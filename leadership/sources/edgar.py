"""SEC EDGAR filings for companies that are public registrants. SPEC 3.1.

Matching is deliberately strict: a wrong match would store another company's
officers as tier-1 facts. A registrant is accepted only when its name equals
the company's name once corporate words are removed ("Robinhood Markets,
Inc." = "Robinhood"), and its listed website, if any, is the same domain.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .web import Fetcher

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_CORPORATE = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
              "plc", "llc", "lp", "sa", "nv", "ag", "holdings", "holding", "group", "markets",
              "technologies", "technology", "platforms", "the", "class", "a", "b", "com"}


@dataclass
class Filing:
    form: str
    filed: str          # ISO date
    url: str
    items: str = ""     # 8-K item numbers


def core_name(name: str) -> str:
    words = re.findall(r"[a-z0-9]+", name.lower())
    return " ".join(w for w in words if w not in _CORPORATE)


def user_agent() -> str:
    return os.environ.get("SEC_USER_AGENT", "").strip()


def match_registrant(tickers: dict, company: str) -> tuple[int, str] | None:
    """(CIK, registrant name) when exactly one registrant has this name."""
    want = core_name(company)
    found = {int(row["cik_str"]): row["title"] for row in tickers.values()
             if want and core_name(row["title"]) == want}
    return next(iter(found.items())) if len(found) == 1 else None


def select_filings(submissions: dict, max_8k: int) -> list[Filing]:
    """Latest annual report, latest proxy statement, recent 8-Ks on officer changes."""
    recent = submissions.get("filings", {}).get("recent", {})
    cik = int(submissions["cik"])
    rows = zip(recent.get("form", []), recent.get("filingDate", []),
               recent.get("accessionNumber", []), recent.get("primaryDocument", []),
               recent.get("items", []))
    out, have = [], {"10-K": 0, "DEF 14A": 0, "8-K": 0}
    for form, filed, accession, document, items in rows:      # newest first
        if form not in have or not document:
            continue
        if form == "8-K" and "5.02" not in items:             # 5.02: officer and director changes
            continue
        if have[form] >= (max_8k if form == "8-K" else 1):
            continue
        have[form] += 1
        out.append(Filing(form, filed, "https://www.sec.gov/Archives/edgar/data/"
                          f"{cik}/{accession.replace('-', '')}/{document}", items))
    return out


def find_filings(fetcher: Fetcher, company: str, company_domains: set[str],
                 max_8k: int) -> tuple[list[Filing], str]:
    """Filings to read, and a one-line explanation of the outcome."""
    from .site import registrable
    agent = user_agent()
    if not agent:
        return [], "skipped: SEC_USER_AGENT is not set in .env"
    headers = {"User-Agent": agent}
    page = fetcher.get(TICKERS_URL, headers=headers)
    if not page.ok:
        return [], f"failed: could not read the SEC company list ({page.error})"
    try:
        match = match_registrant(json.loads(page.body), company)
    except ValueError:
        return [], "failed: the SEC company list was not valid JSON"
    if not match:
        return [], f"no SEC registrant named '{company}'"
    cik, title = match
    page = fetcher.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=headers)
    if not page.ok:
        return [], f"failed: could not read the filing list of {title} ({page.error})"
    try:
        submissions = json.loads(page.body)
    except ValueError:
        return [], f"failed: the filing list of {title} was not valid JSON"
    website = (submissions.get("website") or "").strip()
    if website:
        host = re.sub(r"^https?://", "", website).split("/")[0]
        if registrable(host) not in company_domains:
            return [], f"rejected: SEC registrant {title} lists a different website ({host})"
    filings = select_filings(submissions, max_8k)
    return filings, f"matched SEC registrant {title} (CIK {cik}); {len(filings)} filings selected"

"""Claim-level source check for chat answers, with one correction turn. SPEC section 5, answering rules.

The model is told which source supports which claim, but it can still cite the
source where a role was first seen as the source of its end, miscopy a date,
or say a person left a role without saying they are still at the company.
`problems` finds these in code. It splits the answer into units (paragraphs
and bullet items) and, for each person a unit names, checks the claims the
unit makes against that person's rows in the agent's snapshot:

- role ended ("left", "ceased", ...) with a cited URL: one cited URL must be a
  source that showed the role ending (ended_by_source_url);
- role transition ("became", "appointed", ...) with a cited URL: one cited URL
  must show the end of the old role or the new role;
- still at the company ("remains", "still", ...) with a cited URL: one cited
  URL must be the source of a current role;
- a role-related date must exist in the person's rows or the queried rows; a
  date presented as an end date must be an `ended_on`, not an
  `end_reported_on` or a source date; a date given as a source's date must be
  that source's date;
- a person who left a role but is still employed (employment_status current)
  must be said to be still at the company somewhere in the answer.

A unit that makes no claim of a kind is not checked for it. `grounded_ask`
asks the agent once more with the problems listed; if problems remain, the
answer is shown with a warning that lists them. Nothing here changes the agent
or its cache key: the correction is an ordinary question to the same agent.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import date

from .citations import _named_here, plain

_URL = re.compile(r"https?://[^\s)\]>*,;\"']+")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], 1)}
_LONG = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2}),\s*(\d{4})\b", re.I)

ENDED = re.compile(r"\b(left|leaves|leaving|ended|ceased|stepped down|departed|resigned|no longer|until|"
                   r"end date)\b", re.I)
TRANSITION = re.compile(r"\b(became|transitioned|appointed|promoted|succeeded|moved to)\b", re.I)
STILL_HERE = re.compile(r"\b(still|remains|remained|continues|currently)\b", re.I)
# Words just before a date that present it as when a role ended, and words that say it is a report's date.
_END_CUE = re.compile(r"\b(left|ended|ceased|stepped down|departed|resigned|until|effective|end date)\b[^.;]*$", re.I)
_REPORT_CUE = re.compile(r"\b(report\w*|dated|filing|filed|published|release|according|source|announced)\b", re.I)

DATE_FIELDS = ("starts_on", "ended_on", "end_reported_on", "first_seen", "last_seen", "source_date",
               "ended_by_source_date", "successor_starts_on")


def _clean(text: str) -> str:
    """Plain hyphens and spaces, no Markdown emphasis, long dates written as ISO dates."""
    text = re.sub(r"[‐‑‒–—]", "-", text).replace("**", "").replace("__", "")
    text = re.sub(r"[   ]", " ", text)
    return _LONG.sub(lambda m: f"{int(m[3]):04d}-{_MONTHS[m[1].lower()]:02d}-{int(m[2]):02d}", text)


def _units(text: str) -> list[str]:
    """Paragraphs, with each bullet or numbered item its own unit."""
    units = []
    for para in re.split(r"\n\s*\n", text):
        units += [u for u in re.split(r"\n(?=\s*(?:[-*•]|\d+[.)])\s)", para) if u.strip()]
    return units


def _dates(unit: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _ISO.finditer(unit)
            if _valid(int(m[1]), int(m[2]), int(m[3]))]


def _valid(y: int, m: int, d: int) -> bool:
    try:
        date(y, m, d)
        return True
    except ValueError:
        return False


def _urls(unit: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).rstrip(".:"), m.start(), m.end()) for m in _URL.finditer(unit)]


def _rows(conn: sqlite3.Connection, person: str) -> list[dict]:
    cur = conn.execute("SELECT * FROM leadership WHERE person = ?", (person,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _url_dates(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Every source URL in the snapshot with the dates the data gives it."""
    out: dict[str, set[str]] = {}
    for url, d in conn.execute("SELECT source_url, source_date FROM leadership UNION "
                               "SELECT ended_by_source_url, ended_by_source_date FROM leadership UNION "
                               "SELECT successor_source_url, successor_starts_on FROM leadership"):
        if url:
            out.setdefault(url, set()).update({d} if d else set())
    return out


def problems(text: str, steps, conn: sqlite3.Connection, collected_on: str | None = None) -> list[str]:
    """What in the answer its sources do not support. Empty when nothing was found."""
    people = sorted({row[s.columns.index("person")] for s in steps if not s.error and "person" in s.columns
                     for row in s.rows if row[s.columns.index("person")]})
    queried_dates = {str(v)[:10] for s in steps if not s.error for row in s.rows for v in row
                     if isinstance(v, str) and _ISO.fullmatch(str(v)[:10])}
    url_dates = _url_dates(conn)
    found: list[str] = []
    said_still: set[str] = set()
    left_role: set[str] = set()
    data: dict[str, list[dict]] = {}

    def add(problem: str) -> None:
        if problem not in found:
            found.append(problem)

    for unit in _units(_clean(text)):
        named = [p for p in people if _named_here(unit, p, set(people))]
        if not named:
            continue
        urls = _urls(unit)
        cited = {u for u, _, _ in urls}
        one = len(named) == 1                  # claims are checked only when the unit is about one person
        for p in named:
            rows = data.setdefault(p, _rows(conn, p))
            former = [r for r in rows if r["status"] == "former" and r["ended_by_source_url"]]
            this_role = [r for r in former if r["title"] and plain(r["title"]) in plain(unit)]
            ended_by = {r["ended_by_source_url"] for r in (this_role or former)}
            current_src = {r["source_url"] for r in rows if r["status"] == "current" and r["source_url"]}
            successor = {r["successor_source_url"] for r in rows if r["successor_source_url"]}
            own = {r[c] for r in rows for c in ("source_url", "ended_by_source_url", "location_source_url",
                                                  "conflicting_source_url", "successor_source_url") if r[c]}
            if cited and one and not cited & own:              # role held: the source must be about this person
                add(f"{p}: the answer cites {', '.join(sorted(cited))}, which is not a source of any of {p}'s rows")
            if STILL_HERE.search(unit):
                said_still.add(p)
                if cited and not cited & current_src and one:
                    add(f"{p}: the answer says {p} is still at the company but cites no source of a current role"
                        + (f"; {_current_role(rows)}" if current_src else
                           ", and no current role is in the data"))
            if ENDED.search(unit) and ended_by:
                left_role.add(p)
                if cited and not cited & ended_by and one:
                    add(f"{p}: the end of the role is cited to {', '.join(sorted(cited))}, which is not a source "
                        f"showing it ended; cite ended_by_source_url ({', '.join(sorted(ended_by))})")
            if TRANSITION.search(unit) and cited and not cited & (ended_by | current_src | successor) and one:
                add(f"{p}: the role change is cited to {', '.join(sorted(cited))}, which shows neither the end of "
                    f"the old role nor the new role")
        # Dates. Only units naming a person are checked, against those people's rows and the queried rows.
        rows = [r for p in named for r in data[p]]
        known = {str(r[f])[:10] for r in rows for f in DATE_FIELDS if r.get(f)} | queried_dates
        if collected_on:
            known.add(collected_on)
        ended_on = {r["ended_on"] for r in rows if r.get("ended_on")}
        reported = {r["end_reported_on"] for r in rows if r.get("end_reported_on")}
        for d, start, end in _dates(unit):
            if d not in known:
                add(f"the date {d} is in none of the rows for {', '.join(named)}")
                continue
            before = unit[max(0, start - 45):start]
            if _END_CUE.search(before) and not _REPORT_CUE.search(before) and d not in ended_on:
                kind = "end_reported_on, the date a source reported the end" if d in reported else "not an ended_on"
                add(f"{d} is given as the date a role ended, but it is {kind}; say the end date is not stated")
            for u, us, ue in urls:                          # "..., 2026-04-22: <url>" or "<url> (sec_filing, 2026-04-22)"
                tight = (us >= end and re.fullmatch(r"[\s:,(\-]*", unit[end:us]) is not None) or \
                        (start >= ue and re.fullmatch(r"\s*\(\s*\w+,\s*", unit[ue:start]) is not None)
                if tight and u in url_dates and url_dates[u] and d not in url_dates[u]:
                    add(f"{d} is given as the date of {u}, but the data dates it {', '.join(sorted(url_dates[u]))}")
    for p in sorted(left_role - said_still):
        if any(r["employment_status"] == "current" for r in data[p]):
            add(f"{p}: the answer says a role ended but not that {p} is still at the company "
                f"(employment_status current)" + (f"; {_current_role(data[p])}" if _current_role(data[p]) else ""))
    return found


def _current_role(rows: list[dict]) -> str:
    """The person's most recently sourced current role, as a hint the model can cite."""
    current = [r for r in rows if r["status"] == "current" and r["source_url"]]
    if not current:
        return ""
    r = max(current, key=lambda r: (r["canonical_role"] is not None, r["source_date"] or ""))
    return f"the current role in the data is {r['title']}, source {r['source_url']}, {r['source_date']}"


CORRECTION = """{question}

Your previous answer to this question failed a source check:
{problems}

Previous answer:
{answer}

Answer the question again. Query the rows again, selecting person, title, status, employment_status, ended_on,
end_reported_on, ended_by_source_url, ended_by_source_date, source_url and source_date. Cite each claim with the
source that supports it:
- the person held a role: that row's source_url and source_date;
- the role ended: ended_by_source_url and ended_by_source_date, never source_url;
- ended_on is the date the role ended. end_reported_on is not an end date: write "the end date is not stated;
  first reported as ended in <source>, <ended_by_source_date>";
- a person still at the company (employment_status current): say so, and cite the source_url of their current
  role's row;
- copy every date exactly as it appears in the rows."""


def grounded_ask(agent, question: str, first_problems: list[str] | None = None):
    """Ask, check the answer against its sources, and correct it once if needed.

    `first_problems`, if given, receives the problems found in the first answer (empty: no correction)."""
    from .agent import Answer, collected_on
    first = agent.ask(question)
    if not first.ok:
        return first
    when = collected_on(agent.source_db, agent.domain)
    found = problems(first.text, first.steps, agent.conn, when)
    if first_problems is not None:
        first_problems.extend(found)
    if not found:
        return first
    agent.log.event("grounding", question=question, problems=found)
    retry = agent.ask(CORRECTION.format(question=question, answer=first.text,
                                        problems="\n".join(f"- {p}" for p in found)))
    left = problems(retry.text, retry.steps, agent.conn, when) if retry.ok else found
    agent.log.event("grounding", question=question, problems=left, corrected=not left)
    best = retry if retry.ok and len(left) <= len(found) else first
    text = best.text
    if left:
        shown = left if best is retry else found
        text = (text.rstrip() + "\n\nWarning: parts of this answer do not match their sources:\n"
                + "\n".join(f"- {p}" for p in shown))
    return Answer(question, text, first.steps + (retry.steps if best is retry else []), ok=True,
                  model=best.model, cached=first.cached and retry.cached)

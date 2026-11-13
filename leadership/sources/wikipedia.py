"""The company's Wikipedia article, if one clearly belongs to it. SPEC 3.1.

An article is accepted only when its text mentions the company's own domain,
so "Campfire" the company is not confused with the article about campfires.
"""

from __future__ import annotations

import json
import os
from urllib.parse import quote

from .web import Fetcher

API = "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=3&srsearch="


def find_article(fetcher: Fetcher, company: str,
                 company_domains: set[str]) -> tuple[str | None, dict | None, str]:
    """(article URL or None, request headers, one-line explanation).

    Wikimedia's robot policy refuses clients that do not identify themselves
    with contact details, so the lookup runs only when WIKIPEDIA_USER_AGENT is
    set in .env. It is never bypassed with a browser-like identity.
    """
    agent = os.environ.get("WIKIPEDIA_USER_AGENT", "").strip()
    if not agent:
        return None, None, "skipped: WIKIPEDIA_USER_AGENT is not set in .env"
    headers = {"User-Agent": agent}
    page = fetcher.get(API + quote(f"{company} company"), headers=headers)
    if not page.ok:
        return None, None, f"failed: Wikipedia search ({page.error})"
    try:
        titles = [hit["title"] for hit in json.loads(page.body)["query"]["search"]]
    except (ValueError, KeyError, TypeError):
        return None, None, "failed: Wikipedia search returned an unexpected answer"
    for title in titles:
        url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
        article = fetcher.get(url, headers=headers)
        if article.ok and any(d in article.body.lower() for d in company_domains):
            return url, headers, f"matched Wikipedia article '{title}'"
    return None, None, "no Wikipedia article mentions the company's domain"

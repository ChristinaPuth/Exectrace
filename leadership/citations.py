"""Additions to chat answers made in code, without a model. SPEC section 5, answering rules.

- `add_missing_sources`: the model is told to cite a source URL for every
  person it names, but it sometimes gives only the source type and date.
  The source URL of each named person is then appended from the rows the
  agent actually queried. It adds nothing that was not in those rows, and
  nothing for a person the answer already cites by any of their URLs.
- `add_count_note`: a count is of the people named in the sources that could
  be read. When some could not be read, or the collection did not finish,
  the answer says so, so that "0 found" is not taken as "there are none".
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

# Every column of the views that holds a URL. A person cited by any of them is cited.
URL_COLUMNS = ("source_url", "ended_by_source_url", "location_source_url", "conflicting_source_url",
               "successor_source_url")


def plain(text: str) -> str:
    """Lower case, no accents, typographic apostrophes and spaces made plain."""
    text = unicodedata.normalize("NFKD", text.replace("’", "'").replace("‘", "'"))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()


def named(answer: str, name: str) -> bool:
    """The answer names this person (by surname, as a whole word)."""
    return re.search(rf"\b{re.escape(plain(name).split()[-1])}\b", plain(answer)) is not None


def _named_here(answer: str, person: str, everyone: set[str]) -> bool:
    """Named by full name, or by surname when no other queried person has that surname."""
    if plain(person) in plain(answer):
        return True
    surname = plain(person).split()[-1]
    shared = any(p != person and plain(p).split()[-1] == surname for p in everyone)
    return not shared and named(answer, person)


def add_missing_sources(text: str, steps) -> str:
    """Append "Sources:" lines for named people whom the answer cites by none of their URLs."""
    sources: dict[str, list[tuple[str, str | None, str | None]]] = {}   # the URL to append, best first
    known: dict[str, set[str]] = {}                                      # every URL of the person
    for step in steps:
        if step.error or "person" not in step.columns:
            continue
        col = {c: i for i, c in enumerate(step.columns)}
        for row in step.rows:
            person = row[col["person"]]
            if not person:
                continue
            known.setdefault(person, set()).update(row[col[c]] for c in URL_COLUMNS if c in col and row[col[c]])
            # A former role: the source that ended it is the one that matters.
            for url_c, date_c, kind in (("ended_by_source_url", "ended_by_source_date", "source that ended the role"),
                                        ("source_url", "source_date", None)):
                if url_c in col and row[col[url_c]]:
                    kind = kind or (row[col["source_type"]] if "source_type" in col else None)
                    entry = (row[col[url_c]], row[col[date_c]] if date_c in col else None, kind)
                    if entry not in sources.setdefault(person, []):
                        sources[person].append(entry)
                    break
    lines = []
    for person, entries in sources.items():
        cited = any(url in text for url in known.get(person, ()))
        if entries and not cited and _named_here(text, person, set(sources)):
            url, date, kind = entries[0]
            lines.append(f"- {person}: {url} ({kind or 'source'}, {date or 'date not stated'})")
    if not lines:
        return text
    return text.rstrip() + "\n\nSources:\n" + "\n".join(lines)


def unread_sources(source_db: str | Path, domain: str) -> list[str]:
    """URLs of this company's sources the pipeline was not allowed to read (robots.txt or a refusal)."""
    conn = sqlite3.connect(f"file:{Path(source_db).resolve().as_posix()}?mode=ro", uri=True)
    try:
        return [r[0] for r in conn.execute(
            """SELECT s.url FROM sources s JOIN companies c ON c.id = s.company_id
                WHERE c.domain = ? AND s.stage = 'skipped'
                  AND (s.last_error LIKE '%robots.txt%' OR s.last_error LIKE '%refused automated access%')
                ORDER BY s.url""", (domain,))]
    finally:
        conn.close()


def _counted(steps) -> list[int]:
    """The numbers returned by successful COUNT queries."""
    return [row[0] for s in steps if not s.error and re.search(r"\bcount\s*\(", s.sql or "", re.I)
            for row in s.rows[:1] if row and isinstance(row[0], int)]


def add_count_note(text: str, steps, domain: str, unread: list[str], status: str = "complete") -> str:
    """After a count, say what the count can miss: unread sources and an unfinished collection."""
    counts = _counted(steps)
    if not counts or (not unread and status == "complete"):
        return text
    parts = ["Note: this counts only the people named in the sources that could be read."]
    if unread:
        own = sum(1 for u in unread if (urlparse(u).hostname or "").removeprefix("www.").endswith(domain))
        where = f", {own} of them on the company's own site" if own else ""
        parts.append(f"{len(unread)} sources found for this company could not be read (robots.txt or the site "
                     f"refused automated access){where}.")
    if status != "complete":
        parts.append(f"The data collection did not finish normally (status {status}).")
    parts.append("None found does not mean there are none; the real number may be higher." if 0 in counts
                 else "The real number may be higher.")
    return text.rstrip() + "\n\n" + " ".join(parts)

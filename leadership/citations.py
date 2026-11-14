"""Citations for chat answers, added without a model. SPEC section 5, answering rules.

The model is told to cite a source URL for every person it names, but it
sometimes gives only the source type and date. `add_missing_sources` then
appends the source URL of each named person from the rows the agent actually
queried. It adds nothing that was not in those rows.
"""

from __future__ import annotations

import re
import unicodedata


def plain(text: str) -> str:
    """Lower case, no accents, typographic apostrophes and spaces made plain."""
    text = unicodedata.normalize("NFKD", text.replace("’", "'").replace("‘", "'"))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()


def named(answer: str, name: str) -> bool:
    """The answer names this person (by surname, as a whole word)."""
    return re.search(rf"\b{re.escape(plain(name).split()[-1])}\b", plain(answer)) is not None


def add_missing_sources(text: str, steps) -> str:
    """Append "Sources:" lines for named people whose source URL the answer does not give."""
    sources: dict[str, list[tuple[str, str | None, str | None]]] = {}
    for step in steps:
        if step.error or "person" not in step.columns or "source_url" not in step.columns:
            continue
        i, u = step.columns.index("person"), step.columns.index("source_url")
        d = step.columns.index("source_date") if "source_date" in step.columns else None
        t = step.columns.index("source_type") if "source_type" in step.columns else None
        for row in step.rows:
            if row[i] and row[u]:
                entry = (row[u], row[d] if d is not None else None, row[t] if t is not None else None)
                if entry not in sources.setdefault(row[i], []):
                    sources[row[i]].append(entry)
    lines = []
    for person, entries in sources.items():
        if named(text, person) and not any(url in text for url, _, _ in entries):
            url, date, kind = entries[0]
            lines.append(f"- {person}: {url} ({kind or 'source'}, {date or 'date not stated'})")
    if not lines:
        return text
    return text.rstrip() + "\n\nSources:\n" + "\n".join(lines)

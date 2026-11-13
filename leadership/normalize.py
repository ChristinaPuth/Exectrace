"""Deterministic normalization of titles and names. SPEC sections 3.4 and 3.5.

Pure functions, no I/O and no model, so every rule is unit-testable.
"""

from __future__ import annotations

import re
import unicodedata

from .sources.edgar import core_name

DEPARTMENTS = ("engineering", "product", "marketing", "sales", "finance", "legal", "people",
               "operations", "design", "data", "security", "other")

# Single-holder roles with a common short form. Any other "Chief X Officer"
# keeps its full wording as the canonical role.
_CHIEFS = {
    "executive": "CEO", "financial": "CFO", "finance": "CFO", "technology": "CTO",
    "technical": "CTO", "operating": "COO", "operations": "COO", "marketing": "CMO",
    "product": "CPO", "legal": "CLO", "information": "CIO", "revenue": "CRO",
    "information security": "CISO", "people": "Chief People Officer",
    "human resources": "CHRO",
}
_ABBREVIATIONS = {"ceo": "CEO", "cfo": "CFO", "cto": "CTO", "coo": "COO", "cmo": "CMO",
                  "cpo": "CPO", "clo": "CLO", "cio": "CIO", "cro": "CRO", "ciso": "CISO",
                  "chro": "CHRO"}
_OFFICE = r"\bofficer\b|\b(?:" + "|".join(_ABBREVIATIONS) + r")\b"
_CHIEF = re.compile(r"\bchief\s+([a-z&,\s]+?)\s+officer\b")
_VP = re.compile(r"\b(?:[se]?vp|vice[\s-]+president)\b")
_HEAD = re.compile(r"\b(?:global\s+)?head\s+of\b|\bhead,\s")
_PRESIDENT = re.compile(r"(?<!vice )(?<!vice-)\bpresident\b")
# Words that follow an office without naming a part of the company.
_OTHER_ROLE = re.compile(r"^(?:co-?\s?founder|founder|chair|president|director|chief|general|"
                         r"secretary|treasurer|member|the\s+board|board|principal)\b")


def _plain(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("&", " and ").casefold()).strip()


def level(title: str | None) -> str:
    """c_level, vp, head or other, from the title's wording."""
    if not title:
        return "other"
    t = _plain(title)
    if _CHIEF.search(t) or any(re.search(rf"\b{a}\b", t) for a in _ABBREVIATIONS):
        return "c_level"
    if _VP.search(t):
        return "vp"                     # includes "Senior Vice President" and "VP of ..."
    if _PRESIDENT.search(t) or re.search(r"\bgeneral counsel\b", t):
        return "c_level"
    if _HEAD.search(t):
        return "head"
    return "other"                      # directors, board members, founders without an office


def has_division(title: str, company: str | None = None) -> bool:
    """The title names a part of the company: 'CEO of Robinhood Crypto'.

    'CEO of Robinhood Markets, Inc.' names the company itself, not a part.
    """
    after = re.split(_OFFICE, _plain(title))[-1]
    for part in re.split(r",|\band\b|;|/|\(|\)", after):
        named = re.sub(r"^(?:of|at|for)\s+", "", part.strip(" -–—"))
        if not named or _OTHER_ROLE.match(named) or not core_name(named):
            continue
        if company and core_name(named) == core_name(company):
            continue
        return True
    return False


def canonical_role(title: str | None, company: str | None = None) -> str | None:
    """The single-holder role a title names, or None.

    'President and CEO' -> CEO; 'Chief Compliance Officer' keeps its wording;
    'CEO of Robinhood Crypto' -> None, because it is one division's CEO.
    When a title names two offices the first is used; extraction is asked to
    give one claim per office.
    """
    if not title:
        return None
    t = _plain(title)
    found: str | None = None
    if m := _CHIEF.search(t):
        # Commas and "and" are spelled differently by different sources.
        what = re.sub(r"\s+", " ", m.group(1).replace(",", " ")).strip()
        named = " ".join(w if w == "and" else w.capitalize() for w in what.split())
        found = _CHIEFS.get(what) or f"Chief {named} Officer"
    else:
        first = min(((m.start(), role) for a, role in _ABBREVIATIONS.items()
                     if (m := re.search(rf"\b{a}\b", t))), default=None)
        found = first[1] if first else None
    if found and has_division(title, company):
        return None
    return found


# ----------------------------------------------------------------------- names
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "mba", "cpa", "esq", "cfa", "jd"}
_PREFIXES = {"dr", "mr", "mrs", "ms", "prof", "sir"}
_NICKNAMES = [
    {"vlad", "vladimir"}, {"bill", "will", "william"}, {"bob", "rob", "robert"},
    {"mike", "michael"}, {"dan", "daniel"}, {"jim", "james"}, {"tom", "thomas"},
    {"chris", "christopher"}, {"matt", "matthew"}, {"steve", "steven", "stephen"},
    {"dave", "david"}, {"joe", "joseph"}, {"ben", "benjamin"}, {"alex", "alexander"},
    {"kate", "katherine", "catherine", "kathy", "cathy"}, {"liz", "beth", "elizabeth"},
    {"jeff", "jeffrey"}, {"greg", "gregory"}, {"tony", "anthony"}, {"nick", "nicholas"},
    {"sam", "samuel", "samantha"}, {"andy", "andrew"}, {"jon", "jonathan"},
    {"tim", "timothy"}, {"pat", "patrick", "patricia"}, {"rick", "rich", "richard"},
    {"ed", "edward"}, {"ken", "kenneth"}, {"ron", "ronald"}, {"sue", "susan"},
]


def normalize_name(name: str) -> str:
    """Lower case, no accents, no titles, suffixes, middle initials or punctuation."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    text = re.sub(r"\([^)]*\)|\"[^\"]*\"", " ", text)           # nicknames in brackets or quotes
    words = [w.strip(".") for w in re.sub(r"[^a-z.'\-\s]", " ", text).split()]
    words = [w for w in words if w and w not in _SUFFIXES and w not in _PREFIXES]
    if len(words) > 2:                                          # drop middle initials
        words = [words[0]] + [w for w in words[1:-1] if len(w) > 1] + [words[-1]]
    return " ".join(words)


def _same_given_name(a: str, b: str) -> bool:
    return a == b or any(a in group and b in group for group in _NICKNAMES)


def compare_names(a: str, b: str) -> str:
    """How two normalized names relate: 'same', 'nickname', 'near' or 'different'.

    'nickname' needs a matching role before the two are merged; 'near' is
    never merged, only flagged for a human.
    """
    if a == b:
        return "same"
    wa, wb = a.split(), b.split()
    if len(wa) < 2 or len(wb) < 2 or wa[-1] != wb[-1]:
        return "different"
    if _same_given_name(wa[0], wb[0]):
        # "Vlad Tenev" and "Vladimir Tenev"; or the same first name with an extra middle name.
        return "nickname" if wa[0] != wb[0] else "near"
    if wa[0][0] == wb[0][0] and (len(wa[0]) == 1 or len(wb[0]) == 1):
        return "near"                                           # "J Warnick" and "Jason Warnick"
    return "different"

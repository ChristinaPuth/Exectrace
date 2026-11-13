"""Worked cases for role resolution (SPEC section 3.6).

Each case is a small set of claims and the outcome the rules must produce.
The same list drives the unit tests and `tools/show_cases.py`, which prints
it as a table for manual review.
"""

from datetime import date

from leadership.resolve import (ANNOUNCED, CURRENT, FORMER, R0_ANNOUNCED, R0_STALE,
                                R1_END, R1_FUTURE, R1_PENDING, R2_REPORTED, R3_APPOINTED,
                                R3_LISTED, R4_LISTED, R5_SUPPORTED, R6_WEAK, R7_OUTRANKED,
                                UNVERIFIED, Claim)

TODAY = date(2026, 11, 1)
X, Y, Z = 1, 2, 3     # people


def _c(id, person, kind, title=None, *, tier, src, on, basis="published", eff=None,
       canonical=None, level="c_level", dept="other", scope=None, source=None,
       pending=False):
    return Claim(id=id, person_id=person, source_id=source or id, claim_type=kind,
                 tier=tier, source_type=src, as_of_date=on, as_of_basis=basis,
                 effective_date=eff, raw_title=title, level=level, department=dept,
                 canonical_role=canonical, departure_scope=scope, effective_pending=pending)


def live(id, person, title, **kw):          # company's own leadership page, retrieved today
    return _c(id, person, "role", title, tier=1, src="company_site", on=TODAY, basis="retrieved", **kw)


def filing(id, person, title, on, **kw):    # SEC filing
    return _c(id, person, "role", title, tier=1, src="sec_filing", on=on, **kw)


def release(id, person, title, on, **kw):   # press release on the company's site
    return _c(id, person, "role", title, tier=2, src="press_release", on=on, **kw)


def news(id, person, title, on, **kw):      # third-party article
    return _c(id, person, "role", title, tier=3, src="news", on=on, **kw)


def snippet(id, person, title, **kw):       # search-result snippet
    return _c(id, person, "role", title, tier=4, src="search_snippet", on=TODAY, basis="retrieved", **kw)


def departure(id, person, scope, *, tier, src, on, **kw):
    return _c(id, person, "departure", kw.pop("title", None), tier=tier, src=src, on=on, scope=scope, **kw)


def appointment(id, person, title, *, tier, src, on, **kw):
    return _c(id, person, "appointment", title, tier=tier, src=src, on=on, **kw)


CTO = dict(canonical="CTO")
CFO = dict(canonical="CFO")
CEO = dict(canonical="CEO")

def matches(case, result) -> bool:
    """Does a resolution give exactly the outcome the case expects?"""
    got = {(r.person_id, r.key): (r.status, r.status_reason) for r in result.roles}
    flagged = {(r.person_id, r.key) for r in result.roles if r.conflict}
    return (got == case["expect"] and result.employment == case["employment"]
            and flagged == case.get("conflict", set()))


# expect: {(person, role key): (status, rule)}    employment: {person: status}
# conflict: the roles that must be flagged as a conflict (none when omitted)
CASES = [
    dict(
        name="Only an old press release names X as CTO",
        claims=[release(1, X, "CTO", date(2023, 5, 1), **CTO)],
        expect={(X, "CTO"): (UNVERIFIED, R6_WEAK)},
        employment={X: "unknown"},
    ),
    dict(
        name="Live leadership page lists X",
        claims=[live(1, X, "CTO", **CTO)],
        expect={(X, "CTO"): (CURRENT, R4_LISTED)},
        employment={X: "current"},
    ),
    dict(
        name="Page lists X; a company release says X stepped down as CTO",
        claims=[live(1, X, "CTO", **CTO),
                departure(2, X, "role", tier=2, src="press_release", on=date(2026, 9, 15), title="CTO", **CTO)],
        expect={(X, "CTO"): (FORMER, R1_END)},
        employment={X: "unknown"},
    ),
    dict(
        name="Page lists X; an SEC filing says X left the company",
        claims=[live(1, X, "CTO", **CTO),
                departure(2, X, "company", tier=1, src="sec_filing", on=date(2026, 9, 15))],
        expect={(X, "CTO"): (FORMER, R1_END)},
        employment={X: "former"},
    ),
    dict(
        name="Page lists X; only a news article reports the departure",
        claims=[live(1, X, "CTO", **CTO),
                departure(2, X, "role", tier=3, src="news", on=date(2026, 10, 1), title="CTO", **CTO)],
        expect={(X, "CTO"): (UNVERIFIED, R2_REPORTED)},
        employment={X: "unknown"},
        conflict={(X, "CTO")},
    ),
    dict(
        name="No page lists X; only a news article reports the departure",
        claims=[release(1, X, "CTO", date(2026, 3, 1), **CTO),
                departure(2, X, "role", tier=3, src="news", on=date(2026, 9, 1), title="CTO", **CTO)],
        expect={(X, "CTO"): (UNVERIFIED, R2_REPORTED)},
        employment={X: "unknown"},
        conflict={(X, "CTO")},
    ),
    dict(
        name="Departure announced with a future effective date",
        claims=[live(1, X, "CTO", **CTO),
                departure(2, X, "role", tier=2, src="press_release", on=date(2026, 10, 20),
                          eff=date(2026, 12, 31), title="CTO", **CTO)],
        expect={(X, "CTO"): (CURRENT, R1_FUTURE)},
        employment={X: "current"},
    ),
    dict(
        name="Old release names X; live page shows Y in that role",
        claims=[release(1, X, "CTO", date(2023, 5, 1), **CTO), live(2, Y, "CTO", **CTO)],
        expect={(X, "CTO"): (FORMER, R3_LISTED), (Y, "CTO"): (CURRENT, R4_LISTED)},
        employment={X: "unknown", Y: "current"},
    ),
    dict(
        name="X steps down as CTO, stays as board advisor",
        claims=[release(1, X, "CTO", date(2025, 2, 1), **CTO),
                departure(2, X, "role", tier=2, src="press_release", on=date(2026, 8, 1), title="CTO", **CTO),
                live(3, X, "Board Advisor", level="other")],
        expect={(X, "CTO"): (FORMER, R1_END), (X, "board advisor"): (CURRENT, R4_LISTED)},
        employment={X: "current"},
    ),
    dict(
        name="X's only role ends; nothing says X left the company",
        claims=[release(1, X, "CTO", date(2025, 2, 1), **CTO),
                departure(2, X, "role", tier=2, src="press_release", on=date(2026, 8, 1), title="CTO", **CTO)],
        expect={(X, "CTO"): (FORMER, R1_END)},
        employment={X: "unknown"},
    ),
    dict(
        name="Concurrent roles: X is CEO on the page and Chair in a filing",
        claims=[live(1, X, "Co-founder and CEO", **CEO),
                filing(2, X, "Chair of the Board", date(2026, 4, 1), level="other")],
        expect={(X, "CEO"): (CURRENT, R4_LISTED),
                (X, "chair of the board"): (CURRENT, R5_SUPPORTED)},
        employment={X: "current"},
    ),
    dict(
        name="Successor appointed: Y becomes CFO; X keeps another role",
        claims=[filing(1, X, "CFO", date(2026, 2, 1), source=10, **CFO),
                filing(2, X, "President", date(2026, 2, 1), source=10),
                appointment(3, Y, "CFO", tier=2, src="press_release", on=date(2026, 7, 15),
                            eff=date(2026, 8, 1), **CFO)],
        expect={(X, "CFO"): (FORMER, R3_APPOINTED), (X, "president"): (CURRENT, R5_SUPPORTED),
                (Y, "CFO"): (CURRENT, R5_SUPPORTED)},
        employment={X: "current", Y: "current"},
    ),
    dict(
        name="Company release appoints Y as CTO; the page still lists only X (page lags)",
        claims=[live(1, X, "CTO", **CTO),
                appointment(2, Y, "CTO", tier=2, src="press_release", on=date(2026, 9, 1), **CTO)],
        expect={(X, "CTO"): (FORMER, R3_APPOINTED), (Y, "CTO"): (CURRENT, R5_SUPPORTED)},
        employment={X: "unknown", Y: "current"},
    ),
    dict(
        name="Y appointed co-CEO; the page lists both X and Y",
        claims=[live(1, X, "Co-CEO", **CEO), live(2, Y, "Co-CEO", **CEO),
                appointment(3, Y, "Co-CEO", tier=2, src="press_release", on=date(2026, 9, 1), **CEO)],
        expect={(X, "CEO"): (CURRENT, R4_LISTED), (Y, "CEO"): (CURRENT, R4_LISTED)},
        employment={X: "current", Y: "current"},
    ),
    dict(
        name="Newer news article names Y as CTO; company page lists X",
        claims=[live(1, X, "CTO", **CTO), news(2, Y, "CTO", date(2026, 10, 10), **CTO)],
        expect={(X, "CTO"): (CURRENT, R4_LISTED), (Y, "CTO"): (UNVERIFIED, R7_OUTRANKED)},
        employment={X: "current", Y: "unknown"},
        conflict={(X, "CTO"), (Y, "CTO")},
    ),
    dict(
        name="Co-CEOs: the company page lists both",
        claims=[live(1, X, "Co-CEO", **CEO), live(2, Y, "Co-CEO", **CEO)],
        expect={(X, "CEO"): (CURRENT, R4_LISTED), (Y, "CEO"): (CURRENT, R4_LISTED)},
        employment={X: "current", Y: "current"},
    ),
    dict(
        name="Only a search snippet names X",
        claims=[snippet(1, X, "CTO", **CTO)],
        expect={(X, "CTO"): (UNVERIFIED, R6_WEAK)},
        employment={X: "unknown"},
    ),
    dict(
        name="A snippet reports X's departure; company page lists X",
        claims=[live(1, X, "CTO", **CTO),
                departure(2, X, "role", tier=4, src="search_snippet", on=TODAY, title="CTO", **CTO)],
        expect={(X, "CTO"): (CURRENT, R4_LISTED)},
        employment={X: "current"},
    ),
    dict(
        name="Only a search snippet says X is a former CTO",
        claims=[departure(1, X, "role", tier=4, src="search_snippet", on=TODAY, title="CTO", **CTO)],
        expect={},                          # too weak to record any role
        employment={X: "unknown"},
    ),
    dict(
        name="Title change: X moves from CTO to Chief Architect",
        claims=[release(1, X, "CTO", date(2025, 1, 10), **CTO),
                departure(2, X, "role", tier=2, src="press_release", on=date(2026, 7, 1),
                          eff=date(2026, 7, 1), title="CTO", source=20, **CTO),
                appointment(3, X, "Chief Architect", tier=2, src="press_release", on=date(2026, 7, 1),
                            eff=date(2026, 7, 1), source=20)],
        expect={(X, "CTO"): (FORMER, R1_END), (X, "chief architect"): (CURRENT, R5_SUPPORTED)},
        employment={X: "current"},
    ),
    dict(
        name="Published in March, effective end of June; a later April release still names X",
        claims=[departure(1, X, "role", tier=2, src="press_release", on=date(2026, 3, 1),
                          eff=date(2026, 6, 30), title="CTO", **CTO),
                release(2, X, "CTO", date(2026, 4, 1), **CTO)],
        expect={(X, "CTO"): (FORMER, R1_END)},
        employment={X: "unknown"},
    ),
    dict(
        name="Same claims, but viewed in May before the effective date",
        today=date(2026, 5, 1),
        claims=[departure(1, X, "role", tier=2, src="press_release", on=date(2026, 3, 1),
                          eff=date(2026, 6, 30), title="CTO", **CTO),
                release(2, X, "CTO", date(2026, 4, 1), **CTO)],
        expect={(X, "CTO"): (CURRENT, R1_FUTURE)},
        employment={X: "current"},
    ),
    dict(
        name="X left the company, then was re-appointed CTO later",
        claims=[departure(1, X, "company", tier=2, src="press_release", on=date(2025, 1, 10)),
                appointment(2, X, "CTO", tier=2, src="press_release", on=date(2026, 2, 1),
                            eff=date(2026, 2, 1), **CTO)],
        expect={(X, "CTO"): (CURRENT, R5_SUPPORTED)},
        employment={X: "current"},
    ),
    dict(
        name="Page lists X; X's only dated source is from 2019",
        claims=[live(1, X, "CTO", **CTO), release(2, X, "CTO", date(2019, 6, 1), **CTO)],
        expect={(X, "CTO"): (CURRENT, R4_LISTED)},
        employment={X: "current"},
    ),
    dict(
        name="No page; news article from 5 months ago names X as VP Engineering",
        claims=[news(1, X, "VP of Engineering", date(2026, 6, 1), level="vp", dept="engineering")],
        expect={(X, "vp:engineering"): (CURRENT, R5_SUPPORTED)},
        employment={X: "current"},
    ),
    dict(
        name="No page; the same article is 17 months old",
        claims=[news(1, X, "VP of Engineering", date(2025, 6, 1), level="vp", dept="engineering")],
        expect={(X, "vp:engineering"): (UNVERIFIED, R6_WEAK)},
        employment={X: "unknown"},
    ),
    dict(
        name="No page; an undated article names X",
        claims=[news(1, X, "CTO", TODAY, basis="retrieved", **CTO)],
        expect={(X, "CTO"): (UNVERIFIED, R6_WEAK)},
        employment={X: "unknown"},
    ),
    dict(
        name="Role known only from a company statement calling X 'former CTO'",
        claims=[departure(1, X, "role", tier=2, src="press_release", on=date(2026, 5, 1), title="CTO", **CTO)],
        expect={(X, "CTO"): (FORMER, R1_END)},
        employment={X: "unknown"},
    ),
    dict(
        name="Release in October says Y will become CFO on 1 January; X is the CFO in a filing",
        claims=[filing(1, X, "CFO", date(2026, 2, 1), **CFO),
                appointment(2, Y, "CFO", tier=2, src="press_release", on=date(2026, 10, 15),
                            eff=date(2027, 1, 1), **CFO)],
        expect={(X, "CFO"): (CURRENT, R5_SUPPORTED), (Y, "CFO"): (ANNOUNCED, R0_ANNOUNCED)},
        employment={X: "current", Y: "unknown"},
    ),
    dict(
        name="Same claims, but viewed on 2 January after the effective date",
        today=date(2027, 1, 2),
        claims=[filing(1, X, "CFO", date(2026, 2, 1), **CFO),
                appointment(2, Y, "CFO", tier=2, src="press_release", on=date(2026, 10, 15),
                            eff=date(2027, 1, 1), **CFO)],
        expect={(X, "CFO"): (FORMER, R3_APPOINTED), (Y, "CFO"): (CURRENT, R5_SUPPORTED)},
        employment={X: "unknown", Y: "current"},
    ),
    dict(
        name="Release says Y 'will become CFO' and gives no date; X is the CFO in a filing",
        claims=[filing(1, X, "CFO", date(2026, 2, 1), **CFO),
                appointment(2, Y, "CFO", tier=2, src="press_release", on=date(2026, 9, 1),
                            pending=True, **CFO)],
        expect={(X, "CFO"): (CURRENT, R5_SUPPORTED), (Y, "CFO"): (ANNOUNCED, R0_ANNOUNCED)},
        employment={X: "current", Y: "unknown"},
    ),
    dict(
        name="Same undated announcement; the live page now lists Y as CFO",
        claims=[filing(1, X, "CFO", date(2026, 2, 1), **CFO),
                appointment(2, Y, "CFO", tier=2, src="press_release", on=date(2026, 9, 1),
                            pending=True, **CFO),
                live(3, Y, "CFO", **CFO)],
        expect={(X, "CFO"): (FORMER, R3_APPOINTED), (Y, "CFO"): (CURRENT, R4_LISTED)},
        employment={X: "unknown", Y: "current"},
    ),
    dict(
        name="Same undated announcement; a later news article names Y as CFO",
        claims=[filing(1, X, "CFO", date(2026, 2, 1), **CFO),
                appointment(2, Y, "CFO", tier=2, src="press_release", on=date(2026, 9, 1),
                            pending=True, **CFO),
                news(3, Y, "CFO", date(2026, 10, 10), **CFO)],
        expect={(X, "CFO"): (FORMER, R3_APPOINTED), (Y, "CFO"): (CURRENT, R5_SUPPORTED)},
        employment={X: "unknown", Y: "current"},
    ),
    dict(
        name="Undated announcement from 15 months ago was never confirmed",
        claims=[filing(1, X, "CFO", date(2026, 2, 1), **CFO),
                appointment(2, Y, "CFO", tier=2, src="press_release", on=date(2025, 8, 1),
                            pending=True, **CFO)],
        expect={(X, "CFO"): (CURRENT, R5_SUPPORTED), (Y, "CFO"): (UNVERIFIED, R0_STALE)},
        employment={X: "current", Y: "unknown"},
    ),
    dict(
        name="Page lists X; a company release says X 'will step down as CTO', no date",
        claims=[live(1, X, "CTO", **CTO),
                departure(2, X, "role", tier=2, src="press_release", on=date(2026, 9, 15),
                          title="CTO", pending=True, **CTO)],
        expect={(X, "CTO"): (CURRENT, R1_PENDING)},
        employment={X: "current"},
    ),
    dict(
        name="No page; a company release says X 'will leave the company', no date",
        claims=[release(1, X, "CTO", date(2026, 3, 1), **CTO),
                departure(2, X, "company", tier=2, src="press_release", on=date(2026, 9, 1),
                          pending=True)],
        expect={(X, "CTO"): (CURRENT, R1_PENDING)},
        employment={X: "current"},
    ),
    dict(
        name="Undated departure; a later company release says X has stepped down",
        claims=[release(1, X, "CTO", date(2026, 3, 1), **CTO),
                departure(2, X, "role", tier=2, src="press_release", on=date(2026, 8, 1),
                          title="CTO", pending=True, **CTO),
                departure(3, X, "role", tier=2, src="press_release", on=date(2026, 10, 1),
                          title="CTO", **CTO)],
        expect={(X, "CTO"): (FORMER, R1_END)},
        employment={X: "unknown"},
    ),
    dict(
        name="Undated departure from 17 months ago, never confirmed; the page still lists X",
        claims=[live(1, X, "CTO", **CTO),
                departure(2, X, "role", tier=2, src="press_release", on=date(2025, 6, 1),
                          title="CTO", pending=True, **CTO)],
        expect={(X, "CTO"): (CURRENT, R4_LISTED)},
        employment={X: "current"},
    ),
    dict(
        name="Undated 'will leave the company' from 17 months ago; only an old release names X",
        claims=[release(1, X, "CTO", date(2023, 5, 1), **CTO),
                departure(2, X, "company", tier=2, src="press_release", on=date(2025, 6, 1),
                          pending=True)],
        expect={(X, "CTO"): (UNVERIFIED, R6_WEAK)},
        employment={X: "unknown"},
    ),
    dict(
        name="The company page lists X and Y, both as plain 'CEO'",
        claims=[live(1, X, "CEO", **CEO), live(2, Y, "CEO", **CEO)],
        expect={(X, "CEO"): (CURRENT, R4_LISTED), (Y, "CEO"): (CURRENT, R4_LISTED)},
        employment={X: "current", Y: "current"},
        conflict={(X, "CEO"), (Y, "CEO")},
    ),
    dict(
        name="The page lists X as 'Co-CEO' but Y as plain 'CEO'",
        claims=[live(1, X, "Co-CEO", **CEO), live(2, Y, "CEO", **CEO)],
        expect={(X, "CEO"): (CURRENT, R4_LISTED), (Y, "CEO"): (CURRENT, R4_LISTED)},
        employment={X: "current", Y: "current"},
        conflict={(X, "CEO"), (Y, "CEO")},
    ),
    dict(
        name="The page lists X as 'Co-CEO'; only a news article calls Y 'Co-CEO'",
        claims=[live(1, X, "Co-CEO", **CEO), news(2, Y, "Co-CEO", date(2026, 10, 10), **CEO)],
        expect={(X, "CEO"): (CURRENT, R4_LISTED), (Y, "CEO"): (UNVERIFIED, R7_OUTRANKED)},
        employment={X: "current", Y: "unknown"},
        conflict={(X, "CEO"), (Y, "CEO")},
    ),
    dict(
        name="The page lists X as 'Joint CEO' and Y as 'Joint Chief Executive Officer'",
        claims=[live(1, X, "Joint CEO", **CEO), live(2, Y, "Joint Chief Executive Officer", **CEO)],
        expect={(X, "CEO"): (CURRENT, R4_LISTED), (Y, "CEO"): (CURRENT, R4_LISTED)},
        employment={X: "current", Y: "current"},
    ),
    dict(
        name="The page lists both X and Y as 'Co-founder and CEO'",
        claims=[live(1, X, "Co-founder and CEO", **CEO), live(2, Y, "Co-founder and CEO", **CEO)],
        expect={(X, "CEO"): (CURRENT, R4_LISTED), (Y, "CEO"): (CURRENT, R4_LISTED)},
        employment={X: "current", Y: "current"},
        conflict={(X, "CEO"), (Y, "CEO")},
    ),
    dict(
        name="The page lists X and Y as 'Co-CEO' and a third person, Z, as plain 'CEO'",
        claims=[live(1, X, "Co-CEO", **CEO), live(2, Y, "Co-CEO", **CEO), live(3, Z, "CEO", **CEO)],
        expect={(X, "CEO"): (CURRENT, R4_LISTED), (Y, "CEO"): (CURRENT, R4_LISTED),
                (Z, "CEO"): (CURRENT, R4_LISTED)},
        employment={X: "current", Y: "current", Z: "current"},
        conflict={(X, "CEO"), (Y, "CEO"), (Z, "CEO")},
    ),
]

"""Decide which roles are current, former or unverified. SPEC section 3.6.

`resolve` is a pure function: claims and a date in, roles out. It reads no
clock, database or network, and never asks a model. Every status it returns
names the rule that produced it.

Principles
- Concurrent roles are normal; a different title elsewhere is not a conflict.
- A role ends only on evidence. Age alone can only downgrade to unverified.
- Explicit events (appointments, departures) outrank page listings.
- Reliability outranks recency: a lower tier never overrides a higher one.
- Leaving a role is not leaving the company.
- A change announced for the future does nothing until it takes effect. If no
  date was given, it takes effect only when a later source shows it happened.
"""

from __future__ import annotations

import calendar
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date

CURRENT, FORMER, UNVERIFIED, ANNOUNCED = "current", "former", "unverified", "announced"

# status_reason values, one per rule.
R0_ANNOUNCED = "rule0_appointment_announced_not_yet_effective"
R0_STALE = "rule0_announcement_never_confirmed"
R1_END = "rule1_explicit_end"
R1_FUTURE = "rule1_departure_announced_for_future_date"
R1_PENDING = "rule1_departure_announced_without_date"
R2_REPORTED = "rule2_departure_reported_by_news_only"
R3_APPOINTED = "rule3_successor_appointed"
R3_LISTED = "rule3_successor_listed_on_company_page"
R4_LISTED = "rule4_listed_on_company_page"
R5_SUPPORTED = "rule5_recent_reliable_source"
R6_WEAK = "rule6_stale_or_weak_evidence"
R7_OUTRANKED = "rule7_outranked_in_single_holder_role"


@dataclass(frozen=True)
class Claim:
    """One observation, joined with the tier and type of its source."""

    id: int
    person_id: int
    source_id: int
    claim_type: str                 # role | appointment | departure | location | bio
    tier: int                       # 1 (best) to 4, from the source
    source_type: str
    as_of_date: date                # published date, or retrieval date for live pages
    as_of_basis: str                # published | retrieved
    effective_date: date | None = None
    raw_title: str | None = None
    level: str | None = None
    department: str | None = None
    canonical_role: str | None = None   # set only for single-holder roles (CEO, CFO, ...)
    departure_scope: str | None = None  # role | company
    effective_pending: bool = False     # described as a future change, with no date
    effective_inferred: bool = False    # the date was worked out by the rules, not stated by a source

    @property
    def pending(self) -> bool:
        """Announced for the future without a date: not known to have happened."""
        return self.effective_pending and self.effective_date is None

    @property
    def stated_date(self) -> date | None:
        """The effective date a source wrote down. Never a publication or worked-out date."""
        return None if self.effective_inferred else self.effective_date

    @property
    def when(self) -> date:
        """When the fact took effect: the stated date, else the source's date."""
        return self.effective_date or self.as_of_date

    @property
    def is_live_listing(self) -> bool:
        """A role shown on the company's own site as retrieved now."""
        return (self.claim_type == "role" and self.tier == 1
                and self.source_type == "company_site" and self.as_of_basis == "retrieved")

    @property
    def is_dated(self) -> bool:
        """Has a real date. An undated article only has its retrieval date: weak."""
        return self.effective_date is not None or self.as_of_basis == "published"


@dataclass
class Role:
    person_id: int
    key: str
    title: str
    level: str | None
    department: str | None
    canonical_role: str | None
    status: str
    status_reason: str
    ended_on: date | None = None            # only a date a source states; never a publication date
    end_reported_on: date | None = None     # when no date is stated: the date of the source reporting the end
    ended_by_claim_id: int | None = None
    first_seen: date | None = None
    last_seen: date | None = None
    conflict: bool = False
    conflict_reason: str | None = None
    conflicting_claim_id: int | None = None
    source_count: int = 0
    supporting_claim_id: int | None = None
    starts_on: date | None = None           # announced roles: the stated start date, if any
    successor_claim_id: int | None = None   # current roles: the announcement of a successor


@dataclass
class Resolution:
    roles: list[Role]
    employment: dict[int, str]      # person_id -> current | former | unknown


_SHARED = re.compile(r"\b(?:co|joint)[\s-]*(?:c[a-z]o|chief|president|chair|head)\b", re.I)


def is_shared_title(title: str | None) -> bool:
    """The title itself says the role is shared: 'Co-CEO', 'Joint Chief Executive'.

    'Co-founder and CEO' is not a shared CEO role.
    """
    return bool(title and _SHARED.search(title))


def role_key(c: Claim) -> str | None:
    """What counts as 'the same role' for one person across sources."""
    if c.canonical_role:
        return c.canonical_role
    if c.level in ("c_level", "vp", "head") and c.department and c.department != "other":
        return f"{c.level}:{c.department}"
    if c.raw_title:
        return re.sub(r"[^a-z0-9]+", " ", c.raw_title.casefold()).strip() or None
    return None


def _months_before(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 - months, 12)
    return date(y, m + 1, min(d.day, calendar.monthrange(y, m + 1)[1]))


def _best(claims: list[Claim]) -> Claim:
    """Most reliable, then most recent, then lowest id (for determinism)."""
    return min(claims, key=lambda c: (c.tier, -c.when.toordinal(), c.id))


def resolve(claims: Iterable[Claim], today: date, freshness_months: int = 12) -> Resolution:
    claims = sorted(claims, key=lambda c: c.id)     # order of input must not matter
    cutoff = _months_before(today, freshness_months)

    # An appointment announced without a date takes effect only when a later
    # source shows the person in the role, on the date of that source.
    def shows_in_role(o: Claim, a: Claim) -> bool:
        if o.id == a.id or o.person_id != a.person_id or role_key(o) != role_key(a):
            return False
        if o.is_live_listing:
            return True
        return (o.claim_type in ("role", "appointment") and o.tier <= 3 and o.is_dated
                and not o.pending and a.as_of_date < o.when <= today)

    original, claims = claims, []
    for c in original:
        if c.claim_type == "appointment" and c.pending:
            proof = [o.when for o in original if shows_in_role(o, c)]
            if proof:
                c = replace(c, effective_date=min(proof), effective_pending=False,
                            effective_inferred=True)
        claims.append(c)

    support: dict[tuple[int, str], list[Claim]] = defaultdict(list)
    upcoming: dict[tuple[int, str], list[Claim]] = defaultdict(list)
    departures: dict[int, list[Claim]] = defaultdict(list)
    for c in claims:
        key = role_key(c)
        if c.claim_type in ("role", "appointment") and key:
            if c.claim_type == "appointment" and (c.pending or c.when > today):
                if c.tier <= 3:                     # not in the role yet: kept as announced
                    upcoming[(c.person_id, key)].append(c)
                continue
            support[(c.person_id, key)].append(c)
        elif c.claim_type == "departure" and c.tier <= 3:   # snippets alone change nothing
            departures[c.person_id].append(c)
            if c.departure_scope == "role" and key:
                support.setdefault((c.person_id, key), [])  # role known only from its end

    # Indexes for single-holder roles.
    live_by_role: dict[str, list[Claim]] = defaultdict(list)
    appointed_by_role: dict[str, list[Claim]] = defaultdict(list)
    for c in claims:
        if not c.canonical_role:
            continue
        if c.is_live_listing:
            live_by_role[c.canonical_role].append(c)
        elif (c.claim_type == "appointment" and c.tier <= 2 and c.when <= today
              and not c.pending):
            appointed_by_role[c.canonical_role].append(c)

    def is_fresh(c: Claim) -> bool:
        return c.tier <= 3 and c.is_dated and c.when >= cutoff

    def successor(pid: int, canonical: str | None, sup: list[Claim]) -> tuple[Claim, str] | None:
        if not canonical:
            return None
        listed = any(s.is_live_listing for s in sup)
        dated = [s.when for s in sup if s.is_dated and not s.is_live_listing]
        latest = max(dated, default=None)
        live_holders = {c.person_id for c in live_by_role[canonical]}
        for a in sorted(appointed_by_role[canonical], key=lambda c: (c.when, c.id)):
            if a.person_id == pid:
                continue
            if latest is not None and a.when <= latest:
                continue                            # holder is confirmed after that appointment
            if listed and a.person_id in live_holders:
                continue                            # page shows both: co-holders, not succession
            return a, R3_APPOINTED
        if not listed and not any(is_fresh(s) for s in sup):
            others = [c for c in live_by_role[canonical] if c.person_id != pid]
            if others:
                return _best(others), R3_LISTED
        return None

    roles: list[Role] = []
    for (pid, key), sup in sorted(support.items()):
        deps = [
            d for d in departures[pid]
            if (d.departure_scope == "company" or role_key(d) == key)
            # A later appointment to this role (a return) cancels an earlier departure.
            and not any(a.claim_type == "appointment" and a.when > d.when for a in sup)
        ]
        ref = _best(sup) if sup else _best(deps)
        canonical = next((c.canonical_role for c in sup + deps if c.canonical_role), None)
        seen = [c.when for c in sup] or [d.when for d in deps]
        role = Role(
            person_id=pid, key=key,
            title=ref.raw_title or key, level=ref.level, department=ref.department,
            canonical_role=canonical, status=UNVERIFIED, status_reason=R6_WEAK,
            first_seen=min(seen), last_seen=max(seen),
            source_count=len({c.source_id for c in sup}),
            supporting_claim_id=ref.id,
        )

        strong = [d for d in deps if d.tier <= 2 and not d.pending]
        weak = [d for d in deps if d.tier == 3]
        # The company said the person will leave, gave no date, and said so recently.
        undated = [d for d in deps if d.tier <= 2 and d.pending and d.as_of_date >= cutoff]
        listed = [s for s in sup if s.is_live_listing]

        if strong:                                                  # rule 1
            d = min(strong, key=lambda c: (c.when, c.id))
            # The end date is the one the source states. Without one, only the date
            # of the report is known; it is kept apart and never shown as the end date.
            role.ended_on, role.ended_by_claim_id = d.stated_date, d.id
            role.end_reported_on = None if d.stated_date else d.as_of_date
            if d.when <= today:
                role.status, role.status_reason = FORMER, R1_END
            else:
                role.status, role.status_reason = CURRENT, R1_FUTURE
        elif weak:                                                  # rule 2
            d = _best(weak)
            role.status, role.status_reason = UNVERIFIED, R2_REPORTED
            role.conflict, role.conflicting_claim_id = True, d.id
            role.conflict_reason = (
                "departure reported by news only; company page still lists the role" if listed
                else "departure reported by news only; no company source confirms it")
        elif (succ := successor(pid, canonical, sup)):              # rule 3
            claim, reason = succ
            role.status, role.status_reason = FORMER, reason
            role.ended_by_claim_id = claim.id
            if reason == R3_APPOINTED:
                role.ended_on = claim.stated_date
                role.end_reported_on = None if claim.stated_date else claim.when
        elif undated:                                               # rule 1, no date yet
            role.status, role.status_reason = CURRENT, R1_PENDING
            role.ended_by_claim_id = _best(undated).id
        elif listed:                                                # rule 4
            role.status, role.status_reason = CURRENT, R4_LISTED
            role.supporting_claim_id = _best(listed).id
        elif any(is_fresh(s) for s in sup):                         # rule 5
            role.status, role.status_reason = CURRENT, R5_SUPPORTED
            role.supporting_claim_id = _best([s for s in sup if is_fresh(s)]).id
        # else rule 6: stays unverified
        roles.append(role)

    # Rule 7: more than one current holder of a single-holder role.
    def shared_on_page(r: Role) -> bool:
        return any(s.is_live_listing and is_shared_title(s.raw_title)
                   for s in support[(r.person_id, r.key)])

    def strength(r: Role) -> int:
        sup = support[(r.person_id, r.key)]
        return min((s.tier for s in sup if s.is_live_listing or is_fresh(s)), default=4)

    holders: dict[str, list[Role]] = defaultdict(list)
    for r in roles:
        if r.canonical_role and r.status == CURRENT:
            holders[r.canonical_role].append(r)
    for canonical, group in holders.items():
        if len({r.person_id for r in group}) < 2:
            continue
        if all(shared_on_page(r) for r in group):
            continue                                # the company page shows a shared role
        best_tier = min(strength(r) for r in group)
        winners = [r for r in group if strength(r) == best_tier]
        for r in group:
            other = next(w for w in winners + group if w.person_id != r.person_id)
            r.conflict, r.conflicting_claim_id = True, other.supporting_claim_id
            if strength(r) == best_tier:
                r.conflict_reason = (f"another person is also claimed as {canonical}; "
                                     "sources are equally reliable" if len(winners) > 1
                                     else f"another person is also claimed as {canonical} "
                                          "by a less reliable source")
            else:
                r.status, r.status_reason = UNVERIFIED, R7_OUTRANKED
                r.conflict_reason = (f"another person holds {canonical} according to a more "
                                     f"reliable source (tier {best_tier} vs tier {strength(r)})")

    # Rule 0: appointments that have not taken effect. Never counted as current.
    next_holder: dict[str, list[Claim]] = defaultdict(list)
    for (pid, key), anns in sorted(upcoming.items()):
        if (pid, key) in support:
            continue                                # the person already has a row for this role
        a = min(anns, key=lambda c: (c.pending, c.tier, c.when, c.id))
        stale = a.pending and a.as_of_date < cutoff
        roles.append(Role(
            person_id=pid, key=key, title=a.raw_title or key, level=a.level,
            department=a.department, canonical_role=a.canonical_role,
            status=UNVERIFIED if stale else ANNOUNCED,
            status_reason=R0_STALE if stale else R0_ANNOUNCED,
            starts_on=None if a.pending else a.when,
            first_seen=min(c.as_of_date for c in anns), last_seen=max(c.as_of_date for c in anns),
            source_count=len({c.source_id for c in anns}), supporting_claim_id=a.id,
        ))
        if a.canonical_role and not stale:
            next_holder[a.canonical_role].append(a)
    for r in roles:                                 # show the successor beside the holder
        if r.status == CURRENT and r.canonical_role:
            others = [a for a in next_holder[r.canonical_role] if a.person_id != r.person_id]
            if others:
                r.successor_claim_id = min(
                    others, key=lambda c: (c.pending, c.tier, c.when, c.id)).id
    roles.sort(key=lambda r: (r.person_id, r.key))

    # Employment status: derived per person, separately from any single role.
    employment: dict[int, str] = {}
    by_person: dict[int, list[Role]] = defaultdict(list)
    for r in roles:
        by_person[r.person_id].append(r)
    for pid in sorted({c.person_id for c in claims}):
        if any(r.status == CURRENT for r in by_person[pid]):
            employment[pid] = "current"
            continue
        left = [d for d in departures[pid]
                if d.departure_scope == "company" and d.tier <= 2 and d.when <= today
                and not d.pending]
        returned = any(c.person_id == pid and c.claim_type == "appointment"
                       and any(c.when > d.when for d in left) for c in claims)
        employment[pid] = "former" if left and not returned else "unknown"

    return Resolution(roles=roles, employment=employment)

import random
import re
from datetime import date

import pytest
from worked_cases import (CASES, CEO, CFO, CTO, TODAY, X, Y, _c, appointment, departure,
                          filing, live, matches, news, release)

from leadership.resolve import (ANNOUNCED, CURRENT, UNVERIFIED, is_shared_title, resolve,
                                role_key)


def _by_key(result):
    return {(r.person_id, r.key): r for r in result.roles}


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_worked_case(case):
    result = resolve(case["claims"], case.get("today", TODAY))
    got = {k: (r.status, r.status_reason) for k, r in _by_key(result).items()}
    assert got == case["expect"]
    assert result.employment == case["employment"]
    assert {k for k, r in _by_key(result).items() if r.conflict} == case.get("conflict", set())
    assert matches(case, result)


def test_every_role_names_the_rule_that_decided_it():
    for case in CASES:
        for r in resolve(case["claims"], case.get("today", TODAY)).roles:
            assert r.status_reason.startswith("rule")


def test_resolve_is_pure_and_order_independent():
    for case in CASES:
        claims = list(case["claims"])
        today = case.get("today", TODAY)
        first = resolve(claims, today)
        shuffled = claims[:]
        random.Random(7).shuffle(shuffled)
        assert resolve(shuffled, today) == first       # input order does not matter
        assert resolve(claims, today) == first         # same input, same output
        assert claims == list(case["claims"])          # input not modified


def test_news_only_departure_records_the_evidence_and_the_reason():
    with_page = [live(1, X, "CTO", **CTO),
                 departure(2, X, "role", tier=3, src="news", on=date(2026, 10, 1), title="CTO", **CTO)]
    r = _by_key(resolve(with_page, TODAY))[(X, "CTO")]
    assert r.conflict and r.conflicting_claim_id == 2
    assert "company page still lists the role" in r.conflict_reason

    without_page = [release(1, X, "CTO", date(2026, 3, 1), **CTO),
                    departure(2, X, "role", tier=3, src="news", on=date(2026, 9, 1), title="CTO", **CTO)]
    r = _by_key(resolve(without_page, TODAY))[(X, "CTO")]
    assert r.status == UNVERIFIED                      # never 'former' on news alone
    assert r.conflict and r.conflicting_claim_id == 2
    assert "no company source confirms" in r.conflict_reason


def test_lower_tier_source_raises_a_conflict_instead_of_overriding():
    roles = _by_key(resolve([live(1, X, "CTO", **CTO),
                             news(2, Y, "CTO", date(2026, 10, 10), **CTO)], TODAY))
    loser, winner = roles[(Y, "CTO")], roles[(X, "CTO")]
    assert winner.status == CURRENT and winner.conflict
    assert loser.conflict and loser.conflicting_claim_id == 1      # points at X's listing
    assert "more reliable source (tier 1 vs tier 3)" in loser.conflict_reason


def test_equal_tier_holders_both_stay_current_and_are_flagged():
    roles = resolve([live(1, X, "CEO", **CEO), live(2, Y, "CEO", **CEO)], TODAY).roles
    assert all(r.status == CURRENT and r.conflict for r in roles)
    assert all("equally reliable" in r.conflict_reason for r in roles)


def test_co_ceos_on_the_company_page_are_current_without_a_conflict():
    roles = resolve([live(1, X, "Co-CEO", **CEO), live(2, Y, "Co-CEO", **CEO)], TODAY).roles
    assert len(roles) == 2
    for r in roles:
        assert r.status == CURRENT and not r.conflict
        assert r.conflict_reason is None and r.conflicting_claim_id is None


def test_shared_title_in_a_filing_or_release_alone_does_not_clear_the_conflict():
    roles = resolve([filing(1, X, "Co-CEO", date(2026, 6, 1), **CEO),
                     release(2, Y, "Co-CEO", date(2026, 6, 1), **CEO)], TODAY).roles
    assert all(r.conflict for r in roles)              # only the live page settles it


@pytest.mark.parametrize("title, shared", [
    ("Co-CEO", True), ("co-ceo", True), ("Co CEO", True), ("Joint CEO", True),
    ("Co-Chief Executive Officer", True), ("Co-founder and Co-CEO", True),
    ("Co-President", True), ("Co-Head of Engineering", True),
    ("CEO", False), ("Co-founder and CEO", False), ("Chief Executive Officer", False),
    ("Cofounder", False), ("Joint venture lead", False), (None, False),
])
def test_is_shared_title(title, shared):
    assert is_shared_title(title) is shared


def test_concurrent_roles_of_one_person_are_not_a_conflict():
    roles = resolve([live(1, X, "Co-founder and CEO", **CEO),
                     filing(2, X, "Chair of the Board", date(2026, 4, 1), level="other")], TODAY).roles
    assert len(roles) == 2 and not any(r.conflict for r in roles)


def test_explicit_end_records_date_and_evidence():
    r = _by_key(resolve([live(1, X, "CTO", **CTO),
                         departure(2, X, "role", tier=2, src="press_release", on=date(2026, 9, 15),
                                   eff=date(2026, 9, 30), title="CTO", **CTO)], TODAY))[(X, "CTO")]
    assert r.ended_on == date(2026, 9, 30)             # effective date, not publication date
    assert r.ended_by_claim_id == 2


def test_succession_records_the_appointment_and_leaves_other_roles_alone():
    roles = _by_key(resolve([
        filing(1, X, "CFO", date(2026, 2, 1), source=10, **CFO),
        filing(2, X, "President", date(2026, 2, 1), source=10),
        appointment(3, Y, "CFO", tier=2, src="press_release", on=date(2026, 7, 15),
                    eff=date(2026, 8, 1), **CFO)], TODAY))
    assert roles[(X, "CFO")].ended_on == date(2026, 8, 1)
    assert roles[(X, "CFO")].ended_by_claim_id == 3
    assert roles[(X, "president")].status == CURRENT


def test_appointment_does_not_end_a_holder_confirmed_after_it():
    roles = _by_key(resolve([
        appointment(1, Y, "CFO", tier=2, src="press_release", on=date(2026, 3, 1), **CFO),
        filing(2, X, "CFO", date(2026, 6, 1), **CFO)], TODAY))
    assert roles[(X, "CFO")].status == CURRENT         # the filing is later than the appointment
    assert roles[(X, "CFO")].conflict and roles[(Y, "CFO")].conflict


def test_future_appointment_is_announced_not_current():
    result = resolve([appointment(1, X, "CTO", tier=2, src="press_release", on=date(2026, 10, 1),
                                  eff=date(2027, 1, 1), **CTO)], TODAY)
    (r,) = result.roles
    assert r.status == ANNOUNCED and r.starts_on == date(2027, 1, 1)
    assert r.supporting_claim_id == 1
    assert result.employment == {X: "unknown"}         # not yet an executive here


def test_announced_roles_are_never_counted_as_current():
    for case in CASES:
        for r in resolve(case["claims"], case.get("today", TODAY)).roles:
            assert not (r.status == CURRENT and r.status_reason.startswith("rule0"))
            if r.status == ANNOUNCED:
                assert r.status_reason.startswith("rule0") and not r.conflict


def test_announced_successor_is_shown_beside_the_current_holder():
    dated = [filing(1, X, "CFO", date(2026, 2, 1), **CFO),
             appointment(2, Y, "CFO", tier=2, src="press_release", on=date(2026, 10, 15),
                         eff=date(2027, 1, 1), **CFO)]
    x = _by_key(resolve(dated, TODAY))[(X, "CFO")]
    assert x.status == CURRENT and x.successor_claim_id == 2
    assert not x.conflict and x.ended_on is None       # a note, not a conflict or an end

    undated = [dated[0], appointment(2, Y, "CFO", tier=2, src="press_release",
                                     on=date(2026, 9, 1), pending=True, **CFO)]
    roles = _by_key(resolve(undated, TODAY))
    assert roles[(X, "CFO")].successor_claim_id == 2
    assert roles[(Y, "CFO")].starts_on is None         # no date was given

    after = _by_key(resolve(dated, date(2027, 1, 2)))  # once it took effect, the note is gone
    assert after[(Y, "CFO")].successor_claim_id is None


def test_stale_or_snippet_announcements_are_not_shown_as_a_successor():
    stale = [filing(1, X, "CFO", date(2026, 2, 1), **CFO),
             appointment(2, Y, "CFO", tier=2, src="press_release", on=date(2025, 8, 1),
                         pending=True, **CFO)]
    assert _by_key(resolve(stale, TODAY))[(X, "CFO")].successor_claim_id is None

    snippet_only = [filing(1, X, "CFO", date(2026, 2, 1), **CFO),
                    _c(2, Y, "appointment", "CFO", tier=4, src="search_snippet", on=TODAY,
                       basis="retrieved", pending=True, **CFO)]
    roles = _by_key(resolve(snippet_only, TODAY))
    assert (Y, "CFO") not in roles and roles[(X, "CFO")].successor_claim_id is None


def test_undated_appointment_takes_effect_on_the_date_of_the_confirming_source():
    roles = _by_key(resolve([
        filing(1, X, "CFO", date(2026, 2, 1), **CFO),
        appointment(2, Y, "CFO", tier=2, src="press_release", on=date(2026, 9, 1),
                    pending=True, **CFO),
        news(3, Y, "CFO", date(2026, 10, 10), **CFO)], TODAY))
    x = roles[(X, "CFO")]
    assert x.status == "former" and x.ended_by_claim_id == 2
    # No source states the day. Known only: a source dated 2026-10-10 shows the successor in office.
    assert x.ended_on is None and x.end_reported_on == date(2026, 10, 10)   # not the announcement date


def test_source_older_than_the_announcement_does_not_confirm_it():
    roles = _by_key(resolve([
        filing(1, X, "CFO", date(2026, 2, 1), **CFO),
        news(2, Y, "CFO", date(2026, 8, 1), **CFO),            # before the announcement
        appointment(3, Y, "CFO", tier=2, src="press_release", on=date(2026, 9, 1),
                    pending=True, **CFO)], TODAY))
    assert roles[(X, "CFO")].status == CURRENT


def test_undated_departure_keeps_the_role_and_records_the_announcement():
    r = _by_key(resolve([
        live(1, X, "CTO", **CTO),
        departure(2, X, "role", tier=2, src="press_release", on=date(2026, 9, 15),
                  title="CTO", pending=True, **CTO)], TODAY))[(X, "CTO")]
    assert r.status == CURRENT and r.ended_on is None
    assert r.ended_by_claim_id == 2                    # the evidence of the planned departure


def test_source_count_counts_independent_sources():
    r = resolve([live(1, X, "CTO", source=1, **CTO),
                 release(2, X, "Chief Technology Officer", date(2026, 5, 1), source=2, **CTO),
                 release(3, X, "CTO", date(2026, 5, 1), source=2, **CTO)], TODAY).roles[0]
    assert r.source_count == 2
    assert r.supporting_claim_id == 1                  # the most reliable one
    assert r.first_seen == date(2026, 5, 1) and r.last_seen == TODAY


def test_freshness_window_is_configurable():
    claims = [news(1, X, "VP of Engineering", date(2025, 6, 1), level="vp", dept="engineering")]
    assert resolve(claims, TODAY, freshness_months=12).roles[0].status == UNVERIFIED
    assert resolve(claims, TODAY, freshness_months=24).roles[0].status == CURRENT


def test_freshness_boundary_is_inclusive():
    exactly = [news(1, X, "CTO", date(2025, 11, 1), **CTO)]
    day_older = [news(1, X, "CTO", date(2025, 10, 31), **CTO)]
    assert resolve(exactly, TODAY).roles[0].status == CURRENT
    assert resolve(day_older, TODAY).roles[0].status == UNVERIFIED


def test_role_key_groups_title_variants():
    a = release(1, X, "CTO", TODAY, **CTO)
    b = release(2, X, "Chief Technology Officer", TODAY, **CTO)
    c = news(3, X, "VP, Engineering", TODAY, level="vp", dept="engineering")
    d = news(4, X, "SVP of Engineering", TODAY, level="vp", dept="engineering")
    e = live(5, X, "Board  Advisor!", level="other")
    assert role_key(a) == role_key(b) == "CTO"
    assert role_key(c) == role_key(d) == "vp:engineering"
    assert role_key(e) == "board advisor"


def test_conflict_messages_are_cleanly_worded():
    """No run-together words ('CTOby'), doubled spaces or stray edges in any message."""
    seen = set()
    for case in CASES:
        for r in resolve(case["claims"], case.get("today", TODAY)).roles:
            assert bool(r.conflict) == bool(r.conflict_reason)
            if r.conflict_reason:
                seen.add(r.conflict_reason)
    assert len(seen) >= 5                              # every kind of message is exercised
    for msg in seen:
        assert msg == msg.strip() and "  " not in msg
        assert not re.search(r"[A-Z]{2,}[a-z]", msg), msg      # e.g. 'CTOby', 'CEOaccording'
        assert not re.search(r"[a-z;,)][A-Z(]", msg), msg      # e.g. 'asCTO', 'source(tier'
    roles = _by_key(resolve([live(1, X, "CTO", **CTO),
                             news(2, Y, "CTO", date(2026, 10, 10), **CTO)], TODAY))
    assert roles[(X, "CTO")].conflict_reason == (
        "another person is also claimed as CTO by a less reliable source")
    assert roles[(Y, "CTO")].conflict_reason == (
        "another person holds CTO according to a more reliable source (tier 1 vs tier 3)")


# ------------------------------------------------- end dates (found in M3 review)
def test_an_end_date_is_the_one_the_source_states_not_the_filing_date():
    """Robinhood: a filing dated 2026-02-10 says the CFO ceased on 2026-02-06."""
    roles = _by_key(resolve([
        filing(1, X, "CFO", date(2025, 2, 18), **CFO),
        departure(2, X, "role", tier=1, src="sec_filing", on=date(2026, 2, 10),
                  eff=date(2026, 2, 6), title="CFO", **CFO)], TODAY))
    x = roles[(X, "CFO")]
    assert x.status == "former" and x.ended_on == date(2026, 2, 6) and x.end_reported_on is None


def test_a_departure_without_a_stated_date_gets_no_end_date():
    """The same filing, when the sentence with the date was not extracted."""
    roles = _by_key(resolve([
        filing(1, X, "CFO", date(2025, 2, 18), **CFO),
        departure(2, X, "role", tier=1, src="sec_filing", on=date(2026, 2, 10), title="CFO", **CFO)], TODAY))
    x = roles[(X, "CFO")]
    assert x.status == "former" and x.ended_by_claim_id == 2
    assert x.ended_on is None                               # the filing date is not an end date
    assert x.end_reported_on == date(2026, 2, 10)           # only: reported by a source of that date


def test_an_earlier_title_given_as_past_has_no_end_date():
    """'Prior to his current position he had been our SVP since 2018' states no end date."""
    svp = dict(level="vp", dept="finance")
    roles = _by_key(resolve([
        departure(1, X, "role", tier=1, src="sec_filing", on=date(2026, 4, 22),
                  title="Senior Vice President of Finance", **svp)], TODAY))
    (r,) = roles.values()
    assert r.status == "former" and r.ended_on is None and r.end_reported_on == date(2026, 4, 22)


def test_succession_without_a_stated_date_gives_no_end_date():
    roles = _by_key(resolve([
        _c(1, X, "role", "CLO", tier=2, src="press_release", on=date(2019, 11, 21), canonical="CLO"),
        appointment(2, Y, "CLO", tier=2, src="press_release", on=date(2020, 8, 4), canonical="CLO")], TODAY))
    x = roles[(X, "CLO")]
    assert x.status == "former" and x.status_reason == "rule3_successor_appointed"
    assert x.ended_on is None and x.end_reported_on == date(2020, 8, 4)
    dated = _by_key(resolve([
        _c(1, X, "role", "CLO", tier=2, src="press_release", on=date(2019, 11, 21), canonical="CLO"),
        appointment(2, Y, "CLO", tier=2, src="press_release", on=date(2020, 5, 5),
                    eff=date(2020, 6, 1), canonical="CLO")], TODAY))
    assert dated[(X, "CLO")].ended_on == date(2020, 6, 1) and dated[(X, "CLO")].end_reported_on is None


def test_leaving_later_this_month_never_ends_a_role_on_the_announcement_date():
    """Robinhood 2020: 'our current Chief Legal Officer, who is leaving later this month'."""
    claims = [
        _c(1, X, "appointment", "CLO", tier=2, src="press_release", on=date(2019, 11, 21), canonical="CLO"),
        departure(2, X, "company", tier=2, src="press_release", on=date(2020, 5, 5), title="CLO",
                  canonical="CLO", pending=True),
        appointment(3, Y, "CLO", tier=2, src="press_release", on=date(2020, 5, 5), canonical="CLO",
                    pending=True),
        _c(4, Y, "role", "CLO", tier=2, src="press_release", on=date(2020, 8, 4), canonical="CLO")]
    x = _by_key(resolve(claims, TODAY))[(X, "CLO")]
    assert x.status == "former"                             # a later source shows the successor in office
    assert x.ended_on is None                               # no source gives the day she left
    assert x.end_reported_on == date(2020, 8, 4) != date(2020, 5, 5)
    # While the announcement was fresh she was still the holder.
    then = _by_key(resolve(claims[:3], date(2020, 5, 10)))[(X, "CLO")]
    assert then.status == "current" and then.status_reason == "rule1_departure_announced_without_date"
    assert then.ended_on is None


def test_no_role_ever_gets_an_end_date_that_no_claim_states():
    """Over every worked case: an end date always equals some claim's stated effective date."""
    for case in CASES:
        stated = {c.effective_date for c in case["claims"] if c.effective_date}
        for r in resolve(case["claims"], TODAY).roles:
            assert r.ended_on is None or r.ended_on in stated, case["name"]
            assert not (r.ended_on and r.end_reported_on), case["name"]


def test_a_company_wide_departure_does_not_lend_its_office_to_the_persons_other_roles():
    """Found in M4: 'leaving Robinhood' named the CLO office, which then showed on an unrelated role."""
    roles = _by_key(resolve([
        _c(1, X, "role", "member of our executive leadership team", tier=2, src="press_release",
           on=date(2019, 11, 21), level="other"),
        _c(2, X, "role", "Chief Legal Officer", tier=2, src="press_release", on=date(2019, 11, 21), canonical="CLO"),
        departure(3, X, "company", tier=2, src="press_release", on=date(2026, 5, 5), eff=date(2026, 5, 20),
                  title="Chief Legal Officer", canonical="CLO")], TODAY))
    other = next(r for k, r in roles.items() if k[1] != "CLO")
    assert other.canonical_role is None and other.status == "former"       # ended, but not renamed
    assert roles[(X, "CLO")].canonical_role == "CLO" and roles[(X, "CLO")].status == "former"

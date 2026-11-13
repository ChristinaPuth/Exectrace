"""Title and name rules. SPEC sections 3.4 and 3.5. No network, no keys."""

import pytest

from leadership.normalize import canonical_role, compare_names, has_division, level, normalize_name


@pytest.mark.parametrize("title, expected", [
    ("CEO", "c_level"), ("Chief Executive Officer", "c_level"), ("Co-Founder & CEO", "c_level"),
    ("Chief Compliance Officer", "c_level"), ("President", "c_level"), ("General Counsel", "c_level"),
    ("SVP, Engineering", "vp"), ("EVP of Sales", "vp"), ("Senior Vice President", "vp"),
    ("Executive Vice President of Sales", "vp"), ("VP of Sales", "vp"), ("Vice-President, Product", "vp"),
    ("Head of Product", "head"), ("Global Head of Communications", "head"),
    ("Director", "other"), ("Director of Engineering", "other"), ("Board member", "other"),
    ("Chairman of the Board", "other"), ("Founder", "other"), ("Chief of Staff", "other"),
    ("Software Engineer", "other"), (None, "other"), ("", "other"),
])
def test_level_from_title(title, expected):
    assert level(title) == expected


def test_a_vice_president_is_not_a_president():
    assert level("Vice President") == "vp" and level("President and COO") == "c_level"


@pytest.mark.parametrize("title, expected", [
    ("CEO", "CEO"), ("Chief Executive Officer", "CEO"), ("President and CEO", "CEO"),
    ("Co-Founder & CEO", "CEO"), ("CEO and Co-Founder", "CEO"), ("Co-CEO", "CEO"),
    ("Chief Executive Officer and Chair of the Board", "CEO"),
    ("Chief Executive Officer, President and Director", "CEO"),
    ("Chief Financial Officer", "CFO"), ("EVP & CFO", "CFO"), ("Chief Technology Officer", "CTO"),
    ("Chief Operating Officer", "COO"), ("Chief Marketing Officer", "CMO"), ("Chief Legal Officer", "CLO"),
    ("Chief Compliance Officer", "Chief Compliance Officer"),
    ("Chief Brokerage Officer", "Chief Brokerage Officer"),
    ("Co-Founder and Chief Creative Officer", "Chief Creative Officer"),
    ("President", None), ("SVP, Engineering", None), ("Head of Product", None),
    ("Director", None), ("Chief of Staff", None), (None, None),
])
def test_canonical_role(title, expected):
    assert canonical_role(title) == expected


def test_the_head_of_one_division_is_not_the_company_wide_holder():
    assert canonical_role("CEO of Robinhood Crypto", "Robinhood") is None
    assert canonical_role("Chief Executive Officer, Robinhood Crypto", "Robinhood") is None
    assert has_division("President, Robinhood Money", "Robinhood")
    # Naming the company itself is not a division.
    assert canonical_role("Chief Executive Officer of Robinhood Markets, Inc.", "Robinhood") == "CEO"
    assert canonical_role("CEO, Campfire", "Campfire") == "CEO"
    assert canonical_role("Chief Technology Officer at Campfire", "Campfire") == "CTO"


def test_other_roles_joined_by_punctuation_are_not_divisions():
    assert canonical_role("Founder CEO / Co-founder", "Campfire") == "CEO"      # as a page rendered it
    assert canonical_role("founder and CEO (and still CFO)", "Campfire") == "CEO"
    assert canonical_role("CEO - Founder", "Campfire") == "CEO"


def test_the_same_office_spelled_two_ways_is_one_role():
    a = canonical_role("Chief Legal, Compliance, and Corporate Affairs Officer")
    b = canonical_role("Chief Legal, Compliance and Corporate Affairs Officer")
    c = canonical_role("Chief Legal, Compliance & Corporate Affairs Officer")
    assert a == b == c == "Chief Legal Compliance and Corporate Affairs Officer"


def test_a_title_naming_two_offices_uses_the_first():
    assert canonical_role("CEO, CFO, and Founder", "Campfire") == "CEO"
    assert canonical_role("Chief Operating Officer and Chief Financial Officer") == "COO"


@pytest.mark.parametrize("name, expected", [
    ("Vladimir Tenev", "vladimir tenev"), ("  Vlad   Tenev ", "vlad tenev"),
    ("Dr. Naomi Boyd", "naomi boyd"), ("Jason K. Warnick", "jason warnick"),
    ("José Álvarez Jr.", "jose alvarez"), ("JOHN GLASGOW", "john glasgow"),
    ("Baiju Prafulkumar Bhatt", "baiju prafulkumar bhatt"), ('Daniel "Dan" Gallagher', "daniel gallagher"),
    ("Mary-Anne O'Neil, CPA", "mary-anne o'neil"),
])
def test_normalize_name(name, expected):
    assert normalize_name(name) == expected


def test_name_comparison():
    assert compare_names("jason warnick", "jason warnick") == "same"
    assert compare_names("vlad tenev", "vladimir tenev") == "nickname"        # merged only with a matching role
    assert compare_names("j warnick", "jason warnick") == "near"              # flagged, never merged
    assert compare_names("baiju bhatt", "baiju prafulkumar bhatt") == "near"
    assert compare_names("john smith", "jane smith") == "different"           # same surname is not enough
    assert compare_names("vlad tenev", "vlad bhatt") == "different"
    assert compare_names("madonna", "madonna") == "same" and compare_names("madonna", "madonna x") == "different"

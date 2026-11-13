"""The scoring used by `leadership eval`, and the labelled fixtures. No model, no network."""

import json

from leadership.evals import LABELLED, same_title, score_page
from leadership.extract import Accepted, ExtractedClaim


def _a(person, title, claim_type="role", **kw):
    claim = ExtractedClaim(person=person, organization="Robinhood", claim_type=claim_type,
                           raw_title=title, evidence=f"{person} is {title} at Robinhood", **kw)
    return Accepted(claim=claim, version="v", level="c_level", canonical_role=None)


PAGE = {"file": "p.txt",
        "expected": [{"person": "Jeff Pinner", "title": "CTO", "claim_type": "appointment"},
                     {"person": "Vlad Tenev", "title": "CEO"},
                     {"person": "Shiv Verma", "title": "principal accounting officer",
                      "claim_type": "departure", "effective_date": "2026-06-25"}],
        "optional": [{"person": "Vlad Tenev", "title": "Co-Founder"}],
        "forbidden_people": ["Ron Wangerin"]}


def test_titles_match_by_role_or_by_the_labels_words():
    assert same_title("CTO", "Chief Technology Officer (CTO)", "Robinhood")
    assert same_title("CEO", "CEO and Co-Founder", "Robinhood")
    assert not same_title("CEO", "Chief Financial Officer", "Robinhood")
    assert same_title("Vice President of Operations", "Vice President of Operations", "Robinhood")
    assert same_title("principal accounting officer", "the Company's principal accounting officer", "Robinhood")
    assert not same_title("principal accounting officer", "principal financial officer", "Robinhood")


def test_a_perfect_page_scores_full_marks():
    s = score_page(PAGE, [_a("Jeff Pinner", "Chief Technology Officer", "appointment"),
                          _a("Vlad Tenev", "CEO and Co-Founder"), _a("Vlad Tenev", "Co-Founder"),
                          _a("Shiv Verma", "principal accounting officer", "departure",
                             effective_date="2026-06-25")], "Robinhood")
    assert (s.expected, s.found, s.matched) == (3, 3, 3)            # the optional pair is not counted
    assert s.missed == s.wrong == s.forbidden == s.detail_errors == []


def test_misses_extras_outsiders_and_wrong_details_are_each_reported():
    s = score_page(PAGE, [_a("Jeff Pinner", "CTO", "role"),                     # right pair, wrong claim type
                          _a("Ron Wangerin", "Chief Financial Officer"),        # a customer
                          _a("Shiv Verma", "principal accounting officer", "departure")], "Robinhood")
    assert (s.expected, s.found, s.matched) == (3, 3, 2)
    assert s.missed == ["Vlad Tenev / CEO"]
    assert s.wrong == ["ron wangerin / Chief Financial Officer"] and s.forbidden == ["ron wangerin"]
    assert len(s.detail_errors) == 2
    assert "claim_type should be appointment" in s.detail_errors[0]
    assert "effective_date should be 2026-06-25" in s.detail_errors[1]


def test_labelled_fixtures_are_complete_and_consistent():
    labels = json.loads((LABELLED / "labels.json").read_text(encoding="utf-8"))
    assert len(labels["pages"]) == 5
    for page in labels["pages"]:
        text = (LABELLED / page["file"]).read_text(encoding="utf-8")
        assert len(text) > 500 and page["company"] in labels["companies"]
        # Every labelled or forbidden person really is named on the saved page.
        for label in page["expected"] + page.get("optional", []):
            assert label["person"] in text, (page["file"], label["person"])
        for name in page.get("forbidden_people", []):
            assert name in text, (page["file"], name)
    assert sum(len(p["expected"]) for p in labels["pages"]) == 11

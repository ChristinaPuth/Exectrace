"""Gold-set measurement and the fixtures export. No model, no network."""

import json
import sqlite3
from datetime import date

import pytest

from leadership.db import connect, rebuild_roles
from leadership.export import company_document, export
from leadership.gold import load, measure

TODAY = date(2026, 11, 13)
GOLD = {"company": "Acme Robotics", "domain": "acme-robotics.com", "leaders": [
    {"name": "Vlad Tenev", "also_written": ["Vladimir Tenev"], "title": "Chairman & Chief Executive Officer",
     "role": "CEO", "level": "c_level", "counts_toward": ["c_level"], "scope": "company", "unit": None,
     "status": "current", "verified": True},
    {"name": "Evan McHenry", "title": "Vice President and Chief Information Security Officer", "role": "CISO",
     "level": "c_level", "counts_toward": ["c_level", "vp"], "scope": "company", "unit": None,
     "status": "current", "verified": True},
    {"name": "Walter Koller", "title": "Vice President and Chief Operating Officer, Acme Financial", "role": None,
     "level": "c_level", "counts_toward": ["c_level", "vp"], "scope": "subsidiary_or_unit",
     "unit": "Acme Financial", "status": "current", "verified": True},
    {"name": "Pooja Anand", "title": "Vice President, Enterprise Risk Management", "role": None, "level": "vp",
     "counts_toward": ["vp"], "scope": "company", "unit": None, "status": "current", "verified": True},
    {"name": "Brendan Doyle", "title": "Head of Product", "role": None, "level": "head",
     "counts_toward": ["head"], "scope": "company", "unit": None, "status": "current", "verified": True},
    {"name": "Jason Warnick", "title": "Chief Financial Officer", "role": "CFO", "level": "c_level",
     "counts_toward": ["c_level"], "scope": "company", "unit": None, "status": "former", "verified": True},
], "secondary_candidates": [{"name": "Chris McDonough", "title": "VP of Sales"}]}


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    c.execute("INSERT INTO companies (id, domain, name) VALUES (1, 'acme-robotics.com', 'Acme Robotics')")
    c.execute("INSERT INTO companies (id, domain, name) VALUES (2, 'other.com', 'Other')")
    c.execute("INSERT INTO sources (id, company_id, url, source_type, tier, published_at, retrieved_at, stage) "
              "VALUES (1,1,'https://sec.example/proxy','sec_filing',1,'2026-09-01','2026-11-13T00:00:00+00:00','extracted')")
    c.execute("INSERT INTO sources (id, company_id, url, source_type, tier, published_at, retrieved_at, stage) "
              "VALUES (2,1,'https://acme-robotics.com/press/old','press_release',2,'2020-08-04','2026-11-13T00:00:00+00:00','extracted')")
    c.execute("INSERT INTO sources (id, company_id, url, source_type, tier, retrieved_at) "
              "VALUES (3,2,'https://other.com/','company_site',1,'2026-11-13T00:00:00+00:00')")
    c.execute("INSERT INTO runs (company_id, started_at, status) VALUES (1, '2026-11-13T00:00:00+00:00', 'complete')")
    return c


def _person(conn, pid, name, source, kind, title, canonical, level, **kw):
    conn.execute("INSERT OR IGNORE INTO people (id, company_id, full_name, normalized_name) VALUES (?,1,?,?)",
                 (pid, name, name.lower()))
    conn.execute(
        """INSERT INTO claims (company_id, person_id, source_id, claim_type, departure_scope, raw_title, level,
               department, canonical_role, value, evidence, as_of_date, as_of_basis, extraction_version,
               person_as_written)
           VALUES (1,?,?,?,?,?,?,?,?,?,?,?,'published','v1',?)""",
        (pid, source, kind, kw.get("scope"), title, level, kw.get("dept", "other"), canonical, kw.get("value"),
         f"{name}, {title}", "2026-09-01" if source == 1 else "2020-08-04", name))


def _seed(conn):
    _person(conn, 1, "Vladimir Tenev", 1, "role", "Chair and Chief Executive Officer", "CEO", "c_level")
    _person(conn, 2, "Evan McHenry", 1, "role", "Chief Information Security Officer", "CISO", "c_level")
    _person(conn, 3, "Pooja Anand", 2, "role", "Vice President, Enterprise Risk Management", None, "vp")   # old: unverified
    _person(conn, 4, "Jason Warnick", 1, "role", "Chief Financial Officer", "CFO", "c_level")              # gold: former
    _person(conn, 5, "Chris McDonough", 1, "role", "VP of Sales", None, "vp", dept="sales")
    _person(conn, 6, "Dara Bazzano", 1, "role", "Chief Accounting Officer", "Chief Accounting Officer", "c_level")
    _person(conn, 1, "Vladimir Tenev", 1, "location", None, None, None, value="Menlo Park, California")
    conn.commit()
    rebuild_roles(conn, 1, TODAY)


def test_recall_is_counted_per_group_and_a_vp_chief_counts_in_both(conn):
    _seed(conn)
    report = measure(conn, GOLD)
    c_level, vp, head = report.recall["c_level"], report.recall["vp"], report.recall["head"]
    assert (c_level["gold"], c_level["current"]) == (3, 2)             # Tenev by nickname, McHenry; not Koller
    assert c_level["missing"] == ["Walter Koller"]
    assert (vp["gold"], vp["current"]) == (3, 1)                       # McHenry counts toward VP too
    assert vp["other_status"] == ["Pooja Anand (unverified)"]          # found, but not as current
    assert vp["missing"] == ["Walter Koller"]
    assert (head["gold"], head["current"], head["missing"]) == (1, 0, ["Brendan Doyle"])
    assert report.recall_of("c_level") == pytest.approx(2 / 3)


def test_a_gold_former_leader_stored_as_current_is_reported_and_counted_wrong(conn):
    _seed(conn)
    report = measure(conn, GOLD)
    assert report.former_as_current == ["Jason Warnick / Chief Financial Officer"]
    verdicts = {r["person"]: r["verdict"] for r in report.review}
    assert verdicts == {"Vladimir Tenev": "correct", "Evan McHenry": "correct", "Jason Warnick": "wrong",
                        "Chris McDonough": "not counted", "Dara Bazzano": "to review"}
    correct, wrong, open_ = report.judged
    assert (correct, wrong, open_) == (2, 1, 2)                        # unjudged roles are never counted as correct
    assert report.precision == pytest.approx(2 / 3)
    assert "Pooja Anand" not in verdicts                               # unverified: not a current role at all


def test_hand_verdicts_settle_roles_the_gold_set_does_not_cover(conn):
    _seed(conn)
    report = measure(conn, GOLD, {"Dara Bazzano | Chief Accounting Officer": "correct"})
    assert report.judged == (3, 1, 1) and report.precision == pytest.approx(3 / 4)


def test_a_unit_office_stored_as_the_company_wide_office_is_a_scope_error(conn):
    _person(conn, 7, "Walter Koller", 1, "role", "Chief Operating Officer", "COO", "c_level")   # the unit was lost
    conn.commit()
    rebuild_roles(conn, 1, TODAY)
    report = measure(conn, GOLD)
    assert len(report.scope_errors) == 1 and "Acme Financial" in report.scope_errors[0]
    conn.execute("DELETE FROM roles")
    conn.execute("UPDATE claims SET raw_title = 'Chief Operating Officer, Acme Financial', canonical_role = NULL")
    conn.commit()
    rebuild_roles(conn, 1, TODAY)
    kept = measure(conn, GOLD)
    assert kept.scope_errors == [] and kept.recall["c_level"]["missing"] == ["Evan McHenry", "Vlad Tenev"]


def test_a_gold_file_with_unverified_rows_cannot_be_used(tmp_path):
    path = tmp_path / "acme.json"
    draft = json.loads(json.dumps(GOLD))
    draft["leaders"][0]["verified"] = False
    path.write_text(json.dumps(draft), encoding="utf-8")
    with pytest.raises(ValueError, match="not yet verified by a person: Vlad Tenev"):
        load(path)


def test_the_committed_gold_sets_are_verified_and_well_formed():
    from leadership.config import ROOT
    for name, rows, current_people in (("robinhood.com.json", 28, 25), ("meetcampfire.com.json", 3, 3)):
        gold = load(ROOT / "fixtures" / "gold" / name)
        assert len(gold["leaders"]) == rows and "not fully blind" in gold["disclosure"].lower()
        assert len({g["name"] for g in gold["leaders"] if g["status"] == "current"}) == current_people
        for g in gold["leaders"]:
            assert g["quote"] and g["source"] in gold["sources"] and g["scope"] in ("company", "subsidiary_or_unit")
            assert (g["scope"] == "subsidiary_or_unit") == bool(g["unit"])
            assert not (g["unit"] and g["role"])                       # a unit office is never company-wide
        assert all(s.get("accessed") for s in gold["sources"].values())


# ---------------------------------------------------------------------- export
def test_export_document_carries_every_role_with_its_claims_and_sources(conn):
    _seed(conn)
    doc = company_document(conn, "acme-robotics.com")
    assert doc["company"] == "Acme Robotics" and doc["run"]["status"] == "complete"
    tenev = next(p for p in doc["people"] if p["name"] == "Vladimir Tenev")
    (role,) = tenev["roles"]
    assert role["status"] == "current" and role["canonical_role"] == "CEO" and role["decided_by"].startswith("rule")
    (claim,) = role["claims"]
    assert claim["source_url"] == "https://sec.example/proxy" and claim["evidence"] and claim["tier"] == 1
    assert tenev["location"] == {"value": "Menlo Park, California", "status": "found",
                                 "evidence": "Vladimir Tenev, None", "source_url": "https://sec.example/proxy"}
    anand = next(p for p in doc["people"] if p["name"] == "Pooja Anand")
    assert anand["roles"][0]["status"] == "unverified" and anand["location"] is None
    assert {s["url"] for s in doc["sources"]} == {"https://sec.example/proxy", "https://acme-robotics.com/press/old"}
    json.dumps(doc)                                                    # plain data throughout


def test_export_writes_json_and_a_database_with_only_the_chosen_companies(conn, tmp_path):
    _seed(conn)
    conn.close()
    written = export(tmp_path / "t.db", tmp_path / "fixtures", ["acme-robotics.com"])
    assert [p.name for p in written] == ["acme-robotics.com.json", "leadership.db"]
    saved = json.loads(written[0].read_text(encoding="utf-8"))
    assert len(saved["people"]) == 6
    db = sqlite3.connect(written[1])
    assert [r[0] for r in db.execute("SELECT domain FROM companies")] == ["acme-robotics.com"]
    assert db.execute("SELECT count(*) FROM sources").fetchone()[0] == 2          # the other company's page is gone
    assert db.execute("SELECT count(*) FROM current_leadership").fetchone()[0] == 5    # the views still work
    db.close()
    with pytest.raises(KeyError):
        company_document(connect(tmp_path / "t.db"), "no-such.com")


# ---------------------------------------------------------- reachable-source recall
def test_reachable_recall_leaves_out_only_people_whose_every_cited_source_was_refused(conn):
    _seed(conn)
    conn.execute("INSERT INTO sources (company_id, url, source_type, tier, retrieved_at, stage, last_error) VALUES "
                 "(1, 'https://ir.acme-robotics.com/leadership', 'company_site', 1, '2026-11-13T00:00:00+00:00', "
                 "'skipped', 'disallowed by robots.txt')")
    conn.commit()
    gold = json.loads(json.dumps(GOLD))
    gold["sources"] = {"page": {"url": "https://ir.acme-robotics.com/leadership/"},       # trailing slash: same page
                       "proxy": {"url": "https://sec.example/proxy"},
                       "never": {"url": "https://acme-robotics.com/team"}}               # never asked for
    by_name = {g["name"]: g for g in gold["leaders"]}
    by_name["Vlad Tenev"].update(source="page", also_supported_by={"source": "proxy", "quote": "q"})
    by_name["Evan McHenry"]["source"] = "page"           # found anyway: reachable by definition
    by_name["Walter Koller"]["source"] = "page"          # only on the refused page: unreachable
    by_name["Pooja Anand"]["source"] = "page"            # found, but unverified: still a miss, and unreachable
    by_name["Brendan Doyle"]["source"] = "never"         # not attempted: no evidence, stays in
    report = measure(conn, gold)
    c_level, vp, head = report.recall["c_level"], report.recall["vp"], report.recall["head"]
    assert (c_level["gold"], c_level["reachable"], c_level["current"]) == (3, 2, 2)
    assert report.recall_of("c_level") == pytest.approx(2 / 3)          # the full figure is unchanged
    assert report.reachable_recall_of("c_level") == 1.0
    assert [u.split(":")[0] for u in c_level["unreachable"]] == ["Walter Koller"]
    assert "disallowed by robots.txt" in c_level["unreachable"][0]
    assert (vp["gold"], vp["reachable"], vp["current"]) == (3, 1, 1)    # Koller and Anand left out, McHenry found
    assert (head["gold"], head["reachable"], head["current"]) == (1, 1, 0)   # Doyle: never attempted, so still a miss
    assert report.reachable_recall_of("head") == 0.0


def test_without_cited_sources_nobody_is_unreachable(conn):
    _seed(conn)
    report = measure(conn, GOLD)                         # the test gold set cites no sources
    assert all(r["reachable"] == r["gold"] and not r["unreachable"] for r in report.recall.values())

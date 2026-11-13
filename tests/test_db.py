import json
import sqlite3
from datetime import date

import pytest

from leadership.db import connect, rebuild_roles

TODAY = date(2026, 11, 1)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.db")
    c.execute("INSERT INTO companies (id, domain, name) VALUES (1, 'example.com', 'Example')")
    c.executemany(
        "INSERT INTO sources (id, company_id, url, source_type, tier, published_at, retrieved_at)"
        " VALUES (?,1,?,?,?,?,'2026-11-01T10:00:00')",
        [(1, "https://example.com/team", "company_site", 1, None),
         (2, "https://example.com/press/cfo", "press_release", 2, "2026-09-15"),
         (3, "https://news.example.org/a", "news", 3, "2026-10-01")])
    c.executemany("INSERT INTO people (id, company_id, full_name, normalized_name) VALUES (?,1,?,?)",
                  [(1, "Ada Lovelace", "ada lovelace"), (2, "Grace Hopper", "grace hopper"),
                   (3, "Alan Turing", "alan turing")])
    return c


def _claim(conn, person, source, kind, *, title=None, canonical=None, level="c_level",
           as_of="2026-11-01", basis="retrieved", scope=None, value=None, effective=None,
           evidence="quoted text from the source", pending=0):
    conn.execute(
        """INSERT INTO claims (company_id, person_id, source_id, claim_type, departure_scope,
               raw_title, level, department, canonical_role, value, evidence, as_of_date,
               as_of_basis, effective_date, effective_pending, extraction_version)
           VALUES (1,?,?,?,?,?,?,'other',?,?,?,?,?,?,?,'v1')""",
        (person, source, kind, scope, title, level, canonical, value, evidence, as_of, basis,
         effective, pending))


def _seed(conn):
    _claim(conn, 1, 1, "role", title="Chief Executive Officer", canonical="CEO")
    _claim(conn, 1, 1, "location", value="London, UK")
    _claim(conn, 2, 1, "role", title="Chief Financial Officer", canonical="CFO")
    _claim(conn, 2, 2, "departure", title="CFO", canonical="CFO", scope="company",
           as_of="2026-09-15", basis="published", effective="2026-09-30")
    _claim(conn, 3, 3, "role", title="Board Member", level="other",
           as_of="2026-10-01", basis="published")
    conn.commit()


def test_schema_has_the_tables_and_views(conn):
    names = {r["name"]: r["type"] for r in
             conn.execute("SELECT name, type FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")}
    for table in ("companies", "runs", "sources", "people", "claims", "roles"):
        assert names[table] == "table"
    for view in ("leadership", "current_leadership"):
        assert names[view] == "view"


def test_connect_is_safe_to_repeat(tmp_path):
    connect(tmp_path / "a.db").close()
    connect(tmp_path / "a.db").close()


@pytest.mark.parametrize("missing", ["evidence", "as_of", "source"])
def test_claim_without_source_date_or_evidence_is_rejected(conn, missing):
    kwargs = dict(title="CEO", canonical="CEO")
    if missing == "evidence":
        kwargs["evidence"] = "   "
    if missing == "as_of":
        kwargs["as_of"] = None
    with pytest.raises(sqlite3.IntegrityError):
        _claim(conn, 1, 999 if missing == "source" else 1, "role", **kwargs)


def test_departure_without_scope_is_rejected(conn):
    with pytest.raises(sqlite3.IntegrityError):
        _claim(conn, 1, 2, "departure", title="CEO", canonical="CEO")


def test_invalid_enum_values_are_rejected(conn):
    with pytest.raises(sqlite3.IntegrityError):
        _claim(conn, 1, 1, "rumour", title="CEO")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO runs (company_id, started_at, status) VALUES (1,'now','finished')")


def test_rebuild_writes_roles_and_employment_status(conn):
    _seed(conn)
    assert rebuild_roles(conn, 1, TODAY) == 3
    roles = {r["person"]: r for r in conn.execute("SELECT * FROM leadership")}
    assert roles["Ada Lovelace"]["status"] == "current"
    assert roles["Ada Lovelace"]["status_reason"] == "rule4_listed_on_company_page"
    assert roles["Grace Hopper"]["status"] == "former"
    assert roles["Grace Hopper"]["ended_on"] == "2026-09-30"
    people = dict(conn.execute("SELECT full_name, employment_status FROM people"))
    assert people == {"Ada Lovelace": "current", "Grace Hopper": "former", "Alan Turing": "current"}


def test_rebuild_is_idempotent(conn):
    _seed(conn)
    dump = lambda: [tuple(r)[1:] for r in conn.execute("SELECT * FROM roles ORDER BY person_id, title")]  # noqa: E731
    rebuild_roles(conn, 1, TODAY)
    first = dump()
    conn.execute("DELETE FROM roles")
    rebuild_roles(conn, 1, TODAY)
    rebuild_roles(conn, 1, TODAY)
    assert dump() == first
    assert conn.execute("SELECT count(*) FROM roles").fetchone()[0] == 3


def test_views_expose_provenance_location_and_scope(conn):
    _seed(conn)
    rebuild_roles(conn, 1, TODAY)
    ada = conn.execute("SELECT * FROM current_leadership WHERE person = 'Ada Lovelace'").fetchone()
    assert ada["source_url"] == "https://example.com/team"
    assert ada["evidence"] == "quoted text from the source"
    assert ada["location"] == "London, UK"
    assert ada["location_source_url"] == "https://example.com/team"
    assert ada["source_count"] == 1
    # current_leadership holds only current, in-scope roles: no former CFO, no board member.
    assert [r["person"] for r in conn.execute("SELECT person FROM current_leadership")] == ["Ada Lovelace"]
    assert conn.execute("SELECT count(*) FROM leadership").fetchone()[0] == 3


def test_announced_role_is_visible_but_not_counted_as_current(conn):
    _claim(conn, 1, 1, "role", title="Chief Financial Officer", canonical="CFO")
    _claim(conn, 3, 2, "appointment", title="Chief Financial Officer", canonical="CFO",
           as_of="2026-09-15", basis="published", effective="2027-01-01")
    conn.commit()
    rebuild_roles(conn, 1, TODAY)
    assert [r["person"] for r in conn.execute("SELECT person FROM current_leadership")] == ["Ada Lovelace"]
    ada = conn.execute("SELECT * FROM current_leadership").fetchone()
    assert ada["successor"] == "Alan Turing"
    assert ada["successor_starts_on"] == "2027-01-01"
    assert ada["successor_source_url"] == "https://example.com/press/cfo"
    alan = conn.execute("SELECT * FROM leadership WHERE person = 'Alan Turing'").fetchone()
    assert alan["status"] == "announced" and alan["starts_on"] == "2027-01-01"
    assert alan["employment_status"] == "unknown"
    assert alan["source_url"] == "https://example.com/press/cfo"


def test_claims_are_kept_when_an_announcement_goes_stale(conn):
    _claim(conn, 1, 1, "role", title="Chief Financial Officer", canonical="CFO")
    _claim(conn, 3, 2, "appointment", title="Chief Financial Officer", canonical="CFO",
           as_of="2025-08-01", basis="published", pending=1)
    _claim(conn, 1, 2, "departure", title="Chief Financial Officer", canonical="CFO",
           scope="role", as_of="2025-08-01", basis="published", pending=1)
    conn.commit()
    before = [tuple(r) for r in conn.execute("SELECT * FROM claims ORDER BY id")]
    rebuild_roles(conn, 1, TODAY)
    rebuild_roles(conn, 1, TODAY)
    assert [tuple(r) for r in conn.execute("SELECT * FROM claims ORDER BY id")] == before
    alan = conn.execute("SELECT * FROM leadership WHERE person = 'Alan Turing'").fetchone()
    assert alan["status"] == "unverified"
    assert alan["status_reason"] == "rule0_announcement_never_confirmed"
    assert alan["evidence"] == "quoted text from the source"   # still traceable to its claim
    ada = conn.execute("SELECT * FROM leadership WHERE person = 'Ada Lovelace'").fetchone()
    assert ada["status"] == "current" and ada["successor"] is None


def test_every_current_role_has_a_supporting_claim(conn):
    _seed(conn)
    rebuild_roles(conn, 1, TODAY)
    orphans = conn.execute(
        "SELECT count(*) FROM roles WHERE status = 'current' AND supporting_claim_id IS NULL"
    ).fetchone()[0]
    assert orphans == 0


def test_runs_table_keeps_status_reasons_and_counters(conn):
    reasons = [{"code": "budget_exhausted", "stage": "fetch", "detail": "max_pages=40 reached"},
               {"code": "partial_failure", "stage": "fetch", "detail": "2 sources failed"}]
    conn.execute(
        "INSERT INTO runs (company_id, started_at, status, reasons, counters) VALUES (1,?,?,?,?)",
        ("2026-11-01T10:00:00", "partial_failure", json.dumps(reasons), json.dumps({"pages_fetched": 40})))
    row = conn.execute("SELECT * FROM runs").fetchone()
    assert row["status"] == "partial_failure"
    assert [r["code"] for r in json.loads(row["reasons"])] == ["budget_exhausted", "partial_failure"]
    assert json.loads(row["counters"])["pages_fetched"] == 40


def test_only_current_roles_appear_among_current_leaders(conn):
    """Unverified, former and announced roles are never returned as current leadership."""
    _seed(conn)
    _claim(conn, 3, 2, "role", title="Chief Operating Officer", canonical="COO",
           as_of="2020-08-04", basis="published")                   # an old press release only
    conn.commit()
    rebuild_roles(conn, 1, TODAY)
    statuses = dict(conn.execute("SELECT title, status FROM leadership WHERE person = 'Alan Turing'"))
    assert statuses["Chief Operating Officer"] == "unverified"
    listed = [tuple(r) for r in conn.execute("SELECT person, status FROM current_leadership")]
    assert listed == [("Ada Lovelace", "current")]
    assert conn.execute("SELECT count(*) FROM current_leadership WHERE status != 'current'").fetchone()[0] == 0


def test_end_date_and_report_date_are_separate_columns_in_the_view(conn):
    _seed(conn)                                                     # Grace: departure effective 2026-09-30
    _claim(conn, 3, 2, "role", title="CTO", canonical="CTO", as_of="2025-01-10", basis="published")
    _claim(conn, 3, 2, "departure", title="CTO", canonical="CTO", scope="role",
           as_of="2026-02-10", basis="published")                   # no effective date stated
    conn.commit()
    rebuild_roles(conn, 1, TODAY)
    rows = {r["person"]: (r["status"], r["ended_on"], r["end_reported_on"])
            for r in conn.execute("SELECT * FROM leadership WHERE canonical_role IN ('CFO', 'CTO')")}
    assert rows["Grace Hopper"] == ("former", "2026-09-30", None)
    assert rows["Alan Turing"] == ("former", None, "2026-02-10")    # the filing date is not shown as the end


def test_views_are_rebuilt_for_a_database_made_by_an_older_version(tmp_path):
    import sqlite3

    from leadership.db import connect
    path = tmp_path / "old.db"
    connect(path).close()
    raw = sqlite3.connect(path)
    raw.executescript("DROP VIEW current_leadership; DROP VIEW leadership; "
                      "CREATE VIEW leadership AS SELECT 1 AS old_shape;")
    raw.close()
    conn = connect(path)
    columns = [d[0] for d in conn.execute("SELECT * FROM leadership").description]
    assert "end_reported_on" in columns and "old_shape" not in columns
    conn.close()

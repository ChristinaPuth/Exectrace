"""The SQL guard. SPEC section 5. No model, no network."""

import sqlite3
from datetime import date

import pytest

from leadership.db import connect, rebuild_roles
from leadership.sqlguard import GuardError, open_readonly, run


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.db"
    c = connect(path)
    c.execute("INSERT INTO companies (id, domain, name) VALUES (1, 'acme.com', 'Acme')")
    c.execute("INSERT INTO sources (id, company_id, url, source_type, tier, published_at, retrieved_at) "
              "VALUES (1, 1, 'https://acme.com/team', 'company_site', 1, NULL, '2026-11-13T00:00:00')")
    for pid in range(1, 61):
        c.execute("INSERT INTO people (id, company_id, full_name, normalized_name) VALUES (?, 1, ?, ?)",
                  (pid, f"Person {pid}", f"person {pid}"))
        c.execute("""INSERT INTO claims (company_id, person_id, source_id, claim_type, raw_title, level, department,
                         evidence, as_of_date, as_of_basis, extraction_version)
                     VALUES (1, ?, 1, 'role', 'Vice President', 'vp', 'sales', 'Person; VP', '2026-11-13',
                             'retrieved', 'v1')""", (pid,))
    c.commit()
    rebuild_roles(c, 1, date(2026, 11, 13))
    c.close()
    conn = open_readonly(path)
    yield conn
    conn.close()


def test_reads_through_the_views_are_allowed(db):
    r = run(db, "SELECT person, title FROM current_leadership WHERE level = 'vp' ORDER BY person LIMIT 3")
    assert r.columns == ["person", "title"] and len(r.rows) == 3
    count = run(db, "SELECT COUNT(DISTINCT person) FROM leadership;")
    assert count.rows == [(60,)]


def test_rows_are_capped_and_the_cap_is_reported(db):
    r = run(db, "SELECT person FROM leadership", row_limit=50)
    assert len(r.rows) == 50 and r.truncated
    assert not run(db, "SELECT person FROM leadership LIMIT 5").truncated


@pytest.mark.parametrize("sql", [
    "SELECT * FROM people",                                     # a base table
    "SELECT full_name FROM people JOIN leadership ON 1 = 1",    # a base table beside a view
    "SELECT * FROM claims",
    "SELECT * FROM sqlite_master",
    "SELECT sql FROM sqlite_schema",
    "PRAGMA table_info(people)",
    "ATTACH DATABASE 'x.db' AS x",
    "DELETE FROM roles",
    "UPDATE people SET full_name = 'x'",
    "INSERT INTO companies (domain, name) VALUES ('a', 'b')",
    "DROP VIEW leadership",
    "CREATE TABLE t (x)",
    "BEGIN",
])
def test_everything_but_reading_the_views_is_refused(db, sql):
    with pytest.raises(GuardError):
        run(db, sql)


def test_only_one_statement_per_call(db):
    with pytest.raises(GuardError, match="one statement"):
        run(db, "SELECT 1 FROM leadership; DELETE FROM roles")
    with pytest.raises(GuardError, match="one statement"):
        run(db, "SELECT 1; SELECT 2")
    # A semicolon inside a string is not a second statement.
    assert run(db, "SELECT person FROM leadership WHERE title = 'a;b'").rows == []


def test_a_slow_query_is_stopped(db):
    slow = ("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) "
            "SELECT count(*) FROM n, leadership")
    with pytest.raises(GuardError, match="longer than"):
        run(db, slow, timeout_s=0.2)


def test_the_connection_itself_cannot_write(tmp_path):
    path = tmp_path / "w.db"
    sqlite3.connect(path).close()
    connect(path).close()
    conn = open_readonly(path)
    conn.set_authorizer(None)                                   # even without the authorizer
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("INSERT INTO leadership (person) VALUES ('Jane Doe')")
    assert conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall() == [
        ("current_leadership",), ("leadership",)]                # no base table exists in the snapshot


def test_errors_are_short_and_say_what_is_allowed(db):
    with pytest.raises(GuardError) as e:
        run(db, "SELECT * FROM people")
    assert "only the tables leadership and current_leadership exist" in str(e.value)
    with pytest.raises(GuardError) as e:
        run(db, "PRAGMA table_info(leadership)")
    assert "only SELECT queries on the views leadership and current_leadership" in str(e.value)
    with pytest.raises(GuardError, match="no such column"):
        run(db, "SELECT nonsense FROM leadership")


def test_a_snapshot_can_hold_one_company_only(tmp_path, db):
    from leadership.sqlguard import open_readonly as open_one
    path = tmp_path / "t.db"
    assert run(open_one(path, "acme.com"), "SELECT COUNT(*) FROM leadership").rows == [(60,)]
    assert run(open_one(path, "other.com"), "SELECT COUNT(*) FROM leadership").rows == [(0,)]

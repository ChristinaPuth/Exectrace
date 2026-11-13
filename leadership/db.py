"""SQLite schema and the bridge between stored claims and derived roles.

Claims are append-only observations. Roles and employment status are derived:
`rebuild_roles` deletes and recomputes them from claims with the pure function
`resolve.resolve`, so they can be regenerated at any time. See SPEC section 4.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from .normalize import compare_names, normalize_name
from .resolve import Claim, resolve, role_key

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id            INTEGER PRIMARY KEY,
    domain        TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    collected_at  TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY,
    company_id   INTEGER NOT NULL REFERENCES companies(id),
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT NOT NULL DEFAULT 'running' CHECK (status IN
                 ('running','complete','budget_exhausted','insufficient_data',
                  'partial_failure','interrupted')),
    reasons      TEXT NOT NULL DEFAULT '[]',   -- JSON list: every condition that applied
    counters     TEXT NOT NULL DEFAULT '{}'    -- JSON: usage against each limit
);

CREATE TABLE IF NOT EXISTS sources (
    id            INTEGER PRIMARY KEY,
    company_id    INTEGER NOT NULL REFERENCES companies(id),
    url           TEXT NOT NULL,
    source_type   TEXT NOT NULL CHECK (source_type IN
                  ('company_site','sec_filing','press_release','news','wikipedia','search_snippet')),
    tier          INTEGER NOT NULL CHECK (tier BETWEEN 1 AND 4),
    title         TEXT,
    published_at  TEXT,                         -- NULL when unknown; never guessed
    retrieved_at  TEXT NOT NULL,
    content_hash  TEXT,
    stage         TEXT NOT NULL DEFAULT 'fetched' CHECK (stage IN
                  ('fetched','extracted','failed','skipped')),
    attempts      INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    extraction_version  TEXT,                   -- of the last extraction; NULL if never extracted
    page_kind     TEXT,                         -- official | article | profile_or_directory | other
    UNIQUE (company_id, url)
);

CREATE TABLE IF NOT EXISTS people (
    id                 INTEGER PRIMARY KEY,
    company_id         INTEGER NOT NULL REFERENCES companies(id),
    full_name          TEXT NOT NULL,
    normalized_name    TEXT NOT NULL,
    linkedin_url       TEXT,
    employment_status  TEXT NOT NULL DEFAULT 'unknown' CHECK (employment_status IN
                       ('current','former','unknown')),     -- DERIVED
    possible_duplicate_of  INTEGER REFERENCES people(id)    -- a near-match name, kept separate
);

CREATE TABLE IF NOT EXISTS claims (
    id                  INTEGER PRIMARY KEY,
    company_id          INTEGER NOT NULL REFERENCES companies(id),
    person_id           INTEGER NOT NULL REFERENCES people(id),
    source_id           INTEGER NOT NULL REFERENCES sources(id),
    claim_type          TEXT NOT NULL CHECK (claim_type IN
                        ('role','appointment','departure','location','bio')),
    departure_scope     TEXT CHECK (departure_scope IN ('role','company')),
    raw_title           TEXT,
    level               TEXT CHECK (level IN ('c_level','vp','head','other')),
    department          TEXT,
    canonical_role      TEXT,
    value               TEXT,                   -- location / bio text
    evidence            TEXT NOT NULL CHECK (length(trim(evidence)) > 0),
    as_of_date          TEXT NOT NULL,          -- ISO date
    as_of_basis         TEXT NOT NULL CHECK (as_of_basis IN ('published','retrieved')),
    effective_date      TEXT,                   -- stated in the text, if any
    effective_pending   INTEGER NOT NULL DEFAULT 0 CHECK (effective_pending IN (0, 1)),
                                                -- 1: a future change with no date given
    extraction_version  TEXT NOT NULL,
    person_as_written   TEXT,                   -- the name as this source wrote it ("Vlad Tenev")
    CHECK (claim_type <> 'departure' OR departure_scope IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS roles (                -- DERIVED, rebuilt by rebuild_roles
    id                    INTEGER PRIMARY KEY,
    company_id            INTEGER NOT NULL REFERENCES companies(id),
    person_id             INTEGER NOT NULL REFERENCES people(id),
    title                 TEXT NOT NULL,
    level                 TEXT,
    department            TEXT,
    canonical_role        TEXT,
    status                TEXT NOT NULL CHECK (status IN
                          ('current','former','unverified','announced')),
    status_reason         TEXT NOT NULL,        -- the rule in SPEC 3.6 that decided
    starts_on             TEXT,                 -- announced roles: stated start date, if any
    ended_on              TEXT,                 -- only a date a source states
    end_reported_on       TEXT,                 -- no stated date: date of the source reporting the end
    ended_by_claim_id     INTEGER REFERENCES claims(id),   -- departure or successor evidence
    first_seen            TEXT,
    last_seen             TEXT,
    last_verified_at      TEXT,
    conflict              INTEGER NOT NULL DEFAULT 0,
    conflict_reason       TEXT,
    conflicting_claim_id  INTEGER REFERENCES claims(id),   -- evidence on the other side
    source_count          INTEGER NOT NULL DEFAULT 0,
    supporting_claim_id   INTEGER REFERENCES claims(id),
    successor_claim_id    INTEGER REFERENCES claims(id)    -- announcement of the next holder
);

-- The two views below are the only objects the chat agent may read.
-- Announced rows (appointments not yet in effect) appear only in the first.
CREATE VIEW IF NOT EXISTS leadership AS
SELECT
    c.domain               AS company_domain,
    c.name                 AS company,
    p.full_name            AS person,
    p.employment_status    AS employment_status,
    r.title, r.level, r.department, r.canonical_role,
    r.status, r.status_reason, r.starts_on, r.ended_on, r.end_reported_on,
    r.first_seen, r.last_seen, r.last_verified_at,
    r.conflict, r.conflict_reason, r.source_count,
    (SELECT l.value FROM claims l JOIN sources ls ON ls.id = l.source_id
      WHERE l.person_id = p.id AND l.claim_type = 'location'
      ORDER BY ls.tier, l.as_of_date DESC, l.id LIMIT 1)        AS location,
    (SELECT ls.url FROM claims l JOIN sources ls ON ls.id = l.source_id
      WHERE l.person_id = p.id AND l.claim_type = 'location'
      ORDER BY ls.tier, l.as_of_date DESC, l.id LIMIT 1)        AS location_source_url,
    (SELECT l.value FROM claims l JOIN sources ls ON ls.id = l.source_id
      WHERE l.person_id = p.id AND l.claim_type = 'bio'
      ORDER BY ls.tier, l.as_of_date DESC, l.id LIMIT 1)        AS bio,
    s.url                  AS source_url,
    s.source_type          AS source_type,
    sc.as_of_date          AS source_date,
    substr(sc.evidence, 1, 300) AS evidence,
    cs.url                 AS conflicting_source_url,
    substr(cc.evidence, 1, 300) AS conflicting_evidence,
    np.full_name           AS successor,
    nc.effective_date      AS successor_starts_on,
    ns.url                 AS successor_source_url
FROM roles r
JOIN people p     ON p.id = r.person_id
JOIN companies c  ON c.id = r.company_id
LEFT JOIN claims sc  ON sc.id = r.supporting_claim_id
LEFT JOIN sources s  ON s.id = sc.source_id
LEFT JOIN claims cc  ON cc.id = r.conflicting_claim_id
LEFT JOIN sources cs ON cs.id = cc.source_id
LEFT JOIN claims nc  ON nc.id = r.successor_claim_id
LEFT JOIN people np  ON np.id = nc.person_id
LEFT JOIN sources ns ON ns.id = nc.source_id;

CREATE VIEW IF NOT EXISTS current_leadership AS
SELECT * FROM leadership
WHERE status = 'current' AND level IN ('c_level', 'vp', 'head');
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (and create if needed) the database with the schema applied."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # The views hold no data, so they are rebuilt on every opening and always
    # match this code, also in a database made by an older version.
    conn.executescript("DROP VIEW IF EXISTS current_leadership; DROP VIEW IF EXISTS leadership;")
    tables, _, views = SCHEMA.partition("-- The two views below")
    conn.executescript(tables)
    for table, column, kind in _ADDED_COLUMNS:          # databases made by an older version
        if column not in {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
    conn.executescript("-- The two views below" + views)
    conn.commit()
    return conn


_ADDED_COLUMNS = [("sources", "extraction_version", "TEXT"), ("sources", "page_kind", "TEXT"),
                  ("people", "possible_duplicate_of", "INTEGER REFERENCES people(id)"),
                  ("claims", "person_as_written", "TEXT"), ("roles", "end_reported_on", "TEXT")]


def find_person(conn: sqlite3.Connection, company_id: int, full_name: str,
                key: str | None = None) -> int:
    """The person this name refers to within one company, created if new. SPEC 3.5.

    The same normalized name is the same person. A nickname form ("Vlad" for
    "Vladimir") is merged only when the surname matches and both hold the role
    `key`. Any other near match stays a separate person and is flagged.
    """
    name = normalize_name(full_name)
    people = conn.execute("SELECT * FROM people WHERE company_id = ? ORDER BY id",
                          (company_id,)).fetchall()
    near: int | None = None
    for p in people:
        relation = compare_names(name, p["normalized_name"])
        if relation == "same":
            return p["id"]
        if relation == "nickname" and key and key in _role_keys(conn, p["id"]):
            if len(full_name) > len(p["full_name"]):    # keep the fuller form of the name
                conn.execute("UPDATE people SET full_name = ?, normalized_name = ? WHERE id = ?",
                             (full_name.strip(), name, p["id"]))
            return p["id"]
        if relation in ("nickname", "near") and near is None:
            near = p["id"]
    cur = conn.execute(
        "INSERT INTO people (company_id, full_name, normalized_name, possible_duplicate_of) "
        "VALUES (?,?,?,?)", (company_id, " ".join(full_name.split()), name, near))
    return cur.lastrowid


def _role_keys(conn: sqlite3.Connection, person_id: int) -> set[str]:
    rows = conn.execute("SELECT raw_title, level, department, canonical_role FROM claims "
                        "WHERE person_id = ? AND claim_type IN ('role','appointment','departure')",
                        (person_id,)).fetchall()
    return {k for r in rows if (k := claim_key(r["raw_title"], r["level"], r["department"],
                                               r["canonical_role"]))}


def claim_key(raw_title, level, department, canonical) -> str | None:
    """The role key the resolution rules would give a claim with these fields."""
    return role_key(Claim(id=0, person_id=0, source_id=0, claim_type="role", tier=1,
                          source_type="", as_of_date=date.min, as_of_basis="published",
                          raw_title=raw_title, level=level, department=department,
                          canonical_role=canonical))


def store_document(conn: sqlite3.Connection, company_id: int, source_id: int, accepted: list,
                   *, version: str, page_kind: str, as_of_date: str, as_of_basis: str,
                   tier: int | None = None) -> int:
    """Replace one document's claims in a single transaction. Returns the number stored.

    Older claims of the document, from any extraction version, are removed
    first, so nothing stale survives a prompt or model change. Derived roles
    are dropped as well; the caller rebuilds them at the end of the run.
    """
    with conn:
        conn.execute("DELETE FROM roles WHERE company_id = ?", (company_id,))
        conn.execute("DELETE FROM claims WHERE source_id = ?", (source_id,))
        seen: set[tuple] = set()
        stored = 0
        for a in accepted:
            c = a.claim
            department = c.department if c.claim_type != "location" else None
            key = claim_key(c.raw_title, a.level, department, a.canonical_role)
            person_id = find_person(conn, company_id, c.person, key)
            same = (person_id, c.claim_type, key, c.effective_date, c.value)
            if same in seen:                            # said twice in one document
                continue
            seen.add(same)
            stored += 1
            conn.execute(
                """INSERT INTO claims (company_id, person_id, source_id, claim_type, departure_scope,
                       raw_title, level, department, canonical_role, value, evidence, as_of_date,
                       as_of_basis, effective_date, effective_pending, extraction_version,
                       person_as_written)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (company_id, person_id, source_id, c.claim_type, c.departure_scope,
                 c.raw_title, a.level if c.raw_title else None, department, a.canonical_role,
                 c.value, " ".join(c.evidence.split()), as_of_date, as_of_basis,
                 c.effective_date, int(c.effective_pending), a.version, " ".join(c.person.split())))
        conn.execute("UPDATE sources SET stage = 'extracted', extraction_version = ?, page_kind = ?, "
                     "last_error = NULL, tier = COALESCE(?, tier) WHERE id = ?",
                     (version, page_kind, tier, source_id))
    return stored


def drop_people_without_claims(conn: sqlite3.Connection, company_id: int) -> None:
    """People left with no claim after a re-extraction are removed."""
    with conn:
        conn.execute("UPDATE people SET possible_duplicate_of = NULL WHERE company_id = ? AND "
                     "possible_duplicate_of NOT IN (SELECT DISTINCT person_id FROM claims)", (company_id,))
        conn.execute("DELETE FROM people WHERE company_id = ? AND id NOT IN "
                     "(SELECT DISTINCT person_id FROM claims)", (company_id,))


def load_claims(conn: sqlite3.Connection, company_id: int) -> list[Claim]:
    rows = conn.execute(
        """SELECT cl.*, s.tier, s.source_type
             FROM claims cl JOIN sources s ON s.id = cl.source_id
            WHERE cl.company_id = ? ORDER BY cl.id""",
        (company_id,),
    ).fetchall()
    return [
        Claim(
            id=r["id"], person_id=r["person_id"], source_id=r["source_id"],
            claim_type=r["claim_type"], tier=r["tier"], source_type=r["source_type"],
            as_of_date=date.fromisoformat(r["as_of_date"]), as_of_basis=r["as_of_basis"],
            effective_date=date.fromisoformat(r["effective_date"]) if r["effective_date"] else None,
            raw_title=r["raw_title"], level=r["level"], department=r["department"],
            canonical_role=r["canonical_role"], departure_scope=r["departure_scope"],
            effective_pending=bool(r["effective_pending"]),
        )
        for r in rows
    ]


def rebuild_roles(conn: sqlite3.Connection, company_id: int, today: date,
                  freshness_months: int = 12) -> int:
    """Recompute roles and employment status for one company from its claims.

    Idempotent: running it twice gives the same rows. Returns the role count.
    """
    result = resolve(load_claims(conn, company_id), today, freshness_months)
    iso = lambda d: d.isoformat() if d else None  # noqa: E731
    with conn:
        conn.execute("DELETE FROM roles WHERE company_id = ?", (company_id,))
        conn.executemany(
            """INSERT INTO roles (company_id, person_id, title, level, department,
                   canonical_role, status, status_reason, ended_on, ended_by_claim_id,
                   first_seen, last_seen, conflict, conflict_reason,
                   conflicting_claim_id, source_count, supporting_claim_id,
                   starts_on, successor_claim_id, end_reported_on)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(company_id, r.person_id, r.title, r.level, r.department, r.canonical_role,
              r.status, r.status_reason, iso(r.ended_on), r.ended_by_claim_id,
              iso(r.first_seen), iso(r.last_seen), int(r.conflict), r.conflict_reason,
              r.conflicting_claim_id, r.source_count, r.supporting_claim_id,
              iso(r.starts_on), r.successor_claim_id, iso(r.end_reported_on))
             for r in result.roles],
        )
        conn.execute("UPDATE people SET employment_status = 'unknown' WHERE company_id = ?",
                     (company_id,))
        conn.executemany(
            "UPDATE people SET employment_status = ? WHERE id = ? AND company_id = ?",
            [(status, pid, company_id) for pid, status in result.employment.items()],
        )
    return len(result.roles)

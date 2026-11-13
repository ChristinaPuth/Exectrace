"""SQLite schema and the bridge between stored claims and derived roles.

Claims are append-only observations. Roles and employment status are derived:
`rebuild_roles` deletes and recomputes them from claims with the pure function
`resolve.resolve`, so they can be regenerated at any time. See SPEC section 4.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from .resolve import Claim, resolve

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
    UNIQUE (company_id, url)
);

CREATE TABLE IF NOT EXISTS people (
    id                 INTEGER PRIMARY KEY,
    company_id         INTEGER NOT NULL REFERENCES companies(id),
    full_name          TEXT NOT NULL,
    normalized_name    TEXT NOT NULL,
    linkedin_url       TEXT,
    employment_status  TEXT NOT NULL DEFAULT 'unknown' CHECK (employment_status IN
                       ('current','former','unknown'))      -- DERIVED
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
    ended_on              TEXT,
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
    r.status, r.status_reason, r.starts_on, r.ended_on,
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
    conn.executescript(SCHEMA)
    return conn


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
                   starts_on, successor_claim_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(company_id, r.person_id, r.title, r.level, r.department, r.canonical_role,
              r.status, r.status_reason, iso(r.ended_on), r.ended_by_claim_id,
              iso(r.first_seen), iso(r.last_seen), int(r.conflict), r.conflict_reason,
              r.conflicting_claim_id, r.source_count, r.supporting_claim_id,
              iso(r.starts_on), r.successor_claim_id)
             for r in result.roles],
        )
        conn.execute("UPDATE people SET employment_status = 'unknown' WHERE company_id = ?",
                     (company_id,))
        conn.executemany(
            "UPDATE people SET employment_status = ? WHERE id = ? AND company_id = ?",
            [(status, pid, company_id) for pid, status in result.employment.items()],
        )
    return len(result.roles)

"""Write the fixtures: one JSON file per company and a database holding only those companies. SPEC 3.9.

The JSON is for people and for diffing; the database is what the chat reads.
Every role carries the claims behind it, each with its source and quote.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .db import claim_key, connect


def _rows(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def company_document(conn: sqlite3.Connection, domain: str) -> dict:
    """Everything stored about one company, as plain data in a stable order."""
    company = conn.execute("SELECT * FROM companies WHERE domain = ?", (domain,)).fetchone()
    if company is None:
        raise KeyError(domain)
    cid = company["id"]
    run = conn.execute("SELECT * FROM runs WHERE company_id = ? ORDER BY id DESC LIMIT 1", (cid,)).fetchone()
    people = []
    for p in _rows(conn, "SELECT * FROM people WHERE company_id = ? ORDER BY full_name, id", (cid,)):
        claims = _rows(conn, """SELECT cl.*, s.url, s.tier, s.source_type FROM claims cl
                                JOIN sources s ON s.id = cl.source_id WHERE cl.person_id = ?
                                ORDER BY s.tier, cl.as_of_date DESC, cl.id""", (p["id"],))
        roles = []
        for r in _rows(conn, """SELECT * FROM roles WHERE person_id = ? ORDER BY CASE status WHEN 'current' THEN 0
                                WHEN 'announced' THEN 1 WHEN 'unverified' THEN 2 ELSE 3 END, title""", (p["id"],)):
            key = r["canonical_role"] or claim_key(r["title"], r["level"], r["department"], None)
            roles.append({
                "title": r["title"], "level": r["level"], "department": r["department"],
                "canonical_role": r["canonical_role"], "status": r["status"], "decided_by": r["status_reason"],
                "starts_on": r["starts_on"], "ended_on": r["ended_on"], "end_reported_on": r["end_reported_on"],
                "first_seen": r["first_seen"], "last_seen": r["last_seen"],
                "last_verified_at": r["last_verified_at"], "conflict": bool(r["conflict"]),
                "conflict_reason": r["conflict_reason"], "source_count": r["source_count"],
                "claims": [{
                    "type": c["claim_type"], "scope": c["departure_scope"], "title": c["raw_title"],
                    "name_as_written": c["person_as_written"], "evidence": c["evidence"],
                    "effective_date": c["effective_date"], "pending": bool(c["effective_pending"]),
                    "as_of_date": c["as_of_date"], "as_of_basis": c["as_of_basis"], "source_url": c["url"],
                    "tier": c["tier"], "source_type": c["source_type"]}
                    for c in claims if c["claim_type"] in ("role", "appointment", "departure")
                    and claim_key(c["raw_title"], c["level"], c["department"], c["canonical_role"]) == key]})

        def fact(kind: str) -> dict | None:
            c = next((c for c in claims if c["claim_type"] == kind), None)
            check = conn.execute("SELECT outcome FROM checks WHERE person_id = ? AND kind = ? ORDER BY id DESC "
                                 "LIMIT 1", (p["id"], kind)).fetchone()
            if c:
                return {"value": c["value"], "status": "found", "evidence": c["evidence"], "source_url": c["url"]}
            return {"value": None, "status": check["outcome"], "evidence": None, "source_url": None} if check else None

        other = conn.execute("SELECT full_name FROM people WHERE id = ?", (p["possible_duplicate_of"],)).fetchone() \
            if p["possible_duplicate_of"] else None
        people.append({"name": p["full_name"], "employment_status": p["employment_status"],
                       "may_be_the_same_person_as": other["full_name"] if other else None,
                       "location": fact("location"), "bio": fact("bio"), "roles": roles})
    return {
        "company": company["name"], "domain": company["domain"], "collected_at": company["collected_at"],
        "run": {"status": run["status"], "reasons": json.loads(run["reasons"]),
                "counters": json.loads(run["counters"])} if run else None,
        "people": people,
        "sources": _rows(conn, """SELECT url, source_type, tier, title, published_at, retrieved_at, stage, page_kind,
                                  last_error FROM sources WHERE company_id = ? ORDER BY tier, url""", (cid,)),
        "checks": [{**c, "queries": json.loads(c["queries"])} for c in _rows(
            conn, """SELECT p.full_name AS person, k.kind, k.role_key, k.queries, k.outcome, k.detail, k.checked_at
                     FROM checks k LEFT JOIN people p ON p.id = k.person_id WHERE k.company_id = ?
                     ORDER BY k.kind, p.full_name, k.id""", (cid,))],
    }


def export(source_db: Path, out_dir: Path, domains: list[str]) -> list[Path]:
    """Write <domain>.json for each company and leadership.db with only those companies."""
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(source_db)
    written: list[Path] = []
    for domain in domains:
        path = out_dir / f"{domain}.json"
        path.write_text(json.dumps(company_document(conn, domain), indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")
        written.append(path)
    conn.close()

    target = out_dir / "leadership.db"
    target.unlink(missing_ok=True)
    src = sqlite3.connect(source_db)
    dst = sqlite3.connect(target)
    src.backup(dst)
    src.close()
    marks = ",".join("?" * len(domains))
    keep = f"(SELECT id FROM companies WHERE domain IN ({marks}))"
    dst.execute("PRAGMA foreign_keys = OFF")
    for table in ("roles", "checks", "claims", "people", "sources", "runs"):
        dst.execute(f"DELETE FROM {table} WHERE company_id NOT IN {keep}", domains)
    dst.execute(f"DELETE FROM companies WHERE domain NOT IN ({marks})", domains)
    dst.commit()
    dst.execute("VACUUM")
    dst.close()
    written.append(target)
    return written

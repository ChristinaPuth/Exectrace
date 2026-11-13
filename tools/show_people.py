"""Show the people, roles and claims stored for one company, with the evidence.

    uv run python tools/show_people.py robinhood.com            # leaders (C-level, VP, Head)
    uv run python tools/show_people.py robinhood.com --all      # also directors and others
    uv run python tools/show_people.py robinhood.com --rejected # claims the checks refused (newest log)

For manual review. Reads data/leadership.db.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from leadership.db import connect  # noqa: E402

ORDER = "CASE r.status WHEN 'current' THEN 0 WHEN 'announced' THEN 1 WHEN 'unverified' THEN 2 ELSE 3 END"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__)
        return 2
    domain = args[0].lower().removeprefix("https://").removeprefix("www.").strip("/")
    conn = connect(ROOT / "data" / "leadership.db")
    company = conn.execute("SELECT * FROM companies WHERE domain = ?", (domain,)).fetchone()
    if not company:
        print(f"No company stored for {domain}. Run: uv run leadership collect {domain}")
        return 1
    cid = company["id"]
    if "--rejected" in sys.argv:
        return rejected(domain)
    scope = "" if "--all" in sys.argv else "AND r.level IN ('c_level', 'vp', 'head')"
    roles = conn.execute(
        f"""SELECT r.*, p.full_name, p.employment_status, p.possible_duplicate_of
              FROM roles r JOIN people p ON p.id = r.person_id
             WHERE r.company_id = ? {scope} ORDER BY {ORDER}, r.level, p.full_name""", (cid,)).fetchall()
    print(f"Company: {company['name']} ({domain})    "
          f"{'all roles' if '--all' in sys.argv else 'leaders: C-level, VP, Head'}: {len(roles)}")
    for r in roles:
        print(f"\n{r['status'].upper():<11}{r['full_name']}  |  {r['title']}")
        print(f"    level {r['level']}, department {r['department']}, single-holder role {r['canonical_role']}, "
              f"employment {r['employment_status']}")
        extra = "".join(f", {k.replace('_', ' ')} {r[k]}" for k in ("starts_on", "ended_on") if r[k])
        if r["status"] == "former" and not r["ended_on"]:
            extra += (f", end date not stated (reported by a source dated {r['end_reported_on']})"
                      if r["end_reported_on"] else ", end date not stated")
        print(f"    decided by {r['status_reason']}; first seen {r['first_seen']}, last seen {r['last_seen']}, "
              f"{r['source_count']} source(s){extra}")
        if r["conflict"]:
            print(f"    CONFLICT: {r['conflict_reason']}")
        if r["possible_duplicate_of"]:
            other = conn.execute("SELECT full_name FROM people WHERE id = ?",
                                 (r["possible_duplicate_of"],)).fetchone()
            print(f"    NAME FLAG: may be the same person as {other['full_name']} (kept separate)")
        claims = conn.execute(
            """SELECT cl.*, s.url, s.tier, s.source_type, s.page_kind FROM claims cl
                 JOIN sources s ON s.id = cl.source_id
                WHERE cl.person_id = ? AND cl.claim_type IN ('role', 'appointment', 'departure')
                ORDER BY s.tier, cl.as_of_date DESC, cl.id""", (r["person_id"],)).fetchall()
        for c in claims:
            from leadership.db import claim_key
            if claim_key(c["raw_title"], c["level"], c["department"], c["canonical_role"]) != _key(r):
                continue
            when = f"effective {c['effective_date']}" if c["effective_date"] else (
                "future, no date" if c["effective_pending"] else "no effective date")
            scope_note = f" ({c['departure_scope']})" if c["departure_scope"] else ""
            print(f"      - {c['claim_type']}{scope_note}: \"{c['raw_title']}\"  tier {c['tier']} {c['source_type']}"
                  f"{' [' + c['page_kind'] + ']' if c['page_kind'] else ''}, as of {c['as_of_date']} "
                  f"({c['as_of_basis']}), {when}")
            print(f"        \"{c['evidence'][:220]}\"")
            print(f"        {c['url'][:110]}")
    counts = conn.execute("SELECT status, count(*) FROM roles WHERE company_id = ? GROUP BY 1", (cid,)).fetchall()
    people = conn.execute("SELECT count(*) FROM people WHERE company_id = ?", (cid,)).fetchone()[0]
    claims = conn.execute("SELECT count(*) FROM claims WHERE company_id = ?", (cid,)).fetchone()[0]
    print(f"\nTotals for {company['name']}: {people} people, {claims} claims; roles by status: "
          + ", ".join(f"{n} {s}" for s, n in counts))
    return 0


def _key(role) -> str:
    from leadership.db import claim_key
    return role["canonical_role"] or claim_key(role["title"], role["level"], role["department"], None)


def rejected(domain: str) -> int:
    logs = sorted((ROOT / "logs").glob(f"collect-{domain}-*.jsonl"))
    for log in reversed(logs):
        events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        refused = [e for e in events if e["event"] in ("claim_rejected", "document_rejected")]
        if refused:
            print(f"Rejected in {log.name}: {len(refused)}")
            for e in refused:
                who = f"{e.get('person', '(whole document)')} / {e.get('title')}"
                print(f"  {who[:60]:<62}{e['reason']}\n      {e['url'][:110]}")
            return 0
    print("No rejected claim in any run log for this company.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

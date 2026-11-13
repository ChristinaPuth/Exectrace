"""Show what collection stored for one company: runs, reasons and sources.

    uv run python tools/show_run.py robinhood.com

For manual review. Reads data/leadership.db and the newest run log.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from leadership.db import connect  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: show_run.py <company domain>")
        return 2
    domain = sys.argv[1].lower().removeprefix("https://").removeprefix("www.").strip("/")
    conn = connect(ROOT / "data" / "leadership.db")
    company = conn.execute("SELECT * FROM companies WHERE domain = ?", (domain,)).fetchone()
    if not company:
        print(f"No company stored for {domain}. Run: uv run leadership collect {domain}")
        return 1
    print(f"Company: {company['name']}  ({company['domain']})   collected at {company['collected_at']}")

    print("\nRuns")
    for r in conn.execute("SELECT * FROM runs WHERE company_id = ? ORDER BY id", (company["id"],)):
        c = json.loads(r["counters"])
        print(f"  run {r['id']}: {r['status']:<17} started {r['started_at']}  finished {r['finished_at']}")
        print(f"      pages {c.get('pages', 0)}, failed {c.get('pages_failed', 0)}, "
              f"skipped {c.get('pages_skipped', 0)}, search queries {c.get('search_queries', 0)}, "
              f"LLM calls {c.get('llm_calls', 0)}, network calls {c.get('network_calls_this_process', 0)}")
        for reason in json.loads(r["reasons"]):
            print(f"      reason: {reason['code']} / {reason['stage']}: {reason['detail']}"
                  + (f"  [{reason['url']}]" if reason.get("url") else ""))

    print("\nSources (tier, type, stage, published, URL)")
    rows = conn.execute("SELECT * FROM sources WHERE company_id = ? ORDER BY tier, source_type, id",
                        (company["id"],)).fetchall()
    for s in rows:
        note = f"   <- {s['last_error']}" if s["last_error"] else ""
        print(f"  {s['tier']}  {s['source_type']:<14}{s['stage']:<9}{s['published_at'] or '-':<11}"
              f"{s['url'][:100]}{note}")
    print(f"\n{len(rows)} sources; "
          + ", ".join(f"{n} {stage}" for stage, n in Counter(s["stage"] for s in rows).items()))
    undated = [s for s in rows if s["source_type"] in ("press_release", "news")
               and s["stage"] == "fetched" and not s["published_at"]]
    print(f"{len(undated)} fetched articles have no publication date (left empty, never guessed)")

    logs = sorted((ROOT / "logs").glob(f"collect-{domain}-*.jsonl"))
    if logs:
        events = [json.loads(line) for line in logs[-1].read_text(encoding="utf-8").splitlines()]
        kinds = Counter(e["event"] for e in events)
        hits = sum(e.get("cache") == "hit" for e in events)
        print(f"\nNewest run log: {logs[-1].name}")
        print("  events: " + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items()))
              + f"   (cache hits: {hits})")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

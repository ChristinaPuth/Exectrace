"""Re-check every stored claim against the saved page it came from. No model, no network.

    uv run python tools/check_evidence.py

For each claim in data/leadership.db: the evidence quote must be in the cached
page, the person's full name must be in it, and the quote must be next to the
name. Also checks that every role marked current has a supporting claim.
Exit code 1 if anything fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from leadership.cache import FileCache  # noqa: E402
from leadership.db import connect  # noqa: E402
from leadership.extract import ExtractedClaim, Target, check_claim  # noqa: E402
from leadership.normalize import normalize_name  # noqa: E402
from leadership.sources.web import clean_text  # noqa: E402
from leadership.verify import check_bio, check_location  # noqa: E402


def main() -> int:
    conn = connect(ROOT / "data" / "leadership.db")
    pages = FileCache(ROOT / "cache" / "pages")
    texts: dict[str, str] = {}
    failures = total = 0
    for company in conn.execute("SELECT * FROM companies ORDER BY id").fetchall():
        target = Target(company["name"], {company["domain"]})
        rows = conn.execute(
            """SELECT cl.*, p.full_name, s.url FROM claims cl JOIN people p ON p.id = cl.person_id
                 JOIN sources s ON s.id = cl.source_id WHERE cl.company_id = ? ORDER BY cl.id""",
            (company["id"],)).fetchall()
        bad = 0
        for r in rows:
            if r["url"] not in texts:
                hit = pages.get(r["url"])
                texts[r["url"]] = clean_text(hit["body"]) if hit and hit.get("body") else ""
            claim = ExtractedClaim(person=r["person_as_written"] or r["full_name"],
                                   organization=company["name"],
                                   claim_type=r["claim_type"], raw_title=r["raw_title"],
                                   value=r["value"], evidence=r["evidence"])
            surname = normalize_name(r["full_name"]).split()[-1]
            if not texts[r["url"]]:
                reason = "the page is not in the cache"
            elif r["claim_type"] == "bio":              # found by enrichment: its own checks apply
                reason = check_bio(r["evidence"], texts[r["url"]], surname)
            elif r["claim_type"] == "location":
                reason = check_location(r["value"], r["evidence"], texts[r["url"]], surname)
            else:
                reason = check_claim(claim, texts[r["url"]], target)
                # Extraction checks the title as the model gave it, then appends the business
                # unit ("..., Wise Platform"). The re-check accepts the title without that suffix.
                if reason and reason.startswith("the title is not tied") and ", " in (r["raw_title"] or ""):
                    plain_title = r["raw_title"].rsplit(", ", 1)[0]
                    reason = check_claim(claim.model_copy(update={"raw_title": plain_title}),
                                         texts[r["url"]], target)
            if reason:
                bad += 1
                print(f"  FAIL claim {r['id']} ({r['person_as_written']} / {r['raw_title']}): {reason}\n"
                      f"       {r['url'][:100]}")
        orphans = conn.execute(
            """SELECT count(*) FROM roles r WHERE r.company_id = ? AND r.status = 'current'
                AND NOT EXISTS (SELECT 1 FROM claims c WHERE c.id = r.supporting_claim_id)""",
            (company["id"],)).fetchone()[0]
        current = conn.execute("SELECT count(*) FROM roles WHERE company_id = ? AND status = 'current'",
                               (company["id"],)).fetchone()[0]
        versions = conn.execute("SELECT count(DISTINCT extraction_version) FROM claims WHERE company_id = ?",
                                (company["id"],)).fetchone()[0]
        print(f"{company['name']:<12}{len(rows):>4} claims re-checked, {bad} failed; {current} current roles, "
              f"{orphans} without a supporting claim; {versions} extraction version(s) in use")
        total += len(rows)
        failures += bad + orphans
    print(f"\n{total} claims checked: {failures} problems")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

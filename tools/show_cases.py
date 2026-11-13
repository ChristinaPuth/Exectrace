"""Print every worked case of the role resolution rules as a table.

    uv run python tools/show_cases.py

For manual review against SPEC section 3.6: each row shows what the rules
actually return for one situation, and whether that matches the expectation
written in tests/worked_cases.py. X and Y are two people.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from worked_cases import CASES, TODAY, matches  # noqa: E402

from leadership.resolve import resolve  # noqa: E402

NAMES = {1: "X", 2: "Y", 3: "Z"}


def main() -> int:
    wrong = 0
    for n, case in enumerate(CASES, 1):
        today = case.get("today", TODAY)
        result = resolve(case["claims"], today)
        got = {(r.person_id, r.key): r for r in result.roles}
        ok = matches(case, result)
        wrong += not ok
        print(f"\n{n:>2}. {case['name']}   [as of {today}]   {'OK' if ok else 'MISMATCH'}")
        for (pid, key), r in sorted(got.items()):
            extra = ""
            if r.status == "announced":
                extra += f"  starts {r.starts_on or '(no date given)'}"
            if r.successor_claim_id:
                nxt = next(c for c in case["claims"] if c.id == r.successor_claim_id)
                extra += f"  successor announced: {NAMES[nxt.person_id]}"
            if r.ended_by_claim_id and r.status == "current" and not r.ended_on:
                extra += "  departure announced, no date"
            if r.ended_on:
                extra += f"  ends {r.ended_on}"
            if r.conflict:      # on its own line, so a narrow terminal does not wrap mid-message
                extra += f"\n          CONFLICT: {r.conflict_reason}"
            print(f"      {NAMES[pid]} as {key:<20} {r.status:<11} {r.status_reason}{extra}")
        print("      employment: " + ", ".join(
            f"{NAMES[p]}={s}" for p, s in sorted(result.employment.items())))
    print(f"\n{len(CASES) - wrong} of {len(CASES)} cases match their expected outcome")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())

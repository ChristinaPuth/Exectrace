"""Check that the worked cases can actually fail.

    uv run python tools/mutation_check.py

Passing tests prove little if they would also pass with a broken rule. This
script breaks the resolution rules one at a time, in memory only (the file on
disk is never changed), and re-runs every worked case against the broken
version. Each break must make at least one case fail.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from worked_cases import CASES, TODAY, matches  # noqa: E402

SOURCE = (ROOT / "leadership" / "resolve.py").read_text(encoding="utf-8")

# (what is broken, exact text in resolve.py, replacement)
MUTATIONS = [
    ("A news-only departure ends the role (rule 2 removed)",
     "strong = [d for d in deps if d.tier <= 2 and not d.pending]",
     "strong = [d for d in deps if d.tier <= 3 and not d.pending]"),
    ("Old documents count as fresh (freshness window removed)",
     "return c.tier <= 3 and c.is_dated and c.when >= cutoff",
     "return c.tier <= 3 and c.is_dated"),
    ("Undated articles count as dated",
     "return self.effective_date is not None or self.as_of_basis == \"published\"",
     "return True"),
    ("Leaving the company no longer ends the person's roles",
     "(d.departure_scope == \"company\" or role_key(d) == key)",
     "(role_key(d) == key)"),
    ("Single-holder conflicts are not detected (rule 7 removed)",
     "if len({r.person_id for r in group}) < 2:",
     "if True:"),
    ("Co-CEOs shown on the company page are flagged as a conflict",
     "if all(shared_on_page(r) for r in group):",
     "if False:"),
    ("One 'Co-CEO' title is enough to clear the conflict for everyone",
     "if all(shared_on_page(r) for r in group):",
     "if any(shared_on_page(r) for r in group):"),
    ("A shared title in any source, not only the company page, clears the conflict",
     "return any(s.is_live_listing and is_shared_title(s.raw_title)",
     "return any(is_shared_title(s.raw_title)"),
    ("Any title starting with 'co-' counts as a shared role",
     "(?:c[a-z]o|chief|president|chair|head)\\b",
     "\\w+"),
    ("An appointment ends a co-holder shown on the same page",
     "if listed and a.person_id in live_holders:",
     "if False:"),
    ("Succession is never detected (rule 3 removed)",
     "elif (succ := successor(pid, canonical, sup)):",
     "elif (succ := None):"),
    ("A live page listing no longer makes a role current (rule 4 removed)",
     "elif listed:                                                # rule 4",
     "elif False:"),
    ("A future departure date is treated as already past",
     "if d.when <= today:\n                role.status, role.status_reason = FORMER, R1_END",
     "if True:\n                role.status, role.status_reason = FORMER, R1_END"),
    ("A future appointment ends the current holder's role on the day it is announced",
     "and c.tier <= 2 and c.when <= today\n",
     "and c.tier <= 2\n"),
    ("An undated 'will become' appointment takes effect on the day it is published",
     "if c.claim_type == \"appointment\" and (c.pending or c.when > today):",
     "if c.claim_type == \"appointment\" and c.when > today:"),
    ("An undated appointment is never confirmed by later evidence",
     "if proof:",
     "if False:"),
    ("An announced appointment is counted as a current role",
     "status=UNVERIFIED if stale else ANNOUNCED,",
     "status=UNVERIFIED if stale else CURRENT,"),
    ("A stale, never-confirmed announcement still shows as announced",
     "stale = a.pending and a.as_of_date < cutoff",
     "stale = False"),
    ("An undated 'will step down' ends the role on the day it is published",
     "strong = [d for d in deps if d.tier <= 2 and not d.pending]",
     "strong = [d for d in deps if d.tier <= 2]"),
    ("An undated departure keeps the role current for ever, however old it is",
     "undated = [d for d in deps if d.tier <= 2 and d.pending and d.as_of_date >= cutoff]",
     "undated = [d for d in deps if d.tier <= 2 and d.pending]"),
    ("An undated 'will leave the company' counts as having left",
     "and d.when <= today\n                and not d.pending]",
     "and d.when <= today]"),
    ("A search snippet can report a departure",
     "elif c.claim_type == \"departure\" and c.tier <= 3:",
     "elif c.claim_type == \"departure\" and c.tier <= 4:"),
    ("A later re-appointment no longer cancels an earlier departure",
     "and not any(a.claim_type == \"appointment\" and a.when > d.when for a in sup)",
     ""),
]


def failing_cases(source: str) -> int:
    module = types.ModuleType("mutated_resolve")
    sys.modules[module.__name__] = module       # dataclasses look the module up by name
    exec(compile(source, "mutated_resolve.py", "exec"), module.__dict__)
    failed = 0
    for case in CASES:
        try:
            # Rebuild each claim with the mutated Claim class, so that breaks
            # inside Claim's own properties take effect too.
            claims = [module.Claim(**dataclasses.asdict(c)) for c in case["claims"]]
            ok = matches(case, module.resolve(claims, case.get("today", TODAY)))
        except Exception:
            ok = False
        failed += not ok
    return failed


def main() -> int:
    baseline = failing_cases(SOURCE)
    print(f"Unmodified rules: {baseline} of {len(CASES)} cases fail (must be 0)")
    survived = 0
    for name, old, new in MUTATIONS:
        if SOURCE.count(old) != 1:
            print(f"  NOT APPLIED  {name}  (text not found exactly once)")
            survived += 1
            continue
        failed = failing_cases(SOURCE.replace(old, new))
        survived += failed == 0
        print(f"  {'caught  ' if failed else 'SURVIVED'}  {name}  ->  {failed} case(s) fail")
    print(f"\n{len(MUTATIONS) - survived} of {len(MUTATIONS)} broken rules were caught by the worked cases")
    return 1 if baseline or survived else 0


if __name__ == "__main__":
    sys.exit(main())

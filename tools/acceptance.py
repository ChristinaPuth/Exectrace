"""Run a milestone's acceptance commands and save their real output.

    uv run python tools/acceptance.py M0

Writes acceptance/<stage>_results.log: every command, its full unedited
output, its exit code, and whether that matched the expected code. Configured
secrets are redacted before anything is written. The log is evidence; the
human-readable report and manual test list live in acceptance/<stage>.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from leadership.config import ROOT, load_env  # noqa: E402
from leadership.llm import redact  # noqa: E402

PYTEST = ["uv", "run", "pytest", "-v"]
CHECK = ["uv", "run", "leadership", "check"]
COLLECT = ["uv", "run", "leadership", "collect", "--no-extract"]     # the M2 stage: fetch only
RUN = ["uv", "run", "leadership", "collect"]

# (title, command, environment overrides, expected exit code)
# An override of "" hides that key from the command, as if it were not in .env.
STAGES: dict[str, list[tuple[str, list[str], dict[str, str], int]]] = {
    "M0": [
        ("Fast tests (no network, no keys)", PYTEST, {}, 0),
        ("Real-provider tests: Groq first, pacing, fallback to OpenAI", PYTEST + ["-m", "llm"], {}, 0),
        ("Key check with all keys", CHECK, {}, 0),
        ("Key check with only the Groq key", CHECK, {"OPENAI_API_KEY": ""}, 0),
        ("Key check with only the OpenAI key", CHECK, {"GROQ_API_KEY": ""}, 0),
        ("Key check with an invalid Groq key (must fail without printing the key)",
         CHECK, {"GROQ_API_KEY": "gsk_invalid_key_for_acceptance_test"}, 1),
        ("Key check with no keys (must explain what to do)",
         CHECK, {"OPENAI_API_KEY": "", "GROQ_API_KEY": ""}, 2),
    ],
    "M1": [
        ("Resolution rules: one test per worked case, plus rule-specific tests",
         PYTEST + ["tests/test_resolve.py"], {}, 0),
        ("Database: schema, constraints, views, rebuilding roles from claims",
         PYTEST + ["tests/test_db.py"], {}, 0),
        ("Whole fast suite (no network, no keys)", ["uv", "run", "pytest", "-q"], {}, 0),
        ("Worked cases printed as a table for manual review",
         ["uv", "run", "python", "tools/show_cases.py"], {}, 0),
        ("Mutation check: each deliberately broken rule must fail at least one case",
         ["uv", "run", "python", "tools/mutation_check.py"], {}, 0),
        ("Database schema as created: tables, views and columns",
         ["uv", "run", "python", "tools/show_schema.py"], {}, 0),
    ],
    "M2": [
        ("M2 tests: cache, fetcher, robots, discovery, SEC matching, runs, budgets, resume",
         PYTEST + ["tests/test_sources.py", "tests/test_collect.py"], {}, 0),
        ("Whole fast suite (no network, no keys)", ["uv", "run", "pytest", "-q"], {}, 0),
        ("Real model: picks leadership pages, may choose none, drops a namesake search result",
         PYTEST + ["-m", "llm", "tests/test_sources.py"], {}, 0),
        ("Campfire: real collection, caches ignored", COLLECT + ["https://meetcampfire.com/", "--refresh"], {}, 0),
        ("Campfire: second run must make 0 network calls", COLLECT + ["https://meetcampfire.com/"], {}, 0),
        ("Robinhood: real collection, caches ignored", COLLECT + ["https://robinhood.com/", "--refresh"], {}, 0),
        ("Robinhood: second run must make 0 network calls", COLLECT + ["https://robinhood.com/"], {}, 0),
        ("Campfire: runs, reasons and sources as stored",
         ["uv", "run", "python", "tools/show_run.py", "meetcampfire.com"], {}, 0),
        ("Robinhood: runs, reasons and sources as stored",
         ["uv", "run", "python", "tools/show_run.py", "robinhood.com"], {}, 0),
        ("Unreachable homepage: must stop with a clear error and store nothing",
         COLLECT + ["https://no-such-company-exectrace.invalid/"], {}, 1),
        ("No key or contact address in any file that could be committed, or in the run logs",
         ["uv", "run", "python", "tools/scan_secrets.py"], {}, 0),
    ],
    # M3 reads the pages M2 cached: no page is fetched and no search query is made,
    # unless the cache is empty. It does make real model calls.
    "M3": [
        ("M3 tests: title and name rules, evidence checks, storage, the extraction stage, scoring",
         PYTEST + ["tests/test_normalize.py", "tests/test_extract.py", "tests/test_pipeline.py",
                   "tests/test_evals.py"], {}, 0),
        ("Whole fast suite (no network, no keys)", ["uv", "run", "pytest", "-q"], {}, 0),
        ("Real model: extraction, planted instructions, directory page, resume, outage, invalid output",
         PYTEST + ["-m", "llm", "tests/test_extract.py", "tests/test_pipeline.py"], {}, 0),
        ("Extraction accuracy on five hand-labelled real pages", ["uv", "run", "leadership", "eval"], {}, 0),
        ("Campfire: collect and extract (pages from the cache, real model)", RUN + ["https://meetcampfire.com/"], {}, 0),
        ("Campfire again: must make 0 model calls and repeat no extraction", RUN + ["https://meetcampfire.com/"], {}, 0),
        ("Robinhood: collect and extract (pages from the cache, real model)", RUN + ["https://robinhood.com/"], {}, 0),
        ("Robinhood again: must make 0 model calls and repeat no extraction", RUN + ["https://robinhood.com/"], {}, 0),
        ("Campfire: leaders with every claim and its evidence",
         ["uv", "run", "python", "tools/show_people.py", "meetcampfire.com"], {}, 0),
        ("Robinhood: leaders with every claim and its evidence",
         ["uv", "run", "python", "tools/show_people.py", "robinhood.com"], {}, 0),
        ("Campfire: claims the checks refused",
         ["uv", "run", "python", "tools/show_people.py", "meetcampfire.com", "--rejected"], {}, 0),
        ("Robinhood: claims the checks refused",
         ["uv", "run", "python", "tools/show_people.py", "robinhood.com", "--rejected"], {}, 0),
        ("Every stored claim re-checked against its saved page; every current role has support",
         ["uv", "run", "python", "tools/check_evidence.py"], {}, 0),
        ("M1 rules still hold: mutation check", ["uv", "run", "python", "tools/mutation_check.py"], {}, 0),
        ("No key or contact address in any file that could be committed, or in the run logs",
         ["uv", "run", "python", "tools/scan_secrets.py"], {}, 0),
    ],
}


def _git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "(none)"


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        print(f"usage: acceptance.py <{'|'.join(STAGES)}>")
        return 2
    stage = sys.argv[1]
    load_env()
    out_dir = ROOT / "acceptance"
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / f"{stage}_results.log"

    lines = [
        f"Acceptance run for {stage}",
        f"Date:    {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Branch:  {_git('branch', '--show-current')}",
        f"Commit:  {_git('rev-parse', '--short', 'HEAD')}",
        f"Python:  {sys.version.split()[0]}",
        "",
    ]
    summary = []
    for n, (title, cmd, overrides, expected) in enumerate(STAGES[stage], 1):
        env = {**os.environ, **overrides, "PYTHONIOENCODING": "utf-8"}
        started = time.perf_counter()
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        seconds = time.perf_counter() - started
        verdict = "PASS" if r.returncode == expected else "FAIL"
        summary.append((n, verdict, title))
        shown_env = " ".join(f"{k}={'(hidden)' if v else '(empty)'}" for k, v in overrides.items())
        lines += [
            "=" * 78,
            f"[{n}] {title}",
            f"command:  {' '.join(cmd)}" + (f"    with {shown_env}" if shown_env else ""),
            f"exit code {r.returncode} (expected {expected})  ->  {verdict}    {seconds:.1f}s",
            "-" * 78,
            redact(r.stdout).rstrip(),
        ]
        if r.stderr.strip():
            lines += ["--- stderr ---", redact(r.stderr).rstrip()]
        lines.append("")
        print(f"[{n}] {verdict}  {title}")

    lines += ["=" * 78, "SUMMARY"] + [f"  [{n}] {v}  {t}" for n, v, t in summary]
    failed = sum(v == "FAIL" for _, v, _ in summary)
    lines.append(f"  {len(summary) - failed} of {len(summary)} steps as expected")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {log_path.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

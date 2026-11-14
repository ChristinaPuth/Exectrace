"""Fresh-clone test (SPEC 7.4): clone the committed repository, add one LLM key, and ask.

    uv run python tools/fresh_clone.py [--max-calls 40]

1. `git clone` this repository's current branch into a new temporary folder.
   Only committed files arrive: no cache, no logs, no data/, no .env.
2. Write a .env holding only the LLM keys from this repository's .env.
3. `uv sync`, then the whole fast test suite.
4. Ask the four sample questions for both companies through the same path as
   the page and the command line (agent, source check, sources appended),
   and grade each with the chat evaluation's rules. Real model calls, capped.
5. The page itself: Streamlit's test runner asks a question in the browser
   page's code (tests/test_app.py, `llm` tests).

Exit code 0 only if every step passed. The clone is left in place for
inspection; its path is printed.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SAMPLES = r'''
import json, sys
from pathlib import Path
from leadership import llm
from leadership.agent import Agent, collected_on
from leadership.chateval import grade
from leadership.citations import add_missing_sources
from leadership.config import load_config, load_env
from leadership.grounding import grounded_ask

limit = int(sys.argv[1])
calls = []
real = llm.LLM.chat
def counted(self, *a, **k):
    calls.append(1)
    if len(calls) > limit:
        raise RuntimeError(f"more than {limit} model calls")
    return real(self, *a, **k)
llm.LLM.chat = counted

load_env()
cfg = load_config()
db = Path("fixtures/leadership.db")
spec = json.loads(Path("fixtures/chat_eval.json").read_text(encoding="utf-8"))
failed = 0
for domain, cases in spec["companies"].items():
    agent = Agent(cfg, domain, db)
    for case in [c for c in cases if c.get("sample")]:
        g = grade(case, grounded_ask(agent, case["question"]), agent.conn, collected_on(db, domain))
        failed += not g.passed
        print(f"{'PASS' if g.passed else 'FAIL'}  {domain}  {case['question']}")
        print("      " + g.answer.replace("\n", "\n      "))
        for p in g.problems:
            print(f"      - {p}")
print(f"\nSample questions: {8 - failed} of 8. Model calls: {len(calls)}")
sys.exit(1 if failed else 0)
'''


def run(cmd: list[str], cwd: Path, title: str) -> bool:
    print(f"\n=== {title}\n$ {' '.join(cmd)}", flush=True)
    done = subprocess.run(cmd, cwd=cwd)
    print(f"--> {'PASS' if done.returncode == 0 else 'FAIL'} (exit code {done.returncode})", flush=True)
    return done.returncode == 0


def main() -> int:
    max_calls = int(sys.argv[sys.argv.index("--max-calls") + 1]) if "--max-calls" in sys.argv else 40
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    clone = Path(tempfile.mkdtemp(prefix="exectrace-clone-")) / "repo"
    if not run(["git", "clone", "--quiet", "--branch", branch, str(ROOT), str(clone)], ROOT,
               f"Clone branch {branch}"):
        return 1
    keys = [line for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines()
            if line.split("=", 1)[0].strip() in ("GROQ_API_KEY", "OPENAI_API_KEY") and line.split("=", 1)[1].strip()]
    (clone / ".env").write_text("\n".join(keys) + "\n", encoding="utf-8")
    print(f"\n.env in the clone: {', '.join(k.split('=', 1)[0] for k in keys)} (values not shown)")
    missing = [p for p in ("cache", "logs", "data") if (clone / p).exists()]
    print(f"Local-only folders in the clone: {missing or 'none'}")
    steps = [
        run(["uv", "sync", "--quiet"], clone, "uv sync"),
        run(["uv", "run", "pytest", "-q"], clone, "Fast test suite"),
        run(["uv", "run", "python", "-c", SAMPLES, str(max_calls)], clone,
            "The four sample questions, both companies (real model)"),
        run(["uv", "run", "pytest", "-q", "-m", "llm", "tests/test_app.py"], clone, "The web page (real model)"),
    ]
    print(f"\nFresh-clone test: {sum(steps)} of {len(steps)} steps passed. Clone: {clone}")
    return 0 if all(steps) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the M2 manual tests that can be scripted, and save the full output.

    uv run python tools/manual_checks_m2.py

Writes acceptance/M2_manual_results.log. Uses as few outside calls as it can:

- MT-2.1 to MT-2.4b, 2.7 and 2.8 run the real commands on the working data,
  which is answered from the cache.
- MT-2.5 (limits and resume) runs Microsoft in a separate folder that holds a
  copy of the cache, with only `max_pages` changed. No search or model call.
- MT-2.6 (interrupt and resume) runs Campfire in a separate folder that holds
  the search and model caches but no pages, so the pages are really fetched
  and the process can be stopped in the middle. The process is killed
  outright, which is harsher than Ctrl+C: nothing gets a chance to clean up.

The separate folders are under data/manual_m2/ (not committed) and are
rebuilt on every run. config.toml and the working database are not changed
by MT-2.5 and MT-2.6.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
WORK = ROOT / "data" / "manual_m2"
PY = [sys.executable, str(Path(__file__).resolve()), "--drive"]
CLI = ["uv", "run", "leadership", "collect"]
SHOW = ["uv", "run", "python", "tools/show_run.py"]


def drive(root: str, url: str, max_pages: str) -> int:
    """One collection run in another folder, with only max_pages overridden."""
    from leadership.collect import _db_path, collect, print_report
    from leadership.config import load_config, load_env
    load_env()
    cfg = load_config()
    cfg = cfg.model_copy(update={"limits": cfg.limits.model_copy(update={"max_pages": int(max_pages)})})
    result = collect(url, cfg, root=Path(root))
    print_report(result, cfg, _db_path(Path(root)))
    return 0 if result.status == "complete" else 3


def run(cmd: list[str]) -> tuple[int, str, float]:
    started = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr.strip() else "")), \
        time.perf_counter() - started


def number(text: str, label: str) -> int | None:
    m = re.search(rf"{re.escape(label)}\s+(\d+)", text)
    return int(m.group(1)) if m else None


def fresh(name: str, caches: tuple[str, ...]) -> Path:
    folder = WORK / name
    shutil.rmtree(folder, ignore_errors=True)
    for c in caches:
        shutil.copytree(ROOT / "cache" / c, folder / "cache" / c)
    return folder


def rows(folder: Path, sql: str) -> list[tuple]:
    path = folder / "data" / "leadership.db"
    if not path.exists():                   # the run has not created it yet
        return []
    conn = sqlite3.connect(path, timeout=10)
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.summary: list[tuple[str, str, str]] = []

    def step(self, test: str, title: str, cmd: list[str], checks) -> str:
        code, out, seconds = run(cmd)
        return self.record(test, title, " ".join(cmd), code, out, seconds, checks(code, out))

    def record(self, test, title, shown, code, out, seconds, checks) -> str:
        failed = [name for name, ok in checks if not ok]
        verdict = "FAIL" if failed else "PASS"
        self.summary.append((test, verdict, title))
        self.lines += ["=" * 78, f"[{test}] {title}", f"command:  {shown}",
                       f"exit code {code}    {seconds:.1f}s", "checks:"]
        self.lines += [f"  {'ok  ' if ok else 'FAIL'}  {name}" for name, ok in checks]
        self.lines += [f"->  {verdict}", "-" * 78, out.rstrip(), ""]
        print(f"  {verdict}  {test}  {title}")
        return out


def main() -> int:
    rep = Report()
    rep.lines += ["Manual tests for M2, run by script (tools/manual_checks_m2.py)",
                  f"Date:    {datetime.now().astimezone().isoformat(timespec='seconds')}",
                  f"Python:  {sys.version.split()[0]}", ""]

    rep.step("MT-2.1", "Fast tests", ["uv", "run", "pytest", "-q"], lambda c, o: [
        ("exit code 0", c == 0), ("no test failed", " failed" not in o),
        ("the tests that need keys are deselected, not failed", bool(re.search(r"\d+ passed, \d+ deselected", o)))])

    rep.step("MT-2.2", "Campfire from cache", CLI + ["https://meetcampfire.com/"], lambda c, o: [
        ("Status: complete", "Status: complete" in o), ("network calls 0", number(o, "network calls") == 0),
        ("unread list names www.thecampfire.ai/about",
         "could not be read" in o and "https://www.thecampfire.ai/about" in o)])

    rep.step("MT-2.3", "Campfire sources as stored", SHOW + ["meetcampfire.com"], lambda c, o: [
        ("thecampfire.ai row is skipped as not verified",
         bool(re.search(r"skipped .*thecampfire\.ai/about.*not verified", o))),
        ("no tavily.com row", "tavily.com" not in o),
        ("every fetched or skipped URL is listed", "sources;" in o)])

    rep.step("MT-2.4", "Robinhood from cache", CLI + ["https://robinhood.com/"], lambda c, o: [
        ("Status: complete", "Status: complete" in o), ("network calls 0", number(o, "network calls") == 0),
        ("unread list marks 5 investor pages as the company's own",
         len(re.findall(r"investors\.robinhood\.com\S*  <- the company's own page", o)) == 5)])

    rep.step("MT-2.4", "Robinhood sources as stored", SHOW + ["robinhood.com"], lambda c, o: [
        ("5 SEC filings fetched", len(re.findall(r"sec_filing +fetched", o)) == 5),
        ("5 investor pages skipped", len(re.findall(r"skipped .*investors\.robinhood\.com", o)) == 5),
        ("2 Wikipedia pages skipped", len(re.findall(r"wikipedia +skipped", o)) == 2),
        ("no exa.ai row", "exa.ai" not in o),
        ("5 access_refused reasons on the run that met them", o.count("reason: access_refused") == 5)])

    rep.step("MT-2.4b", "Microsoft, repeated", CLI + ["https://www.microsoft.com/"], lambda c, o: [
        ("Status: complete", "Status: complete" in o), ("pages failed 0", number(o, "pages failed") == 0),
        ("unread list names news.microsoft.com/source/leadership",
         "https://news.microsoft.com/source/leadership  <- the company's own page" in o)])
    rep.step("MT-2.4b", "Microsoft sources as stored", SHOW + ["microsoft.com"], lambda c, o: [
        ("no vectorshift.ai or derrick-app.com row", "vectorshift" not in o and "derrick-app" not in o),
        ("3 access_refused reasons on an earlier run", o.count("reason: access_refused") == 3),
        ("Business Roundtable page has no date",
         bool(re.search(r"fetched +- +https://www\.businessroundtable\.org", o)))])

    # ------------------------------------------------------------------ MT-2.5
    folder = fresh("limits", ("pages", "search", "llm"))
    url = "https://www.microsoft.com/"
    first = rep.step("MT-2.5", "Limit: max_pages = 5 (Microsoft, copy of the cache)",
                     PY + [str(folder), url, "5"], lambda c, o: [
        ("exit code 3", c == 3), ("Status: budget_exhausted", "Status: budget_exhausted" in o),
        ("reason reads max_pages=5 reached", "max_pages=5 reached" in o),
        ("pages 5 of 5", number(o, "pages") == 5),
        ("no search query or model call was made", number(o, "search_queries") == 0)])
    run_id = number(first, "Run:")
    rep.step("MT-2.5", "Same limit again: must resume, not start over", PY + [str(folder), url, "5"],
             lambda c, o: [
        ("same run number", number(o, "Run:") == run_id),
        ("resumed (budget_exhausted)", "resumed (budget_exhausted)" in o),
        ("still 5 pages: the restart did not get around the limit", number(o, "pages") == 5),
        ("Status: budget_exhausted", "Status: budget_exhausted" in o)])
    rep.step("MT-2.5", "Limit raised to max_pages = 40: must continue the same run", PY + [str(folder), url, "40"],
             lambda c, o: [
        ("same run number", number(o, "Run:") == run_id),
        ("resumed (budget_exhausted)", "resumed (budget_exhausted)" in o),
        ("more than 5 pages", (number(o, "pages") or 0) > 5),
        ("Status: complete", "Status: complete" in o)])

    # ------------------------------------------------------------------ MT-2.6
    folder = fresh("interrupt", ("search", "llm"))          # no pages: they must really be fetched
    url = "https://meetcampfire.com/"
    started = time.perf_counter()
    proc = subprocess.Popen(PY + [str(folder), url, "40"], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    stored = 0
    while proc.poll() is None and time.perf_counter() - started < 240:
        got = rows(folder, "SELECT count(*) FROM sources")
        stored = got[0][0] if got else 0
        if stored >= 4:
            break
        time.sleep(0.2)
    killed = proc.poll() is None
    proc.kill()
    out, _ = proc.communicate()
    after = rows(folder, "SELECT id, status, finished_at FROM runs ORDER BY id")
    rep.record("MT-2.6", "Kill the process in the middle of a Campfire run",
               " ".join(PY + [str(folder), url, "40"]) + "   (killed once 4 sources were stored)",
               proc.returncode, out + f"\n[runs table after the kill: {after}]\n"
               f"[sources stored before the kill: {stored}]", time.perf_counter() - started, [
        ("the process was still working when it was killed", killed),
        ("it had stored some sources but not all", 4 <= stored < 12),
        ("the run is left as 'running' with no finish time",
         len(after) == 1 and after[0][1] == "running" and after[0][2] is None)])
    rep.step("MT-2.6", "Start again: must detect the interrupted run and finish it",
             PY + [str(folder), url, "40"], lambda c, o: [
        ("same run number", bool(after) and number(o, "Run:") == after[0][0]),
        ("resumed (interrupted)", "resumed (interrupted)" in o),
        ("Status: complete", "Status: complete" in o),
        ("more sources than before the kill",
         rows(folder, "SELECT count(*) FROM sources")[0][0] > stored),
        ("each source is stored once", rows(folder, "SELECT count(*), count(DISTINCT url) FROM sources")[0][0]
         == rows(folder, "SELECT count(*), count(DISTINCT url) FROM sources")[0][1]),
        ("still one run for the company", len(rows(folder, "SELECT id FROM runs")) == 1)])

    rep.step("MT-2.7", "Unreachable site", CLI + ["https://no-such-company-exectrace.invalid/"],
             lambda c, o: [("exit code 1", c == 1), ("says nothing was stored", "Nothing was stored." in o),
                           ("says the homepage could not be read", "Could not read the homepage" in o)])

    newest = max((ROOT / "logs").glob("collect-*.jsonl"), key=lambda p: p.stat().st_mtime)
    kinds = re.findall(r'"event": "(\w+)"', newest.read_text(encoding="utf-8"))
    rep.record("MT-2.8", "Newest run log has one line per event", f"read {newest.name}", 0,
               "\n".join(f"{k}: {kinds.count(k)}" for k in sorted(set(kinds))), 0.0,
               [("has fetch events", "fetch" in kinds), ("ends with run_end", kinds[-1:] == ["run_end"])])
    rep.step("MT-2.8", "Secret scan", ["uv", "run", "python", "tools/scan_secrets.py"], lambda c, o: [
        ("exit code 0", c == 0), ("0 problems", o.strip().endswith("0 problems"))])

    rep.lines += ["=" * 78, "SUMMARY"]
    rep.lines += [f"  {verdict}  {test:<8} {title}" for test, verdict, title in rep.summary]
    passed = sum(v == "PASS" for _, v, _ in rep.summary)
    rep.lines += ["", f"{passed} of {len(rep.summary)} scripted checks passed.",
                  "Not scripted: MT-2.9 (your decisions) and MT-2.10 (the checklist workbook)."]
    out_path = ROOT / "acceptance" / "M2_manual_results.log"
    out_path.write_text("\n".join(rep.lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\n{passed} of {len(rep.summary)} passed. Wrote {out_path.relative_to(ROOT)}")
    return 0 if passed == len(rep.summary) else 1


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--drive":
        sys.exit(drive(*sys.argv[2:]))
    sys.exit(main())

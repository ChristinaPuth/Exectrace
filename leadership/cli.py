"""Command line entry point: `uv run leadership <command>`."""

from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config, load_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="leadership", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="verify API keys and report usable models and rate limits")
    p = sub.add_parser("collect", help="discover and fetch a company's public leadership sources")
    p.add_argument("url", help="company domain or URL, e.g. robinhood.com")
    p.add_argument("--refresh", action="store_true", help="ignore the page and search caches")
    p.add_argument("--no-verify", action="store_true",
                   help="stop after the first resolution: no verification searches, no enrichment")
    p.add_argument("--no-extract", action="store_true",
                   help="stop after fetching; do not extract people (no LLM extraction calls)")
    c = sub.add_parser("chat", help="ask questions about one company's leadership (real LLM, read-only SQL)")
    c.add_argument("domain", help="company domain, e.g. robinhood.com")
    c.add_argument("-q", "--question", help="ask one question and exit")
    c.add_argument("--show-sql", action="store_true", help="print the SQL that ran and the row counts")
    c.add_argument("--db", help="database to read (default: fixtures/leadership.db)")
    e = sub.add_parser("eval", help="measure extraction on the labelled pages and the fixtures against the gold sets")
    e.add_argument("--gold-only", action="store_true", help="skip the extraction part, which needs an LLM key")
    e.add_argument("--chat", action="store_true", help="evaluate the chat agent only (real LLM, cached)")
    x = sub.add_parser("export", help="write fixtures/<domain>.json and fixtures/leadership.db")
    x.add_argument("domains", nargs="+", help="company domains to export, e.g. robinhood.com")
    args = parser.parse_args(argv)

    try:
        load_env()
        cfg = load_config()
        if args.command == "check":
            from . import check
            return check.run(cfg)
        if args.command == "chat":
            return _chat(cfg, args)
        if args.command == "eval":
            from . import evals
            if args.chat:
                from .chateval import run_chat_eval
                return run_chat_eval(cfg)
            if not args.gold_only:
                evals.run_extraction_eval(cfg)
            return evals.run_gold_eval(cfg)
        if args.command == "export":
            from .config import ROOT
            from .export import export
            try:
                written = export(ROOT / "data" / "leadership.db", ROOT / "fixtures", args.domains)
            except KeyError as e:
                print(f"No company stored for {e}. Run `leadership collect` for it first.", file=sys.stderr)
                return 1
            for path in written:
                print(f"wrote {path.relative_to(ROOT)}")
            return 0
        if args.command == "collect":
            from . import collect
            from .sources.site import HomepageError
            try:
                result = collect.collect(args.url, cfg, refresh=args.refresh,
                                         extract=not args.no_extract, verify=not args.no_verify)
            except HomepageError as e:
                print(f"Stopped: {e}. Nothing was stored.", file=sys.stderr)
                return 1
            collect.print_report(result, cfg, collect._db_path(collect.ROOT))
            return 0 if result.status == "complete" else 3
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    return 0


def _chat(cfg, args) -> int:
    from pathlib import Path

    from .agent import Agent
    from .citations import add_missing_sources
    domain = args.domain.lower().removeprefix("https://").removeprefix("http://").removeprefix("www.").strip("/")
    try:
        agent = Agent(cfg, domain, Path(args.db) if args.db else None)
    except KeyError:
        print(f"No data for {domain}. Run `leadership collect` and `leadership export` first.", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"No database at {e}. Run `leadership export` first.", file=sys.stderr)
        return 1
    note = "" if agent.status == "complete" else f"  (data collection status: {agent.status}; answers are partial)"
    print(f"{agent.company} ({domain}){note}. Answers come only from the collected data.")

    def answer(question: str) -> None:
        result = agent.ask(question)
        print("\n" + add_missing_sources(result.text, result.steps))
        if args.show_sql:
            for step in result.steps:
                outcome = f"error: {step.error}" if step.error else f"{len(step.rows)} rows"
                print(f"\n  SQL ({outcome}): {step.sql}")
            print(f"  [{'from the cache' if result.cached else result.model}]")

    if args.question:
        answer(args.question)
        return 0
    print("Type a question, or an empty line to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not question:
            return 0
        answer(question)


if __name__ == "__main__":
    sys.exit(main())

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
    e = sub.add_parser("eval", help="measure extraction on the labelled pages and the fixtures against the gold sets")
    e.add_argument("--gold-only", action="store_true", help="skip the extraction part, which needs an LLM key")
    x = sub.add_parser("export", help="write fixtures/<domain>.json and fixtures/leadership.db")
    x.add_argument("domains", nargs="+", help="company domains to export, e.g. robinhood.com")
    args = parser.parse_args(argv)

    try:
        load_env()
        cfg = load_config()
        if args.command == "check":
            from . import check
            return check.run(cfg)
        if args.command == "eval":
            from . import evals
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


if __name__ == "__main__":
    sys.exit(main())

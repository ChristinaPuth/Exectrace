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
    p.add_argument("--no-extract", action="store_true",
                   help="stop after fetching; do not extract people (no LLM extraction calls)")
    sub.add_parser("eval", help="measure extraction accuracy on the hand-labelled pages (real LLM)")
    args = parser.parse_args(argv)

    try:
        load_env()
        cfg = load_config()
        if args.command == "check":
            from . import check
            return check.run(cfg)
        if args.command == "eval":
            from . import evals
            return evals.run_extraction_eval(cfg)
        if args.command == "collect":
            from . import collect
            from .sources.site import HomepageError
            try:
                result = collect.collect(args.url, cfg, refresh=args.refresh,
                                         extract=not args.no_extract)
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

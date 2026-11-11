"""Command line entry point: `uv run leadership <command>`."""

from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config, load_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="leadership", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="verify API keys and report usable models and rate limits")
    args = parser.parse_args(argv)

    try:
        load_env()
        cfg = load_config()
        if args.command == "check":
            from . import check
            return check.run(cfg)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

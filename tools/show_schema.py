"""Print the database schema: every table and view with its columns.

    uv run python tools/show_schema.py

Builds the schema in memory, so it touches no file. For manual review against
SPEC section 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from leadership.db import connect  # noqa: E402


def main() -> int:
    conn = connect(":memory:")
    objects = conn.execute(
        "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()
    for obj in objects:
        cols = conn.execute(f"PRAGMA table_xinfo({obj['name']})").fetchall()
        print(f"\n{obj['type'].upper()} {obj['name']}  ({len(cols)} columns)")
        for c in cols:
            flags = " NOT NULL" if c["notnull"] else ""
            print(f"    {c['name']:<24}{c['type']:<8}{flags}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

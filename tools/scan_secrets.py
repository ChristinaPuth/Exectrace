"""Check that no API key or contact address is in any file that could be committed.

    uv run python tools/scan_secrets.py

Scans every file git would track (committed or new, not ignored) plus logs/,
for the exact values in .env and for key-shaped strings. Prints file names
only, never the value found. Exit code 1 if anything is found.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
KEY_SHAPED = re.compile(r"sk-[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9]{20,}|tvly-[A-Za-z0-9_-]{20,}")
# Placeholder keys used on purpose by tests and by the M0 acceptance run.
ALLOWED = ("gsk_invalid_key_for_acceptance_test", "tvly-TESTSECRETVALUE0123456789")


def main() -> int:
    values = [v for v in dotenv_values(ROOT / ".env").values() if v and len(v) >= 8]
    listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                            capture_output=True, text=True).stdout.splitlines()
    files = [ROOT / f for f in listed] + sorted((ROOT / "logs").glob("*"))
    hits = 0
    for path in files:
        try:
            text = path.read_bytes().decode("utf-8", "ignore")
        except OSError:
            continue
        if any(v in text for v in values):
            print(f"FOUND a value from .env in {path.relative_to(ROOT)}")
            hits += 1
        for m in KEY_SHAPED.finditer(text):
            if not m.group(0).startswith(ALLOWED):
                print(f"FOUND a key-shaped string in {path.relative_to(ROOT)}")
                hits += 1
                break
    print(f"{len(files)} files scanned ({len(listed)} that git would track, "
          f"{len(files) - len(listed)} run logs), {len(values)} values from .env checked: "
          f"{hits} problems")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())

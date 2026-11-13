"""Append-only JSON-lines log, one file per collection run. SPEC section 6.4.

Every line is one event. Configured secrets are removed before writing, and
callers never pass request headers, so keys cannot reach the file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .llm import redact


class RunLog:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, **fields) -> None:
        if not self.path:
            return
        line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "event": kind, **fields}, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(redact(line) + "\n")

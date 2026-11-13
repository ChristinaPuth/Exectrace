"""File caches with atomic writes. SPEC section 6.3.

One directory per cache (pages, search, llm), one JSON file per entry. An
entry is written to a temporary file and then renamed, so a crash can never
leave a half-written entry behind: a key is either absent or complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


class FileCache:
    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def path(self, key: str) -> Path:
        return self.dir / (hashlib.sha256(key.encode("utf-8")).hexdigest()[:40] + ".json")

    def get(self, key: str) -> dict | None:
        try:
            entry = json.loads(self.path(key).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return None                 # absent, or not valid JSON: treat as a miss
        return entry["value"] if entry.get("key") == key else None

    def put(self, key: str, value: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"key": key, "value": value}, f, ensure_ascii=False)
            os.replace(tmp, self.path(key))     # atomic on the same volume
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

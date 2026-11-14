"""Copy this project's Claude Code session transcripts into session/, redacted. No network.

    uv run python tools/export_session.py

Claude Code keeps one JSONL transcript per session under
~/.claude/projects/<project path with separators as dashes>/. Each is copied
to session/<date>-<session id>.jsonl, with two kinds of value replaced by
"[REDACTED]":

- every e-mail address, except Anthropic's no-reply commit address and
  addresses at example.com/example.org;
- every value in .env of 8 characters or more (API keys, user agents).

session/index.md lists the files. Run tools/scan_secrets.py afterwards.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "session"
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
KEEP = ("noreply@anthropic.com",)
KEEP_DOMAINS = ("example.com", "example.org")


def transcripts_dir() -> Path:
    slug = re.sub(r"[:\\/]", "-", str(ROOT))
    return Path.home() / ".claude" / "projects" / slug


def env_values() -> list[str]:
    env = ROOT / ".env"
    if not env.exists():
        return []
    values = []
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            if len(v) >= 8:
                values.append(v)
    return sorted(values, key=len, reverse=True)


def redact(text: str, secrets: list[str]) -> tuple[str, int]:
    n = 0
    for s in secrets:
        n += text.count(s)
        text = text.replace(s, "[REDACTED]")

    def mail(m: re.Match) -> str:
        nonlocal n
        addr = m.group(0)
        if addr.lower() in KEEP or addr.lower().split("@")[1] in KEEP_DOMAINS:
            return addr
        n += 1
        return "[REDACTED]"

    return EMAIL.sub(mail, text), n


def first_timestamp(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        for line in f:
            m = re.search(r'"timestamp":"([^"]+)"', line)
            if m:
                return m.group(1)
    return "unknown"


def main() -> int:
    src = transcripts_dir()
    files = sorted(src.glob("*.jsonl"), key=first_timestamp)
    if not files:
        print(f"No transcripts in {src}")
        return 1
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.jsonl"):
        old.unlink()
    secrets = env_values()
    rows = []
    for path in files:
        started = first_timestamp(path)
        text, n = redact(path.read_text(encoding="utf-8"), secrets)
        target = OUT / f"{started[:10]}-{path.stem}.jsonl"
        target.write_text(text, encoding="utf-8", newline="\n")
        lines = text.count("\n")
        users = sum(1 for line in text.splitlines()
                    if '"type":"user"' in line and '"tool_use_id"' not in line)
        rows.append((target.name, started, lines, users, round(target.stat().st_size / 1e6, 1), n))
        print(f"{target.name}: {lines} lines, {n} values redacted")
    (OUT / "index.md").write_text(
        "# Claude Code session export\n\n"
        "The transcripts of every Claude Code session used to build this project, oldest first, as Claude Code "
        "stores them (one JSON object per line). E-mail addresses and `.env` values are replaced by `[REDACTED]`. "
        "Written by `tools/export_session.py`.\n\n"
        "| File | Started (UTC) | Lines | User turns (approx.) | MB |\n|---|---|---|---|---|\n"
        + "".join(f"| `{f}` | {s} | {l} | {u} | {mb} |\n" for f, s, l, u, mb, _ in rows),
        encoding="utf-8", newline="\n")
    print(f"Wrote {len(rows)} transcripts and session/index.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

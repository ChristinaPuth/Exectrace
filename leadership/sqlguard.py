"""The chat agent's only way to read data. SPEC section 5, "SQL guard".

Enforced by SQLite itself, not by inspecting the SQL text:

- the agent's database is a snapshot holding only the two views
  `leadership` and `current_leadership`, copied as plain tables when the
  chat starts. The base tables do not exist in it, so they cannot be read.
  (SQLite reports some reads inside the views' subqueries without the
  view's name, so an authorizer on the full database could not tell them
  from a direct read of a base table.)
- the snapshot is opened read-only (`mode=ro`);
- an authorizer allows reading only those two tables; PRAGMA, ATTACH,
  writes, schema changes and transactions are denied;
- one statement per call, at most `row_limit` rows, and a time limit.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

VIEWS = ("leadership", "current_leadership")
_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)   # WITH RECURSIVE


class GuardError(Exception):
    """The query was refused or failed. The message is safe to show to the model."""


@dataclass
class Result:
    columns: list[str]
    rows: list[tuple]
    truncated: bool                 # more rows existed than the limit


def snapshot(source: str | Path, company_domain: str | None = None) -> Path:
    """Copy the two views into a new database file of their own. Optionally one company only."""
    source = Path(source).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    src = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    target = Path(tempfile.mkdtemp(prefix="exectrace-chat-")) / "chat.db"
    dst = sqlite3.connect(target)
    for view in VIEWS:
        where, args = ("WHERE company_domain = ?", (company_domain,)) if company_domain else ("", ())
        cur = src.execute(f"SELECT * FROM {view} {where}", args)
        columns = [d[0] for d in cur.description]
        dst.execute(f"CREATE TABLE {view} ({', '.join(columns)})")
        dst.executemany(f"INSERT INTO {view} VALUES ({', '.join('?' * len(columns))})", cur.fetchall())
    dst.commit()
    dst.close()
    src.close()
    return target


def open_readonly(source: str | Path, company_domain: str | None = None) -> sqlite3.Connection:
    """A read-only connection to a snapshot of the two views, with an authorizer that admits nothing else."""
    path = snapshot(source, company_domain)
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)

    def authorize(action, arg1, arg2, db_name, trigger):
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, _RECURSIVE):     # reading only
            return sqlite3.SQLITE_OK
        # Only the two view tables exist; other names read are a query's own WITH clauses.
        # SQLite's internal tables (the schema) are refused.
        if action == sqlite3.SQLITE_READ and not (arg1 or "").lower().startswith("sqlite_"):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY                    # the schema, PRAGMA, ATTACH, writes, DDL

    conn.set_authorizer(authorize)
    return conn


def run(conn: sqlite3.Connection, sql: str, *, row_limit: int = 50, timeout_s: float = 5.0) -> Result:
    """Run one SELECT through the guard. Raises GuardError with a short reason."""
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        raise GuardError("empty query")
    if not sqlite3.complete_statement(text + ";") or _has_second_statement(text):
        raise GuardError("only one statement per call is allowed")
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(text)
        rows = cur.fetchmany(row_limit + 1)
    except sqlite3.DatabaseError as e:
        message = str(e)
        if "interrupted" in message:
            message = f"the query took longer than {timeout_s:g} seconds"
        elif message.startswith("no such table"):
            message += "; only the tables leadership and current_leadership exist"
        elif "not authorized" in message or "prohibited" in message:
            message = ("not authorized: only SELECT queries on the views leadership and current_leadership "
                       "are allowed")
        raise GuardError(message) from None
    finally:
        conn.set_progress_handler(None, 0)
    columns = [d[0] for d in cur.description or []]
    return Result(columns, [tuple(r) for r in rows[:row_limit]], len(rows) > row_limit)


def _has_second_statement(text: str) -> bool:
    """True if a statement follows the first one. Semicolons inside strings do not count."""
    for i, ch in enumerate(text):
        if ch == ";" and sqlite3.complete_statement(text[: i + 1]) and text[i + 1:].strip():
            return True
    return False

"""Apply SQL files in ``migrations/`` that are not yet recorded in ``schema_migrations``."""

from __future__ import annotations

import time
from pathlib import Path

from rfp_intel.db.session import connect
from rfp_intel.logging_setup import get_logger

logger = get_logger("migrate")

ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS_DIR = ROOT / "migrations"

_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


def split_sql(script: str) -> list[str]:
    """Split a SQL script on semicolons that are outside quotes and line comments."""
    parts: list[str] = []
    buf: list[str] = []
    in_quote = False
    i = 0
    length = len(script)
    while i < length:
        ch = script[i]
        if in_quote:
            buf.append(ch)
            if ch == "'":
                if i + 1 < length and script[i + 1] == "'":
                    buf.append("'")
                    i += 2
                    continue
                in_quote = False
            i += 1
            continue
        if ch == "'":
            in_quote = True
            buf.append(ch)
            i += 1
            continue
        if ch == "-" and i + 1 < length and script[i + 1] == "-":
            end = script.find("\n", i)
            i = length if end == -1 else end
            continue
        if ch == ";":
            statement = "".join(buf).strip()
            if statement:
                parts.append(statement)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def upgrade() -> None:
    paths = sorted(MIGRATIONS_DIR.glob("*.sql"))
    conn = connect()
    try:
        conn.execute(_MIGRATIONS_TABLE)
        conn.commit()
        applied = {
            row["version"]
            for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
        }
        for path in paths:
            version = path.name
            if version in applied:
                continue
            logger.info("applying migration", extra={"extra_data": {"version": version}})
            for statement in split_sql(path.read_text(encoding="utf-8")):
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (%(version)s)",
                {"version": version},
            )
            conn.commit()
    finally:
        conn.close()


def main() -> None:
    """Retry until Postgres accepts connections, then apply pending migrations."""
    from rfp_intel.config import get_settings
    from rfp_intel.resilience import backoff_delay

    settings = get_settings()
    last: Exception | None = None
    for attempt in range(30):
        try:
            upgrade()
            return
        except Exception as exc:
            last = exc
            logger.exception("database is not ready for migrations")
            time.sleep(max(backoff_delay(attempt, settings.backoff_base_seconds, settings.backoff_cap_seconds), 0.2))
    raise SystemExit(f"migrations failed: {last}")


if __name__ == "__main__":
    main()

"""psycopg connections. Each scope opens one transaction and closes it."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row

from rfp_intel.config import get_settings

Conn = psycopg.Connection[dict[str, Any]]


def database_dsn() -> str:
    """libpq URL. A ``postgresql+psycopg://`` URL is accepted and the driver suffix is dropped."""
    url = get_settings().database_url.strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty. Set it in .env or the environment before connecting to Postgres.")
    if "://" in url:
        scheme, rest = url.split("://", 1)
        scheme = scheme.split("+", 1)[0]
        url = f"{scheme}://{rest}"
    return url


def connect() -> Conn:
    return psycopg.connect(database_dsn(), autocommit=False, row_factory=dict_row)


@contextmanager
def session_scope() -> Iterator[Conn]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

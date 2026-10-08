"""PostgreSQL access: connection helper, versioned migrations, advisory locks."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources

import psycopg

from .. import config as C


def connect(dsn: str | None = None, **kw) -> psycopg.Connection:
    return psycopg.connect(dsn or C.dsn(), **kw)


def _migration_files() -> list[tuple[str, str]]:
    root = resources.files("retailfc.warehouse") / "migrations"
    files = sorted((p for p in root.iterdir() if p.name.endswith(".sql")), key=lambda p: p.name)
    return [(p.name, p.read_text(encoding="utf-8")) for p in files]


def migrate(dsn: str | None = None) -> list[str]:
    """Apply pending migrations in order, each in its own transaction. Detects edited (checksum-changed) files."""
    applied: list[str] = []
    with connect(dsn, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(727001)")
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS public.schema_migrations "
                "(version TEXT PRIMARY KEY, checksum TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            done = dict(conn.execute("SELECT version, checksum FROM public.schema_migrations").fetchall())
            for name, sql in _migration_files():
                chk = hashlib.sha256(sql.encode()).hexdigest()
                if name in done:
                    if done[name] != chk:
                        raise RuntimeError(f"migration {name} was modified after being applied")
                    continue
                with conn.transaction():
                    conn.execute(sql)
                    conn.execute(
                        "INSERT INTO public.schema_migrations(version, checksum) VALUES (%s, %s)", (name, chk)
                    )
                applied.append(name)
        finally:
            conn.execute("SELECT pg_advisory_unlock(727001)")
    return applied


@contextmanager
def advisory_lock(conn: psycopg.Connection, key: int) -> Iterator[bool]:
    """Non-blocking session-level advisory lock; yields whether it was acquired."""
    got = conn.execute("SELECT pg_try_advisory_lock(%s)", (key,)).fetchone()[0]
    try:
        yield bool(got)
    finally:
        if got:
            conn.execute("SELECT pg_advisory_unlock(%s)", (key,))
            conn.commit()


LOCK_TRAINING = 727100  # training / challenger evaluation / promotion are mutually exclusive

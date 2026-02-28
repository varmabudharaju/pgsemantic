"""Database client for pgsemantic.

Provides sync psycopg3 connections with pgvector type registration.
"""
from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector  # type: ignore[import-untyped]
from psycopg.rows import dict_row

from pgsemantic.exceptions import ExtensionNotFoundError

logger = logging.getLogger(__name__)

# Type alias for a psycopg connection using dict rows.
# psycopg.Connection is generic over the row type; dict_row produces
# dict[str, <value>] rows. We use dict[str, object] to avoid Any.
DictConnection = psycopg.Connection[dict[str, object]]

# -- SQL Constants ----------------------------------------------------------

# Check whether the pgvector extension is installed and retrieve its version.
SQL_CHECK_VECTOR_EXTENSION = """
    SELECT extversion
    FROM pg_extension
    WHERE extname = 'vector';
"""

# Attempt to create the pgvector extension if it does not exist.
SQL_CREATE_VECTOR_EXTENSION = """
    CREATE EXTENSION IF NOT EXISTS vector;
"""


def parse_pgvector_version(version_str: str) -> tuple[int, int, int]:
    """Parse a pgvector version string like '0.7.4' into a tuple."""
    parts = version_str.strip().split(".")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


@contextmanager
def get_connection(
    database_url: str,
) -> Generator[DictConnection, None, None]:
    """Open a sync psycopg3 connection with pgvector type registration."""
    conn: DictConnection = psycopg.connect(
        database_url, row_factory=dict_row
    )
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def get_pgvector_version(
    conn: DictConnection,
) -> tuple[int, int, int] | None:
    """Check if pgvector is installed and return its version tuple, or None."""
    result = conn.execute(SQL_CHECK_VECTOR_EXTENSION).fetchone()
    if result is None:
        return None
    return parse_pgvector_version(str(result["extversion"]))


def ensure_pgvector_extension(
    conn: DictConnection,
) -> tuple[int, int, int]:
    """Ensure pgvector extension is installed. Attempts to create if missing.

    Returns the pgvector version tuple.
    Raises ExtensionNotFoundError if it cannot be installed.
    """
    version = get_pgvector_version(conn)
    if version is not None:
        return version

    try:
        conn.execute(SQL_CREATE_VECTOR_EXTENSION)
        conn.commit()
    except psycopg.errors.InsufficientPrivilege as e:
        raise ExtensionNotFoundError(
            "Could not install pgvector extension (permission denied).\n\n"
            "To enable pgvector on your host:\n"
            '  Supabase:   Dashboard → Database → Extensions → enable "vector"\n'
            "  Neon:       Run: CREATE EXTENSION vector;  (connect as project owner)\n"
            "  RDS:        Run: CREATE EXTENSION vector;  (requires rds_superuser role)\n"
            "  Railway:    Run: CREATE EXTENSION vector;  (connect as postgres user)\n"
            "  Bare metal: apt install postgresql-16-pgvector\n"
            "              Then: CREATE EXTENSION vector;"
        ) from e

    version = get_pgvector_version(conn)
    if version is None:
        raise ExtensionNotFoundError(
            "pgvector extension installation failed silently."
        )
    return version

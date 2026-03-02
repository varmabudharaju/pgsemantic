"""Shared fixtures for integration tests.

All integration tests require a running Postgres instance with pgvector.
Start one with: docker-compose up -d

The default connection string points to the docker-compose service:
  postgresql://postgres:password@localhost:5432/pgvector_dev
"""
from __future__ import annotations

import os
from collections.abc import Generator

import psycopg
import pytest
from psycopg.rows import dict_row

from pgsemantic.db.client import DictConnection, get_connection

# Default test database URL — matches docker-compose.yml
DEFAULT_TEST_DB_URL = (
    "postgresql://postgres:password@localhost:5432/pgvector_dev"
)

TEST_TABLE = "pgsemantic_test_products"

# Small test dataset — enough to validate behaviour, fast to embed
TEST_ROWS = [
    (1, "Wireless Headphones", "Premium wireless noise-canceling headphones with 30-hour battery life and comfortable over-ear design."),
    (2, "Bluetooth Speaker", "Portable waterproof Bluetooth speaker with deep bass and 360-degree sound for outdoor adventures."),
    (3, "USB-C Hub", "7-in-1 USB-C hub with HDMI 4K output, SD card reader, and 100W power delivery pass-through."),
    (4, "Mechanical Keyboard", "Full-size mechanical keyboard with Cherry MX Blue switches, RGB backlighting, and aluminum frame."),
    (5, "Laptop Stand", "Adjustable aluminum laptop stand with ergonomic height settings and cable management."),
    (6, "Webcam HD", "1080p HD webcam with auto-focus, built-in microphone, and privacy shutter for video calls."),
    (7, "Wireless Mouse", "Ergonomic wireless mouse with silent clicks, adjustable DPI, and USB-C rechargeable battery."),
    (8, "Monitor Light Bar", "LED monitor light bar with adjustable color temperature and brightness for reduced eye strain."),
    (9, "Power Bank", "20000mAh portable power bank with fast charging, dual USB-C ports, and LED display."),
    (10, "Noise Canceling Earbuds", "True wireless earbuds with active noise cancellation, transparency mode, and 8-hour playback."),
]


def _get_test_db_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_TEST_DB_URL)


def _is_postgres_available(url: str) -> bool:
    """Check if Postgres is reachable."""
    try:
        conn = psycopg.connect(url)
        conn.close()
        return True
    except psycopg.OperationalError:
        return False


@pytest.fixture(scope="session")
def db_url() -> str:
    """Return the test database URL, skipping if Postgres is unreachable."""
    url = _get_test_db_url()
    if not _is_postgres_available(url):
        pytest.skip(
            f"Postgres not available at {url}. Run: docker-compose up -d"
        )
    return url


@pytest.fixture(scope="session")
def _ensure_pgvector(db_url: str) -> None:
    """Ensure the pgvector extension is installed (once per session)."""
    conn = psycopg.connect(db_url)
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(scope="session")
def _create_test_table(
    db_url: str,
    _ensure_pgvector: None,
) -> Generator[None, None, None]:
    """Create and populate the test table once per session, drop at end."""
    conn = psycopg.connect(db_url)
    try:
        # Drop if leftover from a previous failed run
        conn.execute(f"DROP TABLE IF EXISTS {TEST_TABLE} CASCADE;")
        conn.execute(
            f"""
            CREATE TABLE {TEST_TABLE} (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL
            );
            """
        )
        for row_id, name, desc in TEST_ROWS:
            conn.execute(
                f"INSERT INTO {TEST_TABLE} (id, name, description) VALUES (%s, %s, %s);",
                (row_id, name, desc),
            )
        conn.commit()
    finally:
        conn.close()

    yield

    # Teardown: drop table and related objects
    conn = psycopg.connect(db_url)
    try:
        conn.execute(f"DROP TABLE IF EXISTS {TEST_TABLE} CASCADE;")
        conn.execute(
            f"DROP FUNCTION IF EXISTS pgvector_setup_{TEST_TABLE}_fn() CASCADE;"
        )
        # Clean up queue entries for our test table
        conn.execute(
            "DELETE FROM pgvector_setup_queue WHERE table_name = %s;",
            (TEST_TABLE,),
        )
        conn.commit()
    except Exception:
        pass  # Table/function may not exist
    finally:
        conn.close()


@pytest.fixture()
def db_conn(
    db_url: str,
    _create_test_table: None,
) -> Generator[DictConnection, None, None]:
    """Provide a psycopg connection with pgvector registered and test table ready."""
    with get_connection(db_url) as conn:
        yield conn


@pytest.fixture()
def raw_conn(
    db_url: str,
    _create_test_table: None,
) -> Generator[DictConnection, None, None]:
    """Provide a raw psycopg connection (no pgvector registration)."""
    conn: DictConnection = psycopg.connect(db_url, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()

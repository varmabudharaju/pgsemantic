# pgvector-setup MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a working CLI tool + MCP server that bootstraps semantic search on any PostgreSQL database — from `pgvector-setup inspect` through `pgvector-setup serve` with Claude Desktop integration.

**Architecture:** Vertical slice approach — build one complete flow end-to-end (inspect), then layer on apply+index, worker, and MCP server. Sync psycopg3 for CLI commands, async only in worker daemon. Both local (sentence-transformers) and OpenAI embedding providers.

**Tech Stack:** Python 3.10+, Typer, Rich, psycopg3 (sync), pgvector, sentence-transformers, openai, FastMCP, python-dotenv, pytest

**Reference:** All SQL constants, schemas, and specifications are in `CLAUDE.md` (single source of truth). Design decisions documented in `docs/plans/2026-02-27-mvp-design.md`.

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `pgvector_setup/__init__.py`

**Step 1: Create pyproject.toml**

Use the authoritative version from CLAUDE.md section "pyproject.toml (Authoritative)". Copy it exactly, including all dependencies, entry point, ruff config, mypy config, and pytest config.

**Step 2: Create docker-compose.yml**

Use the authoritative version from CLAUDE.md section "Docker Compose (Authoritative)". Uses `ankane/pgvector:pg16` image.

**Step 3: Create .env.example**

Use the authoritative version from CLAUDE.md section "Environment Variables". Include all env vars with comments explaining each one.

**Step 4: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg

# Virtual environments
.venv/
venv/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Secrets — never commit
.env

# Project
.pgvector-setup.json
benchmarks/results/

# OS
.DS_Store
Thumbs.db
```

**Step 5: Create pgvector_setup/__init__.py**

```python
"""pgvector-setup: Zero-config semantic search bootstrap for PostgreSQL."""

__version__ = "0.1.0"
```

**Step 6: Install and verify**

Run: `pip install -e ".[dev]"`
Expected: Successful install, `pgvector-setup` command available (will fail on import since cli.py doesn't exist yet — that's OK)

**Step 7: Initialize git repo and commit**

```bash
git init
git add pyproject.toml docker-compose.yml .env.example .gitignore pgvector_setup/__init__.py CLAUDE.md docs/
git commit -m "feat: project scaffold — pyproject.toml, docker-compose, .env.example"
```

---

## Task 2: Exceptions Module

**Files:**
- Create: `pgvector_setup/exceptions.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`

**Step 1: Create exceptions.py**

All typed exceptions from CLAUDE.md section "Exceptions (exceptions.py — Authoritative)":

```python
"""Typed exceptions for pgvector-setup.

All third-party exceptions (psycopg.Error, openai.APIError) must be caught
and re-raised as one of these typed exceptions with a human-readable message.
Never let a raw third-party exception propagate to CLI output or MCP responses.
"""


class PgvectorSetupError(Exception):
    """Base exception for all pgvector-setup errors."""


class ExtensionNotFoundError(PgvectorSetupError):
    """pgvector extension is not installed or not accessible."""


class TableNotFoundError(PgvectorSetupError):
    """The specified table does not exist in the database."""


class TableNotWatchedError(PgvectorSetupError):
    """The table exists but has not been set up with pgvector-setup apply."""


class ColumnNotFoundError(PgvectorSetupError):
    """The specified column does not exist on the table."""


class ColumnAlreadyExistsError(PgvectorSetupError):
    """The embedding column already exists on the table."""


class DimensionMismatchError(PgvectorSetupError):
    """The existing embedding column has different dimensions than the requested model."""


class EmbeddingProviderError(PgvectorSetupError):
    """Error from the embedding model (API error, model not found, etc.)."""


class QueueError(PgvectorSetupError):
    """Error interacting with the pgvector_setup_queue table."""


class ConfigError(PgvectorSetupError):
    """Error reading or writing .pgvector-setup.json."""


class MigrationError(PgvectorSetupError):
    """Error during zero-downtime model migration."""


class InvalidFilterError(PgvectorSetupError):
    """A filter key in hybrid_search does not correspond to a valid column."""
```

**Step 2: Create test directory structure**

```bash
mkdir -p tests/unit tests/integration tests/fixtures
touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py
```

**Step 3: Verify**

Run: `python -c "from pgvector_setup.exceptions import PgvectorSetupError; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pgvector_setup/exceptions.py tests/
git commit -m "feat: add typed exceptions module"
```

---

## Task 3: Config Module

**Files:**
- Create: `pgvector_setup/config.py`
- Create: `tests/unit/test_config.py`

**Step 1: Write test_config.py**

```python
"""Tests for config module — Settings loading and ProjectConfig read/write."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from pgvector_setup.config import (
    CONFIG_FILE_NAME,
    DEFAULT_INDEX_BATCH_SIZE,
    DEFAULT_LOCAL_DIMENSIONS,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_WORKER_BATCH_SIZE,
    DEFAULT_WORKER_MAX_RETRIES,
    DEFAULT_WORKER_POLL_INTERVAL_MS,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    OPENAI_DIMENSIONS,
    OPENAI_MODEL,
    ProjectConfig,
    Settings,
    TableConfig,
    load_project_config,
    load_settings,
    save_project_config,
)


class TestSettings:
    def test_load_settings_from_env(self) -> None:
        env = {
            "DATABASE_URL": "postgresql://test:test@localhost/testdb",
            "EMBEDDING_PROVIDER": "local",
            "LOG_LEVEL": "debug",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = load_settings()
        assert settings.database_url == "postgresql://test:test@localhost/testdb"
        assert settings.embedding_provider == "local"
        assert settings.log_level == "debug"

    def test_load_settings_defaults(self) -> None:
        env = {"DATABASE_URL": "postgresql://test:test@localhost/testdb"}
        with patch.dict(os.environ, env, clear=False):
            settings = load_settings()
        assert settings.embedding_provider == "local"
        assert settings.worker_batch_size == DEFAULT_WORKER_BATCH_SIZE
        assert settings.worker_poll_interval_ms == DEFAULT_WORKER_POLL_INTERVAL_MS
        assert settings.worker_max_retries == DEFAULT_WORKER_MAX_RETRIES
        assert settings.mcp_transport == "stdio"
        assert settings.log_level == "info"

    def test_load_settings_missing_database_url(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings()
        assert settings.database_url is None


class TestProjectConfig:
    def test_save_and_load_project_config(self, tmp_path: Path) -> None:
        config = ProjectConfig(
            version="0.1.0",
            tables=[
                TableConfig(
                    table="products",
                    schema="public",
                    column="description",
                    embedding_column="embedding",
                    model="local",
                    model_name=DEFAULT_LOCAL_MODEL,
                    dimensions=DEFAULT_LOCAL_DIMENSIONS,
                    hnsw_m=HNSW_M,
                    hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
                    applied_at="2026-02-27T10:00:00Z",
                )
            ],
        )
        config_path = tmp_path / CONFIG_FILE_NAME
        save_project_config(config, config_path)

        loaded = load_project_config(config_path)
        assert loaded.version == "0.1.0"
        assert len(loaded.tables) == 1
        assert loaded.tables[0].table == "products"
        assert loaded.tables[0].dimensions == DEFAULT_LOCAL_DIMENSIONS

    def test_load_missing_config_returns_none(self, tmp_path: Path) -> None:
        config_path = tmp_path / CONFIG_FILE_NAME
        result = load_project_config(config_path)
        assert result is None

    def test_get_table_config(self, tmp_path: Path) -> None:
        config = ProjectConfig(
            version="0.1.0",
            tables=[
                TableConfig(
                    table="products",
                    schema="public",
                    column="description",
                    embedding_column="embedding",
                    model="local",
                    model_name=DEFAULT_LOCAL_MODEL,
                    dimensions=DEFAULT_LOCAL_DIMENSIONS,
                    hnsw_m=HNSW_M,
                    hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
                    applied_at="2026-02-27T10:00:00Z",
                )
            ],
        )
        tc = config.get_table_config("products")
        assert tc is not None
        assert tc.column == "description"

        tc2 = config.get_table_config("nonexistent")
        assert tc2 is None


class TestConstants:
    def test_hnsw_defaults(self) -> None:
        assert HNSW_M == 16
        assert HNSW_EF_CONSTRUCTION == 64

    def test_model_defaults(self) -> None:
        assert DEFAULT_LOCAL_MODEL == "all-MiniLM-L6-v2"
        assert DEFAULT_LOCAL_DIMENSIONS == 384
        assert OPENAI_MODEL == "text-embedding-3-small"
        assert OPENAI_DIMENSIONS == 1536

    def test_worker_defaults(self) -> None:
        assert DEFAULT_WORKER_BATCH_SIZE == 10
        assert DEFAULT_WORKER_POLL_INTERVAL_MS == 500
        assert DEFAULT_WORKER_MAX_RETRIES == 3
        assert DEFAULT_INDEX_BATCH_SIZE == 32
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pgvector_setup.config'`

**Step 3: Write config.py**

```python
"""Configuration for pgvector-setup.

Loads settings from environment variables (via python-dotenv) and manages
the .pgvector-setup.json project config file.

All magic numbers live here as named constants. Never hardcode them inline.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from pgvector_setup.exceptions import ConfigError

logger = logging.getLogger(__name__)

# ── HNSW index parameters ─────────────────────────────────────────────────
# Do not change without updating CLAUDE.md and docs/
HNSW_M: int = 16
HNSW_EF_CONSTRUCTION: int = 64
HNSW_EF_SEARCH: int = 40

# ── Default embedding models ──────────────────────────────────────────────
DEFAULT_LOCAL_MODEL: str = "all-MiniLM-L6-v2"
DEFAULT_LOCAL_DIMENSIONS: int = 384

OPENAI_MODEL: str = "text-embedding-3-small"
OPENAI_DIMENSIONS: int = 1536

# ── Worker defaults ───────────────────────────────────────────────────────
DEFAULT_WORKER_BATCH_SIZE: int = 10
DEFAULT_WORKER_POLL_INTERVAL_MS: int = 500
DEFAULT_WORKER_MAX_RETRIES: int = 3
DEFAULT_INDEX_BATCH_SIZE: int = 32

# ── Worker heartbeat ─────────────────────────────────────────────────────
WORKER_HEARTBEAT_INTERVAL_S: int = 60

# ── Queue table name — never change after initial release ─────────────────
QUEUE_TABLE_NAME: str = "pgvector_setup_queue"

# ── Config file name ─────────────────────────────────────────────────────
CONFIG_FILE_NAME: str = ".pgvector-setup.json"


@dataclass
class Settings:
    """Runtime settings loaded from environment variables."""

    database_url: Optional[str] = None
    embedding_provider: str = "local"
    openai_api_key: Optional[str] = None
    mcp_transport: str = "stdio"
    mcp_port: int = 3000
    mcp_auth_token: Optional[str] = None
    worker_batch_size: int = DEFAULT_WORKER_BATCH_SIZE
    worker_poll_interval_ms: int = DEFAULT_WORKER_POLL_INTERVAL_MS
    worker_max_retries: int = DEFAULT_WORKER_MAX_RETRIES
    log_level: str = "info"


def load_settings() -> Settings:
    """Load settings from .env file and environment variables.

    Never read os.environ directly elsewhere — always go through this function.
    """
    load_dotenv()
    return Settings(
        database_url=os.environ.get("DATABASE_URL"),
        embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "local"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        mcp_transport=os.environ.get("MCP_TRANSPORT", "stdio"),
        mcp_port=int(os.environ.get("MCP_PORT", "3000")),
        mcp_auth_token=os.environ.get("MCP_AUTH_TOKEN"),
        worker_batch_size=int(
            os.environ.get("WORKER_BATCH_SIZE", str(DEFAULT_WORKER_BATCH_SIZE))
        ),
        worker_poll_interval_ms=int(
            os.environ.get(
                "WORKER_POLL_INTERVAL_MS", str(DEFAULT_WORKER_POLL_INTERVAL_MS)
            )
        ),
        worker_max_retries=int(
            os.environ.get("WORKER_MAX_RETRIES", str(DEFAULT_WORKER_MAX_RETRIES))
        ),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )


@dataclass
class TableConfig:
    """Configuration for a single watched table."""

    table: str
    schema: str
    column: str
    embedding_column: str
    model: str
    model_name: str
    dimensions: int
    hnsw_m: int
    hnsw_ef_construction: int
    applied_at: str


@dataclass
class ProjectConfig:
    """Persistent project config stored in .pgvector-setup.json."""

    version: str
    tables: list[TableConfig] = field(default_factory=list)

    def get_table_config(self, table_name: str) -> Optional[TableConfig]:
        """Get config for a specific table, or None if not found."""
        for tc in self.tables:
            if tc.table == table_name:
                return tc
        return None


def save_project_config(config: ProjectConfig, path: Optional[Path] = None) -> None:
    """Write project config to .pgvector-setup.json."""
    if path is None:
        path = Path.cwd() / CONFIG_FILE_NAME
    try:
        data = asdict(config)
        path.write_text(json.dumps(data, indent=2) + "\n")
    except OSError as e:
        raise ConfigError(f"Failed to write config to {path}: {e}") from e


def load_project_config(path: Optional[Path] = None) -> Optional[ProjectConfig]:
    """Read project config from .pgvector-setup.json. Returns None if file doesn't exist."""
    if path is None:
        path = Path.cwd() / CONFIG_FILE_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        tables = [TableConfig(**t) for t in data.get("tables", [])]
        return ProjectConfig(version=data["version"], tables=tables)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ConfigError(f"Invalid config file {path}: {e}") from e
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_config.py -v`
Expected: All tests pass

**Step 5: Run mypy and ruff**

Run: `mypy pgvector_setup/config.py pgvector_setup/exceptions.py --strict && ruff check pgvector_setup/`
Expected: Zero errors

**Step 6: Commit**

```bash
git add pgvector_setup/config.py tests/unit/test_config.py
git commit -m "feat: add config module with Settings and ProjectConfig"
```

---

## Task 4: Database Client

**Files:**
- Create: `pgvector_setup/db/__init__.py`
- Create: `pgvector_setup/db/client.py`

**Step 1: Create db/__init__.py**

Empty `__init__.py` file.

**Step 2: Write client.py**

```python
"""Database client for pgvector-setup.

Provides sync psycopg3 connections with pgvector type registration.
Async connections are used only in worker/daemon.py.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Optional

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from pgvector_setup.exceptions import ExtensionNotFoundError

logger = logging.getLogger(__name__)

# Check if pgvector extension is installed and get its version
SQL_CHECK_VECTOR_EXTENSION = """
    SELECT extversion
    FROM pg_extension
    WHERE extname = 'vector';
"""

# Attempt to install pgvector extension
SQL_CREATE_VECTOR_EXTENSION = """
    CREATE EXTENSION IF NOT EXISTS vector;
"""


def parse_pgvector_version(version_str: str) -> tuple[int, int, int]:
    """Parse a pgvector version string like '0.7.4' into a tuple (0, 7, 4)."""
    parts = version_str.strip().split(".")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


@contextmanager
def get_connection(database_url: str) -> Generator[psycopg.Connection[dict_row], None, None]:
    """Open a sync psycopg3 connection with pgvector type registration.

    Usage:
        with get_connection(url) as conn:
            conn.execute("SELECT ...")
    """
    conn = psycopg.connect(database_url, row_factory=dict_row)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def get_pgvector_version(conn: psycopg.Connection[dict_row]) -> Optional[tuple[int, int, int]]:
    """Check if pgvector is installed and return its version.

    Returns (major, minor, patch) tuple, or None if pgvector is not installed.
    """
    result = conn.execute(SQL_CHECK_VECTOR_EXTENSION).fetchone()
    if result is None:
        return None
    return parse_pgvector_version(result["extversion"])


def ensure_pgvector_extension(conn: psycopg.Connection[dict_row]) -> tuple[int, int, int]:
    """Ensure pgvector extension is installed. Attempts to create it if missing.

    Returns the pgvector version tuple.
    Raises ExtensionNotFoundError if it cannot be installed.
    """
    version = get_pgvector_version(conn)
    if version is not None:
        return version

    # Try to install it
    try:
        conn.execute(SQL_CREATE_VECTOR_EXTENSION)
        conn.commit()
    except psycopg.errors.InsufficientPrivilege as e:
        raise ExtensionNotFoundError(
            "Could not install pgvector extension (permission denied).\n\n"
            "To enable pgvector on your host:\n"
            "  Supabase:   Dashboard → Database → Extensions → enable \"vector\"\n"
            "  Neon:       Run: CREATE EXTENSION vector;  (connect as project owner)\n"
            "  RDS:        Run: CREATE EXTENSION vector;  (requires rds_superuser role)\n"
            "  Railway:    Run: CREATE EXTENSION vector;  (connect as postgres user)\n"
            "  Bare metal: apt install postgresql-16-pgvector\n"
            "              Then: CREATE EXTENSION vector;"
        ) from e

    version = get_pgvector_version(conn)
    if version is None:
        raise ExtensionNotFoundError("pgvector extension installation failed silently.")
    return version
```

**Step 3: Verify import**

Run: `python -c "from pgvector_setup.db.client import get_connection; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pgvector_setup/db/
git commit -m "feat: add sync database client with pgvector registration"
```

---

## Task 5: Introspect Module

**Files:**
- Create: `pgvector_setup/db/introspect.py`
- Create: `tests/unit/test_introspect.py`

**Step 1: Write test_introspect.py**

```python
"""Tests for database introspection — column scoring and schema parsing."""
from unittest.mock import MagicMock, patch

import pytest

from pgvector_setup.db.introspect import (
    ColumnCandidate,
    score_column,
    SEMANTIC_COLUMN_NAMES,
    LOW_VALUE_COLUMN_NAMES,
)


class TestScoreColumn:
    def test_long_text_high_score(self) -> None:
        candidate = ColumnCandidate(
            table_name="products",
            table_schema="public",
            column_name="description",
            data_type="text",
            avg_length=847.0,
            sampled_rows=50,
        )
        score = score_column(candidate)
        assert score >= 2.5  # ★★★ for long text + semantic name bonus

    def test_medium_text_medium_score(self) -> None:
        candidate = ColumnCandidate(
            table_name="users",
            table_schema="public",
            column_name="bio",
            data_type="text",
            avg_length=143.0,
            sampled_rows=50,
        )
        score = score_column(candidate)
        assert 1.5 <= score <= 3.0  # ★★☆ base + semantic name bonus

    def test_short_text_low_score(self) -> None:
        candidate = ColumnCandidate(
            table_name="products",
            table_schema="public",
            column_name="name",
            data_type="character varying",
            avg_length=34.0,
            sampled_rows=50,
        )
        score = score_column(candidate)
        assert score <= 1.5  # ★☆☆

    def test_low_value_column_forced_low(self) -> None:
        candidate = ColumnCandidate(
            table_name="users",
            table_schema="public",
            column_name="email",
            data_type="character varying",
            avg_length=200.0,
            sampled_rows=50,
        )
        score = score_column(candidate)
        assert score == 1.0  # Forced ★☆☆

    def test_semantic_column_name_bonus(self) -> None:
        base_candidate = ColumnCandidate(
            table_name="articles",
            table_schema="public",
            column_name="data",
            data_type="text",
            avg_length=500.0,
            sampled_rows=50,
        )
        semantic_candidate = ColumnCandidate(
            table_name="articles",
            table_schema="public",
            column_name="content",
            data_type="text",
            avg_length=500.0,
            sampled_rows=50,
        )
        base_score = score_column(base_candidate)
        semantic_score = score_column(semantic_candidate)
        assert semantic_score > base_score

    def test_zero_avg_length(self) -> None:
        candidate = ColumnCandidate(
            table_name="test",
            table_schema="public",
            column_name="notes",
            data_type="text",
            avg_length=0.0,
            sampled_rows=0,
        )
        score = score_column(candidate)
        assert score >= 0.0

    def test_semantic_column_names_list(self) -> None:
        assert "description" in SEMANTIC_COLUMN_NAMES
        assert "body" in SEMANTIC_COLUMN_NAMES
        assert "content" in SEMANTIC_COLUMN_NAMES

    def test_low_value_column_names_list(self) -> None:
        assert "id" in LOW_VALUE_COLUMN_NAMES
        assert "email" in LOW_VALUE_COLUMN_NAMES
        assert "uuid" in LOW_VALUE_COLUMN_NAMES
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_introspect.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write introspect.py**

```python
"""Database introspection — scan schemas, score columns for semantic search suitability.

All SQL is defined as module-level constants. Table/column names used in
SQL_SAMPLE_AVG_LENGTH are validated against information_schema before formatting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import psycopg
from psycopg.rows import dict_row

from pgvector_setup.exceptions import TableNotFoundError

logger = logging.getLogger(__name__)

# ── SQL Constants ─────────────────────────────────────────────────────────

# Get all user-created tables (excluding system schemas)
SQL_GET_USER_TABLES = """
    SELECT
        t.table_name,
        t.table_schema
    FROM information_schema.tables t
    WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
      AND t.table_type = 'BASE TABLE'
    ORDER BY t.table_name;
"""

# Get all text-like columns for a specific table
SQL_GET_TEXT_COLUMNS = """
    SELECT
        column_name,
        data_type,
        character_maximum_length,
        is_nullable
    FROM information_schema.columns
    WHERE table_name  = %(table_name)s
      AND table_schema = %(schema_name)s
      AND data_type IN ('text', 'character varying', 'jsonb', 'character')
    ORDER BY ordinal_position;
"""

# Sample avg text length for a column — TABLESAMPLE avoids full sequential scan
# table and column are validated against information_schema before formatting
SQL_SAMPLE_AVG_LENGTH = """
    SELECT
        avg(length({column}::TEXT))  AS avg_length,
        count(*)                     AS sampled_rows
    FROM {table} TABLESAMPLE SYSTEM(5)
    WHERE {column} IS NOT NULL;
"""

# Check if pgvector extension is installed
SQL_CHECK_VECTOR_EXTENSION = """
    SELECT extname, extversion
    FROM pg_extension
    WHERE extname = 'vector';
"""

# Check if a column exists on a table
SQL_COLUMN_EXISTS = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name   = %(table_name)s
      AND column_name  = %(column_name)s
      AND table_schema = %(schema_name)s;
"""

# Get all columns of a table (for building dynamic WHERE in hybrid_search)
SQL_GET_ALL_COLUMNS = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name   = %(table_name)s
      AND table_schema = %(schema_name)s
    ORDER BY ordinal_position;
"""

# Get row count estimate for a table (fast, uses pg_class statistics)
SQL_GET_ROW_COUNT = """
    SELECT reltuples::BIGINT AS estimate
    FROM pg_class
    WHERE relname = %(table_name)s;
"""

# Get primary key column(s) for a table
SQL_GET_PRIMARY_KEY = """
    SELECT kcu.column_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
        AND tc.table_schema = kcu.table_schema
    WHERE tc.constraint_type = 'PRIMARY KEY'
      AND tc.table_name = %(table_name)s
      AND tc.table_schema = %(schema_name)s
    ORDER BY kcu.ordinal_position;
"""

# ── Column name scoring sets ─────────────────────────────────────────────

# Column names that strongly suggest semantic content — gets +0.5 score bonus
SEMANTIC_COLUMN_NAMES: frozenset[str] = frozenset({
    "description", "body", "content", "text", "summary", "notes",
    "bio", "review", "comment", "message", "details", "abstract", "excerpt",
})

# Column names that are almost never useful for semantic search — forced to score 1.0
LOW_VALUE_COLUMN_NAMES: frozenset[str] = frozenset({
    "id", "status", "code", "type", "slug", "url", "email", "phone",
    "uuid", "hash", "token",
})


@dataclass
class ColumnCandidate:
    """A text column that might be a good candidate for semantic search."""

    table_name: str
    table_schema: str
    column_name: str
    data_type: str
    avg_length: float
    sampled_rows: int


def score_column(candidate: ColumnCandidate) -> float:
    """Score a column for semantic search suitability.

    Scoring is heuristic (text length + column name patterns):
    - avg_length > 200 chars → base score 3.0 (★★★)
    - avg_length 50–200 chars → base score 2.0 (★★☆)
    - avg_length < 50 chars → base score 1.0 (★☆☆)
    - Column name in SEMANTIC_COLUMN_NAMES → +0.5
    - Column name in LOW_VALUE_COLUMN_NAMES → forced to 1.0
    """
    # Force low score for low-value columns regardless of text length
    if candidate.column_name in LOW_VALUE_COLUMN_NAMES:
        return 1.0

    # Base score from average text length
    if candidate.avg_length > 200:
        score = 3.0
    elif candidate.avg_length >= 50:
        score = 2.0
    else:
        score = 1.0

    # Bonus for semantically meaningful column names
    if candidate.column_name in SEMANTIC_COLUMN_NAMES:
        score += 0.5

    return score


def score_to_stars(score: float) -> str:
    """Convert a numeric score to a star rating string."""
    if score >= 2.5:
        return "★★★"
    elif score >= 1.5:
        return "★★☆"
    else:
        return "★☆☆"


def get_user_tables(
    conn: psycopg.Connection[dict_row],
) -> list[dict[str, str]]:
    """Get all user-created tables (excluding system schemas)."""
    result = conn.execute(SQL_GET_USER_TABLES).fetchall()
    return [dict(row) for row in result]


def get_text_columns(
    conn: psycopg.Connection[dict_row],
    table_name: str,
    schema_name: str = "public",
) -> list[dict[str, object]]:
    """Get all text-like columns for a specific table."""
    result = conn.execute(
        SQL_GET_TEXT_COLUMNS,
        {"table_name": table_name, "schema_name": schema_name},
    ).fetchall()
    return [dict(row) for row in result]


def sample_avg_length(
    conn: psycopg.Connection[dict_row],
    table_name: str,
    column_name: str,
    schema_name: str = "public",
) -> tuple[float, int]:
    """Sample average text length for a column using TABLESAMPLE.

    Returns (avg_length, sampled_rows). Table and column names are validated
    against information_schema before being formatted into the SQL string.
    """
    # Validate that the table and column exist (prevents SQL injection)
    exists = conn.execute(
        SQL_COLUMN_EXISTS,
        {
            "table_name": table_name,
            "column_name": column_name,
            "schema_name": schema_name,
        },
    ).fetchone()
    if exists is None:
        raise TableNotFoundError(
            f"Column '{column_name}' not found on table '{schema_name}.{table_name}'"
        )

    # Safe to format — names are validated against information_schema
    qualified_table = f'"{schema_name}"."{table_name}"'
    quoted_column = f'"{column_name}"'
    sql = SQL_SAMPLE_AVG_LENGTH.format(table=qualified_table, column=quoted_column)

    result = conn.execute(sql).fetchone()
    if result is None or result["avg_length"] is None:
        return (0.0, 0)
    return (float(result["avg_length"]), int(result["sampled_rows"]))


def get_column_names(
    conn: psycopg.Connection[dict_row],
    table_name: str,
    schema_name: str = "public",
) -> set[str]:
    """Get all column names for a table (used for filter validation in hybrid_search)."""
    result = conn.execute(
        SQL_GET_ALL_COLUMNS,
        {"table_name": table_name, "schema_name": schema_name},
    ).fetchall()
    return {row["column_name"] for row in result}


def column_exists(
    conn: psycopg.Connection[dict_row],
    table_name: str,
    column_name: str,
    schema_name: str = "public",
) -> bool:
    """Check if a specific column exists on a table."""
    result = conn.execute(
        SQL_COLUMN_EXISTS,
        {
            "table_name": table_name,
            "column_name": column_name,
            "schema_name": schema_name,
        },
    ).fetchone()
    return result is not None


def get_row_count_estimate(
    conn: psycopg.Connection[dict_row],
    table_name: str,
) -> int:
    """Get estimated row count from pg_class (fast, no full scan)."""
    result = conn.execute(
        SQL_GET_ROW_COUNT,
        {"table_name": table_name},
    ).fetchone()
    if result is None:
        return 0
    return max(0, int(result["estimate"]))


def get_primary_key_columns(
    conn: psycopg.Connection[dict_row],
    table_name: str,
    schema_name: str = "public",
) -> list[str]:
    """Get primary key column names for a table."""
    result = conn.execute(
        SQL_GET_PRIMARY_KEY,
        {"table_name": table_name, "schema_name": schema_name},
    ).fetchall()
    return [row["column_name"] for row in result]


def inspect_database(
    conn: psycopg.Connection[dict_row],
) -> list[ColumnCandidate]:
    """Scan all user tables and score text columns for semantic search suitability.

    Returns a list of ColumnCandidate objects sorted by score (highest first),
    then by avg_length (longest first).
    """
    candidates: list[ColumnCandidate] = []
    tables = get_user_tables(conn)

    for table_info in tables:
        table_name = table_info["table_name"]
        schema_name = table_info["table_schema"]

        # Skip our own queue table
        if table_name == "pgvector_setup_queue":
            continue

        text_columns = get_text_columns(conn, table_name, schema_name)
        for col in text_columns:
            col_name = str(col["column_name"])
            try:
                avg_len, sampled = sample_avg_length(
                    conn, table_name, col_name, schema_name
                )
            except Exception:
                logger.warning(
                    "Failed to sample %s.%s, skipping", table_name, col_name
                )
                avg_len, sampled = 0.0, 0

            candidates.append(
                ColumnCandidate(
                    table_name=table_name,
                    table_schema=schema_name,
                    column_name=col_name,
                    data_type=str(col["data_type"]),
                    avg_length=avg_len,
                    sampled_rows=sampled,
                )
            )

    # Sort by score descending, then by avg_length descending
    candidates.sort(
        key=lambda c: (score_column(c), c.avg_length),
        reverse=True,
    )
    return candidates
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_introspect.py -v`
Expected: All tests pass

**Step 5: Run mypy**

Run: `mypy pgvector_setup/db/ --strict`
Expected: Zero errors (may need to adjust type annotations)

**Step 6: Commit**

```bash
git add pgvector_setup/db/introspect.py tests/unit/test_introspect.py
git commit -m "feat: add database introspection with column scoring heuristics"
```

---

## Task 6: CLI Root + Inspect Command

**Files:**
- Create: `pgvector_setup/cli.py`
- Create: `pgvector_setup/commands/__init__.py`
- Create: `pgvector_setup/commands/inspect.py`

**Step 1: Write cli.py**

```python
"""CLI root — registers all subcommands for pgvector-setup."""
from __future__ import annotations

import typer

from pgvector_setup import __version__

app = typer.Typer(
    name="pgvector-setup",
    help="Zero-config semantic search bootstrap for any PostgreSQL database.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    if value:
        print(f"pgvector-setup {__version__}")  # noqa: T201
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v", help="Show version and exit.", callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """pgvector-setup: Point it at your Postgres database. Get semantic search in 60 seconds."""


# Import and register subcommands
from pgvector_setup.commands.inspect import inspect_command  # noqa: E402

app.command(name="inspect")(inspect_command)
```

Note: Additional commands will be registered here as they are built. Import them at module level after the app definition.

**Step 2: Write commands/__init__.py**

Empty file.

**Step 3: Write commands/inspect.py**

```python
"""pgvector-setup inspect — scan a database and score columns for semantic search."""
from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pgvector_setup.db.client import get_connection, get_pgvector_version
from pgvector_setup.db.introspect import (
    ColumnCandidate,
    inspect_database,
    score_column,
    score_to_stars,
)

console = Console()


def inspect_command(
    database_url: str = typer.Argument(
        ...,
        help="PostgreSQL connection string (e.g. postgresql://user:pass@host/db)",
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Output results as JSON for programmatic use.",
    ),
) -> None:
    """Scan a database and score columns for semantic search suitability."""
    try:
        with get_connection(database_url) as conn:
            # Check pgvector
            pgv_version = get_pgvector_version(conn)

            candidates = inspect_database(conn)

        if not candidates:
            console.print(Panel(
                "[yellow]No text columns found in this database.[/yellow]\n\n"
                "pgvector-setup looks for TEXT, VARCHAR, and JSONB columns.\n"
                "Make sure your database has tables with text data.",
                title="No Candidates Found",
                border_style="yellow",
            ))
            raise typer.Exit(code=0)

        if output_json:
            _print_json(candidates, pgv_version)
        else:
            _print_rich_table(candidates, database_url, pgv_version)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to inspect database:[/red]\n\n{e}",
            title="Connection Error",
            border_style="red",
        ))
        raise typer.Exit(code=1)


def _print_rich_table(
    candidates: list[ColumnCandidate],
    database_url: str,
    pgv_version: Optional[tuple[int, int, int]],
) -> None:
    """Render inspection results as a Rich table."""
    # Header
    pgv_str = f"pgvector {'.'.join(str(v) for v in pgv_version)}" if pgv_version else "pgvector not installed"
    # Mask password in URL for display
    display_url = database_url.split("@")[-1] if "@" in database_url else database_url

    console.print()
    console.print(Panel(
        f"Semantic Search Candidates — {display_url}\n"
        f"[dim]{pgv_str}[/dim]",
        border_style="blue",
    ))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Table", style="cyan")
    table.add_column("Column", style="green")
    table.add_column("Score")
    table.add_column("Avg Length", justify="right")
    table.add_column("Type", style="dim")

    for c in candidates:
        score = score_column(c)
        stars = score_to_stars(score)
        star_style = "green" if score >= 2.5 else "yellow" if score >= 1.5 else "dim"
        avg_len_str = f"{c.avg_length:,.0f} chars" if c.avg_length > 0 else "—"

        table.add_row(
            c.table_name,
            c.column_name,
            f"[{star_style}]{stars}[/{star_style}]",
            avg_len_str,
            c.data_type,
        )

    console.print(table)
    console.print()
    console.print(
        "[dim]ℹ  Scoring is heuristic (text length + column name patterns).[/dim]\n"
        "[dim]   Always verify recommendations make sense for your use case.[/dim]"
    )

    # Suggest next step with highest-scored candidate
    if candidates:
        best = candidates[0]
        console.print()
        console.print(f"Next step:")
        console.print(
            f"  [cyan]pgvector-setup apply --table {best.table_name} "
            f"--column {best.column_name}[/cyan]"
        )
    console.print()


def _print_json(
    candidates: list[ColumnCandidate],
    pgv_version: Optional[tuple[int, int, int]],
) -> None:
    """Output results as JSON for programmatic use."""
    data = {
        "pgvector_version": ".".join(str(v) for v in pgv_version) if pgv_version else None,
        "candidates": [
            {
                "table": c.table_name,
                "schema": c.table_schema,
                "column": c.column_name,
                "data_type": c.data_type,
                "avg_length": round(c.avg_length, 1),
                "sampled_rows": c.sampled_rows,
                "score": score_column(c),
                "stars": score_to_stars(score_column(c)),
            }
            for c in candidates
        ],
    }
    console.print(json.dumps(data, indent=2))
```

**Step 4: Verify CLI works**

Run: `pip install -e ".[dev]" && pgvector-setup --version`
Expected: `pgvector-setup 0.1.0`

Run: `pgvector-setup --help`
Expected: Shows help with `inspect` subcommand listed

**Step 5: Test against Docker Postgres (manual smoke test)**

Run: `docker-compose up -d && sleep 5 && pgvector-setup inspect postgresql://postgres:password@localhost:5432/pgvector_dev`
Expected: Either shows "No text columns found" (empty DB) or a Rich table. No crashes.

**Step 6: Commit**

```bash
git add pgvector_setup/cli.py pgvector_setup/commands/
git commit -m "feat: add CLI root and inspect command with Rich table output"
```

---

## Task 7: Embedding Providers

**Files:**
- Create: `pgvector_setup/embeddings/__init__.py`
- Create: `pgvector_setup/embeddings/base.py`
- Create: `pgvector_setup/embeddings/local.py`
- Create: `pgvector_setup/embeddings/openai_provider.py`
- Create: `tests/unit/test_embeddings.py`

**Step 1: Write test_embeddings.py**

```python
"""Tests for embedding providers — interface compliance and batching."""
from unittest.mock import MagicMock, patch

import pytest

from pgvector_setup.embeddings.base import EmbeddingProvider, ProviderConfig
from pgvector_setup.embeddings.local import LocalProvider
from pgvector_setup.embeddings.openai_provider import OpenAIProvider
from pgvector_setup.embeddings import get_provider
from pgvector_setup.config import DEFAULT_LOCAL_DIMENSIONS, OPENAI_DIMENSIONS


class TestProviderConfig:
    def test_provider_config_defaults(self) -> None:
        config = ProviderConfig(
            model_name="test-model",
            dimensions=384,
        )
        assert config.batch_size == 32

    def test_provider_config_custom_batch_size(self) -> None:
        config = ProviderConfig(
            model_name="test-model",
            dimensions=384,
            batch_size=64,
        )
        assert config.batch_size == 64


class TestLocalProvider:
    @patch("pgvector_setup.embeddings.local.SentenceTransformer")
    def test_embed_returns_correct_dimensions(self, mock_st_class: MagicMock) -> None:
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1] * DEFAULT_LOCAL_DIMENSIONS,
            [0.2] * DEFAULT_LOCAL_DIMENSIONS,
        ])
        mock_st_class.return_value = mock_model

        provider = LocalProvider()
        result = provider.embed(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == DEFAULT_LOCAL_DIMENSIONS
        mock_model.encode.assert_called_once()

    @patch("pgvector_setup.embeddings.local.SentenceTransformer")
    def test_embed_query_returns_single_vector(self, mock_st_class: MagicMock) -> None:
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1] * DEFAULT_LOCAL_DIMENSIONS,
        ])
        mock_st_class.return_value = mock_model

        provider = LocalProvider()
        result = provider.embed_query("test query")

        assert len(result) == DEFAULT_LOCAL_DIMENSIONS

    @patch("pgvector_setup.embeddings.local.SentenceTransformer")
    def test_embed_empty_list_raises(self, mock_st_class: MagicMock) -> None:
        provider = LocalProvider()
        with pytest.raises(ValueError, match="non-empty"):
            provider.embed([])

    @patch("pgvector_setup.embeddings.local.SentenceTransformer")
    def test_config_has_correct_values(self, mock_st_class: MagicMock) -> None:
        provider = LocalProvider()
        assert provider.config.model_name == "all-MiniLM-L6-v2"
        assert provider.config.dimensions == DEFAULT_LOCAL_DIMENSIONS


class TestOpenAIProvider:
    @patch("pgvector_setup.embeddings.openai_provider.OpenAI")
    def test_embed_returns_correct_dimensions(self, mock_openai_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * OPENAI_DIMENSIONS
        mock_response = MagicMock()
        mock_response.data = [mock_embedding, mock_embedding]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test")
        result = provider.embed(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == OPENAI_DIMENSIONS

    @patch("pgvector_setup.embeddings.openai_provider.OpenAI")
    def test_embed_empty_list_raises(self, mock_openai_class: MagicMock) -> None:
        provider = OpenAIProvider(api_key="sk-test")
        with pytest.raises(ValueError, match="non-empty"):
            provider.embed([])

    @patch("pgvector_setup.embeddings.openai_provider.OpenAI")
    def test_config_has_correct_values(self, mock_openai_class: MagicMock) -> None:
        provider = OpenAIProvider(api_key="sk-test")
        assert provider.config.model_name == "text-embedding-3-small"
        assert provider.config.dimensions == OPENAI_DIMENSIONS


class TestGetProvider:
    @patch("pgvector_setup.embeddings.local.SentenceTransformer")
    def test_get_local_provider(self, mock_st: MagicMock) -> None:
        provider = get_provider("local")
        assert provider.config.model_name == "all-MiniLM-L6-v2"

    @patch("pgvector_setup.embeddings.openai_provider.OpenAI")
    def test_get_openai_provider(self, mock_openai: MagicMock) -> None:
        provider = get_provider("openai", api_key="sk-test")
        assert provider.config.model_name == "text-embedding-3-small"

    def test_get_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            get_provider("unknown")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_embeddings.py -v`
Expected: FAIL — module not found

**Step 3: Write base.py**

```python
"""Abstract embedding provider interface.

Uses typing.Protocol — not ABC. All providers must implement this interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ProviderConfig:
    """Configuration for an embedding provider."""

    model_name: str
    dimensions: int
    batch_size: int = 32


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol that all embedding providers must implement."""

    config: ProviderConfig

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per text.

        All vectors have length == self.config.dimensions.
        texts must be non-empty. Empty list raises ValueError.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text. Used at search time.

        Equivalent to embed([text])[0] but may use a different prompt prefix
        for asymmetric models.
        """
        ...
```

**Step 4: Write local.py**

```python
"""Local embedding provider using sentence-transformers.

Model: all-MiniLM-L6-v2
Dimensions: 384
Download: 22 MB (to ~/.cache/huggingface/ on first use)
Cost: Zero
Requirements: sentence-transformers (included in dependencies)
"""
from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from pgvector_setup.config import DEFAULT_LOCAL_DIMENSIONS, DEFAULT_LOCAL_MODEL
from pgvector_setup.embeddings.base import EmbeddingProvider, ProviderConfig
from pgvector_setup.exceptions import EmbeddingProviderError

logger = logging.getLogger(__name__)


class LocalProvider:
    """Embedding provider using sentence-transformers (all-MiniLM-L6-v2)."""

    config = ProviderConfig(
        model_name=DEFAULT_LOCAL_MODEL,
        dimensions=DEFAULT_LOCAL_DIMENSIONS,
        batch_size=32,
    )

    def __init__(self) -> None:
        try:
            self._model = SentenceTransformer(DEFAULT_LOCAL_MODEL)
        except Exception as e:
            raise EmbeddingProviderError(
                f"Failed to load model '{DEFAULT_LOCAL_MODEL}': {e}"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using sentence-transformers."""
        if not texts:
            raise ValueError("texts must be non-empty")
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()  # type: ignore[union-attr]
        except Exception as e:
            raise EmbeddingProviderError(f"Embedding failed: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self.embed([text])[0]
```

**Step 5: Write openai_provider.py**

```python
"""OpenAI embedding provider.

Model: text-embedding-3-small
Dimensions: 1536
Cost: $0.02 per 1M tokens
Requirements: OPENAI_API_KEY environment variable
"""
from __future__ import annotations

import logging

from openai import OpenAI

from pgvector_setup.config import OPENAI_DIMENSIONS, OPENAI_MODEL
from pgvector_setup.embeddings.base import ProviderConfig
from pgvector_setup.exceptions import EmbeddingProviderError

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """Embedding provider using OpenAI text-embedding-3-small."""

    config = ProviderConfig(
        model_name=OPENAI_MODEL,
        dimensions=OPENAI_DIMENSIONS,
        batch_size=100,
    )

    def __init__(self, api_key: str) -> None:
        try:
            self._client = OpenAI(api_key=api_key)
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to initialize OpenAI client: {e}") from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using OpenAI API."""
        if not texts:
            raise ValueError("texts must be non-empty")
        try:
            response = self._client.embeddings.create(
                model=OPENAI_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingProviderError(f"OpenAI embedding failed: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self.embed([text])[0]
```

**Step 6: Write embeddings/__init__.py**

```python
"""Embedding provider factory."""
from __future__ import annotations

from typing import Optional, Union

from pgvector_setup.embeddings.local import LocalProvider
from pgvector_setup.embeddings.openai_provider import OpenAIProvider


def get_provider(
    name: str,
    api_key: Optional[str] = None,
) -> Union[LocalProvider, OpenAIProvider]:
    """Factory function to get an embedding provider by name.

    Args:
        name: "local" or "openai"
        api_key: Required for "openai" provider

    Returns:
        An embedding provider instance
    """
    if name == "local":
        return LocalProvider()
    elif name == "openai":
        if api_key is None:
            raise ValueError(
                "OpenAI provider requires an API key. "
                "Set OPENAI_API_KEY in your .env file."
            )
        return OpenAIProvider(api_key=api_key)
    else:
        raise ValueError(
            f"Unknown embedding provider: '{name}'. "
            f"Supported providers: 'local', 'openai'"
        )
```

**Step 7: Run tests**

Run: `pytest tests/unit/test_embeddings.py -v`
Expected: All tests pass

**Step 8: Commit**

```bash
git add pgvector_setup/embeddings/ tests/unit/test_embeddings.py
git commit -m "feat: add local and OpenAI embedding providers with Protocol interface"
```

---

## Task 8: Queue Module

**Files:**
- Create: `pgvector_setup/db/queue.py`
- Create: `tests/unit/test_queue.py`

**Step 1: Write test_queue.py**

Test the SQL constants are defined and the function signatures work with mocked connections. Test that `create_queue_table`, `claim_batch`, `complete_job`, `fail_job` call the right SQL.

```python
"""Tests for queue module — job claim/complete/fail logic with mocked DB."""
from unittest.mock import MagicMock, call

import pytest

from pgvector_setup.db.queue import (
    SQL_CLAIM_BATCH,
    SQL_COMPLETE_JOB,
    SQL_CREATE_QUEUE_INDEX,
    SQL_CREATE_QUEUE_TABLE,
    SQL_FAIL_JOB,
    claim_batch,
    complete_job,
    create_queue_table,
    fail_job,
)


class TestCreateQueueTable:
    def test_executes_create_table_and_index(self) -> None:
        mock_conn = MagicMock()
        create_queue_table(mock_conn)
        calls = mock_conn.execute.call_args_list
        assert len(calls) == 2
        assert SQL_CREATE_QUEUE_TABLE in calls[0].args[0]
        assert SQL_CREATE_QUEUE_INDEX in calls[1].args[0]
        mock_conn.commit.assert_called_once()


class TestClaimBatch:
    def test_claim_returns_jobs(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            {"id": 1, "table_name": "products", "row_id": "42", "column_name": "description", "operation": "INSERT"},
        ]
        jobs = claim_batch(mock_conn, batch_size=10)
        assert len(jobs) == 1
        assert jobs[0]["table_name"] == "products"
        mock_conn.commit.assert_called_once()

    def test_claim_empty_queue(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        jobs = claim_batch(mock_conn, batch_size=10)
        assert jobs == []


class TestCompleteJob:
    def test_deletes_completed_job(self) -> None:
        mock_conn = MagicMock()
        complete_job(mock_conn, job_id=42)
        mock_conn.execute.assert_called_once()
        args = mock_conn.execute.call_args
        assert args[1] == {"job_id": 42}
        mock_conn.commit.assert_called_once()


class TestFailJob:
    def test_marks_job_as_failed(self) -> None:
        mock_conn = MagicMock()
        fail_job(mock_conn, job_id=42, error_msg="embedding timeout")
        mock_conn.execute.assert_called_once()
        args = mock_conn.execute.call_args
        assert args[1]["job_id"] == 42
        assert args[1]["error_msg"] == "embedding timeout"
        mock_conn.commit.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_queue.py -v`
Expected: FAIL

**Step 3: Write queue.py**

Use all SQL constants from CLAUDE.md section "db/queue.py SQL Constants". Implement `create_queue_table`, `claim_batch`, `complete_job`, `fail_job`, `count_pending`, `retry_job` functions.

```python
"""Queue table management for pgvector-setup.

The queue table (pgvector_setup_queue) is the CDC mechanism — triggers write jobs
into this table, and the worker daemon claims and processes them. Uses
SELECT FOR UPDATE SKIP LOCKED for atomic, deadlock-free batch claiming.
"""
from __future__ import annotations

import logging

import psycopg
from psycopg.rows import dict_row

from pgvector_setup.exceptions import QueueError

logger = logging.getLogger(__name__)

# ── SQL Constants ─────────────────────────────────────────────────────────

# Create queue table — one table for all watched tables
SQL_CREATE_QUEUE_TABLE = """
    CREATE TABLE IF NOT EXISTS pgvector_setup_queue (
        id          BIGSERIAL       PRIMARY KEY,
        table_name  TEXT            NOT NULL,
        row_id      TEXT            NOT NULL,
        column_name TEXT            NOT NULL,
        operation   TEXT            NOT NULL    CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
        status      TEXT            NOT NULL    DEFAULT 'pending'
                                                CHECK (status IN ('pending', 'processing', 'failed')),
        retries     INT             NOT NULL    DEFAULT 0,
        error_msg   TEXT,
        created_at  TIMESTAMPTZ     NOT NULL    DEFAULT NOW(),
        updated_at  TIMESTAMPTZ     NOT NULL    DEFAULT NOW(),
        UNIQUE (table_name, row_id, column_name)
    );
"""

# Partial index: only index pending jobs (worker polling uses this)
SQL_CREATE_QUEUE_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_pgvs_queue_pending
        ON pgvector_setup_queue (created_at ASC)
        WHERE status = 'pending';
"""

# Claim a batch of jobs atomically (SELECT FOR UPDATE SKIP LOCKED)
SQL_CLAIM_BATCH = """
    UPDATE pgvector_setup_queue
    SET status     = 'processing',
        updated_at = NOW()
    WHERE id IN (
        SELECT id
        FROM pgvector_setup_queue
        WHERE status = 'pending'
        ORDER BY created_at ASC
        LIMIT %(batch_size)s
        FOR UPDATE SKIP LOCKED
    )
    RETURNING id, table_name, row_id, column_name, operation;
"""

# Delete a completed job (completed jobs don't need to be kept)
SQL_COMPLETE_JOB = """
    DELETE FROM pgvector_setup_queue
    WHERE id = %(job_id)s;
"""

# Mark a job as failed with error message
SQL_FAIL_JOB = """
    UPDATE pgvector_setup_queue
    SET status     = 'failed',
        retries    = retries + 1,
        error_msg  = %(error_msg)s,
        updated_at = NOW()
    WHERE id = %(job_id)s;
"""

# Re-queue a failed job for retry (reset to pending if under retry limit)
SQL_RETRY_JOB = """
    UPDATE pgvector_setup_queue
    SET status     = 'pending',
        updated_at = NOW()
    WHERE id = %(job_id)s
      AND retries < %(max_retries)s;
"""

# Count pending jobs for a table
SQL_COUNT_PENDING = """
    SELECT count(*) AS pending_count
    FROM pgvector_setup_queue
    WHERE table_name = %(table_name)s
      AND status     = 'pending';
"""

# Count failed jobs for a table
SQL_COUNT_FAILED = """
    SELECT count(*) AS failed_count
    FROM pgvector_setup_queue
    WHERE table_name = %(table_name)s
      AND status     = 'failed';
"""


def create_queue_table(conn: psycopg.Connection[dict_row]) -> None:
    """Create the queue table and index if they don't exist."""
    try:
        conn.execute(SQL_CREATE_QUEUE_TABLE)
        conn.execute(SQL_CREATE_QUEUE_INDEX)
        conn.commit()
    except psycopg.Error as e:
        raise QueueError(f"Failed to create queue table: {e}") from e


def claim_batch(
    conn: psycopg.Connection[dict_row],
    batch_size: int,
) -> list[dict[str, object]]:
    """Claim a batch of pending jobs atomically.

    Uses SELECT FOR UPDATE SKIP LOCKED — safe for parallel workers.
    Returns list of job dicts with id, table_name, row_id, column_name, operation.
    """
    try:
        result = conn.execute(SQL_CLAIM_BATCH, {"batch_size": batch_size}).fetchall()
        conn.commit()
        return [dict(row) for row in result]
    except psycopg.Error as e:
        conn.rollback()
        raise QueueError(f"Failed to claim batch: {e}") from e


def complete_job(conn: psycopg.Connection[dict_row], job_id: int) -> None:
    """Delete a successfully completed job from the queue."""
    try:
        conn.execute(SQL_COMPLETE_JOB, {"job_id": job_id})
        conn.commit()
    except psycopg.Error as e:
        conn.rollback()
        raise QueueError(f"Failed to complete job {job_id}: {e}") from e


def fail_job(
    conn: psycopg.Connection[dict_row],
    job_id: int,
    error_msg: str,
) -> None:
    """Mark a job as failed with an error message."""
    try:
        conn.execute(SQL_FAIL_JOB, {"job_id": job_id, "error_msg": error_msg})
        conn.commit()
    except psycopg.Error as e:
        conn.rollback()
        raise QueueError(f"Failed to mark job {job_id} as failed: {e}") from e


def count_pending(
    conn: psycopg.Connection[dict_row],
    table_name: str,
) -> int:
    """Count pending jobs for a table."""
    result = conn.execute(SQL_COUNT_PENDING, {"table_name": table_name}).fetchone()
    if result is None:
        return 0
    return int(result["pending_count"])


def count_failed(
    conn: psycopg.Connection[dict_row],
    table_name: str,
) -> int:
    """Count failed jobs for a table."""
    result = conn.execute(SQL_COUNT_FAILED, {"table_name": table_name}).fetchone()
    if result is None:
        return 0
    return int(result["failed_count"])
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_queue.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add pgvector_setup/db/queue.py tests/unit/test_queue.py
git commit -m "feat: add queue module with atomic batch claiming (SKIP LOCKED)"
```

---

## Task 9: Vectors Module

**Files:**
- Create: `pgvector_setup/db/vectors.py`

**Step 1: Write vectors.py**

All SQL constants from CLAUDE.md plus the keyset pagination improvement. Functions: `fetch_unembedded_batch`, `update_embedding`, `count_embedded`, `search_similar`, `hybrid_search`.

```python
"""Vector operations — embedding storage, search, and bulk updates.

All SQL uses parameterized queries. Table and column names are validated
against information_schema before being formatted into SQL strings.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import psycopg
from psycopg.rows import dict_row

from pgvector_setup.db.introspect import get_column_names
from pgvector_setup.exceptions import InvalidFilterError, TableNotFoundError

logger = logging.getLogger(__name__)

# ── SQL Constants ─────────────────────────────────────────────────────────

# Fetch rows that need embedding — uses keyset pagination (WHERE id > last_id)
# for consistent performance on large tables (no OFFSET degradation)
# table_name and column validated before string formatting
SQL_FETCH_UNEMBEDDED = """
    SELECT id, {column}::TEXT AS content
    FROM {table}
    WHERE {column} IS NOT NULL
      AND embedding IS NULL
      AND id > %(last_id)s
    ORDER BY id
    LIMIT %(limit)s;
"""

# Update embedding for a single row
SQL_UPDATE_EMBEDDING = """
    UPDATE {table}
    SET embedding = %(embedding)s::vector
    WHERE id = %(row_id)s;
"""

# Semantic search — pure cosine similarity, no filters
SQL_SEMANTIC_SEARCH = """
    SELECT
        id,
        {column}::TEXT          AS content,
        1 - (embedding <=> %(query_vector)s::vector) AS similarity
    FROM {table}
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %(query_vector)s::vector
    LIMIT %(limit)s;
"""

# Count embedded rows (non-NULL embedding)
SQL_COUNT_EMBEDDED = """
    SELECT count(*) AS embedded_count
    FROM {table}
    WHERE embedding IS NOT NULL;
"""

# Count total rows with non-NULL source column
SQL_COUNT_TOTAL_WITH_CONTENT = """
    SELECT count(*) AS total_count
    FROM {table}
    WHERE {column} IS NOT NULL;
"""

# Fetch source text for a single row by ID (used by worker)
SQL_FETCH_ROW_TEXT = """
    SELECT {column}::TEXT AS content
    FROM {table}
    WHERE id = %(row_id)s;
"""

# Set embedding to NULL for a deleted row (used by worker on DELETE events)
SQL_NULL_EMBEDDING = """
    UPDATE {table}
    SET embedding = NULL
    WHERE id = %(row_id)s;
"""


def _validate_and_format(
    table: str,
    column: str,
    sql_template: str,
    schema: str = "public",
) -> str:
    """Format SQL template with quoted table/column names.

    Table and column names MUST be validated against information_schema
    before calling this function. This function only adds quoting.
    """
    qualified_table = f'"{schema}"."{table}"'
    quoted_column = f'"{column}"'
    return sql_template.format(table=qualified_table, column=quoted_column)


def fetch_unembedded_batch(
    conn: psycopg.Connection[dict_row],
    table: str,
    column: str,
    batch_size: int,
    last_id: int = 0,
    schema: str = "public",
) -> list[dict[str, Any]]:
    """Fetch a batch of rows that need embedding using keyset pagination.

    Returns list of dicts with 'id' and 'content' keys.
    """
    sql = _validate_and_format(table, column, SQL_FETCH_UNEMBEDDED, schema)
    result = conn.execute(sql, {"last_id": last_id, "limit": batch_size}).fetchall()
    return [dict(row) for row in result]


def update_embedding(
    conn: psycopg.Connection[dict_row],
    table: str,
    row_id: int,
    embedding: list[float],
    schema: str = "public",
) -> None:
    """Update the embedding column for a single row."""
    sql = _validate_and_format(table, "embedding", SQL_UPDATE_EMBEDDING, schema)
    conn.execute(sql, {"row_id": row_id, "embedding": embedding})


def bulk_update_embeddings(
    conn: psycopg.Connection[dict_row],
    table: str,
    rows: list[dict[str, Any]],
    embeddings: list[list[float]],
    schema: str = "public",
) -> None:
    """Update embeddings for multiple rows in a single transaction."""
    sql = _validate_and_format(table, "embedding", SQL_UPDATE_EMBEDDING, schema)
    for row, embedding in zip(rows, embeddings):
        conn.execute(sql, {"row_id": row["id"], "embedding": embedding})
    conn.commit()


def count_embedded(
    conn: psycopg.Connection[dict_row],
    table: str,
    schema: str = "public",
) -> int:
    """Count rows that have a non-NULL embedding."""
    sql = SQL_COUNT_EMBEDDED.format(table=f'"{schema}"."{table}"')
    result = conn.execute(sql).fetchone()
    if result is None:
        return 0
    return int(result["embedded_count"])


def count_total_with_content(
    conn: psycopg.Connection[dict_row],
    table: str,
    column: str,
    schema: str = "public",
) -> int:
    """Count rows that have non-NULL content in the source column."""
    sql = _validate_and_format(table, column, SQL_COUNT_TOTAL_WITH_CONTENT, schema)
    result = conn.execute(sql).fetchone()
    if result is None:
        return 0
    return int(result["total_count"])


def search_similar(
    conn: psycopg.Connection[dict_row],
    table: str,
    column: str,
    query_vector: list[float],
    limit: int = 10,
    schema: str = "public",
) -> list[dict[str, Any]]:
    """Semantic search — find rows by cosine similarity to query vector."""
    sql = _validate_and_format(table, column, SQL_SEMANTIC_SEARCH, schema)
    result = conn.execute(
        sql, {"query_vector": query_vector, "limit": limit}
    ).fetchall()
    return [dict(row) for row in result]


def hybrid_search(
    conn: psycopg.Connection[dict_row],
    table: str,
    column: str,
    query_vector: list[float],
    filters: dict[str, Any],
    limit: int = 10,
    pgvector_version: Optional[tuple[int, int, int]] = None,
    schema: str = "public",
) -> list[dict[str, Any]]:
    """Semantic search + SQL WHERE filters in a single query.

    Uses pre-filtering with hnsw.iterative_scan (pgvector >= 0.8.0) for
    correct recall on selective filters.

    Filter key conventions:
    - "column"      → WHERE column = value
    - "column_max"  → WHERE column <= value
    - "column_min"  → WHERE column >= value
    """
    # Enable iterative scan for pre-filtering (pgvector >= 0.8.0)
    if pgvector_version is not None and pgvector_version >= (0, 8, 0):
        conn.execute("SET hnsw.iterative_scan = relaxed_order")

    # Validate filter keys against actual table columns
    valid_columns = get_column_names(conn, table, schema)

    # Build parameterized WHERE clause
    where_clauses = ["embedding IS NOT NULL"]
    params: dict[str, Any] = {
        "query_vector": query_vector,
        "limit": limit,
    }

    for key, value in filters.items():
        if key.endswith("_max"):
            col = key[:-4]
            if col not in valid_columns:
                continue  # silently skip invalid filter columns
            where_clauses.append(f'"{col}" <= %({key})s')
        elif key.endswith("_min"):
            col = key[:-4]
            if col not in valid_columns:
                continue
            where_clauses.append(f'"{col}" >= %({key})s')
        else:
            if key not in valid_columns:
                continue
            where_clauses.append(f'"{key}" = %({key})s')
        params[key] = value

    where_sql = " AND ".join(where_clauses)
    qualified_table = f'"{schema}"."{table}"'
    quoted_column = f'"{column}"'

    sql = f"""
        SELECT
            *,
            1 - (embedding <=> %(query_vector)s::vector) AS similarity
        FROM {qualified_table}
        WHERE {where_sql}
        ORDER BY embedding <=> %(query_vector)s::vector
        LIMIT %(limit)s
    """

    result = conn.execute(sql, params).fetchall()
    return [dict(row) for row in result]


def fetch_row_text(
    conn: psycopg.Connection[dict_row],
    table: str,
    column: str,
    row_id: str,
    schema: str = "public",
) -> Optional[str]:
    """Fetch source text for a single row by ID. Used by worker."""
    sql = _validate_and_format(table, column, SQL_FETCH_ROW_TEXT, schema)
    result = conn.execute(sql, {"row_id": row_id}).fetchone()
    if result is None:
        return None
    return str(result["content"]) if result["content"] is not None else None


def null_embedding(
    conn: psycopg.Connection[dict_row],
    table: str,
    row_id: str,
    schema: str = "public",
) -> None:
    """Set embedding to NULL for a row (used on DELETE events)."""
    sql = SQL_NULL_EMBEDDING.format(table=f'"{schema}"."{table}"')
    conn.execute(sql, {"row_id": row_id})
    conn.commit()
```

**Step 2: Verify import**

Run: `python -c "from pgvector_setup.db.vectors import search_similar; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add pgvector_setup/db/vectors.py
git commit -m "feat: add vectors module with keyset pagination and hybrid search"
```

---

## Task 10: Apply Command

**Files:**
- Create: `pgvector_setup/commands/apply.py`
- Modify: `pgvector_setup/cli.py` (register apply command)

**Step 1: Write apply.py**

The 10-step apply process from CLAUDE.md: check pgvector extension → check existing column → preview SQL → add vector column → create HNSW index → create queue table → install trigger function → install trigger → save config → print summary.

```python
"""pgvector-setup apply — set up semantic search on a table/column.

This is the core setup command. It adds a vector column, HNSW index,
queue table, and trigger to enable semantic search on an existing table.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from pgvector_setup import __version__
from pgvector_setup.config import (
    CONFIG_FILE_NAME,
    DEFAULT_LOCAL_DIMENSIONS,
    DEFAULT_LOCAL_MODEL,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    OPENAI_DIMENSIONS,
    OPENAI_MODEL,
    ProjectConfig,
    TableConfig,
    load_project_config,
    load_settings,
    save_project_config,
)
from pgvector_setup.db.client import ensure_pgvector_extension, get_connection
from pgvector_setup.db.introspect import column_exists
from pgvector_setup.db.queue import create_queue_table
from pgvector_setup.exceptions import ExtensionNotFoundError

console = Console()

# ── SQL for apply steps (table/column names validated before formatting) ──

SQL_ADD_VECTOR_COLUMN = """ALTER TABLE "{schema}"."{table}" ADD COLUMN embedding vector({dimensions});"""

SQL_CREATE_HNSW_INDEX = """
CREATE INDEX CONCURRENTLY ON "{schema}"."{table}"
USING hnsw (embedding vector_cosine_ops)
WITH (m = {m}, ef_construction = {ef_construction});
"""

SQL_CREATE_TRIGGER_FUNCTION = """
CREATE OR REPLACE FUNCTION pgvector_setup_notify_fn()
RETURNS TRIGGER AS $$
DECLARE
    v_row_id TEXT;
BEGIN
    v_row_id := NEW.id::TEXT;

    IF TG_OP = 'DELETE' THEN
        INSERT INTO pgvector_setup_queue (table_name, row_id, column_name, operation)
        VALUES (TG_TABLE_NAME, OLD.id::TEXT, TG_ARGV[0], 'DELETE')
        ON CONFLICT (table_name, row_id, column_name)
        DO UPDATE SET
            status     = 'pending',
            operation  = 'DELETE',
            retries    = 0,
            error_msg  = NULL,
            updated_at = NOW();
        RETURN OLD;
    ELSE
        INSERT INTO pgvector_setup_queue (table_name, row_id, column_name, operation)
        VALUES (TG_TABLE_NAME, v_row_id, TG_ARGV[0], TG_OP)
        ON CONFLICT (table_name, row_id, column_name)
        DO UPDATE SET
            status     = 'pending',
            operation  = TG_OP,
            retries    = 0,
            error_msg  = NULL,
            updated_at = NOW();
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;
"""

SQL_CREATE_TRIGGER = """
CREATE OR REPLACE TRIGGER pgvector_setup_{table}_trigger
AFTER INSERT OR UPDATE OF "{column}" OR DELETE
ON "{schema}"."{table}"
FOR EACH ROW
EXECUTE FUNCTION pgvector_setup_notify_fn('{column}');
"""


def apply_command(
    table: str = typer.Option(..., "--table", "-t", help="Table name to set up"),
    column: str = typer.Option(..., "--column", "-c", help="Text column to embed"),
    model: str = typer.Option("local", "--model", "-m", help="Embedding model: 'local' or 'openai'"),
    db: Optional[str] = typer.Option(None, "--db", help="Database URL (overrides DATABASE_URL env var)"),
    schema: str = typer.Option("public", "--schema", help="Schema name"),
) -> None:
    """Set up semantic search on a table/column (extension + vector column + index + queue + trigger)."""
    settings = load_settings()
    database_url = db or settings.database_url

    if database_url is None:
        console.print(Panel(
            "[red]No database URL provided.[/red]\n\n"
            "Either pass --db or set DATABASE_URL in your .env file.",
            title="Missing Database URL",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    # Determine dimensions based on model
    if model == "openai":
        dimensions = OPENAI_DIMENSIONS
        model_name = OPENAI_MODEL
    else:
        dimensions = DEFAULT_LOCAL_DIMENSIONS
        model_name = DEFAULT_LOCAL_MODEL

    try:
        with get_connection(database_url) as conn:
            # Step 1: Check pgvector extension
            try:
                pgv_version = ensure_pgvector_extension(conn)
                version_str = ".".join(str(v) for v in pgv_version)
                console.print(f"[green]✓[/green] pgvector {version_str} detected")
            except ExtensionNotFoundError as e:
                console.print(Panel(str(e), title="Setup Required", border_style="red"))
                raise typer.Exit(code=1)

            # Step 2: Check if embedding column already exists
            if column_exists(conn, table, "embedding", schema):
                console.print(
                    f"[yellow]⚠[/yellow] Column 'embedding' already exists on {table}."
                )
                if not typer.confirm("Overwrite existing setup?", default=False):
                    raise typer.Exit(code=0)

            # Step 3: Preview SQL and ask for confirmation
            alter_sql = SQL_ADD_VECTOR_COLUMN.format(
                schema=schema, table=table, dimensions=dimensions
            )
            index_sql = SQL_CREATE_HNSW_INDEX.format(
                schema=schema, table=table,
                m=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION,
            )

            console.print()
            console.print("About to run:")
            console.print()
            console.print(Syntax(alter_sql.strip(), "sql", theme="monokai"))
            console.print(Syntax(index_sql.strip(), "sql", theme="monokai"))
            console.print()

            if not typer.confirm("Proceed?", default=False):
                console.print("Aborted.")
                raise typer.Exit(code=0)

            # Step 4: Add vector column
            if not column_exists(conn, table, "embedding", schema):
                conn.execute(alter_sql)
                conn.commit()
            console.print(
                f"[green]✓[/green] Column 'embedding vector({dimensions})' added to {table}"
            )

            # Step 5: Create HNSW index (CONCURRENTLY requires autocommit)
            conn.autocommit = True
            try:
                conn.execute(index_sql)
            finally:
                conn.autocommit = False
            console.print(
                f"[green]✓[/green] HNSW index created (m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION})"
            )

            # Step 6: Create queue table
            create_queue_table(conn)
            console.print("[green]✓[/green] Queue table pgvector_setup_queue created")

            # Step 7: Install trigger function (idempotent)
            conn.execute(SQL_CREATE_TRIGGER_FUNCTION)
            conn.commit()
            console.print("[green]✓[/green] Trigger function pgvector_setup_notify_fn installed")

            # Step 8: Install trigger on table
            trigger_sql = SQL_CREATE_TRIGGER.format(
                schema=schema, table=table, column=column
            )
            conn.execute(trigger_sql)
            conn.commit()
            console.print(
                f"[green]✓[/green] Trigger pgvector_setup_{table}_trigger installed"
            )

        # Step 9: Save config
        config = load_project_config() or ProjectConfig(version=__version__)
        table_config = TableConfig(
            table=table,
            schema=schema,
            column=column,
            embedding_column="embedding",
            model=model,
            model_name=model_name,
            dimensions=dimensions,
            hnsw_m=HNSW_M,
            hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
            applied_at=datetime.now(timezone.utc).isoformat(),
        )
        # Replace existing config for same table or add new
        config.tables = [t for t in config.tables if t.table != table]
        config.tables.append(table_config)
        save_project_config(config)
        console.print(f"[green]✓[/green] Config saved to {CONFIG_FILE_NAME}")

        # Step 10: Print next steps
        console.print()
        console.print("Next steps:")
        console.print(f"  [cyan]pgvector-setup index --table {table}[/cyan]   # embed your existing rows")
        console.print(f"  [cyan]pgvector-setup worker[/cyan]                   # keep new rows embedded automatically")
        console.print(f"  [cyan]pgvector-setup serve[/cyan]                    # start MCP server for Claude/Cursor")
        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        console.print(Panel(
            f"[red]Apply failed:[/red]\n\n{e}",
            title="Error",
            border_style="red",
        ))
        raise typer.Exit(code=1)
```

**Step 2: Register apply command in cli.py**

Add after the inspect import:
```python
from pgvector_setup.commands.apply import apply_command
app.command(name="apply")(apply_command)
```

**Step 3: Verify CLI**

Run: `pgvector-setup apply --help`
Expected: Shows help with --table, --column, --model, --db options

**Step 4: Commit**

```bash
git add pgvector_setup/commands/apply.py pgvector_setup/cli.py
git commit -m "feat: add apply command — sets up vector column, index, queue, trigger"
```

---

## Task 11: Index Command

**Files:**
- Create: `pgvector_setup/commands/index.py`
- Modify: `pgvector_setup/cli.py` (register index command)

**Step 1: Write index.py**

Bulk backfill with keyset pagination and Rich progress bar. Reads config from `.pgvector-setup.json` to determine model and column. Shows rows/min in progress bar.

```python
"""pgvector-setup index — bulk embed all existing rows where embedding IS NULL."""
from __future__ import annotations

import time
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from pgvector_setup.config import (
    DEFAULT_INDEX_BATCH_SIZE,
    load_project_config,
    load_settings,
)
from pgvector_setup.db.client import get_connection
from pgvector_setup.db.vectors import (
    bulk_update_embeddings,
    count_embedded,
    count_total_with_content,
    fetch_unembedded_batch,
)
from pgvector_setup.embeddings import get_provider
from pgvector_setup.exceptions import PgvectorSetupError

console = Console()


def index_command(
    table: str = typer.Option(..., "--table", "-t", help="Table to index"),
    batch_size: int = typer.Option(
        DEFAULT_INDEX_BATCH_SIZE, "--batch-size", "-b",
        help="Number of rows to embed per batch",
    ),
    db: Optional[str] = typer.Option(None, "--db", help="Database URL (overrides DATABASE_URL)"),
) -> None:
    """Bulk embed all existing rows where embedding IS NULL."""
    settings = load_settings()
    database_url = db or settings.database_url

    if database_url is None:
        console.print("[red]No database URL. Pass --db or set DATABASE_URL.[/red]")
        raise typer.Exit(code=1)

    config = load_project_config()
    if config is None:
        console.print("[red]No .pgvector-setup.json found. Run 'pgvector-setup apply' first.[/red]")
        raise typer.Exit(code=1)

    table_config = config.get_table_config(table)
    if table_config is None:
        console.print(f"[red]Table '{table}' not found in config. Run 'pgvector-setup apply --table {table}' first.[/red]")
        raise typer.Exit(code=1)

    try:
        # Initialize embedding provider
        provider = get_provider(
            table_config.model,
            api_key=settings.openai_api_key,
        )
        console.print(f"Using model: [cyan]{provider.config.model_name}[/cyan] ({provider.config.dimensions}d)")

        with get_connection(database_url) as conn:
            # Count total and already embedded
            total = count_total_with_content(conn, table, table_config.column, table_config.schema)
            already_embedded = count_embedded(conn, table, table_config.schema)
            remaining = total - already_embedded

            if remaining == 0:
                console.print(f"[green]✓[/green] All {total} rows already embedded.")
                return

            console.print(f"Indexing {remaining} rows ({already_embedded}/{total} already done)")

            start_time = time.time()
            processed = 0
            last_id = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Indexing {table}.{table_config.column}",
                    total=remaining,
                )

                while True:
                    batch = fetch_unembedded_batch(
                        conn, table, table_config.column,
                        batch_size=batch_size,
                        last_id=last_id,
                        schema=table_config.schema,
                    )
                    if not batch:
                        break

                    texts = [row["content"] for row in batch]
                    embeddings = provider.embed(texts)
                    bulk_update_embeddings(conn, table, batch, embeddings, table_config.schema)

                    last_id = batch[-1]["id"]
                    processed += len(batch)
                    progress.update(task, advance=len(batch))

            elapsed = time.time() - start_time
            rate = processed / (elapsed / 60) if elapsed > 0 else 0

            console.print()
            console.print(f"[green]✓[/green] Indexed {processed} rows in {elapsed:.1f}s ({rate:.0f} rows/min)")

    except PgvectorSetupError as e:
        console.print(f"[red]Index failed:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise typer.Exit(code=1)
```

**Step 2: Register in cli.py**

Add:
```python
from pgvector_setup.commands.index import index_command
app.command(name="index")(index_command)
```

**Step 3: Commit**

```bash
git add pgvector_setup/commands/index.py pgvector_setup/cli.py
git commit -m "feat: add index command — bulk backfill with progress bar"
```

---

## Task 12: Worker Daemon

**Files:**
- Create: `pgvector_setup/worker/__init__.py`
- Create: `pgvector_setup/worker/daemon.py`
- Create: `pgvector_setup/commands/worker.py`
- Modify: `pgvector_setup/cli.py` (register worker command)

**Step 1: Write worker/daemon.py**

Async poll loop with exponential backoff, graceful shutdown, heartbeat logging. This is the only async module in the project.

The daemon:
1. Claims a batch of jobs from the queue using `SELECT FOR UPDATE SKIP LOCKED`
2. Groups jobs by table for efficient batch embedding
3. Fetches source text, embeds, updates embedding column
4. Deletes completed jobs, marks failed ones
5. Logs heartbeat every 60s when idle
6. Handles `SIGINT`/`SIGTERM` for graceful shutdown

**Step 2: Write commands/worker.py**

Thin wrapper that calls `asyncio.run(run_worker(settings))`.

**Step 3: Register in cli.py**

**Step 4: Commit**

```bash
git add pgvector_setup/worker/ pgvector_setup/commands/worker.py pgvector_setup/cli.py
git commit -m "feat: add worker daemon with async poll loop and graceful shutdown"
```

---

## Task 13: MCP Server

**Files:**
- Create: `pgvector_setup/mcp_server/__init__.py`
- Create: `pgvector_setup/mcp_server/server.py`
- Create: `pgvector_setup/commands/serve.py`
- Modify: `pgvector_setup/cli.py` (register serve command)

**Step 1: Write mcp_server/server.py**

FastMCP server with three tools: `semantic_search`, `hybrid_search`, `get_embedding_status`. Each tool is a thin wrapper around functions in `db/vectors.py` and `db/queue.py`.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="pgvector-setup", version="0.1.0")

@mcp.tool()
def semantic_search(query: str, table: str, limit: int = 10) -> list[dict]:
    # Load config, get provider, embed query, call search_similar
    ...

@mcp.tool()
def hybrid_search(query: str, table: str, filters: dict = {}, limit: int = 10) -> list[dict]:
    # Same as semantic_search but calls hybrid_search with filters
    ...

@mcp.tool()
def get_embedding_status(table: str) -> dict:
    # Count total rows, embedded rows, pending queue, failed queue
    ...
```

**Step 2: Write commands/serve.py**

Starts the MCP server with the configured transport (stdio or http).

**Step 3: Register in cli.py**

**Step 4: Commit**

```bash
git add pgvector_setup/mcp_server/ pgvector_setup/commands/serve.py pgvector_setup/cli.py
git commit -m "feat: add MCP server with semantic_search, hybrid_search, get_embedding_status"
```

---

## Task 14: Status Command

**Files:**
- Create: `pgvector_setup/commands/status.py`
- Modify: `pgvector_setup/cli.py` (register status command)

**Step 1: Write status.py**

Rich table showing each watched table's embedding coverage, queue depth, model info. Data from pg_class estimates + exact counts + queue table.

**Step 2: Register in cli.py**

**Step 3: Commit**

```bash
git add pgvector_setup/commands/status.py pgvector_setup/cli.py
git commit -m "feat: add status command — embedding health dashboard"
```

---

## Task 15: Integrate Command

**Files:**
- Create: `pgvector_setup/commands/integrate.py`
- Modify: `pgvector_setup/cli.py` (register integrate command)

**Step 1: Write integrate.py**

Reads `.pgvector-setup.json` and generates Claude Desktop config snippet. Prints the JSON and file location per platform.

**Step 2: Register in cli.py**

**Step 3: Commit**

```bash
git add pgvector_setup/commands/integrate.py pgvector_setup/cli.py
git commit -m "feat: add integrate command — generates Claude Desktop config"
```

---

## Task 16: Unit Tests (Complete Suite)

**Files:**
- Verify/update: `tests/unit/test_config.py`
- Verify/update: `tests/unit/test_embeddings.py`
- Verify/update: `tests/unit/test_introspect.py`
- Verify/update: `tests/unit/test_queue.py`

**Step 1: Run all unit tests**

Run: `pytest tests/unit/ -v`
Expected: All pass

**Step 2: Run mypy**

Run: `mypy pgvector_setup/ --strict`
Fix any errors.

**Step 3: Run ruff**

Run: `ruff check . && ruff check . --fix`
Fix any issues.

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: resolve mypy and ruff issues across all modules"
```

---

## Task 17: Integration Test Fixtures

**Files:**
- Create: `tests/fixtures/products.sql`
- Create: `tests/conftest.py`

**Step 1: Write products.sql**

500 product rows with realistic names, descriptions (varied lengths), categories, and prices. This is the primary test fixture.

**Step 2: Write conftest.py**

Pytest fixtures for:
- `db_url`: connection string for test database
- `db_conn`: psycopg connection with pgvector registered, auto-cleanup
- `setup_products_table`: creates products table and loads fixture data
- Skip integration tests if Docker Postgres is not available

**Step 3: Commit**

```bash
git add tests/
git commit -m "feat: add test fixtures and conftest for integration tests"
```

---

## Task 18: Integration Tests

**Files:**
- Create: `tests/integration/test_apply.py`
- Create: `tests/integration/test_index.py`
- Create: `tests/integration/test_search.py`
- Create: `tests/integration/test_worker.py`

**Step 1: Write test_apply.py**

Test the full apply flow against real Postgres: creates vector column, HNSW index, queue table, trigger. Verify with information_schema queries.

**Step 2: Write test_index.py**

Test bulk indexing: apply → index → verify embeddings are non-NULL.

**Step 3: Write test_search.py**

Test semantic_search and hybrid_search return ranked results after indexing.

**Step 4: Write test_worker.py**

Test: INSERT a row → worker processes → embedding appears.

**Step 5: Run integration tests**

Run: `docker-compose up -d && pytest tests/integration/ -m integration -v`
Expected: All pass

**Step 6: Commit**

```bash
git add tests/integration/
git commit -m "feat: add integration tests for apply, index, search, worker"
```

---

## Task 19: Final Verification

**Step 1: Run full test suite**

Run: `pytest -v`
Expected: All unit and integration tests pass

**Step 2: Run mypy strict**

Run: `mypy pgvector_setup/ --strict`
Expected: Zero errors

**Step 3: Run ruff**

Run: `ruff check .`
Expected: Zero errors

**Step 4: End-to-end smoke test**

```bash
docker-compose up -d
pgvector-setup inspect postgresql://postgres:password@localhost:5432/pgvector_dev
pgvector-setup apply --table products --column description
pgvector-setup index --table products
pgvector-setup status
pgvector-setup serve  # verify it starts, Ctrl+C to stop
pgvector-setup integrate claude
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: pgvector-setup MVP complete — Milestones 1-3"
```

"""Configuration for pgsemantic.

Loads settings from environment variables (via python-dotenv) and manages
the .pgsemantic.json project config file.

All magic numbers live here as named constants. Never hardcode them inline.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from pgsemantic.exceptions import ConfigError

logger = logging.getLogger(__name__)

# ── HNSW index parameters ─────────────────────────────────────────────────
HNSW_M: int = 16
HNSW_EF_CONSTRUCTION: int = 64
HNSW_EF_SEARCH: int = 40

# ── Default embedding models ──────────────────────────────────────────────
DEFAULT_LOCAL_MODEL: str = "all-MiniLM-L6-v2"
DEFAULT_LOCAL_DIMENSIONS: int = 384
OPENAI_MODEL: str = "text-embedding-3-small"
OPENAI_DIMENSIONS: int = 1536
OLLAMA_MODEL: str = "nomic-embed-text"
OLLAMA_DIMENSIONS: int = 768
OLLAMA_BASE_URL: str = "http://localhost:11434"

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
CONFIG_FILE_NAME: str = ".pgsemantic.json"


@dataclass
class Settings:
    """Runtime settings loaded from environment variables."""
    database_url: str | None = None
    embedding_provider: str = "local"
    openai_api_key: str | None = None
    ollama_base_url: str = OLLAMA_BASE_URL
    mcp_transport: str = "stdio"
    mcp_port: int = 3000
    mcp_auth_token: str | None = None
    worker_batch_size: int = DEFAULT_WORKER_BATCH_SIZE
    worker_poll_interval_ms: int = DEFAULT_WORKER_POLL_INTERVAL_MS
    worker_max_retries: int = DEFAULT_WORKER_MAX_RETRIES
    log_level: str = "info"


def load_settings() -> Settings:
    """Load settings from .env file and environment variables."""
    load_dotenv()
    return Settings(
        database_url=os.environ.get("DATABASE_URL"),
        embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "local"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
        mcp_transport=os.environ.get("MCP_TRANSPORT", "stdio"),
        mcp_port=int(os.environ.get("MCP_PORT", "3000")),
        mcp_auth_token=os.environ.get("MCP_AUTH_TOKEN"),
        worker_batch_size=int(os.environ.get("WORKER_BATCH_SIZE", str(DEFAULT_WORKER_BATCH_SIZE))),
        worker_poll_interval_ms=int(os.environ.get("WORKER_POLL_INTERVAL_MS", str(DEFAULT_WORKER_POLL_INTERVAL_MS))),
        worker_max_retries=int(os.environ.get("WORKER_MAX_RETRIES", str(DEFAULT_WORKER_MAX_RETRIES))),
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
    """Persistent project config stored in .pgsemantic.json."""
    version: str
    tables: list[TableConfig] = field(default_factory=list)

    def get_table_config(self, table_name: str) -> TableConfig | None:
        """Get config for a specific table, or None if not found."""
        for tc in self.tables:
            if tc.table == table_name:
                return tc
        return None


def save_project_config(config: ProjectConfig, path: Path | None = None) -> None:
    """Write project config to .pgsemantic.json."""
    if path is None:
        path = Path.cwd() / CONFIG_FILE_NAME
    try:
        data = asdict(config)
        path.write_text(json.dumps(data, indent=2) + "\n")
    except OSError as e:
        raise ConfigError(f"Failed to write config to {path}: {e}") from e


def load_project_config(path: Path | None = None) -> ProjectConfig | None:
    """Read project config from .pgsemantic.json. Returns None if file doesn't exist."""
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

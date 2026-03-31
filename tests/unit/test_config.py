"""Tests for config module — Settings loading and ProjectConfig read/write."""
import os
from pathlib import Path
from unittest.mock import patch

from pgsemantic.config import (
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
    SHADOW_TABLE_PREFIX,
    ProjectConfig,
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

    @patch("pgsemantic.config.load_dotenv")
    def test_load_settings_missing_database_url(self, mock_dotenv) -> None:
        """When DATABASE_URL is not in env (and dotenv loading is skipped), it should be None."""
        env_without_db = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
        with patch.dict(os.environ, env_without_db, clear=True):
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
        assert loaded is not None
        assert loaded.version == "0.1.0"
        assert len(loaded.tables) == 1
        assert loaded.tables[0].table == "products"
        assert loaded.tables[0].dimensions == DEFAULT_LOCAL_DIMENSIONS

    def test_load_missing_config_returns_none(self, tmp_path: Path) -> None:
        config_path = tmp_path / CONFIG_FILE_NAME
        result = load_project_config(config_path)
        assert result is None

    def test_get_table_config(self) -> None:
        config = ProjectConfig(
            version="0.1.0",
            tables=[
                TableConfig(
                    table="products", schema="public", column="description",
                    embedding_column="embedding", model="local",
                    model_name=DEFAULT_LOCAL_MODEL, dimensions=DEFAULT_LOCAL_DIMENSIONS,
                    hnsw_m=HNSW_M, hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
                    applied_at="2026-02-27T10:00:00Z",
                )
            ],
        )
        tc = config.get_table_config("products")
        assert tc is not None
        assert tc.column == "description"
        tc2 = config.get_table_config("nonexistent")
        assert tc2 is None


class TestTableConfigNewFields:
    """Tests for columns, storage_mode, shadow_table fields and backward compat."""

    def test_defaults(self) -> None:
        tc = TableConfig(
            table="products", schema="public", column="description",
            embedding_column="embedding", model="local",
            model_name=DEFAULT_LOCAL_MODEL, dimensions=DEFAULT_LOCAL_DIMENSIONS,
            hnsw_m=HNSW_M, hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
            applied_at="2026-02-27T10:00:00Z",
        )
        assert tc.columns is None
        assert tc.storage_mode == "inline"
        assert tc.shadow_table is None
        assert tc.source_columns == ["description"]

    def test_multi_column(self) -> None:
        tc = TableConfig(
            table="products", schema="public", column="title",
            embedding_column="embedding", model="local",
            model_name=DEFAULT_LOCAL_MODEL, dimensions=DEFAULT_LOCAL_DIMENSIONS,
            hnsw_m=HNSW_M, hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
            applied_at="2026-02-27T10:00:00Z",
            columns=["title", "description"],
        )
        assert tc.source_columns == ["title", "description"]

    def test_external_mode(self) -> None:
        tc = TableConfig(
            table="products", schema="public", column="description",
            embedding_column="embedding", model="local",
            model_name=DEFAULT_LOCAL_MODEL, dimensions=DEFAULT_LOCAL_DIMENSIONS,
            hnsw_m=HNSW_M, hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
            applied_at="2026-02-27T10:00:00Z",
            storage_mode="external",
            shadow_table="pgsemantic_embeddings_products",
        )
        assert tc.storage_mode == "external"
        assert tc.shadow_table == "pgsemantic_embeddings_products"

    def test_backward_compat_load(self, tmp_path: Path) -> None:
        """Old config files without new fields should load fine."""
        import json
        old_config = {
            "version": "0.1.0",
            "tables": [{
                "table": "products",
                "schema": "public",
                "column": "description",
                "embedding_column": "embedding",
                "model": "local",
                "model_name": "all-MiniLM-L6-v2",
                "dimensions": 384,
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "applied_at": "2026-02-27T10:00:00Z",
                "primary_key": ["id"],
            }],
        }
        config_path = tmp_path / CONFIG_FILE_NAME
        config_path.write_text(json.dumps(old_config))
        loaded = load_project_config(config_path)
        assert loaded is not None
        tc = loaded.tables[0]
        assert tc.columns is None
        assert tc.storage_mode == "inline"
        assert tc.shadow_table is None
        assert tc.source_columns == ["description"]

    def test_roundtrip_with_new_fields(self, tmp_path: Path) -> None:
        """Config with new fields should save and load correctly."""
        config = ProjectConfig(
            version="0.1.0",
            tables=[
                TableConfig(
                    table="products", schema="public", column="title",
                    embedding_column="embedding", model="local",
                    model_name=DEFAULT_LOCAL_MODEL, dimensions=DEFAULT_LOCAL_DIMENSIONS,
                    hnsw_m=HNSW_M, hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
                    applied_at="2026-02-27T10:00:00Z",
                    columns=["title", "description"],
                    storage_mode="external",
                    shadow_table="pgsemantic_embeddings_products",
                )
            ],
        )
        config_path = tmp_path / CONFIG_FILE_NAME
        save_project_config(config, config_path)
        loaded = load_project_config(config_path)
        assert loaded is not None
        tc = loaded.tables[0]
        assert tc.columns == ["title", "description"]
        assert tc.storage_mode == "external"
        assert tc.shadow_table == "pgsemantic_embeddings_products"
        assert tc.source_columns == ["title", "description"]


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

    def test_shadow_table_prefix(self) -> None:
        assert SHADOW_TABLE_PREFIX == "pgsemantic_embeddings_"

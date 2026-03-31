"""Tests for embedding visualization endpoint."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from pgsemantic.config import (
    DEFAULT_LOCAL_DIMENSIONS,
    DEFAULT_LOCAL_MODEL,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    ProjectConfig,
    TableConfig,
)


def _make_config() -> ProjectConfig:
    return ProjectConfig(
        version="0.3.0",
        tables=[
            TableConfig(
                table="articles",
                schema="public",
                column="body",
                embedding_column="embedding",
                model="local",
                model_name=DEFAULT_LOCAL_MODEL,
                dimensions=DEFAULT_LOCAL_DIMENSIONS,
                hnsw_m=HNSW_M,
                hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
                applied_at="2026-03-30T00:00:00Z",
            ),
        ],
    )


class TestVisualizeEndpoint:
    @patch("pgsemantic.web.app.get_connection")
    @patch("pgsemantic.web.app.load_project_config")
    @patch("pgsemantic.web.app.load_settings")
    def test_returns_points(self, mock_settings, mock_config, mock_conn_ctx):
        import numpy as np
        from fastapi.testclient import TestClient

        from pgsemantic.web.app import app

        mock_settings.return_value = MagicMock(
            database_url="postgresql://test@localhost/db"
        )
        mock_config.return_value = _make_config()

        mock_conn = MagicMock()
        mock_conn_ctx.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=False)

        # Create 5 fake embeddings with distinct directions
        rng = np.random.default_rng(42)
        fake_rows = []
        for i in range(5):
            emb = rng.standard_normal(DEFAULT_LOCAL_DIMENSIONS).tolist()
            fake_rows.append(
                {
                    "row_id": str(i),
                    "embedding": emb,
                    "label": f"Article {i}",
                }
            )

        mock_result = MagicMock()
        mock_result.fetchall.return_value = fake_rows
        mock_conn.execute.return_value = mock_result

        client = TestClient(app)
        resp = client.get("/api/visualize?table=articles&sample=5")

        assert resp.status_code == 200
        data = resp.json()
        assert "points" in data
        assert len(data["points"]) == 5
        assert "x" in data["points"][0]
        assert "y" in data["points"][0]
        assert 0 <= data["points"][0]["x"] <= 1
        assert 0 <= data["points"][0]["y"] <= 1

    @patch("pgsemantic.web.app.load_project_config")
    @patch("pgsemantic.web.app.load_settings")
    def test_no_config_returns_error(self, mock_settings, mock_config):
        from fastapi.testclient import TestClient

        from pgsemantic.web.app import app

        mock_settings.return_value = MagicMock(
            database_url="postgresql://test@localhost/db"
        )
        mock_config.return_value = None

        client = TestClient(app)
        resp = client.get("/api/visualize?table=articles")
        assert resp.status_code == 400

    @patch("pgsemantic.web.app.get_connection")
    @patch("pgsemantic.web.app.load_project_config")
    @patch("pgsemantic.web.app.load_settings")
    def test_empty_embeddings_returns_empty(
        self, mock_settings, mock_config, mock_conn_ctx
    ):
        from fastapi.testclient import TestClient

        from pgsemantic.web.app import app

        mock_settings.return_value = MagicMock(
            database_url="postgresql://test@localhost/db"
        )
        mock_config.return_value = _make_config()

        mock_conn = MagicMock()
        mock_conn_ctx.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=False)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_result

        client = TestClient(app)
        resp = client.get("/api/visualize?table=articles&sample=100")

        assert resp.status_code == 200
        data = resp.json()
        assert data["points"] == []

    @patch("pgsemantic.web.app.load_project_config")
    @patch("pgsemantic.web.app.load_settings")
    def test_unconfigured_table_returns_error(self, mock_settings, mock_config):
        from fastapi.testclient import TestClient

        from pgsemantic.web.app import app

        mock_settings.return_value = MagicMock(
            database_url="postgresql://test@localhost/db"
        )
        mock_config.return_value = _make_config()

        client = TestClient(app)
        resp = client.get("/api/visualize?table=nonexistent")
        assert resp.status_code == 400
        assert "not configured" in resp.json()["detail"]

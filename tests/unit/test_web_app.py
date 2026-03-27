"""Tests for FastAPI web UI endpoints — all DB interactions mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from pgsemantic.web.app import app


@pytest.fixture()
def client():
    """Create a test client."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def csrf_token(client):
    """Get a valid CSRF token."""
    resp = client.get("/api/csrf-token")
    assert resp.status_code == 200
    return resp.json()["token"]


class TestCSRF:
    def test_get_csrf_token(self, client) -> None:
        resp = client.get("/api/csrf-token")
        assert resp.status_code == 200
        assert "token" in resp.json()
        assert len(resp.json()["token"]) == 64

    def test_post_without_csrf_returns_403(self, client) -> None:
        resp = client.post("/api/retry", json={"table": "t"})
        assert resp.status_code == 403

    def test_post_with_wrong_csrf_returns_403(self, client) -> None:
        resp = client.post(
            "/api/retry",
            json={"table": "t"},
            headers={"X-CSRF-Token": "wrong"},
        )
        assert resp.status_code == 403

    def test_post_with_valid_csrf_passes(self, client, csrf_token) -> None:
        """CSRF check passes (endpoint may fail for other reasons, but not 403)."""
        with patch("pgsemantic.web.app.load_settings") as mock:
            mock.return_value = MagicMock(database_url=None)
            resp = client.post(
                "/api/retry",
                json={"table": "t"},
                headers={"X-CSRF-Token": csrf_token},
            )
        # Should get past CSRF (not 403) — may be 400 due to no DB
        assert resp.status_code != 403


class TestSecurityHeaders:
    def test_security_headers_present(self, client) -> None:
        resp = client.get("/api/csrf-token")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert "Content-Security-Policy" in resp.headers


class TestConnection:
    @patch("pgsemantic.web.app.get_connection")
    @patch("pgsemantic.web.app.get_pgvector_version")
    @patch("pgsemantic.web.app.load_settings")
    def test_connection_info_success(self, mock_settings, mock_pgv, mock_conn, client) -> None:
        mock_settings.return_value = MagicMock(
            database_url="postgresql://user:secret@localhost/testdb",
        )
        mock_pgv.return_value = (0, 7, 0)
        mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        resp = client.get("/api/connection")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is True
        # Password should be masked
        assert "secret" not in data.get("database_url", "")

    @patch("pgsemantic.web.app.load_settings")
    def test_connection_info_no_db(self, mock_settings, client) -> None:
        mock_settings.return_value = MagicMock(database_url=None)
        resp = client.get("/api/connection")
        assert resp.status_code == 400


class TestStatus:
    @patch("pgsemantic.web.app.load_project_config")
    @patch("pgsemantic.web.app.load_settings")
    def test_status_no_config(self, mock_settings, mock_config, client) -> None:
        mock_settings.return_value = MagicMock(database_url="postgresql://u:p@localhost/db")
        mock_config.return_value = None
        resp = client.get("/api/status")
        assert resp.status_code == 200
        assert resp.json()["tables"] == []

    @patch("pgsemantic.web.app.count_failed")
    @patch("pgsemantic.web.app.count_pending")
    @patch("pgsemantic.web.app.count_total_with_content")
    @patch("pgsemantic.web.app.count_embedded")
    @patch("pgsemantic.web.app.get_connection")
    @patch("pgsemantic.web.app.load_settings")
    @patch("pgsemantic.web.app.load_project_config")
    def test_status_with_table(
        self, mock_config, mock_settings, mock_conn,
        mock_embedded, mock_total, mock_pending, mock_failed, client,
    ) -> None:
        from pgsemantic.config import ProjectConfig, TableConfig
        tc = TableConfig(
            table="news", schema="public", column="body",
            embedding_column="embedding", model="local",
            model_name="all-MiniLM-L6-v2", dimensions=384,
            hnsw_m=16, hnsw_ef_construction=64,
            applied_at="2026-03-26T00:00:00Z",
        )
        mock_config.return_value = ProjectConfig(version="0.3.0", tables=[tc])
        mock_settings.return_value = MagicMock(database_url="postgresql://u:p@localhost/db")
        mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_embedded.return_value = 100
        mock_total.return_value = 200
        mock_pending.return_value = 5
        mock_failed.return_value = 2
        resp = client.get("/api/status")
        assert resp.status_code == 200
        tables = resp.json()["tables"]
        assert len(tables) == 1
        assert tables[0]["table"] == "news"
        assert tables[0]["coverage_pct"] == 50.0
        assert tables[0]["failed"] == 2


class TestRetry:
    @patch("pgsemantic.db.queue.retry_failed_jobs")
    @patch("pgsemantic.web.app.get_connection")
    @patch("pgsemantic.web.app.load_settings")
    def test_retry_success(self, mock_settings, mock_conn, mock_retry, client, csrf_token) -> None:
        mock_settings.return_value = MagicMock(database_url="postgresql://u:p@localhost/db")
        mock_inner = MagicMock()
        mock_conn.return_value.__enter__ = MagicMock(return_value=mock_inner)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_retry.return_value = 3
        resp = client.post(
            "/api/retry",
            json={"table": "news"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        assert resp.json()["retried"] == 3

    def test_retry_missing_table_and_all(self, client, csrf_token) -> None:
        with patch("pgsemantic.web.app.load_settings") as mock:
            mock.return_value = MagicMock(database_url="postgresql://u:p@localhost/db")
            resp = client.post(
                "/api/retry",
                json={},
                headers={"X-CSRF-Token": csrf_token},
            )
        assert resp.status_code == 400


class TestWorkerHealth:
    @patch("pgsemantic.db.queue.get_worker_health")
    @patch("pgsemantic.web.app.get_connection")
    @patch("pgsemantic.web.app.load_settings")
    def test_worker_health_empty(self, mock_settings, mock_conn, mock_health, client) -> None:
        mock_settings.return_value = MagicMock(database_url="postgresql://u:p@localhost/db")
        mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_health.return_value = []
        resp = client.get("/api/worker-health")
        assert resp.status_code == 200
        assert resp.json()["workers"] == []

    @patch("pgsemantic.db.queue.get_worker_health")
    @patch("pgsemantic.web.app.get_connection")
    @patch("pgsemantic.web.app.load_settings")
    def test_worker_health_with_worker(self, mock_settings, mock_conn, mock_health, client) -> None:
        from datetime import datetime, timezone
        mock_settings.return_value = MagicMock(database_url="postgresql://u:p@localhost/db")
        mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        now = datetime.now(tz=timezone.utc)
        mock_health.return_value = [
            {"worker_id": "mac-1234", "last_heartbeat": now,
             "jobs_processed": 42, "started_at": now},
        ]
        resp = client.get("/api/worker-health")
        assert resp.status_code == 200
        workers = resp.json()["workers"]
        assert len(workers) == 1
        assert workers[0]["worker_id"] == "mac-1234"
        assert workers[0]["status"] == "running"
        assert workers[0]["jobs_processed"] == 42


class TestSearch:
    @patch("pgsemantic.web.app.search_similar")
    @patch("pgsemantic.web.app.get_provider")
    @patch("pgsemantic.web.app.get_connection")
    @patch("pgsemantic.web.app.load_settings")
    @patch("pgsemantic.web.app.load_project_config")
    def test_search_success(
        self, mock_config, mock_settings, mock_conn, mock_provider, mock_search,
        client, csrf_token,
    ) -> None:
        from pgsemantic.config import ProjectConfig, TableConfig
        tc = TableConfig(
            table="news", schema="public", column="body",
            embedding_column="embedding", model="local",
            model_name="all-MiniLM-L6-v2", dimensions=384,
            hnsw_m=16, hnsw_ef_construction=64,
            applied_at="2026-03-26T00:00:00Z",
        )
        mock_config.return_value = ProjectConfig(version="0.3.0", tables=[tc])
        mock_settings.return_value = MagicMock(database_url="postgresql://u:p@localhost/db")
        mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_prov_instance = MagicMock()
        mock_prov_instance.embed_query.return_value = [0.1] * 384
        mock_provider.return_value = mock_prov_instance
        mock_search.return_value = [
            {"body": "test content", "similarity": 0.85, "id": 1},
        ]
        resp = client.post(
            "/api/search",
            json={"query": "test", "table": "news", "limit": 5},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test"
        assert len(data["results"]) == 1

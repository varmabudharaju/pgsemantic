# High-Impact Improvements: Retry, Worker Health, Web UI Tests

**Date:** 2026-03-26
**Status:** Approved

## Overview

Three improvements to pgsemantic targeting operational reliability and test coverage:
1. Retry failed embedding jobs (CLI + Web UI)
2. Worker health visibility in status dashboard
3. Web UI endpoint test coverage

---

## Feature 1: Retry Failed Jobs

### Problem
Failed jobs in `pgvector_setup_queue` have no recovery path. Users must manually UPDATE the database.

### Design

**New function in `pgsemantic/db/queue.py`:**
```python
def retry_failed_jobs(conn, table_name: str | None = None) -> int:
    """Reset failed jobs to pending. Returns count of jobs retried."""
```
- `table_name=None` retries all tables
- Sets `status='pending', retries=0, error_msg=NULL, updated_at=NOW()`
- Returns count of rows updated

**New CLI command `pgsemantic/commands/retry.py`:**
- `pgsemantic retry --table T` — retry failed jobs for one table
- `pgsemantic retry --all` — retry all failed jobs across all tables
- Shows: "Retried 12 failed jobs for news_articles"
- Shows: "No failed jobs found" when count is 0

**Register in `cli.py`** as a new command.

**New web endpoint in `pgsemantic/web/app.py`:**
- `POST /api/retry` with body `{"table": "news_articles"}`
- Returns `{"retried": 12}`
- CSRF protected (already handled by middleware)

**UI change in `pgsemantic/web/static/index.html`:**
- "Retry Failed" button on Status page, next to each table with failed > 0
- Button calls `POST /api/retry`, refreshes status on success
- Button hidden when failed count is 0

---

## Feature 2: Worker Health

### Problem
`pgsemantic status` and the web UI show queue counts but not whether the worker is actually running.

### Design

**New table `pgvector_setup_worker_health`:**
```sql
CREATE TABLE IF NOT EXISTS pgvector_setup_worker_health (
    worker_id       TEXT PRIMARY KEY,
    last_heartbeat  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    jobs_processed  INT NOT NULL DEFAULT 0,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**New functions in `pgsemantic/db/queue.py`:**
```python
def create_worker_health_table(conn) -> None
def upsert_worker_heartbeat(conn, worker_id: str, jobs_processed: int) -> None
def get_worker_health(conn) -> list[dict]  # all workers
def delete_worker_health(conn, worker_id: str) -> None
```

**Worker daemon changes (`pgsemantic/worker/daemon.py`):**
- Generate a `worker_id` on startup (hostname + PID, e.g. `macbook-12345`)
- Create health table alongside queue table at startup
- Upsert heartbeat every poll cycle (piggyback on existing loop — no extra thread)
- Track cumulative `jobs_processed` counter
- Delete health row on graceful shutdown (SIGINT/SIGTERM)

**Status command changes (`pgsemantic/commands/status.py`):**
- Query `pgvector_setup_worker_health` table
- Show per-worker: status (running/stale/stopped), last seen, jobs processed, uptime
- "Running" if last_heartbeat < 30s ago, "Stale" if 30s-5min, "Not running" if >5min or no row
- Gracefully handle missing health table (worker never started)

**Web UI changes:**
- Worker health section on Status/Dashboard page
- Show same info: status badge (green/yellow/red), last seen, jobs processed

**Web endpoint:**
- `GET /api/worker-health` returns worker health rows

---

## Feature 3: Web UI Tests

### Problem
Zero test coverage on FastAPI endpoints. The web UI is the primary user interface.

### Design

**New file: `tests/unit/test_web_app.py`**

**Framework:** pytest + `httpx.AsyncClient` via `fastapi.testclient.TestClient`

**Mocking strategy:** Patch `pgsemantic.db.client.get_connection` and relevant DB functions at the module level. No real database needed.

**Test coverage targets:**
1. CSRF token flow — GET token, POST with token, POST without token (403)
2. `GET /api/connection` — returns masked URL + pgvector version
3. `POST /api/connection/test` — success and failure cases
4. `GET /api/database/tables` — returns table list
5. `GET /api/inspect` — returns scored columns
6. `POST /api/apply` — success case
7. `POST /api/search` — returns results with similarity scores
8. `GET /api/status` — returns per-table status
9. `POST /api/retry` — returns retried count (new endpoint)
10. `POST /api/teardown` — success case
11. `GET /api/worker-health` — returns worker status (new endpoint)
12. Rate limiting — verify 429 after threshold
13. Security headers — CSP, X-Frame-Options present

**Not in scope:** Integration tests hitting real DB, browser-based UI tests, MCP endpoint tests.

---

## Files Changed

| File | Change |
|------|--------|
| `pgsemantic/db/queue.py` | Add retry + worker health functions |
| `pgsemantic/commands/retry.py` | New retry command |
| `pgsemantic/cli.py` | Register retry command |
| `pgsemantic/worker/daemon.py` | Add heartbeat writes |
| `pgsemantic/commands/status.py` | Show worker health |
| `pgsemantic/web/app.py` | Add /api/retry + /api/worker-health endpoints |
| `pgsemantic/web/static/index.html` | Retry button + worker health display |
| `tests/unit/test_web_app.py` | New test file |
| `tests/unit/test_queue.py` | Tests for retry + health functions |

# High-Impact Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add retry for failed jobs (CLI + UI), worker health visibility in status, and web UI test coverage.

**Architecture:** Three independent features sharing the queue module as foundation. Retry and worker health add new DB functions + new endpoints. Web UI tests mock the DB layer and test all endpoints including the new ones.

**Tech Stack:** Python 3.10+, Typer, FastAPI, psycopg3, pytest + httpx TestClient

---

## File Structure

| File | Responsibility |
|------|---------------|
| `pgsemantic/db/queue.py` | Add `retry_failed_jobs()`, `create_worker_health_table()`, `upsert_worker_heartbeat()`, `get_worker_health()`, `delete_worker_health()` |
| `pgsemantic/commands/retry.py` | New CLI command: `pgsemantic retry` |
| `pgsemantic/cli.py` | Register retry command |
| `pgsemantic/worker/daemon.py` | Add heartbeat writes to existing poll loop |
| `pgsemantic/commands/status.py` | Show worker health info |
| `pgsemantic/web/app.py` | Add `POST /api/retry`, `GET /api/worker-health` endpoints + Pydantic model |
| `pgsemantic/web/static/index.html` | Retry button on Status page, worker health badge on Dashboard/Status |
| `tests/unit/test_queue.py` | Tests for retry + health DB functions |
| `tests/unit/test_web_app.py` | New file: endpoint tests for the full web API |

---

### Task 1: Retry Failed Jobs — DB Layer

**Files:**
- Modify: `pgsemantic/db/queue.py` (add function after `count_failed` at line ~210)
- Modify: `tests/unit/test_queue.py` (add tests at end)

- [ ] **Step 1: Write failing tests for `retry_failed_jobs`**

Add to `tests/unit/test_queue.py`:

```python
from pgsemantic.db.queue import retry_failed_jobs


class TestRetryFailedJobs:
    def test_retries_for_specific_table(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.rowcount = 5
        result = retry_failed_jobs(mock_conn, table_name="products")
        assert result == 5
        mock_conn.commit.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "status = 'pending'" in sql
        assert "table_name = %(table_name)s" in sql

    def test_retries_all_tables(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.rowcount = 12
        result = retry_failed_jobs(mock_conn, table_name=None)
        assert result == 12
        sql = mock_conn.execute.call_args[0][0]
        assert "table_name" not in sql

    def test_returns_zero_when_no_failed_jobs(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.rowcount = 0
        result = retry_failed_jobs(mock_conn, table_name="products")
        assert result == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_queue.py::TestRetryFailedJobs -v`
Expected: ImportError — `retry_failed_jobs` not found

- [ ] **Step 3: Implement `retry_failed_jobs` in queue.py**

Add these SQL constants after `SQL_COUNT_FAILED` (around line 106):

```python
# Reset failed jobs to pending for retry.
SQL_RETRY_FAILED_FOR_TABLE = """
    UPDATE pgvector_setup_queue
    SET status = 'pending', retries = 0, error_msg = NULL, updated_at = NOW()
    WHERE status = 'failed'
      AND table_name = %(table_name)s;
"""

SQL_RETRY_FAILED_ALL = """
    UPDATE pgvector_setup_queue
    SET status = 'pending', retries = 0, error_msg = NULL, updated_at = NOW()
    WHERE status = 'failed';
"""
```

Add this function after `count_failed` (around line 210):

```python
def retry_failed_jobs(conn: DictConnection, table_name: str | None = None) -> int:
    """Reset failed jobs to pending for retry. Returns count of jobs retried."""
    if table_name is not None:
        result = conn.execute(SQL_RETRY_FAILED_FOR_TABLE, {"table_name": table_name})
    else:
        result = conn.execute(SQL_RETRY_FAILED_ALL)
    conn.commit()
    count = result.rowcount
    logger.info("Retried %d failed jobs%s", count, f" for {table_name}" if table_name else "")
    return count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_queue.py::TestRetryFailedJobs -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pgsemantic/db/queue.py tests/unit/test_queue.py
git commit -m "feat: add retry_failed_jobs to queue module"
```

---

### Task 2: Retry Failed Jobs — CLI Command

**Files:**
- Create: `pgsemantic/commands/retry.py`
- Modify: `pgsemantic/cli.py` (add import + registration)

- [ ] **Step 1: Create `pgsemantic/commands/retry.py`**

```python
"""pgsemantic retry — reset failed embedding jobs for retry."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from pgsemantic.config import load_settings
from pgsemantic.db.client import get_connection
from pgsemantic.db.queue import retry_failed_jobs

console = Console()


def retry_command(
    table: str = typer.Option(
        "",
        "--table",
        "-t",
        help="Table to retry failed jobs for.",
    ),
    all_tables: bool = typer.Option(
        False,
        "--all",
        help="Retry failed jobs for all tables.",
    ),
    db: str = typer.Option(
        "",
        "--db",
        "-d",
        help="Database URL (overrides DATABASE_URL env var).",
    ),
) -> None:
    """Retry failed embedding jobs."""
    if not table and not all_tables:
        console.print(Panel(
            "[red]Specify --table <name> or --all.[/red]",
            title="Missing Option",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    settings = load_settings()
    database_url = db if db else settings.database_url

    if not database_url:
        console.print(Panel(
            "[red]No database URL provided.[/red]\n\n"
            "Set DATABASE_URL in your .env file or pass --db.",
            title="Missing Database URL",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    try:
        with get_connection(database_url) as conn:
            table_name = table if table else None
            count = retry_failed_jobs(conn, table_name=table_name)

        if count == 0:
            scope = f"for {table}" if table else "across all tables"
            console.print(f"[yellow]No failed jobs found {scope}.[/yellow]")
        else:
            scope = table if table else "all tables"
            console.print(f"[green]Retried {count} failed job(s) for {scope}.[/green]")

    except Exception as e:
        console.print(Panel(
            f"[red]Retry failed:[/red]\n\n{e}",
            title="Error",
            border_style="red",
        ))
        raise typer.Exit(code=1) from e
```

- [ ] **Step 2: Register in `cli.py`**

Add after the existing worker import (line 44):

```python
from pgsemantic.commands.retry import retry_command  # noqa: E402
```

Add after the worker registration (line 54):

```python
app.command(name="retry")(retry_command)
```

- [ ] **Step 3: Verify CLI registration**

Run: `python -m pgsemantic retry --help`
Expected: Shows help text with --table, --all, --db options

- [ ] **Step 4: Commit**

```bash
git add pgsemantic/commands/retry.py pgsemantic/cli.py
git commit -m "feat: add pgsemantic retry CLI command"
```

---

### Task 3: Retry Failed Jobs — Web UI Endpoint + Button

**Files:**
- Modify: `pgsemantic/web/app.py` (add Pydantic model + endpoint)
- Modify: `pgsemantic/web/static/index.html` (add retry button)

- [ ] **Step 1: Add Pydantic model and endpoint to `app.py`**

Add `RetryRequest` model after `BulkApplyRequest` (around line 275):

```python
class RetryRequest(BaseModel):
    table: str = Field("", max_length=128)
    all_tables: bool = False

    @field_validator("table")
    @classmethod
    def validate_ident(cls, v: str) -> str:
        return _validate_identifier(v) if v else v
```

Add the retry endpoint after the `status_dashboard` endpoint (around line 1021):

```python
@app.post("/api/retry")
async def retry_failed(req: RetryRequest):
    """Retry failed embedding jobs."""
    from pgsemantic.db.queue import retry_failed_jobs

    if not req.table and not req.all_tables:
        raise HTTPException(400, "Specify table or set all_tables=true")

    db_url = _get_db_url()
    try:
        with get_connection(db_url) as conn:
            table_name = req.table if req.table else None
            count = retry_failed_jobs(conn, table_name=table_name)
        return {"retried": count, "table": req.table or "all"}
    except Exception as e:
        logger.exception("retry_failed failed")
        raise HTTPException(500, "Retry failed. Check server logs.")
```

- [ ] **Step 2: Add retry button to Status section in `index.html`**

Find the `loadStatus` function (around line 629) and the status table rendering. After the failed count cell in the status table row, add a retry button:

In the JS where status table rows are built, modify the failed cell to include a retry button when `t.failed > 0`:

```javascript
// Inside the status table row building, replace the failed cell with:
const failedCell = t.failed > 0
    ? `<td style="color:var(--danger)">${t.failed} <button class="btn btn-sm" onclick="retryFailed('${t.table}')" style="margin-left:6px;padding:2px 8px;font-size:0.8em">Retry</button></td>`
    : `<td style="color:var(--success)">${t.failed}</td>`;
```

Add the `retryFailed` function near the other status functions:

```javascript
async function retryFailed(table) {
    try {
        const d = await api('/api/retry', {
            method: 'POST',
            body: JSON.stringify({ table })
        });
        toast(`Retried ${d.retried} failed job(s) for ${table}`);
        loadStatus();
    } catch(e) {
        toast(e.message, false);
    }
}
```

- [ ] **Step 3: Test manually**

Run: `python -m pgsemantic ui --reload`
Navigate to Status page, verify retry button appears when failed > 0.

- [ ] **Step 4: Commit**

```bash
git add pgsemantic/web/app.py pgsemantic/web/static/index.html
git commit -m "feat: add retry endpoint and button to web UI"
```

---

### Task 4: Worker Health — DB Layer

**Files:**
- Modify: `pgsemantic/db/queue.py` (add health table + functions)
- Modify: `tests/unit/test_queue.py` (add tests)

- [ ] **Step 1: Write failing tests for worker health functions**

Add to `tests/unit/test_queue.py`:

```python
from pgsemantic.db.queue import (
    create_worker_health_table,
    delete_worker_health,
    get_worker_health,
    upsert_worker_heartbeat,
)


class TestCreateWorkerHealthTable:
    def test_executes_create_table(self) -> None:
        mock_conn = MagicMock()
        create_worker_health_table(mock_conn)
        sql = mock_conn.execute.call_args[0][0]
        assert "pgvector_setup_worker_health" in sql
        mock_conn.commit.assert_called_once()


class TestUpsertWorkerHeartbeat:
    def test_upserts_heartbeat(self) -> None:
        mock_conn = MagicMock()
        upsert_worker_heartbeat(mock_conn, worker_id="mac-1234", jobs_processed=42)
        args = mock_conn.execute.call_args
        params = args[0][1]
        assert params["worker_id"] == "mac-1234"
        assert params["jobs_processed"] == 42
        mock_conn.commit.assert_called_once()


class TestGetWorkerHealth:
    def test_returns_workers(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            {"worker_id": "mac-1234", "last_heartbeat": "2026-03-26T12:00:00Z",
             "jobs_processed": 42, "started_at": "2026-03-26T11:00:00Z"},
        ]
        result = get_worker_health(mock_conn)
        assert len(result) == 1
        assert result[0]["worker_id"] == "mac-1234"

    def test_returns_empty_list_when_no_workers(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        result = get_worker_health(mock_conn)
        assert result == []

    def test_handles_missing_table(self) -> None:
        """get_worker_health returns [] if the health table doesn't exist."""
        import psycopg
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = psycopg.errors.UndefinedTable("relation does not exist")
        result = get_worker_health(mock_conn)
        assert result == []


class TestDeleteWorkerHealth:
    def test_deletes_worker(self) -> None:
        mock_conn = MagicMock()
        delete_worker_health(mock_conn, worker_id="mac-1234")
        args = mock_conn.execute.call_args
        assert args[0][1] == {"worker_id": "mac-1234"}
        mock_conn.commit.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_queue.py::TestCreateWorkerHealthTable tests/unit/test_queue.py::TestUpsertWorkerHeartbeat tests/unit/test_queue.py::TestGetWorkerHealth tests/unit/test_queue.py::TestDeleteWorkerHealth -v`
Expected: ImportError — functions not found

- [ ] **Step 3: Implement worker health functions in queue.py**

Add SQL constants after `SQL_COUNT_FAILED`:

```python
# -- Worker health table ---------------------------------------------------

SQL_CREATE_WORKER_HEALTH_TABLE = """
    CREATE TABLE IF NOT EXISTS pgvector_setup_worker_health (
        worker_id       TEXT PRIMARY KEY,
        last_heartbeat  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        jobs_processed  INT NOT NULL DEFAULT 0,
        started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
"""

SQL_UPSERT_WORKER_HEARTBEAT = """
    INSERT INTO pgvector_setup_worker_health (worker_id, last_heartbeat, jobs_processed, started_at)
    VALUES (%(worker_id)s, NOW(), %(jobs_processed)s, NOW())
    ON CONFLICT (worker_id)
    DO UPDATE SET
        last_heartbeat = NOW(),
        jobs_processed = %(jobs_processed)s;
"""

SQL_GET_WORKER_HEALTH = """
    SELECT worker_id, last_heartbeat, jobs_processed, started_at
    FROM pgvector_setup_worker_health
    ORDER BY last_heartbeat DESC;
"""

SQL_DELETE_WORKER_HEALTH = """
    DELETE FROM pgvector_setup_worker_health
    WHERE worker_id = %(worker_id)s;
"""
```

Add functions after `retry_failed_jobs`:

```python
def create_worker_health_table(conn: DictConnection) -> None:
    """Create the worker health table if it doesn't exist."""
    conn.execute(SQL_CREATE_WORKER_HEALTH_TABLE)
    conn.commit()
    logger.info("Worker health table ready")


def upsert_worker_heartbeat(
    conn: DictConnection, worker_id: str, jobs_processed: int
) -> None:
    """Update worker heartbeat timestamp and job count."""
    conn.execute(
        SQL_UPSERT_WORKER_HEARTBEAT,
        {"worker_id": worker_id, "jobs_processed": jobs_processed},
    )
    conn.commit()


def get_worker_health(conn: DictConnection) -> list[dict[str, object]]:
    """Get all worker health records. Returns [] if table doesn't exist."""
    try:
        result = conn.execute(SQL_GET_WORKER_HEALTH).fetchall()
        return [dict(row) for row in result]
    except Exception:
        conn.rollback()
        return []


def delete_worker_health(conn: DictConnection, worker_id: str) -> None:
    """Remove a worker's health record (on graceful shutdown)."""
    conn.execute(SQL_DELETE_WORKER_HEALTH, {"worker_id": worker_id})
    conn.commit()
    logger.info("Removed health record for worker %s", worker_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_queue.py -v`
Expected: All tests pass (existing + new)

- [ ] **Step 5: Commit**

```bash
git add pgsemantic/db/queue.py tests/unit/test_queue.py
git commit -m "feat: add worker health table and functions"
```

---

### Task 5: Worker Health — Daemon Integration

**Files:**
- Modify: `pgsemantic/worker/daemon.py`

- [ ] **Step 1: Add heartbeat writes to the worker loop**

Add imports at top of `daemon.py`:

```python
import os
import platform
```

Add new import from queue module (update existing import line):

```python
from pgsemantic.db.queue import (
    claim_batch,
    complete_job,
    create_worker_health_table,
    delete_worker_health,
    fail_job,
    upsert_worker_heartbeat,
)
```

In `run_worker`, after the config validation block (around line 64) and before the poll loop, add:

```python
    # Generate a stable worker ID for health tracking
    worker_id = f"{platform.node()}-{os.getpid()}"
    jobs_processed = 0

    # Ensure health table exists and register this worker
    try:
        with get_connection(database_url) as conn:
            create_worker_health_table(conn)
            upsert_worker_heartbeat(conn, worker_id, jobs_processed)
    except Exception as e:
        logger.warning("Could not initialize worker health table: %s", e)
```

Inside the main loop, after successful job processing (after `_process_job` call, around line 106), increment and heartbeat:

```python
                    _process_job(conn, job, config, provider, max_retries)
                    jobs_processed += 1
```

After the `for job in jobs` loop completes (still inside the `if not jobs` branch is wrong — put this after the jobs loop), add heartbeat:

```python
                # Heartbeat after processing batch
                try:
                    upsert_worker_heartbeat(conn, worker_id, jobs_processed)
                except Exception:
                    pass  # Non-critical
```

Also add heartbeat in the idle branch (inside the `if not jobs:` block, alongside the existing heartbeat log):

```python
                if not jobs:
                    now = time.time()
                    if now - last_heartbeat >= WORKER_HEARTBEAT_INTERVAL_S:
                        logger.info("Worker alive, queue empty.")
                        try:
                            upsert_worker_heartbeat(conn, worker_id, jobs_processed)
                        except Exception:
                            pass
                        last_heartbeat = now
                    time.sleep(poll_interval_s)
                    backoff_s = 1.0
                    continue
```

At the end of `run_worker`, after the loop, clean up:

```python
    # Clean up health record on shutdown
    try:
        with get_connection(database_url) as conn:
            delete_worker_health(conn, worker_id)
    except Exception:
        pass  # Best effort cleanup

    logger.info("Worker stopped.")
```

- [ ] **Step 2: Verify worker still starts**

Run: `python -m pgsemantic worker --help`
Expected: Shows help text (confirms import works)

- [ ] **Step 3: Commit**

```bash
git add pgsemantic/worker/daemon.py
git commit -m "feat: worker writes heartbeat to health table"
```

---

### Task 6: Worker Health — Status Command + Web UI

**Files:**
- Modify: `pgsemantic/commands/status.py`
- Modify: `pgsemantic/web/app.py` (add endpoint)
- Modify: `pgsemantic/web/static/index.html` (add health display)

- [ ] **Step 1: Add worker health to CLI status command**

Add import in `status.py`:

```python
from pgsemantic.db.queue import count_failed, count_pending, get_worker_health
```

After the table is printed (after `console.print(table)` at line 131), add:

```python
    # Worker health section
    try:
        with get_connection(db_url) as conn:
            workers = get_worker_health(conn)

        if workers:
            from datetime import datetime, timezone

            console.print()
            worker_table = Table(title="Worker Health", show_lines=True)
            worker_table.add_column("Worker ID", style="cyan")
            worker_table.add_column("Status")
            worker_table.add_column("Last Seen", style="dim")
            worker_table.add_column("Jobs Processed", justify="right")
            worker_table.add_column("Uptime", style="dim")

            now = datetime.now(tz=timezone.utc)
            for w in workers:
                last_hb = w["last_heartbeat"]
                if hasattr(last_hb, "timestamp"):
                    age_s = (now - last_hb).total_seconds()
                else:
                    age_s = 999

                if age_s < 30:
                    status_str = "[green]Running[/green]"
                elif age_s < 300:
                    status_str = "[yellow]Stale[/yellow]"
                else:
                    status_str = "[red]Not running[/red]"

                last_seen = str(w["last_heartbeat"])[:19] if w["last_heartbeat"] else "—"
                started = w.get("started_at")
                if started and hasattr(started, "timestamp"):
                    uptime_s = (now - started).total_seconds()
                    hours, remainder = divmod(int(uptime_s), 3600)
                    minutes, _ = divmod(remainder, 60)
                    uptime_str = f"{hours}h {minutes}m"
                else:
                    uptime_str = "—"

                worker_table.add_row(
                    str(w["worker_id"]),
                    status_str,
                    last_seen,
                    str(w["jobs_processed"]),
                    uptime_str,
                )

            console.print(worker_table)
        else:
            console.print("\n[dim]No worker health data. Start a worker with: pgsemantic worker[/dim]")
    except Exception:
        pass  # Worker health is supplementary, don't fail status
```

- [ ] **Step 2: Add `/api/worker-health` endpoint to `app.py`**

Add after the existing `/api/worker/status` endpoint (around line 1053):

```python
@app.get("/api/worker-health")
async def worker_health():
    """Get worker health from the database health table."""
    from pgsemantic.db.queue import get_worker_health
    from datetime import datetime, timezone

    db_url = _get_db_url()
    try:
        with get_connection(db_url) as conn:
            workers = get_worker_health(conn)

        now = datetime.now(tz=timezone.utc)
        result = []
        for w in workers:
            last_hb = w["last_heartbeat"]
            if hasattr(last_hb, "timestamp"):
                age_s = (now - last_hb).total_seconds()
            else:
                age_s = 999

            if age_s < 30:
                status = "running"
            elif age_s < 300:
                status = "stale"
            else:
                status = "stopped"

            result.append({
                "worker_id": str(w["worker_id"]),
                "status": status,
                "last_heartbeat": str(w["last_heartbeat"]),
                "jobs_processed": w["jobs_processed"],
                "started_at": str(w.get("started_at")),
                "age_seconds": round(age_s, 1),
            })

        return {"workers": result}
    except Exception:
        return {"workers": []}
```

- [ ] **Step 3: Add worker health display to `index.html`**

In the Dashboard section's `loadDashboard` function, add a worker health fetch and display. After the existing dashboard stats, add:

```javascript
async function loadWorkerHealth() {
    try {
        const d = await api('/api/worker-health');
        const el = $('worker-health-display');
        if (!el) return;
        if (!d.workers || d.workers.length === 0) {
            el.innerHTML = '<span class="muted">No worker health data</span>';
            return;
        }
        el.innerHTML = d.workers.map(w => {
            const color = w.status === 'running' ? 'var(--success)' : w.status === 'stale' ? 'var(--warning)' : 'var(--danger)';
            const dot = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:6px"></span>`;
            return `<div style="padding:4px 0">${dot}<strong>${w.worker_id}</strong> — ${w.status} · ${w.jobs_processed} jobs processed · last seen ${Math.round(w.age_seconds)}s ago</div>`;
        }).join('');
    } catch(e) { /* non-critical */ }
}
```

Add a `<div id="worker-health-display"></div>` in both the Dashboard and Status HTML sections, inside a card/panel.

In the Dashboard section (around line 355), after existing content:

```html
<div class="card" style="margin-top:1rem">
    <h3>Worker Health</h3>
    <div id="worker-health-display"><span class="muted">Loading...</span></div>
</div>
```

Call `loadWorkerHealth()` from `loadDashboard()` and `loadStatus()`.

- [ ] **Step 4: Verify manually**

Run: `python -m pgsemantic ui --reload`
Check Dashboard and Status pages show worker health section.

- [ ] **Step 5: Commit**

```bash
git add pgsemantic/commands/status.py pgsemantic/web/app.py pgsemantic/web/static/index.html
git commit -m "feat: show worker health in status command and web UI"
```

---

### Task 7: Web UI Tests

**Files:**
- Create: `tests/unit/test_web_app.py`

- [ ] **Step 1: Create test file with CSRF and connection tests**

```python
"""Tests for FastAPI web UI endpoints — all DB interactions mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Create a test client with mocked DB URL."""
    with patch("pgsemantic.web.app.load_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            database_url="postgresql://user:pass@localhost/testdb",
            openai_api_key=None,
            ollama_base_url="http://localhost:11434",
            embedding_provider="local",
            worker_batch_size=10,
            worker_poll_interval_ms=500,
            worker_max_retries=3,
        )
        from pgsemantic.web.app import app
        yield TestClient(app)


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
        assert len(resp.json()["token"]) == 64  # hex(32)

    def test_post_without_csrf_returns_403(self, client) -> None:
        resp = client.post("/api/search", json={"query": "test", "table": "t", "limit": 5})
        assert resp.status_code == 403

    def test_post_with_wrong_csrf_returns_403(self, client) -> None:
        resp = client.post(
            "/api/search",
            json={"query": "test", "table": "t", "limit": 5},
            headers={"X-CSRF-Token": "wrong"},
        )
        assert resp.status_code == 403


class TestConnection:
    @patch("pgsemantic.web.app.get_connection")
    @patch("pgsemantic.web.app.get_pgvector_version")
    def test_connection_info_success(self, mock_pgv, mock_conn, client) -> None:
        mock_pgv.return_value = (0, 7, 0)
        mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        resp = client.get("/api/connection")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is True
        assert "pass" not in data["database_url"]  # Password masked

    def test_connection_info_no_db(self, client) -> None:
        with patch("pgsemantic.web.app.load_settings") as mock:
            mock.return_value = MagicMock(database_url=None)
            resp = client.get("/api/connection")
            assert resp.status_code == 400


class TestStatus:
    @patch("pgsemantic.web.app.get_connection")
    @patch("pgsemantic.web.app.load_project_config")
    def test_status_no_config(self, mock_config, mock_conn, client) -> None:
        mock_config.return_value = None
        resp = client.get("/api/status")
        assert resp.status_code == 200
        assert resp.json()["tables"] == []

    @patch("pgsemantic.web.app.get_connection")
    @patch("pgsemantic.web.app.load_project_config")
    @patch("pgsemantic.web.app.count_embedded")
    @patch("pgsemantic.web.app.count_total_with_content")
    @patch("pgsemantic.web.app.count_pending")
    @patch("pgsemantic.web.app.count_failed")
    def test_status_with_table(
        self, mock_failed, mock_pending, mock_total, mock_embedded,
        mock_config, mock_conn, client,
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
    @patch("pgsemantic.web.app.get_connection")
    def test_retry_success(self, mock_conn, client, csrf_token) -> None:
        mock_inner = MagicMock()
        mock_conn.return_value.__enter__ = MagicMock(return_value=mock_inner)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_inner.execute.return_value.rowcount = 3
        with patch("pgsemantic.web.app.retry_failed_jobs", return_value=3) as mock_retry:
            resp = client.post(
                "/api/retry",
                json={"table": "news"},
                headers={"X-CSRF-Token": csrf_token},
            )
        assert resp.status_code == 200
        assert resp.json()["retried"] == 3

    def test_retry_missing_table(self, client, csrf_token) -> None:
        resp = client.post(
            "/api/retry",
            json={},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400


class TestWorkerHealth:
    @patch("pgsemantic.web.app.get_connection")
    def test_worker_health_empty(self, mock_conn, client) -> None:
        mock_inner = MagicMock()
        mock_conn.return_value.__enter__ = MagicMock(return_value=mock_inner)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        with patch("pgsemantic.web.app.get_worker_health", return_value=[]):
            resp = client.get("/api/worker-health")
        assert resp.status_code == 200
        assert resp.json()["workers"] == []


class TestSecurityHeaders:
    def test_security_headers_present(self, client) -> None:
        resp = client.get("/api/csrf-token")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert "Content-Security-Policy" in resp.headers


class TestSearch:
    @patch("pgsemantic.web.app.get_connection")
    @patch("pgsemantic.web.app.load_project_config")
    @patch("pgsemantic.web.app.get_provider")
    @patch("pgsemantic.web.app.search_similar")
    def test_search_success(
        self, mock_search, mock_provider, mock_config, mock_conn,
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
        mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_provider.return_value.embed_query.return_value = [0.1] * 384
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
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/unit/test_web_app.py -v`
Expected: All tests pass. Some may need mock adjustments — fix any that fail due to import ordering or mock paths.

- [ ] **Step 3: Fix any failing tests**

Common issues: mock paths need to match where the function is imported in `app.py`, not where it's defined. Adjust patch targets as needed.

- [ ] **Step 4: Run full unit test suite**

Run: `pytest tests/unit/ -v`
Expected: All existing tests still pass + new tests pass

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_web_app.py
git commit -m "test: add web UI endpoint tests"
```

---

### Task 8: Final Verification

- [ ] **Step 1: Run full unit test suite**

Run: `pytest tests/unit/ -v`
Expected: All pass

- [ ] **Step 2: Run ruff linter**

Run: `ruff check pgsemantic/ tests/`
Expected: No errors

- [ ] **Step 3: Verify CLI commands**

Run: `python -m pgsemantic --help`
Expected: Shows retry in the command list

Run: `python -m pgsemantic retry --help`
Expected: Shows --table, --all, --db options

- [ ] **Step 4: Commit any remaining fixes**

```bash
git add -A && git commit -m "chore: final cleanup for high-impact improvements"
```

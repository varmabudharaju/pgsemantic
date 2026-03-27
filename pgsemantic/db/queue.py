"""Queue table management for pgsemantic.

Uses a plain Postgres table (pgvector_setup_queue) with SELECT FOR UPDATE
SKIP LOCKED to claim work atomically. Jobs are deleted on completion and
marked failed with an error message on failure. Never loses events.

All SQL is defined as module-level constants.
"""
from __future__ import annotations

import logging

import psycopg

from pgsemantic.config import DEFAULT_WORKER_MAX_RETRIES
from pgsemantic.exceptions import QueueError

logger = logging.getLogger(__name__)

# Type alias matching client.py — a psycopg connection returning dict rows.
DictConnection = psycopg.Connection[dict[str, object]]

# -- SQL Constants ----------------------------------------------------------

# Create the queue table with a UNIQUE constraint on (table_name, row_id,
# column_name) so duplicate events for the same row/column are upserted
# rather than duplicated.
SQL_CREATE_QUEUE_TABLE = """
    CREATE TABLE IF NOT EXISTS pgvector_setup_queue (
        id          BIGSERIAL PRIMARY KEY,
        table_name  TEXT NOT NULL,
        row_id      TEXT NOT NULL,
        column_name TEXT NOT NULL,
        operation   TEXT NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
        status      TEXT NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending', 'processing', 'failed')),
        retries     INT NOT NULL DEFAULT 0,
        error_msg   TEXT,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (table_name, row_id, column_name)
    );
"""

# Partial index on pending jobs ordered by creation time for efficient
# queue polling. Only indexes rows with status = 'pending'.
SQL_CREATE_QUEUE_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_pgvs_queue_pending
        ON pgvector_setup_queue (created_at ASC)
        WHERE status = 'pending';
"""

# Atomically claim a batch of pending jobs using SELECT FOR UPDATE SKIP LOCKED.
# Updates their status to 'processing' and returns the claimed rows.
# SKIP LOCKED ensures multiple workers never claim the same job.
SQL_CLAIM_BATCH = """
    WITH claimed AS (
        SELECT id
        FROM pgvector_setup_queue
        WHERE status = 'pending'
        ORDER BY created_at ASC
        LIMIT %(batch_size)s
        FOR UPDATE SKIP LOCKED
    )
    UPDATE pgvector_setup_queue q
    SET status = 'processing', updated_at = NOW()
    FROM claimed c
    WHERE q.id = c.id
    RETURNING q.id, q.table_name, q.row_id, q.column_name, q.operation;
"""

# Delete a completed job — no need to keep finished work in the queue.
SQL_COMPLETE_JOB = """
    DELETE FROM pgvector_setup_queue
    WHERE id = %(job_id)s;
"""

# Mark a job as failed, increment retry count, and record the error message.
# If retries exceed the max, the job stays failed for manual inspection.
SQL_FAIL_JOB = """
    UPDATE pgvector_setup_queue
    SET status = CASE
            WHEN retries + 1 < %(max_retries)s THEN 'pending'
            ELSE 'failed'
        END,
        retries = retries + 1,
        error_msg = %(error_msg)s,
        updated_at = NOW()
    WHERE id = %(job_id)s;
"""

# Count pending jobs for a specific table (used by status command).
SQL_COUNT_PENDING = """
    SELECT COUNT(*) AS cnt
    FROM pgvector_setup_queue
    WHERE table_name = %(table_name)s
      AND status = 'pending';
"""

# Count failed jobs for a specific table (used by status command).
SQL_COUNT_FAILED = """
    SELECT COUNT(*) AS cnt
    FROM pgvector_setup_queue
    WHERE table_name = %(table_name)s
      AND status = 'failed';
"""

# Reset failed jobs to pending for a specific table so the worker retries them.
SQL_RETRY_FAILED_FOR_TABLE = """
    UPDATE pgvector_setup_queue
    SET status = 'pending', retries = 0, error_msg = NULL, updated_at = NOW()
    WHERE status = 'failed'
      AND table_name = %(table_name)s;
"""

# Reset ALL failed jobs to pending (across every table).
SQL_RETRY_FAILED_ALL = """
    UPDATE pgvector_setup_queue
    SET status = 'pending', retries = 0, error_msg = NULL, updated_at = NOW()
    WHERE status = 'failed';
"""

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


# -- Functions --------------------------------------------------------------


def create_queue_table(conn: DictConnection) -> None:
    """Create the queue table and partial index if they don't exist.

    Commits the transaction so the table is immediately available.
    Raises QueueError if creation fails.
    """
    try:
        conn.execute(SQL_CREATE_QUEUE_TABLE)
        conn.execute(SQL_CREATE_QUEUE_INDEX)
        conn.commit()
        logger.info("Queue table pgvector_setup_queue ready")
    except psycopg.Error as e:
        raise QueueError(
            f"Failed to create queue table: {e}"
        ) from e


def claim_batch(
    conn: DictConnection,
    batch_size: int = 10,
) -> list[dict[str, object]]:
    """Claim a batch of pending jobs atomically.

    Uses SELECT FOR UPDATE SKIP LOCKED so multiple workers can poll
    concurrently without double-processing. Returns a list of job dicts
    with keys: id, table_name, row_id, column_name, operation.
    """
    try:
        result = conn.execute(
            SQL_CLAIM_BATCH, {"batch_size": batch_size}
        ).fetchall()
        conn.commit()
        return [dict(row) for row in result]
    except psycopg.Error as e:
        raise QueueError(f"Failed to claim batch: {e}") from e


def complete_job(conn: DictConnection, job_id: int) -> None:
    """Delete a completed job from the queue.

    Commits the transaction immediately.
    """
    try:
        conn.execute(SQL_COMPLETE_JOB, {"job_id": job_id})
        conn.commit()
        logger.debug("Completed job %d", job_id)
    except psycopg.Error as e:
        raise QueueError(
            f"Failed to complete job {job_id}: {e}"
        ) from e


def fail_job(
    conn: DictConnection,
    job_id: int,
    error_msg: str,
    max_retries: int = DEFAULT_WORKER_MAX_RETRIES,
) -> None:
    """Mark a job as failed with an error message.

    If retries < max_retries, the job is returned to 'pending' for retry.
    Otherwise it stays 'failed' for manual inspection.
    Commits the transaction immediately.
    """
    try:
        conn.execute(
            SQL_FAIL_JOB,
            {
                "job_id": job_id,
                "error_msg": error_msg,
                "max_retries": max_retries,
            },
        )
        conn.commit()
        logger.warning("Job %d failed: %s", job_id, error_msg)
    except psycopg.Error as e:
        raise QueueError(
            f"Failed to mark job {job_id} as failed: {e}"
        ) from e


def count_pending(conn: DictConnection, table_name: str) -> int:
    """Count pending jobs for a specific table."""
    result = conn.execute(
        SQL_COUNT_PENDING, {"table_name": table_name}
    ).fetchone()
    if result is None:
        return 0
    return int(str(result["cnt"]))


def count_failed(conn: DictConnection, table_name: str) -> int:
    """Count failed jobs for a specific table."""
    result = conn.execute(
        SQL_COUNT_FAILED, {"table_name": table_name}
    ).fetchone()
    if result is None:
        return 0
    return int(str(result["cnt"]))


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

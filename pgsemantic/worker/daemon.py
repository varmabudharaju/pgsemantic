"""Worker daemon -- polls queue and processes embedding jobs.

Runs as a sync loop. Claims batches from pgvector_setup_queue,
generates embeddings, and writes them back. Reconnects with
exponential backoff on connection errors.
"""
from __future__ import annotations

import logging
import signal
import time
from types import FrameType

import psycopg

from pgsemantic.config import (
    WORKER_HEARTBEAT_INTERVAL_S,
    ProjectConfig,
    Settings,
    TableConfig,
    load_project_config,
)
from pgsemantic.db.client import get_connection
from pgsemantic.db.queue import claim_batch, complete_job, fail_job
from pgsemantic.db.vectors import fetch_row_text, null_embedding, update_embedding
from pgsemantic.embeddings import get_provider
from pgsemantic.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Sentinel for graceful shutdown
_shutdown_requested: bool = False


def _handle_signal(signum: int, frame: FrameType | None) -> None:
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = True
    logger.info("Shutdown signal received (signal %d). Finishing current batch...", signum)


def run_worker(settings: Settings) -> None:
    """Main worker loop. Runs until SIGINT/SIGTERM.

    Polls pgvector_setup_queue for pending jobs, claims them in batches,
    generates embeddings, and writes them back to the source table.
    Uses exponential backoff (1s -> 2s -> 4s -> ... -> 30s max) on
    connection errors.

    Raises ValueError if DATABASE_URL or project config is missing.
    """
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    database_url = settings.database_url
    if database_url is None:
        raise ValueError("DATABASE_URL is required for the worker")

    config = load_project_config()
    if config is None or not config.tables:
        raise ValueError("No tables configured. Run 'pgsemantic apply' first.")

    # Initialize embedding provider
    provider = get_provider(
        settings.embedding_provider,
        api_key=settings.openai_api_key,
    )

    poll_interval_s: float = settings.worker_poll_interval_ms / 1000.0
    max_retries: int = settings.worker_max_retries
    backoff_s: float = 1.0
    max_backoff_s: float = 30.0
    last_heartbeat: float = time.time()

    logger.info(
        "Worker started. Polling every %dms, batch size %d.",
        settings.worker_poll_interval_ms,
        settings.worker_batch_size,
    )

    while not _shutdown_requested:
        try:
            with get_connection(database_url) as conn:
                jobs = claim_batch(conn, batch_size=settings.worker_batch_size)

                if not jobs:
                    # Heartbeat log when idle
                    now = time.time()
                    if now - last_heartbeat >= WORKER_HEARTBEAT_INTERVAL_S:
                        logger.info("Worker alive, queue empty.")
                        last_heartbeat = now
                    time.sleep(poll_interval_s)
                    backoff_s = 1.0  # Reset backoff on successful connection
                    continue

                logger.info("Claimed %d jobs from queue.", len(jobs))
                backoff_s = 1.0
                last_heartbeat = time.time()

                for job in jobs:
                    if _shutdown_requested:
                        break
                    _process_job(conn, job, config, provider, max_retries)

        except psycopg.OperationalError as e:
            logger.warning(
                "Connection lost: %s. Retrying in %.0fs...", e, backoff_s
            )
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 2, max_backoff_s)
        except Exception as e:
            logger.error(
                "Unexpected error: %s. Retrying in %.0fs...", e, backoff_s
            )
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 2, max_backoff_s)

    logger.info("Worker stopped.")


def _process_job(
    conn: psycopg.Connection[dict[str, object]],
    job: dict[str, object],
    config: ProjectConfig,
    provider: EmbeddingProvider,
    max_retries: int,
) -> None:
    """Process a single queue job.

    For INSERT/UPDATE operations: fetches source text, generates embedding,
    and updates the embedding column.
    For DELETE operations: sets the embedding to NULL.
    On failure, marks the job as failed with retry logic.
    """
    job_id = int(str(job["id"]))
    table_name = str(job["table_name"])
    row_id = str(job["row_id"])
    column_name = str(job["column_name"])
    operation = str(job["operation"])

    # Find table config
    table_config: TableConfig | None = config.get_table_config(table_name)
    if table_config is None:
        logger.warning(
            "No config for table '%s', skipping job %d", table_name, job_id
        )
        fail_job(
            conn,
            job_id=job_id,
            error_msg=f"Table '{table_name}' not in project config",
            max_retries=max_retries,
        )
        return

    pk_columns = table_config.primary_key

    try:
        if operation == "DELETE":
            null_embedding(
                conn, table_name, row_id,
                pk_columns=pk_columns, schema=table_config.schema,
            )
            complete_job(conn, job_id=job_id)
            logger.info(
                "Cleared embedding for %s#%s (DELETE)", table_name, row_id
            )
        else:
            # INSERT or UPDATE -- fetch text, embed, update
            text = fetch_row_text(
                conn,
                table_name,
                column_name,
                row_id,
                pk_columns=pk_columns,
                schema=table_config.schema,
            )
            if text is None:
                # Row was deleted before we could process it
                complete_job(conn, job_id=job_id)
                logger.info("Row %s#%s gone, marking complete.", table_name, row_id)
                return

            # Generate embedding
            embedding: list[float] = provider.embed_query(text)

            # Write embedding back to source table
            update_embedding(
                conn,
                table_name,
                row_id,
                embedding,
                pk_columns=pk_columns,
                schema=table_config.schema,
            )
            conn.commit()
            complete_job(conn, job_id=job_id)
            logger.info(
                "Embedded %s#%s (%s, %dd)",
                table_name,
                row_id,
                column_name,
                len(embedding),
            )

    except Exception as e:
        logger.error("Failed to process job %d: %s", job_id, e)
        try:
            conn.rollback()
            fail_job(
                conn,
                job_id=job_id,
                error_msg=str(e),
                max_retries=max_retries,
            )
        except Exception:
            logger.error("Failed to mark job %d as failed", job_id)

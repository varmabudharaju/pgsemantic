"""Integration tests for the worker queue processing flow.

Tests the full cycle: INSERT → trigger → queue → claim → embed → update.
"""
from __future__ import annotations

import psycopg
import pytest

from pgsemantic.db.client import get_connection
from pgsemantic.db.introspect import column_exists
from pgsemantic.db.queue import (
    claim_batch,
    complete_job,
    count_pending,
    create_queue_table,
    fail_job,
)
from pgsemantic.db.vectors import fetch_row_text, update_embedding

from .conftest import TEST_TABLE

pytestmark = pytest.mark.integration

DIMENSIONS = 384


@pytest.fixture()
def worker_ready_table(db_url: str, db_conn: psycopg.Connection) -> None:
    """Ensure the table has embedding column, queue table, and trigger installed."""
    # Embedding column
    if not column_exists(db_conn, TEST_TABLE, "embedding"):
        db_conn.execute(
            f'ALTER TABLE "{TEST_TABLE}" ADD COLUMN "embedding" vector({DIMENSIONS});'
        )
        db_conn.commit()

    # Queue table
    create_queue_table(db_conn)

    # Trigger function
    trigger_fn = f"""
    CREATE OR REPLACE FUNCTION pgvector_setup_{TEST_TABLE}_fn()
    RETURNS TRIGGER AS $$
    BEGIN
        IF TG_OP = 'DELETE' THEN
            INSERT INTO pgvector_setup_queue
                (table_name, row_id, column_name, operation)
            VALUES (TG_TABLE_NAME, OLD.id::TEXT, TG_ARGV[0], 'DELETE')
            ON CONFLICT (table_name, row_id, column_name)
            DO UPDATE SET status='pending', operation='DELETE',
                retries=0, error_msg=NULL, updated_at=NOW();
            RETURN OLD;
        ELSE
            INSERT INTO pgvector_setup_queue
                (table_name, row_id, column_name, operation)
            VALUES (TG_TABLE_NAME, NEW.id::TEXT, TG_ARGV[0], TG_OP)
            ON CONFLICT (table_name, row_id, column_name)
            DO UPDATE SET status='pending', operation=TG_OP,
                retries=0, error_msg=NULL, updated_at=NOW();
            RETURN NEW;
        END IF;
    END;
    $$ LANGUAGE plpgsql;
    """
    db_conn.execute(trigger_fn)
    db_conn.commit()

    trigger_sql = f"""
    CREATE OR REPLACE TRIGGER pgvector_setup_{TEST_TABLE}_trigger
    AFTER INSERT OR UPDATE OF "description" OR DELETE
    ON "{TEST_TABLE}"
    FOR EACH ROW
    EXECUTE FUNCTION pgvector_setup_{TEST_TABLE}_fn('description');
    """
    db_conn.execute(trigger_sql)
    db_conn.commit()

    # Clean queue before each test
    db_conn.execute(
        "DELETE FROM pgvector_setup_queue WHERE table_name = %s;",
        (TEST_TABLE,),
    )
    db_conn.commit()


class TestQueueClaiming:
    def test_claim_empty_queue(
        self, db_conn: psycopg.Connection, worker_ready_table: None
    ) -> None:
        """Claiming from an empty queue should return no jobs."""
        jobs = claim_batch(db_conn, batch_size=10)
        # Filter to our test table (other tests may have entries)
        our_jobs = [j for j in jobs if str(j["table_name"]) == TEST_TABLE]
        assert our_jobs == []

    def test_insert_creates_queue_entry(
        self, db_conn: psycopg.Connection, worker_ready_table: None
    ) -> None:
        """INSERT should trigger a queue entry."""
        db_conn.execute(
            f"INSERT INTO {TEST_TABLE} (id, name, description) "
            f"VALUES (200, 'Queue Test', 'Testing the queue mechanism.');",
        )
        db_conn.commit()

        pending = count_pending(db_conn, TEST_TABLE)
        assert pending >= 1

        # Clean up
        db_conn.execute(f"DELETE FROM {TEST_TABLE} WHERE id = 200;")
        db_conn.commit()

    def test_claim_and_complete_job(
        self, db_conn: psycopg.Connection, worker_ready_table: None
    ) -> None:
        """Full claim → process → complete cycle."""
        # Insert to create a queue entry
        db_conn.execute(
            f"INSERT INTO {TEST_TABLE} (id, name, description) "
            f"VALUES (201, 'Claim Test', 'Testing claim and complete.');",
        )
        db_conn.commit()

        # Claim the job
        jobs = claim_batch(db_conn, batch_size=10)
        our_jobs = [j for j in jobs if str(j["table_name"]) == TEST_TABLE]
        assert len(our_jobs) >= 1

        job = our_jobs[0]
        assert str(job["operation"]) in ("INSERT", "UPDATE")

        # Simulate worker: fetch text, generate embedding, write it
        row_id = str(job["row_id"])
        text = fetch_row_text(
            db_conn, TEST_TABLE, "description", row_id
        )
        assert text is not None

        embedding = [0.5] * DIMENSIONS
        update_embedding(db_conn, TEST_TABLE, row_id, embedding)
        db_conn.commit()

        # Complete the job
        complete_job(db_conn, int(str(job["id"])))

        # Verify embedding was written
        result = db_conn.execute(
            f'SELECT "embedding" FROM "{TEST_TABLE}" WHERE id = %s;',
            (int(row_id),),
        ).fetchone()
        assert result is not None
        assert result["embedding"] is not None

        # Clean up
        db_conn.execute(f"DELETE FROM {TEST_TABLE} WHERE id = 201;")
        db_conn.commit()


class TestFailAndRetry:
    def test_fail_job_with_retry(
        self, db_conn: psycopg.Connection, worker_ready_table: None
    ) -> None:
        """Failed job should be requeued for retry when retries < max."""
        # Insert to create a queue entry
        db_conn.execute(
            f"INSERT INTO {TEST_TABLE} (id, name, description) "
            f"VALUES (202, 'Fail Test', 'Testing failure and retry.');",
        )
        db_conn.commit()

        # Claim
        jobs = claim_batch(db_conn, batch_size=10)
        our_jobs = [j for j in jobs if str(j["table_name"]) == TEST_TABLE]
        assert len(our_jobs) >= 1

        job = our_jobs[0]
        job_id = int(str(job["id"]))

        # Fail the job (should re-queue since retries < max_retries=3)
        fail_job(db_conn, job_id, "test error", max_retries=3)

        # Should be back to pending (retries < max)
        result = db_conn.execute(
            "SELECT status, retries FROM pgvector_setup_queue WHERE id = %s;",
            (job_id,),
        ).fetchone()
        assert result is not None
        assert str(result["status"]) == "pending"
        assert int(str(result["retries"])) == 1

        # Clean up
        db_conn.execute(
            "DELETE FROM pgvector_setup_queue WHERE id = %s;", (job_id,)
        )
        db_conn.execute(f"DELETE FROM {TEST_TABLE} WHERE id = 202;")
        db_conn.commit()


class TestUpdateTrigger:
    def test_update_requeues_row(
        self, db_conn: psycopg.Connection, worker_ready_table: None
    ) -> None:
        """UPDATE on the watched column should create a new queue entry."""
        # Insert and claim/complete the initial job
        db_conn.execute(
            f"INSERT INTO {TEST_TABLE} (id, name, description) "
            f"VALUES (203, 'Update Test', 'Original description.');",
        )
        db_conn.commit()

        # Drain the INSERT job
        jobs = claim_batch(db_conn, batch_size=10)
        for job in jobs:
            if str(job["table_name"]) == TEST_TABLE:
                complete_job(db_conn, int(str(job["id"])))

        # Now update the description
        db_conn.execute(
            f"UPDATE {TEST_TABLE} SET description = 'Updated description.' "
            f"WHERE id = 203;",
        )
        db_conn.commit()

        # Should have a new pending entry
        pending = count_pending(db_conn, TEST_TABLE)
        assert pending >= 1

        # Clean up
        db_conn.execute(
            "DELETE FROM pgvector_setup_queue WHERE table_name = %s;",
            (TEST_TABLE,),
        )
        db_conn.execute(f"DELETE FROM {TEST_TABLE} WHERE id = 203;")
        db_conn.commit()

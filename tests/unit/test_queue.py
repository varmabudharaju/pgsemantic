"""Tests for queue module -- job claim/complete/fail logic with mocked DB."""
from unittest.mock import MagicMock

from pgsemantic.db.queue import (
    claim_batch,
    complete_job,
    count_failed,
    count_pending,
    create_queue_table,
    create_worker_health_table,
    delete_worker_health,
    fail_job,
    get_worker_health,
    retry_failed_jobs,
    upsert_worker_heartbeat,
)


class TestCreateQueueTable:
    def test_executes_create_table_and_index(self) -> None:
        mock_conn = MagicMock()
        create_queue_table(mock_conn)
        calls = mock_conn.execute.call_args_list
        assert len(calls) == 2
        # Verify both SQL constants are used
        first_sql = calls[0].args[0]
        second_sql = calls[1].args[0]
        assert "pgvector_setup_queue" in first_sql
        assert "idx_pgvs_queue_pending" in second_sql
        mock_conn.commit.assert_called_once()


class TestClaimBatch:
    def test_returns_jobs(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            {"id": 1, "table_name": "products", "row_id": "42",
             "column_name": "description", "operation": "INSERT"},
        ]
        jobs = claim_batch(mock_conn, batch_size=10)
        assert len(jobs) == 1
        assert jobs[0]["table_name"] == "products"
        mock_conn.commit.assert_called_once()

    def test_empty_queue(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        jobs = claim_batch(mock_conn, batch_size=10)
        assert jobs == []

    def test_batch_size_passed_as_param(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        claim_batch(mock_conn, batch_size=25)
        args = mock_conn.execute.call_args
        assert args[0][1] == {"batch_size": 25}

    def test_multiple_jobs_returned(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            {"id": 1, "table_name": "products", "row_id": "10",
             "column_name": "description", "operation": "INSERT"},
            {"id": 2, "table_name": "products", "row_id": "11",
             "column_name": "description", "operation": "UPDATE"},
            {"id": 3, "table_name": "articles", "row_id": "5",
             "column_name": "body", "operation": "INSERT"},
        ]
        jobs = claim_batch(mock_conn, batch_size=10)
        assert len(jobs) == 3
        assert jobs[2]["table_name"] == "articles"


class TestCompleteJob:
    def test_deletes_job(self) -> None:
        mock_conn = MagicMock()
        complete_job(mock_conn, job_id=42)
        mock_conn.execute.assert_called_once()
        args = mock_conn.execute.call_args
        assert args[0][1] == {"job_id": 42}
        mock_conn.commit.assert_called_once()


class TestFailJob:
    def test_marks_failed(self) -> None:
        mock_conn = MagicMock()
        fail_job(mock_conn, job_id=42, error_msg="timeout")
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    def test_passes_error_msg_and_max_retries(self) -> None:
        mock_conn = MagicMock()
        fail_job(mock_conn, job_id=7, error_msg="connection lost", max_retries=5)
        args = mock_conn.execute.call_args
        params = args[0][1]
        assert params["job_id"] == 7
        assert params["error_msg"] == "connection lost"
        assert params["max_retries"] == 5


class TestCountPending:
    def test_returns_count(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"cnt": 15}
        result = count_pending(mock_conn, table_name="products")
        assert result == 15

    def test_returns_zero_when_no_result(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        result = count_pending(mock_conn, table_name="products")
        assert result == 0


class TestCountFailed:
    def test_returns_count(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"cnt": 3}
        result = count_failed(mock_conn, table_name="products")
        assert result == 3

    def test_returns_zero_when_no_result(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        result = count_failed(mock_conn, table_name="products")
        assert result == 0


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
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("relation does not exist")
        result = get_worker_health(mock_conn)
        assert result == []


class TestDeleteWorkerHealth:
    def test_deletes_worker(self) -> None:
        mock_conn = MagicMock()
        delete_worker_health(mock_conn, worker_id="mac-1234")
        args = mock_conn.execute.call_args
        assert args[0][1] == {"worker_id": "mac-1234"}
        mock_conn.commit.assert_called_once()

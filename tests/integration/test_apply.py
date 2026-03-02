"""Integration tests for the apply workflow.

Tests the full setup flow: pgvector extension, vector column, HNSW index,
queue table, trigger function, and trigger installation.
"""
from __future__ import annotations

import psycopg
import pytest
from psycopg.rows import dict_row

from pgsemantic.db.client import ensure_pgvector_extension, get_connection
from pgsemantic.db.introspect import column_exists, get_primary_key_columns
from pgsemantic.db.queue import create_queue_table

from .conftest import TEST_TABLE

pytestmark = pytest.mark.integration


class TestPgvectorExtension:
    def test_extension_is_available(self, db_conn: psycopg.Connection) -> None:
        """pgvector extension should be installed and return a version."""
        version = ensure_pgvector_extension(db_conn)
        assert version is not None
        assert len(version) == 3
        assert version[0] >= 0

    def test_extension_version_is_reasonable(
        self, db_conn: psycopg.Connection
    ) -> None:
        result = db_conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
        ).fetchone()
        assert result is not None
        version_str = str(result["extversion"])
        parts = version_str.split(".")
        assert len(parts) >= 2


class TestApplyVectorColumn:
    def test_add_embedding_column(
        self, db_url: str, db_conn: psycopg.Connection
    ) -> None:
        """Adding an embedding column should work and be detectable."""
        # Skip if column already exists (from a previous test or run)
        if column_exists(db_conn, TEST_TABLE, "embedding"):
            return

        db_conn.execute(
            f'ALTER TABLE "{TEST_TABLE}" ADD COLUMN "embedding" vector(384);'
        )
        db_conn.commit()

        assert column_exists(db_conn, TEST_TABLE, "embedding")

    def test_primary_key_detection(
        self, db_conn: psycopg.Connection
    ) -> None:
        """Should detect the 'id' primary key column."""
        pk_cols = get_primary_key_columns(db_conn, TEST_TABLE)
        assert pk_cols == ["id"]

    def test_embedding_column_accepts_vectors(
        self, db_url: str
    ) -> None:
        """Embedding column should accept vector data."""
        with get_connection(db_url) as conn:
            # Ensure column exists
            if not column_exists(conn, TEST_TABLE, "embedding"):
                conn.execute(
                    f'ALTER TABLE "{TEST_TABLE}" '
                    f'ADD COLUMN "embedding" vector(384);'
                )
                conn.commit()

            # Write a test vector
            test_vector = [0.1] * 384
            conn.execute(
                f'UPDATE "{TEST_TABLE}" SET "embedding" = %s::vector WHERE id = 1;',
                (str(test_vector),),
            )
            conn.commit()

            # Read it back
            result = conn.execute(
                f'SELECT "embedding" FROM "{TEST_TABLE}" WHERE id = 1;'
            ).fetchone()
            assert result is not None
            assert result["embedding"] is not None


class TestQueueTable:
    def test_create_queue_table(
        self, db_conn: psycopg.Connection
    ) -> None:
        """Queue table should be created idempotently."""
        create_queue_table(db_conn)

        result = db_conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = 'pgvector_setup_queue';"
        ).fetchone()
        assert result is not None

    def test_queue_table_has_expected_columns(
        self, db_conn: psycopg.Connection
    ) -> None:
        """Queue table should have all required columns."""
        create_queue_table(db_conn)

        result = db_conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'pgvector_setup_queue' "
            "ORDER BY ordinal_position;"
        ).fetchall()
        col_names = [str(row["column_name"]) for row in result]
        expected = [
            "id", "table_name", "row_id", "column_name", "operation",
            "status", "retries", "error_msg", "created_at", "updated_at",
        ]
        for exp in expected:
            assert exp in col_names, f"Missing column: {exp}"


class TestHnswIndex:
    def test_create_hnsw_index(self, db_url: str) -> None:
        """HNSW index should be created CONCURRENTLY."""
        # Ensure embedding column exists first
        with get_connection(db_url) as conn:
            if not column_exists(conn, TEST_TABLE, "embedding"):
                conn.execute(
                    f'ALTER TABLE "{TEST_TABLE}" '
                    f'ADD COLUMN "embedding" vector(384);'
                )
                conn.commit()

        # CREATE INDEX CONCURRENTLY requires autocommit
        with psycopg.connect(db_url, autocommit=True) as conn:
            conn.execute(
                f'CREATE INDEX CONCURRENTLY IF NOT EXISTS '
                f'"idx_{TEST_TABLE}_embedding_hnsw" '
                f'ON "{TEST_TABLE}" '
                f'USING hnsw ("embedding" vector_cosine_ops) '
                f'WITH (m = 16, ef_construction = 64);'
            )

        # Verify index exists
        with get_connection(db_url) as conn:
            result = conn.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = %s AND indexname LIKE '%%hnsw%%';",
                (TEST_TABLE,),
            ).fetchone()
            assert result is not None


class TestTrigger:
    def test_trigger_function_and_trigger(
        self, db_url: str
    ) -> None:
        """Trigger should enqueue jobs on INSERT/UPDATE."""
        with get_connection(db_url) as conn:
            # Ensure prerequisites
            if not column_exists(conn, TEST_TABLE, "embedding"):
                conn.execute(
                    f'ALTER TABLE "{TEST_TABLE}" '
                    f'ADD COLUMN "embedding" vector(384);'
                )
                conn.commit()
            create_queue_table(conn)

            # Create trigger function
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
            conn.execute(trigger_fn)
            conn.commit()

            # Install trigger
            trigger_sql = f"""
            CREATE OR REPLACE TRIGGER pgvector_setup_{TEST_TABLE}_trigger
            AFTER INSERT OR UPDATE OF "description" OR DELETE
            ON "{TEST_TABLE}"
            FOR EACH ROW
            EXECUTE FUNCTION pgvector_setup_{TEST_TABLE}_fn('description');
            """
            conn.execute(trigger_sql)
            conn.commit()

            # Clean queue for our test table
            conn.execute(
                "DELETE FROM pgvector_setup_queue WHERE table_name = %s;",
                (TEST_TABLE,),
            )
            conn.commit()

            # Insert a new row — should create a queue entry
            conn.execute(
                f"INSERT INTO {TEST_TABLE} (id, name, description) "
                f"VALUES (100, 'Test Product', 'A test product for integration testing.');",
            )
            conn.commit()

            # Check queue
            result = conn.execute(
                "SELECT * FROM pgvector_setup_queue "
                "WHERE table_name = %s AND row_id = '100';",
                (TEST_TABLE,),
            ).fetchone()
            assert result is not None
            assert str(result["operation"]) == "INSERT"
            assert str(result["status"]) == "pending"

            # Clean up the test row
            conn.execute(f"DELETE FROM {TEST_TABLE} WHERE id = 100;")
            conn.commit()

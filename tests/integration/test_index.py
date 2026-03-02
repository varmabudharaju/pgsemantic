"""Integration tests for bulk embedding (index command logic).

Tests that embeddings are correctly generated and written to the database.
"""
from __future__ import annotations

import psycopg
import pytest

from pgsemantic.db.introspect import column_exists
from pgsemantic.db.vectors import (
    bulk_update_embeddings,
    count_embedded,
    count_total_with_content,
    fetch_unembedded_batch,
)

from .conftest import TEST_TABLE, TEST_ROWS

pytestmark = pytest.mark.integration

DIMENSIONS = 384


@pytest.fixture()
def table_with_embedding(db_url: str, db_conn: psycopg.Connection) -> None:
    """Ensure the test table has an embedding column and clear existing embeddings."""
    if not column_exists(db_conn, TEST_TABLE, "embedding"):
        db_conn.execute(
            f'ALTER TABLE "{TEST_TABLE}" ADD COLUMN "embedding" vector({DIMENSIONS});'
        )
        db_conn.commit()

    # Clear all embeddings so tests start fresh
    db_conn.execute(f'UPDATE "{TEST_TABLE}" SET "embedding" = NULL;')
    db_conn.commit()


class TestFetchUnembedded:
    def test_fetch_all_unembedded(
        self, db_conn: psycopg.Connection, table_with_embedding: None
    ) -> None:
        """Should return all rows when none have embeddings."""
        rows = fetch_unembedded_batch(
            db_conn, TEST_TABLE, "description", batch_size=100
        )
        assert len(rows) == len(TEST_ROWS)

    def test_fetch_respects_batch_size(
        self, db_conn: psycopg.Connection, table_with_embedding: None
    ) -> None:
        """Should return at most batch_size rows."""
        rows = fetch_unembedded_batch(
            db_conn, TEST_TABLE, "description", batch_size=3
        )
        assert len(rows) == 3

    def test_keyset_pagination(
        self, db_conn: psycopg.Connection, table_with_embedding: None
    ) -> None:
        """Subsequent batches should not overlap."""
        batch1 = fetch_unembedded_batch(
            db_conn, TEST_TABLE, "description", batch_size=5
        )
        assert len(batch1) == 5

        # Get last PK for keyset pagination
        last_params = {"last_pk": batch1[-1]["id"]}
        batch2 = fetch_unembedded_batch(
            db_conn, TEST_TABLE, "description", batch_size=5,
            last_pk_params=last_params,
        )
        assert len(batch2) == 5

        # No overlap
        ids1 = {row["id"] for row in batch1}
        ids2 = {row["id"] for row in batch2}
        assert ids1.isdisjoint(ids2)


class TestBulkUpdateEmbeddings:
    def test_writes_embeddings(
        self, db_conn: psycopg.Connection, table_with_embedding: None
    ) -> None:
        """Bulk update should write embeddings and they should be readable."""
        rows = fetch_unembedded_batch(
            db_conn, TEST_TABLE, "description", batch_size=5
        )
        embeddings = [[0.1] * DIMENSIONS for _ in rows]

        bulk_update_embeddings(
            db_conn, TEST_TABLE, rows, embeddings
        )

        embedded_count = count_embedded(db_conn, TEST_TABLE)
        assert embedded_count == 5

    def test_count_total_with_content(
        self, db_conn: psycopg.Connection, table_with_embedding: None
    ) -> None:
        """Should count all rows with non-NULL description."""
        total = count_total_with_content(db_conn, TEST_TABLE, "description")
        assert total == len(TEST_ROWS)


class TestEmbeddingWithMockedModel:
    def test_embed_and_store(
        self,
        db_conn: psycopg.Connection,
        table_with_embedding: None,
    ) -> None:
        """Generate mock embeddings and store them in the database."""
        rows = fetch_unembedded_batch(
            db_conn, TEST_TABLE, "description", batch_size=3
        )
        assert len(rows) > 0

        # Use simple deterministic vectors instead of a real model
        embeddings = [[0.1 * (i + 1)] * DIMENSIONS for i in range(len(rows))]

        bulk_update_embeddings(db_conn, TEST_TABLE, rows, embeddings)

        embedded = count_embedded(db_conn, TEST_TABLE)
        assert embedded == len(rows)

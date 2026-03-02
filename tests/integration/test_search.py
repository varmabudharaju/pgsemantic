"""Integration tests for semantic search and hybrid search.

Tests that search queries return correctly ranked results from the database.
"""
from __future__ import annotations

import psycopg
import pytest

from pgsemantic.db.client import get_connection, get_pgvector_version
from pgsemantic.db.introspect import column_exists
from pgsemantic.db.vectors import (
    bulk_update_embeddings,
    fetch_unembedded_batch,
    hybrid_search,
    search_similar,
)
from pgsemantic.exceptions import InvalidFilterError

from .conftest import TEST_TABLE, TEST_ROWS

pytestmark = pytest.mark.integration

DIMENSIONS = 384


@pytest.fixture()
def table_with_embeddings(
    db_url: str, db_conn: psycopg.Connection
) -> None:
    """Ensure all rows have embeddings for search tests."""
    if not column_exists(db_conn, TEST_TABLE, "embedding"):
        db_conn.execute(
            f'ALTER TABLE "{TEST_TABLE}" ADD COLUMN "embedding" vector({DIMENSIONS});'
        )
        db_conn.commit()

    # Give each row a directionally distinct embedding.
    # Cosine similarity measures direction, not magnitude, so uniform vectors
    # [c, c, c, ...] all point the same direction regardless of c.
    # Instead, give row N a "hot" dimension at index N to make them distinguishable.
    for row_id, _, _ in TEST_ROWS:
        vec = [0.01] * DIMENSIONS
        # Set a unique "hot" dimension based on row_id
        vec[row_id % DIMENSIONS] = 1.0
        db_conn.execute(
            f'UPDATE "{TEST_TABLE}" SET "embedding" = %s::vector WHERE id = %s;',
            (str(vec), row_id),
        )
    db_conn.commit()


class TestSemanticSearch:
    def test_returns_results(
        self, db_conn: psycopg.Connection, table_with_embeddings: None
    ) -> None:
        """Semantic search should return results with similarity scores."""
        query_vector = [0.1] * DIMENSIONS
        results = search_similar(
            db_conn, TEST_TABLE, "description", query_vector, limit=5
        )
        assert len(results) > 0
        assert len(results) <= 5

    def test_results_have_similarity(
        self, db_conn: psycopg.Connection, table_with_embeddings: None
    ) -> None:
        """Each result should have a similarity score."""
        query_vector = [0.1] * DIMENSIONS
        results = search_similar(
            db_conn, TEST_TABLE, "description", query_vector, limit=3
        )
        for row in results:
            assert "similarity" in row
            sim = float(str(row["similarity"]))
            assert 0.0 <= sim <= 1.0

    def test_results_are_ordered_by_similarity(
        self, db_conn: psycopg.Connection, table_with_embeddings: None
    ) -> None:
        """Results should be ordered by descending similarity."""
        query_vector = [0.1] * DIMENSIONS
        results = search_similar(
            db_conn, TEST_TABLE, "description", query_vector, limit=10
        )
        similarities = [float(str(row["similarity"])) for row in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_most_similar_result(
        self, db_conn: psycopg.Connection, table_with_embeddings: None
    ) -> None:
        """Row 1 (hot dim at index 1) should be most similar to a query hot at index 1."""
        # Build a query vector that matches row 1's pattern (hot at index 1)
        query_vector = [0.01] * DIMENSIONS
        query_vector[1] = 1.0
        results = search_similar(
            db_conn, TEST_TABLE, "description", query_vector, limit=1
        )
        assert len(results) == 1
        assert int(str(results[0]["id"])) == 1

    def test_limit_respected(
        self, db_conn: psycopg.Connection, table_with_embeddings: None
    ) -> None:
        """Should never return more than limit results."""
        query_vector = [0.1] * DIMENSIONS
        results = search_similar(
            db_conn, TEST_TABLE, "description", query_vector, limit=2
        )
        assert len(results) == 2

    def test_results_contain_source_content(
        self, db_conn: psycopg.Connection, table_with_embeddings: None
    ) -> None:
        """Results should include the source text columns."""
        query_vector = [0.1] * DIMENSIONS
        results = search_similar(
            db_conn, TEST_TABLE, "description", query_vector, limit=1
        )
        row = results[0]
        assert "name" in row
        assert "description" in row or "content" in row


class TestHybridSearch:
    def test_returns_results(
        self, db_conn: psycopg.Connection, table_with_embeddings: None
    ) -> None:
        """Hybrid search should return results."""
        query_vector = [0.1] * DIMENSIONS
        results = hybrid_search(
            db_conn, TEST_TABLE, "description", query_vector,
            filters={}, limit=5,
        )
        assert len(results) > 0

    def test_exact_filter(
        self, db_conn: psycopg.Connection, table_with_embeddings: None
    ) -> None:
        """Exact column filter should restrict results."""
        query_vector = [0.1] * DIMENSIONS
        results = hybrid_search(
            db_conn, TEST_TABLE, "description", query_vector,
            filters={"name": "Wireless Headphones"},
            limit=10,
        )
        for row in results:
            assert str(row["name"]) == "Wireless Headphones"

    def test_range_filter_max(
        self, db_conn: psycopg.Connection, table_with_embeddings: None
    ) -> None:
        """Range filter with _max suffix should work."""
        query_vector = [0.1] * DIMENSIONS
        results = hybrid_search(
            db_conn, TEST_TABLE, "description", query_vector,
            filters={"id_max": 3},
            limit=10,
        )
        for row in results:
            assert int(str(row["id"])) <= 3

    def test_range_filter_min(
        self, db_conn: psycopg.Connection, table_with_embeddings: None
    ) -> None:
        """Range filter with _min suffix should work."""
        query_vector = [0.1] * DIMENSIONS
        results = hybrid_search(
            db_conn, TEST_TABLE, "description", query_vector,
            filters={"id_min": 8},
            limit=10,
        )
        for row in results:
            assert int(str(row["id"])) >= 8

    def test_invalid_filter_raises(
        self, db_conn: psycopg.Connection, table_with_embeddings: None
    ) -> None:
        """Invalid filter column should raise InvalidFilterError."""
        query_vector = [0.1] * DIMENSIONS
        with pytest.raises(InvalidFilterError):
            hybrid_search(
                db_conn, TEST_TABLE, "description", query_vector,
                filters={"nonexistent_column": "value"},
                limit=5,
            )

    def test_empty_filters_same_as_semantic(
        self, db_conn: psycopg.Connection, table_with_embeddings: None
    ) -> None:
        """Hybrid search with empty filters should behave like semantic search."""
        query_vector = [0.1] * DIMENSIONS
        semantic = search_similar(
            db_conn, TEST_TABLE, "description", query_vector, limit=5
        )
        hybrid = hybrid_search(
            db_conn, TEST_TABLE, "description", query_vector,
            filters={}, limit=5,
        )
        # Same row IDs in the same order
        semantic_ids = [row["id"] for row in semantic]
        hybrid_ids = [row["id"] for row in hybrid]
        assert semantic_ids == hybrid_ids

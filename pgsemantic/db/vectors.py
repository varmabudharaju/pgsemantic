"""Vector operations for pgsemantic.

Handles embedding storage, retrieval, and similarity search. Uses keyset
pagination for consistent bulk-fetch performance. Hybrid search validates
filter columns against information_schema before building parameterized SQL.

All SQL is defined as module-level constants. Table/column names are quoted
with double quotes to prevent injection; values use %(name)s placeholders.
"""
from __future__ import annotations

import logging

import psycopg

from pgsemantic.db.introspect import get_column_names
from pgsemantic.exceptions import InvalidFilterError

logger = logging.getLogger(__name__)

# Type alias matching client.py — a psycopg connection returning dict rows.
DictConnection = psycopg.Connection[dict[str, object]]

# -- SQL Constants ----------------------------------------------------------

# Fetch rows that have source text but no embedding yet, using keyset
# pagination (WHERE id > last_id) for consistent performance regardless
# of how many rows have already been processed.
SQL_FETCH_UNEMBEDDED = """
    SELECT "id", {column}
    FROM {table}
    WHERE {column} IS NOT NULL
      AND "embedding" IS NULL
      AND "id" > %(last_id)s
    ORDER BY "id" ASC
    LIMIT %(batch_size)s;
"""

# Update a single row's embedding vector.
SQL_UPDATE_EMBEDDING = """
    UPDATE {table}
    SET "embedding" = %(embedding)s
    WHERE "id" = %(row_id)s;
"""

# Count rows that have a non-NULL embedding.
SQL_COUNT_EMBEDDED = """
    SELECT COUNT(*) AS cnt
    FROM {table}
    WHERE "embedding" IS NOT NULL;
"""

# Count rows that have non-NULL source text (the total that should be embedded).
SQL_COUNT_TOTAL_WITH_CONTENT = """
    SELECT COUNT(*) AS cnt
    FROM {table}
    WHERE {column} IS NOT NULL;
"""

# Cosine similarity search using pgvector's <=> operator.
# Returns rows ordered by similarity (descending), with similarity = 1 - distance.
SQL_SEARCH_SIMILAR = """
    SELECT
        "id",
        {column} AS content,
        1 - ("embedding" <=> %(query_vector)s::vector) AS similarity
    FROM {table}
    WHERE "embedding" IS NOT NULL
    ORDER BY "embedding" <=> %(query_vector)s::vector
    LIMIT %(limit)s;
"""

# Hybrid search with pre-filtering. Dynamic WHERE clauses are appended as
# parameterized conditions. Uses hnsw.iterative_scan on pgvector >= 0.8.0
# to avoid recall degradation with selective filters.
SQL_HYBRID_SEARCH_BASE = """
    SELECT
        "id",
        {column} AS content,
        1 - ("embedding" <=> %(query_vector)s::vector) AS similarity
    FROM {table}
    WHERE "embedding" IS NOT NULL
"""

# Enable iterative scan for pre-filtered HNSW queries (pgvector >= 0.8.0).
SQL_SET_ITERATIVE_SCAN = """
    SET LOCAL hnsw.iterative_scan = relaxed_order;
"""

# Fetch the source text for a single row (used by the worker to get text
# before embedding).
SQL_FETCH_ROW_TEXT = """
    SELECT {column}
    FROM {table}
    WHERE "id" = %(row_id)s;
"""

# Set a row's embedding to NULL (used for DELETE events to clear stale data).
SQL_NULL_EMBEDDING = """
    UPDATE {table}
    SET "embedding" = NULL
    WHERE "id" = %(row_id)s;
"""


# -- Helper -----------------------------------------------------------------


def _qualified_table(table: str, schema: str) -> str:
    """Build a double-quoted schema.table identifier."""
    return f'"{schema}"."{table}"'


def _quoted(name: str) -> str:
    """Double-quote a single identifier (column name, etc.)."""
    return f'"{name}"'


# -- Functions --------------------------------------------------------------


def fetch_unembedded_batch(
    conn: DictConnection,
    table: str,
    column: str,
    batch_size: int,
    last_id: int = 0,
    schema: str = "public",
) -> list[dict[str, object]]:
    """Fetch rows with source text but no embedding, using keyset pagination.

    Returns a list of dicts with 'id' and the source column value.
    Keyset pagination (WHERE id > last_id ORDER BY id LIMIT N) ensures
    consistent performance regardless of offset depth.
    """
    qualified = _qualified_table(table, schema)
    quoted_col = _quoted(column)
    sql = SQL_FETCH_UNEMBEDDED.format(table=qualified, column=quoted_col)
    result = conn.execute(
        sql, {"last_id": last_id, "batch_size": batch_size}
    ).fetchall()
    return [dict(row) for row in result]


def update_embedding(
    conn: DictConnection,
    table: str,
    row_id: int,
    embedding: list[float],
    schema: str = "public",
) -> None:
    """Update a single row's embedding vector."""
    qualified = _qualified_table(table, schema)
    sql = SQL_UPDATE_EMBEDDING.format(table=qualified)
    conn.execute(sql, {"row_id": row_id, "embedding": embedding})


def bulk_update_embeddings(
    conn: DictConnection,
    table: str,
    rows: list[dict[str, object]],
    embeddings: list[list[float]],
    schema: str = "public",
) -> None:
    """Update embeddings for multiple rows in a single transaction.

    Each row dict must have an 'id' key. The embeddings list must be the
    same length as the rows list. Commits after all updates.
    """
    if len(rows) != len(embeddings):
        raise ValueError(
            f"rows ({len(rows)}) and embeddings ({len(embeddings)}) "
            "must have the same length"
        )

    qualified = _qualified_table(table, schema)
    sql = SQL_UPDATE_EMBEDDING.format(table=qualified)

    for row, embedding in zip(rows, embeddings, strict=True):
        row_id = row["id"]
        conn.execute(sql, {"row_id": row_id, "embedding": embedding})

    conn.commit()
    logger.debug("Bulk updated %d embeddings in %s", len(rows), table)


def count_embedded(
    conn: DictConnection,
    table: str,
    schema: str = "public",
) -> int:
    """Count rows that have a non-NULL embedding."""
    qualified = _qualified_table(table, schema)
    sql = SQL_COUNT_EMBEDDED.format(table=qualified)
    result = conn.execute(sql).fetchone()
    if result is None:
        return 0
    return int(str(result["cnt"]))


def count_total_with_content(
    conn: DictConnection,
    table: str,
    column: str,
    schema: str = "public",
) -> int:
    """Count rows that have non-NULL source text."""
    qualified = _qualified_table(table, schema)
    quoted_col = _quoted(column)
    sql = SQL_COUNT_TOTAL_WITH_CONTENT.format(
        table=qualified, column=quoted_col
    )
    result = conn.execute(sql).fetchone()
    if result is None:
        return 0
    return int(str(result["cnt"]))


def search_similar(
    conn: DictConnection,
    table: str,
    column: str,
    query_vector: list[float],
    limit: int = 10,
    schema: str = "public",
) -> list[dict[str, object]]:
    """Find rows semantically similar to a query vector using cosine similarity.

    Returns rows ordered by similarity (highest first), each containing
    id, content (the source column text), and similarity score.
    """
    qualified = _qualified_table(table, schema)
    quoted_col = _quoted(column)
    sql = SQL_SEARCH_SIMILAR.format(table=qualified, column=quoted_col)
    result = conn.execute(
        sql, {"query_vector": query_vector, "limit": limit}
    ).fetchall()
    return [dict(row) for row in result]


def hybrid_search(
    conn: DictConnection,
    table: str,
    column: str,
    query_vector: list[float],
    filters: dict[str, object],
    limit: int = 10,
    pgvector_version: tuple[int, int, int] | None = None,
    schema: str = "public",
) -> list[dict[str, object]]:
    """Semantic similarity + SQL WHERE filters in one query.

    Validates filter keys against information_schema columns before building
    parameterized SQL. Sets hnsw.iterative_scan = relaxed_order on pgvector
    >= 0.8.0 to avoid recall degradation with selective pre-filters.

    Raises InvalidFilterError if a filter key doesn't match a table column.
    """
    # Validate filter keys against actual table columns.
    # Strip _min/_max suffixes before checking, since those are comparison operators
    # applied to the base column name (e.g., price_max → price).
    if filters:
        valid_columns = get_column_names(conn, table, schema)
        for key in filters:
            actual_col = key
            if key.endswith("_min") or key.endswith("_max"):
                actual_col = key[:-4]
            if actual_col not in valid_columns:
                raise InvalidFilterError(
                    f"Invalid filter column: '{actual_col}' (from filter key '{key}'). "
                    f"Valid columns: {', '.join(sorted(valid_columns))}"
                )

    # Enable iterative scan for pgvector >= 0.8.0
    if pgvector_version is not None and pgvector_version >= (0, 8, 0):
        conn.execute(SQL_SET_ITERATIVE_SCAN)

    # Build the query with dynamic WHERE clauses
    qualified = _qualified_table(table, schema)
    quoted_col = _quoted(column)
    sql = SQL_HYBRID_SEARCH_BASE.format(table=qualified, column=quoted_col)

    params: dict[str, object] = {
        "query_vector": query_vector,
        "limit": limit,
    }

    # Append parameterized filter conditions
    for key, value in filters.items():
        quoted_key = _quoted(key)
        # Handle special suffixes for comparison operators
        if key.endswith("_min"):
            actual_col = key[:-4]
            quoted_key = _quoted(actual_col)
            sql += f"  AND {quoted_key} >= %({key})s\n"
        elif key.endswith("_max"):
            actual_col = key[:-4]
            quoted_key = _quoted(actual_col)
            sql += f"  AND {quoted_key} <= %({key})s\n"
        else:
            sql += f"  AND {quoted_key} = %({key})s\n"
        params[key] = value

    sql += '    ORDER BY "embedding" <=> %(query_vector)s::vector\n'
    sql += "    LIMIT %(limit)s;"

    result = conn.execute(sql, params).fetchall()
    return [dict(row) for row in result]


def fetch_row_text(
    conn: DictConnection,
    table: str,
    column: str,
    row_id: str,
    schema: str = "public",
) -> str | None:
    """Fetch the source text for a single row (used by the worker).

    Returns the text value or None if the row doesn't exist or the
    column is NULL.
    """
    qualified = _qualified_table(table, schema)
    quoted_col = _quoted(column)
    sql = SQL_FETCH_ROW_TEXT.format(table=qualified, column=quoted_col)
    result = conn.execute(sql, {"row_id": row_id}).fetchone()
    if result is None:
        return None
    value = result[column]
    if value is None:
        return None
    return str(value)


def null_embedding(
    conn: DictConnection,
    table: str,
    row_id: str,
    schema: str = "public",
) -> None:
    """Set a row's embedding to NULL (used for DELETE events)."""
    qualified = _qualified_table(table, schema)
    sql = SQL_NULL_EMBEDDING.format(table=qualified)
    conn.execute(sql, {"row_id": row_id})
    conn.commit()
    logger.debug("Nulled embedding for row %s in %s", row_id, table)

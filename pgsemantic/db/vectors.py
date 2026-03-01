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
# pagination for consistent performance. PK columns are substituted at runtime.
# Two variants: one for the first page (no PK filter), one for subsequent pages.
SQL_FETCH_UNEMBEDDED_FIRST = """
    SELECT {pk_select}, {column}
    FROM {table}
    WHERE {column} IS NOT NULL
      AND "embedding" IS NULL
    ORDER BY {pk_order}
    LIMIT %(batch_size)s;
"""

SQL_FETCH_UNEMBEDDED_NEXT = """
    SELECT {pk_select}, {column}
    FROM {table}
    WHERE {column} IS NOT NULL
      AND "embedding" IS NULL
      AND ({pk_where_gt})
    ORDER BY {pk_order}
    LIMIT %(batch_size)s;
"""

# Update a single row's embedding vector. PK WHERE clause is substituted at runtime.
SQL_UPDATE_EMBEDDING_TEMPLATE = """
    UPDATE {table}
    SET "embedding" = %(embedding)s
    WHERE {pk_where_eq};
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
        *,
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
        *,
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
# before embedding). PK WHERE clause is substituted at runtime.
SQL_FETCH_ROW_TEXT_TEMPLATE = """
    SELECT {column}
    FROM {table}
    WHERE {pk_where_eq};
"""

# Set a row's embedding to NULL (used for DELETE events to clear stale data).
SQL_NULL_EMBEDDING_TEMPLATE = """
    UPDATE {table}
    SET "embedding" = NULL
    WHERE {pk_where_eq};
"""


# -- Helper -----------------------------------------------------------------


def _qualified_table(table: str, schema: str) -> str:
    """Build a double-quoted schema.table identifier."""
    return f'"{schema}"."{table}"'


def _quoted(name: str) -> str:
    """Double-quote a single identifier (column name, etc.)."""
    return f'"{name}"'


def _pk_select(pk_columns: list[str]) -> str:
    """Build SELECT list for PK columns, e.g. '"id"' or '"org_id", "user_id"'."""
    return ", ".join(_quoted(c) for c in pk_columns)


def _pk_order(pk_columns: list[str]) -> str:
    """Build ORDER BY clause for PK columns."""
    return ", ".join(f"{_quoted(c)} ASC" for c in pk_columns)


def _pk_where_eq(pk_columns: list[str]) -> str:
    """Build WHERE clause for PK equality, e.g. '"id" = %(pk_id)s'."""
    if len(pk_columns) == 1:
        return f'{_quoted(pk_columns[0])} = %(pk_val)s'
    return " AND ".join(
        f'{_quoted(c)} = %(pk_{c})s' for c in pk_columns
    )


def _pk_where_gt(pk_columns: list[str]) -> str:
    """Build WHERE clause for keyset pagination (row > last_row)."""
    if len(pk_columns) == 1:
        return f'{_quoted(pk_columns[0])} > %(last_pk)s'
    cols = ", ".join(_quoted(c) for c in pk_columns)
    params = ", ".join(f"%(last_pk_{c})s" for c in pk_columns)
    return f"ROW({cols}) > ROW({params})"


def _pk_params_from_row_id(pk_columns: list[str], row_id: str) -> dict[str, str]:
    """Parse a stringified row_id back into PK parameter dict.

    Single PK: "42" -> {"pk_val": "42"}
    Composite PK: "org1,user2" -> {"pk_org_id": "org1", "pk_user_id": "user2"}
    """
    if len(pk_columns) == 1:
        return {"pk_val": row_id}
    values = row_id.split(",")
    return {f"pk_{c}": v for c, v in zip(pk_columns, values, strict=True)}


def _pk_row_id(pk_columns: list[str], row: dict[str, object]) -> str:
    """Extract the stringified row_id from a row dict."""
    if len(pk_columns) == 1:
        return str(row[pk_columns[0]])
    return ",".join(str(row[c]) for c in pk_columns)


def _pk_last_params(pk_columns: list[str], row: dict[str, object]) -> dict[str, object]:
    """Build keyset pagination params from the last row in a batch."""
    if len(pk_columns) == 1:
        return {"last_pk": row[pk_columns[0]]}
    return {f"last_pk_{c}": row[c] for c in pk_columns}


def _pk_initial_params(pk_columns: list[str]) -> dict[str, object]:
    """Build initial keyset pagination params (start from beginning).

    Uses None to signal "no previous row" — the SQL WHERE clause
    handles this with IS NULL check.
    """
    if len(pk_columns) == 1:
        return {"last_pk": None}
    return {f"last_pk_{c}": None for c in pk_columns}


# -- Functions --------------------------------------------------------------


def fetch_unembedded_batch(
    conn: DictConnection,
    table: str,
    column: str,
    batch_size: int,
    pk_columns: list[str] | None = None,
    last_pk_params: dict[str, object] | None = None,
    schema: str = "public",
) -> list[dict[str, object]]:
    """Fetch rows with source text but no embedding, using keyset pagination.

    Returns a list of dicts with PK columns and the source column value.
    Keyset pagination ensures consistent performance regardless of offset depth.

    Pass last_pk_params=None for the first page (no PK filter).
    """
    if pk_columns is None:
        pk_columns = ["id"]

    qualified = _qualified_table(table, schema)
    quoted_col = _quoted(column)

    if last_pk_params is None:
        # First page — no PK filter
        sql = SQL_FETCH_UNEMBEDDED_FIRST.format(
            table=qualified,
            column=quoted_col,
            pk_select=_pk_select(pk_columns),
            pk_order=_pk_order(pk_columns),
        )
        params: dict[str, object] = {"batch_size": batch_size}
    else:
        # Subsequent pages — use keyset pagination
        sql = SQL_FETCH_UNEMBEDDED_NEXT.format(
            table=qualified,
            column=quoted_col,
            pk_select=_pk_select(pk_columns),
            pk_where_gt=_pk_where_gt(pk_columns),
            pk_order=_pk_order(pk_columns),
        )
        params = {"batch_size": batch_size, **last_pk_params}

    result = conn.execute(sql, params).fetchall()
    return [dict(row) for row in result]


def update_embedding(
    conn: DictConnection,
    table: str,
    row_id: str,
    embedding: list[float],
    pk_columns: list[str] | None = None,
    schema: str = "public",
) -> None:
    """Update a single row's embedding vector."""
    if pk_columns is None:
        pk_columns = ["id"]
    qualified = _qualified_table(table, schema)
    sql = SQL_UPDATE_EMBEDDING_TEMPLATE.format(
        table=qualified,
        pk_where_eq=_pk_where_eq(pk_columns),
    )
    params = {"embedding": embedding, **_pk_params_from_row_id(pk_columns, row_id)}
    conn.execute(sql, params)


def bulk_update_embeddings(
    conn: DictConnection,
    table: str,
    rows: list[dict[str, object]],
    embeddings: list[list[float]],
    pk_columns: list[str] | None = None,
    schema: str = "public",
) -> None:
    """Update embeddings for multiple rows in a single transaction.

    Each row dict must have the PK column(s). The embeddings list must be the
    same length as the rows list. Commits after all updates.
    """
    if pk_columns is None:
        pk_columns = ["id"]
    if len(rows) != len(embeddings):
        raise ValueError(
            f"rows ({len(rows)}) and embeddings ({len(embeddings)}) "
            "must have the same length"
        )

    qualified = _qualified_table(table, schema)
    sql = SQL_UPDATE_EMBEDDING_TEMPLATE.format(
        table=qualified,
        pk_where_eq=_pk_where_eq(pk_columns),
    )

    for row, embedding in zip(rows, embeddings, strict=True):
        row_id = _pk_row_id(pk_columns, row)
        params = {"embedding": embedding, **_pk_params_from_row_id(pk_columns, row_id)}
        conn.execute(sql, params)

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
    # Keys ending in _min/_max are range operators (e.g., price_max → price <= value),
    # UNLESS the key itself is a real column name (e.g., cook_time_min is a column).
    valid_columns: set[str] = set()
    if filters:
        valid_columns = get_column_names(conn, table, schema)
        for key in filters:
            actual_col = key
            # Only strip suffix if the raw key isn't itself a column
            if key not in valid_columns and (key.endswith("_min") or key.endswith("_max")):
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

    # Append parameterized filter conditions.
    # _min/_max suffixes become >= / <= operators, but only when the key
    # is NOT itself a real column name (avoids collision with columns
    # like "cook_time_min" which should be treated as exact match).
    for key, value in filters.items():
        is_range_op = (
            (key.endswith("_min") or key.endswith("_max"))
            and key not in valid_columns
        )
        if is_range_op and key.endswith("_min"):
            actual_col = key[:-4]
            sql += f"  AND {_quoted(actual_col)} >= %({key})s\n"
        elif is_range_op and key.endswith("_max"):
            actual_col = key[:-4]
            sql += f"  AND {_quoted(actual_col)} <= %({key})s\n"
        else:
            sql += f"  AND {_quoted(key)} = %({key})s\n"
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
    pk_columns: list[str] | None = None,
    schema: str = "public",
) -> str | None:
    """Fetch the source text for a single row (used by the worker).

    Returns the text value or None if the row doesn't exist or the
    column is NULL.
    """
    if pk_columns is None:
        pk_columns = ["id"]
    qualified = _qualified_table(table, schema)
    quoted_col = _quoted(column)
    sql = SQL_FETCH_ROW_TEXT_TEMPLATE.format(
        table=qualified,
        column=quoted_col,
        pk_where_eq=_pk_where_eq(pk_columns),
    )
    params = _pk_params_from_row_id(pk_columns, row_id)
    result = conn.execute(sql, params).fetchone()
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
    pk_columns: list[str] | None = None,
    schema: str = "public",
) -> None:
    """Set a row's embedding to NULL (used for DELETE events)."""
    if pk_columns is None:
        pk_columns = ["id"]
    qualified = _qualified_table(table, schema)
    sql = SQL_NULL_EMBEDDING_TEMPLATE.format(
        table=qualified,
        pk_where_eq=_pk_where_eq(pk_columns),
    )
    params = _pk_params_from_row_id(pk_columns, row_id)
    conn.execute(sql, params)
    conn.commit()
    logger.debug("Nulled embedding for row %s in %s", row_id, table)

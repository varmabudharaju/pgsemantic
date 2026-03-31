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

from pgsemantic.config import ProjectConfig
from pgsemantic.db.introspect import get_column_names
from pgsemantic.exceptions import InvalidFilterError

logger = logging.getLogger(__name__)

# Type alias matching client.py — a psycopg connection returning dict rows.
DictConnection = psycopg.Connection[dict[str, object]]

# -- SQL Constants (Inline mode — embeddings on source table) ---------------

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

# -- SQL Constants (External mode — embeddings in shadow table) -------------

# Create a shadow table that stores embeddings separately from the source table.
SQL_CREATE_SHADOW_TABLE = """
    CREATE TABLE IF NOT EXISTS {shadow_table} (
        row_id          TEXT            PRIMARY KEY,
        embedding       vector({dimensions}),
        source_column   TEXT            NOT NULL,
        model_name      TEXT            NOT NULL,
        updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
    );
"""

# Fetch rows that don't yet have a shadow-table embedding, using LEFT JOIN.
SQL_EXT_FETCH_UNEMBEDDED_FIRST = """
    SELECT {pk_select}, {content_expr}
    FROM {table} s
    LEFT JOIN {shadow_table} e ON {pk_cast_join}
    WHERE {not_null_check}
      AND e.row_id IS NULL
    ORDER BY {pk_order_prefixed}
    LIMIT %(batch_size)s;
"""

SQL_EXT_FETCH_UNEMBEDDED_NEXT = """
    SELECT {pk_select}, {content_expr}
    FROM {table} s
    LEFT JOIN {shadow_table} e ON {pk_cast_join}
    WHERE {not_null_check}
      AND e.row_id IS NULL
      AND ({pk_where_gt_prefixed})
    ORDER BY {pk_order_prefixed}
    LIMIT %(batch_size)s;
"""

# Upsert an embedding into the shadow table.
SQL_EXT_UPSERT_EMBEDDING = """
    INSERT INTO {shadow_table} (row_id, embedding, source_column, model_name, updated_at)
    VALUES (%(row_id)s, %(embedding)s::vector, %(source_column)s, %(model_name)s, NOW())
    ON CONFLICT (row_id)
    DO UPDATE SET
        embedding     = EXCLUDED.embedding,
        source_column = EXCLUDED.source_column,
        model_name    = EXCLUDED.model_name,
        updated_at    = NOW();
"""

# Count embedded rows in shadow table.
SQL_EXT_COUNT_EMBEDDED = """
    SELECT COUNT(*) AS cnt
    FROM {shadow_table}
    WHERE "embedding" IS NOT NULL;
"""

# Semantic search via JOIN between source and shadow table.
SQL_EXT_SEARCH_SIMILAR = """
    SELECT
        s.*,
        {content_expr} AS content,
        1 - (e.embedding <=> %(query_vector)s::vector) AS similarity
    FROM {table} s
    JOIN {shadow_table} e ON {pk_cast_join}
    WHERE e.embedding IS NOT NULL
    ORDER BY e.embedding <=> %(query_vector)s::vector
    LIMIT %(limit)s;
"""

# Hybrid search via JOIN between source and shadow table.
SQL_EXT_HYBRID_SEARCH_BASE = """
    SELECT
        s.*,
        {content_expr} AS content,
        1 - (e.embedding <=> %(query_vector)s::vector) AS similarity
    FROM {table} s
    JOIN {shadow_table} e ON {pk_cast_join}
    WHERE e.embedding IS NOT NULL
"""

# Delete an embedding from the shadow table (for DELETE events).
SQL_EXT_DELETE_EMBEDDING = """
    DELETE FROM {shadow_table}
    WHERE row_id = %(row_id)s;
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


# -- Multi-column / External helpers ----------------------------------------


def _content_expr(columns: list[str], alias: str = "") -> str:
    """Build a SQL expression that concatenates multiple columns with labels.

    Single column: "col"  (with optional alias prefix)
    Multi-column: 'title: ' || s."title" || E'\\n' || 'description: ' || s."description"
    """
    prefix = f'{alias}.' if alias else ''
    if len(columns) == 1:
        return f'{prefix}{_quoted(columns[0])}'
    parts: list[str] = []
    for col in columns:
        parts.append(f"'{col}: ' || {prefix}{_quoted(col)}")
    return " || E'\\n' || ".join(parts)


def _content_text(columns: list[str], row: dict[str, object]) -> str:
    """Build concatenated text from a row dict for multi-column embedding."""
    if len(columns) == 1:
        return str(row[columns[0]])
    parts = [f"{col}: {row[col]}" for col in columns]
    return "\n".join(parts)


def _pk_cast_join(pk_columns: list[str]) -> str:
    """Build JOIN condition: s."id"::TEXT = e.row_id (for shadow table)."""
    if len(pk_columns) == 1:
        return f's.{_quoted(pk_columns[0])}::TEXT = e.row_id'
    pk_concat = " || ',' || ".join(f's.{_quoted(c)}::TEXT' for c in pk_columns)
    return f"({pk_concat}) = e.row_id"


def _pk_select_prefixed(pk_columns: list[str], alias: str = "s") -> str:
    """Build SELECT list for PK columns with table alias prefix."""
    return ", ".join(f'{alias}.{_quoted(c)}' for c in pk_columns)


def _pk_order_prefixed(pk_columns: list[str], alias: str = "s") -> str:
    """Build ORDER BY clause for PK columns with table alias prefix."""
    return ", ".join(f'{alias}.{_quoted(c)} ASC' for c in pk_columns)


def _pk_where_gt_prefixed(pk_columns: list[str], alias: str = "s") -> str:
    """Build WHERE clause for keyset pagination with table alias prefix."""
    if len(pk_columns) == 1:
        return f'{alias}.{_quoted(pk_columns[0])} > %(last_pk)s'
    cols = ", ".join(f'{alias}.{_quoted(c)}' for c in pk_columns)
    params = ", ".join(f"%(last_pk_{c})s" for c in pk_columns)
    return f"ROW({cols}) > ROW({params})"


def _not_null_check(columns: list[str], alias: str = "s") -> str:
    """Build NOT NULL check for one or more source columns."""
    prefix = f'{alias}.' if alias else ''
    parts = [f'{prefix}{_quoted(c)} IS NOT NULL' for c in columns]
    return " AND ".join(parts)


# -- Functions --------------------------------------------------------------


def fetch_unembedded_batch(
    conn: DictConnection,
    table: str,
    column: str,
    batch_size: int,
    pk_columns: list[str] | None = None,
    last_pk_params: dict[str, object] | None = None,
    schema: str = "public",
    storage_mode: str = "inline",
    shadow_table: str | None = None,
    source_columns: list[str] | None = None,
) -> list[dict[str, object]]:
    """Fetch rows with source text but no embedding, using keyset pagination.

    Returns a list of dicts with PK columns and the source column value(s).
    Keyset pagination ensures consistent performance regardless of offset depth.

    Pass last_pk_params=None for the first page (no PK filter).

    In external mode, uses LEFT JOIN against the shadow table to find
    rows without embeddings.
    """
    if pk_columns is None:
        pk_columns = ["id"]

    columns = source_columns or [column]
    qualified = _qualified_table(table, schema)

    if storage_mode == "external" and shadow_table:
        qualified_shadow = _qualified_table(shadow_table, schema)
        content = _content_expr(columns, alias="s")

        if last_pk_params is None:
            sql = SQL_EXT_FETCH_UNEMBEDDED_FIRST.format(
                table=qualified,
                shadow_table=qualified_shadow,
                pk_select=_pk_select_prefixed(pk_columns),
                content_expr=f"{content} AS content",
                pk_cast_join=_pk_cast_join(pk_columns),
                not_null_check=_not_null_check(columns, alias="s"),
                pk_order_prefixed=_pk_order_prefixed(pk_columns),
            )
            params: dict[str, object] = {"batch_size": batch_size}
        else:
            sql = SQL_EXT_FETCH_UNEMBEDDED_NEXT.format(
                table=qualified,
                shadow_table=qualified_shadow,
                pk_select=_pk_select_prefixed(pk_columns),
                content_expr=f"{content} AS content",
                pk_cast_join=_pk_cast_join(pk_columns),
                not_null_check=_not_null_check(columns, alias="s"),
                pk_where_gt_prefixed=_pk_where_gt_prefixed(pk_columns),
                pk_order_prefixed=_pk_order_prefixed(pk_columns),
            )
            params = {"batch_size": batch_size, **last_pk_params}
    elif len(columns) > 1:
        # Inline multi-column: select all source columns, check all NOT NULL
        col_select = ", ".join(_quoted(c) for c in columns)
        not_null = _not_null_check(columns, alias="")
        pk_sel = _pk_select(pk_columns)

        if last_pk_params is None:
            sql = (
                f"SELECT {pk_sel}, {col_select}\n"
                f"FROM {qualified}\n"
                f"WHERE {not_null}\n"
                f'  AND "embedding" IS NULL\n'
                f"ORDER BY {_pk_order(pk_columns)}\n"
                f"LIMIT %(batch_size)s;"
            )
            params = {"batch_size": batch_size}
        else:
            sql = (
                f"SELECT {pk_sel}, {col_select}\n"
                f"FROM {qualified}\n"
                f"WHERE {not_null}\n"
                f'  AND "embedding" IS NULL\n'
                f"  AND ({_pk_where_gt(pk_columns)})\n"
                f"ORDER BY {_pk_order(pk_columns)}\n"
                f"LIMIT %(batch_size)s;"
            )
            params = {"batch_size": batch_size, **last_pk_params}
    else:
        # Inline single-column (original behavior)
        quoted_col = _quoted(column)

        if last_pk_params is None:
            sql = SQL_FETCH_UNEMBEDDED_FIRST.format(
                table=qualified,
                column=quoted_col,
                pk_select=_pk_select(pk_columns),
                pk_order=_pk_order(pk_columns),
            )
            params = {"batch_size": batch_size}
        else:
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
    storage_mode: str = "inline",
    shadow_table: str | None = None,
    source_column: str = "",
    model_name: str = "",
) -> None:
    """Update a single row's embedding vector.

    In external mode, upserts into the shadow table instead of updating
    the source table.
    """
    if pk_columns is None:
        pk_columns = ["id"]

    if storage_mode == "external" and shadow_table:
        qualified_shadow = _qualified_table(shadow_table, schema)
        sql = SQL_EXT_UPSERT_EMBEDDING.format(shadow_table=qualified_shadow)
        params: dict[str, object] = {
            "row_id": row_id,
            "embedding": embedding,
            "source_column": source_column,
            "model_name": model_name,
        }
        conn.execute(sql, params)
    else:
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
    storage_mode: str = "inline",
    shadow_table: str | None = None,
    source_column: str = "",
    model_name: str = "",
) -> None:
    """Update embeddings for multiple rows in a single transaction.

    Each row dict must have the PK column(s). The embeddings list must be the
    same length as the rows list. Commits after all updates.

    In external mode, upserts into the shadow table.
    """
    if pk_columns is None:
        pk_columns = ["id"]
    if len(rows) != len(embeddings):
        raise ValueError(
            f"rows ({len(rows)}) and embeddings ({len(embeddings)}) "
            "must have the same length"
        )

    if storage_mode == "external" and shadow_table:
        qualified_shadow = _qualified_table(shadow_table, schema)
        sql = SQL_EXT_UPSERT_EMBEDDING.format(shadow_table=qualified_shadow)

        for row, embedding in zip(rows, embeddings, strict=True):
            row_id = _pk_row_id(pk_columns, row)
            params: dict[str, object] = {
                "row_id": row_id,
                "embedding": embedding,
                "source_column": source_column,
                "model_name": model_name,
            }
            conn.execute(sql, params)
    else:
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
    storage_mode: str = "inline",
    shadow_table: str | None = None,
) -> int:
    """Count rows that have a non-NULL embedding."""
    if storage_mode == "external" and shadow_table:
        qualified_shadow = _qualified_table(shadow_table, schema)
        sql = SQL_EXT_COUNT_EMBEDDED.format(shadow_table=qualified_shadow)
    else:
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
    source_columns: list[str] | None = None,
) -> int:
    """Count rows that have non-NULL source text.

    For multi-column mode, counts rows where ALL source columns are non-NULL.
    """
    qualified = _qualified_table(table, schema)
    columns = source_columns or [column]
    if len(columns) == 1:
        quoted_col = _quoted(columns[0])
        sql = SQL_COUNT_TOTAL_WITH_CONTENT.format(
            table=qualified, column=quoted_col
        )
    else:
        where = _not_null_check(columns, alias="")
        sql = f"SELECT COUNT(*) AS cnt FROM {qualified} WHERE {where};"
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
    storage_mode: str = "inline",
    shadow_table: str | None = None,
    source_columns: list[str] | None = None,
    pk_columns: list[str] | None = None,
) -> list[dict[str, object]]:
    """Find rows semantically similar to a query vector using cosine similarity.

    Returns rows ordered by similarity (highest first), each containing
    id, content (the source column text), and similarity score.

    In external mode, JOINs the shadow table to retrieve embeddings.
    """
    if pk_columns is None:
        pk_columns = ["id"]
    columns = source_columns or [column]
    qualified = _qualified_table(table, schema)

    if storage_mode == "external" and shadow_table:
        qualified_shadow = _qualified_table(shadow_table, schema)
        content = _content_expr(columns, alias="s")
        sql = SQL_EXT_SEARCH_SIMILAR.format(
            table=qualified,
            shadow_table=qualified_shadow,
            content_expr=content,
            pk_cast_join=_pk_cast_join(pk_columns),
        )
    else:
        quoted_col = _quoted(column)
        sql = SQL_SEARCH_SIMILAR.format(table=qualified, column=quoted_col)

    # ef_search must be >= limit or HNSW returns fewer results than requested.
    ef_search = max(limit * 2, 100)
    conn.execute(f"SET hnsw.ef_search = {ef_search}")
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
    storage_mode: str = "inline",
    shadow_table: str | None = None,
    source_columns: list[str] | None = None,
    pk_columns: list[str] | None = None,
) -> list[dict[str, object]]:
    """Semantic similarity + SQL WHERE filters in one query.

    Validates filter keys against information_schema columns before building
    parameterized SQL. Sets hnsw.iterative_scan = relaxed_order on pgvector
    >= 0.8.0 to avoid recall degradation with selective pre-filters.

    In external mode, JOINs the shadow table and prefixes filter columns
    with the source table alias.

    Raises InvalidFilterError if a filter key doesn't match a table column.
    """
    if pk_columns is None:
        pk_columns = ["id"]
    columns = source_columns or [column]
    is_external = storage_mode == "external" and shadow_table is not None

    # Validate filter keys against actual table columns.
    valid_columns: set[str] = set()
    if filters:
        valid_columns = get_column_names(conn, table, schema)
        for key in filters:
            actual_col = key
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

    qualified = _qualified_table(table, schema)

    if is_external:
        qualified_shadow = _qualified_table(shadow_table, schema)
        content = _content_expr(columns, alias="s")
        sql = SQL_EXT_HYBRID_SEARCH_BASE.format(
            table=qualified,
            shadow_table=qualified_shadow,
            content_expr=content,
            pk_cast_join=_pk_cast_join(pk_columns),
        )
        col_prefix = "s."
        emb_prefix = "e."
    else:
        quoted_col = _quoted(column)
        sql = SQL_HYBRID_SEARCH_BASE.format(table=qualified, column=quoted_col)
        col_prefix = ""
        emb_prefix = ""

    params: dict[str, object] = {
        "query_vector": query_vector,
        "limit": limit,
    }

    for key, value in filters.items():
        is_range_op = (
            (key.endswith("_min") or key.endswith("_max"))
            and key not in valid_columns
        )
        if is_range_op and key.endswith("_min"):
            actual_col = key[:-4]
            sql += f"  AND {col_prefix}{_quoted(actual_col)} >= %({key})s\n"
        elif is_range_op and key.endswith("_max"):
            actual_col = key[:-4]
            sql += f"  AND {col_prefix}{_quoted(actual_col)} <= %({key})s\n"
        else:
            sql += f"  AND {col_prefix}{_quoted(key)} ILIKE %({key})s\n"
        params[key] = value

    sql += f'    ORDER BY {emb_prefix}"embedding" <=> %(query_vector)s::vector\n'
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
    source_columns: list[str] | None = None,
) -> str | None:
    """Fetch the source text for a single row (used by the worker).

    Returns the text value or None if the row doesn't exist or the
    column is NULL. For multi-column mode, concatenates all columns
    with field labels.
    """
    if pk_columns is None:
        pk_columns = ["id"]

    columns = source_columns or [column]
    qualified = _qualified_table(table, schema)

    # Select all source columns
    col_list = ", ".join(_quoted(c) for c in columns)
    sql = SQL_FETCH_ROW_TEXT_TEMPLATE.format(
        table=qualified,
        column=col_list,
        pk_where_eq=_pk_where_eq(pk_columns),
    )
    params = _pk_params_from_row_id(pk_columns, row_id)
    result = conn.execute(sql, params).fetchone()
    if result is None:
        return None

    # Check if any required column is NULL
    for col in columns:
        if result[col] is None:
            return None

    return _content_text(columns, result)


def create_shadow_table(
    conn: DictConnection,
    shadow_table: str,
    dimensions: int,
    schema: str = "public",
) -> None:
    """Create the shadow table for external embedding storage."""
    qualified_shadow = _qualified_table(shadow_table, schema)
    sql = SQL_CREATE_SHADOW_TABLE.format(
        shadow_table=qualified_shadow,
        dimensions=dimensions,
    )
    conn.execute(sql)
    conn.commit()
    logger.info("Created shadow table %s", qualified_shadow)


def null_embedding(
    conn: DictConnection,
    table: str,
    row_id: str,
    pk_columns: list[str] | None = None,
    schema: str = "public",
    storage_mode: str = "inline",
    shadow_table: str | None = None,
) -> None:
    """Set a row's embedding to NULL (inline) or delete from shadow table (external)."""
    if pk_columns is None:
        pk_columns = ["id"]

    if storage_mode == "external" and shadow_table:
        qualified_shadow = _qualified_table(shadow_table, schema)
        sql = SQL_EXT_DELETE_EMBEDDING.format(shadow_table=qualified_shadow)
        conn.execute(sql, {"row_id": row_id})
    else:
        qualified = _qualified_table(table, schema)
        sql = SQL_NULL_EMBEDDING_TEMPLATE.format(
            table=qualified,
            pk_where_eq=_pk_where_eq(pk_columns),
        )
        params = _pk_params_from_row_id(pk_columns, row_id)
        conn.execute(sql, params)

    conn.commit()
    logger.debug("Nulled embedding for row %s in %s", row_id, table)


def search_all(
    conn: DictConnection,
    query: str,
    providers: dict[str, object],
    project_config: ProjectConfig,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Search across all configured tables, merging results by similarity.

    Embeds the query once per unique model, searches each table independently,
    then merges and sorts all results by similarity score (descending).

    Each result is tagged with _source_table and _source_schema.
    """
    if not project_config.tables:
        return []

    # Embed query once per unique model
    query_vectors: dict[str, list[float]] = {}
    for tc in project_config.tables:
        if tc.model not in query_vectors:
            provider = providers.get(tc.model)
            if provider is None:
                logger.warning("No provider for model %s, skipping tables using it", tc.model)
                continue
            query_vectors[tc.model] = provider.embed_query(query)

    # Search each table and tag results
    all_results: list[dict[str, object]] = []
    for tc in project_config.tables:
        qv = query_vectors.get(tc.model)
        if qv is None:
            continue

        table_results = search_similar(
            conn=conn,
            table=tc.table,
            column=tc.column,
            query_vector=qv,
            limit=limit,
            schema=tc.schema,
            storage_mode=tc.storage_mode,
            shadow_table=tc.shadow_table,
            source_columns=tc.source_columns,
            pk_columns=tc.primary_key,
        )

        for row in table_results:
            row["_source_table"] = tc.table
            row["_source_schema"] = tc.schema
            all_results.append(row)

    # Sort by similarity descending, take top N
    all_results.sort(key=lambda r: float(str(r.get("similarity", 0))), reverse=True)
    return all_results[:limit]

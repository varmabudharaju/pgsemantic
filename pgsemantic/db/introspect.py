"""Database introspection -- scan schemas, score columns for semantic search suitability.

All SQL is defined as module-level constants. Table/column names used in
SQL_SAMPLE_AVG_LENGTH are validated against information_schema before formatting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg

from pgsemantic.exceptions import TableNotFoundError

logger = logging.getLogger(__name__)

# Type alias matching client.py — a psycopg connection returning dict rows.
DictConnection = psycopg.Connection[dict[str, object]]

# -- SQL Constants ----------------------------------------------------------

# Retrieve all user-created base tables, excluding system schemas.
SQL_GET_USER_TABLES = """
    SELECT t.table_name, t.table_schema
    FROM information_schema.tables t
    WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
      AND t.table_type = 'BASE TABLE'
    ORDER BY t.table_name;
"""

# Retrieve text-like columns for a specific table (text, varchar, jsonb, char).
SQL_GET_TEXT_COLUMNS = """
    SELECT column_name, data_type, character_maximum_length, is_nullable
    FROM information_schema.columns
    WHERE table_name = %(table_name)s
      AND table_schema = %(schema_name)s
      AND data_type IN ('text', 'character varying', 'jsonb', 'character')
    ORDER BY ordinal_position;
"""

# Sample ~5% of table pages to estimate average text length without a full scan.
# Table and column are formatted in (quoted) after validation against information_schema.
SQL_SAMPLE_AVG_LENGTH = """
    SELECT
        avg(length({column}::TEXT)) AS avg_length,
        count(*) AS sampled_rows
    FROM {table} TABLESAMPLE SYSTEM(5)
    WHERE {column} IS NOT NULL;
"""

# Fallback for small tables where TABLESAMPLE returns 0 rows.
SQL_FULL_AVG_LENGTH = """
    SELECT
        avg(length({column}::TEXT)) AS avg_length,
        count(*) AS sampled_rows
    FROM {table}
    WHERE {column} IS NOT NULL;
"""

# Check whether the pgvector extension is installed.
SQL_CHECK_VECTOR_EXTENSION = """
    SELECT extname, extversion
    FROM pg_extension
    WHERE extname = 'vector';
"""

# Verify that a specific column exists on a table.
SQL_COLUMN_EXISTS = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = %(table_name)s
      AND column_name = %(column_name)s
      AND table_schema = %(schema_name)s;
"""

# Retrieve all column names and types for a table (used for filter validation).
SQL_GET_ALL_COLUMNS = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = %(table_name)s
      AND table_schema = %(schema_name)s
    ORDER BY ordinal_position;
"""

# Fast estimated row count from pg_class (avoids full table scan).
SQL_GET_ROW_COUNT = """
    SELECT reltuples::BIGINT AS estimate
    FROM pg_class
    WHERE relname = %(table_name)s;
"""

# Retrieve primary key column(s) for a table, ordered by position.
SQL_GET_PRIMARY_KEY = """
    SELECT kcu.column_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
        AND tc.table_schema = kcu.table_schema
    WHERE tc.constraint_type = 'PRIMARY KEY'
      AND tc.table_name = %(table_name)s
      AND tc.table_schema = %(schema_name)s
    ORDER BY kcu.ordinal_position;
"""

# -- Heuristic Sets ---------------------------------------------------------

# Column names that strongly suggest semantic content worth embedding.
SEMANTIC_COLUMN_NAMES: frozenset[str] = frozenset({
    "description",
    "body",
    "content",
    "text",
    "summary",
    "notes",
    "note",
    "bio",
    "review",
    "comment",
    "message",
    "details",
    "abstract",
    "excerpt",
    "title",
    "question",
    "answer",
})

# Column names that are almost never useful for semantic search.
LOW_VALUE_COLUMN_NAMES: frozenset[str] = frozenset({
    "id",
    "status",
    "code",
    "type",
    "slug",
    "url",
    "email",
    "phone",
    "uuid",
    "hash",
    "token",
})


# -- Data Classes -----------------------------------------------------------


@dataclass
class ColumnCandidate:
    """A text column that might be a good candidate for semantic search."""

    table_name: str
    table_schema: str
    column_name: str
    data_type: str
    avg_length: float
    sampled_rows: int


# -- Scoring ----------------------------------------------------------------


def score_column(candidate: ColumnCandidate) -> float:
    """Score a column for semantic search suitability (heuristic).

    Scoring rules:
      - Low-value column names (id, email, etc.) are forced to 1.0.
      - Long avg text (>200 chars) scores 3.0; medium (>=50) scores 2.0; short scores 1.0.
      - Semantic column names (description, body, etc.) get a +0.5 bonus.
    """
    if candidate.column_name in LOW_VALUE_COLUMN_NAMES:
        return 1.0

    if candidate.avg_length > 200:
        score = 3.0
    elif candidate.avg_length >= 50:
        score = 2.0
    else:
        score = 1.0

    if candidate.column_name in SEMANTIC_COLUMN_NAMES:
        score += 0.5

    return score


def score_to_stars(score: float) -> str:
    """Convert a numeric score to a star rating string."""
    if score >= 2.5:
        return "\u2605\u2605\u2605"
    elif score >= 1.5:
        return "\u2605\u2605\u2606"
    else:
        return "\u2605\u2606\u2606"


# -- Query Functions --------------------------------------------------------


def get_user_tables(
    conn: DictConnection,
) -> list[dict[str, str]]:
    """Get all user-created tables (excluding system schemas)."""
    result = conn.execute(SQL_GET_USER_TABLES).fetchall()
    return [{k: str(v) for k, v in row.items()} for row in result]


def get_text_columns(
    conn: DictConnection,
    table_name: str,
    schema_name: str = "public",
) -> list[dict[str, object]]:
    """Get all text-like columns for a specific table."""
    result = conn.execute(
        SQL_GET_TEXT_COLUMNS,
        {"table_name": table_name, "schema_name": schema_name},
    ).fetchall()
    return [dict(row) for row in result]


def sample_avg_length(
    conn: DictConnection,
    table_name: str,
    column_name: str,
    schema_name: str = "public",
) -> tuple[float, int]:
    """Sample average text length for a column using TABLESAMPLE.

    Validates column existence against information_schema before building
    the sampling query to prevent SQL injection.
    """
    exists = conn.execute(
        SQL_COLUMN_EXISTS,
        {
            "table_name": table_name,
            "column_name": column_name,
            "schema_name": schema_name,
        },
    ).fetchone()
    if exists is None:
        raise TableNotFoundError(
            f"Column '{column_name}' not found on table "
            f"'{schema_name}.{table_name}'"
        )

    qualified_table = f'"{schema_name}"."{table_name}"'
    quoted_column = f'"{column_name}"'
    sql = SQL_SAMPLE_AVG_LENGTH.format(
        table=qualified_table, column=quoted_column
    )

    result = conn.execute(sql).fetchone()
    if result is None or result["avg_length"] is None or int(str(result["sampled_rows"])) == 0:
        # TABLESAMPLE returned 0 rows (table too small) — fall back to full scan
        fallback_sql = SQL_FULL_AVG_LENGTH.format(
            table=qualified_table, column=quoted_column
        )
        result = conn.execute(fallback_sql).fetchone()
        if result is None or result["avg_length"] is None:
            return (0.0, 0)
    avg_val: float = float(str(result["avg_length"]))
    count_val: int = int(str(result["sampled_rows"]))
    return (avg_val, count_val)


def get_column_names(
    conn: DictConnection,
    table_name: str,
    schema_name: str = "public",
) -> set[str]:
    """Get all column names for a table (used for filter validation)."""
    result = conn.execute(
        SQL_GET_ALL_COLUMNS,
        {"table_name": table_name, "schema_name": schema_name},
    ).fetchall()
    return {str(row["column_name"]) for row in result}


def column_exists(
    conn: DictConnection,
    table_name: str,
    column_name: str,
    schema_name: str = "public",
) -> bool:
    """Check if a specific column exists on a table."""
    result = conn.execute(
        SQL_COLUMN_EXISTS,
        {
            "table_name": table_name,
            "column_name": column_name,
            "schema_name": schema_name,
        },
    ).fetchone()
    return result is not None


def get_row_count_estimate(
    conn: DictConnection,
    table_name: str,
) -> int:
    """Get estimated row count from pg_class (fast, no full scan)."""
    result = conn.execute(
        SQL_GET_ROW_COUNT, {"table_name": table_name}
    ).fetchone()
    if result is None:
        return 0
    estimate: int = int(str(result["estimate"]))
    return max(0, estimate)


def get_primary_key_columns(
    conn: DictConnection,
    table_name: str,
    schema_name: str = "public",
) -> list[str]:
    """Get primary key column names for a table."""
    result = conn.execute(
        SQL_GET_PRIMARY_KEY,
        {"table_name": table_name, "schema_name": schema_name},
    ).fetchall()
    return [str(row["column_name"]) for row in result]


def inspect_database(
    conn: DictConnection,
) -> list[ColumnCandidate]:
    """Scan all user tables and score text columns for semantic search suitability.

    Skips the internal pgvector_setup_queue table. Returns candidates sorted
    by descending score then descending average text length.
    """
    candidates: list[ColumnCandidate] = []
    tables = get_user_tables(conn)

    for table_info in tables:
        table_name = table_info["table_name"]
        schema_name = table_info["table_schema"]

        # Skip internal queue table
        if table_name == "pgvector_setup_queue":
            continue

        text_columns = get_text_columns(conn, table_name, schema_name)
        for col in text_columns:
            col_name = str(col["column_name"])
            try:
                avg_len, sampled = sample_avg_length(
                    conn, table_name, col_name, schema_name
                )
            except Exception:
                logger.warning(
                    "Failed to sample %s.%s, skipping", table_name, col_name
                )
                avg_len, sampled = 0.0, 0

            candidates.append(
                ColumnCandidate(
                    table_name=table_name,
                    table_schema=schema_name,
                    column_name=col_name,
                    data_type=str(col["data_type"]),
                    avg_length=avg_len,
                    sampled_rows=sampled,
                )
            )

    candidates.sort(
        key=lambda c: (score_column(c), c.avg_length), reverse=True
    )
    return candidates

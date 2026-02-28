"""FastMCP server for pgsemantic.

Exposes semantic_search, hybrid_search, and get_embedding_status as MCP tools
that AI agents (Claude Desktop, Cursor, Copilot) can call over stdio or HTTP/SSE.

This is library code — uses logging only, no Rich or print().
"""
from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from pgsemantic.config import load_project_config, load_settings
from pgsemantic.db.client import get_connection, get_pgvector_version
from pgsemantic.db.queue import count_failed, count_pending
from pgsemantic.db.vectors import (
    count_embedded,
    count_total_with_content,
    search_similar,
)
from pgsemantic.db.vectors import (
    hybrid_search as db_hybrid_search,
)
from pgsemantic.embeddings import get_provider
from pgsemantic.exceptions import PgvectorSetupError

logger = logging.getLogger(__name__)

mcp = FastMCP(name="pgsemantic")


def _get_table_config(
    table: str,
) -> tuple[str, str, str, str, str | None, str]:
    """Load settings/config and return provider info for a table.

    Returns (database_url, column, model, model_name, openai_api_key, ollama_base_url).
    Loads settings once so callers don't need a redundant load_settings() call.
    Raises ToolError if settings or table config is missing.
    """
    settings = load_settings()
    if not settings.database_url:
        raise ToolError(
            "DATABASE_URL not set. "
            "Set it in your .env file or as an environment variable."
        )

    config = load_project_config()
    if config is None:
        raise ToolError(
            "No pgsemantic config found. "
            "Run: pgsemantic apply --table <table> --column <column> first."
        )

    table_config = config.get_table_config(table)
    if table_config is None:
        available = [tc.table for tc in config.tables]
        raise ToolError(
            f"Table '{table}' not found in pgsemantic config. "
            f"Available tables: {', '.join(available) if available else '(none)'}. "
            f"Run: pgsemantic apply --table {table} --column <column>"
        )

    return (
        settings.database_url,
        table_config.column,
        table_config.model,
        table_config.model_name,
        settings.openai_api_key,
        settings.ollama_base_url,
    )


@mcp.tool()
def semantic_search(
    query: str,
    table: str,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Find rows semantically similar to a natural-language query.

    Args:
        query: The natural-language search query.
        table: The table to search in.
        limit: Maximum number of results to return (default 10).

    Returns:
        List of matching rows with id, content, and similarity score.
    """
    database_url, column, model, _model_name, api_key, ollama_url = _get_table_config(table)

    try:
        provider = get_provider(model, api_key=api_key, ollama_base_url=ollama_url)
        query_vector = provider.embed_query(query)

        with get_connection(database_url) as conn:
            results = search_similar(
                conn=conn,
                table=table,
                column=column,
                query_vector=query_vector,
                limit=limit,
            )

        # Strip the raw embedding vector — it's huge and useless for the AI
        clean_results = [
            {k: v for k, v in row.items() if k != "embedding"}
            for row in results
        ]

        logger.info(
            "semantic_search: query=%r table=%s results=%d",
            query[:50],
            table,
            len(clean_results),
        )
        return clean_results

    except PgvectorSetupError as e:
        raise ToolError(str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error in semantic_search")
        raise ToolError(f"Search failed: {e}") from e


@mcp.tool()
def hybrid_search(
    query: str,
    table: str,
    filters: dict[str, object] | None = None,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Semantic similarity search with SQL WHERE filters.

    Combines pgvector ANN similarity search with column filters in a single query.
    Uses pre-filtering with hnsw.iterative_scan to avoid recall degradation.

    Args:
        query: The natural-language search query.
        table: The table to search in.
        filters: Optional column filters (e.g. {"category": "laptop", "price_max": 1500}).
        limit: Maximum number of results to return (default 10).

    Returns:
        List of matching rows with id, content, and similarity score.
    """
    database_url, column, model, _model_name, api_key, ollama_url = _get_table_config(table)
    effective_filters: dict[str, object] = filters if filters is not None else {}

    try:
        provider = get_provider(model, api_key=api_key, ollama_base_url=ollama_url)
        query_vector = provider.embed_query(query)

        with get_connection(database_url) as conn:
            pgvector_version = get_pgvector_version(conn)
            results = db_hybrid_search(
                conn=conn,
                table=table,
                column=column,
                query_vector=query_vector,
                filters=effective_filters,
                limit=limit,
                pgvector_version=pgvector_version,
            )

        clean_results = [
            {k: v for k, v in row.items() if k != "embedding"}
            for row in results
        ]

        logger.info(
            "hybrid_search: query=%r table=%s filters=%s results=%d",
            query[:50],
            table,
            effective_filters,
            len(clean_results),
        )
        return clean_results

    except PgvectorSetupError as e:
        raise ToolError(str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error in hybrid_search")
        raise ToolError(f"Search failed: {e}") from e


@mcp.tool()
def get_embedding_status(table: str) -> dict[str, object]:
    """Report embedding coverage and queue depth for a table.

    Args:
        table: The table to check status for.

    Returns:
        Dict with table, total_rows, embedded_rows, coverage_pct,
        pending_queue, failed_queue, model, and applied_at.
    """
    database_url, column, _model, model_name, _api_key, _ollama_url = _get_table_config(table)

    try:
        config = load_project_config()
        if config is None:
            raise ToolError("Internal error: config became unavailable.")
        table_config = config.get_table_config(table)
        if table_config is None:
            raise ToolError(f"Internal error: table '{table}' config became unavailable.")

        with get_connection(database_url) as conn:
            embedded = count_embedded(conn, table, schema=table_config.schema)
            total = count_total_with_content(
                conn, table, column, schema=table_config.schema
            )
            pending = count_pending(conn, table)
            failed = count_failed(conn, table)

        coverage_pct = round((embedded / total) * 100, 1) if total > 0 else 0.0

        status: dict[str, object] = {
            "table": table,
            "total_rows": total,
            "embedded_rows": embedded,
            "coverage_pct": coverage_pct,
            "pending_queue": pending,
            "failed_queue": failed,
            "model": model_name,
            "applied_at": table_config.applied_at,
        }

        logger.info("get_embedding_status: table=%s coverage=%.1f%%", table, coverage_pct)
        return status

    except PgvectorSetupError as e:
        raise ToolError(str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error in get_embedding_status")
        raise ToolError(f"Status check failed: {e}") from e

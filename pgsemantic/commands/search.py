"""pgsemantic search — semantic search from the command line."""
from __future__ import annotations

import csv
import json
import sys
from typing import Any

import typer
from rich.console import Console

from pgsemantic.config import load_project_config, load_settings
from pgsemantic.db.client import get_connection
from pgsemantic.db.vectors import search_all, search_chunked, search_similar
from pgsemantic.embeddings import get_provider

console = Console()

# Columns to hide from search results (internal/large)
_TABLE_HIDDEN_COLUMNS = frozenset({"embedding", "similarity", "content"})


def _results_to_csv(results: list[dict[str, Any]]) -> None:
    """Write results as CSV to stdout."""
    if not results:
        return
    writer = csv.DictWriter(sys.stdout, fieldnames=list(results[0].keys()))
    writer.writeheader()
    for row in results:
        writer.writerow({k: str(v) if v is not None else "" for k, v in row.items()})


def _results_to_json(results: list[dict[str, Any]]) -> None:
    """Write results as a JSON array to stdout."""
    print(json.dumps(results, indent=2, default=str))


def _results_to_jsonl(results: list[dict[str, Any]]) -> None:
    """Write results as newline-delimited JSON to stdout."""
    for row in results:
        print(json.dumps(row, default=str))


def _output_results(
    results: list[dict[str, Any]],
    query: str,
    table_label: str,
    column: str,
    fmt: str,
) -> None:
    """Format and output search results in the requested format."""
    if not results:
        if fmt == "table":
            console.print("\n[yellow]No results found.[/yellow]\n")
        elif fmt == "json":
            print("[]")
        raise typer.Exit(code=0)

    if fmt == "csv":
        _results_to_csv(results)
        return
    if fmt == "json":
        _results_to_json(results)
        return
    if fmt == "jsonl":
        _results_to_jsonl(results)
        return

    # Rich table output
    console.print()
    console.print(
        f"[dim]Results for:[/dim] [cyan]\"{query}\"[/cyan] "
        f"[dim]in {table_label}[/dim]"
    )

    for i, row in enumerate(results, 1):
        sim = float(str(row["similarity"]))
        score_color = "green" if sim >= 0.5 else "yellow" if sim >= 0.3 else "dim"

        meta_parts: list[str] = []
        for key, val in row.items():
            if key in _TABLE_HIDDEN_COLUMNS or key == column:
                continue
            meta_parts.append(f"[dim]{key}:[/dim] {val}")
        meta_line = "  ".join(meta_parts)

        content = str(row.get("content", row.get(column, "")))
        if len(content) > 300:
            content = content[:300] + "..."

        console.print()
        console.print(
            f"  [{score_color}]{i}. (score: {sim:.3f})[/{score_color}]  {meta_line}"
        )
        console.print(f"  [white]{content}[/white]")

    console.print()


def search_command(
    query: str = typer.Argument(
        ...,
        help="Natural language search query.",
    ),
    table: str = typer.Option(
        "",
        "--table",
        "-t",
        help="Table to search. Omit to search all configured tables.",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-n",
        help="Number of results to return.",
    ),
    db: str = typer.Option(
        "",
        "--db",
        "-d",
        help="Database URL (overrides DATABASE_URL env var).",
    ),
    fmt: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table (default), csv, json, jsonl.",
    ),
) -> None:
    """Search your database using natural language.

    Omit --table to search across all configured tables at once.
    """
    if fmt not in ("table", "csv", "json", "jsonl"):
        console.print("[red]--format must be one of: table, csv, json, jsonl[/red]")
        raise typer.Exit(code=1)

    settings = load_settings()
    database_url = db if db else settings.database_url

    if not database_url:
        console.print(
            "[red]No database URL provided.[/red] "
            "Set DATABASE_URL in your .env file or pass --db."
        )
        raise typer.Exit(code=1)

    config = load_project_config()
    if config is None:
        console.print(
            "[red]No .pgsemantic.json found.[/red] "
            "Run [cyan]pgsemantic apply --table <table> --column <column>[/cyan] first."
        )
        raise typer.Exit(code=1)

    # Single-table search (original behavior)
    if table:
        table_config = config.get_table_config(table)
        if table_config is None:
            console.print(
                f"[red]Table '{table}' not found in config.[/red] "
                f"Run [cyan]pgsemantic apply --table {table}[/cyan] first."
            )
            raise typer.Exit(code=1)

        column_name = table_config.column
        model = table_config.model
        source_columns = table_config.source_columns

        try:
            api_key = settings.openai_api_key if model == "openai" else None
            provider = get_provider(model, api_key=api_key)
            query_vector = provider.embed_query(query)

            with get_connection(database_url) as conn:
                if table_config.chunked:
                    results = search_chunked(
                        conn, table, column_name, query_vector,
                        shadow_table=table_config.shadow_table,
                        schema=table_config.schema,
                        pk_columns=table_config.primary_key,
                        limit=limit,
                    )
                else:
                    results = search_similar(
                        conn, table, column_name, query_vector, limit=limit,
                        schema=table_config.schema,
                        storage_mode=table_config.storage_mode,
                        shadow_table=table_config.shadow_table,
                        source_columns=source_columns,
                        pk_columns=table_config.primary_key,
                    )

            _output_results(results, query, f"{table}.{column_name}", column_name, fmt)

        except typer.Exit:
            raise
        except Exception as e:
            console.print()
            console.print(f"[red]Search failed:[/red] {e}")
            raise typer.Exit(code=1) from None
        return

    # Cross-table search (--table omitted)
    if not config.tables:
        console.print("[red]No tables configured.[/red] Run pgsemantic apply first.")
        raise typer.Exit(code=1)

    try:
        providers: dict[str, object] = {}
        for tc in config.tables:
            if tc.model not in providers:
                api_key = settings.openai_api_key if tc.model == "openai" else None
                ollama_url = settings.ollama_base_url if tc.model == "ollama" else None
                providers[tc.model] = get_provider(
                    tc.model, api_key=api_key, ollama_base_url=ollama_url
                )

        with get_connection(database_url) as conn:
            results = search_all(
                conn=conn,
                query=query,
                providers=providers,
                project_config=config,
                limit=limit,
            )

        _output_results(results, query, "all tables", "content", fmt)

    except typer.Exit:
        raise
    except Exception as e:
        console.print()
        console.print(f"[red]Search failed:[/red] {e}")
        raise typer.Exit(code=1) from None

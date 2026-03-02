"""pgsemantic search — semantic search from the command line."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from pgsemantic.config import load_project_config, load_settings
from pgsemantic.db.client import get_connection
from pgsemantic.db.vectors import search_similar
from pgsemantic.embeddings import get_provider

console = Console()

# Columns to hide from search results (internal/large)
_HIDDEN_COLUMNS = frozenset({"embedding", "similarity", "content"})


def search_command(
    query: str = typer.Argument(
        ...,
        help="Natural language search query.",
    ),
    table: str = typer.Option(
        ...,
        "--table",
        "-t",
        help="Table to search.",
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
) -> None:
    """Search your database using natural language."""
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
            f"Run [cyan]pgsemantic apply --table {table}[/cyan] first."
        )
        raise typer.Exit(code=1)

    table_config = config.get_table_config(table)
    if table_config is None:
        console.print(
            f"[red]Table '{table}' not found in config.[/red] "
            f"Run [cyan]pgsemantic apply --table {table}[/cyan] first."
        )
        raise typer.Exit(code=1)

    column = table_config.column
    model = table_config.model
    source_columns = table_config.source_columns

    try:
        api_key = settings.openai_api_key if model == "openai" else None
        provider = get_provider(model, api_key=api_key)

        query_vector = provider.embed_query(query)

        with get_connection(database_url) as conn:
            results = search_similar(
                conn, table, column, query_vector, limit=limit,
                schema=table_config.schema,
                storage_mode=table_config.storage_mode,
                shadow_table=table_config.shadow_table,
                source_columns=source_columns,
                pk_columns=table_config.primary_key,
            )

        if not results:
            console.print("\n[yellow]No results found.[/yellow]\n")
            raise typer.Exit(code=0)

        console.print()
        console.print(
            f"[dim]Results for:[/dim] [cyan]\"{query}\"[/cyan] "
            f"[dim]in {table}.{column}[/dim]"
        )

        for i, row in enumerate(results, 1):
            sim = float(str(row["similarity"]))
            score_color = "green" if sim >= 0.5 else "yellow" if sim >= 0.3 else "dim"

            # Build metadata line from non-hidden, non-content columns
            meta_parts: list[str] = []
            for key, val in row.items():
                if key in _HIDDEN_COLUMNS or key == column:
                    continue
                meta_parts.append(f"[dim]{key}:[/dim] {val}")
            meta_line = "  ".join(meta_parts)

            # Truncate content for preview
            content = str(row.get(column, ""))
            if len(content) > 300:
                content = content[:300] + "..."

            console.print()
            console.print(
                f"  [{score_color}]{i}. (score: {sim:.3f})[/{score_color}]  {meta_line}"
            )
            console.print(f"  [white]{content}[/white]")

        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        console.print()
        console.print(f"[red]Search failed:[/red] {e}")
        raise typer.Exit(code=1) from None

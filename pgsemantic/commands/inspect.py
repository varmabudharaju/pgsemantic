"""pgsemantic inspect — scan a database and score columns for semantic search."""
from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pgsemantic.db.client import get_connection, get_pgvector_version
from pgsemantic.db.introspect import (
    ColumnCandidate,
    inspect_database,
    score_column,
    score_to_stars,
)

console = Console()


def inspect_command(
    database_url: str = typer.Argument(
        ...,
        help="PostgreSQL connection string (e.g. postgresql://user:pass@host/db)",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON for programmatic use.",
    ),
) -> None:
    """Scan a database and score columns for semantic search suitability."""
    try:
        with get_connection(database_url, register_vector_type=False) as conn:
            pgv_version = get_pgvector_version(conn)
            candidates = inspect_database(conn)

        if not candidates:
            console.print(Panel(
                "[yellow]No text columns found in this database.[/yellow]\n\n"
                "pgsemantic looks for TEXT, VARCHAR, and JSONB columns.\n"
                "Make sure your database has tables with text data.",
                title="No Candidates Found",
                border_style="yellow",
            ))
            raise typer.Exit(code=0)

        if output_json:
            _print_json(candidates, pgv_version)
        else:
            _print_rich_table(candidates, database_url, pgv_version)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to inspect database:[/red]\n\n{e}",
            title="Connection Error",
            border_style="red",
        ))
        raise typer.Exit(code=1) from None


def _print_rich_table(
    candidates: list[ColumnCandidate],
    database_url: str,
    pgv_version: tuple[int, int, int] | None,
) -> None:
    """Render inspection results as a Rich table."""
    pgv_str = (
        f"pgvector {'.'.join(str(v) for v in pgv_version)}"
        if pgv_version else "pgvector not installed"
    )
    display_url = database_url.split("@")[-1] if "@" in database_url else database_url

    console.print()
    console.print(Panel(
        f"Semantic Search Candidates — {display_url}\n[dim]{pgv_str}[/dim]",
        border_style="blue",
    ))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Table", style="cyan")
    table.add_column("Column", style="green")
    table.add_column("Score")
    table.add_column("Avg Length", justify="right")
    table.add_column("Type", style="dim")

    for c in candidates:
        score = score_column(c)
        stars = score_to_stars(score)
        star_style = "green" if score >= 2.5 else "yellow" if score >= 1.5 else "dim"
        avg_len_str = f"{c.avg_length:,.0f} chars" if c.avg_length > 0 else "—"
        table.add_row(
            c.table_name,
            c.column_name,
            f"[{star_style}]{stars}[/{star_style}]",
            avg_len_str,
            c.data_type,
        )

    console.print(table)
    console.print()
    console.print(
        "[dim]Scoring is heuristic (text length + column name patterns).[/dim]\n"
        "[dim]Always verify recommendations make sense for your use case.[/dim]"
    )

    if candidates:
        best = candidates[0]
        console.print()
        console.print("Next step:")
        console.print(
            f"  [cyan]pgsemantic apply --table {best.table_name} "
            f"--column {best.column_name}[/cyan]"
        )
    console.print()


def _print_json(
    candidates: list[ColumnCandidate],
    pgv_version: tuple[int, int, int] | None,
) -> None:
    """Output results as JSON for programmatic use."""
    data = {
        "pgvector_version": (
            ".".join(str(v) for v in pgv_version) if pgv_version else None
        ),
        "candidates": [
            {
                "table": c.table_name,
                "schema": c.table_schema,
                "column": c.column_name,
                "data_type": c.data_type,
                "avg_length": round(c.avg_length, 1),
                "sampled_rows": c.sampled_rows,
                "score": score_column(c),
                "stars": score_to_stars(score_column(c)),
            }
            for c in candidates
        ],
    }
    console.print(json.dumps(data, indent=2))

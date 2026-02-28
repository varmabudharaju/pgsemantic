"""CLI command: pgsemantic status — embedding health dashboard.

Shows a Rich table with embedding coverage, queue depth, and model info
for each watched table in the project config.
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pgsemantic.config import load_project_config, load_settings
from pgsemantic.db.client import get_connection
from pgsemantic.db.queue import count_failed, count_pending
from pgsemantic.db.vectors import count_embedded, count_total_with_content

console = Console()


def status_command(
    database_url: str = typer.Option(
        None,
        "--database-url",
        "-d",
        help="PostgreSQL connection string. Defaults to DATABASE_URL env var.",
    ),
) -> None:
    """Show embedding health dashboard for all watched tables."""
    settings = load_settings()
    db_url = database_url or settings.database_url

    if not db_url:
        console.print(Panel(
            "[red]DATABASE_URL not set.[/red]\n\n"
            "Provide it via [cyan]--database-url[/cyan] or set [cyan]DATABASE_URL[/cyan] "
            "in your .env file.",
            title="Configuration Required",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    config = load_project_config()
    if config is None or len(config.tables) == 0:
        console.print(Panel(
            "[yellow]No pgsemantic config found.[/yellow]\n\n"
            "Run [cyan]pgsemantic apply --table <table> --column <column>[/cyan] first.",
            title="No Configuration",
            border_style="yellow",
        ))
        raise typer.Exit(code=1)

    table = Table(title="Embedding Status", show_lines=True)
    table.add_column("Table", style="cyan", no_wrap=True)
    table.add_column("Column", style="white")
    table.add_column("Model", style="white")
    table.add_column("Coverage", justify="right")
    table.add_column("Pending", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Applied", style="dim")

    try:
        with get_connection(db_url) as conn:
            for tc in config.tables:
                embedded = count_embedded(conn, tc.table, schema=tc.schema)
                total = count_total_with_content(
                    conn, tc.table, tc.column, schema=tc.schema
                )
                pending = count_pending(conn, tc.table)
                failed = count_failed(conn, tc.table)

                pct = (embedded / total) * 100 if total > 0 else 0.0

                # Color the coverage based on percentage
                if pct >= 100.0:
                    coverage_str = f"[green]{embedded}/{total} ({pct:.1f}%)[/green]"
                elif pct >= 50.0:
                    coverage_str = f"[yellow]{embedded}/{total} ({pct:.1f}%)[/yellow]"
                else:
                    coverage_str = f"[red]{embedded}/{total} ({pct:.1f}%)[/red]"

                # Color pending/failed counts
                pending_str = (
                    f"[yellow]{pending}[/yellow]" if pending > 0
                    else f"[green]{pending}[/green]"
                )
                failed_str = (
                    f"[red]{failed}[/red]" if failed > 0
                    else f"[green]{failed}[/green]"
                )

                table.add_row(
                    tc.table,
                    tc.column,
                    tc.model_name,
                    coverage_str,
                    pending_str,
                    failed_str,
                    tc.applied_at,
                )

    except Exception as e:
        console.print(Panel(
            f"[red]Failed to connect to database:[/red]\n\n{e}",
            title="Connection Error",
            border_style="red",
        ))
        raise typer.Exit(code=1) from e

    console.print(table)

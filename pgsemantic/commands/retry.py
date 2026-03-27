"""pgsemantic retry — reset failed embedding jobs for retry."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from pgsemantic.config import load_settings
from pgsemantic.db.client import get_connection
from pgsemantic.db.queue import retry_failed_jobs

console = Console()


def retry_command(
    table: str = typer.Option(
        "",
        "--table",
        "-t",
        help="Table to retry failed jobs for.",
    ),
    all_tables: bool = typer.Option(
        False,
        "--all",
        help="Retry failed jobs for all tables.",
    ),
    db: str = typer.Option(
        "",
        "--db",
        "-d",
        help="Database URL (overrides DATABASE_URL env var).",
    ),
) -> None:
    """Retry failed embedding jobs."""
    if table and all_tables:
        console.print(Panel(
            "[red]--table and --all are mutually exclusive.[/red]",
            title="Invalid Options",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    if not table and not all_tables:
        console.print(Panel(
            "[red]Specify --table <name> or --all.[/red]",
            title="Missing Option",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    settings = load_settings()
    database_url = db if db else settings.database_url

    if not database_url:
        console.print(Panel(
            "[red]No database URL provided.[/red]\n\n"
            "Set DATABASE_URL in your .env file or pass --db.",
            title="Missing Database URL",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    try:
        with get_connection(database_url) as conn:
            table_name = table if table else None
            count = retry_failed_jobs(conn, table_name=table_name)

        if count == 0:
            scope = f"for {table}" if table else "across all tables"
            console.print(f"[yellow]No failed jobs found {scope}.[/yellow]")
        else:
            scope = table if table else "all tables"
            console.print(f"[green]Retried {count} failed job(s) for {scope}.[/green]")

    except Exception as e:
        console.print(Panel(
            f"[red]Retry failed:[/red]\n\n{e}",
            title="Error",
            border_style="red",
        ))
        raise typer.Exit(code=1) from e

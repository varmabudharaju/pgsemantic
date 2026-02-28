"""pgsemantic worker -- background daemon that keeps embeddings in sync."""
from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.panel import Panel

from pgsemantic.config import load_settings

console = Console()


def worker_command(
    db: str | None = typer.Option(
        None, "--db", help="Database URL (overrides DATABASE_URL)"
    ),
) -> None:
    """Start the background worker daemon that processes the embedding queue."""
    settings = load_settings()
    if db is not None:
        settings.database_url = db

    if settings.database_url is None:
        console.print(
            Panel(
                "[red]No database URL provided.[/red]\n\n"
                "Either pass --db or set DATABASE_URL in your .env file.",
                title="Missing Database URL",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    # Configure logging based on settings
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Mask credentials in displayed URL
    display_url = settings.database_url
    if "@" in display_url:
        display_url = display_url.split("@")[-1]

    console.print("[green]Starting worker...[/green]")
    console.print(f"Database: [dim]{display_url}[/dim]")
    console.print(
        f"Batch size: {settings.worker_batch_size}, "
        f"Poll interval: {settings.worker_poll_interval_ms}ms"
    )
    console.print("[dim]Press Ctrl+C to stop.[/dim]")
    console.print()

    try:
        from pgsemantic.worker.daemon import run_worker

        run_worker(settings)
    except ValueError as e:
        console.print(
            Panel(
                f"[red]{e}[/red]",
                title="Configuration Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from e
    except KeyboardInterrupt:
        console.print("\n[yellow]Worker stopped.[/yellow]")
    except Exception as e:
        console.print(
            Panel(
                f"[red]Worker crashed:[/red]\n\n{e}",
                title="Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from e

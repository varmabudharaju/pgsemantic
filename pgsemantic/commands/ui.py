"""pgsemantic ui — launch the web dashboard."""
from __future__ import annotations

import typer
from rich.console import Console

from pgsemantic.config import load_settings

console = Console()


def ui_command(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind to. Use 0.0.0.0 for network access.",
    ),
    port: int = typer.Option(
        8080,
        "--port",
        "-p",
        help="Port to bind to.",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development.",
    ),
) -> None:
    """Launch the pgsemantic web dashboard."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]uvicorn is required for the web UI.[/red]\n\n"
            "Install it with: [cyan]pip install uvicorn[/cyan]"
        )
        raise typer.Exit(code=1)

    console.print()
    console.print(
        f"[green]pgsemantic UI[/green] starting at "
        f"[cyan]http://{host}:{port}[/cyan]"
    )
    console.print("[dim]Press Ctrl+C to stop.[/dim]")
    console.print()

    settings = load_settings()
    uvicorn.run(
        "pgsemantic.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=getattr(settings, "log_level", "info"),
    )

"""CLI command: pgsemantic serve — start MCP server.

Starts the FastMCP server in stdio (for Claude Desktop/Cursor) or
SSE/HTTP mode (for remote agents).
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from pgsemantic.config import load_settings

console = Console()


def serve_command(
    transport: str | None = typer.Option(
        None,
        help="Transport: 'stdio' (default, for Claude Desktop/Cursor) or 'http' (SSE). "
        "Defaults to MCP_TRANSPORT env var or 'stdio'.",
    ),
    port: int | None = typer.Option(
        None,
        help="Port for HTTP/SSE transport (ignored for stdio). "
        "Defaults to MCP_PORT env var or 3000.",
    ),
) -> None:
    """Start the pgsemantic MCP server."""
    settings = load_settings()
    effective_transport = transport or settings.mcp_transport
    effective_port = port or settings.mcp_port

    if not settings.database_url:
        console.print(Panel(
            "[red]DATABASE_URL not set.[/red]\n\n"
            "Set it in your [cyan].env[/cyan] file or as an environment variable:\n"
            "  [cyan]export DATABASE_URL=postgresql://user:pass@host/db[/cyan]",
            title="Configuration Required",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    # Import here to avoid loading mcp at CLI parse time
    from pgsemantic.mcp_server.server import mcp  # noqa: E402

    if effective_transport == "stdio":
        console.print(
            "[green]Starting MCP server[/green] (transport: stdio)"
        )
        mcp.run(transport="stdio")
    elif effective_transport in ("http", "sse"):
        console.print(
            f"[green]Starting MCP server[/green] (transport: SSE, port: {effective_port})"
        )
        # Port is configured via FastMCP settings, not run().
        mcp.settings.port = effective_port
        mcp.run(transport="sse")
    else:
        console.print(Panel(
            f"[red]Unknown transport: '{effective_transport}'[/red]\n\n"
            "Supported transports:\n"
            "  [cyan]stdio[/cyan] — for Claude Desktop, Cursor (default)\n"
            "  [cyan]http[/cyan]  — for remote agents via SSE",
            title="Invalid Transport",
            border_style="red",
        ))
        raise typer.Exit(code=1)

"""CLI command: pgsemantic integrate — set up AI agent integrations.

Currently supports Claude Desktop. Automatically writes the MCP config
so Claude can search your database.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from pgsemantic.config import load_settings

console = Console()


def _get_config_path() -> Path:
    """Return the Claude Desktop config file path for the current platform."""
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        return Path(appdata) / "Claude" / "claude_desktop_config.json"
    else:
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def integrate_command(
    target: str = typer.Argument(
        "claude",
        help="Integration target. Currently supported: 'claude'.",
    ),
) -> None:
    """Set up AI agent integrations (auto-configures Claude Desktop)."""
    if target != "claude":
        console.print(Panel(
            f"[red]Unknown integration target: '{target}'[/red]\n\n"
            "Currently supported targets:\n"
            "  [cyan]claude[/cyan] — Claude Desktop (MCP over stdio)",
            title="Unsupported Target",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    # Get database URL from settings
    settings = load_settings()
    database_url = settings.database_url or "<your-database-url>"

    # Build the MCP server entry
    pgsemantic_bin = shutil.which("pgsemantic")
    if pgsemantic_bin:
        server_entry = {
            "command": pgsemantic_bin,
            "args": ["serve"],
            "env": {
                "DATABASE_URL": database_url,
            },
        }
    else:
        server_entry = {
            "command": sys.executable,
            "args": ["-m", "pgsemantic", "serve"],
            "env": {
                "DATABASE_URL": database_url,
            },
        }

    config_path = _get_config_path()

    # Read existing config or start fresh
    existing_config: dict[str, object] = {}
    if config_path.exists():
        try:
            existing_config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing_config = {}

    # Merge in our server entry
    if "mcpServers" not in existing_config:
        existing_config["mcpServers"] = {}

    mcp_servers = existing_config["mcpServers"]
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
        existing_config["mcpServers"] = mcp_servers

    already_configured = "pgsemantic" in mcp_servers
    mcp_servers["pgsemantic"] = server_entry

    # Write the config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(existing_config, indent=2) + "\n")

    # Show what we did
    action = "Updated" if already_configured else "Added"
    console.print()
    console.print(Panel(
        f"[green]{action} pgsemantic in Claude Desktop config.[/green]\n\n"
        f"Config file: [cyan]{config_path}[/cyan]\n\n"
        "Claude will have access to:\n"
        "  [cyan]semantic_search[/cyan]  — search any indexed table by meaning\n"
        "  [cyan]hybrid_search[/cyan]   — semantic search + SQL filters\n"
        "  [cyan]get_embedding_status[/cyan] — check embedding health",
        title="Claude Desktop Integration",
        border_style="green",
    ))

    # Show the config we wrote
    console.print(Syntax(
        json.dumps({"mcpServers": {"pgsemantic": server_entry}}, indent=2),
        "json",
        theme="monokai",
    ))
    console.print()

    if database_url == "<your-database-url>":
        console.print(
            "[yellow]Note:[/yellow] Set DATABASE_URL in your .env file, "
            "then run this command again to update the config."
        )
        console.print()

    console.print(
        "[green]Restart Claude Desktop[/green] to connect. Then try asking:\n"
        "  [dim]\"Search the medical_qa table for chest pain symptoms\"[/dim]"
    )
    console.print()

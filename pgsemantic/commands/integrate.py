"""CLI command: pgsemantic integrate — generate AI agent config snippets.

Currently supports Claude Desktop. Generates the JSON config needed to
register pgsemantic as an MCP server.
"""
from __future__ import annotations

import json
import platform
import shutil
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def _get_config_path() -> str:
    """Return the Claude Desktop config file path for the current platform."""
    system = platform.system()
    paths = {
        "Darwin": "~/Library/Application Support/Claude/claude_desktop_config.json",
        "Linux": "~/.config/Claude/claude_desktop_config.json",
        "Windows": "%APPDATA%/Claude/claude_desktop_config.json",
    }
    return paths.get(system, "~/.config/Claude/claude_desktop_config.json")


def integrate_command(
    target: str = typer.Argument(
        "claude",
        help="Integration target. Currently supported: 'claude'.",
    ),
) -> None:
    """Generate config snippets for AI agent integrations."""
    if target != "claude":
        console.print(Panel(
            f"[red]Unknown integration target: '{target}'[/red]\n\n"
            "Currently supported targets:\n"
            "  [cyan]claude[/cyan] — Claude Desktop (MCP over stdio)",
            title="Unsupported Target",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    # Use full python path to avoid PATH issues in Claude Desktop
    python_path = sys.executable
    pgsemantic_bin = shutil.which("pgsemantic")

    if pgsemantic_bin:
        config = {
            "mcpServers": {
                "pgsemantic": {
                    "command": pgsemantic_bin,
                    "args": ["serve"],
                    "env": {
                        "DATABASE_URL": "<your-database-url>",
                    },
                },
            },
        }
    else:
        # Fallback: use python3 -m pgsemantic serve
        config = {
            "mcpServers": {
                "pgsemantic": {
                    "command": python_path,
                    "args": ["-m", "pgsemantic", "serve"],
                    "env": {
                        "DATABASE_URL": "<your-database-url>",
                    },
                },
            },
        }

    config_json = json.dumps(config, indent=2)
    config_path = _get_config_path()

    console.print(Panel(
        "Add the following to your Claude Desktop config file:",
        title="Claude Desktop Integration",
        border_style="green",
    ))
    console.print()
    console.print(Syntax(config_json, "json", theme="monokai"))
    console.print()
    console.print(
        f"Config file location: [cyan]{config_path}[/cyan]"
    )
    console.print()
    console.print(
        "[dim]Replace [cyan]<your-database-url>[/cyan] with your actual "
        "PostgreSQL connection string.[/dim]"
    )

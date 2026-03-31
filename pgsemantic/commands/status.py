"""CLI command: pgsemantic status — embedding health dashboard.

Shows a Rich table with embedding coverage, queue depth, and model info
for each watched table in the project config.
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pgsemantic.config import SHADOW_TABLE_PREFIX, load_project_config, load_settings
from pgsemantic.db.client import get_connection
from pgsemantic.db.queue import count_failed, count_pending, get_worker_health
from pgsemantic.db.vectors import count_embedded, count_total_with_content

console = Console()


def status_command(
    database_url: str = typer.Option(
        None,
        "--db",
        "-d",
        help="Database URL (overrides DATABASE_URL env var).",
    ),
) -> None:
    """Show embedding health dashboard for all watched tables."""
    settings = load_settings()
    db_url = database_url or settings.database_url

    if not db_url:
        console.print(Panel(
            "[red]DATABASE_URL not set.[/red]\n\n"
            "Provide it via [cyan]--db[/cyan] or set [cyan]DATABASE_URL[/cyan] "
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
    table.add_column("Storage", style="white")
    table.add_column("Coverage", justify="right")
    table.add_column("Pending", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Migration", style="white")
    table.add_column("Applied", style="dim")

    try:
        with get_connection(db_url) as conn:
            for tc in config.tables:
                try:
                    embedded = count_embedded(
                        conn, tc.table, schema=tc.schema,
                        storage_mode=tc.storage_mode,
                        shadow_table=tc.shadow_table,
                        chunked=tc.chunked,
                    )
                    total = count_total_with_content(
                        conn, tc.table, tc.column, schema=tc.schema,
                        source_columns=tc.source_columns,
                    )
                    pending = count_pending(conn, tc.table)
                    failed = count_failed(conn, tc.table)
                except Exception:
                    # Table may have been dropped — show as missing
                    table.add_row(
                        tc.table,
                        ", ".join(tc.source_columns),
                        tc.model_name,
                        tc.storage_mode,
                        "[red]TABLE MISSING[/red]",
                        "\u2014",
                        "\u2014",
                        "\u2014",
                        tc.applied_at,
                    )
                    continue

                pct = min((embedded / total) * 100, 100.0) if total > 0 else 0.0

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

                # Check for in-progress migration
                migration_table = f"{SHADOW_TABLE_PREFIX}{tc.table}_migration"
                try:
                    result = conn.execute(
                        "SELECT COUNT(*) AS cnt FROM information_schema.tables "
                        "WHERE table_schema = %(schema)s AND table_name = %(tname)s",
                        {"schema": tc.schema, "tname": migration_table},
                    ).fetchone()
                    has_migration = bool(result and int(str(result["cnt"])) > 0)
                except Exception:
                    has_migration = False

                migration_str = (
                    "[yellow]In progress[/yellow]" if has_migration
                    else "[dim]\u2014[/dim]"
                )

                table.add_row(
                    tc.table,
                    ", ".join(tc.source_columns),
                    tc.model_name,
                    tc.storage_mode,
                    coverage_str,
                    pending_str,
                    failed_str,
                    migration_str,
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

    # Worker health section
    try:
        with get_connection(db_url) as conn:
            workers = get_worker_health(conn)

        if workers:
            from datetime import datetime, timezone

            console.print()
            worker_table = Table(title="Worker Health", show_lines=True)
            worker_table.add_column("Worker ID", style="cyan")
            worker_table.add_column("Status")
            worker_table.add_column("Last Seen", style="dim")
            worker_table.add_column("Jobs Processed", justify="right")
            worker_table.add_column("Uptime", style="dim")

            now = datetime.now(tz=timezone.utc)
            for w in workers:
                last_hb = w["last_heartbeat"]
                age_s = (now - last_hb).total_seconds() if hasattr(last_hb, "timestamp") else 999

                if age_s < 30:
                    status_str = "[green]Running[/green]"
                elif age_s < 300:
                    status_str = "[yellow]Stale[/yellow]"
                else:
                    status_str = "[red]Not running[/red]"

                last_seen = str(w["last_heartbeat"])[:19] if w["last_heartbeat"] else "—"
                started = w.get("started_at")
                if started and hasattr(started, "timestamp"):
                    uptime_s = (now - started).total_seconds()
                    hours, remainder = divmod(int(uptime_s), 3600)
                    minutes, _ = divmod(remainder, 60)
                    uptime_str = f"{hours}h {minutes}m"
                else:
                    uptime_str = "—"

                worker_table.add_row(
                    str(w["worker_id"]),
                    status_str,
                    last_seen,
                    str(w["jobs_processed"]),
                    uptime_str,
                )

            console.print(worker_table)
        else:
            console.print("\n[dim]No worker health data. Start a worker with: pgsemantic worker[/dim]")
    except Exception:
        pass  # Worker health is supplementary, don't fail status

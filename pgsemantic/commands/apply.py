"""pgsemantic apply — set up semantic search on a table.

Performs a 10-step setup: extension check, column creation, HNSW index,
queue table, trigger function, trigger installation, config save.
"""
from __future__ import annotations

from datetime import datetime, timezone

import psycopg
import typer
from rich.console import Console
from rich.panel import Panel

from pgsemantic.config import (
    DEFAULT_LOCAL_DIMENSIONS,
    DEFAULT_LOCAL_MODEL,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    OPENAI_DIMENSIONS,
    OPENAI_MODEL,
    ProjectConfig,
    TableConfig,
    load_project_config,
    load_settings,
    save_project_config,
)
from pgsemantic.db.client import ensure_pgvector_extension, get_connection
from pgsemantic.db.introspect import column_exists
from pgsemantic.db.queue import create_queue_table
from pgsemantic.exceptions import ExtensionNotFoundError

console = Console()

# -- SQL Constants ----------------------------------------------------------

# Add a vector column to an existing table.
SQL_ADD_VECTOR_COLUMN = """
    ALTER TABLE "{schema}"."{table}"
    ADD COLUMN "embedding" vector({dimensions});
"""

# Create an HNSW index for cosine similarity search.
# Must be run with conn.autocommit = True because CREATE INDEX CONCURRENTLY
# cannot execute inside a transaction block.
SQL_CREATE_HNSW_INDEX = """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS "idx_{table}_embedding_hnsw"
    ON "{schema}"."{table}"
    USING hnsw ("embedding" vector_cosine_ops)
    WITH (m = {m}, ef_construction = {ef_construction});
"""

# Trigger function that enqueues row changes into pgvector_setup_queue.
# Handles INSERT, UPDATE, and DELETE operations. Uses UPSERT to avoid
# duplicate queue entries for the same row/column combination.
SQL_CREATE_TRIGGER_FUNCTION = """
    CREATE OR REPLACE FUNCTION pgvector_setup_notify_fn()
    RETURNS TRIGGER AS $$
    DECLARE
        v_row_id TEXT;
    BEGIN
        v_row_id := NEW.id::TEXT;
        IF TG_OP = 'DELETE' THEN
            INSERT INTO pgvector_setup_queue (table_name, row_id, column_name, operation)
            VALUES (TG_TABLE_NAME, OLD.id::TEXT, TG_ARGV[0], 'DELETE')
            ON CONFLICT (table_name, row_id, column_name)
            DO UPDATE SET status = 'pending', operation = 'DELETE', retries = 0, error_msg = NULL, updated_at = NOW();
            RETURN OLD;
        ELSE
            INSERT INTO pgvector_setup_queue (table_name, row_id, column_name, operation)
            VALUES (TG_TABLE_NAME, v_row_id, TG_ARGV[0], TG_OP)
            ON CONFLICT (table_name, row_id, column_name)
            DO UPDATE SET status = 'pending', operation = TG_OP, retries = 0, error_msg = NULL, updated_at = NOW();
            RETURN NEW;
        END IF;
    END;
    $$ LANGUAGE plpgsql;
"""

# Install a trigger on a specific table that fires on INSERT, UPDATE, or DELETE
# and enqueues the affected row_id into the queue table.
SQL_CREATE_TRIGGER = """
    CREATE OR REPLACE TRIGGER pgvector_setup_{table}_trigger
    AFTER INSERT OR UPDATE OF "{column}" OR DELETE
    ON "{schema}"."{table}"
    FOR EACH ROW
    EXECUTE FUNCTION pgvector_setup_notify_fn('{column}');
"""


def apply_command(
    table: str = typer.Option(
        ...,
        "--table",
        "-t",
        help="Table to enable semantic search on.",
    ),
    column: str = typer.Option(
        ...,
        "--column",
        "-c",
        help="Text column to embed.",
    ),
    model: str = typer.Option(
        "local",
        "--model",
        "-m",
        help="Embedding provider: 'local' (384d, free) or 'openai' (1536d).",
    ),
    db: str = typer.Option(
        "",
        "--db",
        help="Database URL (overrides DATABASE_URL env var).",
    ),
    schema: str = typer.Option(
        "public",
        "--schema",
        "-s",
        help="Database schema.",
    ),
) -> None:
    """Set up semantic search on a table: extension, column, index, trigger."""
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

    # Resolve model dimensions
    if model == "openai":
        dimensions = OPENAI_DIMENSIONS
        model_name = OPENAI_MODEL
    else:
        dimensions = DEFAULT_LOCAL_DIMENSIONS
        model_name = DEFAULT_LOCAL_MODEL

    console.print()
    console.print(Panel(
        f"Setting up semantic search on [cyan]{schema}.{table}.{column}[/cyan]\n"
        f"Model: [green]{model_name}[/green] ({dimensions} dimensions)",
        title="pgsemantic apply",
        border_style="blue",
    ))
    console.print()

    try:
        with get_connection(database_url, register_vector_type=False) as conn:
            # Step 1: Check/install pgvector extension
            console.print("  [dim][1/9][/dim] Checking pgvector extension...", end=" ")
            pgv_version = ensure_pgvector_extension(conn)
            version_str = ".".join(str(v) for v in pgv_version)
            console.print(f"[green]v{version_str}[/green]")

            # Step 2: Check if source column exists
            console.print("  [dim][2/9][/dim] Verifying source column...", end=" ")
            if not column_exists(conn, table, column, schema):
                console.print("[red]not found[/red]")
                console.print(Panel(
                    f"[red]Column '{column}' not found on table "
                    f"'{schema}.{table}'.[/red]\n\n"
                    "Check the table and column names and try again.",
                    title="Column Not Found",
                    border_style="red",
                ))
                raise typer.Exit(code=1)
            console.print("[green]found[/green]")

            # Step 3: Check if embedding column already exists
            console.print("  [dim][3/9][/dim] Checking embedding column...", end=" ")
            if column_exists(conn, table, "embedding", schema):
                console.print("[yellow]already exists[/yellow]")
                console.print(
                    "  [dim]    Skipping ALTER TABLE — embedding column "
                    "already present.[/dim]"
                )
            else:
                console.print("[dim]needs creation[/dim]")

                # Preview SQL and ask for confirmation
                alter_sql = (
                    f'ALTER TABLE "{schema}"."{table}" '
                    f'ADD COLUMN "embedding" vector({dimensions});'
                )
                console.print()
                console.print(Panel(
                    f"[cyan]{alter_sql}[/cyan]",
                    title="SQL Preview — ALTER TABLE",
                    border_style="yellow",
                ))

                if not typer.confirm(
                    "This will add an embedding column to your table. Continue?"
                ):
                    console.print("[yellow]Aborted.[/yellow]")
                    raise typer.Exit(code=0)

                # Step 4: Add embedding column
                console.print(
                    "  [dim][4/9][/dim] Adding embedding column...",
                    end=" ",
                )
                sql = SQL_ADD_VECTOR_COLUMN.format(
                    schema=schema, table=table, dimensions=dimensions
                )
                conn.execute(sql)
                conn.commit()
                console.print("[green]done[/green]")

            # Step 5: Create HNSW index (requires autocommit)
            console.print(
                "  [dim][5/9][/dim] Creating HNSW index (CONCURRENTLY)...",
                end=" ",
            )

        # CREATE INDEX CONCURRENTLY cannot run inside a transaction block.
        # Open a fresh connection with autocommit enabled.
        with psycopg.connect(database_url, autocommit=True) as autocommit_conn:
            index_sql = SQL_CREATE_HNSW_INDEX.format(
                schema=schema,
                table=table,
                m=HNSW_M,
                ef_construction=HNSW_EF_CONSTRUCTION,
            )
            autocommit_conn.execute(index_sql)
        console.print("[green]done[/green]")

        # Reopen the regular connection for the remaining steps
        with get_connection(database_url) as conn:
            # Step 6: Create queue table
            console.print(
                "  [dim][6/9][/dim] Creating queue table...", end=" "
            )
            create_queue_table(conn)
            console.print("[green]done[/green]")

            # Step 7: Install trigger function
            console.print(
                "  [dim][7/9][/dim] Installing trigger function...", end=" "
            )
            conn.execute(SQL_CREATE_TRIGGER_FUNCTION)
            conn.commit()
            console.print("[green]done[/green]")

            # Step 8: Install trigger on table
            console.print(
                "  [dim][8/9][/dim] Installing trigger on table...", end=" "
            )
            trigger_sql = SQL_CREATE_TRIGGER.format(
                schema=schema, table=table, column=column
            )
            conn.execute(trigger_sql)
            conn.commit()
            console.print("[green]done[/green]")

        # Step 9: Save config
        console.print("  [dim][9/9][/dim] Saving config...", end=" ")
        _save_config(
            table=table,
            schema=schema,
            column=column,
            model=model,
            model_name=model_name,
            dimensions=dimensions,
        )
        console.print("[green]done[/green]")

        # Summary
        console.print()
        console.print(Panel(
            f"[green]Semantic search is ready on "
            f"{schema}.{table}.{column}[/green]\n\n"
            "Next steps:\n"
            f"  1. Embed existing rows:  "
            f"[cyan]pgsemantic index --table {table}[/cyan]\n"
            f"  2. Start live sync:      "
            f"[cyan]pgsemantic worker[/cyan]\n"
            f"  3. Start MCP server:     "
            f"[cyan]pgsemantic serve[/cyan]",
            title="Setup Complete",
            border_style="green",
        ))
        console.print()

    except typer.Exit:
        raise
    except ExtensionNotFoundError as e:
        console.print()
        console.print(Panel(
            f"[red]pgvector extension not found.[/red]\n\n{e}",
            title="Setup Required",
            border_style="red",
        ))
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print()
        console.print(Panel(
            f"[red]Apply failed:[/red]\n\n{e}",
            title="Error",
            border_style="red",
        ))
        raise typer.Exit(code=1) from None


def _save_config(
    table: str,
    schema: str,
    column: str,
    model: str,
    model_name: str,
    dimensions: int,
) -> None:
    """Save or update the .pgsemantic.json project config."""
    from pgsemantic import __version__

    config = load_project_config()
    if config is None:
        config = ProjectConfig(version=__version__)

    # Remove existing config for this table if present
    config.tables = [tc for tc in config.tables if tc.table != table]

    table_config = TableConfig(
        table=table,
        schema=schema,
        column=column,
        embedding_column="embedding",
        model=model,
        model_name=model_name,
        dimensions=dimensions,
        hnsw_m=HNSW_M,
        hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
        applied_at=datetime.now(tz=timezone.utc).isoformat(),
    )
    config.tables.append(table_config)
    save_project_config(config)

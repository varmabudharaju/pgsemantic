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
    SHADOW_TABLE_PREFIX,
    ProjectConfig,
    TableConfig,
    load_project_config,
    load_settings,
    save_project_config,
)
from pgsemantic.db.client import ensure_pgvector_extension, get_connection
from pgsemantic.db.introspect import column_exists, get_primary_key_columns
from pgsemantic.db.queue import create_queue_table
from pgsemantic.db.vectors import create_shadow_table
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

def _build_trigger_function_sql(table: str, pk_columns: list[str]) -> str:
    """Build a per-table trigger function with the correct PK columns.

    Each table gets its own function (pgvector_setup_{table}_fn) so that
    tables with different primary keys don't conflict.

    For single-column PKs: row_id = NEW.pk_col::TEXT
    For composite PKs: row_id = NEW.col1::TEXT || ',' || NEW.col2::TEXT
    """
    if len(pk_columns) == 1:
        new_pk_expr = f'NEW."{pk_columns[0]}"::TEXT'
        old_pk_expr = f'OLD."{pk_columns[0]}"::TEXT'
    else:
        new_pk_expr = " || ',' || ".join(f'NEW."{col}"::TEXT' for col in pk_columns)
        old_pk_expr = " || ',' || ".join(f'OLD."{col}"::TEXT' for col in pk_columns)

    fn_name = f"pgvector_setup_{table}_fn"

    return f"""
    CREATE OR REPLACE FUNCTION {fn_name}()
    RETURNS TRIGGER AS $$
    DECLARE
        v_row_id TEXT;
    BEGIN
        IF TG_OP = 'DELETE' THEN
            v_row_id := {old_pk_expr};
            INSERT INTO pgvector_setup_queue (table_name, row_id, column_name, operation)
            VALUES (TG_TABLE_NAME, v_row_id, TG_ARGV[0], 'DELETE')
            ON CONFLICT (table_name, row_id, column_name)
            DO UPDATE SET status = 'pending', operation = 'DELETE', retries = 0, error_msg = NULL, updated_at = NOW();
            RETURN OLD;
        ELSE
            v_row_id := {new_pk_expr};
            INSERT INTO pgvector_setup_queue (table_name, row_id, column_name, operation)
            VALUES (TG_TABLE_NAME, v_row_id, TG_ARGV[0], TG_OP)
            ON CONFLICT (table_name, row_id, column_name)
            DO UPDATE SET status = 'pending', operation = TG_OP, retries = 0, error_msg = NULL, updated_at = NOW();
            RETURN NEW;
        END IF;
    END;
    $$ LANGUAGE plpgsql;
    """

# Install a per-table trigger that fires on INSERT, UPDATE, or DELETE
# and enqueues the affected row_id into the queue table.
SQL_CREATE_TRIGGER = """
    CREATE OR REPLACE TRIGGER pgvector_setup_{table}_trigger
    AFTER INSERT OR UPDATE OF "{column}" OR DELETE
    ON "{schema}"."{table}"
    FOR EACH ROW
    EXECUTE FUNCTION pgvector_setup_{table}_fn('{column}');
"""

# Trigger for multi-column mode: fires on UPDATE of any watched column.
SQL_CREATE_TRIGGER_MULTI = """
    CREATE OR REPLACE TRIGGER pgvector_setup_{table}_trigger
    AFTER INSERT OR UPDATE OF {column_list} OR DELETE
    ON "{schema}"."{table}"
    FOR EACH ROW
    EXECUTE FUNCTION pgvector_setup_{table}_fn('{column_arg}');
"""

# HNSW index on shadow table (CONCURRENTLY).
SQL_CREATE_SHADOW_HNSW_INDEX = """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS "idx_{shadow_table}_embedding_hnsw"
    ON "{schema}"."{shadow_table}"
    USING hnsw ("embedding" vector_cosine_ops)
    WITH (m = {m}, ef_construction = {ef_construction});
"""


def apply_command(
    table: str = typer.Option(
        ...,
        "--table",
        "-t",
        help="Table to enable semantic search on.",
    ),
    column: str = typer.Option(
        "",
        "--column",
        "-c",
        help="Text column to embed (mutually exclusive with --columns).",
    ),
    columns: str = typer.Option(
        "",
        "--columns",
        help="Comma-separated columns to concatenate and embed (e.g. 'title,description').",
    ),
    model: str = typer.Option(
        "local",
        "--model",
        "-m",
        help="Embedding provider: 'local' (384d, free) or 'openai' (1536d).",
    ),
    external: bool = typer.Option(
        False,
        "--external",
        help="Store embeddings in a separate shadow table (don't modify source table).",
    ),
    db: str = typer.Option(
        "",
        "--db",
        "-d",
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

    # Validate --column / --columns (mutually exclusive, one required)
    if column and columns:
        console.print(Panel(
            "[red]Cannot use both --column and --columns.[/red]\n\n"
            "Use --column for a single column or --columns for multiple.",
            title="Invalid Options",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    if not column and not columns:
        console.print(Panel(
            "[red]Either --column or --columns is required.[/red]\n\n"
            "Example: --column description\n"
            "    or: --columns title,description",
            title="Missing Column",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    # Parse column list
    source_columns: list[str] = (
        [c.strip() for c in columns.split(",") if c.strip()]
        if columns
        else [column]
    )
    # Use the first column as the "primary" column for backward compat
    primary_column = source_columns[0]
    is_multi = len(source_columns) > 1

    # Resolve storage mode
    storage_mode = "external" if external else "inline"
    shadow_table_name = f"{SHADOW_TABLE_PREFIX}{table}" if external else None

    # Resolve model dimensions
    if model == "openai":
        dimensions = OPENAI_DIMENSIONS
        model_name = OPENAI_MODEL
    else:
        dimensions = DEFAULT_LOCAL_DIMENSIONS
        model_name = DEFAULT_LOCAL_MODEL

    col_display = ", ".join(source_columns)
    console.print()
    console.print(Panel(
        f"Setting up semantic search on [cyan]{schema}.{table}[/cyan]\n"
        f"Column(s): [cyan]{col_display}[/cyan]\n"
        f"Model: [green]{model_name}[/green] ({dimensions} dimensions)\n"
        f"Storage: [green]{storage_mode}[/green]"
        + (f" → {shadow_table_name}" if shadow_table_name else ""),
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

            # Step 2: Check if source column(s) exist and detect primary key
            console.print("  [dim][2/9][/dim] Verifying source column(s)...", end=" ")
            for src_col in source_columns:
                if not column_exists(conn, table, src_col, schema):
                    console.print("[red]not found[/red]")
                    console.print(Panel(
                        f"[red]Column '{src_col}' not found on table "
                        f"'{schema}.{table}'.[/red]\n\n"
                        "Check the table and column names and try again.",
                        title="Column Not Found",
                        border_style="red",
                    ))
                    raise typer.Exit(code=1)

            pk_columns = get_primary_key_columns(conn, table, schema)
            if not pk_columns:
                console.print("[red]no primary key[/red]")
                console.print(Panel(
                    f"[red]Table '{schema}.{table}' has no primary key.[/red]\n\n"
                    "pgsemantic requires a primary key to track row changes.\n"
                    "Add one with: ALTER TABLE {table} ADD PRIMARY KEY (column);",
                    title="Primary Key Required",
                    border_style="red",
                ))
                raise typer.Exit(code=1)
            pk_display = ", ".join(pk_columns)
            console.print(f"[green]found[/green] (PK: {pk_display})")

            # Step 3: Embedding storage setup
            if external:
                # External mode: create shadow table
                console.print(
                    "  [dim][3/9][/dim] Creating shadow table...", end=" "
                )
                create_shadow_table(
                    conn,
                    shadow_table=shadow_table_name,
                    dimensions=dimensions,
                    schema=schema,
                )
                console.print("[green]done[/green]")
                console.print(
                    f"  [dim]    Your table remains unchanged. Embeddings "
                    f"stored in {shadow_table_name}[/dim]"
                )
            else:
                # Inline mode: check/add embedding column on source table
                console.print("  [dim][3/9][/dim] Checking embedding column...", end=" ")
                if column_exists(conn, table, "embedding", schema):
                    console.print("[yellow]already exists[/yellow]")
                    console.print(
                        "  [dim]    Skipping ALTER TABLE — embedding column "
                        "already present.[/dim]"
                    )
                else:
                    console.print("[dim]needs creation[/dim]")

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
        with psycopg.connect(database_url, autocommit=True) as autocommit_conn:
            if external:
                index_sql = SQL_CREATE_SHADOW_HNSW_INDEX.format(
                    schema=schema,
                    shadow_table=shadow_table_name,
                    m=HNSW_M,
                    ef_construction=HNSW_EF_CONSTRUCTION,
                )
            else:
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

            # Step 7: Install trigger function (PK-aware)
            console.print(
                "  [dim][7/9][/dim] Installing trigger function...", end=" "
            )
            trigger_fn_sql = _build_trigger_function_sql(table, pk_columns)
            conn.execute(trigger_fn_sql)
            conn.commit()
            console.print("[green]done[/green]")

            # Step 8: Install trigger on table
            console.print(
                "  [dim][8/9][/dim] Installing trigger on table...", end=" "
            )
            if is_multi:
                # Multi-column trigger: fires on UPDATE OF col1, col2, ...
                column_list = ", ".join(f'"{c}"' for c in source_columns)
                # Escape single quotes in column names for the SQL string literal
                column_arg = "+".join(c.replace("'", "''") for c in source_columns)
                trigger_sql = SQL_CREATE_TRIGGER_MULTI.format(
                    schema=schema,
                    table=table,
                    column_list=column_list,
                    column_arg=column_arg,
                )
            else:
                trigger_sql = SQL_CREATE_TRIGGER.format(
                    schema=schema, table=table,
                    column=primary_column.replace("'", "''"),
                )
            conn.execute(trigger_sql)
            conn.commit()
            console.print("[green]done[/green]")

        # Step 9: Save config
        console.print("  [dim][9/9][/dim] Saving config...", end=" ")
        _save_config(
            table=table,
            schema=schema,
            column=primary_column,
            model=model,
            model_name=model_name,
            dimensions=dimensions,
            primary_key=pk_columns,
            columns=source_columns if is_multi else None,
            storage_mode=storage_mode,
            shadow_table=shadow_table_name,
        )
        console.print("[green]done[/green]")

        # Summary
        console.print()
        storage_note = (
            f"\nEmbeddings stored in: [cyan]{shadow_table_name}[/cyan]"
            if external else ""
        )
        console.print(Panel(
            f"[green]Semantic search is ready on "
            f"{schema}.{table} ({col_display})[/green]"
            f"{storage_note}\n\n"
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
    primary_key: list[str] | None = None,
    columns: list[str] | None = None,
    storage_mode: str = "inline",
    shadow_table: str | None = None,
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
        primary_key=primary_key or ["id"],
        columns=columns,
        storage_mode=storage_mode,
        shadow_table=shadow_table,
    )
    config.tables.append(table_config)
    save_project_config(config)

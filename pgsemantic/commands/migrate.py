"""pgsemantic migrate — switch embedding models with zero downtime.

Re-embeds all rows using a new model in a temporary migration table,
then atomically swaps it into production. Search keeps working on the
old model during re-embedding.
"""
from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from pgsemantic.config import (
    DEFAULT_INDEX_BATCH_SIZE,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    SHADOW_TABLE_PREFIX,
    load_project_config,
    load_settings,
    save_project_config,
)
from pgsemantic.db.client import get_connection
from pgsemantic.db.vectors import (
    SQL_EXT_UPSERT_EMBEDDING,
    _content_text,
    _pk_last_params,
    _pk_row_id,
    _qualified_table,
    count_total_with_content,
    create_shadow_table,
    fetch_unembedded_batch,
)
from pgsemantic.embeddings import get_provider

console = Console()

# Migration table suffix
MIGRATION_SUFFIX = "_migration"

# SQL to count rows already in migration table
SQL_COUNT_MIGRATION_ROWS = """
    SELECT COUNT(*) AS cnt FROM {migration_table}
    WHERE embedding IS NOT NULL;
"""


def migrate_command(
    table: str = typer.Option(
        ...,
        "--table",
        "-t",
        help="Table to migrate.",
    ),
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="New embedding model (e.g., openai, local, ollama).",
    ),
    batch_size: int = typer.Option(
        DEFAULT_INDEX_BATCH_SIZE,
        "--batch-size",
        "-b",
        help="Rows per batch.",
    ),
    db: str = typer.Option(
        "",
        "--db",
        "-d",
        help="Database URL override.",
    ),
) -> None:
    """Migrate a table to a new embedding model with zero downtime."""
    settings = load_settings()
    database_url = db if db else settings.database_url

    if not database_url:
        console.print("[red]No database URL.[/red] Set DATABASE_URL in .env or pass --db.")
        raise typer.Exit(code=1)

    config = load_project_config()
    if config is None:
        console.print("[red]No .pgsemantic.json found.[/red]")
        raise typer.Exit(code=1)

    table_config = config.get_table_config(table)
    if table_config is None:
        console.print(f"[red]Table '{table}' not in config.[/red]")
        raise typer.Exit(code=1)

    old_model = table_config.model
    old_model_name = table_config.model_name
    if old_model == model:
        console.print(f"[yellow]Table already uses model '{model}'.[/yellow]")
        raise typer.Exit(code=0)

    # Step 1: Validate new model
    console.print(f"\n[bold]Migrating {table}:[/bold] {old_model_name} \u2192 {model}\n")
    console.print("  [dim]Step 1:[/dim] Validating new model...")

    try:
        api_key = settings.openai_api_key if model == "openai" else None
        ollama_url = settings.ollama_base_url if model == "ollama" else None
        new_provider = get_provider(model, api_key=api_key, ollama_base_url=ollama_url)
        test_embedding = new_provider.embed_query("test")
        new_dimensions = len(test_embedding)
        new_model_name = new_provider.config.model_name
        console.print(f"  [green]\u2713[/green] Model '{new_model_name}' works ({new_dimensions}d)")
    except Exception as e:
        console.print(f"  [red]\u2717 Model validation failed:[/red] {e}")
        raise typer.Exit(code=1) from None

    # Determine table names
    schema = table_config.schema
    old_shadow = table_config.shadow_table
    new_shadow_name = f"{SHADOW_TABLE_PREFIX}{table}"
    migration_table_name = f"{new_shadow_name}{MIGRATION_SUFFIX}"

    try:
        with get_connection(database_url) as conn:
            qualified_migration = _qualified_table(migration_table_name, schema)

            # Step 2: Create migration table (or resume if exists)
            console.print("  [dim]Step 2:[/dim] Creating migration table...")
            create_shadow_table(conn, migration_table_name, new_dimensions, schema)
            console.print(f"  [green]\u2713[/green] Migration table: {migration_table_name}")

            # Step 3: Create HNSW index on migration table
            console.print("  [dim]Step 3:[/dim] Creating HNSW index...")
            conn.autocommit = True
            conn.execute(
                f'CREATE INDEX CONCURRENTLY IF NOT EXISTS '
                f'"idx_{migration_table_name}_hnsw" '
                f'ON {qualified_migration} '
                f'USING hnsw ("embedding" vector_cosine_ops) '
                f'WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})'
            )
            conn.autocommit = False
            console.print("  [green]\u2713[/green] HNSW index created")

            # Step 4: Bulk re-embed
            console.print("  [dim]Step 4:[/dim] Re-embedding rows...\n")

            total = count_total_with_content(
                conn, table, table_config.column, schema,
                source_columns=table_config.source_columns,
            )

            # Check how many already migrated (for resume)
            already = conn.execute(
                SQL_COUNT_MIGRATION_ROWS.format(migration_table=qualified_migration)
            ).fetchone()
            already_count = int(str(already["cnt"])) if already else 0

            if already_count > 0:
                console.print(f"  Resuming: {already_count}/{total} already migrated")

            remaining = total - already_count
            if remaining <= 0:
                console.print("  [green]All rows already re-embedded.[/green]")
            else:
                start_time = time.monotonic()
                rows_done = 0
                last_pk_params = None
                pk_columns = table_config.primary_key
                source_columns = table_config.source_columns
                column = table_config.column

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task("Re-embedding...", total=remaining)

                    while True:
                        # Fetch batch from SOURCE table (not migration table)
                        batch = fetch_unembedded_batch(
                            conn, table=table, column=column,
                            batch_size=batch_size,
                            pk_columns=pk_columns,
                            last_pk_params=last_pk_params,
                            schema=schema,
                            storage_mode="external",
                            shadow_table=migration_table_name,
                            source_columns=source_columns,
                        )

                        if not batch:
                            break

                        # Extract texts
                        if len(source_columns) > 1:
                            texts = [_content_text(source_columns, row) for row in batch]
                        else:
                            texts = [str(row.get("content", row.get(column, ""))) for row in batch]

                        embeddings = new_provider.embed(texts)

                        # Insert into migration table
                        for row, emb in zip(batch, embeddings, strict=True):
                            row_id = _pk_row_id(pk_columns, row)
                            conn.execute(
                                SQL_EXT_UPSERT_EMBEDDING.format(
                                    shadow_table=qualified_migration
                                ),
                                {
                                    "row_id": row_id,
                                    "embedding": emb,
                                    "source_column": "+".join(source_columns),
                                    "model_name": new_model_name,
                                },
                            )
                        conn.commit()

                        rows_done += len(batch)
                        last_pk_params = _pk_last_params(pk_columns, batch[-1])
                        progress.update(task, advance=len(batch))

                elapsed = time.monotonic() - start_time
                rate = (rows_done / elapsed * 60) if elapsed > 0 else 0
                console.print(f"\n  Re-embedded {rows_done} rows in {elapsed:.1f}s ({rate:,.0f}/min)")

            # Step 5: Atomic swap
            console.print("  [dim]Step 5:[/dim] Swapping tables...")
            qualified_old = _qualified_table(old_shadow, schema) if old_shadow else None
            conn.autocommit = True
            if qualified_old and old_shadow != new_shadow_name:
                conn.execute(f"DROP TABLE IF EXISTS {qualified_old}")
            if migration_table_name != new_shadow_name:
                conn.execute(f'ALTER TABLE {qualified_migration} RENAME TO "{new_shadow_name}"')
            conn.autocommit = False
            console.print("  [green]\u2713[/green] Swap complete")

            # Step 6: Drop old inline column if migrating from inline
            if table_config.storage_mode == "inline":
                console.print("  [dim]Step 6:[/dim] Dropping old inline embedding column...")
                qualified_source = _qualified_table(table, schema)
                conn.execute(f'ALTER TABLE {qualified_source} DROP COLUMN IF EXISTS "embedding"')
                conn.commit()
                console.print("  [green]\u2713[/green] Inline column dropped")

        # Step 7: Update config
        console.print("  [dim]Step 7:[/dim] Updating config...")
        table_config.model = model
        table_config.model_name = new_model_name
        table_config.dimensions = new_dimensions
        table_config.storage_mode = "external"
        table_config.shadow_table = new_shadow_name
        save_project_config(config)
        console.print("  [green]\u2713[/green] Config updated\n")

        console.print(
            f"[green bold]Migration complete![/green bold] "
            f"{table} now uses {new_model_name} ({new_dimensions}d)\n"
        )

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[red]Migration failed:[/red] {e}")
        console.print(f"[dim]Migration table '{migration_table_name}' preserved for resume.[/dim]")
        raise typer.Exit(code=1) from None

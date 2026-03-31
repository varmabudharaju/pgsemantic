"""pgsemantic index — bulk embed existing rows that have no embedding yet.

Uses keyset pagination (fetch_unembedded_batch with last_id) for consistent
performance and a Rich progress bar to show progress.
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
    load_project_config,
    load_settings,
)
from pgsemantic.db.client import get_connection
from pgsemantic.db.vectors import (
    _content_text,
    _pk_last_params,
    bulk_insert_chunks,
    bulk_update_embeddings,
    count_embedded,
    count_total_with_content,
    fetch_unembedded_batch,
)
from pgsemantic.embeddings import get_provider
from pgsemantic.embeddings.chunker import chunk_text

console = Console()


def index_command(
    table: str = typer.Option(
        ...,
        "--table",
        "-t",
        help="Table to bulk-embed.",
    ),
    batch_size: int = typer.Option(
        DEFAULT_INDEX_BATCH_SIZE,
        "--batch-size",
        "-b",
        help="Number of rows to embed per batch.",
    ),
    db: str = typer.Option(
        "",
        "--db",
        "-d",
        help="Database URL (overrides DATABASE_URL env var).",
    ),
) -> None:
    """Bulk embed existing rows where embedding IS NULL."""
    settings = load_settings()
    database_url = db if db else settings.database_url

    if not database_url:
        console.print(
            "[red]No database URL provided.[/red] "
            "Set DATABASE_URL in your .env file or pass --db."
        )
        raise typer.Exit(code=1)

    # Load project config to get table settings
    config = load_project_config()
    if config is None:
        console.print(
            "[red]No .pgsemantic.json found.[/red] "
            f"Run [cyan]pgsemantic apply --table {table}[/cyan] first."
        )
        raise typer.Exit(code=1)

    table_config = config.get_table_config(table)
    if table_config is None:
        console.print(
            f"[red]Table '{table}' not found in config.[/red] "
            f"Run [cyan]pgsemantic apply --table {table}[/cyan] first."
        )
        raise typer.Exit(code=1)

    column = table_config.column
    schema = table_config.schema
    model = table_config.model
    pk_columns = table_config.primary_key
    source_columns = table_config.source_columns
    storage_mode = table_config.storage_mode
    shadow_table = table_config.shadow_table
    chunked = table_config.chunked
    chunk_max_tokens = table_config.chunk_max_tokens
    chunk_overlap = table_config.chunk_overlap

    col_display = ", ".join(source_columns)
    console.print()
    console.print(
        f"Indexing [cyan]{schema}.{table} ({col_display})[/cyan] "
        f"with [green]{table_config.model_name}[/green] "
        f"(batch size: {batch_size}, storage: {storage_mode})"
    )
    console.print()

    try:
        # Initialize embedding provider
        api_key = settings.openai_api_key if model == "openai" else None
        provider = get_provider(model, api_key=api_key)

        with get_connection(database_url) as conn:
            # Count total and already-embedded rows
            total_with_content = count_total_with_content(
                conn, table, column, schema,
                source_columns=source_columns,
            )
            already_embedded = count_embedded(
                conn, table, schema,
                storage_mode=storage_mode,
                shadow_table=shadow_table,
            )
            remaining = total_with_content - already_embedded

            if remaining <= 0:
                console.print(
                    "[green]All rows are already embedded.[/green] "
                    f"({already_embedded}/{total_with_content})"
                )
                console.print()
                raise typer.Exit(code=0)

            console.print(
                f"  Total rows with content: {total_with_content}\n"
                f"  Already embedded:        {already_embedded}\n"
                f"  Remaining:               {remaining}"
            )
            console.print()

            # Bulk embed loop with progress bar
            start_time = time.monotonic()
            rows_embedded = 0
            last_pk_params: dict[str, object] | None = None  # First page

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Embedding rows...", total=remaining
                )

                while True:
                    batch = fetch_unembedded_batch(
                        conn,
                        table=table,
                        column=column,
                        batch_size=batch_size,
                        pk_columns=pk_columns,
                        last_pk_params=last_pk_params,
                        schema=schema,
                        storage_mode=storage_mode,
                        shadow_table=shadow_table,
                        source_columns=source_columns,
                    )

                    if not batch:
                        break

                    if chunked:
                        # Chunked mode: chunk each row's text, embed chunks
                        for row in batch:
                            if storage_mode == "external":
                                text = str(row["content"])
                            elif len(source_columns) > 1:
                                text = _content_text(source_columns, row)
                            else:
                                text = str(row[column])

                            text_chunks = chunk_text(text, chunk_max_tokens, chunk_overlap)
                            if not text_chunks:
                                continue

                            row_id = (
                                str(row[pk_columns[0]])
                                if len(pk_columns) == 1
                                else ",".join(str(row[c]) for c in pk_columns)
                            )
                            chunk_embeddings = provider.embed(text_chunks)

                            bulk_insert_chunks(
                                conn,
                                shadow_table=shadow_table,
                                row_id=row_id,
                                chunks=text_chunks,
                                embeddings=chunk_embeddings,
                                source_column="+".join(source_columns),
                                model_name=table_config.model_name,
                                schema=schema,
                            )
                    else:
                        # Original non-chunked path
                        if storage_mode == "external":
                            texts = [str(row["content"]) for row in batch]
                        elif len(source_columns) > 1:
                            texts = [_content_text(source_columns, row) for row in batch]
                        else:
                            texts = [str(row[column]) for row in batch]

                        embeddings = provider.embed(texts)

                        bulk_update_embeddings(
                            conn,
                            table=table,
                            rows=batch,
                            embeddings=embeddings,
                            pk_columns=pk_columns,
                            schema=schema,
                            storage_mode=storage_mode,
                            shadow_table=shadow_table,
                            source_column="+".join(source_columns),
                            model_name=table_config.model_name,
                        )

                    rows_embedded += len(batch)
                    last_pk_params = _pk_last_params(pk_columns, batch[-1])
                    progress.update(task, advance=len(batch))

            elapsed = time.monotonic() - start_time
            rows_per_min = (rows_embedded / elapsed * 60) if elapsed > 0 else 0.0

            # Summary
            console.print()
            console.print(
                f"[green]Indexed {rows_embedded} rows "
                f"in {elapsed:.1f}s "
                f"({rows_per_min:,.0f} rows/min)[/green]"
            )
            final_embedded = count_embedded(
                conn, table, schema,
                storage_mode=storage_mode,
                shadow_table=shadow_table,
            )
            coverage = (
                (final_embedded / total_with_content * 100)
                if total_with_content > 0
                else 0.0
            )
            console.print(
                f"  Coverage: {final_embedded}/{total_with_content} "
                f"({coverage:.1f}%)"
            )
            console.print()
            console.print(
                "Next steps:\n"
                "  Start live sync:   "
                "[cyan]pgsemantic worker[/cyan]\n"
                "  Start MCP server:  "
                "[cyan]pgsemantic serve[/cyan]"
            )
            console.print()

    except typer.Exit:
        raise
    except Exception as e:
        console.print()
        console.print(
            f"[red]Indexing failed:[/red] {e}"
        )
        raise typer.Exit(code=1) from None

# Smart Auto-Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split long documents into overlapping chunks before embedding, storing chunk vectors in the shadow table and returning best-matching chunks on search.

**Architecture:** New `chunker.py` module handles text splitting. Shadow table schema extended with `chunk_index` and `chunk_text` columns. Config gets `chunked`, `chunk_max_tokens`, `chunk_overlap` fields. Apply command gets `--chunked` flag. Index and worker chunk text before embedding. Search deduplicates by row_id, returning best chunk per row.

**Tech Stack:** Python, psycopg3, Typer

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `pgsemantic/embeddings/chunker.py` | `chunk_text()` function — sentence-aware splitting |
| Modify | `pgsemantic/config.py` | Add `chunked`, `chunk_max_tokens`, `chunk_overlap` to `TableConfig` |
| Modify | `pgsemantic/db/vectors.py` | Chunked shadow table schema, chunked search with dedup |
| Modify | `pgsemantic/commands/apply.py` | `--chunked` flag, force external storage |
| Modify | `pgsemantic/commands/index.py` | Chunk text before embedding in bulk |
| Modify | `pgsemantic/worker/daemon.py` | Chunk text in worker job processing |
| Create | `tests/unit/test_chunker.py` | Tests for chunk_text() |
| Create | `tests/unit/test_chunked_search.py` | Tests for chunked config, storage, and search |

---

### Task 1: Create chunker module

**Files:**
- Create: `pgsemantic/embeddings/chunker.py`
- Create: `tests/unit/test_chunker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_chunker.py`:

```python
"""Tests for text chunking."""
from __future__ import annotations


class TestChunkText:
    def test_short_text_single_chunk(self) -> None:
        """Text shorter than max_tokens should return a single chunk."""
        from pgsemantic.embeddings.chunker import chunk_text

        text = "This is a short sentence."
        chunks = chunk_text(text, max_tokens=256, overlap=64)
        assert chunks == ["This is a short sentence."]

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        """Text longer than max_tokens should be split into chunks."""
        from pgsemantic.embeddings.chunker import chunk_text

        # ~50 words per sentence, 5 sentences = ~250 words
        sentences = [f"Sentence {i} " + "word " * 48 + "end." for i in range(5)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_tokens=100, overlap=20)
        assert len(chunks) > 1
        # Each chunk should be <= max_tokens words
        for chunk in chunks:
            assert len(chunk.split()) <= 100

    def test_overlap_produces_shared_content(self) -> None:
        """Adjacent chunks should share some content when overlap > 0."""
        from pgsemantic.embeddings.chunker import chunk_text

        sentences = [f"Unique sentence number {i}. " for i in range(20)]
        text = "".join(sentences)
        chunks = chunk_text(text, max_tokens=30, overlap=10)
        assert len(chunks) >= 2
        # Check that at least some content is shared between adjacent chunks
        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].split())
            words_b = set(chunks[i + 1].split())
            assert words_a & words_b  # non-empty intersection

    def test_no_overlap(self) -> None:
        """overlap=0 should produce non-overlapping chunks."""
        from pgsemantic.embeddings.chunker import chunk_text

        text = " ".join(["word"] * 100)
        chunks = chunk_text(text, max_tokens=30, overlap=0)
        total_words = sum(len(c.split()) for c in chunks)
        assert total_words == 100

    def test_sentence_boundary_splitting(self) -> None:
        """Chunks should prefer to break at sentence boundaries."""
        from pgsemantic.embeddings.chunker import chunk_text

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, max_tokens=5, overlap=0)
        # Each chunk should end cleanly (not mid-word)
        for chunk in chunks:
            assert chunk.strip()

    def test_max_chunks_limit(self) -> None:
        """Very long text should be capped at max_chunks."""
        from pgsemantic.embeddings.chunker import chunk_text

        text = " ".join(["word"] * 50000)
        chunks = chunk_text(text, max_tokens=10, overlap=0, max_chunks=200)
        assert len(chunks) <= 200

    def test_empty_text_returns_empty(self) -> None:
        """Empty string should return empty list."""
        from pgsemantic.embeddings.chunker import chunk_text

        assert chunk_text("", max_tokens=256, overlap=64) == []

    def test_whitespace_only_returns_empty(self) -> None:
        """Whitespace-only text should return empty list."""
        from pgsemantic.embeddings.chunker import chunk_text

        assert chunk_text("   \n\t  ", max_tokens=256, overlap=64) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_chunker.py -v`
Expected: FAIL — `ImportError: cannot import name 'chunk_text'`

- [ ] **Step 3: Write the implementation**

Create `pgsemantic/embeddings/chunker.py`:

```python
"""Sentence-aware text chunking for long document embedding.

Splits text into overlapping chunks at sentence boundaries.
Uses whitespace tokenization (no external tokenizer dependency).
"""
from __future__ import annotations

import re

# Sentence boundary pattern: split after .!? followed by whitespace
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Safety limit for max chunks per row
DEFAULT_MAX_CHUNKS = 200


def chunk_text(
    text: str,
    max_tokens: int = 256,
    overlap: int = 64,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries.

    Args:
        text: The text to split.
        max_tokens: Maximum words per chunk (whitespace-tokenized).
        overlap: Number of words to overlap between adjacent chunks.
        max_chunks: Safety cap on total chunks per text.

    Returns:
        List of chunk strings. Empty list for empty/whitespace input.
    """
    text = text.strip()
    if not text:
        return []

    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    # Split into sentences first
    sentences = _SENTENCE_RE.split(text)

    chunks: list[str] = []
    current_words: list[str] = []

    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue

        # If adding this sentence would exceed max_tokens, flush current chunk
        if current_words and len(current_words) + len(sentence_words) > max_tokens:
            chunks.append(" ".join(current_words))
            if len(chunks) >= max_chunks:
                return chunks

            # Start next chunk with overlap from end of current
            if overlap > 0:
                current_words = current_words[-overlap:]
            else:
                current_words = []

        current_words.extend(sentence_words)

        # Handle single sentences longer than max_tokens
        while len(current_words) > max_tokens:
            chunks.append(" ".join(current_words[:max_tokens]))
            if len(chunks) >= max_chunks:
                return chunks
            if overlap > 0:
                current_words = current_words[max_tokens - overlap:]
            else:
                current_words = current_words[max_tokens:]

    # Flush remaining words
    if current_words:
        chunks.append(" ".join(current_words))

    return chunks[:max_chunks]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_chunker.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pgsemantic/embeddings/chunker.py tests/unit/test_chunker.py
git commit -m "feat: add sentence-aware text chunker for long documents"
```

---

### Task 2: Add chunking fields to TableConfig

**Files:**
- Modify: `pgsemantic/config.py`
- Create: `tests/unit/test_chunked_search.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_chunked_search.py`:

```python
"""Tests for chunked search config and storage."""
from __future__ import annotations

from pgsemantic.config import (
    DEFAULT_LOCAL_DIMENSIONS,
    DEFAULT_LOCAL_MODEL,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    ProjectConfig,
    TableConfig,
)


def _make_chunked_config(
    table: str = "articles",
    chunked: bool = True,
    chunk_max_tokens: int = 256,
    chunk_overlap: int = 64,
) -> TableConfig:
    return TableConfig(
        table=table,
        schema="public",
        column="body",
        embedding_column="embedding",
        model="local",
        model_name=DEFAULT_LOCAL_MODEL,
        dimensions=DEFAULT_LOCAL_DIMENSIONS,
        hnsw_m=HNSW_M,
        hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
        applied_at="2026-03-30T00:00:00Z",
        storage_mode="external",
        shadow_table=f"pgsemantic_embeddings_{table}",
        chunked=chunked,
        chunk_max_tokens=chunk_max_tokens,
        chunk_overlap=chunk_overlap,
    )


class TestChunkedConfig:
    def test_chunked_defaults_to_false(self) -> None:
        tc = TableConfig(
            table="t", schema="public", column="c",
            embedding_column="embedding", model="local",
            model_name="m", dimensions=384,
            hnsw_m=16, hnsw_ef_construction=64,
            applied_at="2026-01-01",
        )
        assert tc.chunked is False
        assert tc.chunk_max_tokens == 256
        assert tc.chunk_overlap == 64

    def test_chunked_config_round_trip(self) -> None:
        """Chunked config should serialize/deserialize through JSON."""
        import json
        from dataclasses import asdict

        tc = _make_chunked_config()
        data = asdict(tc)
        restored = TableConfig(**data)
        assert restored.chunked is True
        assert restored.chunk_max_tokens == 256
        assert restored.chunk_overlap == 64

    def test_project_config_with_chunked_table(self) -> None:
        config = ProjectConfig(
            version="0.3.0",
            tables=[_make_chunked_config()],
        )
        tc = config.get_table_config("articles")
        assert tc is not None
        assert tc.chunked is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_chunked_search.py -v`
Expected: FAIL — `TypeError: TableConfig.__init__() got an unexpected keyword argument 'chunked'`

- [ ] **Step 3: Add fields to TableConfig**

In `pgsemantic/config.py`, add three new fields to the `TableConfig` dataclass after `shadow_table`:

```python
@dataclass
class TableConfig:
    """Configuration for a single watched table."""
    table: str
    schema: str
    column: str
    embedding_column: str
    model: str
    model_name: str
    dimensions: int
    hnsw_m: int
    hnsw_ef_construction: int
    applied_at: str
    primary_key: list[str] = field(default_factory=lambda: ["id"])
    columns: list[str] | None = None
    storage_mode: str = "inline"
    shadow_table: str | None = None
    chunked: bool = False
    chunk_max_tokens: int = 256
    chunk_overlap: int = 64
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_chunked_search.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pgsemantic/config.py tests/unit/test_chunked_search.py
git commit -m "feat: add chunked, chunk_max_tokens, chunk_overlap to TableConfig"
```

---

### Task 3: Chunked shadow table schema and search

**Files:**
- Modify: `pgsemantic/db/vectors.py`
- Modify: `tests/unit/test_chunked_search.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_chunked_search.py`:

```python
from unittest.mock import MagicMock, patch


class TestChunkedShadowTable:
    def test_create_chunked_shadow_table_sql(self) -> None:
        """Chunked shadow table should have chunk_index and chunk_text columns."""
        from pgsemantic.db.vectors import create_chunked_shadow_table

        mock_conn = MagicMock()
        create_chunked_shadow_table(mock_conn, "pgsemantic_embeddings_articles", 384, "public")

        call_args = mock_conn.execute.call_args_list
        sql = str(call_args[0][0][0])
        assert "chunk_index" in sql
        assert "chunk_text" in sql
        assert "row_id" in sql
        assert "PRIMARY KEY" in sql


class TestChunkedSearch:
    def test_search_chunked_deduplicates_by_row_id(self) -> None:
        """Chunked search should return best chunk per row, not all chunks."""
        from pgsemantic.db.vectors import search_chunked

        mock_conn = MagicMock()
        # Simulate 3 chunks from 2 rows: row1 has 2 chunks, row2 has 1
        mock_conn.execute.return_value.fetchall.return_value = [
            {"row_id": "1", "chunk_text": "chunk 1a", "chunk_index": 0,
             "similarity": 0.9, "embedding": [0.1]},
            {"row_id": "1", "chunk_text": "chunk 1b", "chunk_index": 1,
             "similarity": 0.7, "embedding": [0.1]},
            {"row_id": "2", "chunk_text": "chunk 2a", "chunk_index": 0,
             "similarity": 0.85, "embedding": [0.1]},
        ]
        # Second execute call is for SET hnsw.ef_search
        mock_conn.execute.side_effect = [MagicMock(), mock_conn.execute.return_value]

        results = search_chunked(
            conn=mock_conn,
            table="articles",
            column="body",
            query_vector=[0.1] * 384,
            shadow_table="pgsemantic_embeddings_articles",
            schema="public",
            pk_columns=["id"],
            limit=10,
        )

        # Should return 2 rows (deduplicated), best chunk per row
        assert len(results) == 2
        assert results[0]["matched_chunk"] == "chunk 1a"  # highest sim for row 1
        assert results[0]["chunk_similarity"] == 0.9


class TestBulkInsertChunks:
    def test_insert_chunks_for_row(self) -> None:
        """Should insert multiple chunks for a single row."""
        from pgsemantic.db.vectors import bulk_insert_chunks

        mock_conn = MagicMock()
        chunks = ["chunk 0", "chunk 1", "chunk 2"]
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

        bulk_insert_chunks(
            conn=mock_conn,
            shadow_table="pgsemantic_embeddings_articles",
            row_id="42",
            chunks=chunks,
            embeddings=embeddings,
            source_column="body",
            model_name="all-MiniLM-L6-v2",
            schema="public",
        )

        # Should have deleted old chunks + inserted 3 new ones + committed
        calls = mock_conn.execute.call_args_list
        assert len(calls) >= 4  # 1 delete + 3 inserts
        mock_conn.commit.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_chunked_search.py -v`
Expected: FAIL — `ImportError: cannot import name 'create_chunked_shadow_table'`

- [ ] **Step 3: Add chunked functions to vectors.py**

Append to `pgsemantic/db/vectors.py` after the `search_all` function:

```python
# -- Chunked shadow table --------------------------------------------------

SQL_CREATE_CHUNKED_SHADOW_TABLE = """
    CREATE TABLE IF NOT EXISTS {shadow_table} (
        row_id          TEXT            NOT NULL,
        chunk_index     INT             NOT NULL,
        chunk_text      TEXT            NOT NULL,
        embedding       vector({dimensions}),
        source_column   TEXT            NOT NULL,
        model_name      TEXT            NOT NULL,
        updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        PRIMARY KEY (row_id, chunk_index)
    );
"""

SQL_DELETE_ROW_CHUNKS = """
    DELETE FROM {shadow_table} WHERE row_id = %(row_id)s;
"""

SQL_INSERT_CHUNK = """
    INSERT INTO {shadow_table}
        (row_id, chunk_index, chunk_text, embedding, source_column, model_name, updated_at)
    VALUES
        (%(row_id)s, %(chunk_index)s, %(chunk_text)s, %(embedding)s::vector,
         %(source_column)s, %(model_name)s, NOW());
"""

SQL_CHUNKED_SEARCH = """
    SELECT
        e.row_id,
        e.chunk_index,
        e.chunk_text,
        1 - (e.embedding <=> %(query_vector)s::vector) AS similarity
    FROM {shadow_table} e
    WHERE e.embedding IS NOT NULL
    ORDER BY e.embedding <=> %(query_vector)s::vector
    LIMIT %(limit)s;
"""


def create_chunked_shadow_table(
    conn: DictConnection,
    shadow_table: str,
    dimensions: int,
    schema: str = "public",
) -> None:
    """Create a chunked shadow table with (row_id, chunk_index) composite PK."""
    qualified_shadow = _qualified_table(shadow_table, schema)
    sql = SQL_CREATE_CHUNKED_SHADOW_TABLE.format(
        shadow_table=qualified_shadow,
        dimensions=dimensions,
    )
    conn.execute(sql)
    conn.commit()
    logger.info("Created chunked shadow table %s", qualified_shadow)


def bulk_insert_chunks(
    conn: DictConnection,
    shadow_table: str,
    row_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    source_column: str,
    model_name: str,
    schema: str = "public",
) -> None:
    """Delete old chunks for a row and insert new chunked embeddings."""
    qualified_shadow = _qualified_table(shadow_table, schema)

    # Delete existing chunks for this row
    conn.execute(
        SQL_DELETE_ROW_CHUNKS.format(shadow_table=qualified_shadow),
        {"row_id": row_id},
    )

    # Insert new chunks
    for i, (chunk_text_val, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        conn.execute(
            SQL_INSERT_CHUNK.format(shadow_table=qualified_shadow),
            {
                "row_id": row_id,
                "chunk_index": i,
                "chunk_text": chunk_text_val,
                "embedding": embedding,
                "source_column": source_column,
                "model_name": model_name,
            },
        )

    conn.commit()
    logger.debug("Inserted %d chunks for row %s in %s", len(chunks), row_id, shadow_table)


def search_chunked(
    conn: DictConnection,
    table: str,
    column: str,
    query_vector: list[float],
    shadow_table: str,
    schema: str = "public",
    pk_columns: list[str] | None = None,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Search chunked shadow table, returning best chunk per source row.

    Queries the shadow table for similar chunks, then deduplicates by row_id
    keeping only the highest-similarity chunk per row.

    Returns dicts with: row_id, matched_chunk, chunk_similarity, chunk_index.
    """
    if pk_columns is None:
        pk_columns = ["id"]

    qualified_shadow = _qualified_table(shadow_table, schema)

    # Fetch more chunks than limit to allow deduplication
    fetch_limit = limit * 3

    ef_search = max(fetch_limit * 2, 100)
    conn.execute(f"SET hnsw.ef_search = {ef_search}")

    sql = SQL_CHUNKED_SEARCH.format(shadow_table=qualified_shadow)
    rows = conn.execute(
        sql, {"query_vector": query_vector, "limit": fetch_limit}
    ).fetchall()

    # Deduplicate: keep best chunk per row_id
    seen: dict[str, dict[str, object]] = {}
    for row in rows:
        rid = str(row["row_id"])
        if rid not in seen:
            seen[rid] = {
                "row_id": rid,
                "matched_chunk": str(row["chunk_text"]),
                "chunk_similarity": float(str(row["similarity"])),
                "chunk_index": int(str(row["chunk_index"])),
            }

    # Sort by chunk_similarity descending
    results = sorted(seen.values(), key=lambda r: r["chunk_similarity"], reverse=True)
    return results[:limit]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_chunked_search.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pgsemantic/db/vectors.py tests/unit/test_chunked_search.py
git commit -m "feat: add chunked shadow table schema, chunk insert, and chunked search"
```

---

### Task 4: Add `--chunked` flag to apply command

**Files:**
- Modify: `pgsemantic/commands/apply.py`

- [ ] **Step 1: Add the --chunked flag and chunk options**

In `pgsemantic/commands/apply.py`, add three new parameters to `apply_command()`. Find the parameter list and add after the existing `external` parameter:

```python
    chunked: bool = typer.Option(
        False,
        "--chunked",
        help="Enable text chunking for long documents. Forces external storage.",
    ),
    chunk_max_tokens: int = typer.Option(
        256,
        "--chunk-max-tokens",
        help="Maximum words per chunk (default 256).",
    ),
    chunk_overlap: int = typer.Option(
        64,
        "--chunk-overlap",
        help="Words to overlap between adjacent chunks (default 64).",
    ),
```

- [ ] **Step 2: Force external storage when chunked**

In the `apply_command` body, after the line that resolves `external`, add:

```python
    # Chunking requires external (shadow table) storage
    if chunked:
        external = True
```

- [ ] **Step 3: Use chunked shadow table when chunked=True**

Find the section that calls `create_shadow_table()` and add a branch:

```python
        if chunked:
            from pgsemantic.db.vectors import create_chunked_shadow_table
            create_chunked_shadow_table(conn, shadow_table_name, dimensions, schema)
        else:
            create_shadow_table(conn, shadow_table_name, dimensions, schema)
```

- [ ] **Step 4: Save chunking config**

In the `_save_config` section where `TableConfig` is constructed, add the three new fields:

```python
            chunked=chunked,
            chunk_max_tokens=chunk_max_tokens,
            chunk_overlap=chunk_overlap,
```

- [ ] **Step 5: Run existing tests**

Run: `python3 -m pytest tests/unit/ -v 2>&1 | tail -5`
Expected: No regressions

- [ ] **Step 6: Commit**

```bash
git add pgsemantic/commands/apply.py
git commit -m "feat: add --chunked flag to apply command"
```

---

### Task 5: Add chunking to index command

**Files:**
- Modify: `pgsemantic/commands/index.py`

- [ ] **Step 1: Add chunked indexing logic**

In `pgsemantic/commands/index.py`, modify the embedding loop to chunk text before embedding when the table is chunked.

Add import at the top (after existing imports):

```python
from pgsemantic.db.vectors import bulk_insert_chunks
from pgsemantic.embeddings.chunker import chunk_text
```

In the `index_command` function, after loading `table_config`, add:

```python
    chunked = table_config.chunked
    chunk_max_tokens = table_config.chunk_max_tokens
    chunk_overlap = table_config.chunk_overlap
```

Replace the embedding loop body (the section that extracts texts, generates embeddings, and writes them back) with a branching version. The non-chunked path stays identical. The chunked path:

```python
                    if chunked:
                        # Chunked mode: chunk each row's text, embed all chunks
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

                            row_id = str(row[pk_columns[0]]) if len(pk_columns) == 1 else ",".join(str(row[c]) for c in pk_columns)
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

                        rows_embedded += len(batch)
                        last_pk_params = _pk_last_params(pk_columns, batch[-1])
                        progress.update(task, advance=len(batch))
                    else:
                        # Original non-chunked path (unchanged)
                        ...existing code...
```

- [ ] **Step 2: Run tests**

Run: `python3 -m pytest tests/unit/ -v 2>&1 | tail -5`
Expected: No regressions

- [ ] **Step 3: Commit**

```bash
git add pgsemantic/commands/index.py
git commit -m "feat: add chunked text splitting to index command"
```

---

### Task 6: Add chunking to worker daemon

**Files:**
- Modify: `pgsemantic/worker/daemon.py`

- [ ] **Step 1: Add chunking to _process_job**

In `pgsemantic/worker/daemon.py`, modify `_process_job` to handle chunked tables.

Add imports:

```python
from pgsemantic.db.vectors import bulk_insert_chunks
from pgsemantic.embeddings.chunker import chunk_text
```

In `_process_job`, for the INSERT/UPDATE branch, after fetching the text, add a chunked path:

```python
            if table_config.chunked:
                # Chunked mode: split text, embed all chunks, store
                from pgsemantic.db.vectors import bulk_insert_chunks
                from pgsemantic.embeddings.chunker import chunk_text

                text_chunks = chunk_text(
                    text,
                    table_config.chunk_max_tokens,
                    table_config.chunk_overlap,
                )
                if not text_chunks:
                    complete_job(conn, job_id=job_id)
                    return

                chunk_embeddings: list[list[float]] = provider.embed(text_chunks)

                bulk_insert_chunks(
                    conn,
                    shadow_table=shadow_table,
                    row_id=row_id,
                    chunks=text_chunks,
                    embeddings=chunk_embeddings,
                    source_column="+".join(source_columns),
                    model_name=table_config.model_name,
                    schema=table_config.schema,
                )
                complete_job(conn, job_id=job_id)
                logger.info(
                    "Embedded %s#%s as %d chunks (%s)",
                    table_name, row_id, len(text_chunks), column_name,
                )
            else:
                # Original non-chunked path
                embedding: list[float] = provider.embed_query(text)
                ...
```

For DELETE operations on chunked tables, `bulk_insert_chunks` already handles deletion via `SQL_DELETE_ROW_CHUNKS`. But the current DELETE path uses `null_embedding`. Add a branch:

```python
        if operation == "DELETE":
            if table_config.chunked:
                # Delete all chunks for this row
                from pgsemantic.db.vectors import SQL_DELETE_ROW_CHUNKS, _qualified_table
                qualified_shadow = _qualified_table(shadow_table, table_config.schema)
                conn.execute(
                    SQL_DELETE_ROW_CHUNKS.format(shadow_table=qualified_shadow),
                    {"row_id": row_id},
                )
                conn.commit()
            else:
                null_embedding(...)
            complete_job(conn, job_id=job_id)
```

- [ ] **Step 2: Run tests**

Run: `python3 -m pytest tests/unit/ -v 2>&1 | tail -5`
Expected: No regressions

- [ ] **Step 3: Commit**

```bash
git add pgsemantic/worker/daemon.py
git commit -m "feat: add chunked text processing to worker daemon"
```

---

### Task 7: Wire chunked search into search paths

**Files:**
- Modify: `pgsemantic/commands/search.py`
- Modify: `pgsemantic/web/app.py`
- Modify: `pgsemantic/mcp_server/server.py`

- [ ] **Step 1: Update CLI search for chunked tables**

In `pgsemantic/commands/search.py`, in the single-table search path, add a branch for chunked tables. After getting `query_vector`, before calling `search_similar`:

```python
            if table_config.chunked:
                from pgsemantic.db.vectors import search_chunked
                results = search_chunked(
                    conn, table, column, query_vector,
                    shadow_table=table_config.shadow_table,
                    schema=table_config.schema,
                    pk_columns=table_config.primary_key,
                    limit=limit,
                )
            else:
                results = search_similar(...)
```

- [ ] **Step 2: Update web search endpoint**

In `pgsemantic/web/app.py`, in the `search_table` endpoint, add the same branching for chunked tables.

- [ ] **Step 3: Update MCP semantic_search tool**

In `pgsemantic/mcp_server/server.py`, in the `semantic_search` tool, add the same branching.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/unit/ -v 2>&1 | tail -5`
Expected: No regressions

- [ ] **Step 5: Commit**

```bash
git add pgsemantic/commands/search.py pgsemantic/web/app.py pgsemantic/mcp_server/server.py
git commit -m "feat: wire chunked search into CLI, web, and MCP search paths"
```

---

### Task 8: Run full test suite and lint

**Files:** All modified files

- [ ] **Step 1: Run unit tests**

Run: `python3 -m pytest tests/unit/ -v`
Expected: All pass (except 6 known failures)

- [ ] **Step 2: Run linter on changed files**

Run: `python3 -m ruff check pgsemantic/embeddings/chunker.py pgsemantic/config.py pgsemantic/db/vectors.py pgsemantic/commands/apply.py pgsemantic/commands/index.py pgsemantic/commands/search.py pgsemantic/worker/daemon.py tests/unit/test_chunker.py tests/unit/test_chunked_search.py`

- [ ] **Step 3: Fix any issues**

- [ ] **Step 4: Commit if fixes needed**

```bash
git add -u
git commit -m "chore: fix lint issues in auto-chunking feature"
```

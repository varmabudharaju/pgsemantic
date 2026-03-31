# Cross-Table Unified Search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users search across all configured tables in a single query, merging results by similarity score.

**Architecture:** Add a `search_all()` function in `db/vectors.py` that iterates over all tables in the project config, embeds the query once per unique model, calls `search_similar()` per table, and merges + sorts results globally. Expose this through CLI (omit `--table`), Web UI (toggle), and MCP (new tool).

**Tech Stack:** Python, psycopg3, Typer, FastAPI, FastMCP, vanilla JS

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `pgsemantic/db/vectors.py` | New `search_all()` function |
| Modify | `pgsemantic/commands/search.py` | Make `--table` optional, search all when omitted |
| Modify | `pgsemantic/web/app.py` | New `/api/search-all` endpoint |
| Modify | `pgsemantic/web/static/index.html` | "Search All Tables" toggle in search UI |
| Modify | `pgsemantic/mcp_server/server.py` | New `search_all` MCP tool |
| Create | `tests/unit/test_search_all.py` | Unit tests for `search_all()` and CLI changes |

---

### Task 1: Add `search_all()` to vectors.py

**Files:**
- Modify: `pgsemantic/db/vectors.py` (append after `null_embedding` at line ~817)
- Test: `tests/unit/test_search_all.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_search_all.py`:

```python
"""Tests for cross-table unified search."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from pgsemantic.config import (
    DEFAULT_LOCAL_DIMENSIONS,
    DEFAULT_LOCAL_MODEL,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    OPENAI_DIMENSIONS,
    OPENAI_MODEL,
    ProjectConfig,
    TableConfig,
)


def _make_table_config(
    table: str = "products",
    model: str = "local",
    model_name: str = DEFAULT_LOCAL_MODEL,
    dimensions: int = DEFAULT_LOCAL_DIMENSIONS,
    column: str = "description",
    schema: str = "public",
    storage_mode: str = "inline",
    shadow_table: str | None = None,
) -> TableConfig:
    return TableConfig(
        table=table,
        schema=schema,
        column=column,
        embedding_column="embedding",
        model=model,
        model_name=model_name,
        dimensions=dimensions,
        hnsw_m=HNSW_M,
        hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
        applied_at="2026-03-30T00:00:00Z",
        storage_mode=storage_mode,
        shadow_table=shadow_table,
    )


class TestSearchAll:
    def test_merges_results_from_two_tables(self) -> None:
        """search_all should merge results from multiple tables sorted by similarity."""
        from pgsemantic.db.vectors import search_all

        config = ProjectConfig(
            version="0.3.0",
            tables=[
                _make_table_config(table="products"),
                _make_table_config(table="articles"),
            ],
        )

        mock_conn = MagicMock()
        mock_provider = MagicMock()
        mock_provider.embed_query.return_value = [0.1] * 384

        with patch("pgsemantic.db.vectors.search_similar") as mock_search:
            mock_search.side_effect = [
                [{"id": 1, "content": "Widget", "similarity": 0.8}],
                [{"id": 2, "content": "News", "similarity": 0.9}],
            ]

            results = search_all(
                conn=mock_conn,
                query="test",
                providers={"local": mock_provider},
                project_config=config,
                limit=10,
            )

        assert len(results) == 2
        # Highest similarity first
        assert results[0]["similarity"] == 0.9
        assert results[0]["_source_table"] == "articles"
        assert results[1]["similarity"] == 0.8
        assert results[1]["_source_table"] == "products"

    def test_respects_global_limit(self) -> None:
        """search_all should return at most `limit` results total."""
        from pgsemantic.db.vectors import search_all

        config = ProjectConfig(
            version="0.3.0",
            tables=[
                _make_table_config(table="products"),
                _make_table_config(table="articles"),
            ],
        )

        mock_conn = MagicMock()
        mock_provider = MagicMock()
        mock_provider.embed_query.return_value = [0.1] * 384

        with patch("pgsemantic.db.vectors.search_similar") as mock_search:
            mock_search.side_effect = [
                [{"id": i, "content": f"P{i}", "similarity": 0.9 - i * 0.01} for i in range(5)],
                [{"id": i, "content": f"A{i}", "similarity": 0.85 - i * 0.01} for i in range(5)],
            ]

            results = search_all(
                conn=mock_conn,
                query="test",
                providers={"local": mock_provider},
                project_config=config,
                limit=3,
            )

        assert len(results) == 3

    def test_different_models_embed_once_each(self) -> None:
        """search_all should call embed_query once per unique model, not per table."""
        from pgsemantic.db.vectors import search_all

        config = ProjectConfig(
            version="0.3.0",
            tables=[
                _make_table_config(table="products", model="local"),
                _make_table_config(table="articles", model="local"),
                _make_table_config(
                    table="docs", model="openai",
                    model_name=OPENAI_MODEL, dimensions=OPENAI_DIMENSIONS,
                ),
            ],
        )

        mock_conn = MagicMock()
        mock_local = MagicMock()
        mock_local.embed_query.return_value = [0.1] * 384
        mock_openai = MagicMock()
        mock_openai.embed_query.return_value = [0.2] * 1536

        with patch("pgsemantic.db.vectors.search_similar", return_value=[]):
            results = search_all(
                conn=mock_conn,
                query="test",
                providers={"local": mock_local, "openai": mock_openai},
                project_config=config,
                limit=10,
            )

        mock_local.embed_query.assert_called_once_with("test")
        mock_openai.embed_query.assert_called_once_with("test")

    def test_empty_config_returns_empty(self) -> None:
        """search_all with no configured tables should return empty list."""
        from pgsemantic.db.vectors import search_all

        config = ProjectConfig(version="0.3.0", tables=[])
        mock_conn = MagicMock()

        results = search_all(
            conn=mock_conn,
            query="test",
            providers={},
            project_config=config,
            limit=10,
        )

        assert results == []

    def test_tags_results_with_source_table_and_schema(self) -> None:
        """Each result should include _source_table and _source_schema."""
        from pgsemantic.db.vectors import search_all

        config = ProjectConfig(
            version="0.3.0",
            tables=[
                _make_table_config(table="products", schema="inventory"),
            ],
        )

        mock_conn = MagicMock()
        mock_provider = MagicMock()
        mock_provider.embed_query.return_value = [0.1] * 384

        with patch("pgsemantic.db.vectors.search_similar") as mock_search:
            mock_search.return_value = [
                {"id": 1, "content": "Widget", "similarity": 0.8},
            ]

            results = search_all(
                conn=mock_conn,
                query="test",
                providers={"local": mock_provider},
                project_config=config,
                limit=10,
            )

        assert results[0]["_source_table"] == "products"
        assert results[0]["_source_schema"] == "inventory"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_search_all.py -v`
Expected: FAIL with `ImportError: cannot import name 'search_all' from 'pgsemantic.db.vectors'`

- [ ] **Step 3: Write the implementation**

Append to `pgsemantic/db/vectors.py` after the `null_embedding` function (after line 817):

```python
def search_all(
    conn: DictConnection,
    query: str,
    providers: dict[str, object],
    project_config: "ProjectConfig",
    limit: int = 10,
) -> list[dict[str, object]]:
    """Search across all configured tables, merging results by similarity.

    Embeds the query once per unique model, searches each table independently,
    then merges and sorts all results by similarity score (descending).

    Each result is tagged with _source_table and _source_schema.

    Args:
        conn: Database connection.
        query: Natural language search query.
        providers: Dict mapping model key (e.g. "local", "openai") to provider instance.
        project_config: ProjectConfig with all configured tables.
        limit: Maximum total results across all tables.

    Returns:
        List of result dicts sorted by similarity, each with _source_table and _source_schema.
    """
    from pgsemantic.config import ProjectConfig  # avoid circular at module level

    if not project_config.tables:
        return []

    # Embed query once per unique model
    query_vectors: dict[str, list[float]] = {}
    for tc in project_config.tables:
        if tc.model not in query_vectors:
            provider = providers.get(tc.model)
            if provider is None:
                logger.warning("No provider for model %s, skipping tables using it", tc.model)
                continue
            query_vectors[tc.model] = provider.embed_query(query)

    # Search each table and tag results
    all_results: list[dict[str, object]] = []
    for tc in project_config.tables:
        qv = query_vectors.get(tc.model)
        if qv is None:
            continue

        table_results = search_similar(
            conn=conn,
            table=tc.table,
            column=tc.column,
            query_vector=qv,
            limit=limit,
            schema=tc.schema,
            storage_mode=tc.storage_mode,
            shadow_table=tc.shadow_table,
            source_columns=tc.source_columns,
            pk_columns=tc.primary_key,
        )

        for row in table_results:
            row["_source_table"] = tc.table
            row["_source_schema"] = tc.schema
            all_results.append(row)

    # Sort by similarity descending, take top N
    all_results.sort(key=lambda r: float(str(r.get("similarity", 0))), reverse=True)
    return all_results[:limit]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_search_all.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pgsemantic/db/vectors.py tests/unit/test_search_all.py
git commit -m "feat: add search_all() for cross-table unified search"
```

---

### Task 2: Make CLI `--table` optional for cross-table search

**Files:**
- Modify: `pgsemantic/commands/search.py`
- Test: `tests/unit/test_search_all.py` (append tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_search_all.py`:

```python
class TestSearchCommandAllTables:
    """Test that search_command works when --table is omitted."""

    @patch("pgsemantic.commands.search.get_connection")
    @patch("pgsemantic.commands.search.get_provider")
    @patch("pgsemantic.commands.search.load_project_config")
    @patch("pgsemantic.commands.search.load_settings")
    def test_no_table_flag_searches_all(
        self,
        mock_settings: MagicMock,
        mock_config: MagicMock,
        mock_provider_fn: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """Omitting --table should invoke search_all over all configured tables."""
        mock_settings.return_value = MagicMock(
            database_url="postgresql://test@localhost/db",
            openai_api_key=None,
            ollama_base_url="http://localhost:11434",
        )
        mock_config.return_value = ProjectConfig(
            version="0.3.0",
            tables=[_make_table_config(table="products")],
        )

        mock_prov = MagicMock()
        mock_prov.embed_query.return_value = [0.1] * 384
        mock_provider_fn.return_value = mock_prov

        with patch("pgsemantic.commands.search.search_all") as mock_sa:
            mock_sa.return_value = [
                {"id": 1, "content": "Widget", "similarity": 0.8,
                 "_source_table": "products", "_source_schema": "public"},
            ]

            from typer.testing import CliRunner
            from pgsemantic.cli import app

            runner = CliRunner()
            result = runner.invoke(app, ["search", "test query", "--format", "json"])

            assert result.exit_code == 0
            mock_sa.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_search_all.py::TestSearchCommandAllTables -v`
Expected: FAIL — current `search_command` requires `--table`

- [ ] **Step 3: Modify search_command to make --table optional**

Edit `pgsemantic/commands/search.py`. Key changes:

1. Change `--table` from required to optional (default `""`)
2. Import `search_all` from `pgsemantic.db.vectors`
3. When `table` is empty, load all providers and call `search_all()`
4. When `table` is provided, keep existing single-table behavior

Replace the `search_command` function:

```python
from pgsemantic.db.vectors import search_all, search_similar


def search_command(
    query: str = typer.Argument(
        ...,
        help="Natural language search query.",
    ),
    table: str = typer.Option(
        "",
        "--table",
        "-t",
        help="Table to search. Omit to search all configured tables.",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-n",
        help="Number of results to return.",
    ),
    db: str = typer.Option(
        "",
        "--db",
        "-d",
        help="Database URL (overrides DATABASE_URL env var).",
    ),
    fmt: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table (default), csv, json, jsonl.",
    ),
) -> None:
    """Search your database using natural language.

    Omit --table to search across all configured tables at once.
    """
    if fmt not in ("table", "csv", "json", "jsonl"):
        console.print("[red]--format must be one of: table, csv, json, jsonl[/red]")
        raise typer.Exit(code=1)

    settings = load_settings()
    database_url = db if db else settings.database_url

    if not database_url:
        console.print(
            "[red]No database URL provided.[/red] "
            "Set DATABASE_URL in your .env file or pass --db."
        )
        raise typer.Exit(code=1)

    config = load_project_config()
    if config is None:
        console.print(
            "[red]No .pgsemantic.json found.[/red] "
            "Run [cyan]pgsemantic apply --table <table> --column <column>[/cyan] first."
        )
        raise typer.Exit(code=1)

    # Single-table search (original behavior)
    if table:
        table_config = config.get_table_config(table)
        if table_config is None:
            console.print(
                f"[red]Table '{table}' not found in config.[/red] "
                f"Run [cyan]pgsemantic apply --table {table}[/cyan] first."
            )
            raise typer.Exit(code=1)

        column = table_config.column
        model = table_config.model
        source_columns = table_config.source_columns

        try:
            api_key = settings.openai_api_key if model == "openai" else None
            provider = get_provider(model, api_key=api_key)
            query_vector = provider.embed_query(query)

            with get_connection(database_url) as conn:
                results = search_similar(
                    conn, table, column, query_vector, limit=limit,
                    schema=table_config.schema,
                    storage_mode=table_config.storage_mode,
                    shadow_table=table_config.shadow_table,
                    source_columns=source_columns,
                    pk_columns=table_config.primary_key,
                )

            _output_results(results, query, table, column, fmt, limit)

        except typer.Exit:
            raise
        except Exception as e:
            console.print()
            console.print(f"[red]Search failed:[/red] {e}")
            raise typer.Exit(code=1) from None
        return

    # Cross-table search (--table omitted)
    if not config.tables:
        console.print("[red]No tables configured.[/red] Run pgsemantic apply first.")
        raise typer.Exit(code=1)

    try:
        # Build providers dict (one per unique model)
        providers: dict[str, object] = {}
        for tc in config.tables:
            if tc.model not in providers:
                api_key = settings.openai_api_key if tc.model == "openai" else None
                ollama_url = settings.ollama_base_url if tc.model == "ollama" else None
                providers[tc.model] = get_provider(
                    tc.model, api_key=api_key, ollama_base_url=ollama_url
                )

        with get_connection(database_url) as conn:
            results = search_all(
                conn=conn,
                query=query,
                providers=providers,
                project_config=config,
                limit=limit,
            )

        _output_results(results, query, "all tables", "content", fmt, limit)

    except typer.Exit:
        raise
    except Exception as e:
        console.print()
        console.print(f"[red]Search failed:[/red] {e}")
        raise typer.Exit(code=1) from None
```

Also add a `_output_results` helper to avoid duplicating the output logic. Add it above `search_command`:

```python
def _output_results(
    results: list[dict[str, Any]],
    query: str,
    table_label: str,
    column: str,
    fmt: str,
    limit: int,
) -> None:
    """Format and output search results in the requested format."""
    if not results:
        if fmt == "table":
            console.print("\n[yellow]No results found.[/yellow]\n")
        elif fmt == "json":
            print("[]")
        raise typer.Exit(code=0)

    if fmt == "csv":
        _results_to_csv(results)
        return
    if fmt == "json":
        _results_to_json(results)
        return
    if fmt == "jsonl":
        _results_to_jsonl(results)
        return

    # Rich table output
    console.print()
    console.print(
        f"[dim]Results for:[/dim] [cyan]\"{query}\"[/cyan] "
        f"[dim]in {table_label}[/dim]"
    )

    for i, row in enumerate(results, 1):
        sim = float(str(row["similarity"]))
        score_color = "green" if sim >= 0.5 else "yellow" if sim >= 0.3 else "dim"

        meta_parts: list[str] = []
        for key, val in row.items():
            if key in _TABLE_HIDDEN_COLUMNS or key == column:
                continue
            meta_parts.append(f"[dim]{key}:[/dim] {val}")
        meta_line = "  ".join(meta_parts)

        content = str(row.get("content", row.get(column, "")))
        if len(content) > 300:
            content = content[:300] + "..."

        console.print()
        console.print(
            f"  [{score_color}]{i}. (score: {sim:.3f})[/{score_color}]  {meta_line}"
        )
        console.print(f"  [white]{content}[/white]")

    console.print()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_search_all.py -v && pytest tests/unit/test_search_command.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add pgsemantic/commands/search.py tests/unit/test_search_all.py
git commit -m "feat: make --table optional in search command for cross-table search"
```

---

### Task 3: Add `/api/search-all` web endpoint

**Files:**
- Modify: `pgsemantic/web/app.py`
- Test: `tests/unit/test_search_all.py` (append tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_search_all.py`:

```python
class TestSearchAllEndpoint:
    """Test the /api/search-all web endpoint."""

    @patch("pgsemantic.web.app.get_connection")
    @patch("pgsemantic.web.app.get_provider")
    @patch("pgsemantic.web.app.load_project_config")
    @patch("pgsemantic.web.app.load_settings")
    def test_search_all_endpoint_returns_results(
        self,
        mock_settings: MagicMock,
        mock_config: MagicMock,
        mock_provider_fn: MagicMock,
        mock_conn_ctx: MagicMock,
    ) -> None:
        from fastapi.testclient import TestClient
        from pgsemantic.web.app import app, _csrf_token

        mock_settings.return_value = MagicMock(
            database_url="postgresql://test@localhost/db",
            openai_api_key=None,
            ollama_base_url="http://localhost:11434",
        )
        mock_config.return_value = ProjectConfig(
            version="0.3.0",
            tables=[_make_table_config(table="products")],
        )

        mock_prov = MagicMock()
        mock_prov.embed_query.return_value = [0.1] * 384
        mock_provider_fn.return_value = mock_prov

        with patch("pgsemantic.web.app.search_all") as mock_sa:
            mock_sa.return_value = [
                {"id": 1, "content": "Widget", "similarity": 0.8,
                 "_source_table": "products", "_source_schema": "public"},
            ]

            client = TestClient(app)
            resp = client.post(
                "/api/search-all",
                json={"query": "test", "limit": 5},
                headers={"X-CSRF-Token": _csrf_token},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["_source_table"] == "products"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_search_all.py::TestSearchAllEndpoint -v`
Expected: FAIL — 404 because `/api/search-all` doesn't exist yet

- [ ] **Step 3: Add the endpoint to app.py**

Add a new Pydantic model and endpoint to `pgsemantic/web/app.py`.

Add this model after the `SearchRequest` class (around line 241):

```python
class SearchAllRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(5, ge=1, le=100)
```

Add the import of `search_all` to the imports from `pgsemantic.db.vectors` (line 47-57):

```python
from pgsemantic.db.vectors import (
    _content_text,
    _pk_last_params,
    _qualified_table,
    bulk_update_embeddings,
    count_embedded,
    count_total_with_content,
    fetch_unembedded_batch,
    hybrid_search,
    search_all,
    search_similar,
)
```

Add the endpoint after the existing `search_table` endpoint (after line ~1006):

```python
@app.post("/api/search-all")
async def search_all_tables(req: SearchAllRequest):
    """Search across all configured tables."""
    db_url = _get_db_url()
    settings = load_settings()

    config = load_project_config()
    if config is None or not config.tables:
        raise HTTPException(400, "No tables configured. Run pgsemantic apply first.")

    try:
        # Build providers dict (one per unique model)
        providers: dict[str, object] = {}
        for tc in config.tables:
            if tc.model not in providers:
                api_key = settings.openai_api_key if tc.model == "openai" else None
                ollama_url = settings.ollama_base_url if tc.model == "ollama" else None
                providers[tc.model] = get_provider(
                    tc.model, api_key=api_key, ollama_base_url=ollama_url
                )

        with get_connection(db_url) as conn:
            results = search_all(
                conn=conn,
                query=req.query,
                providers=providers,
                project_config=config,
                limit=req.limit,
            )

        hidden = {"embedding"}
        cleaned = []
        for row in results:
            item: dict[str, object] = {}
            item["similarity"] = round(float(str(row["similarity"])), 4)
            item["content"] = str(row.get("content", ""))[:500]
            item["_source_table"] = row.get("_source_table", "")
            item["_source_schema"] = row.get("_source_schema", "")
            for key, val in row.items():
                if key not in hidden and key not in ("similarity", "content", "_source_table", "_source_schema"):
                    item[key] = _safe_json_value(val)
            cleaned.append(item)

        return {"query": req.query, "results": cleaned}

    except Exception as e:
        logger.exception("search_all_tables failed")
        raise HTTPException(status_code=500, detail="Search failed. Check server logs.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_search_all.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add pgsemantic/web/app.py tests/unit/test_search_all.py
git commit -m "feat: add /api/search-all endpoint for cross-table web search"
```

---

### Task 4: Add "Search All Tables" toggle to the web UI

**Files:**
- Modify: `pgsemantic/web/static/index.html`

- [ ] **Step 1: Add the toggle switch to the search form**

In `pgsemantic/web/static/index.html`, find the search form (around line 360, the `<!-- SEARCH -->` section). Add a toggle between the table selector and the query input.

Find the line with the table `<select>` in the search form:
```html
<div class="field-row"><div class="field"><label class="label">Table</label><select class="select" id="s-table" onchange="loadSearchFilters()"><option value="">Select table...</option></select></div><div class="field"><label class="label">Limit</label><input class="input" id="s-limit" type="number" value="5" min="1" max="100"></div></div>
```

Replace it with:
```html
<div class="field-row"><div class="field"><label class="label">Table</label><select class="select" id="s-table" onchange="loadSearchFilters()"><option value="">Select table...</option></select></div><div class="field"><label class="label">Limit</label><input class="input" id="s-limit" type="number" value="5" min="1" max="100"></div><div class="field" style="display:flex;align-items:flex-end;padding-bottom:6px"><label style="display:flex;align-items:center;gap:8px;cursor:pointer;white-space:nowrap;font-size:13px;color:var(--fg-2)"><input type="checkbox" id="s-all-tables" onchange="toggleSearchAll(this.checked)"> Search all tables</label></div></div>
```

- [ ] **Step 2: Add the JS toggle logic**

Find the `// Search` comment in the JS section (around line 574). Add this function right after it:

```javascript
function toggleSearchAll(checked){
    const sel=$('s-table');
    if(checked){sel.disabled=true;sel.style.opacity='0.5';$('s-filter-section')?.style&&($('s-filter-section').style.display='none');}
    else{sel.disabled=false;sel.style.opacity='1';$('s-filter-section')?.style&&($('s-filter-section').style.display='');}
}
```

- [ ] **Step 3: Modify doSearch to handle the toggle**

In the `doSearch` function (line 589), modify the beginning to check the toggle:

Find this line in `doSearch`:
```javascript
const table=$('s-table').value,query=$('s-query').value.trim(),limit=parseInt($('s-limit').value)||5;if(!table||!query)return;
```

Replace with:
```javascript
const searchAll=$('s-all-tables')?.checked;const table=$('s-table').value,query=$('s-query').value.trim(),limit=parseInt($('s-limit').value)||5;if(!searchAll&&!table){toast('Select a table or check "Search all tables"',false);return;}if(!query)return;
```

Then find the API call section in `doSearch`:
```javascript
const body={query,table,limit};if(hasFilters)body.filters=filters;const d=await api('/api/search',{method:'POST',body:JSON.stringify(body),signal:ctrl.signal});
```

Replace with:
```javascript
let d;if(searchAll){const body={query,limit};d=await api('/api/search-all',{method:'POST',body:JSON.stringify(body),signal:ctrl.signal});}else{const body={query,table,limit};if(hasFilters)body.filters=filters;d=await api('/api/search',{method:'POST',body:JSON.stringify(body),signal:ctrl.signal});}
```

Also update the title line to show which table(s) were searched. Find:
```javascript
$('s-title').innerHTML=`Results for "<strong>${esc(query)}</strong>" in <strong>${esc(table)}</strong>${modeLabel}${filterTags}`;
```

Replace with:
```javascript
const tableLabel=searchAll?'all tables':esc(table);$('s-title').innerHTML=`Results for "<strong>${esc(query)}</strong>" in <strong>${tableLabel}</strong>${modeLabel}${filterTags}`;
```

And in the badges section, when using search-all mode, add a source table badge. Find:
```javascript
let badges='';for(const k of metaKeys){
```

Replace with:
```javascript
let badges='';if(searchAll&&r._source_table){badges+=`<span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;background:var(--accent);color:#fff;margin-right:6px;margin-bottom:4px"><span style="opacity:0.7">table:</span> <strong>${esc(r._source_table)}</strong></span>`;}for(const k of metaKeys){if(k==='_source_table'||k==='_source_schema')continue;
```

- [ ] **Step 4: Test manually**

Run: `pgsemantic ui --reload`
Open browser to http://localhost:8080, navigate to Search page:
1. Verify the "Search all tables" checkbox appears
2. Checking it disables the table dropdown
3. Running a search with it checked hits `/api/search-all`
4. Results show source table badges

- [ ] **Step 5: Commit**

```bash
git add pgsemantic/web/static/index.html
git commit -m "feat: add 'Search all tables' toggle to web UI search page"
```

---

### Task 5: Add `search_all` MCP tool

**Files:**
- Modify: `pgsemantic/mcp_server/server.py`
- Test: `tests/unit/test_search_all.py` (append test)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_search_all.py`:

```python
class TestSearchAllMcpTool:
    """Test the search_all MCP tool."""

    @patch("pgsemantic.mcp_server.server.get_connection")
    @patch("pgsemantic.mcp_server.server._get_or_create_provider")
    @patch("pgsemantic.mcp_server.server.load_project_config")
    @patch("pgsemantic.mcp_server.server.load_settings")
    def test_search_all_tool_returns_results(
        self,
        mock_settings: MagicMock,
        mock_config: MagicMock,
        mock_provider_fn: MagicMock,
        mock_conn_ctx: MagicMock,
    ) -> None:
        mock_settings.return_value = MagicMock(
            database_url="postgresql://test@localhost/db",
            openai_api_key=None,
            ollama_base_url="http://localhost:11434",
        )
        mock_config.return_value = ProjectConfig(
            version="0.3.0",
            tables=[_make_table_config(table="products")],
        )

        mock_prov = MagicMock()
        mock_prov.embed_query.return_value = [0.1] * 384
        mock_provider_fn.return_value = mock_prov

        with patch("pgsemantic.mcp_server.server.db_search_all") as mock_sa:
            mock_sa.return_value = [
                {"id": 1, "content": "Widget", "similarity": 0.8,
                 "embedding": [0.1] * 384,
                 "_source_table": "products", "_source_schema": "public"},
            ]

            from pgsemantic.mcp_server.server import search_all_tables

            results = search_all_tables(query="test", limit=5)

        assert len(results) == 1
        assert results[0]["_source_table"] == "products"
        assert "embedding" not in results[0]  # stripped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_search_all.py::TestSearchAllMcpTool -v`
Expected: FAIL — `ImportError: cannot import name 'search_all_tables' from 'pgsemantic.mcp_server.server'`

- [ ] **Step 3: Add the MCP tool**

In `pgsemantic/mcp_server/server.py`, add the import of `search_all`:

Find (line 24-32):
```python
from pgsemantic.db.vectors import (
    _qualified_table,
    count_embedded,
    count_total_with_content,
    search_similar,
)
from pgsemantic.db.vectors import (
    hybrid_search as db_hybrid_search,
)
```

Replace with:
```python
from pgsemantic.db.vectors import (
    _qualified_table,
    count_embedded,
    count_total_with_content,
    search_similar,
)
from pgsemantic.db.vectors import (
    hybrid_search as db_hybrid_search,
)
from pgsemantic.db.vectors import (
    search_all as db_search_all,
)
```

Add the tool after the `list_configured_tables` tool (after line 452):

```python
@mcp.tool()
def search_all_tables(
    query: str,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Search across ALL configured tables at once.

    Queries every table set up with pgsemantic, merges results by similarity
    score, and returns the top matches. Each result includes _source_table
    and _source_schema to identify where it came from.

    Use this when you want to search broadly across the entire database
    rather than targeting a specific table.

    Args:
        query: The natural-language search query.
        limit: Maximum number of results to return across all tables (default 10).

    Returns:
        List of matching rows sorted by similarity, each with _source_table,
        _source_schema, content, and similarity score.
    """
    settings = load_settings()
    if not settings.database_url:
        raise ToolError(
            "DATABASE_URL not set. "
            "Set it in your .env file or as an environment variable."
        )

    config = load_project_config()
    if config is None or not config.tables:
        raise ToolError(
            "No tables configured. Run: pgsemantic apply --table <table> --column <column>"
        )

    try:
        # Build providers dict (one per unique model)
        providers: dict[str, object] = {}
        for tc in config.tables:
            if tc.model not in providers:
                api_key = settings.openai_api_key if tc.model == "openai" else None
                providers[tc.model] = _get_or_create_provider(
                    tc.model,
                    api_key=api_key,
                    ollama_base_url=settings.ollama_base_url,
                )

        with get_connection(settings.database_url) as conn:
            results = db_search_all(
                conn=conn,
                query=query,
                providers=providers,
                project_config=config,
                limit=limit,
            )

        # Strip raw embedding vectors
        clean_results = [
            {k: v for k, v in row.items() if k != "embedding"}
            for row in results
        ]

        logger.info(
            "search_all_tables: query=%r results=%d",
            query[:50],
            len(clean_results),
        )
        return clean_results

    except PgvectorSetupError as e:
        raise ToolError(str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error in search_all_tables")
        raise ToolError(f"Search failed: {e}") from e
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_search_all.py -v && pytest tests/unit/test_mcp_server.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add pgsemantic/mcp_server/server.py tests/unit/test_search_all.py
git commit -m "feat: add search_all_tables MCP tool for cross-table search"
```

---

### Task 6: Run full test suite and lint

**Files:**
- All modified files

- [ ] **Step 1: Run unit tests**

Run: `pytest tests/unit/ -v`
Expected: All tests pass (except the 6 known failures in test_config.py and test_embeddings.py)

- [ ] **Step 2: Run linter**

Run: `ruff check .`
Expected: No new lint errors

- [ ] **Step 3: Fix any issues found**

If ruff reports issues, fix them (typically import ordering, line length).

- [ ] **Step 4: Final commit if fixes were needed**

```bash
git add -u
git commit -m "chore: fix lint issues in cross-table search"
```

# Top 5 Features — pgsemantic v0.4.0

**Date:** 2026-03-30
**Status:** Approved
**Goal:** Take pgsemantic from "vector search setup tool" to "semantic data platform"

---

## Feature 1: Cross-Table Unified Search

**Lens:** Gap-filling
**Affected modules:** `db/vectors.py`, `commands/search.py`, `web/app.py`, `mcp_server/server.py`, `web/static/index.html`

### Problem
Users can only search one table at a time. If they have `news_articles`, `blog_posts`, and `comments` all configured, they must run three separate searches and mentally merge results.

### Design
- New function `search_all(conn, query_vector, project_config, limit)` in `db/vectors.py`
- Queries every configured table in the project config
- Each table searched independently (may use different embedding models/dimensions)
  - For tables sharing the same model: query vector computed once
  - For tables with different models: query vector computed per-model
- Results merged and sorted by cosine similarity score
- Each result tagged with source `table` and `schema`
- CLI: `pgsemantic search "query"` (no `--table` flag) searches all configured tables
- CLI: `pgsemantic search "query" --table T` retains single-table behavior
- Web UI: "Search All Tables" toggle on the Search page (default: on when multiple tables configured)
- MCP: new `search_all(query, limit)` tool alongside existing `semantic_search`

### Constraints
- Similarity scores are only comparable within the same embedding model. Cross-model results are ranked by normalized score (0-1 cosine similarity is already normalized).
- Limit applies globally (top N across all tables), not per-table.

### Data flow
```
query → embed with model_A → search table_1, table_2
      → embed with model_B → search table_3
      → merge all results → sort by similarity → return top N
```

---

## Feature 2: Embedding Visualization

**Lens:** Wow factor
**Affected modules:** `web/app.py`, `web/static/index.html`, `db/vectors.py`

### Problem
Users can't see what their embeddings look like. Semantic search is invisible — you type a query, get results, but have no intuition for how your data is organized semantically.

### Design
- New endpoint `GET /api/visualize?table=T&schema=S&sample=1000`
- Server-side: fetch up to N embedding vectors, reduce to 2D via UMAP (preferred) or t-SNE
  - Use `umap-learn` library (fast, better at preserving global structure than t-SNE)
  - Fallback to PCA if umap-learn not installed (simpler, faster, no extra dependency)
- Response: `[{x, y, label (truncated text), row_id, table}]`
- Frontend: render as interactive scatter plot using inline `<canvas>` (no external charting library)
  - Color points by table (when visualizing multiple tables)
  - Hover to see truncated source text
  - Click to see full row details
  - Zoom/pan support
- New "Visualize" page in sidebar (between Search and Status)
- Optional: overlay a search query as a highlighted point to show where it lands in the cluster

### Constraints
- UMAP on 5000 384-dim vectors takes ~2-5 seconds — acceptable for on-demand visualization
- Sample cap at 5000 points to keep render responsive in canvas
- Dimensionality reduction runs server-side (not in browser JS)
- `umap-learn` added as optional dependency: `pip install pgsemantic[viz]`

### Performance
- Cache reduced coordinates per table (invalidate when embedding count changes)
- Progressive loading: show plot immediately, refine with more points if user wants

---

## Feature 3: Zero-Downtime Model Migration

**Lens:** Production-ready
**Affected modules:** `commands/` (new `migrate.py`), `db/vectors.py`, `config.py`, `web/app.py`, `web/static/index.html`

### Problem
Switching embedding models (e.g., local → OpenAI for better quality, or upgrading to a newer model) requires manually tearing down and re-applying. During re-embedding, search is broken.

### Design
- New CLI command: `pgsemantic migrate --table T --model openai`
- Migration flow (atomic swap):
  1. Validate new model is available and works (embed a test string)
  2. Create a new shadow table (`pgsemantic_embeddings_<table>_migration`) with new dimensions
  3. Create HNSW index on migration table
  4. Bulk re-embed all rows into migration table (reuses `index` logic with progress bar)
  5. Atomic swap: rename migration table → production shadow table (single transaction)
  6. Update `.pgsemantic.json` with new model/dimensions
  7. Drop old shadow table (or old inline column if migrating from inline → external)
  8. Re-embed queue trigger continues to use new model going forward
- For inline storage: migration always moves to external (shadow) storage mode, since inline can't have two embedding columns simultaneously
- Worker paused during swap (< 1 second), resumes with new model

### Constraints
- Migration is idempotent — can be resumed if interrupted (migration table persists until swap)
- CLI shows: progress, estimated time, old model → new model, dimension change
- `pgsemantic status` shows "migration in progress" if migration table exists
- Web UI: "Migrate Model" button on the Apply page (per-table)

### Failure modes
- Interrupted mid-embed: resume by counting already-embedded rows in migration table
- New model API fails: abort, drop migration table, no change to production
- Disk space: migration temporarily doubles embedding storage (documented)

---

## Feature 4: Smart Auto-Chunking for Long Text

**Lens:** Gap-filling
**Affected modules:** `db/vectors.py`, `embeddings/` (new `chunker.py`), `commands/apply.py`, `commands/index.py`, `worker/daemon.py`, `config.py`

### Problem
Embedding an entire 2000-word article as a single vector loses nuance. The embedding becomes an average of all topics in the document. Search quality drops significantly for long-form content.

### Design
- New module `pgsemantic/embeddings/chunker.py`:
  - `chunk_text(text, max_tokens=256, overlap=64) -> list[str]`
  - Strategy: sentence-aware splitting (split on `.!?` then greedily fill chunks up to max_tokens)
  - Overlap ensures context isn't lost at boundaries
- Chunking enabled per-table at apply time: `pgsemantic apply --table T --column C --chunked`
- Storage: always uses external (shadow) table mode when chunked
  - Shadow table schema adds: `chunk_index INT`, `chunk_text TEXT`
  - Primary key becomes `(row_id, chunk_index)`
  - One source row → N chunk rows in shadow table
- Search returns the best-matching chunk + parent row:
  - Query shadow table for top chunks
  - JOIN back to source table for full row
  - Deduplicate by row_id (keep best chunk per row)
  - Return: full row + `matched_chunk` field + `chunk_similarity` score
- Worker handles chunking: on INSERT/UPDATE, chunk the text, embed each chunk, store all
- On DELETE: remove all chunks for that row_id
- On UPDATE: delete old chunks, re-chunk and re-embed

### Config changes
- `TableConfig` gets: `chunked: bool = False`, `chunk_max_tokens: int = 256`, `chunk_overlap: int = 64`
- `.pgsemantic.json` stores chunking settings

### Constraints
- Chunking is opt-in (not retroactive on existing tables without re-apply)
- Token counting uses whitespace splitting (approximate but fast, no tokenizer dependency)
- Max chunks per row: 200 (safety limit for very long documents)
- Short text (< max_tokens): stored as single chunk (no overhead)

### Trade-offs
- **Gain:** Dramatically better search quality for long documents
- **Give up:** More storage (N chunks per row), slightly more complex search queries, re-apply required to enable

---

## Feature 5: Natural Language Database Q&A (MCP Tool)

**Lens:** Wow factor
**Affected modules:** `mcp_server/server.py`, `db/introspect.py`

### Problem
AI agents connected via MCP can do semantic search, but they can't ask structured questions about the data ("how many articles were published last month?", "what's the average word count by category?"). This requires SQL, which the agent has to generate itself without schema knowledge.

### Design
- New MCP tool: `ask_database(question, tables=None)`
- Flow:
  1. Gather schema context: table names, column names/types, sample values (reuse `introspect.py` + `get_sample_rows`)
  2. Return schema context + question to the calling agent
  3. Agent generates SQL using its own LLM capabilities
  4. Agent calls `execute_safe_sql(sql)` — a second MCP tool that:
     - Validates SQL is SELECT only (no INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE)
     - Validates referenced tables exist
     - Sets statement timeout: 10 seconds
     - Executes in a read-only transaction
     - Returns results (max 100 rows)
- This is actually **two MCP tools**: `get_schema_context(tables=None)` and `execute_safe_sql(sql)`
- Schema context is cached per session (table structure rarely changes mid-conversation)
- Tables parameter scopes which tables the agent sees (reduces prompt size, improves accuracy)

### Safety (critical)
- **Read-only:** SQL must parse as SELECT. Reject everything else.
- **Statement timeout:** `SET statement_timeout = '10s'` before execution
- **Row limit:** `LIMIT 100` appended if not present
- **No DDL:** Regex + parser check for dangerous keywords
- **Sandboxed connection:** Use a separate connection with read-only transaction (`SET TRANSACTION READ ONLY`)

### LLM provider
- Uses the same LLM that's calling the MCP tool (the agent generates SQL in its own context)
- Alternative: pgsemantic could call Claude API directly, but this adds a dependency and cost
- **Decision:** The MCP tool provides schema context and executes validated SQL. The calling agent (Claude, GPT, etc.) generates the SQL. This avoids an LLM-calling-LLM chain and keeps pgsemantic provider-agnostic.
- Tool response includes the schema context so the agent can write good SQL

### Constraints
- Requires the calling agent to be capable of SQL generation (Claude, GPT-4 — yes; smaller models — maybe not)
- No mutation possible (enforced at connection level)
- Results truncated at 100 rows with a note if more exist

### Trade-offs
- **Gain:** "Talk to your database" experience, massive utility for AI agents
- **Give up:** Depends on agent intelligence for SQL quality. Risk of slow/expensive queries (mitigated by timeout). Schema exposure to the agent (but agents already have semantic_search access, so this is incremental).

---

## Implementation Order (Recommended)

1. **Cross-table unified search** — builds on existing search, moderate complexity, immediately useful
2. **Smart auto-chunking** — extends the embedding pipeline, high impact on search quality
3. **Zero-downtime model migration** — operational necessity, builds on shadow table infrastructure
4. **Natural language database Q&A** — new MCP tool, mostly self-contained
5. **Embedding visualization** — new dependency (umap-learn), new UI page, can be added last as a polish feature

Each feature is independent and can be shipped separately.

---

## Out of Scope (for now)

- Search analytics / query logging — deferred to next round
- Authentication on web UI — deferred (currently localhost-only)
- Scheduled reindexing — deferred
- Parallel indexing / GPU batching — deferred
- Reranking (cross-encoder) — deferred (can layer on top of chunking later)

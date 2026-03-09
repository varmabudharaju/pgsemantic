# pgsemantic

**Semantic search for your existing PostgreSQL database — set up in under a minute.**

[![PyPI](https://img.shields.io/pypi/v/pgsemantic)](https://pypi.org/project/pgsemantic/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

pgsemantic adds production-quality semantic (vector) search to any PostgreSQL database you already have. Point it at your database, pick a text column, and get natural-language search working in under a minute — no data migration, no new infrastructure, no pgvector expertise required.

It works on your existing tables, keeps embeddings in sync automatically as rows change, and exposes your search as MCP tools for AI agents like Claude.

---

## See it in action

We loaded 5,980 real clinical trials from ClinicalTrials.gov and searched them in plain English. Here's what the full flow looks like.

**Browse your existing tables — no setup required**

<img src="docs/screenshots/01-tables.png" width="100%" alt="Table Browser showing clinical_trials and news_articles tables with row counts"/>

*pgsemantic connects to your existing database and shows every table. Nothing is modified until you say so.*

---

**Inspect scores your text columns for semantic search suitability**

<img src="docs/screenshots/02-inspect.png" width="100%" alt="Inspect page scoring clinical_trials columns — eligibility and brief_summary rated 3 stars"/>

*Columns are ranked by average text length and content quality. `brief_summary` and `eligibility` score highest — these are the best candidates for embedding.*

---

**Apply sets up everything with one click**

<img src="docs/screenshots/03-apply.png" width="100%" alt="Apply page with column picker checkboxes and embedding model selection"/>

*Pick your columns, choose an embedding model (local runs free on your machine, no API key), click Apply Setup. pgsemantic adds a vector column, HNSW index, and a change-tracking trigger.*

---

**Dashboard shows your embedding coverage**

<img src="docs/screenshots/04-dashboard.png" width="100%" alt="Dashboard showing 5,980 embeddings, 100% coverage, 0 pending"/>

*5,980 rows. 100% coverage. 0 pending. Every clinical trial is now searchable by meaning.*

---

**Search in plain English — results that share zero words with your query**

<img src="docs/screenshots/05-search.png" width="100%" alt="Search results for 'kidney complications in diabetics' returning trials about diabetic nephropathy and DKD"/>

*Query: `kidney complications in diabetics`*

*Top results: trials about **diabetic nephropathy**, **DKD**, **renal insufficiency** — none of which contain the words "kidney complications in diabetics". That's semantic search.*

Results export to CSV or JSON directly from the UI, or from the CLI:

```bash
pgsemantic search "cardiovascular risk factors" --table clinical_trials --format csv > results.csv
```

---

## What is semantic search?

Traditional SQL search finds rows where a column contains the exact words you typed. Semantic search understands *meaning* — it finds results that are conceptually similar even when they share no words with your query.

| Query | Keyword search finds | Semantic search also finds |
|---|---|---|
| "comfortable headphones for travel" | rows containing "comfortable", "headphones", "travel" | noise-canceling earbuds, in-flight audio, ANC headphones |
| "high blood pressure treatment" | rows containing those exact words | hypertension therapy, antihypertensive medication, cardiac care |
| "budget accommodation Paris" | rows with all three words | cheap hotels, affordable hostels, low-cost stays in France |

---

## Install

```bash
pip install pgsemantic
```

Requirements: Python 3.10+, PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension enabled.

---

## Two ways to use it

### Option 1 — Web UI (recommended for first-time setup)

Launch a local dashboard and walk through setup with a visual interface:

```bash
pgsemantic ui
```

Open [http://localhost:8080](http://localhost:8080) in your browser.

The UI guides you through every step:

1. **Connection** — paste your PostgreSQL URL and test it
2. **Tables** — browse every table in your database, see columns and row counts
3. **Inspect** — pgsemantic scores your text columns for semantic search suitability
4. **Apply** — pick a table and column, choose an embedding model, click Apply
5. **Index** — embed all existing rows (runs in the browser, shows progress)
6. **Search** — test natural-language queries immediately
7. **Status** — monitor embedding coverage across all configured tables
8. **MCP** — copy the config snippet to connect Claude or other AI agents

### Option 2 — Terminal (CLI)

For scripting, CI, or if you prefer the command line:

```bash
# 1. Scan your database and find the best columns for semantic search
pgsemantic inspect postgresql://user:pass@host/db

# 2. Set up semantic search on a table
pgsemantic apply postgresql://user:pass@host/db --table products --column description

# 3. Embed all existing rows
pgsemantic index postgresql://user:pass@host/db --table products

# 4. Search
pgsemantic search "lightweight running shoes" --table products

# 5. Check coverage across all tables
pgsemantic status
```

---

## Web UI walkthrough

### Starting the UI

```bash
# Default — binds to localhost:8080
pgsemantic ui

# Custom port
pgsemantic ui --port 3000

# Allow access from other machines on your network
pgsemantic ui --host 0.0.0.0 --port 8080

# Development mode with auto-reload
pgsemantic ui --reload
```

Open [http://localhost:8080](http://localhost:8080) in your browser.

### Step 1: Connection

Paste your PostgreSQL connection string and click **Test Connection**. pgsemantic checks that pgvector is installed and shows you the version.

```
postgresql://user:password@host:5432/database
```

Click **Save & Connect** to store it in a local `.env` file. Your database URL never leaves your machine — it stays server-side only.

**Supabase users:** Enable pgvector first via Dashboard → Database → Extensions → search "vector" → toggle ON. Use the connection pooler URL (port 6543).

### Step 2: Browse Tables

The Tables page shows every user table in your database with column types and row counts. Click any table to inspect it, or click **Apply** to jump straight to setup.

System tables (Supabase auth/storage/realtime, pgsemantic internals) are filtered out automatically.

### Step 3: Inspect

pgsemantic scans your text columns and scores each one for semantic search suitability based on average text length and column name patterns. Higher star ratings = better candidates.

| Stars | Meaning |
|-------|---------|
| ★★★ | Excellent — long descriptive text, ideal for semantic search |
| ★★☆ | Good — medium-length text, will work well |
| ★☆☆ | Poor — short values, codes, or IDs — search will be less useful |

### Step 4: Apply

Select a table and column (or multiple columns to embed their combined text), choose an embedding model, and click **Apply**. pgsemantic will:

- Add a `vector` column to your table (or create a separate shadow table)
- Install an HNSW index for fast similarity search
- Install a database trigger to keep embeddings in sync when rows change

**Embedding models:**

| Model | Dimensions | Speed | Cost | Best for |
|-------|-----------|-------|------|---------|
| Local (MiniLM) | 384 | Fast | Free | Getting started, most use cases |
| Local (MPNet) | 768 | Medium | Free | Higher accuracy, larger datasets |
| OpenAI Small | 1536 | Fast | ~$0.02/1M tokens | Production, multilingual |
| OpenAI Large | 3072 | Medium | ~$0.13/1M tokens | Maximum accuracy |
| Ollama | Custom | Local | Free | Privacy-sensitive data |

For OpenAI models, add your API key in the **OpenAI API Key** card on the Apply page before clicking Apply.

### Step 5: Index

Click **Index Table** to embed all existing rows. Progress is shown live. For large tables:

| Rows | MiniLM (local) | OpenAI Small |
|------|---------------|--------------|
| 1,000 | ~30 seconds | ~10 seconds |
| 10,000 | ~5 minutes | ~1 minute |
| 100,000 | ~45 minutes | ~10 minutes |

Indexing is resumable — if it stops, run it again and it picks up where it left off.

### Step 6: Search

Type any natural-language query and see the most semantically similar rows ranked by similarity score (0–1, higher = more similar).

### Step 7: Status

Shows embedding coverage across all configured tables: how many rows are embedded, how many are pending, and whether the background worker is running.

### Step 8: MCP (AI Agent Integration)

Copy the config snippet and paste it into your Claude Desktop `claude_desktop_config.json` to give Claude access to your database via natural-language queries.

---

## CLI reference

### `pgsemantic ui` — launch the web dashboard

```
pgsemantic ui [--host HOST] [--port PORT] [--reload]

Options:
  --host    Host to bind to (default: 127.0.0.1)
  --port    Port to bind to (default: 8080)
  --reload  Enable auto-reload for development
```

### `pgsemantic inspect` — scan and score columns

```
pgsemantic inspect DATABASE_URL [--json]
```

Scans every text column in your database and outputs a ranked list of candidates with star ratings. Pass `--json` for machine-readable output.

### `pgsemantic apply` — set up semantic search

```
pgsemantic apply DATABASE_URL --table TABLE --column COLUMN [options]

Options:
  --table     Table name (required)
  --column    Column to embed (required)
  --model     Embedding model: local (default), local-mpnet, openai, openai-large, ollama
  --external  Store embeddings in a separate shadow table instead of adding a column
  --schema    Schema name (default: public)
```

Multiple columns (combined text):

```bash
pgsemantic apply DATABASE_URL --table products \
  --column "name,description,category" --model local
```

### `pgsemantic index` — embed existing rows

```
pgsemantic index DATABASE_URL --table TABLE [--batch-size N]
```

Embeds all rows where `embedding IS NULL`. Safe to run multiple times — skips already-embedded rows. Default batch size: 100.

### `pgsemantic search` — natural-language search

```
pgsemantic search QUERY --table TABLE [--limit N]
```

Returns the top N rows ranked by semantic similarity to the query.

### `pgsemantic status` — embedding health dashboard

```
pgsemantic status [DATABASE_URL]
```

Shows embedding coverage, pending queue depth, and failed jobs for every configured table.

### `pgsemantic worker` — background sync daemon

```
pgsemantic worker [DATABASE_URL]
```

Runs a background process that watches the job queue and keeps embeddings in sync as rows are inserted, updated, or deleted. The web UI can start/stop this for you.

### `pgsemantic serve` — MCP server

```
pgsemantic serve [DATABASE_URL]
```

Starts the MCP server over stdio for use with Claude Desktop and other MCP-compatible AI agents.

---

## MCP tools for AI agents

Once configured, AI agents get seven tools:

| Tool | Description |
|------|-------------|
| `semantic_search` | Find rows similar to a natural-language query |
| `hybrid_search` | Semantic search with SQL WHERE filters |
| `get_embedding_status` | Report embedding coverage for a table |
| `list_tables` | List all tables with columns and row counts |
| `get_sample_rows` | Sample rows to understand a table's data |
| `inspect_columns` | Score text columns for semantic search suitability |
| `list_configured_tables` | List tables with semantic search already configured |

### Claude Desktop setup

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "pgsemantic": {
      "command": "pgsemantic",
      "args": ["serve"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@host/db"
      }
    }
  }
}
```

Or use the SSE endpoint if the web UI is running:

```json
{
  "mcpServers": {
    "pgsemantic": {
      "url": "http://localhost:8080/mcp/sse"
    }
  }
}
```

---

## Features in depth

### Multi-column embedding

Embed the combined text of multiple columns so search works across all of them at once:

```bash
# CLI
pgsemantic apply $DB --table products --column "name,description,tags"

# Web UI — check multiple columns in the Apply page column picker
```

### External (shadow) storage

For tables where you can't alter the schema, store embeddings in a separate table:

```bash
pgsemantic apply $DB --table products --column description --external
```

pgsemantic creates `pgsemantic_embeddings_products` and joins it transparently at search time.

### Automatic sync

A database trigger fires on every INSERT, UPDATE, and DELETE and writes to pgsemantic's job queue. The background worker picks up the queue and keeps embeddings fresh. Zero application code changes needed.

### Hybrid search

Combine semantic similarity with exact SQL filters via the MCP `hybrid_search` tool or by calling the search function directly in your application.

---

## Supported databases

Any PostgreSQL instance with pgvector installed:

- Local PostgreSQL + pgvector
- Supabase (enable pgvector in Dashboard → Database → Extensions)
- Neon
- Amazon RDS / Aurora PostgreSQL (pgvector available as an extension)
- Google Cloud SQL for PostgreSQL
- Self-hosted

---

## Configuration files

pgsemantic stores two local files (both gitignored by default):

| File | Purpose |
|------|---------|
| `.env` | `DATABASE_URL` and optional API keys (`OPENAI_API_KEY`, `OLLAMA_BASE_URL`) |
| `.pgsemantic.json` | Auto-generated by `apply` — tracks which tables are configured, which model, dimensions, etc. |

Both files are created in the directory where you run `pgsemantic`. The `.env` is written with `chmod 600` (owner-read-only).

---

## Security

- **Database URL is never sent to the browser** — it stays server-side, masked in the UI
- **CSRF protection** on all POST endpoints
- **SQL injection prevention** via Pydantic validation + regex on all identifiers
- **Rate limiting** — 120 requests/minute per IP
- **Content Security Policy**, `X-Frame-Options: DENY`, and other security headers on all responses
- **`.env` written with chmod 600** — no world-readable credentials

---

## FAQ

**Does it work with tables that already have data?**
Yes. Run `pgsemantic index` (or click Index in the UI) to embed existing rows. New and updated rows are handled automatically by the trigger.

**Which embedding model should I pick?**
Start with Local (MiniLM) — it's free, runs offline, and works well for most use cases. Switch to OpenAI if you need multilingual support or higher accuracy.

**Can I search across multiple tables?**
Yes. Run `apply` on each table, then use `pgsemantic search --table TABLE` per table, or use the MCP tools which let an AI agent query across all configured tables.

**Can I change the model after indexing?**
Yes. Re-run `apply` with the new model, then `index` again. This re-embeds all rows with the new model.

**What happens if indexing fails halfway through?**
pgsemantic tracks which rows are embedded. Re-run `index` and it resumes from where it left off — already-embedded rows are skipped.

**Does it slow down writes to my table?**
The trigger adds minimal overhead — it writes a small job record to a queue table asynchronously. The actual embedding happens in the background worker, not in your transaction.

**Can I use it with a managed database (Supabase, Neon)?**
Yes. You need pgvector enabled and the ability to create tables and triggers. Most managed providers support this. For Supabase, enable pgvector in Dashboard → Database → Extensions.

**How do I uninstall semantic search from a table?**
Use the Teardown button in the web UI (Status page → table menu), or it will be supported via CLI in a future release. This drops the embedding column, trigger, index, and config entry.

# pgsemantic

**Semantic search for your existing Postgres database — set up in under a minute.**

[![PyPI](https://img.shields.io/pypi/v/pgsemantic)](https://pypi.org/project/pgsemantic/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

pgsemantic bootstraps production-quality semantic (vector) search on any PostgreSQL database you already have — no data migration, no new infrastructure, no pgvector expertise required. It works on your existing tables, automatically keeps embeddings in sync as rows change, and connects to AI agents out of the box.

---

## What is semantic search?

Traditional search finds rows where a column contains the exact word you typed. Semantic search understands *meaning* — it finds results that are conceptually similar even when they don't share a single word with your query.

| Query | Keyword search finds | Semantic search also finds |
|---|---|---|
| "comfortable headphones for travel" | rows containing "comfortable", "headphones", "travel" | noise-canceling earbuds, in-flight audio, ANC headphones |
| "high blood pressure treatment" | rows containing those exact words | hypertension therapy, antihypertensive medication, cardiac care |
| "quick recipe for dinner tonight" | rows with "quick", "recipe", "dinner" | easy weeknight meals, 30-minute meals, fast cooking ideas |

This is powered by vector embeddings — every piece of text gets converted into a point in high-dimensional space. Similar meaning = nearby points. pgsemantic handles all of this automatically using pgvector inside your existing Postgres database.

---

## The problem

Most teams already have their data in Postgres. Adding semantic search means choosing between three bad options:

**Option A — Migrate to a vector database (Pinecone, Weaviate, Qdrant)**
You copy your data into a new system, build a sync pipeline to keep it current, pay for another service, and now you have two sources of truth. When Postgres has new rows, your vector DB might be stale. When you get search results back, you still have to JOIN against Postgres to get full row data.

**Option B — Set up pgvector yourself**
Read through the pgvector docs, write `CREATE EXTENSION`, choose between HNSW and IVFFlat and understand the tradeoffs, write a trigger to queue new rows, build a worker to consume the queue, handle retries on failures, write the cosine distance search query, wire up an embedding model, manage model downloads and lifecycle, repeat for every table. It's a full day of work, and that's before you've built anything for your users.

**Option C — pgsemantic**
Your data stays in Postgres. One command sets up everything. New rows are embedded automatically. Search works immediately.

---

## What you get

### A complete semantic search stack, not just the index

pgsemantic handles the entire lifecycle that most tools leave to you:

- **Schema detection** — scans your tables and scores which columns are worth embedding, so you don't have to guess
- **One-command setup** — creates the pgvector extension (if missing), adds an embedding column, creates an HNSW index, and installs a trigger — all behind a single `apply` command
- **Bulk indexing** — embeds all your existing rows in batches, with a live progress bar and safe resume if interrupted
- **Automatic sync** — a background worker watches the trigger queue and embeds new and updated rows as they arrive, so search is never stale
- **Natural language search** — query your database in plain English from the terminal or the web UI
- **Hybrid search** — combine semantic similarity with SQL filters (e.g. find products semantically similar to "portable speaker" where `category = 'electronics'` and `price < 200`)
- **Health monitoring** — per-table dashboard showing embedding coverage, pending rows, and failed rows
- **AI agent integration** — ships with an MCP server that lets Claude Desktop, Cursor, and other AI tools search your database directly

### Two interfaces for every skill level

**Web UI** — A full visual dashboard for people who prefer not to work in a terminal. Connect your database, browse tables, inspect columns, pick a model, and start searching — entirely from the browser. No config files to edit, no commands to memorize beyond the one that starts it.

**Terminal (CLI)** — A complete command-line interface for automation, scripting, and CI/CD. Every operation is a single command. Output is formatted for both humans and pipes. Works headless on servers.

Both interfaces do the same thing — pick the one that fits your workflow.

### No new infrastructure

pgsemantic is a Python package. It talks to your existing Postgres. It runs on your machine. The web UI is a built-in FastAPI server, not a hosted service. The worker is a single background process. Nothing requires Docker Compose additions, Kubernetes, or a cloud account.

### Free by default

The default embedding model (all-MiniLM-L6-v2 from Hugging Face) runs entirely on your machine — no API key, no internet after the first download, no cost per query. It's fast enough for most use cases. When you need better quality, switch to OpenAI or Ollama with a single flag — no re-architecture required.

### Your data never leaves Postgres

No migration. No sync pipeline. Embeddings live right next to your rows in the same table. They're in your backup, under your access controls, visible to your existing queries. If you drop the extension later, you drop a column — that's it.

### Production-grade internals

- **HNSW index** (not IVFFlat) — no training step, handles inserts without a full rebuild, >95% recall
- **Persistent queue** (not LISTEN/NOTIFY) — LISTEN/NOTIFY silently drops events under load. pgsemantic uses a queue table with `SELECT FOR UPDATE SKIP LOCKED`, the same pattern used by Sidekiq, Oban, and every serious job processor. Events survive process crashes and restarts.
- **Pre-filtered hybrid search** — uses `hnsw.iterative_scan = relaxed_order` (pgvector ≥ 0.8.0) to combine vector similarity with SQL WHERE filters in a single query, without post-filtering accuracy loss.

---

## vs. the alternatives

| | pgsemantic | Pinecone / Weaviate / Qdrant | Manual pgvector |
|---|---|---|---|
| Data location | Stays in your Postgres | Migrated to vendor system | Stays in Postgres |
| Setup time | < 1 minute | Hours (migration + sync pipeline) | Hours (docs + plumbing) |
| Works on existing tables | Yes | No — re-ingest required | Partially |
| Sync with Postgres | Built-in trigger + worker | You build it | You build it |
| Cost | Free (local model) | $70–$700+/month | Free, but costs your time |
| Vendor lock-in | None | High | None |
| Crash-safe queue | Yes | Managed for you | You build it |
| Web UI | Built-in | Vendor dashboard | None |
| AI agent integration (MCP) | Built-in | Via their API | You build it |
| Hybrid search | Built-in | Paid feature / manual | You build it |
| Requires pgvector expertise | No | No | Yes |

---

## Install

```bash
pip install pgsemantic
```

Requires Python 3.10+ and a Postgres database with pgvector. If pgvector isn't installed, pgsemantic tries to install it automatically when you run `apply`.

---

## Two ways to use pgsemantic

| | Web UI | Terminal (CLI) |
|---|---|---|
| Best for | First-time setup, exploring data, visual feedback | Automation, scripting, CI/CD, headless servers |
| How to start | `pgsemantic ui` then open your browser | `pgsemantic inspect`, `apply`, `index`, ... |
| Config | Enter DB URL in the browser, click Save | Write a `.env` file or pass `--db` |

- [Option 1: Web UI](#option-1-web-ui) — point-and-click, no commands to memorize
- [Option 2: Terminal (CLI)](#option-2-terminal-cli) — full command-line walkthrough

---

## Option 1: Web UI

Launch with one command. Everything else happens in the browser.

```bash
pgsemantic ui
# Opens at http://localhost:8080
```

The sidebar walks you through the full flow left to right. Press **Cmd+K** (or **Ctrl+K**) to open a command palette for quick navigation between pages.

---

### Step 1 — Connect to your database

The app opens on the **Connection** page.

1. Paste your database URL into the **Database URL** field
2. Click **Test Connection** — this checks connectivity without saving anything
3. Click **Save to .env** — saves permanently so you never have to enter it again

**Finding your connection string:**

| Provider | Where to find it |
|----------|-----------------|
| Local Docker | From your `docker-compose.yml`. If using ours: `postgresql://postgres:password@localhost:5432/pgvector_dev` |
| Supabase | Dashboard → Settings → Database → Connection string → URI |
| Neon | Dashboard → your project → Connection Details → Connection string |
| Railway | Dashboard → your Postgres service → Variables → `DATABASE_URL` |
| AWS RDS | `postgresql://USER:PASS@your-instance.region.rds.amazonaws.com:5432/dbname` |

> **Supabase:** pgvector must be enabled manually first. Go to Dashboard → Database → Extensions → find "vector" and toggle it ON.

---

### Step 2 — Browse your tables

Click **Tables** in the sidebar. You'll see every table in your database with row counts, column counts, and schema. Click any table row to open a side drawer with column types and a preview of sample data.

System and internal tables (Postgres internals, Supabase auth/storage/realtime) are automatically filtered out so you only see your own tables.

---

### Step 3 — Find the right column to search

Click **Inspect** in the sidebar. pgsemantic scans every text column across all your tables and scores them by search suitability — based on average text length, column name patterns, and data type.

You'll see a ranked list like:

```
Table            Column         Score   Avg Length
────────────     ────────────   ─────   ──────────
news_articles    body           ★★★     1,200 chars
news_articles    headline       ★★★        85 chars
products         description    ★★★       108 chars
products         name           ★☆☆        20 chars
```

Three stars means it's a strong candidate — good text length, meaningful content. One star means the column is likely too short or categorical to benefit from semantic search (e.g. a `status` or `name` field).

Click **Quick Apply** on any row to jump straight to setup with that table and column pre-filled.

---

### Step 4 — Set up semantic search

On the **Apply** page:

**1. Select a table** from the dropdown.

**2. Choose your columns.** Checkboxes appear automatically showing all text columns in the selected table. Check one for single-column search, or check multiple to combine them — useful when you want search to consider both a `headline` and a `body` together.

**3. Pick an embedding model:**

| Model | Quality | Cost | What you need |
|-------|---------|------|---------------|
| Local Fast | Good | Free | Nothing — runs on your machine |
| Local Better | Better | Free | Nothing — still on your machine, higher quality |
| OpenAI Small | Great | $0.02 / 1M tokens | OpenAI API key |
| OpenAI Large | Best | $0.13 / 1M tokens | OpenAI API key |
| Ollama | Good | Free | Ollama installed and running locally |

Start with **Local Fast** if you're unsure. You can always switch later by re-running Apply.

**4. If you chose OpenAI:** An **OpenAI API Key** section appears below the model cards. Paste your key and click **Save API Key**. It's stored in your local `.env` file — it never touches the browser after you submit it.

**5. Click Apply Setup.**

pgsemantic shows you the exact SQL it's about to run and asks for confirmation. It makes only two changes to your database: one embedding column and one trigger. Nothing else is touched. If you'd rather not modify your source table at all, check **Use external shadow table** — embeddings go into a separate table and your original stays untouched.

---

### Step 5 — Embed your existing rows

Click **Index** in the sidebar.

1. Select your table
2. Click **Start Indexing**

A live progress bar shows rows being embedded in real time, with rows/second and an estimated completion time. You can safely stop and resume — it only processes rows that haven't been embedded yet.

Approximate timing as a reference:

| Dataset size | Local model | OpenAI |
|-------------|------------|--------|
| 1,000 rows | ~30 seconds | ~5 seconds |
| 5,000 rows | ~3–5 minutes | ~20 seconds |
| 50,000 rows | ~30–45 minutes | ~3 minutes |

When indexing completes, scroll down to **Worker — Auto-embed New Rows** and click **Start Worker**. The worker runs in the background and automatically embeds any rows you insert or update from this point forward. You don't need to re-index manually as your data grows.

---

### Step 6 — Search your data

Click **Search** in the sidebar.

1. Select your table
2. Type any natural language query

Results appear as you type — ranked by semantic similarity, with a visual score bar on each result. Click any result card to open a full-row detail view in a side drawer.

Example queries and what they find — even without exact word matches:

| Query | What it finds |
|---|---|
| `"comfortable headphones for long flights"` | Noise-canceling headphones, travel audio gear |
| `"articles about government AI policy"` | News about AI regulation, tech legislation |
| `"quick vegetarian dinner"` | Easy plant-based recipes, meatless weeknight meals |
| `"treatment for high blood pressure"` | Hypertension medication, cardiac care articles |

---

### Step 7 — Monitor embedding health

Click **Status** in the sidebar for a per-table health dashboard:

- **Coverage** — what percentage of rows have embeddings (you want this at 100%)
- **Pending** — rows in the queue waiting to be embedded
- **Failed** — rows that couldn't be embedded (usually empty content or encoding issues)

Enable **Auto-refresh** to watch coverage climb live while indexing runs.

---

### Step 8 — Connect AI agents (optional)

Click **MCP** in the sidebar. This is where you connect Claude Desktop, Cursor, or any other AI tool that supports the Model Context Protocol.

1. Select **stdio** (recommended for Claude Desktop / Cursor)
2. Click **Configure Claude Desktop** — this auto-writes the config file
3. Restart Claude Desktop

Now you can ask Claude things like:
- *"Search the products table for portable speakers under $150"*
- *"Find news articles related to climate change from the last year"*
- *"What are the most relevant clinical trials for diabetes treatment?"*

Claude calls pgsemantic's search tools directly and returns ranked results from your actual database.

---

## Option 2: Terminal (CLI)

The full flow from zero to semantic search using only the command line. Every output shown below is real.

### Step 1: Set up your connection

```bash
# Pass the URL directly (good for a quick try)
pgsemantic inspect postgresql://postgres:password@localhost:5432/mydb

# Or save it in a .env file (recommended for regular use)
echo 'DATABASE_URL=postgresql://postgres:password@localhost:5432/mydb' > .env
pgsemantic inspect   # reads DATABASE_URL automatically
```

> Add `.env` to your `.gitignore` — it contains your database password.

### Step 2: Find the right column

```
$ pgsemantic inspect

╭──────────────────────────────────────────────────────────────────────────────╮
│ Semantic Search Candidates — localhost:5432/pgvector_dev                      │
│ pgvector 0.8.2                                                               │
╰──────────────────────────────────────────────────────────────────────────────╯
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━┓
┃ Table           ┃ Column        ┃ Score ┃  Avg Length ┃ Type ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━┩
│ clinical_trials │ eligibility   │ ★★★   │ 1,382 chars │ text │
│ clinical_trials │ brief_summary │ ★★★   │   711 chars │ text │
│ products        │ description   │ ★★★   │   108 chars │ text │
│ clinical_trials │ title         │ ★★★   │    85 chars │ text │
│ products        │ name          │ ★☆☆   │    20 chars │ text │
└─────────────────┴───────────────┴───────┴─────────────┴──────┘

Next step:
  pgsemantic apply --table products --column description
```

### Step 3: Set up semantic search

```
$ pgsemantic apply --table products --column description

╭────────────────────────────── pgsemantic apply ──────────────────────────────╮
│ Setting up semantic search on public.products                                │
│ Column(s): description                                                       │
│ Model: all-MiniLM-L6-v2 (384 dimensions)                                     │
│ Storage: inline                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

  [1/9] Checking pgvector extension... v0.8.2
  [2/9] Verifying source column(s)... found (PK: id)
  [3/9] Checking embedding column... needs creation

╭───────────────────────── SQL Preview — ALTER TABLE ──────────────────────────╮
│ ALTER TABLE "public"."products" ADD COLUMN "embedding" vector(384);          │
╰──────────────────────────────────────────────────────────────────────────────╯
This will add an embedding column to your table. Continue? [y/N]: y

  [4/9] Adding embedding column... done
  [5/9] Creating HNSW index (CONCURRENTLY)... done
  [6/9] Creating queue table... done
  [7/9] Installing trigger function... done
  [8/9] Installing trigger on table... done
  [9/9] Saving config... done

╭─────────────────────────────── Setup Complete ───────────────────────────────╮
│ Semantic search is ready on public.products (description)                    │
│                                                                              │
│ Next steps:                                                                  │
│   1. Embed existing rows:  pgsemantic index --table products                 │
│   2. Start live sync:      pgsemantic worker                                 │
│   3. Start MCP server:     pgsemantic serve                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

`apply` always shows you the exact SQL before running it and requires confirmation. It makes two changes and nothing else: one `embedding` column, one trigger.

### Step 4: Embed existing rows

```
$ pgsemantic index --table products

Indexing public.products (description) with all-MiniLM-L6-v2
(batch size: 32, storage: inline)

  Total rows with content: 20
  Already embedded:        0
  Remaining:               20

  Embedding rows... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 20/20 0:00:00 0:00:00

Indexed 20 rows in 0.3s (4,631 rows/min)
  Coverage: 20/20 (100.0%)
```

Safe to stop and re-run — only un-embedded rows are processed.

### Step 5: Search

```
$ pgsemantic search "wireless headphones with good battery" --table products

Results for: "wireless headphones with good battery" in products.description

  1. (score: 0.698)  Bose QuietComfort 45
  Comfortable over-ear wireless headphones with world-class noise cancellation
  and 24-hour battery life

  2. (score: 0.673)  Sony WH-1000XM5
  Premium wireless noise-canceling headphones with 30-hour battery life and
  exceptional sound quality

  3. (score: 0.604)  Apple AirPods Max
  High-fidelity wireless over-ear headphones with active noise cancellation
  and spatial audio
```

### Step 6: Keep embeddings in sync

```bash
$ pgsemantic worker

INFO  Worker started. Polling every 500ms, batch size 10.
INFO  Claimed 3 jobs from queue.
INFO  Embedded products#21 (description, 384d)
INFO  Embedded products#22 (description, 384d)
INFO  Worker alive, queue empty.
```

The worker runs in the background. Every INSERT or UPDATE fires your trigger, the job lands in the queue, and the worker embeds it within 500ms. Stop with `Ctrl+C` — no events are lost, they'll be picked up on the next run.

### Step 7: Check health

```
$ pgsemantic status

Embedding Status
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Table      ┃ Column      ┃ Model            ┃ Storage  ┃ Coverage   ┃ Pending ┃ Failed ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ products   │ description │ all-MiniLM-L6-v2 │ inline   │   20/20    │       0 │      0 │
│            │             │                  │          │ (100.0%)   │         │        │
└────────────┴─────────────┴──────────────────┴──────────┴────────────┴─────────┴────────┘
```

### Step 8: Connect Claude Desktop

```bash
pgsemantic integrate claude
# Restart Claude Desktop
# Now ask: "Find products similar to a portable bluetooth speaker under $100"
```

---

## Features in depth

### Multi-column search

Embed multiple columns together so a single search considers all of them. Useful when meaning is split across fields — for example, searching products by both name and description, or articles by both headline and body.

```bash
pgsemantic apply --table products --columns title,description
```

Columns are joined with labeled separators before embedding:

```
title: Sony WH-1000XM5
description: Wireless noise-canceling headphones with 30-hour battery life...
```

Search works identically after that. In the Web UI, check multiple column checkboxes on the Apply page.

---

### External storage (shadow table)

By default, pgsemantic adds an `embedding` column to your source table. If you can't or won't ALTER your table — strict schema policies, no permissions, compliance requirements — use external storage:

```bash
pgsemantic apply --table products --column description --external
```

pgsemantic creates a separate shadow table (`pgsemantic_embeddings_products`) and stores all embeddings there. Your source table is completely untouched. Search JOINs against the shadow table transparently — results look identical.

In the Web UI: check **Use external shadow table** on the Apply page.

You can combine external storage with multi-column search:

```bash
pgsemantic apply --table products --columns title,description --external
```

---

### Hybrid search (semantic + SQL filters)

Use the MCP tools or query directly to combine vector similarity with WHERE conditions:

```python
# Find products semantically similar to "portable speaker"
# but only where category = 'electronics' and price < 200
hybrid_search(
    table="products",
    query="portable speaker",
    filters={"category": "electronics", "price_max": 200}
)
```

This uses `hnsw.iterative_scan = relaxed_order` under the hood, which means the vector index is used alongside the SQL filter — no accuracy loss from post-filtering.

---

### Embedding models

| Model | CLI key | Dimensions | Cost | Notes |
|-------|---------|-----------|------|-------|
| all-MiniLM-L6-v2 | `local` *(default)* | 384 | Free | Fast, on-device, no API key |
| all-mpnet-base-v2 | `local-mpnet` | 768 | Free | Higher quality, still free and on-device |
| text-embedding-3-small | `openai` | 1536 | $0.02/1M tokens | Best balance of quality and speed |
| text-embedding-3-large | `openai-large` | 3072 | $0.13/1M tokens | Highest quality available |
| nomic-embed-text (Ollama) | `ollama` | 768 | Free | Self-hosted, requires Ollama running |

**CLI:**
```bash
pgsemantic apply --table products --column description --model openai
```

**Web UI:** Pick from the visual model cards on the Apply page. OpenAI models show an API key field below the cards.

The model is set per-table at apply time. To switch, run `apply` again with the new model and then re-index.

---

### MCP server tools

The MCP server exposes your database to AI agents that support the Model Context Protocol (Claude Desktop, Cursor, and others).

| Tool | What it does |
|------|-------------|
| `semantic_search` | Natural-language similarity search across a configured table |
| `hybrid_search` | Semantic search combined with SQL WHERE filters |
| `get_embedding_status` | Coverage %, queue depth, model info for a table |
| `list_tables` | List all database tables with column types and row counts |
| `get_sample_rows` | Preview rows from any table |
| `inspect_columns` | Score text columns for semantic search suitability |
| `list_configured_tables` | List tables that already have semantic search set up |

Start the MCP server:
```bash
pgsemantic serve              # stdio (for Claude Desktop / Cursor)
pgsemantic ui                 # also runs MCP over SSE at /mcp/sse
pgsemantic integrate claude   # auto-writes Claude Desktop config
```

---

## Supported Postgres hosts

Works with any Postgres that has the pgvector extension installed.

| Host | How to enable pgvector |
|------|----------------------|
| Docker (local) | Use `ankane/pgvector:pg16` image — pgvector is pre-installed |
| Supabase | Dashboard → Database → Extensions → "vector" toggle ON |
| Neon | `CREATE EXTENSION vector;` as project owner |
| Railway | `CREATE EXTENSION vector;` as postgres user |
| AWS RDS | `CREATE EXTENSION vector;` with `rds_superuser` role |
| Google Cloud SQL | `CREATE EXTENSION vector;` as `cloudsqlsuperuser` |
| Azure Database | `CREATE EXTENSION vector;` (Flexible Server, PG 15+) |
| Bare metal | `apt install postgresql-16-pgvector` then `CREATE EXTENSION vector;` |

If pgvector isn't installed when you run `apply`, pgsemantic tries to install it automatically. If that fails due to permissions, it detects your host and prints the exact steps to enable it, then exits cleanly.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Your Application                      │
│                                                           │
│   INSERT INTO products (name, description)                │
│   VALUES ('Widget', 'A great widget for...');             │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                    PostgreSQL + pgvector                   │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  products table                                      │  │
│  │  id | name | description | embedding vector(384)     │  │
│  └───────────────────────────┬─────────────────────────┘  │
│                              │                             │
│  Trigger fires on INSERT/UPDATE ──► pgvector_setup_queue  │
│                                     (table, row_id, op)   │
└──────────────────────────────────────┬────────────────────┘
                                       │
              ┌────────────────────────┘
              ▼
┌─────────────────────────────┐     ┌───────────────────────┐
│   pgsemantic worker         │     │  pgsemantic serve     │
│                             │     │  (MCP Server)         │
│  Polls queue every 500ms    │     │                       │
│  Fetches source text        │     │  semantic_search()    │
│  Generates embedding        │     │  hybrid_search()      │
│  Writes back to table       │     │  list_tables()        │
│  Deletes completed job      │     │  get_sample_rows()    │
└─────────────────────────────┘     │  inspect_columns()    │
                                    │  + 2 more tools       │
┌─────────────────────────────┐     └───────┼───────────────┘
│   pgsemantic ui             │             │
│   (Web Dashboard)           │             ▼
│                             │     ┌───────────────┐
│  Search, Inspect, Browse    │     │ Claude Desktop │
│  Apply, Index, Status       │     │ Cursor / AI    │
│  Embedded MCP SSE server    │     └───────────────┘
│  Worker control             │
└─────────────────────────────┘
```

**Key design decisions:**

- **HNSW over IVFFlat** — HNSW requires no training phase, handles incremental inserts without index rebuilds, and delivers >95% recall. IVFFlat needs a training step and degrades as data distribution shifts.
- **Persistent queue over LISTEN/NOTIFY** — LISTEN/NOTIFY drops events under high load with no indication that events were lost. A queue table with `SELECT FOR UPDATE SKIP LOCKED` guarantees every event is processed exactly once, survives worker crashes, and scales to concurrent workers.
- **Embeddings on source table by default** — keeps your data model simple. One table, one backup, no JOIN overhead at query time. `--external` (shadow table) is available for schemas you can't modify.
- **Pre-filtered hybrid search** — combines the vector index with SQL WHERE in a single query using `hnsw.iterative_scan = relaxed_order`. No post-filtering means full HNSW recall even with narrow filters.

---

## Commands reference

| Command | Description |
|---------|-------------|
| `pgsemantic inspect [DB_URL]` | Scan and score columns for semantic search suitability |
| `pgsemantic apply --table T --column C` | Set up semantic search: extension, column, HNSW index, trigger |
| `pgsemantic index --table T` | Bulk embed all existing rows |
| `pgsemantic search "query" --table T` | Natural language search from the terminal |
| `pgsemantic worker` | Background daemon that keeps embeddings in sync |
| `pgsemantic serve` | Start MCP server (stdio) for Claude Desktop / Cursor |
| `pgsemantic status` | Show per-table embedding health dashboard |
| `pgsemantic integrate claude` | Auto-configure Claude Desktop |
| `pgsemantic ui` | Launch web dashboard at http://localhost:8080 |

### `apply` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--table`, `-t` | required | Table to set up |
| `--column`, `-c` | required* | Single text column to embed |
| `--columns` | — | Comma-separated columns to embed together (e.g. `title,description`) |
| `--model`, `-m` | `local` | `local`, `local-mpnet`, `openai`, `openai-large`, or `ollama` |
| `--external` | off | Store embeddings in a shadow table instead of modifying source table |
| `--db`, `-d` | `DATABASE_URL` env | Postgres connection string |
| `--schema`, `-s` | `public` | Postgres schema name |

*Use `--column` or `--columns`, not both.

### `ui` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--host`, `-h` | `127.0.0.1` | Bind address. Use `0.0.0.0` for network access |
| `--port`, `-p` | `8080` | Port to listen on |
| `--reload` | off | Auto-reload on file changes (for development) |

---

## Configuration

### `.env` file

Create this in the directory where you run pgsemantic commands:

```env
# Required
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# For OpenAI embedding models
OPENAI_API_KEY=sk-...

# For Ollama (if not running on default port)
OLLAMA_BASE_URL=http://localhost:11434
```

Most users only need `DATABASE_URL`. When using the Web UI, you can set all of these from the browser without touching files.

### `.pgsemantic.json`

Auto-generated by `pgsemantic apply`. Stores your table configurations (which columns, which model, which storage mode). The `index`, `worker`, `search`, and `serve` commands read this file. Don't delete it. Safe to commit to git — it contains no credentials.

---

## FAQ

**Does this modify my production tables?**
By default, it adds one `embedding` column and one trigger. That's it — no other columns, no schema changes, no data touched. Use `--external` if you can't or won't ALTER the table at all.

**What if pgvector isn't installed on my database?**
`apply` tries to install it automatically (`CREATE EXTENSION IF NOT EXISTS vector`). If that fails due to insufficient permissions, it detects your host provider and prints the exact steps to enable it, then exits without making any changes.

**Do I need to move my data anywhere?**
No. pgsemantic connects to your database directly. Embeddings are generated on your local machine and written back. Your data never leaves your database.

**Can I search across multiple columns at once?**
Yes — `--columns title,description` on the CLI, or check multiple checkboxes on the Apply page in the UI. The columns are concatenated and embedded as a single unit.

**What happens to very long text?**
Local models (MiniLM, mpnet) have a 256-token limit — longer text is truncated at the embedding stage. For long documents (articles, legal text, clinical notes), use `openai` or `openai-large`, which support up to 8,191 tokens.

**Can I switch embedding models later?**
Yes. Re-run `apply` with `--model <new-model>`, then re-run `index`. The old embeddings are replaced. Note that you can't mix models within a table — all rows use the same model.

**How does the worker handle failures?**
If embedding a row fails (e.g., OpenAI API error, network issue), the job stays in the queue and is retried up to 3 times (configurable with `WORKER_MAX_RETRIES`). Failed rows are marked and visible in `pgsemantic status`.

**Is the Web UI secure?**
Yes. The database URL is stored server-side only — never sent to the browser. All write endpoints require a CSRF token validated per-request. Table and column names are validated against strict regex before any SQL to prevent injection. Responses include `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, and Content Security Policy headers. Rate limited at 120 requests/minute per IP.

**Can I run multiple tables at once?**
Yes. Run `apply` separately for each table, then run one `pgsemantic worker` — it processes all configured tables from the shared queue.

---

## Development

```bash
git clone https://github.com/varmabudharaju/pgsemantic.git
cd pgsemantic
pip install -e ".[dev]"

docker-compose up -d                           # local Postgres with pgvector
pytest tests/unit/ -v                          # unit tests (no DB needed)
pytest tests/integration/ -m integration -v    # integration tests (needs DB)
ruff check .                                   # lint
```

---

## License

MIT

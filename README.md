# pgsemantic

**Point it at your existing Postgres database. Get semantic search in 60 seconds.**

[![PyPI](https://img.shields.io/pypi/v/pgsemantic)](https://pypi.org/project/pgsemantic/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Zero-config CLI that bootstraps production-quality semantic search on any existing PostgreSQL database — local or remote (Supabase, Neon, RDS, Railway). No pgvector expertise required.

**You don't download your data.** You don't move anything. You give pgsemantic your Postgres connection string — whether it's localhost or a cloud database on the other side of the world — and it sets up semantic search in place.

## What it does

- Scans your database and scores columns for semantic search suitability
- Adds a vector embedding column + HNSW index to your existing table (nothing else changes)
- Embeds all your existing rows using local models (no API key needed) or OpenAI
- Installs a trigger so new/updated rows are automatically queued for embedding
- Provides an MCP server so Claude Desktop and Cursor can search your database by meaning
- Works with **any** Postgres host — local Docker, Supabase, Neon, RDS, Railway, bare metal

## Installation

```bash
pip install pgsemantic
```

## Quickstart (Local Database)

```bash
# 1. Start a local Postgres with pgvector
docker-compose up -d

# 2. Scan your database — see which columns are good candidates
pgsemantic inspect postgresql://postgres:password@localhost:5432/mydb

# 3. Set up semantic search on a table
pgsemantic apply --table products --column description

# 4. Embed all existing rows
pgsemantic index --table products

# 5. Search!
pgsemantic search "wireless noise-canceling headphones" --table products
```

## Quickstart (Remote Database)

**Works with any Postgres that has pgvector enabled.** Just pass your connection string:

```bash
# Supabase
pgsemantic inspect postgresql://postgres.xxxx:password@aws-0-us-east-1.pooler.supabase.com:6543/postgres
pgsemantic apply --table products --column description --db postgresql://postgres.xxxx:password@aws-0-us-east-1.pooler.supabase.com:6543/postgres

# Neon
pgsemantic apply --table articles --column content --db postgresql://user:pass@ep-cool-rain-123456.us-east-2.aws.neon.tech/mydb

# Amazon RDS
pgsemantic apply --table tickets --column body --db postgresql://admin:pass@mydb.abc123.us-east-1.rds.amazonaws.com:5432/mydb

# Railway
pgsemantic apply --table docs --column content --db postgresql://postgres:pass@roundhouse.proxy.rlwy.net:12345/railway
```

All commands accept `--db` to override the `DATABASE_URL` environment variable. Set it once in `.env` and you won't need `--db` at all:

```bash
# .env
DATABASE_URL=postgresql://user:pass@your-cloud-host:5432/yourdb
```

Then every command just works:

```bash
pgsemantic inspect
pgsemantic apply --table products --column description
pgsemantic index --table products
pgsemantic search "noise canceling" --table products
```

---

## Commands Reference

### `pgsemantic inspect`

Scan a database and score every text column for semantic search suitability.

```bash
pgsemantic inspect <DATABASE_URL>
pgsemantic inspect --json    # machine-readable output
```

**Example output:**

```
╭────────────────────────────────────────────────────────────────────╮
│     Semantic Search Candidates — postgresql://localhost/mydb        │
╰────────────────────────────────────────────────────────────────────╯

 Table               Column           Score   Avg Length   Type
 ──────────────────────────────────────────────────────────────────
 products            description      ★★★    847 chars    text
 support_tickets     body             ★★★    1,203 chars  text
 articles            content          ★★★    4,821 chars  text
 users               bio              ★★☆    143 chars    text
 products            name             ★☆☆    34 chars     varchar

ℹ  Scoring is heuristic (text length + column name patterns).
   Always verify recommendations make sense for your use case.

Next step:
  pgsemantic apply --table products --column description
```

Scoring is based on average text length and column name patterns — longer text in columns named `description`, `body`, `content`, etc. score higher. It's a heuristic to help you decide, not a guarantee.

---

### `pgsemantic apply`

Set up semantic search on a table. This is the main setup command.

```bash
pgsemantic apply --table <TABLE> --column <COLUMN> [--model local|openai] [--db <URL>]
```

**What it does (9 steps):**

1. Checks/installs pgvector extension
2. Detects the table's primary key
3. Shows you the exact SQL it will run and asks for confirmation
4. Adds an `embedding vector(384)` column to your table
5. Creates an HNSW index (CONCURRENTLY — no downtime)
6. Creates the job queue table (`pgvector_setup_queue`)
7. Installs the trigger function
8. Installs a trigger on your table for INSERT/UPDATE/DELETE
9. Saves config to `.pgsemantic.json`

**Example output:**

```
✓ pgvector 0.8.0 detected
✓ Primary key detected: id (integer)

About to run:

  ALTER TABLE products ADD COLUMN embedding vector(384);
  CREATE INDEX CONCURRENTLY ON products USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

Proceed? [y/N]: y

✓ Column 'embedding vector(384)' added to products
✓ HNSW index created (m=16, ef_construction=64)
✓ Queue table pgvector_setup_queue created
✓ Trigger function pgvector_setup_notify_fn installed
✓ Trigger pgvector_setup_products_trigger installed
✓ Config saved to .pgsemantic.json

Next steps:
  pgsemantic index --table products    # embed existing rows
  pgsemantic worker                    # keep new rows embedded automatically
  pgsemantic serve                     # start MCP server for Claude/Cursor
```

**Important:** pgsemantic only adds ONE column (`embedding`) and ONE trigger to your table. It never modifies, renames, or drops any existing columns, constraints, or indexes.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--table` | required | Table to set up |
| `--column` | required | Text column to embed |
| `--model` | `local` | Embedding model: `local` (all-MiniLM-L6-v2, 384d), `openai` (text-embedding-3-small, 1536d) |
| `--db` | `DATABASE_URL` | Database connection string |
| `--schema` | `public` | Postgres schema |

---

### `pgsemantic index`

Bulk embed all existing rows that don't have embeddings yet.

```bash
pgsemantic index --table <TABLE> [--batch-size 32] [--db <URL>]
```

**Example output:**

```
Indexing products.description ━━━━━━━━━━━━━━━━━━━━ 12,000/12,000  100%  542 rows/min
✓ Indexed 12,000 rows in products.description
```

- **Idempotent** — re-running only processes rows where `embedding IS NULL`
- **Resumable** — if interrupted, just re-run and it picks up where it left off
- Uses keyset pagination for consistent performance on large tables

---

### `pgsemantic search`

Search your database using natural language, right from the terminal.

```bash
pgsemantic search "<QUERY>" --table <TABLE> [--limit 5] [--db <URL>]
```

**Example:**

```bash
pgsemantic search "wireless headphones with good battery" --table products
```

```
Results for: "wireless headphones with good battery" in products.description

  1. (score: 0.847)  id: 142  name: Sony WH-1000XM5
  Wireless noise-canceling headphones with 30-hour battery life...

  2. (score: 0.812)  id: 89  name: Bose QuietComfort 45
  Premium wireless headphones featuring 24-hour battery...

  3. (score: 0.798)  id: 234  name: Apple AirPods Max
  Over-ear wireless headphones with 20-hour battery...
```

---

### `pgsemantic worker`

Start the background worker that keeps embeddings in sync as your data changes.

```bash
pgsemantic worker [--db <URL>]
```

**How it works:**

1. Your app inserts/updates/deletes rows normally
2. The trigger (installed by `apply`) writes a job to `pgvector_setup_queue`
3. The worker polls the queue, generates embeddings, and writes them back
4. Uses `SELECT FOR UPDATE SKIP LOCKED` — safe for multiple concurrent workers

```
INFO  Worker started. Polling pgvector_setup_queue every 500ms.
INFO  Claimed 3 jobs from queue.
INFO  Embedded products#142 (description, 384d) in 38ms
INFO  Embedded products#143 (description, 384d) in 41ms
INFO  Embedded products#144 (description, 384d) in 39ms
INFO  Queue empty. Sleeping 500ms.
```

The worker handles connection drops with exponential backoff (1s → 2s → 4s → ... → 30s max) and never crashes permanently.

---

### `pgsemantic status`

Show embedding health for all watched tables.

```bash
pgsemantic status [--db <URL>]
```

**Example output:**

```
╭────────────────────────────────────────────────────────────────────────────╮
│                      Embedding Health Dashboard                            │
╰────────────────────────────────────────────────────────────────────────────╯

 Table               Column       Total   Embedded  Coverage  Queue     Model
 ───────────────────────────────────────────────────────────────────────────
 products            description  12,847  12,843    99.97%    0 pending  all-MiniLM-L6-v2
 support_tickets     body          3,201   3,201    100%      1 pending  all-MiniLM-L6-v2
 articles            content         847     847    100%      0 pending  all-MiniLM-L6-v2

⚠  4 rows in 'products' are missing embeddings.
   Run: pgsemantic index --table products

✓  Worker is keeping embeddings in sync.
```

---

### `pgsemantic serve`

Start the MCP server so AI agents (Claude Desktop, Cursor) can search your database.

```bash
pgsemantic serve                              # stdio (for Claude Desktop / Cursor)
pgsemantic serve --transport http --port 3000  # HTTP/SSE (for remote agents)
```

The MCP server exposes three tools:

| Tool | Description |
|------|-------------|
| `semantic_search` | Natural-language similarity search |
| `hybrid_search` | Semantic search + SQL WHERE filters (e.g. `{"category": "laptop", "price_max": 1500}`) |
| `get_embedding_status` | Coverage %, queue depth, model info |

---

### `pgsemantic integrate`

Auto-configure Claude Desktop to connect to pgsemantic.

```bash
pgsemantic integrate claude
```

This writes the MCP server configuration directly to your Claude Desktop config file:

```
✓ Claude Desktop config updated at:
  ~/Library/Application Support/Claude/claude_desktop_config.json

  Added MCP server "pgsemantic" with:
    command: pgsemantic serve
    DATABASE_URL: postgresql://...

Restart Claude Desktop to activate.

Claude will now have access to:
  • semantic_search — search any indexed table by meaning
  • hybrid_search — semantic search + SQL filters
  • get_embedding_status — check embedding health

Try asking Claude:
  "Search the products table for wireless headphones under $100"
```

---

## Claude Desktop Integration

Full setup in 4 steps:

```bash
# 1. Install pgsemantic
pip install pgsemantic

# 2. Set up your table
pgsemantic apply --table products --column description --db postgresql://...
pgsemantic index --table products --db postgresql://...

# 3. Auto-configure Claude Desktop
pgsemantic integrate claude

# 4. Restart Claude Desktop — done!
```

Now you can ask Claude things like:
- "Search the products table for wireless headphones under $100"
- "Find support tickets about login issues"
- "Show me articles related to machine learning"

Claude calls `semantic_search` or `hybrid_search` behind the scenes and gets ranked results from your actual database.

---

## Supported Postgres Hosts

pgsemantic works with any Postgres that has the pgvector extension available.

| Host | pgvector Support | How to Enable | Notes |
|------|-----------------|---------------|-------|
| **Docker (local)** | Pre-installed | Use `ankane/pgvector:pg16` image | Included in our docker-compose.yml |
| **Supabase** | Built-in | Dashboard → Database → Extensions → "vector" | Works on free tier |
| **Neon** | Built-in | `CREATE EXTENSION vector;` as project owner | Works on free tier |
| **Railway** | Postgres 15+ | `CREATE EXTENSION vector;` as postgres user | Works |
| **AWS RDS** | Postgres 15+ | `CREATE EXTENSION vector;` with `rds_superuser` role | Note: not `superuser` |
| **Google Cloud SQL** | Postgres 15+ | `CREATE EXTENSION vector;` as `cloudsqlsuperuser` | Flexible Server |
| **Azure Database** | Flexible Server, PG 15+ | `CREATE EXTENSION vector;` | Works |
| **Bare metal** | Manual install | `apt install postgresql-16-pgvector` then `CREATE EXTENSION vector;` | Full superuser access |

When pgvector isn't installed, `pgsemantic apply` tries to install it automatically. If that fails (permissions), it prints host-specific instructions and exits cleanly.

---

## Embedding Models

| Model | Provider | Dimensions | Cost | Speed | API Key Required |
|-------|----------|-----------|------|-------|-----------------|
| **all-MiniLM-L6-v2** | Local (sentence-transformers) | 384 | Free | ~500+ rows/min (CPU) | No |
| **nomic-embed-text** | Ollama (local) | 768 | Free | ~300+ rows/min (CPU) | No |
| **text-embedding-3-small** | OpenAI | 1536 | $0.02/1M tokens | ~1000+ rows/min | Yes |

**Default:** `all-MiniLM-L6-v2` — runs locally, no API key needed, 22 MB download on first use. Good for development, demos, and small-to-medium datasets.

```bash
# Use default local model (no API key needed)
pgsemantic apply --table products --column description

# Use OpenAI for higher quality embeddings
pgsemantic apply --table products --column description --model openai

# Use Ollama for better local quality (requires Ollama running)
pgsemantic apply --table products --column description --model ollama
```

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
│  Writes back to table       │     │  get_embedding_status │
│  Deletes completed job      │     │       │               │
└─────────────────────────────┘     └───────┼───────────────┘
                                            │
                                            ▼
                                    ┌───────────────┐
                                    │ Claude Desktop │
                                    │ Cursor / AI    │
                                    └───────────────┘
```

**Key design decisions:**

- **Embedding column lives on your source table** — no central embeddings table. This avoids cross-table HNSW recall degradation and keeps queries simple.
- **Persistent queue, not LISTEN/NOTIFY** — LISTEN/NOTIFY drops events silently under load. The queue table with `SELECT FOR UPDATE SKIP LOCKED` never loses events and survives process crashes.
- **HNSW index, not IVFFlat** — HNSW requires no training step, handles inserts without full rebuild, and achieves >95% recall.
- **Pre-filtered hybrid search** — Uses `hnsw.iterative_scan = relaxed_order` (pgvector ≥0.8.0) to combine vector similarity with SQL WHERE filters efficiently.

---

## FAQ

### Does this modify my existing tables?

Minimally. pgsemantic adds exactly ONE column (`embedding vector(N)`) and ONE trigger to your table. It never modifies, renames, or drops any existing columns, constraints, or indexes.

### What if pgvector is not installed?

`pgsemantic apply` tries to install it automatically (`CREATE EXTENSION IF NOT EXISTS vector`). If that fails due to permissions, it prints host-specific instructions (Supabase, Neon, RDS, etc.) and exits cleanly.

### Do I need to download my data?

No. pgsemantic connects directly to your Postgres database (local or remote) and works in place. Your data never leaves your database. Embeddings are generated on your machine and written back to the database.

### Can I use my own embedding model?

Yes. pgsemantic supports three providers out of the box:
- `local` — sentence-transformers (all-MiniLM-L6-v2, 384d, free)
- `openai` — text-embedding-3-small (1536d, requires API key)
- `ollama` — nomic-embed-text (768d, requires Ollama running locally)

### How does the worker keep embeddings in sync?

A Postgres trigger fires on every INSERT, UPDATE, or DELETE on your watched column. It writes a job to `pgvector_setup_queue`. The worker polls this queue, generates the embedding, and writes it back. Uses `SELECT FOR UPDATE SKIP LOCKED` so multiple workers can run safely in parallel.

### Can I use this with multiple tables?

Yes. Run `pgsemantic apply` for each table/column you want to search. The worker processes all watched tables from a single queue.

### What about very large text fields?

all-MiniLM-L6-v2 has a 256-token limit. Longer text is silently truncated. For long documents, consider using OpenAI's text-embedding-3-small (8191 token limit) or splitting text into chunks before embedding.

### Is there a web UI?

No. pgsemantic is a CLI tool + MCP server. The primary interface is the command line and AI agents (Claude Desktop, Cursor) via MCP.

---

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```env
# Required — your Postgres connection string
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Embedding provider: local (default), openai, or ollama
EMBEDDING_PROVIDER=local

# Required only for openai provider
OPENAI_API_KEY=sk-...

# Ollama settings (only if using ollama provider)
OLLAMA_BASE_URL=http://localhost:11434

# MCP transport: stdio (default) or http
MCP_TRANSPORT=stdio
MCP_PORT=3000

# Worker settings
WORKER_BATCH_SIZE=10
WORKER_POLL_INTERVAL_MS=500
WORKER_MAX_RETRIES=3

# Logging: debug, info, warn, error
LOG_LEVEL=info
```

### Project Config (`.pgsemantic.json`)

Created automatically by `pgsemantic apply`. Tracks which tables are set up:

```json
{
  "version": "0.1.0",
  "tables": [
    {
      "table": "products",
      "schema": "public",
      "column": "description",
      "embedding_column": "embedding",
      "model": "local",
      "model_name": "all-MiniLM-L6-v2",
      "dimensions": 384,
      "hnsw_m": 16,
      "hnsw_ef_construction": 64,
      "applied_at": "2026-03-01T10:00:00Z",
      "primary_key": ["id"]
    }
  ]
}
```

---

## Development

```bash
# Clone and install in dev mode
git clone https://github.com/yourusername/pgsemantic.git
cd pgsemantic
pip install -e ".[dev]"

# Start local Postgres with pgvector
docker-compose up -d

# Run unit tests (no database needed)
pytest tests/unit/ -v

# Run integration tests (requires docker-compose up)
pytest tests/integration/ -m integration -v

# Lint
ruff check .

# Type check
mypy pgsemantic/ --strict
```

---

## License

MIT

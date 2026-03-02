# pgsemantic

**Point it at your existing Postgres database. Get semantic search in 60 seconds.**

[![PyPI](https://img.shields.io/pypi/v/pgsemantic)](https://pypi.org/project/pgsemantic/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Zero-config CLI that bootstraps production-quality semantic search on any existing PostgreSQL database — local or remote (Supabase, Neon, RDS, Railway). No pgvector expertise required.

---

## Install

```bash
pip install pgsemantic
```

```
$ pgsemantic --help

 Usage: pgsemantic [OPTIONS] COMMAND [ARGS]...

 Zero-config semantic search bootstrap for any PostgreSQL database.

╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ inspect    Scan a database and score columns for semantic search suitability │
│ apply      Set up semantic search on a table                                │
│ index      Bulk embed existing rows where embedding IS NULL                 │
│ search     Search your database using natural language                      │
│ worker     Start the background worker daemon                               │
│ serve      Start the pgsemantic MCP server                                  │
│ status     Show embedding health dashboard for all watched tables           │
│ integrate  Set up AI agent integrations (Claude Desktop)                    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## Full Walkthrough

Here's the complete flow from zero to semantic search. Every output below is real terminal output, not mockups.

### Step 1: Connect to your database

pgsemantic needs a Postgres connection string. You can either pass it with `--db` or save it in a `.env` file:

```bash
# Option A: pass directly (good for trying things out)
pgsemantic inspect postgresql://postgres:password@localhost:5432/mydb

# Option B: save in a .env file (recommended)
echo 'DATABASE_URL=postgresql://postgres:password@localhost:5432/mydb' > .env
pgsemantic inspect   # reads from .env automatically
```

**Where to find your connection string:**

| Provider | Where to find it |
|----------|-----------------|
| **Local Docker** | The URL you set when creating the container. With our docker-compose.yml: `postgresql://postgres:password@localhost:5432/pgvector_dev` |
| **Supabase** | Dashboard → Settings → Database → Connection string → URI |
| **Neon** | Dashboard → your project → Connection Details → Connection string |
| **Railway** | Dashboard → your Postgres service → Variables → `DATABASE_URL` |
| **AWS RDS** | `postgresql://USER:PASS@your-instance.region.rds.amazonaws.com:5432/dbname` |

> **Tip:** The `.env` file contains your database password. Add `.env` to your `.gitignore` so it's never committed to git.

### Step 2: Scan your database

```
$ pgsemantic inspect postgresql://postgres:password@localhost:5432/pgvector_dev

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
│ products        │ category      │ ★☆☆   │     9 chars │ text │
└─────────────────┴───────────────┴───────┴─────────────┴──────┘

Scoring is heuristic (text length + column name patterns).
Always verify recommendations make sense for your use case.

Next step:
  pgsemantic apply --table clinical_trials --column eligibility
```

`inspect` scans every text column in your database, samples average text length, and scores them. Longer text in columns named `description`, `body`, `content`, etc. scores higher. It's a heuristic to help you decide what to embed.

### Step 3: Set up semantic search on a table

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

`apply` shows you the exact SQL it will run and asks for confirmation before touching your table. It adds ONE column (`embedding`) and ONE trigger. Nothing else changes.

### Step 4: Embed your existing rows

```
$ pgsemantic index --table products

Indexing public.products (description) with all-MiniLM-L6-v2 (batch size: 32,
storage: inline)

  Total rows with content: 20
  Already embedded:        0
  Remaining:               20

  Embedding rows... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 20/20 0:00:00 0:00:00

Indexed 20 rows in 0.3s (4,631 rows/min)
  Coverage: 20/20 (100.0%)

Next steps:
  Start live sync:   pgsemantic worker
  Start MCP server:  pgsemantic serve
```

`index` generates vector embeddings for every row. It uses the local model by default — no API key, no internet required (after first download). Re-running only processes rows that haven't been embedded yet, so it's safe to interrupt and resume.

### Step 5: Search!

```
$ pgsemantic search "wireless headphones with good battery" --table products

Results for: "wireless headphones with good battery" in products.description

  1. (score: 0.698)  id: 2  name: Bose QuietComfort 45  category: electronics  price: 329.99
  Comfortable over-ear wireless headphones with world-class noise cancellation,
  24-hour battery life, and high-fidelity audio

  2. (score: 0.673)  id: 1  name: Sony WH-1000XM5  category: electronics  price: 349.99
  Premium wireless noise-canceling headphones with 30-hour battery life,
  exceptional sound quality, and comfortable fit for all-day listening

  3. (score: 0.604)  id: 3  name: Apple AirPods Max  category: electronics  price: 549.99
  High-fidelity wireless over-ear headphones with active noise cancellation,
  spatial audio, and premium build quality

  4. (score: 0.569)  id: 4  name: Samsung Galaxy Buds Pro  category: electronics  price: 199.99
  True wireless earbuds with intelligent active noise cancellation, 360 audio,
  and water resistance for workouts

  5. (score: 0.533)  id: 17  name: Sony WF-1000XM5 Earbuds  category: electronics  price: 279.99
  Premium true wireless earbuds with industry-leading noise cancellation and
  high-resolution audio in a compact design
```

Natural language in, ranked results out. It works with any query:

```
$ pgsemantic search "espresso machine" --table products --limit 3

Results for: "espresso machine" in products.description

  1. (score: 0.738)  id: 19  name: Breville Barista Express  category: kitchen  price: 699.99
  Semi-automatic espresso machine with built-in grinder, precise temperature
  control, and micro-foam milk texturing

  2. (score: 0.303)  id: 11  name: Instant Pot Duo 7-in-1  category: kitchen  price: 89.99
  Multi-use programmable pressure cooker, slow cooker, rice cooker, steamer, and
  more in one appliance

  3. (score: 0.253)  id: 13  name: Yeti Rambler Tumbler  category: kitchen  price: 35.00
  Vacuum-insulated stainless steel tumbler that keeps drinks cold for hours,
  dishwasher safe
```

### Step 6: Keep embeddings in sync (optional)

```bash
$ pgsemantic worker

INFO  Worker started. Polling pgvector_setup_queue every 500ms, batch size 10.
INFO  Claimed 3 jobs from queue.
INFO  Embedded products#21 (description, 384d)
INFO  Embedded products#22 (description, 384d)
INFO  Embedded products#23 (description, 384d)
INFO  Worker alive, queue empty.
```

The worker runs in the background. When your app inserts/updates/deletes rows, the trigger fires, the worker picks up the job, and the embedding is updated automatically. Stop it with `Ctrl+C`.

### Step 7: Check health

```
$ pgsemantic status

Embedding Status
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Table           ┃ Column    ┃ Model            ┃ Storage  ┃ Coverage   ┃ Pending ┃ Failed ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ clinical_trials │ brief_sum │ all-MiniLM-L6-v2 │ inline   │ 3003/3003  │       0 │      0 │
│                 │           │                  │          │ (100.0%)   │         │        │
│ products        │ descript  │ all-MiniLM-L6-v2 │ inline   │   20/20    │       0 │      0 │
│                 │           │                  │          │ (100.0%)   │         │        │
│ test_products   │ title,    │ all-MiniLM-L6-v2 │ external │   10/10    │       0 │      0 │
│                 │ descript  │                  │          │ (100.0%)   │         │        │
└─────────────────┴───────────┴──────────────────┴──────────┴────────────┴─────────┴────────┘
```

### Step 8: Connect Claude Desktop (optional)

```bash
# Auto-configure Claude Desktop
pgsemantic integrate claude

# Restart Claude Desktop, then ask:
# "Search the products table for wireless headphones under $100"
# "Find articles about machine learning"
```

Claude calls `semantic_search` or `hybrid_search` behind the scenes and gets ranked results from your actual database.

---

## Multi-Column Search (`--columns`)

Search across multiple columns at once — for example, finding products by both their name and description:

```
$ pgsemantic apply --table test_products --columns title,description --external

╭────────────────────────────── pgsemantic apply ──────────────────────────────╮
│ Setting up semantic search on public.test_products                           │
│ Column(s): title, description                                                │
│ Model: all-MiniLM-L6-v2 (384 dimensions)                                     │
│ Storage: external → pgsemantic_embeddings_test_products                      │
╰──────────────────────────────────────────────────────────────────────────────╯

  [1/9] Checking pgvector extension... v0.8.2
  [2/9] Verifying source column(s)... found (PK: id)
  [3/9] Creating shadow table... done
      Your table remains unchanged. Embeddings stored in
      pgsemantic_embeddings_test_products
  [5/9] Creating HNSW index (CONCURRENTLY)... done
  [6/9] Creating queue table... done
  [7/9] Installing trigger function... done
  [8/9] Installing trigger on table... done
  [9/9] Saving config... done

╭─────────────────────────────── Setup Complete ───────────────────────────────╮
│ Semantic search is ready on public.test_products (title, description)        │
│ Embeddings stored in: pgsemantic_embeddings_test_products                    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Columns are concatenated with field labels before embedding:

```
title: Sony WH-1000XM5
description: Wireless noise-canceling headphones with 30-hour battery life...
```

Then index and search work exactly the same:

```
$ pgsemantic index --table test_products

Indexing public.test_products (title, description) with all-MiniLM-L6-v2
(batch size: 32, storage: external)

  Total rows with content: 10
  Already embedded:        0
  Remaining:               10

  Embedding rows... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10/10 0:00:00 0:00:00

Indexed 10 rows in 0.4s (1,370 rows/min)
  Coverage: 10/10 (100.0%)
```

```
$ pgsemantic search "warm jacket for hiking" --table test_products

Results for: "warm jacket for hiking" in test_products.title

  1. (score: 0.581)  id: 5  North Face Thermoball
  Warm and compressible insulated jacket with synthetic fill for wet conditions

  2. (score: 0.507)  id: 4  Patagonia Nano Puff
  Lightweight insulated jacket perfect for cold weather hiking and outdoor adventures
```

---

## External Storage Mode (`--external`)

By default, pgsemantic adds an `embedding` column directly to your source table. If you can't or don't want to alter your production table (no ALTER permissions, strict schema policies, etc.), use `--external`:

```bash
pgsemantic apply --table products --column description --external
```

This creates a separate **shadow table** to store embeddings. Your source table stays completely untouched:

```
-- Your source table — NO embedding column, no changes at all
SELECT column_name FROM information_schema.columns WHERE table_name = 'test_products';

 column_name
─────────────
 id
 title
 description
 category
 price

-- Embeddings live in a separate shadow table
SELECT row_id, source_column, model_name FROM pgsemantic_embeddings_test_products LIMIT 3;

 row_id │  source_column    │    model_name
────────┼───────────────────┼──────────────────
 1      │ title+description │ all-MiniLM-L6-v2
 2      │ title+description │ all-MiniLM-L6-v2
 3      │ title+description │ all-MiniLM-L6-v2
```

**How updates and deletes work with `--external`:**

- **INSERT/UPDATE** — the worker generates a new embedding and upserts it into the shadow table (`INSERT ... ON CONFLICT DO UPDATE`)
- **DELETE** — the worker deletes the corresponding row from the shadow table
- **Search** — pgsemantic automatically JOINs the shadow table to your source table at query time, so search results look exactly the same

The shadow table schema:

```sql
CREATE TABLE pgsemantic_embeddings_{table} (
    row_id        TEXT PRIMARY KEY,          -- stringified PK from your source table
    embedding     vector(384),               -- the embedding vector
    source_column TEXT NOT NULL,             -- which column(s) were embedded
    model_name    TEXT NOT NULL,             -- model used to generate the embedding
    updated_at    TIMESTAMPTZ DEFAULT NOW()  -- when the embedding was last updated
);
```

You can combine `--external` with `--columns`:

```bash
# Multi-column + external: search across title and description,
# without modifying the products table at all
pgsemantic apply --table products --columns title,description --external
```

---

## Embedding Models

| Model | Provider | Dimensions | Cost | API Key Required |
|-------|----------|-----------|------|-----------------|
| **all-MiniLM-L6-v2** (default) | Local (sentence-transformers) | 384 | Free | No |
| **nomic-embed-text** | Ollama (local) | 768 | Free | No |
| **text-embedding-3-small** | OpenAI | 1536 | $0.02/1M tokens | Yes |

```bash
# Use default local model (no API key needed)
pgsemantic apply --table products --column description

# Use OpenAI (set OPENAI_API_KEY in .env first)
pgsemantic apply --table products --column description --model openai

# Use Ollama (requires: ollama serve && ollama pull nomic-embed-text)
pgsemantic apply --table products --column description --model ollama
```

---

## Supported Postgres Hosts

Works with any Postgres that has the pgvector extension.

| Host | How to Enable pgvector |
|------|----------------------|
| **Docker (local)** | Use `ankane/pgvector:pg16` image (included in our docker-compose.yml) |
| **Supabase** | Dashboard → Database → Extensions → "vector" |
| **Neon** | `CREATE EXTENSION vector;` as project owner |
| **Railway** | `CREATE EXTENSION vector;` as postgres user |
| **AWS RDS** | `CREATE EXTENSION vector;` with `rds_superuser` role |
| **Google Cloud SQL** | `CREATE EXTENSION vector;` as `cloudsqlsuperuser` |
| **Azure Database** | `CREATE EXTENSION vector;` (Flexible Server, PG 15+) |
| **Bare metal** | `apt install postgresql-16-pgvector` then `CREATE EXTENSION vector;` |

When pgvector isn't installed, `pgsemantic apply` tries to install it automatically. If that fails (permissions), it prints host-specific instructions and exits cleanly.

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

- **Embedding column lives on your source table** (by default) — no central embeddings table. Use `--external` if you prefer a separate shadow table.
- **Persistent queue, not LISTEN/NOTIFY** — LISTEN/NOTIFY drops events silently under load. The queue table with `SELECT FOR UPDATE SKIP LOCKED` never loses events and survives process crashes.
- **HNSW index, not IVFFlat** — no training step, handles inserts without full rebuild, >95% recall.
- **Pre-filtered hybrid search** — Uses `hnsw.iterative_scan = relaxed_order` (pgvector >=0.8.0) to combine vector similarity with SQL WHERE filters in a single query.

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `pgsemantic inspect <DB_URL>` | Scan and score columns for semantic search suitability |
| `pgsemantic apply --table T --column C` | Set up semantic search: extension, column, index, trigger |
| `pgsemantic index --table T` | Bulk embed all existing rows |
| `pgsemantic search "query" --table T` | Natural language search from the terminal |
| `pgsemantic worker` | Background daemon that keeps embeddings in sync |
| `pgsemantic serve` | Start MCP server for Claude Desktop / Cursor |
| `pgsemantic status` | Show embedding health dashboard |
| `pgsemantic integrate claude` | Auto-configure Claude Desktop |

### `apply` options

| Flag | Default | Description |
|------|---------|-------------|
| `--table`, `-t` | required | Table to set up |
| `--column`, `-c` | required* | Single text column to embed |
| `--columns` | — | Comma-separated columns to embed together (e.g. `title,description`) |
| `--model`, `-m` | `local` | `local`, `openai`, or `ollama` |
| `--external` | off | Store embeddings in a shadow table (don't modify source table) |
| `--db`, `-d` | `DATABASE_URL` | Database connection string |
| `--schema`, `-s` | `public` | Postgres schema |

*Use `--column` or `--columns`, not both.

### MCP server tools

| Tool | Description |
|------|-------------|
| `semantic_search` | Natural-language similarity search |
| `hybrid_search` | Semantic search + SQL WHERE filters (e.g. `{"category": "laptop", "price_max": 1500}`) |
| `get_embedding_status` | Coverage %, queue depth, model info |

---

## Configuration

### `.env` file

Create in the directory where you run pgsemantic:

```env
# Required — your Postgres connection string
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Only needed for OpenAI embeddings
OPENAI_API_KEY=sk-...

# Only needed for Ollama embeddings
OLLAMA_BASE_URL=http://localhost:11434
```

Most users only need `DATABASE_URL`. Everything else has sensible defaults.

### `.pgsemantic.json`

Created automatically by `pgsemantic apply`. Tracks which tables are set up and with what settings. Don't delete it — `index`, `worker`, `search`, and `serve` read it. Safe to commit to git.

---

## FAQ

**Does this modify my existing tables?**
By default, it adds ONE column (`embedding`) and ONE trigger. Nothing else changes. Use `--external` to avoid even that — embeddings go in a separate shadow table.

**What if pgvector is not installed?**
`apply` tries to install it automatically. If that fails (permissions), it prints host-specific instructions and exits cleanly.

**Do I need to download my data?**
No. pgsemantic connects to your database (local or remote) and works in place. Embeddings are generated on your machine and written back over the connection.

**Can I search across multiple columns?**
Yes. Use `--columns title,description` to concatenate multiple columns into one embedding.

**What about very large text fields?**
all-MiniLM-L6-v2 has a 256-token limit (longer text is truncated). For long documents, use OpenAI's text-embedding-3-small (8191 token limit).

**What is `.pgsemantic.json`?**
Config file created by `apply`. Tracks your setup. Don't delete it. Safe to commit to git.

---

## Development

```bash
git clone https://github.com/varmabudharaju/pgsemantic.git
cd pgsemantic
pip install -e ".[dev]"

docker-compose up -d                           # local Postgres with pgvector
pytest tests/unit/ -v                          # unit tests (no DB needed)
pytest tests/integration/ -m integration -v    # integration tests
ruff check .                                   # lint
```

---

## License

MIT

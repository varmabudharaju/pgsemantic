# CLAUDE.md — pgsemantic

> This file is the single source of truth for every architectural decision, convention,
> command, SQL statement, and build instruction in this project. Read it fully before
> touching any code. Update it whenever a new term, command, or convention is introduced.

---

## Project Identity

| Field | Value |
|---|---|
| **Project name** | pgsemantic |
| **Type** | Open-source Python CLI tool + MCP server |
| **Language** | Python 3.10+ |
| **Status** | Active Development — MVP Phase |
| **PyPI package** | `pgsemantic` |
| **Install command** | `pip install pgsemantic` |
| **Binary produced** | `pgsemantic` (available in PATH after pip install) |
| **License** | MIT |
| **Maintainer** | Sai Ram Varma Budharaju |

**One-line description:**
Zero-config CLI that bootstraps production-quality semantic search on any existing
PostgreSQL database — no pgvector expertise required.

**The problem this solves:**
pgai Vectorizer (Timescale) — the only mature tool that automated embedding sync on
Postgres — was archived in February 2026. There is no single tool that takes an existing
Postgres database and produces working semantic search end-to-end. Every RAG tutorial says
"first, set up pgvector" and waves away 2–4 days of real engineering work involving schema
design, backfill scripting, trigger setup, index tuning, and worker infrastructure.
This project packages all of that work into one CLI tool.

**Target user:**
A Python or backend developer who has an existing PostgreSQL database (products table,
support tickets, documents, articles — any table with text columns) and wants to add
semantic search so that Claude, Cursor, or their own application can query it intelligently.
They do not want to become a pgvector expert. They want one command that works.

**Viral hook:**
> "Point it at your existing Postgres database. Get semantic search in 60 seconds."

**Why this will get GitHub stars:**
- pgai Vectorizer was just archived — its users are stranded and actively looking for alternatives
- The "set up pgvector" pain point is the most complained-about step in every RAG tutorial
- The MCP + Postgres bootstrap combination does not exist in any polished open-source tool
- The `inspect` command with its beautiful terminal output is GIF-able and shareable
- Zero cost to try: no API key required, no cloud account, works on any Postgres

---

## What This Project Is NOT

Before writing any code, understand what we explicitly do NOT do:

| Claim | Truth |
|---|---|
| "Cuts latency in half vs two-step retrieval" | FALSE — was a hallucination in a prior spec. LangChain/LlamaIndex already do single-query pgvector. We make no latency claims. |
| Uses LISTEN/NOTIFY for change detection | NO — LISTEN/NOTIFY drops events silently. We use a persistent queue table. |
| Central embeddings table | NO — We add a vector column to the source table. Central table causes cross-table post-filter recall degradation. |
| Inline ONNX inference in the event loop | NO — sentence-transformers runs synchronously; we use asyncio.to_thread() for async contexts. |
| 100 rows/minute bulk indexer | NO — that was a design flaw in a prior spec. sentence-transformers in batched mode achieves ≥500 rows/min on CPU. We benchmark and label hardware. |
| "LangChain is heavyweight" | NOT our claim — LangChain's PGVector integration is ~500 lines. We solve a different problem. |

---

## Domain Glossary

| Term | Definition |
|---|---|
| **pgvector** | PostgreSQL extension that adds a `vector(n)` column type and ANN search operators: `<=>` (cosine distance), `<->` (L2/Euclidean), `<#>` (negative inner product). |
| **HNSW** | Hierarchical Navigable Small World — the ANN index type used on vector columns. No training step required (unlike IVFFlat). Handles inserts without full rebuild. Our default. |
| **IVFFlat** | Alternative ANN index. Requires a training step (needs existing data). Better at extreme scale (100M+ vectors). We do NOT use this. |
| **ANN** | Approximate Nearest Neighbor — fast similarity search that trades perfect recall for speed. HNSW achieves >95% recall at <10ms with proper tuning. |
| **Embedding** | A float array (`vector(384)` or `vector(1536)`) that encodes the semantic meaning of a text string. Similar texts produce embeddings that are close in vector space. |
| **Cosine similarity** | The primary similarity metric. `1 - (embedding <=> query_vector)`. Ranges from 0 (unrelated) to 1 (identical meaning). |
| **sentence-transformers** | Python library that runs embedding models in-process via ONNX/PyTorch. Our default local provider. No API key, no separate server, works offline on CPU. |
| **all-MiniLM-L6-v2** | Default embedding model. 384 dimensions, 22 MB download, zero cost. Good for development and demos. Downloaded automatically to `~/.cache/huggingface/` on first use. |
| **nomic-embed-text** | Better local model available via Ollama. 1024 dimensions, 274 MB. Outperforms OpenAI text-embedding-ada-002 on MTEB benchmarks (62.39 score vs ~61). |
| **text-embedding-3-small** | OpenAI's production embedding model. 1536 dimensions. Requires `OPENAI_API_KEY`. Used with `--model openai`. |
| **MTEB** | Massive Text Embedding Benchmark. The standard benchmark for comparing embedding models. We cite MTEB scores where we reference model quality. |
| **Dimensions** | The length of the embedding vector. Must be fixed at table creation time. `vector(384)` for all-MiniLM, `vector(1024)` for nomic-embed-text, `vector(1536)` for OpenAI. Changing dimensions requires re-embedding all rows. |
| **Persistent Queue** | Our CDC (change data capture) mechanism. A plain Postgres table (`pgvector_setup_queue`) that records which rows need embedding. Uses `SELECT FOR UPDATE SKIP LOCKED` to claim jobs atomically. Never loses events. Survives process crashes and Postgres restarts. |
| **SELECT FOR UPDATE SKIP LOCKED** | A Postgres locking primitive. Claims a row for processing without waiting for other workers (SKIP LOCKED skips rows locked by other transactions). Enables safe parallel workers with no deadlocks. Requires no extensions. |
| **Worker** | The `pgsemantic worker` daemon. Polls `pgvector_setup_queue`, fetches source text, generates embeddings, writes them back to the source table's `embedding` column. Runs continuously alongside your application. |
| **Trigger** | A Postgres trigger installed by `pgsemantic apply` on watched tables. Fires on INSERT or UPDATE of the watched column. Writes a job into `pgvector_setup_queue`. Uses a Postgres function (`pgvector_setup_notify_fn`). |
| **MCP** | Model Context Protocol. A JSON-RPC protocol (over stdio or HTTP/SSE) that lets AI agents (Claude, Cursor, Copilot) discover and call tools exposed by a server. Developed by Anthropic, now open standard. |
| **FastMCP** | The high-level server API from the official `mcp` Python package. Lets you define MCP tools with a `@mcp.tool()` decorator. Handles protocol, validation, and serialization. |
| **MCP Tool** | A named function an MCP server exposes to AI agents. Tools have input schemas (auto-generated from Python type hints) and return values. Agents call tools by name. |
| **hybrid_search** | A search mode that combines pgvector ANN similarity with SQL WHERE filters in a single query. Uses pre-filtering + `hnsw.iterative_scan` to avoid the recall degradation of post-filtering. |
| **Pre-filtering** | Applying WHERE clauses before the vector scan. With `hnsw.iterative_scan = relaxed_order` (pgvector ≥0.8.0), the index scans enough candidates to satisfy both the filter and the limit. Correct approach for selective filters. |
| **Post-filtering** | The naive approach: retrieve K approximate neighbors from the HNSW index, then apply WHERE clause. When the filter is selective, you get far fewer than K results. Do not use for selective filters. |
| **Coverage** | The percentage of rows in a watched table that have a non-NULL `embedding` column. 100% coverage means all rows are searchable. Shown by `pgsemantic status`. |
| **Stale embedding** | An embedding generated from an older version of the text. The row was updated but the embedding was not yet refreshed. The worker resolves staleness. |
| **Dual-column migration** | The zero-downtime strategy for switching embedding models. Add `embedding_new` column, backfill it while `embedding` continues to serve, validate overlap, atomic rename swap, drop old column. |
| **TABLESAMPLE SYSTEM(5)** | Postgres clause that samples ~5% of table pages randomly. Used by `inspect` to estimate average text length without a full sequential scan. Much faster than `SELECT avg(length(col)) FROM table` on large tables. |
| **CONCURRENTLY** | A modifier for `CREATE INDEX` that builds the index without taking an `AccessShareLock` on the table. Allows reads and writes during index creation. Always use this on live tables. |
| **pyproject.toml** | The standard Python packaging configuration file (PEP 517/518). Replaces `setup.py` and `setup.cfg`. Defines dependencies, entry points, metadata. |
| **Entry point** | The `[project.scripts]` section in `pyproject.toml` that maps a CLI command name to a Python function. What makes `pgsemantic` available as a terminal command after `pip install`. |
| **EmbeddingProvider** | A Python Protocol (interface) defined in `pgsemantic/embeddings/base.py`. Both `local.py` and `openai_provider.py` implement this interface. Allows the rest of the codebase to be provider-agnostic. |

---

## Tech Stack (All Verified)

| Layer | Technology | Version | Reason |
|---|---|---|---|
| Language | Python | 3.10+ | Best ML ecosystem, user's choice |
| CLI framework | Typer | ≥0.9.0 | Type-hint-driven, native Rich integration, modern (30% faster than Click in v0.9) |
| Terminal UI | Rich | ≥13.0.0 | Tables, progress bars, panels, syntax highlighting, colors — all in terminal |
| DB client | psycopg[binary] | ≥3.1.0 | Psycopg3 — modern, async-capable, actively maintained. Binary wheels = no build toolchain needed. |
| pgvector Python | pgvector | ≥0.3.0 | Official package. `from pgvector.psycopg import register_vector` registers the vector type with psycopg3. |
| Embeddings (local) | sentence-transformers | ≥3.0.0 | In-process inference via ONNX/PyTorch. No server, no API key. Works on CPU, Windows/Mac/Linux. |
| Embeddings (cloud) | openai | ≥1.0.0 | Official OpenAI Python SDK. `text-embedding-3-small` (1536d). |
| MCP server | mcp (FastMCP) | ≥1.0.0 | Official Anthropic Python MCP SDK. `@mcp.tool()` decorator auto-generates schemas. |
| Config / secrets | python-dotenv | ≥1.0.0 | Loads `DATABASE_URL`, `OPENAI_API_KEY` etc. from `.env` file. |
| Testing | pytest | ≥8.0.0 | Standard Python test runner. |
| Async testing | pytest-asyncio | ≥0.23.0 | `@pytest.mark.asyncio` for async test functions. |
| Linting | ruff | latest | Fast, opinionated, replaces flake8 + isort + pyupgrade. |
| Type checking | mypy | latest | Strict mode. Catches type errors before runtime. |
| Benchmarking | pytest-benchmark | latest | Reproducible benchmarks integrated into pytest. |
| Packaging | pyproject.toml (PEP 517) | — | Standard 2025 packaging. `python -m build` → wheel + sdist. |
| Dev database | Docker Compose | — | `ankane/pgvector:pg16` image. Postgres 16 + pgvector ready on `docker-compose up -d`. |

---

## Project Structure (Complete)

```
pgsemantic/
├── CLAUDE.md                              ← This file. Single source of truth.
├── README.md                              ← User-facing docs, GIF demo, quickstart, benchmarks
├── pyproject.toml                         ← Package metadata, all dependencies, entry point
├── docker-compose.yml                     ← Postgres 16 + pgvector for local dev
├── .env.example                           ← Template for all environment variables
├── .gitignore                             ← .env, __pycache__, dist/, .venv/, benchmarks/results/
├── .github/
│   └── workflows/
│       ├── ci.yml                         ← Run unit + integration tests on every push
│       └── publish.yml                    ← Publish to PyPI on version tag push
│
├── pgsemantic/
│   ├── __init__.py                        ← Package version: __version__ = "0.1.0"
│   ├── cli.py                             ← Typer root app; registers all subcommands
│   ├── exceptions.py                      ← All typed exceptions used across the project
│   ├── config.py                          ← Read/write .pgsemantic.json; env var loading
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── client.py                      ← psycopg3 connection pool; register_vector(); version check
│   │   ├── introspect.py                  ← Schema inspection: tables, columns, scoring
│   │   ├── queue.py                       ← Queue table: create, enqueue, claim_batch, complete, fail
│   │   └── vectors.py                     ← bulk_update_embeddings(); search_similar(); hybrid_search()
│   │
│   ├── embeddings/
│   │   ├── __init__.py                    ← get_provider(name) factory function
│   │   ├── base.py                        ← EmbeddingProvider Protocol; ProviderConfig dataclass
│   │   ├── local.py                       ← LocalProvider: sentence-transformers, batched embed()
│   │   └── openai_provider.py             ← OpenAIProvider: text-embedding-3-small, batched embed()
│   │
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── inspect.py                     ← pgsemantic inspect <db_url>
│   │   ├── apply.py                       ← pgsemantic apply --table --column [--model] [--db]
│   │   ├── index.py                       ← pgsemantic index --table [--batch-size] [--db]
│   │   ├── worker.py                      ← pgsemantic worker [--db] [--concurrency]
│   │   ├── serve.py                       ← pgsemantic serve [--transport] [--port]
│   │   ├── status.py                      ← pgsemantic status [--db]
│   │   ├── migrate.py                     ← pgsemantic migrate --from --to [--table] [--db]
│   │   └── integrate.py                   ← pgsemantic integrate claude
│   │
│   ├── mcp_server/
│   │   ├── __init__.py
│   │   └── server.py                      ← FastMCP server; all tool definitions
│   │
│   └── worker/
│       ├── __init__.py
│       └── daemon.py                      ← Queue consumer loop; reconnect logic; batch processing
│
├── tests/
│   ├── conftest.py                        ← Shared fixtures: db_conn, docker-compose lifecycle
│   ├── fixtures/
│   │   ├── products.sql                   ← 500 product rows with realistic descriptions
│   │   ├── articles.sql                   ← 100 article rows for documentation demo
│   │   └── tickets.sql                    ← 200 support ticket rows
│   ├── unit/
│   │   ├── test_introspect.py             ← Column scoring, schema parsing (mocked DB)
│   │   ├── test_embeddings.py             ← EmbeddingProvider interface (mocked models)
│   │   ├── test_queue.py                  ← Queue claim/complete logic (mocked DB)
│   │   └── test_config.py                 ← Config read/write, validation
│   └── integration/                       ← All marked @pytest.mark.integration
│       ├── test_apply.py                  ← Full apply against real Postgres
│       ├── test_index.py                  ← Backfill with real embeddings
│       ├── test_search.py                 ← semantic_search + hybrid_search queries
│       └── test_worker.py                 ← End-to-end: INSERT row → worker → embedding appears
│
├── benchmarks/
│   ├── bench_index.py                     ← Measures bulk indexer rows/min (labeled output)
│   ├── bench_search.py                    ← Measures semantic_search latency at 1K/10K/100K rows
│   └── results/                           ← Committed benchmark result files (hardware-labeled)
│
└── examples/
    ├── e-commerce/
    │   └── README.md                      ← How to run e-commerce demo with 500 products
    └── support-tickets/
        └── README.md                      ← How to run support ticket search demo
```

---

## Environment Variables

Create a `.env` file in the repo root. **Never commit this file.**

```env
# ── Required ──────────────────────────────────────────────────────────────
# Standard Postgres connection string. Works with any host.
DATABASE_URL=postgresql://postgres:password@localhost:5432/pgvector_dev

# ── Embedding Provider ────────────────────────────────────────────────────
# "local"  → sentence-transformers all-MiniLM-L6-v2 (default, no API key)
# "openai" → OpenAI text-embedding-3-small (requires OPENAI_API_KEY)
EMBEDDING_PROVIDER=local

# Required only if EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...

# ── MCP Server ────────────────────────────────────────────────────────────
# "stdio" → for Claude Desktop and Cursor (default)
# "http"  → for remote agents and API access
MCP_TRANSPORT=stdio

# Required only if MCP_TRANSPORT=http
MCP_PORT=3000

# Bearer token for HTTP transport auth (leave empty to disable auth in dev)
MCP_AUTH_TOKEN=

# ── Worker Settings ───────────────────────────────────────────────────────
# How many queue jobs to claim per poll loop iteration
WORKER_BATCH_SIZE=10

# Milliseconds to sleep when the queue is empty (avoids busy polling)
WORKER_POLL_INTERVAL_MS=500

# How many times to retry a failed embedding job before marking it 'failed'
WORKER_MAX_RETRIES=3

# ── Logging ───────────────────────────────────────────────────────────────
# "debug" | "info" | "warn" | "error"
LOG_LEVEL=info
```

Rules:
- Every new env var must be added to this CLAUDE.md and to `.env.example` in the same commit.
- Never read `os.environ` directly — always go through `pgsemantic/config.py`.
- Config is loaded once at startup via `Settings` dataclass in `config.py`.

---

## All Key Commands

```bash
# ── Setup ─────────────────────────────────────────────────────────────────

# Install for development (editable install with dev extras)
pip install -e ".[dev]"

# Start local Postgres 16 + pgvector (required for integration tests and local dev)
docker-compose up -d

# Verify local database is ready
pgsemantic inspect postgresql://postgres:password@localhost:5432/pgvector_dev


# ── Main CLI Commands ─────────────────────────────────────────────────────

# 1. Scan a database and score columns for semantic search suitability
pgsemantic inspect <DATABASE_URL>

# 2. Set up semantic search on a table/column (extension + vector column + index + queue + trigger)
pgsemantic apply --table products --column description
pgsemantic apply --table products --column description --model openai
pgsemantic apply --table products --column description --db postgresql://...

# 3. Bulk embed all existing rows (backfill)
pgsemantic index --table products
pgsemantic index --table products --batch-size 64
pgsemantic index --table products --db postgresql://...

# 4. Start the background worker (keeps embeddings fresh as data changes)
pgsemantic worker
pgsemantic worker --db postgresql://...
pgsemantic worker --concurrency 2

# 5. Start the MCP server (for Claude Desktop / Cursor)
pgsemantic serve                            # stdio transport (default)
pgsemantic serve --transport http           # HTTP/SSE transport
pgsemantic serve --transport http --port 3000

# 6. Show embedding health dashboard
pgsemantic status
pgsemantic status --db postgresql://...

# 7. Migrate embedding model (zero-downtime dual-column strategy)
pgsemantic migrate --from all-MiniLM-L6-v2 --to text-embedding-3-small
pgsemantic migrate --from all-MiniLM-L6-v2 --to text-embedding-3-small --table products
pgsemantic migrate --from all-MiniLM-L6-v2 --to text-embedding-3-small --confirm

# 8. Generate Claude Desktop config
pgsemantic integrate claude


# ── Development ───────────────────────────────────────────────────────────

# Run all unit tests (no Docker required)
pytest tests/unit/ -v

# Run all integration tests (requires docker-compose up)
pytest tests/integration/ -m integration -v

# Run all tests
pytest

# Run linter
ruff check .

# Auto-fix lint issues
ruff check . --fix

# Type check
mypy pgsemantic/ --strict

# Run benchmarks (outputs hardware-labeled results to benchmarks/results/)
pytest benchmarks/ -m benchmark --benchmark-save=results


# ── Packaging ────────────────────────────────────────────────────────────

# Build wheel + source distribution
python -m build

# Publish to PyPI (maintainer only, after tagging a release)
twine upload dist/*

# Smoke test on clean virtualenv (verify pip install works)
python -m venv /tmp/pgvs-test && source /tmp/pgvs-test/bin/activate
pip install pgsemantic
pgsemantic --help
pgsemantic --version
```

---

## pyproject.toml (Authoritative)

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pgsemantic"
version = "0.1.0"
description = "Zero-config semantic search bootstrap for any PostgreSQL database"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.10"
authors = [
  { name = "Sai Ram Varma Budharaju" }
]
keywords = ["pgvector", "postgresql", "semantic-search", "rag", "embeddings", "mcp"]
classifiers = [
  "Development Status :: 3 - Alpha",
  "Intended Audience :: Developers",
  "License :: OSI Approved :: MIT License",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
]

dependencies = [
  "typer>=0.9.0",           # CLI framework with Rich integration
  "rich>=13.0.0",           # Terminal tables, progress bars, colors
  "psycopg[binary]>=3.1.0", # Postgres client (psycopg3, binary wheels)
  "pgvector>=0.3.0",        # register_vector() for psycopg3
  "sentence-transformers>=3.0.0",  # Local embedding models (all-MiniLM-L6-v2)
  "openai>=1.0.0",          # OpenAI text-embedding-3-small
  "mcp>=1.0.0",             # Official Anthropic MCP Python SDK (FastMCP)
  "python-dotenv>=1.0.0",   # Load DATABASE_URL etc. from .env
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0.0",
  "pytest-asyncio>=0.23.0",
  "pytest-benchmark>=4.0.0",
  "ruff",
  "mypy>=1.8.0",
  "types-psycopg2",          # mypy stubs
]

[project.scripts]
# This line creates the `pgsemantic` binary after pip install
pgsemantic = "pgsemantic.cli:app"

[project.urls]
Homepage = "https://github.com/yourusername/pgsemantic"
Issues = "https://github.com/yourusername/pgsemantic/issues"

[tool.ruff]
target-version = "py310"
line-length = 100
select = ["E", "F", "I", "UP", "N", "B", "SIM"]
ignore = ["E501"]  # line length handled by formatter

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = false

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
  "integration: requires a running Postgres instance (docker-compose up)",
  "benchmark: performance benchmarks, not run in CI",
]
testpaths = ["tests"]
```

---

## Docker Compose (Authoritative)

```yaml
# docker-compose.yml
version: "3.8"
services:
  postgres:
    image: ankane/pgvector:pg16       # Postgres 16 with pgvector pre-installed
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: pgvector_dev
    ports:
      - "5432:5432"
    volumes:
      - pgvector_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  pgvector_data:
```

---

## Database Conventions

### Naming — all pgsemantic objects use the `pgvector_setup_` prefix

| Object | Full name | Notes |
|---|---|---|
| Queue table | `pgvector_setup_queue` | One table for all watched tables |
| Trigger PL/pgSQL function | `pgvector_setup_notify_fn` | One shared function, uses `TG_TABLE_NAME` |
| Trigger per table | `pgvector_setup_{table}_trigger` | One trigger per watched table |
| Queue index | `idx_pgvs_queue_pending` | Partial index on pending jobs |
| Vector column | `embedding` | Added to source table |

The `pgvector_setup_` prefix ensures our objects are always distinguishable from user objects
in `pg_catalog`, `\dt`, and any schema introspection tools.

### Source Table Policy (CRITICAL)
- We add **exactly one column** to the user's source table: `embedding vector(N)`
- We add **exactly one trigger** to the user's source table
- We **never** rename, drop, or modify any existing column
- We **never** change any existing constraint, index, or policy on the user's table
- We always print a preview of the exact SQL and ask for confirmation before executing `ALTER TABLE`
- We label our trigger clearly so users can identify and remove it if needed

### Queue Table Schema (AUTHORITATIVE — never change this without a migration file)
```sql
CREATE TABLE IF NOT EXISTS pgvector_setup_queue (
    id          BIGSERIAL       PRIMARY KEY,
    table_name  TEXT            NOT NULL,
    row_id      TEXT            NOT NULL,   -- stringified PK (handles composite PKs)
    column_name TEXT            NOT NULL,   -- which column triggered this job
    operation   TEXT            NOT NULL    CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    status      TEXT            NOT NULL    DEFAULT 'pending'
                                            CHECK (status IN ('pending', 'processing', 'failed')),
    retries     INT             NOT NULL    DEFAULT 0,
    error_msg   TEXT,                       -- set when status = 'failed'
    created_at  TIMESTAMPTZ     NOT NULL    DEFAULT NOW(),
    updated_at  TIMESTAMPTZ     NOT NULL    DEFAULT NOW(),
    UNIQUE (table_name, row_id, column_name)   -- deduplication: one job per row per column
);

-- Partial index: only index pending jobs (not processing/failed)
-- This is the only index the worker uses for polling
CREATE INDEX IF NOT EXISTS idx_pgvs_queue_pending
    ON pgvector_setup_queue (created_at ASC)
    WHERE status = 'pending';
```

Why `UNIQUE (table_name, row_id, column_name)`: If a row is updated 10 times before the
worker runs, we only need to embed it once. The `ON CONFLICT DO UPDATE SET status = 'pending'`
in the trigger re-queues the job and resets it to pending status.

### Trigger SQL (AUTHORITATIVE)

The trigger function (one shared function for all tables):
```sql
CREATE OR REPLACE FUNCTION pgvector_setup_notify_fn()
RETURNS TRIGGER AS $$
DECLARE
    v_row_id TEXT;
BEGIN
    -- Handle composite primary keys by concatenating all PK columns
    -- For simple integer PKs, this is just NEW.id::TEXT
    v_row_id := NEW.id::TEXT;

    IF TG_OP = 'DELETE' THEN
        INSERT INTO pgvector_setup_queue (table_name, row_id, column_name, operation)
        VALUES (TG_TABLE_NAME, OLD.id::TEXT, TG_ARGV[0], 'DELETE')
        ON CONFLICT (table_name, row_id, column_name)
        DO UPDATE SET
            status     = 'pending',
            operation  = 'DELETE',
            retries    = 0,
            error_msg  = NULL,
            updated_at = NOW();
        RETURN OLD;
    ELSE
        INSERT INTO pgvector_setup_queue (table_name, row_id, column_name, operation)
        VALUES (TG_TABLE_NAME, v_row_id, TG_ARGV[0], TG_OP)
        ON CONFLICT (table_name, row_id, column_name)
        DO UPDATE SET
            status     = 'pending',
            operation  = TG_OP,
            retries    = 0,
            error_msg  = NULL,
            updated_at = NOW();
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

The trigger for a specific table (TG_ARGV[0] passes the column name to the function):
```sql
CREATE OR REPLACE TRIGGER pgvector_setup_{table}_trigger
AFTER INSERT OR UPDATE OF {column} OR DELETE
ON {table}
FOR EACH ROW
EXECUTE FUNCTION pgvector_setup_notify_fn('{column}');
```

### HNSW Index Defaults (verified from pgvector docs + Neon recommendations)

```sql
CREATE INDEX CONCURRENTLY ON {table}
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

| Parameter | Value | Source | Notes |
|---|---|---|---|
| `m` | 16 | pgvector default | Max connections per HNSW node. Safe for general use. |
| `ef_construction` | 64 | pgvector default | Build quality. Neon recommends ≥ 2×m (32 min), 64 is safe. |
| `ef_search` | 40 | pgvector default | Set at query time via `SET hnsw.ef_search = 40`. |
| `iterative_scan` | relaxed_order | pgvector ≥0.8.0 | Enables pre-filtering for hybrid search. Set per-session. |

Always use `CONCURRENTLY` — never block the table during index creation on a live database.

### pgvector Version Check

Different pgvector versions have different features. Check on connect:
```sql
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```

| Feature | Minimum pgvector version |
|---|---|
| HNSW index | 0.5.0 |
| `hnsw.iterative_scan` | 0.8.0 |
| Vectors > 2000 dimensions | Not supported (index limitation) |

Store the detected version in the connection pool's app state and use it to conditionally
enable features.

---

## All SQL Constants (Authoritative)

All SQL is defined as module-level constants in the `db/` files. Never write SQL inline in
command files. Here are the authoritative versions:

### `db/introspect.py` SQL Constants

```python
# Get all user-created tables (excluding system schemas)
SQL_GET_USER_TABLES = """
    SELECT
        t.table_name,
        t.table_schema
    FROM information_schema.tables t
    WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
      AND t.table_type = 'BASE TABLE'
    ORDER BY t.table_name;
"""

# Get all text-like columns for a specific table
SQL_GET_TEXT_COLUMNS = """
    SELECT
        column_name,
        data_type,
        character_maximum_length,
        is_nullable
    FROM information_schema.columns
    WHERE table_name  = %(table_name)s
      AND table_schema = %(schema_name)s
      AND data_type IN ('text', 'character varying', 'jsonb', 'character')
    ORDER BY ordinal_position;
"""

# Sample avg text length for a column — TABLESAMPLE avoids full sequential scan
# Substituted at runtime — table/column names validated against information_schema first
SQL_SAMPLE_AVG_LENGTH = """
    SELECT
        avg(length({column}::TEXT))  AS avg_length,
        count(*)                     AS sampled_rows
    FROM {table} TABLESAMPLE SYSTEM(5)
    WHERE {column} IS NOT NULL;
"""

# Check if pgvector extension is installed
SQL_CHECK_VECTOR_EXTENSION = """
    SELECT extname, extversion
    FROM pg_extension
    WHERE extname = 'vector';
"""

# Check if a column exists on a table
SQL_COLUMN_EXISTS = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name   = %(table_name)s
      AND column_name  = %(column_name)s
      AND table_schema = %(schema_name)s;
"""

# Get all columns of a table (for building dynamic WHERE in hybrid_search)
SQL_GET_ALL_COLUMNS = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name   = %(table_name)s
      AND table_schema = %(schema_name)s
    ORDER BY ordinal_position;
"""

# Get row count for a table
SQL_GET_ROW_COUNT = """
    SELECT reltuples::BIGINT AS estimate
    FROM pg_class
    WHERE relname = %(table_name)s;
"""
```

### `db/vectors.py` SQL Constants

```python
# Fetch rows that need embedding (embedding IS NULL)
# table_name and column validated before string formatting
SQL_FETCH_UNEMBEDDED = """
    SELECT id, {column}::TEXT AS content
    FROM {table}
    WHERE {column} IS NOT NULL
      AND embedding IS NULL
    ORDER BY id
    LIMIT %(limit)s
    OFFSET %(offset)s;
"""

# Update embedding for a single row
SQL_UPDATE_EMBEDDING = """
    UPDATE {table}
    SET embedding = %(embedding)s::vector
    WHERE id = %(row_id)s;
"""

# Semantic search — pure cosine similarity, no filters
SQL_SEMANTIC_SEARCH = """
    SELECT
        id,
        {column}::TEXT          AS content,
        1 - (embedding <=> %(query_vector)s::vector) AS similarity
    FROM {table}
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %(query_vector)s::vector
    LIMIT %(limit)s;
"""

# Count embedded rows
SQL_COUNT_EMBEDDED = """
    SELECT count(*) AS embedded_count
    FROM {table}
    WHERE embedding IS NOT NULL;
"""
```

### `db/queue.py` SQL Constants

```python
# Create queue table
SQL_CREATE_QUEUE_TABLE = """
    CREATE TABLE IF NOT EXISTS pgvector_setup_queue (
        id          BIGSERIAL       PRIMARY KEY,
        table_name  TEXT            NOT NULL,
        row_id      TEXT            NOT NULL,
        column_name TEXT            NOT NULL,
        operation   TEXT            NOT NULL    CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
        status      TEXT            NOT NULL    DEFAULT 'pending'
                                                CHECK (status IN ('pending', 'processing', 'failed')),
        retries     INT             NOT NULL    DEFAULT 0,
        error_msg   TEXT,
        created_at  TIMESTAMPTZ     NOT NULL    DEFAULT NOW(),
        updated_at  TIMESTAMPTZ     NOT NULL    DEFAULT NOW(),
        UNIQUE (table_name, row_id, column_name)
    );
"""

# Create partial index on pending jobs only
SQL_CREATE_QUEUE_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_pgvs_queue_pending
        ON pgvector_setup_queue (created_at ASC)
        WHERE status = 'pending';
"""

# Claim a batch of jobs atomically (SELECT FOR UPDATE SKIP LOCKED)
SQL_CLAIM_BATCH = """
    UPDATE pgvector_setup_queue
    SET status     = 'processing',
        updated_at = NOW()
    WHERE id IN (
        SELECT id
        FROM pgvector_setup_queue
        WHERE status = 'pending'
        ORDER BY created_at ASC
        LIMIT %(batch_size)s
        FOR UPDATE SKIP LOCKED
    )
    RETURNING id, table_name, row_id, column_name, operation;
"""

# Mark a job as successfully completed (delete it — completed jobs don't need to be kept)
SQL_COMPLETE_JOB = """
    DELETE FROM pgvector_setup_queue
    WHERE id = %(job_id)s;
"""

# Mark a job as failed
SQL_FAIL_JOB = """
    UPDATE pgvector_setup_queue
    SET status     = 'failed',
        retries    = retries + 1,
        error_msg  = %(error_msg)s,
        updated_at = NOW()
    WHERE id = %(job_id)s;
"""

# Re-queue a failed job for retry (reset to pending)
SQL_RETRY_JOB = """
    UPDATE pgvector_setup_queue
    SET status     = 'pending',
        updated_at = NOW()
    WHERE id = %(job_id)s
      AND retries < %(max_retries)s;
"""

# Count pending jobs for a table
SQL_COUNT_PENDING = """
    SELECT count(*) AS pending_count
    FROM pgvector_setup_queue
    WHERE table_name = %(table_name)s
      AND status     = 'pending';
"""
```

---

## Commands — Full Specification

### `pgsemantic inspect <database_url>`

**Purpose:** Scan a database schema and score every text column for semantic search suitability.
Helps the user decide what to embed before running `apply`.

**Algorithm:**
1. Connect to the database
2. Query `information_schema.tables` for all user tables (exclude pg_catalog, information_schema)
3. For each table, query `information_schema.columns` for all TEXT/VARCHAR/JSONB columns
4. For each qualifying column:
   a. Run `TABLESAMPLE SYSTEM(5)` to estimate average text length (~5% of pages sampled)
   b. Apply heuristic scoring:
      - avg_length > 200 chars → base score ★★★
      - avg_length 50–200 chars → base score ★★☆
      - avg_length < 50 chars → base score ★☆☆
      - column name in (`description`, `body`, `content`, `text`, `summary`, `notes`,
        `bio`, `review`, `comment`, `message`, `details`, `abstract`, `excerpt`) → +½ star
      - column name in (`id`, `status`, `code`, `type`, `slug`, `url`, `email`, `phone`,
        `uuid`, `hash`, `token`) → score → ★☆☆ (forced low)
5. Sort all columns by score (★★★ first), then by avg_length descending
6. Render a Rich table

**Output:**
```
╭────────────────────────────────────────────────────────────────────╮
│         Semantic Search Candidates — postgresql://localhost/myapp   │
╰────────────────────────────────────────────────────────────────────╯

 Table               Column          Score   Avg Length   Type
 ──────────────────────────────────────────────────────────────────
 products            description     ★★★    847 chars    text
 support_tickets     body            ★★★    1,203 chars  text
 articles            content         ★★★    4,821 chars  text
 users               bio             ★★☆    143 chars    text
 products            name            ★☆☆    34 chars     varchar
 products            category        ★☆☆    12 chars     varchar

ℹ  Scoring is heuristic (text length + column name patterns).
   Always verify recommendations make sense for your use case.

Next step:
  pgsemantic apply --table products --column description
```

**Important:** The scoring is heuristic, not ML-based. Document this clearly in the output
and in the README. Never claim the scoring is smart or intelligent.

---

### `pgsemantic apply --table <t> --column <c> [--model local|openai] [--db <url>]`

**Purpose:** Install everything needed for semantic search on one table/column.

**Steps (in order, with confirmation before any destructive action):**

**Step 1 — Check pgvector extension**
```sql
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```
- Found → print `✓ pgvector X.Y.Z detected`
- Not found → attempt `CREATE EXTENSION IF NOT EXISTS vector;`
  - Success → print `✓ pgvector extension installed`
  - Failure (permissions) → print host-specific instructions and exit 1:
    ```
    ✗ Could not install pgvector extension (permission denied).

    To enable pgvector on your host:
      Supabase:   Dashboard → Database → Extensions → enable "vector"
      Neon:       Run: CREATE EXTENSION vector;  (connect as project owner)
      RDS:        Run: CREATE EXTENSION vector;  (requires rds_superuser role)
      Railway:    Run: CREATE EXTENSION vector;  (connect as postgres user)
      Bare metal: apt install postgresql-16-pgvector
                  Then: CREATE EXTENSION vector;
    ```

**Step 2 — Check if embedding column already exists**
```sql
SELECT column_name FROM information_schema.columns
WHERE table_name = %(table)s AND column_name = 'embedding';
```
- Already exists → print warning and ask user to confirm overwrite or exit

**Step 3 — Preview SQL and ask for confirmation**
Print the exact SQL that will be executed. Ask:
```
About to run:

  ALTER TABLE products ADD COLUMN embedding vector(384);
  CREATE INDEX CONCURRENTLY ON products USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

Proceed? [y/N]:
```
Never run `ALTER TABLE` without explicit confirmation.

**Step 4 — Add vector column**
```sql
ALTER TABLE {table} ADD COLUMN embedding vector({dimensions});
```
dimensions = 384 for local (all-MiniLM-L6-v2), 1536 for openai (text-embedding-3-small)

**Step 5 — Create HNSW index (CONCURRENTLY)**
```sql
CREATE INDEX CONCURRENTLY ON {table}
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Step 6 — Create queue table (if not exists)**
Run `SQL_CREATE_QUEUE_TABLE` and `SQL_CREATE_QUEUE_INDEX` from `db/queue.py`

**Step 7 — Install trigger function (idempotent)**
Run the `CREATE OR REPLACE FUNCTION pgvector_setup_notify_fn()` SQL

**Step 8 — Install trigger on table**
Run `CREATE OR REPLACE TRIGGER pgvector_setup_{table}_trigger ...`

**Step 9 — Write `.pgsemantic.json`**
```json
{
  "version": "0.1.0",
  "database_url": "postgresql://...",
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
      "applied_at": "2026-02-27T10:00:00Z"
    }
  ]
}
```

**Step 10 — Print summary and next steps**
```
✓ pgvector 0.7.4 detected
✓ Column 'embedding vector(384)' added to products
✓ HNSW index created (m=16, ef_construction=64)
✓ Queue table pgvector_setup_queue created
✓ Trigger pgvector_setup_products_trigger installed
✓ Config saved to .pgsemantic.json

Next steps:
  pgsemantic index --table products   # embed your existing rows
  pgsemantic worker                   # keep new rows embedded automatically
  pgsemantic serve                    # start MCP server for Claude/Cursor
```

---

### `pgsemantic index --table <t> [--batch-size 32] [--db <url>]`

**Purpose:** Bulk embed all existing rows where `embedding IS NULL`.

**Algorithm:**
1. Load config from `.pgsemantic.json` to get model name and column name
2. Initialize embedding provider
3. Count unembedded rows: `SELECT count(*) FROM {table} WHERE embedding IS NULL`
4. Loop in pages of `batch_size` rows:
   a. Fetch batch: `SQL_FETCH_UNEMBEDDED` with `LIMIT batch_size OFFSET offset`
   b. Extract text content from each row
   c. Call `provider.embed(texts)` — single batch call (not one-by-one)
   d. Execute `SQL_UPDATE_EMBEDDING` for each row
   e. Update Rich progress bar
5. Print completion summary

**Progress bar output:**
```
Indexing products.description ━━━━━━━━━━━━━━━━━━━━━━━ 2,847/12,000  487 rows/min  ETA 18m
```

**Honest performance claim:**
- We claim ≥500 rows/min on modern CPU with default batch_size=32
- This is a conservative estimate based on sentence-transformers batched inference
- We do NOT publish a specific number until we run `benchmarks/bench_index.py`
  on documented hardware and commit the results to `benchmarks/results/`

**Resume support:**
- `index` is idempotent: re-running it only processes rows where `embedding IS NULL`
- If interrupted, just re-run — it picks up where it left off

---

### `pgsemantic worker [--db <url>] [--concurrency 1]`

**Purpose:** Long-running daemon that processes the queue table, keeping embeddings
in sync as source data changes.

**Main loop (in `worker/daemon.py`):**
```python
async def run_worker(settings: Settings) -> None:
    while True:
        async with get_connection(settings.database_url) as conn:
            try:
                jobs = await claim_batch(conn, batch_size=settings.worker_batch_size)
                if not jobs:
                    await asyncio.sleep(settings.worker_poll_interval_ms / 1000)
                    continue
                await process_jobs(conn, jobs, provider)
            except psycopg.OperationalError:
                # Reconnect on connection loss
                await asyncio.sleep(backoff.next())
                continue
            else:
                backoff.reset()
```

**For each job:**
- `INSERT` or `UPDATE`: fetch source text → embed → `UPDATE {table} SET embedding = ... WHERE id = {row_id}`
- `DELETE`: `UPDATE {table} SET embedding = NULL WHERE id = {row_id}` (or leave NULL if row gone)

**Reconnection strategy:**
- Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s (max)
- Never crash permanently on connection loss
- Log reconnection attempts at WARNING level

**Logging:**
```
INFO  Worker started. Polling pgvector_setup_queue every 500ms.
INFO  Claimed 10 jobs from queue.
INFO  Embedded products#1234 (description, 384d) in 42ms
INFO  Queue empty. Sleeping 500ms.
```

---

### `pgsemantic serve [--transport stdio|http] [--port 3000]`

**Purpose:** Start the MCP server so Claude Desktop, Cursor, or any MCP-compatible
agent can call semantic_search, hybrid_search, and get_embedding_status.

**Implementation in `mcp_server/server.py`:**

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="pgsemantic", version="0.1.0")

@mcp.tool()
def semantic_search(query: str, table: str, limit: int = 10) -> list[dict]:
    """
    Search for rows in a Postgres table semantically similar to a natural-language query.

    Args:
        query: Natural language search query (e.g. "wireless headphones under $100")
        table: Name of the table to search (must have been set up with pgsemantic apply)
        limit: Maximum number of results to return (default 10, max 100)

    Returns:
        List of matching rows with their similarity scores, ordered by relevance.
    """
    ...

@mcp.tool()
def hybrid_search(
    query: str,
    table: str,
    filters: dict = {},
    limit: int = 10
) -> list[dict]:
    """
    Semantic search combined with exact SQL filters in a single query.

    Args:
        query: Natural language search query
        table: Name of the table to search
        filters: Key-value pairs for exact match filters (e.g. {"category": "laptop", "in_stock": true})
        limit: Maximum number of results to return

    Returns:
        List of matching rows filtered by both semantic similarity and exact conditions.
    """
    ...

@mcp.tool()
def get_embedding_status(table: str) -> dict:
    """
    Get embedding coverage and sync health for a table.

    Args:
        table: Name of the table to check

    Returns:
        Coverage percentage, pending queue depth, model info, last sync time.
    """
    ...
```

**Transport:**
- `stdio` (default): for Claude Desktop and Cursor — communicates over stdin/stdout
- `http`: for remote agents — HTTP/SSE on `localhost:{port}`

**Starting with stdio:**
```python
if settings.mcp_transport == "stdio":
    mcp.run()  # uses stdio by default
elif settings.mcp_transport == "http":
    mcp.run(transport="sse", port=settings.mcp_port)
```

---

### Hybrid Search — Full SQL Implementation

```python
async def hybrid_search(
    conn: psycopg.AsyncConnection,
    table: str,
    column: str,
    query_vector: list[float],
    filters: dict[str, Any],
    limit: int,
    pgvector_version: tuple[int, int, int],
) -> list[dict[str, Any]]:

    # Enable iterative scan for pre-filtering (pgvector >= 0.8.0 only)
    if pgvector_version >= (0, 8, 0):
        await conn.execute("SET hnsw.iterative_scan = relaxed_order")

    # Validate filter keys against actual table columns (prevent injection)
    valid_columns = await get_column_names(conn, table)
    safe_filters = {k: v for k, v in filters.items() if k in valid_columns}

    # Build parameterized WHERE clause
    where_clauses = ["embedding IS NOT NULL"]
    params: dict[str, Any] = {
        "query_vector": query_vector,
        "limit": limit,
    }

    for key, value in safe_filters.items():
        if key.endswith("_max"):
            col = key[:-4]
            where_clauses.append(f"{col} <= %({key})s")
        elif key.endswith("_min"):
            col = key[:-4]
            where_clauses.append(f"{col} >= %({key})s")
        else:
            where_clauses.append(f"{key} = %({key})s")
        params[key] = value

    where_sql = " AND ".join(where_clauses)

    # Final query — pre-filtered, single round-trip
    sql = f"""
        SELECT
            *,
            1 - (embedding <=> %(query_vector)s::vector) AS similarity
        FROM {table}
        WHERE {where_sql}
        ORDER BY embedding <=> %(query_vector)s::vector
        LIMIT %(limit)s
    """

    async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        await cur.execute(sql, params)
        return await cur.fetchall()
```

**Why this is correct:**
- Pre-filtering: WHERE is applied before ORDER BY, so pgvector scans enough candidates
- With `hnsw.iterative_scan = relaxed_order`: the index iterates until it finds enough
  results that satisfy the filter (pgvector 0.8.0+)
- Without iterative scan (pgvector < 0.8.0): degrades gracefully to post-filter behavior
  with a warning logged
- Never uses string interpolation for user-supplied filter values — only for validated
  column names

---

### `pgsemantic status [--db <url>]`

**Purpose:** Show a real-time health dashboard for all watched tables.

**Data sources (all standard SQL, no extensions):**
```sql
-- Total rows (fast estimate from pg_class statistics)
SELECT reltuples::BIGINT AS estimate FROM pg_class WHERE relname = %(table)s;

-- Exact embedded count
SELECT count(*) FROM {table} WHERE embedding IS NOT NULL;

-- Queue depth (pending jobs)
SELECT count(*) FROM pgvector_setup_queue
WHERE table_name = %(table)s AND status = 'pending';

-- Failed jobs
SELECT count(*) FROM pgvector_setup_queue
WHERE table_name = %(table)s AND status = 'failed';
```

**Rich terminal output:**
```
╭────────────────────────────────────────────────────────────────────────────────╮
│                        Embedding Health Dashboard                              │
╰────────────────────────────────────────────────────────────────────────────────╯

 Table               Column       Total    Embedded   Coverage   Queue      Model
 ────────────────────────────────────────────────────────────────────────────────
 products            description  12,847   12,843     99.97%     0 pending  all-MiniLM-L6-v2
 support_tickets     body          3,201    3,201     100%       1 pending  all-MiniLM-L6-v2
 articles            content         847      847     100%       0 pending  all-MiniLM-L6-v2

⚠  4 rows in 'products' are missing embeddings.
   Run: pgsemantic index --table products

✓  Worker is keeping embeddings in sync.
   Last activity: 2 minutes ago
```

---

### `pgsemantic migrate --from <model> --to <model> [--table <t>] [--confirm]`

**Purpose:** Zero-downtime embedding model migration using dual-column strategy.

**Steps:**

1. **Validate:** Check that `--from` matches the model recorded in `.pgsemantic.json`
2. **Add new column:** `ALTER TABLE {table} ADD COLUMN embedding_new vector({new_dims})`
3. **Backfill new column:** Same as `index` command but writes to `embedding_new`
4. **Validate overlap:**
   - Run 20 sample queries against both columns
   - For each query, compare top-10 results from old and new embeddings
   - Report Jaccard overlap: "Old and new models agree on X% of top-10 results for 20 sample queries"
5. **Print comparison report:**
   ```
   Migration Validation (20 sample queries)
   ─────────────────────────────────────────
   Average top-10 Jaccard overlap: 73%

   Sample query: "wireless noise-canceling headphones"
   Old model (all-MiniLM-L6-v2) top 3:
     1. Sony WH-1000XM5 (score: 0.91)
     2. Bose QuietComfort 45 (score: 0.89)
     3. Apple AirPods Max (score: 0.87)

   New model (text-embedding-3-small) top 3:
     1. Sony WH-1000XM5 (score: 0.94)
     2. Bose QuietComfort 45 (score: 0.92)
     3. Jabra Evolve2 85 (score: 0.88)

   Proceed with migration? Run with --confirm to apply.
   ```
6. **Atomic swap (only with `--confirm`):**
   ```sql
   BEGIN;
   ALTER TABLE {table} RENAME COLUMN embedding TO embedding_old;
   ALTER TABLE {table} RENAME COLUMN embedding_new TO embedding;
   COMMIT;
   ```
7. **Rebuild HNSW index on new column**
8. **Drop old column:** `ALTER TABLE {table} DROP COLUMN embedding_old`
9. **Update `.pgsemantic.json`** with new model info

---

### `pgsemantic integrate claude`

**Purpose:** Generate the `claude_desktop_config.json` snippet so Claude Desktop can
connect to the MCP server automatically.

**Reads:** `.pgsemantic.json` for database URL and tables.

**Generates and prints:**
```json
{
  "mcpServers": {
    "pgsemantic": {
      "command": "pgsemantic",
      "args": ["serve"],
      "env": {
        "DATABASE_URL": "postgresql://user:password@host/db",
        "EMBEDDING_PROVIDER": "local"
      }
    }
  }
}
```

**Also prints:**
```
Add the above to your Claude Desktop config file:

  macOS:    ~/Library/Application Support/Claude/claude_desktop_config.json
  Windows:  %APPDATA%\Claude\claude_desktop_config.json
  Linux:    ~/.config/claude/claude_desktop_config.json

Then restart Claude Desktop. Claude will now have access to:
  • semantic_search  — search any indexed table by meaning
  • hybrid_search    — semantic search + SQL filters
  • get_embedding_status — check embedding health

Try asking Claude: "Search the products table for wireless headphones under $100"
```

---

## Embedding Providers — Full Specification

### Abstract Interface (`embeddings/base.py`)

```python
from typing import Protocol
from dataclasses import dataclass

@dataclass
class ProviderConfig:
    model_name: str
    dimensions: int
    batch_size: int = 32

class EmbeddingProvider(Protocol):
    config: ProviderConfig

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts. Returns one vector per text.
        All vectors have length == self.config.dimensions.
        texts must be non-empty strings. Empty strings raise ValueError.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text. Used at search time.
        Equivalent to embed([text])[0] but may use a different prompt prefix.
        """
        ...
```

### Local Provider (`embeddings/local.py`)

```python
# Model: all-MiniLM-L6-v2
# Dimensions: 384
# Download size: 22 MB (to ~/.cache/huggingface/ on first use)
# Speed: ≥500 rows/min in batched mode on modern CPU (conservative estimate)
# Requirements: sentence-transformers (included in dependencies)
# No API key, no internet required after first download

from sentence_transformers import SentenceTransformer
import asyncio

MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSIONS = 384

class LocalProvider:
    config = ProviderConfig(model_name=MODEL_NAME, dimensions=DIMENSIONS, batch_size=32)

    def __init__(self) -> None:
        # Model downloads automatically on first init
        self._model = SentenceTransformer(MODEL_NAME)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must be non-empty")
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        # sentence-transformers is sync — run in thread pool to avoid blocking event loop
        return await asyncio.to_thread(self.embed, texts)
```

### OpenAI Provider (`embeddings/openai_provider.py`)

```python
# Model: text-embedding-3-small
# Dimensions: 1536
# Cost: $0.02 per 1M tokens (verify before publishing in README)
# Rate limits: 1,000,000 TPM on Tier 1 (verify current limits)
# Requirements: OPENAI_API_KEY in environment

from openai import OpenAI

MODEL_NAME = "text-embedding-3-small"
DIMENSIONS = 1536
MAX_BATCH_SIZE = 2048  # OpenAI supports up to 2048 inputs per request

class OpenAIProvider:
    config = ProviderConfig(model_name=MODEL_NAME, dimensions=DIMENSIONS, batch_size=100)

    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must be non-empty")
        response = self._client.embeddings.create(
            model=MODEL_NAME,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
```

### Adding a New Provider

1. Create `pgsemantic/embeddings/{name}_provider.py`
2. Implement the `EmbeddingProvider` Protocol
3. Register in `pgsemantic/embeddings/__init__.py`'s `get_provider()` factory
4. Add `--model {name}` to the `apply` command's options
5. Add unit test in `tests/unit/test_embeddings.py`
6. Add to this CLAUDE.md under Embedding Providers
7. Document required env vars and installation steps

---

## MCP Tools — Full Reference

These are the exact tool signatures. **Never rename a tool.** It breaks existing agent configs.

### `semantic_search`
```
Input:
  query: str          — natural language query
  table: str          — must be a watched table (in .pgsemantic.json)
  limit: int = 10     — max results (1–100)

Output: list[dict] where each dict contains:
  - all columns from the source table row
  - "similarity": float (0.0 to 1.0, higher = more similar)

Errors:
  - TableNotFoundError if table is not watched
  - EmbeddingProviderError if embedding fails
```

### `hybrid_search`
```
Input:
  query: str                — natural language query
  table: str                — must be a watched table
  filters: dict = {}        — column → value pairs for exact matching
                              supports "_max" and "_min" suffixes for range filters
                              e.g. {"category": "laptop", "price_max": 1500}
  limit: int = 10           — max results

Output: same as semantic_search

Filter key conventions:
  - "column"      → WHERE column = value
  - "column_max"  → WHERE column <= value
  - "column_min"  → WHERE column >= value
  - Only columns that exist in the table schema are accepted (others silently ignored)

Errors:
  - TableNotFoundError if table is not watched
  - InvalidFilterError if a filter key is not a valid column name pattern
```

### `get_embedding_status`
```
Input:
  table: str   — table name

Output: dict
  - table: str
  - total_rows: int
  - embedded_rows: int
  - coverage_pct: float (0.0 to 100.0)
  - pending_queue: int
  - failed_queue: int
  - model_name: str
  - dimensions: int
  - applied_at: str (ISO 8601)
```

---

## Exceptions (`exceptions.py` — Authoritative)

```python
class PgvectorSetupError(Exception):
    """Base exception for all pgsemantic errors."""

class ExtensionNotFoundError(PgvectorSetupError):
    """pgvector extension is not installed or not accessible."""

class TableNotFoundError(PgvectorSetupError):
    """The specified table does not exist in the database."""

class TableNotWatchedError(PgvectorSetupError):
    """The table exists but has not been set up with pgsemantic apply."""

class ColumnNotFoundError(PgvectorSetupError):
    """The specified column does not exist on the table."""

class ColumnAlreadyExistsError(PgvectorSetupError):
    """The embedding column already exists on the table."""

class DimensionMismatchError(PgvectorSetupError):
    """The existing embedding column has different dimensions than the requested model."""

class EmbeddingProviderError(PgvectorSetupError):
    """Error from the embedding model (API error, model not found, etc.)."""

class QueueError(PgvectorSetupError):
    """Error interacting with the pgvector_setup_queue table."""

class ConfigError(PgvectorSetupError):
    """Error reading or writing .pgsemantic.json."""

class MigrationError(PgvectorSetupError):
    """Error during zero-downtime model migration."""

class InvalidFilterError(PgvectorSetupError):
    """A filter key in hybrid_search does not correspond to a valid column."""
```

All `psycopg.Error`, `openai.APIError`, and other third-party exceptions must be caught
and re-raised as one of the above typed exceptions with a human-readable message.
**Never let a raw third-party exception propagate to CLI output or MCP responses.**

---

## Code Style Rules

### Python Style
- **Python 3.10+.** Use `match` statements where appropriate, `X | Y` union types.
- **Type hints on every function and method.** No untyped parameters or return values.
- **No `Any`.** Use specific types. If you find yourself needing `Any`, use `TypeVar` or narrow with `isinstance`.
- **`Protocol` for interfaces.** Use `typing.Protocol`, not abstract base classes.
- **`dataclass` for data structures.** Named parameters, not positional tuples or plain dicts.
- **`async/await` for all I/O.** All database calls are async using psycopg3's async API.
- **`asyncio.to_thread()` for blocking calls.** sentence-transformers is synchronous.
  Wrap it in `await asyncio.to_thread(provider.embed, texts)` when called from async context.
- **No bare `except:` or `except Exception:` without re-raising.** Catch specific exceptions.
  If you must catch broadly, always log and re-raise.
- **No `print()` in library code.** Use `logging.getLogger(__name__)` in all files under
  `db/`, `embeddings/`, `worker/`, `mcp_server/`. Use `rich.console.Console()` only in `commands/`.
- **All SQL is module-level constants.** SQL strings in `UPPER_CASE` at the top of the file.
  Never write SQL inline in business logic.
- **All SQL uses `%(name)s` parameterized placeholders.** Never use `%s` positional for
  complex queries (hard to maintain). Exception: simple single-param queries.
- **No string interpolation in SQL for user input.** Table names and column names are
  validated against `information_schema` before being formatted into SQL.
  Values are always parameterized. SQL injection is always wrong.
- **No ORMs.** Use psycopg3 directly with `dict_row` row factory.

### File Naming
- All Python files: `snake_case.py`
- CLI command names (as shown to users): `kebab-case` (handled by Typer automatically from function names)

### Imports
```python
# Standard library
import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

# Third-party (blank line separator)
import psycopg
from rich.console import Console

# Local (blank line separator)
from pgsemantic.config import Settings
from pgsemantic.exceptions import TableNotFoundError
```

### Configuration Constants in `config.py`

```python
# HNSW index parameters — do not change without updating CLAUDE.md and docs/
HNSW_M: int = 16
HNSW_EF_CONSTRUCTION: int = 64
HNSW_EF_SEARCH: int = 40

# Default embedding model
DEFAULT_LOCAL_MODEL: str = "all-MiniLM-L6-v2"
DEFAULT_LOCAL_DIMENSIONS: int = 384

OPENAI_MODEL: str = "text-embedding-3-small"
OPENAI_DIMENSIONS: int = 1536

# Worker defaults
DEFAULT_WORKER_BATCH_SIZE: int = 10
DEFAULT_WORKER_POLL_INTERVAL_MS: int = 500
DEFAULT_WORKER_MAX_RETRIES: int = 3
DEFAULT_INDEX_BATCH_SIZE: int = 32

# Queue table name — never change after initial release
QUEUE_TABLE_NAME: str = "pgvector_setup_queue"

# Config file name
CONFIG_FILE_NAME: str = ".pgsemantic.json"
```

---

## Testing Strategy

### Unit Tests (`tests/unit/`)
- Test each module in complete isolation.
- Mock the database with `pytest-mock` or `unittest.mock.AsyncMock`.
- Mock embedding models to return **deterministic fixed vectors** (e.g., `[0.1] * 384`).
- No real Postgres. No real HTTP calls. No real model inference.
- Fast: entire unit test suite must run in < 10 seconds.
- Run with: `pytest tests/unit/ -v`

### Integration Tests (`tests/integration/`)
- Use a real Postgres instance from docker-compose.
- All marked `@pytest.mark.integration`.
- The `conftest.py` `db_conn` fixture:
  1. Checks docker-compose is up (skip with message if not)
  2. Creates a fresh test database for each test session
  3. Loads fixture data from `tests/fixtures/`
  4. Drops the database after the session
- Run with: `pytest tests/integration/ -m integration`
- Must be run before any pull request is merged.

### Benchmark Tests (`tests/benchmarks/` / `benchmarks/`)
- NOT run in CI (marked `@pytest.mark.benchmark`).
- Must output hardware-labeled results.
- Run manually: `pytest benchmarks/ -m benchmark --benchmark-save=results`
- Results committed to `benchmarks/results/` with hardware tag in filename.
- Format: `bench_index_M2Pro_16GB_pg16_pgvector071_20260227.json`

### Test Data (Fixtures)
- All test data in `tests/fixtures/` as SQL files.
- `products.sql`: 500 rows with realistic product names + descriptions (varied lengths)
- `articles.sql`: 100 rows with realistic article titles + content
- `tickets.sql`: 200 rows with realistic support ticket subjects + bodies
- Never hardcode test data inline in test functions.

### Coverage Targets
- Milestone 1–3: ≥60% line coverage
- Pre-release v1.0: ≥75% line coverage
- Check with: `pytest --cov=pgsemantic --cov-report=term-missing`

### Never Skip a Failing Test
If a test fails, either fix the code or open a GitHub issue with the failure details.
Never use `@pytest.mark.skip` without a linked issue in the reason string.

---

## What Claude Should Never Do

- **Never claim a specific performance number without a benchmark.** All perf claims require
  a result from `benchmarks/` labeled with hardware and software versions.
- **Never use LISTEN/NOTIFY.** It drops events silently under load. The queue table is the
  only correct CDC approach.
- **Never put all embeddings in a central table.** Always add `embedding` column to source table.
  The central table design causes cross-table HNSW post-filter recall degradation.
- **Never interpolate user input into SQL.** Always use parameterized queries. Validate
  table and column names against `information_schema` before formatting SQL strings.
- **Never run `ALTER TABLE` without explicit user confirmation via `typer.confirm()`.**
- **Never rename an MCP tool.** It breaks existing agent configurations and Claude Desktop setups.
- **Never run `CREATE INDEX` without `CONCURRENTLY`** on a live table.
- **Never add a dependency without justification** in the commit message.
- **Never commit `.env` or any file with secrets.**
- **Never use `print()` in library code** (`db/`, `embeddings/`, `worker/`, `mcp_server/`).
  Use `logging.getLogger(__name__)`.
- **Never let a raw third-party exception reach the user.** Catch and convert to typed exceptions.
- **Never change the `pgvector_setup_queue` schema** without creating a migration script
  and bumping the package version.
- **Never modify user table columns** other than adding the `embedding` column.
- **Never claim to "cut latency in half."** This was a hallucination in an earlier spec.
- **Never use LISTEN/NOTIFY.** (Worth repeating.)

---

## Postgres Host Compatibility

This tool requires **only** standard PostgreSQL and the `pgvector` extension.
No `pg_cron`, no `pgmq`, no `shared_preload_libraries` changes.

| Host | pgvector support | How to enable | Notes |
|---|---|---|---|
| Bare metal / self-hosted | Manual install | `apt install postgresql-16-pgvector` then `CREATE EXTENSION vector;` | Full superuser access |
| Docker (local dev) | Pre-installed | Use `ankane/pgvector:pg16` image | Included in our docker-compose.yml |
| Supabase | Built-in | Dashboard → Database → Extensions → "vector" | No superuser needed |
| Neon | Built-in | `CREATE EXTENSION vector;` as project owner | Works on free tier |
| Railway | Postgres 15+ | `CREATE EXTENSION vector;` as postgres user | Works |
| AWS RDS | Postgres 15+ | `CREATE EXTENSION vector;` requires `rds_superuser` | Note: not `superuser` |
| Google Cloud SQL | Postgres 15+ | `CREATE EXTENSION vector;` as cloudsqlsuperuser | Works |
| Azure Database | Flexible Server, PG 15+ | `CREATE EXTENSION vector;` | Works |

When `CREATE EXTENSION vector` fails, the `apply` command prints host-specific instructions
from the table above and exits with code 1. Never crash with a raw Postgres error.

---

## Performance Targets (Honest)

Numbers marked `*` are estimates pending benchmarks. We run `benchmarks/bench_*.py` and
commit labeled results before publishing any numbers in README.

| Metric | Target | Status | Notes |
|---|---|---|---|
| `semantic_search` p95 (1K rows) | < 10ms | Pending benchmark | HNSW on local Postgres |
| `semantic_search` p95 (100K rows) | < 100ms | Pending benchmark | HNSW m=16 |
| `hybrid_search` p95 (100K rows, 1 filter) | < 150ms | Pending benchmark | With iterative_scan |
| Bulk indexer (local CPU, batched) | ≥ 500 rows/min* | Conservative estimate | sentence-transformers, batch=32 |
| Worker sync latency (trigger → embedding ready) | < 5s p95 | Pending benchmark | Queue poll interval 500ms + embed |
| `pgsemantic apply` (table < 1M rows) | < 30s | Pending benchmark | Excluding index build time |

**Benchmark policy:**
- Every number in README is labeled: `(hardware: M2 Pro, 16GB RAM | PG 16.2 | pgvector 0.7.4)`
- Benchmark script included in repo (`benchmarks/`) so anyone can reproduce
- If we can't measure it, we don't claim it

---

## Open Questions (Decisions Pending)

- [ ] **JSONB column embedding:** When a TEXT column contains JSON, should we stringify the
      whole object or extract specific keys?
      MVP decision: stringify entire JSONB as text.
      Limitation documented in README. Multi-key support in v1.1.

- [ ] **Multi-column embedding:** Should one embedding represent multiple columns
      (e.g., `title + description + tags`) concatenated?
      MVP decision: single column per embedding. If user wants multi-column, they create
      a generated column or a view and point pgsemantic at that.
      Format if concatenating: `"{col1_name}: {value}\n{col2_name}: {value}"` (preserves field labels).

- [ ] **Tables without `id` column:** Queue uses `row_id TEXT`. For composite PKs:
      serialize as `"col1=val1,col2=val2"`. Need a test case.
      → Add to Milestone 2 integration tests.

- [ ] **Very large text (> 8KB per row):** all-MiniLM-L6-v2 has max 256 tokens.
      text-embedding-3-small has max 8191 tokens.
      For very long documents, we truncate silently in MVP.
      Chunking is a v1.1 feature. Document the limitation clearly.

- [ ] **HTTP transport auth:** Bearer token for MVP (simple `Authorization: Bearer {token}`
      header check). OAuth PKCE for v1.0 stable.

- [ ] **pgvector version detection for iterative_scan:** Must check `extversion` from
      `pg_extension` and only set `hnsw.iterative_scan` on pgvector ≥ 0.8.0.
      → Add to `db/client.py` on connect: store `pgvector_version: tuple[int,int,int]`.

- [ ] **Composite primary key support:** Current implementation assumes `id` column.
      Need to handle tables where PK is not named `id` or is composite.
      → Detect PK from `information_schema.key_column_usage` in `apply` step.

---

## Build Order (Authoritative — Do Not Skip Steps)

### Milestone 1 — Core Infrastructure (Days 1–3)
**Goal:** All foundational modules written and unit tests passing. No CLI yet.

**Acceptance criteria:**
- [ ] `pip install -e ".[dev]"` succeeds on Python 3.10, 3.11, 3.12
- [ ] `pytest tests/unit/ -v` — all unit tests pass
- [ ] `mypy pgsemantic/ --strict` — zero errors
- [ ] `ruff check .` — zero errors

**Build order:**
1. `pyproject.toml`
2. `docker-compose.yml`
3. `.env.example`
4. `pgsemantic/__init__.py` (just `__version__ = "0.1.0"`)
5. `pgsemantic/exceptions.py`
6. `pgsemantic/config.py` (Settings dataclass, constants, load_settings())
7. `pgsemantic/db/client.py` (connection pool, register_vector, pgvector version check)
8. `pgsemantic/embeddings/base.py` (EmbeddingProvider Protocol, ProviderConfig)
9. `pgsemantic/embeddings/local.py` (sentence-transformers, batched embed)
10. `pgsemantic/embeddings/openai_provider.py` (OpenAI, batched embed)
11. `pgsemantic/embeddings/__init__.py` (get_provider() factory)
12. `pgsemantic/db/introspect.py` (get_user_tables, get_text_columns, score_column)
13. `pgsemantic/db/queue.py` (create_queue_table, enqueue, claim_batch, complete, fail)
14. `pgsemantic/db/vectors.py` (bulk_update_embeddings, search_similar, hybrid_search)
15. `tests/unit/test_embeddings.py`
16. `tests/unit/test_introspect.py`
17. `tests/unit/test_queue.py`
18. `tests/unit/test_config.py`

---

### Milestone 2 — CLI Commands (Days 4–6)
**Goal:** All four core commands working against real Postgres. Integration tests passing.

**Do NOT build yet:** MCP server, status command, migrate command.

**Acceptance criteria:**
- [ ] `docker-compose up -d && pgsemantic inspect postgresql://postgres:password@localhost:5432/pgvector_dev` renders Rich table
- [ ] `pgsemantic apply --table products --column description` runs through all steps with ✓ checkmarks
- [ ] `pgsemantic index --table products` shows Rich progress bar, reaches 100%
- [ ] INSERT a new row into products → `pgsemantic worker` picks it up within 5 seconds
- [ ] UPDATE a row's description → worker updates embedding within 5 seconds
- [ ] DELETE a row → worker cleans up embedding within 5 seconds
- [ ] `pytest tests/integration/ -m integration` — all integration tests pass

**Build order:**
1. `pgsemantic/cli.py` (Typer app root, version command)
2. `pgsemantic/commands/inspect.py`
3. `pgsemantic/commands/apply.py`
4. `pgsemantic/worker/daemon.py`
5. `pgsemantic/commands/worker.py`
6. `pgsemantic/commands/index.py`
7. `tests/fixtures/products.sql` (500 realistic product rows)
8. `tests/fixtures/articles.sql` (100 article rows)
9. `tests/fixtures/tickets.sql` (200 support ticket rows)
10. `tests/conftest.py` (docker-compose fixture, db_conn fixture)
11. `tests/integration/test_apply.py`
12. `tests/integration/test_index.py`
13. `tests/integration/test_worker.py`

---

### Milestone 3 — MCP Server + Status + Claude Integration (Days 7–9)
**Goal:** Claude Desktop connects and searches. Developer sees embedding health.

**Acceptance criteria:**
- [ ] `pgsemantic serve` starts without errors
- [ ] Claude Desktop connects via stdio MCP transport
- [ ] Claude can call `semantic_search` and receive ranked results
- [ ] `hybrid_search` with `{"category": "electronics"}` filter returns correctly filtered results
- [ ] `pgsemantic status` shows correct coverage % and queue depth
- [ ] `pgsemantic integrate claude` writes a valid claude_desktop_config.json snippet

**Build order:**
1. `pgsemantic/mcp_server/server.py` (FastMCP, all three tools)
2. `pgsemantic/commands/serve.py`
3. `pgsemantic/commands/status.py`
4. `pgsemantic/commands/integrate.py`
5. `tests/integration/test_search.py`

---

### Milestone 4 — Migration + Error Handling + PyPI Release (Days 10–12)
**Goal:** Zero-downtime migration. Clean errors. Published to PyPI. README with GIF.

**Acceptance criteria:**
- [ ] `pgsemantic migrate --from all-MiniLM-L6-v2 --to text-embedding-3-small` completes for products table
- [ ] Old `embedding` column serves searches during migration (no downtime)
- [ ] After `--confirm` swap, `semantic_search` uses new model's embeddings
- [ ] No raw exception tracebacks reach CLI output — all errors are friendly Rich panels
- [ ] `pip install pgsemantic` on a fresh virtualenv works end-to-end
- [ ] README has animated GIF of full flow (inspect → apply → index → serve → Claude searching)

**Build order:**
1. `pgsemantic/commands/migrate.py`
2. Error handling audit — ensure all exceptions in `commands/` display Rich panels
3. `benchmarks/bench_index.py` (measures rows/min, outputs labeled result)
4. `benchmarks/bench_search.py` (measures latency at 1K/10K/100K rows)
5. Run benchmarks, commit labeled results to `benchmarks/results/`
6. Write `README.md` (with GIF, verified benchmark numbers, quickstart)
7. `.github/workflows/ci.yml`
8. `.github/workflows/publish.yml`
9. Bump version to `0.1.0`, tag `v0.1.0`, publish to PyPI

---

## GitHub Actions CI (`/.github/workflows/ci.yml`)

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    services:
      postgres:
        image: ankane/pgvector:pg16
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: password
          POSTGRES_DB: pgvector_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint
        run: ruff check .

      - name: Type check
        run: mypy pgsemantic/ --strict

      - name: Unit tests
        run: pytest tests/unit/ -v

      - name: Integration tests
        env:
          DATABASE_URL: postgresql://postgres:password@localhost:5432/pgvector_test
          EMBEDDING_PROVIDER: local
        run: pytest tests/integration/ -m integration -v
```

---

## README Structure (For Launch)

The README must have these sections in this order:

1. **One-line hook** — "Point it at your existing Postgres database. Get semantic search in 60 seconds."
2. **Animated GIF** — full demo: inspect → apply → index → worker → serve → Claude searching
3. **Why this exists** — 2-3 sentences on pgai being archived + the bootstrap gap
4. **Quickstart** (5 steps, copy-pasteable)
5. **Commands reference** — all 8 commands with examples
6. **Postgres host compatibility table**
7. **Embedding models table** (local vs OpenAI, dimensions, cost, speed)
8. **Architecture diagram** (text-based, from this CLAUDE.md)
9. **Benchmarks table** — labeled with hardware, versions (only after running benchmarks/bench_*.py)
10. **FAQ** — "Does this modify my existing tables?", "What if pgvector is not installed?", "Can I use my own embedding model?"
11. **Contributing** — link to CONTRIBUTING.md
12. **License** — MIT

**Claims we make in README (verified):**
- Works on RDS, Neon, Supabase, Railway, bare metal
- Uses `SELECT FOR UPDATE SKIP LOCKED` for reliable CDC (no LISTEN/NOTIFY)
- Purely additive: only adds `embedding` column and a trigger to your table
- HNSW index with pgvector-default parameters (m=16, ef_construction=64)
- Pre-filtered hybrid search with `hnsw.iterative_scan` (pgvector ≥0.8.0)
- Local embeddings via sentence-transformers: no API key, no server, works offline

**Claims we do NOT make until benchmarked:**
- Any specific latency number
- Any specific rows/minute number
- Any quality comparison vs Pinecone or Weaviate

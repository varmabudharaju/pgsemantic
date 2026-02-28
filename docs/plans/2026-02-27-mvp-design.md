# pgvector-setup MVP Design (Milestones 1–3)

**Date:** 2026-02-27
**Status:** Approved
**Scope:** Full MVP — core infrastructure, CLI commands, MCP server

---

## Summary

Build a working `pgvector-setup` CLI tool that takes an existing PostgreSQL database and
bootstraps production-quality semantic search. The MVP covers: schema inspection, automated
setup (vector column + HNSW index + queue + trigger), bulk embedding, a background worker
for live sync, an MCP server for AI agent integration, and a status dashboard.

---

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Build approach | Vertical slices | See working results fastest; each slice is testable immediately |
| DB client model | Sync (psycopg3 sync API) | CLI commands are sync (Typer). Async only in worker daemon. |
| Embedding providers | Both local + OpenAI | Local for zero-config demo; OpenAI for production. OpenAI provider is ~30 lines. |
| Type checking | Strict mypy from the start | Catches bugs early, avoids painful retrofit |
| Pagination (index cmd) | Keyset (`WHERE id > last_id`) | Stays fast on large tables, unlike OFFSET which degrades |
| MCP tool functions | Sync | FastMCP handles threading internally; tools just do DB queries |

---

## Architecture

```
User
 │
 ├── pgvector-setup inspect   → db/client.py → db/introspect.py → Rich table
 ├── pgvector-setup apply      → db/client.py → db/vectors.py + db/queue.py → ALTER TABLE + trigger
 ├── pgvector-setup index      → embeddings/ → db/vectors.py → bulk UPDATE
 ├── pgvector-setup worker     → worker/daemon.py (async) → db/queue.py → embeddings/ → UPDATE
 ├── pgvector-setup serve      → mcp_server/server.py → db/vectors.py → JSON-RPC
 ├── pgvector-setup status     → db/vectors.py + db/queue.py → Rich table
 └── pgvector-setup integrate  → config.py → print JSON snippet
```

### Module Dependency Graph

```
cli.py
 └── commands/*
      ├── inspect.py  → db/client, db/introspect
      ├── apply.py    → db/client, db/introspect, db/queue, db/vectors, config
      ├── index.py    → db/client, db/vectors, embeddings/, config
      ├── worker.py   → worker/daemon
      ├── serve.py    → mcp_server/server
      ├── status.py   → db/client, db/vectors, db/queue, config
      └── integrate.py → config

worker/daemon.py → db/queue, db/vectors, embeddings/ (async)
mcp_server/server.py → db/client, db/vectors, embeddings/, config
```

---

## Build Slices (Execution Order)

### Slice 0: Scaffold
- pyproject.toml (all deps, entry point)
- docker-compose.yml (Postgres 16 + pgvector)
- .env.example
- .gitignore
- pgvector_setup/__init__.py
- pgvector_setup/exceptions.py (all typed exceptions)
- pgvector_setup/config.py (Settings dataclass, ProjectConfig, constants)

### Slice 1: Inspect (first working command)
- pgvector_setup/db/__init__.py
- pgvector_setup/db/client.py (sync connection, register_vector, pgvector version check)
- pgvector_setup/db/introspect.py (SQL constants, get_user_tables, get_text_columns, score_column)
- pgvector_setup/cli.py (Typer root app)
- pgvector_setup/commands/__init__.py
- pgvector_setup/commands/inspect.py
- **Milestone check:** `pgvector-setup inspect postgresql://...` renders Rich table

### Slice 2: Apply + Index + Embeddings
- pgvector_setup/embeddings/base.py (EmbeddingProvider Protocol, ProviderConfig)
- pgvector_setup/embeddings/local.py (sentence-transformers, batched)
- pgvector_setup/embeddings/openai_provider.py (OpenAI, batched)
- pgvector_setup/embeddings/__init__.py (get_provider factory)
- pgvector_setup/db/queue.py (SQL constants, create/enqueue/claim/complete/fail)
- pgvector_setup/db/vectors.py (SQL constants, bulk_update, search_similar, hybrid_search)
- pgvector_setup/commands/apply.py (10-step setup with confirmation)
- pgvector_setup/commands/index.py (bulk backfill with keyset pagination + Rich progress)
- **Milestone check:** Full `apply` → `index` flow works against Docker Postgres

### Slice 3: Worker Daemon
- pgvector_setup/worker/__init__.py
- pgvector_setup/worker/daemon.py (async poll loop, batch processing, reconnect, graceful shutdown)
- pgvector_setup/commands/worker.py
- **Milestone check:** INSERT/UPDATE/DELETE rows → worker picks up and embeds within 5s

### Slice 4: MCP Server + Status + Integrate
- pgvector_setup/mcp_server/__init__.py
- pgvector_setup/mcp_server/server.py (FastMCP, semantic_search, hybrid_search, get_embedding_status)
- pgvector_setup/commands/serve.py
- pgvector_setup/commands/status.py
- pgvector_setup/commands/integrate.py
- **Milestone check:** Claude Desktop connects and can search

### Slice 5: Tests
- tests/conftest.py (fixtures, docker-compose lifecycle)
- tests/fixtures/products.sql (500 rows)
- tests/unit/test_config.py
- tests/unit/test_embeddings.py
- tests/unit/test_introspect.py
- tests/unit/test_queue.py
- tests/integration/test_apply.py
- tests/integration/test_index.py
- tests/integration/test_worker.py
- tests/integration/test_search.py

---

## Improvements Over Original Spec

1. **Sync DB client** — CLI commands don't need async. Async only in worker daemon.
   Reduces complexity without sacrificing capability.

2. **Keyset pagination for bulk index** — `WHERE id > last_id` instead of `OFFSET`.
   Stays fast on tables with millions of rows.

3. **Worker heartbeat log** — Every 60s when idle, logs "Worker alive, queue empty."
   Operators can confirm the worker is running without checking process tables.

4. **`--json` flag on inspect** — Pipe-friendly output for CI/automation.

---

## Error Handling Strategy

- **CLI commands:** Catch typed exceptions → display Rich panels with actionable messages
- **Library code (db/, embeddings/):** Catch third-party exceptions → re-raise as typed PgvectorSetupError subclasses
- **MCP tools:** Catch exceptions → return structured MCP error responses
- **Worker daemon:** Catch connection errors → exponential backoff. Catch embedding errors → mark job failed in queue.
- **Never:** Let raw psycopg.Error, openai.APIError, or stack traces reach the user

---

## Testing Strategy

- **Unit tests:** Mock DB + mock embedding models. Deterministic vectors. Fast (< 10s).
- **Integration tests:** Real Postgres via docker-compose. @pytest.mark.integration.
- **Coverage target:** ≥60% for MVP
- **Strict mypy** on all code from day one

---

## What's NOT in the MVP

- `migrate` command (Milestone 4)
- Benchmark suite (Milestone 4)
- README with GIF (Milestone 4)
- GitHub Actions CI/CD (Milestone 4)
- Composite primary key support (v1.1)
- JSONB key extraction (v1.1)
- Multi-column concatenated embeddings (v1.1)
- Document chunking for large text (v1.1)
- OAuth for HTTP transport (v1.0 stable)

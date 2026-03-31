"""FastAPI application for the pgsemantic web UI.

Exposes REST endpoints that wrap existing pgsemantic functions.
Database URL is kept server-side only (from .env / settings).
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import re
import secrets
import signal
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from pgsemantic.config import (
    CONFIG_FILE_NAME,
    DEFAULT_INDEX_BATCH_SIZE,
    DEFAULT_LOCAL_DIMENSIONS,
    DEFAULT_LOCAL_MODEL,
    OLLAMA_DIMENSIONS,
    OLLAMA_MODEL,
    OPENAI_DIMENSIONS,
    OPENAI_MODEL,
    load_project_config,
    load_settings,
    save_project_config,
)
from pgsemantic.db.client import get_connection, get_pgvector_version
from pgsemantic.db.introspect import (
    get_row_count_estimate,
    get_user_tables,
    inspect_database,
    score_column,
    score_to_stars,
)
from pgsemantic.db.queue import count_failed, count_pending
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
    search_chunked,
    search_similar,
)
from pgsemantic.embeddings import get_provider

logger = logging.getLogger(__name__)

# Rate limiting state
_rate_limit: dict[str, list[float]] = {}
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 120  # requests per window (increased for live search)

STATIC_DIR = Path(__file__).parent / "static"

# CSRF token — generated once per server process
_csrf_token = secrets.token_hex(32)

# Worker process tracking
_worker_process: multiprocessing.Process | None = None


def _check_rate_limit(client_ip: str) -> None:
    """Simple in-memory rate limiter per IP."""
    now = time.monotonic()
    if client_ip not in _rate_limit:
        _rate_limit[client_ip] = []
    _rate_limit[client_ip] = [t for t in _rate_limit[client_ip] if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit[client_ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    _rate_limit[client_ip].append(now)


def _get_db_url() -> str:
    """Get database URL from settings. Never expose to client."""
    settings = load_settings()
    if not settings.database_url:
        raise HTTPException(
            status_code=400,
            detail="DATABASE_URL not configured. Set it in your .env file or use the Connection Setup page.",
        )
    return settings.database_url


def _mask_db_url(url: str) -> str:
    """Mask password in database URL for display."""
    if "@" not in url:
        return url
    prefix, rest = url.split("@", 1)
    if ":" in prefix:
        scheme_user = prefix.rsplit(":", 1)[0]
        return f"{scheme_user}:****@{rest}"
    return url


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("pgsemantic UI started")
    yield
    # Stop worker if running
    global _worker_process
    if _worker_process and _worker_process.is_alive():
        _worker_process.terminate()
        _worker_process.join(timeout=5)
    logger.info("pgsemantic UI stopped")


app = FastAPI(
    title="pgsemantic",
    description="Semantic search management UI",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-CSRF-Token"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- Middleware ---


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        client_ip = request.client.host if request.client else "unknown"
        _check_rate_limit(client_ip)
    return await call_next(request)


@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    if request.method == "POST" and request.url.path.startswith("/api/"):
        token = request.headers.get("X-CSRF-Token", "")
        if not secrets.compare_digest(token, _csrf_token):
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid or missing CSRF token"},
            )
    return await call_next(request)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'"
    )
    return response


# --- Pydantic Models ---

_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_identifier(v: str) -> str:
    if not _IDENT_RE.match(v):
        raise ValueError(f"Invalid identifier: {v!r}")
    if len(v) > 128:
        raise ValueError("Identifier too long (max 128 chars)")
    return v


class ApplyRequest(BaseModel):
    table: str = Field(..., min_length=1, max_length=128)
    column: str = Field("", max_length=128)
    columns: str = Field("", max_length=512)
    model: str = Field("local", pattern=r"^(local|local-mpnet|openai|openai-large|ollama)$")
    external: bool = False
    schema_name: str = Field("public", max_length=128)

    @field_validator("table", "schema_name")
    @classmethod
    def validate_ident(cls, v: str) -> str:
        return _validate_identifier(v)

    @field_validator("column")
    @classmethod
    def validate_column(cls, v: str) -> str:
        return _validate_identifier(v) if v else v

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, v: str) -> str:
        if v:
            for part in v.split(","):
                _validate_identifier(part.strip())
        return v


class IndexRequest(BaseModel):
    table: str = Field(..., min_length=1, max_length=128)
    batch_size: int = Field(DEFAULT_INDEX_BATCH_SIZE, ge=1, le=1000)

    @field_validator("table")
    @classmethod
    def validate_ident(cls, v: str) -> str:
        return _validate_identifier(v)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    table: str = Field(..., min_length=1, max_length=128)
    limit: int = Field(5, ge=1, le=100)
    filters: dict[str, str] = Field(default_factory=dict)

    @field_validator("table")
    @classmethod
    def validate_ident(cls, v: str) -> str:
        return _validate_identifier(v)


class SearchAllRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(5, ge=1, le=100)


class ConnectionTestRequest(BaseModel):
    database_url: str = Field(..., min_length=10, max_length=1000)


class SaveApiKeyRequest(BaseModel):
    key_name: str = Field(..., pattern=r"^(OPENAI_API_KEY|OLLAMA_BASE_URL)$")
    key_value: str = Field(..., min_length=1, max_length=500)


class TeardownRequest(BaseModel):
    table: str = Field(..., min_length=1, max_length=128)
    schema_name: str = Field("public", max_length=128)

    @field_validator("table", "schema_name")
    @classmethod
    def validate_ident(cls, v: str) -> str:
        return _validate_identifier(v)


class BulkApplyItem(BaseModel):
    table: str = Field(..., min_length=1, max_length=128)
    column: str = Field(..., min_length=1, max_length=128)
    schema_name: str = Field("public", max_length=128)

    @field_validator("table", "column", "schema_name")
    @classmethod
    def validate_ident(cls, v: str) -> str:
        return _validate_identifier(v)


class BulkApplyRequest(BaseModel):
    items: list[BulkApplyItem] = Field(..., min_length=1, max_length=20)
    model: str = Field("local", pattern=r"^(local|local-mpnet|openai|openai-large|ollama)$")
    external: bool = False


class RetryRequest(BaseModel):
    table: str = Field("", max_length=128)
    all_tables: bool = False

    @field_validator("table")
    @classmethod
    def validate_ident(cls, v: str) -> str:
        return _validate_identifier(v) if v else v


# --- Routes ---


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/csrf-token")
async def get_csrf_token():
    return {"token": _csrf_token}


@app.get("/api/connection")
async def connection_info():
    """Return masked connection info and pgvector version."""
    try:
        db_url = _get_db_url()
        masked = _mask_db_url(db_url)
        with get_connection(db_url, register_vector_type=False) as conn:
            pgv_version = get_pgvector_version(conn)
        version_str = ".".join(str(v) for v in pgv_version) if pgv_version else None
        return {
            "connected": True,
            "database_url": masked,
            "pgvector_version": version_str,
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"connected": False, "error": str(e)}


@app.post("/api/connection/test")
async def test_connection(req: ConnectionTestRequest):
    """Test a database URL without saving it."""
    import psycopg

    try:
        with psycopg.connect(req.database_url, connect_timeout=5) as conn:
            result = conn.execute("SELECT version()").fetchone()
            db_version = result[0] if result else "Unknown"

            # Check pgvector
            pgv = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            pgvector_version = pgv[0] if pgv else None

        return {
            "success": True,
            "database_version": db_version,
            "pgvector_version": pgvector_version,
            "masked_url": _mask_db_url(req.database_url),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/connection/save")
async def save_connection(req: ConnectionTestRequest):
    """Save database URL to .env file."""
    env_path = Path.cwd() / ".env"
    try:
        # Read existing .env content
        existing = {}
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    existing[key.strip()] = val.strip()

        existing["DATABASE_URL"] = req.database_url

        # Write back
        lines = [f"{k}={v}" for k, v in existing.items()]
        env_path.write_text("\n".join(lines) + "\n")
        env_path.chmod(0o600)

        # Update running process so new URL takes effect immediately
        os.environ["DATABASE_URL"] = req.database_url

        return {"success": True, "message": "DATABASE_URL saved to .env"}
    except Exception as e:
        logger.exception("save_connection failed")
        raise HTTPException(500, "Failed to save connection. Check server logs.")


@app.get("/api/connection/api-keys")
async def get_api_keys():
    """Return which API keys are configured. Never returns actual key values."""
    settings = load_settings()
    return {
        "openai_configured": bool(settings.openai_api_key),
        "ollama_url": settings.ollama_base_url,
    }


@app.post("/api/connection/save-api-key")
async def save_api_key(req: SaveApiKeyRequest):
    """Save an API key to .env file. Only whitelisted key names accepted."""
    env_path = Path.cwd() / ".env"
    try:
        existing: dict[str, str] = {}
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    existing[key.strip()] = val.strip()

        existing[req.key_name] = req.key_value
        lines = [f"{k}={v}" for k, v in existing.items()]
        env_path.write_text("\n".join(lines) + "\n")
        env_path.chmod(0o600)
        os.environ[req.key_name] = req.key_value

        return {"success": True, "message": f"{req.key_name} saved to .env"}
    except Exception as e:
        logger.exception("save_api_key failed")
        raise HTTPException(500, "Failed to save API key. Check server logs.")


@app.get("/api/inspect")
async def inspect_db(table: str | None = None):
    """Scan database and score columns for semantic search suitability.

    Optional ?table=name filters results to a single table server-side.
    """
    if table is not None:
        _validate_identifier(table)
    db_url = _get_db_url()
    try:
        with get_connection(db_url, register_vector_type=False) as conn:
            pgv_version = get_pgvector_version(conn)
            candidates = inspect_database(conn)

        if table is not None:
            candidates = [c for c in candidates if c.table_name == table]

        return {
            "pgvector_version": ".".join(str(v) for v in pgv_version) if pgv_version else None,
            "candidates": [
                {
                    "table": c.table_name,
                    "schema": c.table_schema,
                    "column": c.column_name,
                    "data_type": c.data_type,
                    "avg_length": round(c.avg_length, 1),
                    "sampled_rows": c.sampled_rows,
                    "score": score_column(c),
                    "stars": score_to_stars(score_column(c)),
                }
                for c in candidates
            ],
        }
    except Exception as e:
        logger.exception("inspect_db failed")
        raise HTTPException(status_code=500, detail="Inspection failed. Check server logs.")


@app.get("/api/database/tables")
async def list_all_database_tables():
    """List all tables in the database with columns, types, and row counts."""
    db_url = _get_db_url()
    try:
        with get_connection(db_url, register_vector_type=False) as conn:
            tables = get_user_tables(conn)
            result = []
            for t in tables:
                tname = t["table_name"]
                tschema = t["table_schema"]
                # Skip internal pgsemantic tables and Supabase system tables
                _skip_prefixes = (
                    "pgvector_setup_", "pgsemantic_",
                    "auth_", "storage_", "realtime_",
                    "supabase_", "_realtime", "schema_migrations",
                    "extensions", "buckets", "objects", "s3_multipart",
                    "hooks", "mfa_", "sso_", "saml_", "flow_state",
                    "refresh_tokens", "audit_log", "instances",
                    "sessions", "users", "identities", "one_time_tokens",
                )
                _skip_schemas = ("auth", "storage", "realtime", "supabase_functions", "extensions", "vault", "pgsodium")
                if tschema in _skip_schemas:
                    continue
                if any(tname.startswith(p) or tname == p.rstrip("_") for p in _skip_prefixes):
                    continue

                # Get columns
                cols_result = conn.execute(
                    "SELECT column_name, data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_name = %(tname)s AND table_schema = %(tschema)s "
                    "ORDER BY ordinal_position",
                    {"tname": tname, "tschema": tschema},
                ).fetchall()

                columns = [
                    {
                        "name": str(r["column_name"]),
                        "type": str(r["data_type"]),
                        "nullable": str(r["is_nullable"]) == "YES",
                    }
                    for r in cols_result
                ]

                row_count = get_row_count_estimate(conn, tname)

                result.append({
                    "name": tname,
                    "schema": tschema,
                    "columns": columns,
                    "row_count": row_count,
                })

        return {"tables": result}
    except Exception as e:
        logger.exception("list_all_database_tables failed")
        raise HTTPException(500, "Failed to list tables. Check server logs.")


@app.get("/api/database/tables/{table_name}/sample")
async def sample_table_rows(table_name: str, schema: str = "public", limit: int = 5):
    """Return sample rows from a table (max 10 rows, truncated values)."""
    _validate_identifier(table_name)
    _validate_identifier(schema)
    if limit > 10:
        limit = 10

    db_url = _get_db_url()
    try:
        with get_connection(db_url, register_vector_type=False) as conn:
            qualified = _qualified_table(table_name, schema)
            rows = conn.execute(f"SELECT * FROM {qualified} LIMIT %(limit)s", {"limit": limit}).fetchall()

            # Truncate long values and convert to safe JSON
            cleaned = []
            for row in rows:
                item = {}
                for key, val in dict(row).items():
                    if key == "embedding":
                        item[key] = "[vector]"
                    else:
                        s = str(val) if val is not None else None
                        if s and len(s) > 200:
                            s = s[:200] + "..."
                        item[key] = s
                cleaned.append(item)

        return {"table": table_name, "rows": cleaned}
    except Exception as e:
        logger.exception("sample_table_rows failed for %s", table_name)
        raise HTTPException(500, "Failed to fetch sample rows. Check server logs.")


@app.post("/api/apply")
async def apply_setup(req: ApplyRequest):
    """Set up semantic search on a table."""
    from datetime import datetime, timezone

    import psycopg

    from pgsemantic.config import (
        HNSW_EF_CONSTRUCTION,
        HNSW_M,
        SHADOW_TABLE_PREFIX,
        ProjectConfig,
        TableConfig,
    )
    from pgsemantic.db.client import ensure_pgvector_extension
    from pgsemantic.db.introspect import column_exists, get_primary_key_columns
    from pgsemantic.db.queue import create_queue_table
    from pgsemantic.db.vectors import create_shadow_table
    from pgsemantic.commands.apply import (
        SQL_ADD_VECTOR_COLUMN,
        SQL_CREATE_HNSW_INDEX,
        SQL_CREATE_SHADOW_HNSW_INDEX,
        SQL_CREATE_TRIGGER,
        SQL_CREATE_TRIGGER_MULTI,
        _build_trigger_function_sql,
    )

    db_url = _get_db_url()

    if req.column and req.columns:
        raise HTTPException(400, "Cannot use both column and columns")
    if not req.column and not req.columns:
        raise HTTPException(400, "Either column or columns is required")

    source_columns = (
        [c.strip() for c in req.columns.split(",") if c.strip()]
        if req.columns
        else [req.column]
    )
    primary_column = source_columns[0]
    is_multi = len(source_columns) > 1
    storage_mode = "external" if req.external else "inline"
    shadow_table_name = f"{SHADOW_TABLE_PREFIX}{req.table}" if req.external else None

    if req.model == "openai":
        dimensions, model_name = OPENAI_DIMENSIONS, OPENAI_MODEL
    elif req.model == "ollama":
        dimensions, model_name = OLLAMA_DIMENSIONS, OLLAMA_MODEL
    else:
        dimensions, model_name = DEFAULT_LOCAL_DIMENSIONS, DEFAULT_LOCAL_MODEL

    # Build SQL preview for confirmation
    if req.external:
        sql_preview = (
            f'CREATE TABLE IF NOT EXISTS "{req.schema_name}"."{SHADOW_TABLE_PREFIX}{req.table}" (\n'
            f"    row_id TEXT PRIMARY KEY,\n"
            f"    embedding vector({dimensions}),\n"
            f"    source_column TEXT NOT NULL,\n"
            f"    model_name TEXT NOT NULL,\n"
            f"    updated_at TIMESTAMPTZ DEFAULT NOW()\n"
            f");"
        )
    else:
        sql_preview = (
            f'ALTER TABLE "{req.schema_name}"."{req.table}" '
            f'ADD COLUMN "embedding" vector({dimensions});'
        )

    steps: list[dict] = []

    try:
        with get_connection(db_url, register_vector_type=False) as conn:
            pgv_version = ensure_pgvector_extension(conn)
            steps.append({"step": "Check pgvector extension", "status": "ok", "version": ".".join(str(v) for v in pgv_version)})

            for src_col in source_columns:
                if not column_exists(conn, req.table, src_col, req.schema_name):
                    raise HTTPException(400, f"Column '{src_col}' not found on '{req.schema_name}.{req.table}'")

            pk_columns = get_primary_key_columns(conn, req.table, req.schema_name)
            if not pk_columns:
                raise HTTPException(400, f"Table '{req.schema_name}.{req.table}' has no primary key")
            steps.append({"step": "Verify columns & primary key", "status": "ok", "pk": pk_columns})

            if req.external:
                create_shadow_table(conn, shadow_table=shadow_table_name, dimensions=dimensions, schema=req.schema_name)
                steps.append({"step": "Create shadow table", "status": "created"})
            else:
                if column_exists(conn, req.table, "embedding", req.schema_name):
                    steps.append({"step": "Check embedding column", "status": "already_exists"})
                else:
                    sql = SQL_ADD_VECTOR_COLUMN.format(schema=req.schema_name, table=req.table, dimensions=dimensions)
                    conn.execute(sql)
                    conn.commit()
                    steps.append({"step": "Add embedding column", "status": "created"})

        with psycopg.connect(db_url, autocommit=True) as autocommit_conn:
            if req.external:
                index_sql = SQL_CREATE_SHADOW_HNSW_INDEX.format(
                    schema=req.schema_name, shadow_table=shadow_table_name,
                    m=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION,
                )
            else:
                index_sql = SQL_CREATE_HNSW_INDEX.format(
                    schema=req.schema_name, table=req.table,
                    m=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION,
                )
            autocommit_conn.execute(index_sql)
        steps.append({"step": "Create HNSW index", "status": "ok"})

        with get_connection(db_url) as conn:
            create_queue_table(conn)
            steps.append({"step": "Create queue table", "status": "ok"})

            trigger_fn_sql = _build_trigger_function_sql(req.table, pk_columns)
            conn.execute(trigger_fn_sql)
            conn.commit()
            steps.append({"step": "Install trigger function", "status": "ok"})

            if is_multi:
                column_list = ", ".join(f'"{c}"' for c in source_columns)
                column_arg = "+".join(source_columns)
                trigger_sql = SQL_CREATE_TRIGGER_MULTI.format(
                    schema=req.schema_name, table=req.table,
                    column_list=column_list, column_arg=column_arg,
                )
            else:
                trigger_sql = SQL_CREATE_TRIGGER.format(
                    schema=req.schema_name, table=req.table, column=primary_column,
                )
            conn.execute(trigger_sql)
            conn.commit()
            steps.append({"step": "Install trigger", "status": "ok"})

        from pgsemantic import __version__

        config = load_project_config()
        if config is None:
            from pgsemantic.config import ProjectConfig
            config = ProjectConfig(version=__version__)
        config.tables = [tc for tc in config.tables if tc.table != req.table]
        from pgsemantic.config import TableConfig
        from datetime import datetime, timezone
        table_config = TableConfig(
            table=req.table, schema=req.schema_name, column=primary_column,
            embedding_column="embedding", model=req.model, model_name=model_name,
            dimensions=dimensions, hnsw_m=HNSW_M, hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
            applied_at=datetime.now(tz=timezone.utc).isoformat(),
            primary_key=pk_columns, columns=source_columns if is_multi else None,
            storage_mode=storage_mode, shadow_table=shadow_table_name,
        )
        config.tables.append(table_config)
        save_project_config(config)
        steps.append({"step": "Save configuration", "status": "ok"})

        return {
            "success": True, "table": req.table, "columns": source_columns,
            "model": model_name, "storage_mode": storage_mode,
            "sql_preview": sql_preview, "steps": steps,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("apply_setup failed for %s", req.table)
        raise HTTPException(status_code=500, detail="Apply failed. Check server logs.")


@app.post("/api/apply/bulk")
async def bulk_apply(req: BulkApplyRequest):
    """Apply semantic search to multiple table/column combos."""
    results = []
    for item in req.items:
        try:
            single = ApplyRequest(
                table=item.table, column=item.column, columns="",
                model=req.model, external=req.external,
                schema_name=item.schema_name,
            )
            result = await apply_setup(single)
            results.append({"table": item.table, "column": item.column, "success": True})
        except HTTPException as e:
            results.append({"table": item.table, "column": item.column, "success": False, "error": e.detail})
        except Exception as e:
            results.append({"table": item.table, "column": item.column, "success": False, "error": str(e)})
    return {"results": results}


@app.post("/api/index")
def index_table(req: IndexRequest):
    """Bulk embed existing rows."""
    db_url = _get_db_url()
    settings = load_settings()

    config = load_project_config()
    if config is None:
        raise HTTPException(400, "No .pgsemantic.json found. Run apply first.")

    table_config = config.get_table_config(req.table)
    if table_config is None:
        raise HTTPException(400, f"Table '{req.table}' not found in config.")

    try:
        api_key = settings.openai_api_key if table_config.model == "openai" else None
        provider = get_provider(table_config.model, api_key=api_key)

        with get_connection(db_url) as conn:
            total = count_total_with_content(
                conn, req.table, table_config.column, table_config.schema,
                source_columns=table_config.source_columns,
            )
            already = count_embedded(
                conn, req.table, table_config.schema,
                storage_mode=table_config.storage_mode,
                shadow_table=table_config.shadow_table,
            )
            remaining = total - already

            if remaining <= 0:
                return {
                    "success": True, "rows_embedded": 0, "total": total,
                    "final_embedded": already,
                    "coverage": min(round(already / total * 100, 1), 100.0) if total > 0 else 100.0,
                    "message": "All rows already embedded",
                }

            rows_embedded = 0
            last_pk_params = None

            while True:
                batch = fetch_unembedded_batch(
                    conn, table=req.table, column=table_config.column,
                    batch_size=req.batch_size, pk_columns=table_config.primary_key,
                    last_pk_params=last_pk_params, schema=table_config.schema,
                    storage_mode=table_config.storage_mode,
                    shadow_table=table_config.shadow_table,
                    source_columns=table_config.source_columns,
                )
                if not batch:
                    break

                source_columns = table_config.source_columns
                if table_config.storage_mode == "external":
                    texts = [str(row["content"]) for row in batch]
                elif len(source_columns) > 1:
                    texts = [_content_text(source_columns, row) for row in batch]
                else:
                    texts = [str(row[table_config.column]) for row in batch]

                embeddings = provider.embed(texts)
                bulk_update_embeddings(
                    conn, table=req.table, rows=batch, embeddings=embeddings,
                    pk_columns=table_config.primary_key, schema=table_config.schema,
                    storage_mode=table_config.storage_mode,
                    shadow_table=table_config.shadow_table,
                    source_column="+".join(source_columns),
                    model_name=table_config.model_name,
                )
                rows_embedded += len(batch)
                last_pk_params = _pk_last_params(table_config.primary_key, batch[-1])

            final_embedded = count_embedded(
                conn, req.table, table_config.schema,
                storage_mode=table_config.storage_mode,
                shadow_table=table_config.shadow_table,
            )

        return {
            "success": True, "rows_embedded": rows_embedded, "total": total,
            "final_embedded": final_embedded,
            "coverage": min(round(final_embedded / total * 100, 1), 100.0) if total > 0 else 100.0,
        }

    except Exception as e:
        logger.exception("index_table failed for %s", req.table)
        raise HTTPException(status_code=500, detail="Indexing failed. Check server logs.")


@app.post("/api/reindex")
def reindex_table(req: IndexRequest):
    """Re-embed a table by nulling existing embeddings then re-indexing."""
    db_url = _get_db_url()

    config = load_project_config()
    if config is None:
        raise HTTPException(400, "No .pgsemantic.json found.")

    table_config = config.get_table_config(req.table)
    if table_config is None:
        raise HTTPException(400, f"Table '{req.table}' not in config.")

    try:
        with get_connection(db_url) as conn:
            qualified = _qualified_table(req.table, table_config.schema)
            if table_config.storage_mode == "external" and table_config.shadow_table:
                shadow_qualified = _qualified_table(table_config.shadow_table, table_config.schema)
                conn.execute(f"DELETE FROM {shadow_qualified}")
            else:
                conn.execute(f'UPDATE {qualified} SET "embedding" = NULL')
            conn.commit()

        # Now run normal index
        return index_table(req)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("reindex_table failed for %s", req.table)
        raise HTTPException(500, "Reindex failed. Check server logs.")


@app.post("/api/teardown")
async def teardown_table(req: TeardownRequest):
    """Remove semantic search from a table — drop embedding column, trigger, queue entries."""
    db_url = _get_db_url()

    config = load_project_config()
    if config is None:
        raise HTTPException(400, "No .pgsemantic.json found.")

    table_config = config.get_table_config(req.table)
    if table_config is None:
        raise HTTPException(400, f"Table '{req.table}' not in config.")

    removed = []
    try:
        with get_connection(db_url) as conn:
            # Drop trigger (use config schema, not req.schema_name, to match what was created)
            trigger_name = f"pgvector_setup_{table_config.table}_trigger"
            fn_name = f"pgvector_setup_{table_config.table}_fn"
            schema = table_config.schema or req.schema_name
            try:
                conn.execute(f'DROP TRIGGER IF EXISTS "{trigger_name}" ON "{schema}"."{table_config.table}"')
                conn.commit()
                removed.append("trigger")
            except Exception:
                conn.rollback()

            # Drop trigger function
            try:
                conn.execute(f'DROP FUNCTION IF EXISTS "{fn_name}"()')
                conn.commit()
                removed.append("trigger_function")
            except Exception:
                conn.rollback()

            # Drop embedding column or shadow table
            if table_config.storage_mode == "external" and table_config.shadow_table:
                try:
                    shadow_qualified = _qualified_table(table_config.shadow_table, req.schema_name)
                    conn.execute(f"DROP TABLE IF EXISTS {shadow_qualified}")
                    conn.commit()
                    removed.append("shadow_table")
                except Exception:
                    conn.rollback()

                # Drop HNSW index on shadow table
                try:
                    conn.execute(f'DROP INDEX IF EXISTS "idx_{table_config.shadow_table}_embedding_hnsw"')
                    conn.commit()
                    removed.append("hnsw_index")
                except Exception:
                    conn.rollback()
            else:
                try:
                    conn.execute(f'ALTER TABLE "{schema}"."{table_config.table}" DROP COLUMN IF EXISTS "embedding"')
                    conn.commit()
                    removed.append("embedding_column")
                except Exception:
                    conn.rollback()

                # Drop HNSW index
                try:
                    conn.execute(f'DROP INDEX IF EXISTS "idx_{table_config.table}_embedding_hnsw"')
                    conn.commit()
                    removed.append("hnsw_index")
                except Exception:
                    conn.rollback()

            # Clean queue entries
            try:
                conn.execute(
                    "DELETE FROM pgvector_setup_queue WHERE table_name = %(table)s",
                    {"table": req.table},
                )
                conn.commit()
                removed.append("queue_entries")
            except Exception:
                conn.rollback()

        # Remove from config
        config.tables = [tc for tc in config.tables if tc.table != req.table]
        save_project_config(config)
        removed.append("config_entry")

        return {"success": True, "table": req.table, "removed": removed}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("teardown_table failed for %s", req.table)
        raise HTTPException(500, "Teardown failed. Check server logs.")


@app.post("/api/search")
async def search_table(req: SearchRequest):
    """Semantic search on a table."""
    db_url = _get_db_url()
    settings = load_settings()

    config = load_project_config()
    if config is None:
        raise HTTPException(400, "No .pgsemantic.json found.")

    table_config = config.get_table_config(req.table)
    if table_config is None:
        raise HTTPException(400, f"Table '{req.table}' not in config.")

    try:
        api_key = settings.openai_api_key if table_config.model == "openai" else None
        provider = get_provider(table_config.model, api_key=api_key)
        query_vector = provider.embed_query(req.query)

        with get_connection(db_url) as conn:
            if table_config.chunked:
                results = search_chunked(
                    conn, req.table, table_config.column, query_vector,
                    shadow_table=table_config.shadow_table,
                    schema=table_config.schema,
                    pk_columns=table_config.primary_key,
                    limit=req.limit,
                )
            elif req.filters:
                pgv = get_pgvector_version(conn)
                results = hybrid_search(
                    conn, req.table, table_config.column, query_vector,
                    filters=req.filters,
                    limit=req.limit, schema=table_config.schema,
                    pgvector_version=pgv,
                    storage_mode=table_config.storage_mode,
                    shadow_table=table_config.shadow_table,
                    source_columns=table_config.source_columns,
                    pk_columns=table_config.primary_key,
                )
            else:
                results = search_similar(
                    conn, req.table, table_config.column, query_vector,
                    limit=req.limit, schema=table_config.schema,
                    storage_mode=table_config.storage_mode,
                    shadow_table=table_config.shadow_table,
                    source_columns=table_config.source_columns,
                    pk_columns=table_config.primary_key,
                )

        hidden = {"embedding", "similarity"}
        cleaned = []
        for row in results:
            item = {}
            item["similarity"] = round(float(str(row["similarity"])), 4)
            item["content"] = str(row.get(table_config.column, row.get("content", "")))[:500]
            for key, val in row.items():
                if key not in hidden and key != table_config.column:
                    item[key] = _safe_json_value(val)
            cleaned.append(item)

        return {"query": req.query, "table": req.table, "results": cleaned}

    except Exception as e:
        logger.exception("search_table failed for %s", req.table)
        raise HTTPException(status_code=500, detail="Search failed. Check server logs.")


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

    except Exception:
        logger.exception("search_all_tables failed")
        raise HTTPException(status_code=500, detail="Search failed. Check server logs.")


@app.get("/api/status")
async def status_dashboard():
    """Embedding health dashboard for all watched tables."""
    db_url = _get_db_url()

    config = load_project_config()
    if config is None or len(config.tables) == 0:
        return {"tables": [], "message": "No tables configured yet."}

    tables = []
    try:
        with get_connection(db_url) as conn:
            for tc in config.tables:
                try:
                    embedded = count_embedded(conn, tc.table, schema=tc.schema, storage_mode=tc.storage_mode, shadow_table=tc.shadow_table)
                    total = count_total_with_content(conn, tc.table, tc.column, schema=tc.schema, source_columns=tc.source_columns)
                    pending = count_pending(conn, tc.table)
                    failed = count_failed(conn, tc.table)
                    pct = min(round(embedded / total * 100, 1), 100.0) if total > 0 else 0.0

                    tables.append({
                        "table": tc.table, "columns": tc.source_columns,
                        "model": tc.model_name, "storage_mode": tc.storage_mode,
                        "embedded": embedded, "total": total, "coverage_pct": pct,
                        "pending": pending, "failed": failed, "applied_at": tc.applied_at,
                    })
                except Exception as table_err:
                    tables.append({
                        "table": tc.table, "columns": tc.source_columns,
                        "model": tc.model_name, "storage_mode": tc.storage_mode,
                        "error": str(table_err) or "Table missing or inaccessible",
                    })
    except Exception as e:
        logger.exception("status_dashboard failed")
        raise HTTPException(status_code=500, detail="Status check failed. Check server logs.")

    return {"tables": tables}


@app.post("/api/retry")
async def retry_failed(req: RetryRequest):
    """Retry failed embedding jobs."""
    from pgsemantic.db.queue import retry_failed_jobs

    if not req.table and not req.all_tables:
        raise HTTPException(400, "Specify table or set all_tables=true")

    db_url = _get_db_url()
    try:
        with get_connection(db_url) as conn:
            table_name = req.table if req.table else None
            count = retry_failed_jobs(conn, table_name=table_name)
        return {"retried": count, "table": req.table or "all"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("retry_failed failed")
        raise HTTPException(500, "Retry failed. Check server logs.")


@app.get("/api/tables")
async def list_configured_tables():
    """List tables configured with pgsemantic apply."""
    config = load_project_config()
    if config is None:
        return {"tables": []}
    return {
        "tables": [
            {"name": tc.table, "columns": tc.source_columns, "model": tc.model_name}
            for tc in config.tables
        ]
    }


@app.get("/api/config")
async def get_config():
    """Return current .pgsemantic.json contents."""
    config = load_project_config()
    if config is None:
        return {"exists": False, "config": None}
    from dataclasses import asdict
    return {"exists": True, "config": asdict(config)}


@app.get("/api/worker/status")
async def worker_status():
    """Check if the worker process is running."""
    global _worker_process
    if _worker_process and _worker_process.is_alive():
        return {"running": True, "pid": _worker_process.pid}
    return {"running": False, "pid": None}


@app.get("/api/worker-health")
async def worker_health():
    """Get worker health from the database health table."""
    from pgsemantic.db.queue import get_worker_health
    from datetime import datetime, timezone

    db_url = _get_db_url()
    try:
        with get_connection(db_url) as conn:
            workers = get_worker_health(conn)

        now = datetime.now(tz=timezone.utc)
        result = []
        for w in workers:
            last_hb = w["last_heartbeat"]
            if hasattr(last_hb, "timestamp"):
                age_s = (now - last_hb).total_seconds()
            else:
                age_s = 999

            if age_s < 30:
                status = "running"
            elif age_s < 300:
                status = "stale"
            else:
                status = "stopped"

            result.append({
                "worker_id": str(w["worker_id"]),
                "status": status,
                "last_heartbeat": str(w["last_heartbeat"]),
                "jobs_processed": w["jobs_processed"],
                "started_at": str(w.get("started_at")),
                "age_seconds": round(age_s, 1),
            })

        return {"workers": result}
    except Exception:
        return {"workers": []}


def _run_worker_process():
    """Target function for the worker subprocess."""
    from pgsemantic.config import load_settings
    from pgsemantic.worker.daemon import run_worker
    settings = load_settings()
    run_worker(settings)


@app.post("/api/worker/start")
async def start_worker():
    """Start the background worker process."""
    global _worker_process
    if _worker_process and _worker_process.is_alive():
        return {"success": False, "message": "Worker is already running", "pid": _worker_process.pid}

    settings = load_settings()
    if not settings.database_url:
        raise HTTPException(400, "DATABASE_URL not configured.")

    config = load_project_config()
    if config is None or not config.tables:
        raise HTTPException(400, "No tables configured. Run apply first.")

    _worker_process = multiprocessing.Process(target=_run_worker_process, daemon=True)
    _worker_process.start()

    return {"success": True, "message": "Worker started", "pid": _worker_process.pid}


@app.post("/api/worker/stop")
async def stop_worker():
    """Stop the background worker process."""
    global _worker_process
    if not _worker_process or not _worker_process.is_alive():
        return {"success": False, "message": "Worker is not running"}

    _worker_process.terminate()
    _worker_process.join(timeout=5)
    if _worker_process.is_alive():
        _worker_process.kill()

    return {"success": True, "message": "Worker stopped"}


@app.get("/api/mcp/status")
async def mcp_status():
    """Check MCP integration status — is Claude Desktop configured?"""
    from pgsemantic.commands.integrate import _get_config_path

    config_path = _get_config_path()
    result: dict[str, object] = {
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "pgsemantic_configured": False,
        "sse_endpoint": None,
    }

    # Check if the embedded MCP SSE server is mounted
    request_host = "localhost:8080"  # default
    result["sse_endpoint"] = f"http://{request_host}/mcp/sse"

    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            servers = cfg.get("mcpServers", {})
            if "pgsemantic" in servers:
                result["pgsemantic_configured"] = True
                result["current_config"] = servers["pgsemantic"]
        except (json.JSONDecodeError, OSError):
            pass

    return result


@app.post("/api/mcp/configure")
async def mcp_configure(request: Request):
    """Configure Claude Desktop MCP integration."""
    import platform
    import shutil
    import sys

    from pgsemantic.commands.integrate import _get_config_path

    body = await request.json()
    mode = body.get("mode", "stdio")  # "stdio" or "sse"

    settings = load_settings()
    database_url = settings.database_url or "<your-database-url>"
    project_dir = str(Path.cwd())

    if mode == "sse":
        # Point Claude Desktop at the embedded SSE endpoint
        host = str(body.get("host", "localhost"))
        port = int(body.get("port", 8080))
        # Validate host: alphanumeric, dots, hyphens only (no injection)
        if not re.match(r"^[a-zA-Z0-9.\-]{1,253}$", host):
            raise HTTPException(400, "Invalid host value")
        if not (1 <= port <= 65535):
            raise HTTPException(400, "Invalid port value")
        server_entry = {
            "url": f"http://{host}:{port}/mcp/sse",
        }
    else:
        # stdio mode — run pgsemantic serve as a subprocess
        pgsemantic_bin = shutil.which("pgsemantic")
        if pgsemantic_bin:
            server_entry = {
                "command": pgsemantic_bin,
                "args": ["serve"],
                "env": {
                    "DATABASE_URL": database_url,
                    "PGSEMANTIC_PROJECT_DIR": project_dir,
                },
            }
        else:
            server_entry = {
                "command": sys.executable,
                "args": ["-m", "pgsemantic", "serve"],
                "env": {
                    "DATABASE_URL": database_url,
                    "PGSEMANTIC_PROJECT_DIR": project_dir,
                },
            }

    config_path = _get_config_path()
    existing_config: dict[str, object] = {}
    if config_path.exists():
        try:
            existing_config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing_config = {}

    if "mcpServers" not in existing_config:
        existing_config["mcpServers"] = {}

    mcp_servers = existing_config["mcpServers"]
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
        existing_config["mcpServers"] = mcp_servers

    already = "pgsemantic" in mcp_servers
    mcp_servers["pgsemantic"] = server_entry

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(existing_config, indent=2) + "\n")

    action = "Updated" if already else "Added"
    return {
        "success": True,
        "message": f"{action} pgsemantic in Claude Desktop config.",
        "config_path": str(config_path),
        "server_entry": server_entry,
        "mode": mode,
    }


@app.get("/api/mcp/config-snippet")
async def mcp_config_snippet(request: Request):
    """Return MCP config snippets for copy-paste into various clients."""
    import shutil
    import sys

    settings = load_settings()
    database_url = settings.database_url or "<your-database-url>"
    project_dir = str(Path.cwd())

    # Determine the host/port from the request
    host = request.headers.get("host", "localhost:8080")

    pgsemantic_bin = shutil.which("pgsemantic") or sys.executable

    stdio_config = {
        "mcpServers": {
            "pgsemantic": {
                "command": pgsemantic_bin if shutil.which("pgsemantic") else sys.executable,
                "args": ["serve"] if shutil.which("pgsemantic") else ["-m", "pgsemantic", "serve"],
                "env": {
                    "DATABASE_URL": database_url,
                    "PGSEMANTIC_PROJECT_DIR": project_dir,
                },
            }
        }
    }

    sse_config = {
        "mcpServers": {
            "pgsemantic": {
                "url": f"http://{host}/mcp/sse",
            }
        }
    }

    return {
        "stdio": json.dumps(stdio_config, indent=2),
        "sse": json.dumps(sse_config, indent=2),
        "sse_endpoint": f"http://{host}/mcp/sse",
        "tools": [
            {"name": "semantic_search", "description": "Find rows semantically similar to a natural-language query"},
            {"name": "hybrid_search", "description": "Semantic search with SQL WHERE filters"},
            {"name": "get_embedding_status", "description": "Report embedding coverage and queue depth for a table"},
            {"name": "list_tables", "description": "List all database tables with columns, types, and row counts"},
            {"name": "get_sample_rows", "description": "Get sample rows from a table to understand its data"},
            {"name": "inspect_columns", "description": "Score text columns for semantic search suitability"},
            {"name": "list_configured_tables", "description": "List tables with semantic search already configured"},
            {"name": "search_all_tables", "description": "Search across all configured tables at once"},
            {"name": "get_schema_context", "description": "Get database schema for SQL generation"},
            {"name": "execute_safe_sql", "description": "Execute read-only SQL queries safely"},
        ],
    }


def _safe_json_value(val: object) -> object:
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (list, tuple)):
        return [_safe_json_value(v) for v in val]
    return str(val)


# Mount MCP SSE server AFTER all API routes are defined.
# Using /mcp-sse to avoid path conflicts with /api/mcp/* endpoints.
try:
    from pgsemantic.mcp_server.server import mcp as _mcp_server
    _mcp_sse_app = _mcp_server.sse_app()
    app.mount("/mcp", _mcp_sse_app)
    logger.info("MCP SSE server mounted at /mcp/sse")
except Exception as _mcp_err:
    logger.warning("Could not mount MCP SSE server: %s", _mcp_err)

"""FastAPI ingestion and health application for project-backend.
Routes:
    GET  /health                  (unauthenticated health check & counters)
    POST /api/v1/ingest/{stream}  (authenticated HTTP ingestion endpoint)
"""
from contextlib import asynccontextmanager
import json
import os
import secrets
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes_query import router as query_router
from backend.config import verify_startup_safety
from backend.ingest.pipeline import get_counters, get_default_writer, process_records

load_dotenv()

MAX_HTTP_PAYLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_HTTP_RECORDS = 5000
ALLOWED_STREAMS = {"readings", "health", "prediction", "twin"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify prefix and safety guards on startup
    verify_startup_safety()
    yield


app = FastAPI(
    title="Project Backend Ingestion & Query API",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for Dashboard
cors_origins_env = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["X-API-Key"],
)

# Mount read/query routes
app.include_router(query_router)

# Register ReadError exception handler returning 503
from backend.influx_reader import ReadError
import logging

logger = logging.getLogger("backend.api")


@app.exception_handler(ReadError)
async def handle_read_error(request: Request, exc: ReadError):
    logger.error("Read API encountered %s", type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "InfluxDB unreachable"},
    )


# Allow injecting a custom/fake writer and reader for unit tests
app.state.writer = None
app.state.reader = None


@app.get("/health")
def health(request: Request):
    """Health check: verifies InfluxDB ping and reports current ingestion counters.
    Does not require an API key.
    """
    writer = getattr(request.app.state, "writer", None)
    if writer is None:
        try:
            writer = get_default_writer()
            influx_ok = writer.ping()
        except Exception:
            influx_ok = False
    else:
        influx_ok = writer.ping()

    return {
        "status": "ok" if influx_ok else "degraded",
        "influx_connected": influx_ok,
        "counters": get_counters(),
    }


@app.post("/api/v1/ingest/{stream}")
async def ingest_stream(
    stream: str,
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Ingests a single record or an array of records for a specific stream.
    Validates API key, payload size, and data contract.
    """
    # 1. Check server-side API_KEY configuration
    configured_api_key = os.environ.get("API_KEY", "").strip()
    if not configured_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingest service unavailable: API key not configured",
        )

    # Read key must NOT open ingest routes
    configured_read_key = os.environ.get("READ_API_KEY", "").strip()
    if (
        configured_read_key
        and x_api_key
        and secrets.compare_digest(x_api_key, configured_read_key)
        and not secrets.compare_digest(configured_api_key, configured_read_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Read API key cannot access ingest routes",
        )

    # 2. Authenticate X-API-Key header with constant-time comparison
    if not x_api_key or not secrets.compare_digest(x_api_key, configured_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing API key",
        )

    # 3. Validate target stream
    if stream not in ALLOWED_STREAMS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown stream: '{stream}'. Allowed: {sorted(ALLOWED_STREAMS)}",
        )

    # 4. Read raw request body to enforce 5 MB limit
    raw_body = await request.body()
    if len(raw_body) > MAX_HTTP_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Payload size {len(raw_body)} bytes exceeds maximum of {MAX_HTTP_PAYLOAD_BYTES} bytes",
        )

    # 5. Parse JSON
    try:
        data = json.loads(raw_body.decode("utf-8"))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed JSON body: {e}",
        )

    # 6. Normalize to list and check record count <= 5000
    if isinstance(data, list):
        if len(data) > MAX_HTTP_RECORDS:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Record count {len(data)} exceeds maximum batch limit of {MAX_HTTP_RECORDS}",
            )
        records = data
    elif isinstance(data, dict):
        records = [data]
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Body must be a JSON object or array, got {type(data).__name__}",
        )

    # 7. Process through ingestion pipeline
    active_writer = getattr(request.app.state, "writer", None)
    result = process_records(
        stream=stream,
        records=records,
        transport="http",
        topic_site=None,
        writer=active_writer,
    )

    return JSONResponse(status_code=status.HTTP_200_OK, content=result)

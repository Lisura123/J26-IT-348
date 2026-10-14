"""Query routes for project-backend dashboard API.
All routes require X-API-Key matching READ_API_KEY.
Strict input validation enforces data contracts and prevents Flux injection.
"""
from datetime import datetime, timedelta, timezone
import logging
import os
import re
import secrets
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Header, HTTPException, Query, Request, status

from backend.config import get_stream_config
from backend.influx_reader import InfluxReader, ReadError
from backend.sites import get_all_sites, valid_site_ids
from backend.validate import parse_utc_timestamp

load_dotenv()

logger = logging.getLogger("backend.api.query")

router = APIRouter(prefix="/api/v1", tags=["query"])

ALLOWED_STREAMS = {"readings", "health", "prediction", "twin"}
FIELD_NAME_REGEX = re.compile(r"^[a-z][a-z0-9_]*$")
ALLOWED_EVERY = {"1m", "5m", "15m", "1h", "1d"}
ALLOWED_FN = {"mean", "min", "max", "last", "count"}
MAX_SERIES_RANGE_DAYS = 31


def get_active_reader(request: Request):
    """Retrieves reader from app state or instantiates default InfluxReader."""
    reader = getattr(request.app.state, "reader", None)
    if reader is None:
        reader = InfluxReader.from_env()
    return reader


def verify_read_auth(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> None:
    """Authenticates read requests against READ_API_KEY.
    Rejects missing or wrong key (401), rejects ingest key (401),
    and raises 503 if READ_API_KEY is not configured.
    """
    configured_read_key = os.environ.get("READ_API_KEY", "").strip()
    if not configured_read_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Read service unavailable: READ_API_KEY not configured",
        )

    # Ingest key must NOT open read routes
    configured_ingest_key = os.environ.get("API_KEY", "").strip()
    if (
        configured_ingest_key
        and x_api_key
        and secrets.compare_digest(x_api_key, configured_ingest_key)
        and not secrets.compare_digest(configured_ingest_key, configured_read_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Ingest API key cannot access read routes",
        )

    if not x_api_key or not secrets.compare_digest(x_api_key, configured_read_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing API key",
        )


@router.get("/sites")
def list_sites(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Returns the list of canonical sites from reference/sites.csv."""
    verify_read_auth(x_api_key)
    sites_map = get_all_sites()
    result = []
    for sid, info in sites_map.items():
        result.append({
            "site_id": sid,
            "device_id": info.get("device_id", ""),
            "panel_id": info.get("panel_id", ""),
            "display_name": info.get("display_name", ""),
            "protocol": info.get("protocol", ""),
            "gateway_id": info.get("gateway_id", ""),
            "link_profile": info.get("link_profile", ""),
        })
    return result


@router.get("/query/latest")
def query_latest(
    request: Request,
    stream: str = Query(..., description="Target stream name"),
    site_id: str = Query(..., description="Target site identifier"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Returns the newest row of that stream for that site:
    {stream, site_id, time (UTC with Z), tags, fields}.
    404 if no rows exist.
    """
    verify_read_auth(x_api_key)

    # Input validation
    if stream not in ALLOWED_STREAMS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid stream: '{stream}'. Allowed: {sorted(ALLOWED_STREAMS)}",
        )

    if site_id not in valid_site_ids():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid site_id: '{site_id}'. Must exist in reference/sites.csv",
        )

    reader = get_active_reader(request)
    try:
        result = reader.get_latest(stream=stream, site_id=site_id)
    except ReadError as exc:
        logger.error("InfluxDB read error in query_latest: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="InfluxDB unreachable",
        )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No data found for stream '{stream}' and site '{site_id}'",
        )
    return result


@router.get("/query/series")
def query_series(
    request: Request,
    stream: str = Query(..., description="Target stream name"),
    site_id: str = Query(..., description="Target site identifier"),
    field: str = Query(..., description="Target field name"),
    start: str = Query(..., description="Start UTC timestamp with Z"),
    stop: str = Query(..., description="Stop UTC timestamp with Z"),
    every: Optional[str] = Query(None, description="Aggregation window (1m, 5m, 15m, 1h, 1d)"),
    fn: Optional[str] = Query(None, description="Aggregation function (mean, min, max, last, count)"),
    limit: int = Query(1000, description="Max points to return (1..10000)"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Returns time series points:
    {stream, site_id, field, points: [{time, value}]}
    """
    verify_read_auth(x_api_key)

    # 1. Validate stream
    if stream not in ALLOWED_STREAMS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid stream: '{stream}'. Allowed: {sorted(ALLOWED_STREAMS)}",
        )

    # 2. Validate site_id
    if site_id not in valid_site_ids():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid site_id: '{site_id}'. Must exist in reference/sites.csv",
        )

    # 3. Validate field name regex
    if not FIELD_NAME_REGEX.match(field):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid field name '{field}': must match pattern '^[a-z][a-z0-9_]*$'",
        )

    # 4. Strict stream declared field validation
    stream_cfg = get_stream_config(stream) or {}
    if stream_cfg.get("mode") == "strict":
        declared_fields = (
            set(stream_cfg.get("float_fields", []))
            | set(stream_cfg.get("int_fields", []))
            | set(stream_cfg.get("string_fields", []))
        )
        if field not in declared_fields:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Field '{field}' is not a declared field for strict stream '{stream}'",
            )

    # 5. Parse and validate start / stop timestamps
    dt_start, err_code_start, reason_start = parse_utc_timestamp(start)
    if dt_start is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid start timestamp '{start}': {reason_start}",
        )

    dt_stop, err_code_stop, reason_stop = parse_utc_timestamp(stop)
    if dt_stop is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid stop timestamp '{stop}': {reason_stop}",
        )

    if dt_start >= dt_stop:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Start timestamp '{start}' must be strictly before stop timestamp '{stop}'",
        )

    if (dt_stop - dt_start) > timedelta(days=MAX_SERIES_RANGE_DAYS):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Time range exceeds maximum allowed range of {MAX_SERIES_RANGE_DAYS} days",
        )

    # 6. Validate aggregation options
    if every is not None or fn is not None:
        if not every or not fn:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Both 'every' and 'fn' must be provided together for aggregation",
            )
        if every not in ALLOWED_EVERY:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid 'every': '{every}'. Allowed: {sorted(ALLOWED_EVERY)}",
            )
        if fn not in ALLOWED_FN:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid 'fn': '{fn}'. Allowed: {sorted(ALLOWED_FN)}",
            )

    # 7. Validate limit
    if limit < 1 or limit > 10000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"limit must be between 1 and 10000, got {limit}",
        )

    # Execute query
    reader = get_active_reader(request)
    try:
        points = reader.get_series(
            stream=stream,
            site_id=site_id,
            field=field,
            start=dt_start,
            stop=dt_stop,
            every=every,
            fn=fn,
            limit=limit,
        )
    except ReadError as exc:
        logger.error("InfluxDB read error in query_series: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="InfluxDB unreachable",
        )

    return {
        "stream": stream,
        "site_id": site_id,
        "field": field,
        "points": points,
    }

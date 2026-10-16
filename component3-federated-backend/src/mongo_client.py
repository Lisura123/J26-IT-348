"""MongoDB Atlas Client & Helper Utilities for Component 3.
Handles connection setup, timeouts, safe bulk upserts, retries, and document mapping.
"""
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pymongo
from pymongo import MongoClient, ReplaceOne
from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    ExecutionTimeout,
    NetworkTimeout,
    PyMongoError,
    ServerSelectionTimeoutError,
)

load_dotenv()

# Unique keys per collection for ReplaceOne upsert
COLLECTION_UNIQUE_KEYS: Dict[str, List[str]] = {
    "network_link_metrics": ["run_id", "site_id", "timestamp_utc"],
    "network_outages": ["run_id", "site_id", "start_utc"],
    "fl_client_updates": ["run_id", "update_id"],
    "fl_global_models": ["run_id", "round_id"],
    "fl_site_state": ["run_id", "round_id", "site_id"],
    "ota_deployments": ["run_id", "site_id", "round_id"],
}

# Primary datetime fields converted to timezone-aware UTC BSON dates
BSON_DATETIME_FIELDS = {
    "timestamp_utc",
    "start_utc",
    "sent_at_utc",
    "published_at_utc",
    "deployed_at_utc",
    "round_opened_utc",
}

_ensured_indexes = set()


def scrub_secrets(text: Any) -> str:
    """Removes passwords, credentials, and tokens from URIs or error messages."""
    if not text:
        return ""
    s = str(text)
    # Scrub mongodb credentials (everything between scheme and @host)
    s = re.sub(r"(mongodb(?:\+srv)?://)[^/]+@([a-zA-Z0-9.-]+)", r"\1***:***@\2", s)
    # Scrub tokens or secret queries
    s = re.sub(r"([?&](?:token|password|auth|apiKey)=)[^&]+", r"\1***", s)
    return s


def get_mongo_client(uri: Optional[str] = None) -> MongoClient:
    """Creates a MongoClient with Atlas-recommended timeouts."""
    conn_uri = uri or os.environ.get("MONGO_URI")
    if not conn_uri:
        raise ValueError("MONGO_URI environment variable is not set.")

    return MongoClient(
        conn_uri,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        socketTimeoutMS=60000,
        retryWrites=True,
    )


def get_mongo_db(client: Optional[MongoClient] = None, db_name: Optional[str] = None):
    """Returns the target database (default from MONGO_DB or 'solar_pv')."""
    c = client if client is not None else get_mongo_client()
    target_db = db_name or os.environ.get("MONGO_DB") or "solar_pv"
    return c[target_db]


def ensure_unique_index(collection, collection_name: str) -> None:
    """Ensures the unique compound index on first use."""
    global _ensured_indexes
    if collection_name in _ensured_indexes:
        return
    keys = COLLECTION_UNIQUE_KEYS.get(collection_name)
    if keys:
        index_spec = [(k, pymongo.ASCENDING) for k in keys]
        collection.create_index(index_spec, unique=True)
    _ensured_indexes.add(collection_name)


def to_utc_datetime(val: Any) -> Optional[datetime]:
    """Converts a value to timezone-aware UTC datetime."""
    if val is None or pd.isna(val) or val == "":
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    try:
        ts = pd.to_datetime(val, utc=True)
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def row_to_document(row: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a row dict to a clean MongoDB document:
    - float stays float, int stays int, str stays str, bool stays bool
    - NaN/None omitted (never stored as 0 or null)
    - datetimes converted to timezone-aware UTC BSON dates
    - numpy scalars converted to standard python primitives
    """
    doc: Dict[str, Any] = {}
    for k, v in row.items():
        if v is None:
            continue
        # Check pd.isna for pandas/numpy NaN, NaT
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass

        # Handle datetime fields
        if k in BSON_DATETIME_FIELDS:
            dt = to_utc_datetime(v)
            if dt is not None:
                doc[k] = dt
            continue

        # Convert numpy types
        if hasattr(v, "item"):
            v = v.item()

        # Handle booleans (must check before int since bool inherits from int)
        if isinstance(v, bool):
            doc[k] = v
        elif isinstance(v, (int, np.integer)):
            doc[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            if np.isnan(v):
                continue
            doc[k] = float(v)
        elif isinstance(v, str):
            doc[k] = v
        else:
            doc[k] = v

    return doc


def _is_transient_error(exc: Exception) -> bool:
    """Identifies transient network or cluster errors suitable for backoff retry."""
    if isinstance(exc, (AutoReconnect, NetworkTimeout, ServerSelectionTimeoutError, ConnectionFailure, ExecutionTimeout)):
        return True
    if hasattr(exc, "has_error_label") and exc.has_error_label("TransientTransactionError"):
        return True
    msg = str(exc).lower()
    transient_indicators = [
        "timed out",
        "timeout",
        "connection reset",
        "socket timeout",
        "network",
        "temporary",
    ]
    return any(ind in msg for ind in transient_indicators)


def write_documents(
    collection,
    collection_name: str,
    docs: List[Dict[str, Any]],
    batch_size: Optional[int] = None,
    pause_s: Optional[float] = None,
    max_retries: int = 3,
) -> int:
    """Writes documents via bulk_write with ReplaceOne(upsert=True), chunked by batch_size,
    with backoff on transient errors and throttle pacing.
    """
    if not docs:
        return 0

    ensure_unique_index(collection, collection_name)
    key_fields = COLLECTION_UNIQUE_KEYS.get(collection_name)
    if not key_fields:
        raise ValueError(f"Unknown collection unique keys for '{collection_name}'")

    write_batch = batch_size or int(os.environ.get("MONGO_WRITE_BATCH", 100))
    write_pause = pause_s if pause_s is not None else float(os.environ.get("MONGO_WRITE_PAUSE_S", 0.0))

    total = len(docs)
    if total >= 500:
        est_sec = total / 100.0
        print(
            f"[{collection_name}] Loading {total} records in batches of {write_batch} "
            f"(estimated ~{est_sec:.1f}s at Atlas ~100 ops/s throttle)..."
        )

    written = 0
    start_time = time.time()

    for i in range(0, total, write_batch):
        chunk = docs[i : i + write_batch]
        ops = []
        for d in chunk:
            filter_doc = {k: d[k] for k in key_fields if k in d}
            ops.append(ReplaceOne(filter_doc, d, upsert=True))

        # Retry loop for chunk
        success = False
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                collection.bulk_write(ops, ordered=False)
                success = True
                break
            except Exception as e:
                last_err = e
                if attempt < max_retries and _is_transient_error(e):
                    backoff = 0.5 * (2 ** attempt)
                    time.sleep(backoff)
                else:
                    break

        if not success:
            clean_err = scrub_secrets(str(last_err))
            print(f"Error: bulk_write failed for {collection_name} chunk {i}-{i+len(chunk)}: {clean_err}", file=sys.stderr)
            raise last_err

        written += len(chunk)
        if write_pause > 0.0:
            time.sleep(write_pause)

        if total >= 500 and (written % 2000 < write_batch or written == total):
            elapsed = max(0.001, time.time() - start_time)
            rate = written / elapsed
            print(f"  [{collection_name}] Progress: {written}/{total} rows written ({rate:.1f} rows/s)...")

    return written

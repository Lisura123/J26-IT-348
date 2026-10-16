"""MongoDB Atlas Writer client for project-backend.
Performs batched, idempotent upserts into MongoDB Atlas collections using ReplaceOne
with ordered=False, ensuring unique indexes and UTC BSON datetimes.
"""
import os
import time
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from pymongo import ReplaceOne
from pymongo.errors import (
    AutoReconnect,
    BulkWriteError,
    ConnectionFailure,
    ExecutionTimeout,
    NetworkTimeout,
    PyMongoError,
    ServerSelectionTimeoutError,
)

from backend.points import PointSpec
from backend.storage import (
    get_database,
    get_mongo_client,
    sanitize_error,
)
from backend.validate import is_missing, parse_utc_timestamp

logger = logging.getLogger("backend.mongo_writer")


def is_transient_error(exc: Exception) -> bool:
    """Returns True if the exception represents a transient timeout or network error."""
    if isinstance(
        exc,
        (
            AutoReconnect,
            ConnectionFailure,
            NetworkTimeout,
            ServerSelectionTimeoutError,
            ExecutionTimeout,
        ),
    ):
        return True
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in (
            "timeout",
            "timed out",
            "connection reset",
            "connection closed",
            "connection refused",
            "network unreachable",
            "broken pipe",
            "temporarily unavailable",
        )
    )


def point_to_document(point: PointSpec) -> Dict[str, Any]:
    """Converts a PointSpec into a MongoDB document adhering to the contract:
    - site_id, device_id, protocol tags as strings
    - timestamp_utc as a BSON UTC datetime
    - schema fields with preserved types (int, float, bool, str)
    - missing/NaN/null fields strictly omitted
    """
    doc: Dict[str, Any] = {
        "site_id": str(point.tags.get("site_id", "")).strip(),
        "device_id": str(point.tags.get("device_id", "")).strip(),
        "protocol": str(point.tags.get("protocol", "")).strip(),
    }

    # Normalize timestamp to UTC datetime
    t = point.time
    if isinstance(t, datetime):
        if t.tzinfo is None:
            t_utc = t.replace(tzinfo=timezone.utc)
        else:
            t_utc = t.astimezone(timezone.utc)
    else:
        dt, _, _ = parse_utc_timestamp(t)
        t_utc = dt if dt is not None else datetime.now(timezone.utc)
    doc["timestamp_utc"] = t_utc

    # Append valid fields, preserving types and omitting missing values
    for k, v in point.fields.items():
        if is_missing(v):
            continue
        if isinstance(v, bool):
            doc[k] = bool(v)
        elif isinstance(v, int):
            doc[k] = int(v)
        elif isinstance(v, float):
            doc[k] = float(v)
        elif isinstance(v, str):
            doc[k] = str(v)
        else:
            doc[k] = v

    return doc


class MongoWriter:
    """Synchronous MongoDB Atlas writer for ingesting streams.
    Uses ReplaceOne upserts for idempotent replay and enforces unique indexes.
    Atlas free tier throttles writes at ~100 operations/sec; chunks writes by
    MONGO_WRITE_BATCH (default 100) with MONGO_WRITE_PAUSE_S (default 0.0)
    and retries transient network/timeout errors up to 3 times with exponential backoff.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        db: Optional[Any] = None,
        write_batch: Optional[int] = None,
        write_pause_s: Optional[float] = None,
    ):
        self._external_client = client is not None
        self.client = client
        self.db = db
        self._initialized_collections: Set[str] = set()
        self.write_batch = write_batch
        self.write_pause_s = write_pause_s

    @classmethod
    def from_env(cls) -> "MongoWriter":
        """Instantiates MongoWriter using application writer role from environment."""
        client = get_mongo_client(role="app")
        db = get_database(client=client, role="app")
        return cls(client=client, db=db)

    def _get_batch_size(self) -> int:
        if self.write_batch is not None:
            return max(1, self.write_batch)
        try:
            return max(1, int(os.environ.get("MONGO_WRITE_BATCH", "100").strip()))
        except ValueError:
            return 100

    def _get_pause_s(self) -> float:
        if self.write_pause_s is not None:
            return max(0.0, self.write_pause_s)
        try:
            return max(0.0, float(os.environ.get("MONGO_WRITE_PAUSE_S", "0.0").strip()))
        except ValueError:
            return 0.0

    def _ensure_connected(self) -> None:
        if self.client is None or self.db is None:
            self.client = get_mongo_client(role="app")
            self.db = get_database(client=self.client, role="app")

    def ensure_indexes(self, collection_name: str) -> None:
        """Creates unique index on (site_id, timestamp_utc) and index on timestamp_utc.
        Safely caches initialized collections to minimize round trips.
        """
        if collection_name in self._initialized_collections:
            return
        self._ensure_connected()
        col = self.db[collection_name]
        try:
            # Justified unique index: preserves every distinct valid record per site and timestamp
            col.create_index(
                [("site_id", 1), ("timestamp_utc", 1)],
                unique=True,
                name="uniq_site_timestamp",
            )
            col.create_index(
                [("timestamp_utc", 1)],
                name="idx_timestamp_utc",
            )
            self._initialized_collections.add(collection_name)
        except Exception as exc:
            sanitized = sanitize_error(exc)
            logger.warning(
                "Could not create indexes on collection '%s': %s",
                collection_name,
                sanitized,
            )

    def ping(self) -> bool:
        """Checks connectivity to MongoDB Atlas. Returns True if ping succeeds, False otherwise."""
        try:
            self._ensure_connected()
            res = self.client.admin.command("ping")
            return bool(res.get("ok", 0) == 1)
        except Exception as exc:
            logger.warning("MongoDB ping failed: %s", sanitize_error(exc))
            return False

    def write(self, points: List[PointSpec]) -> Tuple[int, int, Optional[str]]:
        """Batches and writes points to MongoDB using ReplaceOne upserts with ordered=False.
        Splits operations into chunks of MONGO_WRITE_BATCH (default 100), applies
        MONGO_WRITE_PAUSE_S between chunks, and retries failed chunks up to 3 times
        with backoff on transient errors.
        Returns:
            Tuple[written_count, failed_count, sanitized_error_text_or_None]
        """
        if not points:
            return 0, 0, None

        self._ensure_connected()

        # Group operations by target collection (stream measurement name)
        grouped_docs: Dict[str, List[Dict[str, Any]]] = {}
        for pt in points:
            col_name = pt.measurement
            doc = point_to_document(pt)
            grouped_docs.setdefault(col_name, []).append(doc)

        total_written = 0
        total_failed = 0
        error_messages: List[str] = []

        batch_size = self._get_batch_size()
        pause_s = self._get_pause_s()

        for col_name, docs in grouped_docs.items():
            self.ensure_indexes(col_name)
            col = self.db[col_name]

            # Construct ReplaceOne upserts
            operations = [
                ReplaceOne(
                    filter={
                        "site_id": doc["site_id"],
                        "timestamp_utc": doc["timestamp_utc"],
                    },
                    replacement=doc,
                    upsert=True,
                )
                for doc in docs
            ]

            # Split into chunks of MONGO_WRITE_BATCH
            chunks = [
                operations[i : i + batch_size]
                for i in range(0, len(operations), batch_size)
            ]

            for chunk_idx, chunk in enumerate(chunks):
                if chunk_idx > 0 and pause_s > 0.0:
                    time.sleep(pause_s)

                max_retries = 3
                backoff = 0.5
                attempt = 0
                while True:
                    try:
                        col.bulk_write(chunk, ordered=False)
                        total_written += len(chunk)
                        break
                    except BulkWriteError as bwe:
                        details = bwe.details or {}
                        write_errors = details.get("writeErrors", [])
                        failed_in_batch = len(write_errors)
                        written_in_batch = len(chunk) - failed_in_batch
                        total_written += max(0, written_in_batch)
                        total_failed += failed_in_batch
                        sanitized = sanitize_error(bwe)
                        error_messages.append(sanitized)
                        logger.error("BulkWriteError on %s: %s", col_name, sanitized)
                        break
                    except Exception as exc:
                        sanitized = sanitize_error(exc)
                        if is_transient_error(exc) and attempt < max_retries:
                            attempt += 1
                            logger.warning(
                                "Transient error writing batch to %s (retry %d/%d, backoff %.1fs): %s",
                                col_name,
                                attempt,
                                max_retries,
                                backoff,
                                sanitized,
                            )
                            time.sleep(backoff)
                            backoff = min(backoff * 2.0, 5.0)
                            continue

                        total_failed += len(chunk)
                        error_messages.append(sanitized)
                        logger.error(
                            "Failed writing chunk to %s after %d retries: %s",
                            col_name,
                            attempt,
                            sanitized,
                        )
                        break

        err_text = "; ".join(error_messages) if error_messages else None
        return total_written, total_failed, err_text

    def count_points(self, measurement: str) -> int:
        """Returns the document count in the given measurement collection."""
        try:
            self._ensure_connected()
            if measurement not in self.db.list_collection_names():
                return 0
            return int(self.db[measurement].count_documents({}))
        except Exception:
            return 0

    def close(self) -> None:
        """Closes the MongoDB client connection if managed."""
        if self.client and not self._external_client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
            self.db = None

"""MongoDB Atlas Reader client for project-backend.
Implements secure, read-only queries with strict validation against query injection ($where, $gt, etc.),
enforces allow-lists, field regex ^[a-z][a-z0-9_]*$, maximum 31-day range, 10,000 points limit,
and maps database connection/authentication failures to safe ReadError.
Never exposes credentials in logs or errors.
"""
from datetime import datetime, timedelta, timezone
import logging
import re
from typing import Any, Dict, List, Optional, Set

from pymongo.errors import PyMongoError

from backend.config import get_final_measurement, get_stream_config
from backend.sites import valid_site_ids
from backend.storage import (
    get_database,
    get_mongo_client,
    sanitize_error,
)

logger = logging.getLogger("backend.mongo_reader")

ALLOWED_STREAMS = {"readings", "health", "prediction", "twin"}
FIELD_NAME_REGEX = re.compile(r"^[a-z][a-z0-9_]*$")
ALLOWED_EVERY = {"1m", "5m", "15m", "1h", "1d"}
ALLOWED_FN = {"mean", "min", "max", "last", "count"}
MAX_SERIES_RANGE_DAYS = 31
MAX_SERIES_LIMIT = 10000

# Disallowed injection tokens
FORBIDDEN_OPERATORS = {"$where", "$gt", "$gte", "$lt", "$lte", "$ne", "$in", "$nin", "$regex", "$expr"}


class ReadError(Exception):
    """Raised when reading from MongoDB fails. Message is sanitized of credentials."""
    pass


class MongoReader:
    """Read-only MongoDB reader client. Uses the least-privilege reader account when configured."""

    def __init__(self, client: Optional[Any] = None, db: Optional[Any] = None):
        self._external_client = client is not None
        self.client = client
        self.db = db

    @classmethod
    def from_env(cls) -> "MongoReader":
        """Instantiates MongoReader using read-only reader role from environment."""
        client = get_mongo_client(role="reader")
        db = get_database(client=client, role="reader")
        return cls(client=client, db=db)

    def _ensure_connected(self) -> None:
        if self.client is None or self.db is None:
            self.client = get_mongo_client(role="reader")
            self.db = get_database(client=self.client, role="reader")

    def ping(self) -> bool:
        """Pings MongoDB Atlas. Returns True on success, False on failure."""
        try:
            self._ensure_connected()
            res = self.client.admin.command("ping")
            return bool(res.get("ok", 0) == 1)
        except Exception as exc:
            logger.warning("MongoDB reader ping failed: %s", sanitize_error(exc))
            return False

    def validate_stream(self, stream: str) -> None:
        """Validates stream against allowed list."""
        if stream not in ALLOWED_STREAMS:
            raise ValueError(f"Invalid stream '{stream}'. Allowed: {sorted(ALLOWED_STREAMS)}")

    def validate_site_id(self, site_id: str) -> None:
        """Validates site_id against canonical reference/sites.csv."""
        if site_id not in valid_site_ids():
            raise ValueError(f"Invalid site_id '{site_id}'. Must exist in reference/sites.csv")

    def validate_field_name(self, stream: str, field: str) -> None:
        """Strictly validates field name against regex and injection patterns."""
        if not isinstance(field, str) or not field:
            raise ValueError("Field name must be a non-empty string")

        if not FIELD_NAME_REGEX.match(field):
            raise ValueError(
                f"Invalid field name '{field}': must match pattern '^[a-z][a-z0-9_]*$'"
            )

        if "." in field or "$" in field or any(op in field for op in FORBIDDEN_OPERATORS):
            raise ValueError(f"Prohibited field pattern or operator detected: '{field}'")

        # Strict stream schema field check
        stream_cfg = get_stream_config(stream) or {}
        if stream_cfg.get("mode") == "strict":
            declared_fields = (
                set(stream_cfg.get("float_fields", []))
                | set(stream_cfg.get("int_fields", []))
                | set(stream_cfg.get("string_fields", []))
            )
            if field not in declared_fields:
                raise ValueError(
                    f"Field '{field}' is not a declared field for strict stream '{stream}'"
                )

    def validate_time_range(self, start: datetime, stop: datetime) -> None:
        """Validates that start is strictly before stop and range does not exceed 31 days."""
        if start >= stop:
            raise ValueError(f"Start timestamp ({start}) must be strictly before stop ({stop})")
        if (stop - start) > timedelta(days=MAX_SERIES_RANGE_DAYS):
            raise ValueError(f"Time range exceeds maximum allowed range of {MAX_SERIES_RANGE_DAYS} days")

    def validate_limit(self, limit: int) -> None:
        """Validates limit within 1..10000."""
        if limit < 1 or limit > MAX_SERIES_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_SERIES_LIMIT}, got {limit}")

    def validate_aggregation(self, every: Optional[str], fn: Optional[str]) -> None:
        """Validates every and fn aggregation parameters."""
        if bool(every) != bool(fn):
            raise ValueError("Both 'every' and 'fn' must be provided together for aggregation")
        if every and every not in ALLOWED_EVERY:
            raise ValueError(f"Invalid 'every': '{every}'. Allowed: {sorted(ALLOWED_EVERY)}")
        if fn and fn not in ALLOWED_FN:
            raise ValueError(f"Invalid 'fn': '{fn}'. Allowed: {sorted(ALLOWED_FN)}")

    def get_latest(self, stream: str, site_id: str) -> Optional[Dict[str, Any]]:
        """Returns the newest document for that stream and site.
        Format: {stream, site_id, time, tags, fields} or None if not found.
        """
        self.validate_stream(stream)
        self.validate_site_id(site_id)
        collection_name = get_final_measurement(stream)

        try:
            self._ensure_connected()
            col = self.db[collection_name]
            doc = col.find_one(
                {"site_id": site_id},
                sort=[("timestamp_utc", -1)],
            )
        except Exception as exc:
            sanitized = sanitize_error(exc)
            logger.error("MongoDB read error in get_latest: %s", sanitized)
            raise ReadError(sanitized) from exc

        if not doc:
            return None

        dt = doc.get("timestamp_utc")
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            time_str = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            time_str = str(dt)

        tags = {
            "site_id": doc.get("site_id", site_id),
            "device_id": doc.get("device_id", ""),
            "protocol": doc.get("protocol", ""),
        }
        exclude_keys = {"_id", "site_id", "device_id", "protocol", "timestamp_utc"}
        fields = {k: v for k, v in doc.items() if k not in exclude_keys}

        return {
            "stream": stream,
            "site_id": site_id,
            "time": time_str,
            "tags": tags,
            "fields": fields,
        }

    latest = get_latest

    def get_series(
        self,
        stream: str,
        site_id: str,
        field: str,
        start: datetime,
        stop: datetime,
        every: Optional[str] = None,
        fn: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Returns time series points: [{time, value}]. Supports raw query or bucketing."""
        self.validate_stream(stream)
        self.validate_site_id(site_id)
        self.validate_field_name(stream, field)
        self.validate_time_range(start, stop)
        self.validate_limit(limit)
        if every or fn:
            self.validate_aggregation(every, fn)

        collection_name = get_final_measurement(stream)

        try:
            self._ensure_connected()
            col = self.db[collection_name]

            if not every:
                # Raw points query
                cursor = col.find(
                    {
                        "site_id": site_id,
                        "timestamp_utc": {"$gte": start, "$lt": stop},
                        field: {"$exists": True, "$ne": None},
                    },
                    projection={"timestamp_utc": 1, field: 1, "_id": 0},
                    sort=[("timestamp_utc", 1)],
                    limit=limit,
                )
                points: List[Dict[str, Any]] = []
                for doc in cursor:
                    dt = doc.get("timestamp_utc")
                    if isinstance(dt, datetime):
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        t_str = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    else:
                        t_str = str(dt)
                    points.append({"time": t_str, "value": doc.get(field)})
                return points

            else:
                # Aggregation pipeline with dateTrunc
                interval_map = {
                    "1m": ("minute", 1),
                    "5m": ("minute", 5),
                    "15m": ("minute", 15),
                    "1h": ("hour", 1),
                    "1d": ("day", 1),
                }
                unit, bin_size = interval_map[every]

                # Select accumulator
                if fn == "mean":
                    accum = {"$avg": f"${field}"}
                elif fn == "min":
                    accum = {"$min": f"${field}"}
                elif fn == "max":
                    accum = {"$max": f"${field}"}
                elif fn == "last":
                    accum = {"$last": f"${field}"}
                elif fn == "count":
                    accum = {"$sum": 1}
                else:
                    raise ValueError(f"Unsupported fn: {fn}")

                pipeline = [
                    {
                        "$match": {
                            "site_id": site_id,
                            "timestamp_utc": {"$gte": start, "$lt": stop},
                            field: {"$exists": True, "$ne": None},
                        }
                    },
                    {"$sort": {"timestamp_utc": 1}},
                    {
                        "$group": {
                            "_id": {
                                "$dateTrunc": {
                                    "date": "$timestamp_utc",
                                    "unit": unit,
                                    "binSize": bin_size,
                                    "timezone": "UTC",
                                }
                            },
                            "value": accum,
                        }
                    },
                    {"$sort": {"_id": 1}},
                    {"$limit": limit},
                ]

                cursor = col.aggregate(pipeline)
                points = []
                for doc in cursor:
                    bucket_dt = doc.get("_id")
                    if isinstance(bucket_dt, datetime):
                        if bucket_dt.tzinfo is None:
                            bucket_dt = bucket_dt.replace(tzinfo=timezone.utc)
                        t_str = bucket_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    else:
                        t_str = str(bucket_dt)
                    val = doc.get("value")
                    if isinstance(val, float):
                        val = round(val, 4)
                    points.append({"time": t_str, "value": val})
                return points

        except Exception as exc:
            sanitized = sanitize_error(exc)
            logger.error("MongoDB read error in get_series: %s", sanitized)
            raise ReadError(sanitized) from exc

    series = get_series

    def count_rows(
        self,
        stream: str,
        sites: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        stop: Optional[datetime] = None,
    ) -> int:
        """Counts documents in target stream collection matching site and time boundaries."""
        self.validate_stream(stream)
        collection_name = get_final_measurement(stream)
        query: Dict[str, Any] = {}
        if sites:
            query["site_id"] = {"$in": sites}
        if start is not None and stop is not None:
            query["timestamp_utc"] = {"$gte": start, "$lte": stop}
        elif start is not None:
            query["timestamp_utc"] = {"$gte": start}
        elif stop is not None:
            query["timestamp_utc"] = {"$lte": stop}

        try:
            self._ensure_connected()
            col = self.db[collection_name]
            return int(col.count_documents(query))
        except Exception as exc:
            sanitized = sanitize_error(exc)
            logger.error("MongoDB count error: %s", sanitized)
            raise ReadError(sanitized) from exc

    count_distinct_rows = count_rows

    def get_stored_record_ids(
        self,
        stream: str,
        sites: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        stop: Optional[datetime] = None,
    ) -> Set[str]:
        """Returns the set of distinct record_id values present in the database."""
        self.validate_stream(stream)
        collection_name = get_final_measurement(stream)
        query: Dict[str, Any] = {}
        if sites:
            query["site_id"] = {"$in": sites}
        if start is not None and stop is not None:
            query["timestamp_utc"] = {"$gte": start, "$lte": stop}
        elif start is not None:
            query["timestamp_utc"] = {"$gte": start}
        elif stop is not None:
            query["timestamp_utc"] = {"$lte": stop}

        try:
            self._ensure_connected()
            col = self.db[collection_name]
            ids = col.distinct("record_id", query)
            return set(ids)
        except Exception as exc:
            sanitized = sanitize_error(exc)
            logger.error("MongoDB distinct error: %s", sanitized)
            raise ReadError(sanitized) from exc

    def count_field_values(self, collection_or_measurement: str) -> int:
        """Counts documents in a collection. Used to verify Member 3's reserved measurements are untouched."""
        try:
            self._ensure_connected()
            if collection_or_measurement not in self.db.list_collection_names():
                return 0
            col = self.db[collection_or_measurement]
            return int(col.count_documents({}))
        except Exception as exc:
            sanitized = sanitize_error(exc)
            logger.error("MongoDB count error on %s: %s", collection_or_measurement, sanitized)
            raise ReadError(sanitized) from exc

    def close(self) -> None:
        """Closes the reader client connection if managed."""
        if self.client and not self._external_client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
            self.db = None


class FakeReader:
    """Mock reader for testing Read API, Replay, and Verify tools in-process without network."""

    def __init__(self, should_raise: bool = False):
        self.rows: List[Dict[str, Any]] = []
        self.should_raise = should_raise
        self.queries_executed: List[Any] = []
        self.bucket = "solar_pv"

    def add_row(
        self,
        stream: str,
        site_id: str,
        time_utc: datetime,
        tags: Dict[str, str],
        fields: Dict[str, Any],
    ) -> None:
        self.rows.append({
            "stream": stream,
            "site_id": site_id,
            "time_utc": time_utc,
            "tags": dict(tags),
            "fields": dict(fields),
        })

    def ping(self) -> bool:
        if self.should_raise:
            return False
        return True

    def get_latest(self, stream: str, site_id: str) -> Optional[Dict[str, Any]]:
        self.queries_executed.append({"op": "latest", "stream": stream, "site_id": site_id})
        if self.should_raise:
            raise ReadError("Simulated database failure")
        matching = [r for r in self.rows if r["stream"] == stream and r["site_id"] == site_id]
        if not matching:
            return None
        newest = max(matching, key=lambda r: r["time_utc"])
        dt = newest["time_utc"]
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return {
            "stream": stream,
            "site_id": site_id,
            "time": dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tags": newest["tags"],
            "fields": newest["fields"],
        }

    latest = get_latest

    def get_series(
        self,
        stream: str,
        site_id: str,
        field: str,
        start: datetime,
        stop: datetime,
        every: Optional[str] = None,
        fn: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        self.queries_executed.append({
            "op": "series",
            "stream": stream,
            "site_id": site_id,
            "field": field,
            "start": start,
            "stop": stop,
            "every": every,
            "fn": fn,
            "limit": limit,
        })
        if self.should_raise:
            raise ReadError("Simulated database failure")
        matching = [
            r for r in self.rows
            if r["stream"] == stream
            and r["site_id"] == site_id
            and start <= r["time_utc"] < stop
            and field in r["fields"]
        ]
        matching.sort(key=lambda r: r["time_utc"])

        if not every or not fn:
            matching = matching[:limit]
            return [
                {
                    "time": r["time_utc"].astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "value": r["fields"][field],
                }
                for r in matching
            ]

        # In-memory bucket aggregation for FakeReader
        interval_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "1d": 1440,
        }.get(every, 60)

        buckets: Dict[datetime, List[float]] = {}
        for r in matching:
            t = r["time_utc"].astimezone(timezone.utc)
            if interval_minutes == 1440:
                b_dt = datetime(t.year, t.month, t.day, tzinfo=timezone.utc)
            elif interval_minutes == 60:
                b_dt = datetime(t.year, t.month, t.day, t.hour, tzinfo=timezone.utc)
            else:
                minute_bucket = (t.minute // interval_minutes) * interval_minutes
                b_dt = datetime(t.year, t.month, t.day, t.hour, minute_bucket, tzinfo=timezone.utc)
            buckets.setdefault(b_dt, []).append(float(r["fields"][field]))

        points = []
        for b_dt in sorted(buckets.keys()):
            vals = buckets[b_dt]
            if fn == "mean":
                v = sum(vals) / len(vals)
            elif fn == "min":
                v = min(vals)
            elif fn == "max":
                v = max(vals)
            elif fn == "last":
                v = vals[-1]
            elif fn == "count":
                v = len(vals)
            else:
                v = vals[-1]
            points.append({
                "time": b_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "value": round(v, 4) if isinstance(v, float) else v,
            })
        return points[:limit]

    series = get_series

    def count_distinct_rows(
        self,
        stream: str,
        sites: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        stop: Optional[datetime] = None,
    ) -> int:
        if self.should_raise:
            raise ReadError("Simulated database failure")
        matching = [r for r in self.rows if r["stream"] == stream]
        if sites:
            matching = [r for r in matching if r["site_id"] in sites]
        if start:
            matching = [r for r in matching if r["time_utc"] >= start]
        if stop:
            matching = [r for r in matching if r["time_utc"] <= stop]
        keys = {(r["site_id"], r["time_utc"]) for r in matching}
        return len(keys)

    count_rows = count_distinct_rows

    def get_stored_record_ids(
        self,
        stream: str,
        sites: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        stop: Optional[datetime] = None,
    ) -> Set[str]:
        if self.should_raise:
            raise ReadError("Simulated database failure")
        matching = [r for r in self.rows if r["stream"] == stream]
        if sites:
            matching = [r for r in matching if r["site_id"] in sites]
        if start:
            matching = [r for r in matching if r["time_utc"] >= start]
        if stop:
            matching = [r for r in matching if r["time_utc"] <= stop]
        return {r["fields"].get("record_id") for r in matching if "record_id" in r["fields"]}

    def count_field_values(self, collection_or_measurement: str) -> int:
        if self.should_raise:
            raise ReadError("Simulated database failure")
        return len([r for r in self.rows if r["stream"] == collection_or_measurement])

    def close(self) -> None:
        pass

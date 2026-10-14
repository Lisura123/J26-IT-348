"""Read-only InfluxDB reader client and FakeReader for project-backend.
Strictly read-only; never writes, deletes, or modifies anything.
No secrets are logged or exposed.
"""
from datetime import datetime, timedelta, timezone
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib3.util

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient
from influxdb_client.rest import ApiException

from backend.config import get_final_measurement, get_stream_config

load_dotenv()


class ReadError(Exception):
    """Raised when an InfluxDB read operation fails (connection, timeout, auth, HTTP).
    Guarantees no secrets are exposed in the error message.
    """
    pass


def sanitize_read_error(e: Exception, secrets: Optional[List[Optional[str]]] = None) -> str:
    """Builds a short, informative reason for a read failure with all secrets redacted."""
    all_secrets = [
        os.environ.get("INFLUX_TOKEN"),
        os.environ.get("API_KEY"),
        os.environ.get("READ_API_KEY"),
    ]
    if secrets:
        all_secrets.extend(secrets)

    # 1. Handle ApiException from InfluxDB client
    if isinstance(e, ApiException):
        status = e.status
        reason = (e.reason or "").strip()
        if status == 401:
            return "Authentication failed (HTTP 401 Unauthorized)"
        if status == 403:
            return "Forbidden (HTTP 403)"
        if status == 404:
            return f"Not found (HTTP 404): {reason or 'Resource not found'}"
        if status:
            return f"HTTP error {status}: {reason or 'ApiException'}"
        return f"InfluxDB API error: {reason or 'Unknown'}"

    err_str = str(e)

    # 2. Check for timeout
    if "timeout" in type(e).__name__.lower() or "timed out" in err_str.lower():
        return f"Request timed out ({type(e).__name__})"

    # 3. Check for connection refused / connection error
    if (
        "connection" in type(e).__name__.lower()
        or "connection" in err_str.lower()
        or "10061" in err_str
        or "111" in err_str
        or "refused" in err_str.lower()
    ):
        if "refused" in err_str.lower() or "10061" in err_str or "111" in err_str:
            return "Connection refused: InfluxDB is unreachable"
        return f"Connection error: InfluxDB is unreachable ({type(e).__name__})"

    # 4. Redact secrets from generic message
    msg = f"{type(e).__name__}: {err_str}"
    for s in all_secrets:
        if s and len(s) >= 4 and s in msg:
            msg = msg.replace(s, "***")
    # Also strip any URL tokens like token=...
    msg = re.sub(r'token=[^&\s]+', 'token=***', msg)
    return msg


class InfluxReader:
    """Read-only InfluxDB query client."""

    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
    ) -> None:
        self.url = url
        self._token = token
        self.org = org
        self.bucket = bucket

        retries = urllib3.util.Retry(
            total=3,
            connect=3,
            read=False,
            redirect=False,
            status=3,
            backoff_factor=0.5,
        )
        self.client = InfluxDBClient(
            url=self.url,
            token=self._token,
            org=self.org,
            timeout=(10000, 30000),  # connect 10s, read 30s
            retries=retries,
        )
        self.query_api = self.client.query_api()

    @classmethod
    def from_env(cls) -> "InfluxReader":
        """Instantiates InfluxReader from environment variables."""
        url = os.environ.get("INFLUX_URL", "http://localhost:8087")
        token = os.environ.get("INFLUX_TOKEN", "")
        org = os.environ.get("INFLUX_ORG", "j26")
        bucket = os.environ.get("INFLUX_BUCKET", "solar_pv")
        return cls(url=url, token=token, org=org, bucket=bucket)

    def ping(self) -> bool:
        """Pings InfluxDB. Returns True if alive."""
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def query(self, flux: str) -> Any:
        """Executes a Flux query. Raises ReadError on any connection, timeout,
        authentication, or HTTP error with secrets removed.
        """
        try:
            return self.query_api.query(flux, org=self.org)
        except Exception as e:
            reason = sanitize_read_error(e, [self._token])
            raise ReadError(reason) from None

    def get_latest(self, stream: str, site_id: str) -> Optional[Dict[str, Any]]:
        """Returns the newest row of that stream for that site:
        {stream, site_id, time (UTC with Z), tags, fields}
        All fields of the latest timestamp are pivoted into one object.
        Returns None if no rows exist.
        Raises ReadError on read failure.
        """
        meas = get_final_measurement(stream)
        stream_cfg = get_stream_config(stream) or {}
        tag_keys = set(stream_cfg.get("tags", ["site_id", "device_id", "protocol"]))

        flux = f'''
        from(bucket: "{self.bucket}")
          |> range(start: 1970-01-01T00:00:00Z)
          |> filter(fn: (r) => r._measurement == "{meas}" and r.site_id == "{site_id}")
          |> last()
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> group()
          |> sort(columns: ["_time"], desc: true)
          |> limit(n: 1)
        '''
        tables = self.query(flux)
        if not tables or not tables[0].records:
            return None

        rec = tables[0].records[0]
        t = rec.get_time()
        if t is None:
            return None
        time_str = t.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Extract tags and fields
        meta_cols = {"_time", "_start", "_stop", "_measurement", "table", "result"}
        tags: Dict[str, Any] = {}
        for k in tag_keys:
            if k in rec.values and rec.values[k] is not None:
                tags[k] = rec.values[k]

        fields: Dict[str, Any] = {}
        for k, v in rec.values.items():
            if k not in meta_cols and k not in tag_keys and v is not None:
                fields[k] = v

        return {
            "stream": stream,
            "site_id": site_id,
            "time": time_str,
            "tags": tags,
            "fields": fields,
        }

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
        """Queries time series points for a given stream, site, and field.
        Raises ReadError on read failure.
        """
        meas = get_final_measurement(stream)
        start_iso = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        stop_iso = stop.strftime("%Y-%m-%dT%H:%M:%SZ")

        if every and fn:
            flux = f'''
            from(bucket: "{self.bucket}")
              |> range(start: {start_iso}, stop: {stop_iso})
              |> filter(fn: (r) => r._measurement == "{meas}" and r.site_id == "{site_id}" and r._field == "{field}")
              |> aggregateWindow(every: {every}, fn: {fn}, createEmpty: false)
              |> limit(n: {limit})
            '''
        else:
            flux = f'''
            from(bucket: "{self.bucket}")
              |> range(start: {start_iso}, stop: {stop_iso})
              |> filter(fn: (r) => r._measurement == "{meas}" and r.site_id == "{site_id}" and r._field == "{field}")
              |> limit(n: {limit})
            '''

        tables = self.query(flux)
        points: List[Dict[str, Any]] = []
        for t in tables:
            for rec in t.records:
                rec_time = rec.get_time()
                time_str = rec_time.strftime("%Y-%m-%dT%H:%M:%SZ") if rec_time else ""
                points.append({
                    "time": time_str,
                    "value": rec.get_value(),
                })
        return points

    def count_distinct_rows(
        self,
        stream: str,
        sites: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        stop: Optional[datetime] = None,
    ) -> int:
        """Counts distinct (site_id, _time) rows in InfluxDB.
        Raises ReadError on read failure.
        """
        meas = get_final_measurement(stream)
        start_iso = start.strftime("%Y-%m-%dT%H:%M:%SZ") if start else "1970-01-01T00:00:00Z"
        # Make stop range inclusive of the final millisecond if provided
        if stop:
            stop_dt = stop + timedelta(seconds=1)
            stop_iso = stop_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            stop_iso = "2030-01-01T00:00:00Z"

        site_filter = ""
        if sites:
            conds = [f'r.site_id == "{s}"' for s in sites]
            site_filter = " and (" + " or ".join(conds) + ")"

        flux = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start_iso}, stop: {stop_iso})
          |> filter(fn: (r) => r._measurement == "{meas}"{site_filter})
          |> group(columns: ["site_id", "_time"])
          |> limit(n: 1)
          |> map(fn: (r) => ({{_time: r._time, site_id: r.site_id, _value: 1}}))
          |> group()
          |> count()
        '''
        tables = self.query(flux)
        if tables and tables[0].records:
            val = tables[0].records[0].get_value()
            return int(val) if val is not None else 0
        return 0

    def get_stored_record_ids(
        self,
        stream: str,
        sites: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        stop: Optional[datetime] = None,
    ) -> Set[str]:
        """Returns set of record_id values stored for strict streams.
        Raises ReadError on read failure.
        """
        meas = get_final_measurement(stream)
        start_iso = start.strftime("%Y-%m-%dT%H:%M:%SZ") if start else "1970-01-01T00:00:00Z"
        if stop:
            stop_dt = stop + timedelta(seconds=1)
            stop_iso = stop_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            stop_iso = "2030-01-01T00:00:00Z"

        site_filter = ""
        if sites:
            conds = [f'r.site_id == "{s}"' for s in sites]
            site_filter = " and (" + " or ".join(conds) + ")"

        flux = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start_iso}, stop: {stop_iso})
          |> filter(fn: (r) => r._measurement == "{meas}" and r._field == "record_id"{site_filter})
        '''
        tables = self.query(flux)
        ids: Set[str] = set()
        for t in tables:
            for r in t.records:
                v = r.get_value()
                if v is not None:
                    ids.add(str(v))
        return ids

    def count_field_values(
        self,
        measurement: str,
        start: str = "2026-09-01T00:00:00Z",
        stop: str = "2026-09-20T00:00:00Z",
    ) -> int:
        """Count all stored field values of every row grouped.
        Raises ReadError on read failure.
        """
        flux = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start}, stop: {stop})
          |> filter(fn: (r) => r._measurement == "{measurement}")
          |> group()
          |> count()
        '''
        tables = self.query(flux)
        if tables and tables[0].records:
            val = tables[0].records[0].get_value()
            return int(val) if val is not None else 0
        return 0

    def close(self) -> None:
        """Closes InfluxDB client."""
        try:
            self.client.close()
        except Exception:
            pass


class FakeReader:
    """In-memory mock reader for unit testing without InfluxDB or Docker."""

    def __init__(
        self,
        should_raise: bool = False,
        error_to_raise: Optional[Exception] = None,
    ) -> None:
        # Key: (stream, site_id) -> List of rows: dict with time (datetime), tags (dict), fields (dict)
        self.rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self.queries_executed: List[str] = []
        self.should_raise = should_raise
        self.error_to_raise = error_to_raise
        self.reserved_counts: Dict[str, int] = {
            "network_link_metrics": 74170,
            "network_outages": 82,
            "fl_client_updates": 4058,
            "fl_global_models": 503,
            "fl_site_state": 600,
            "ota_deployments": 1083,
        }

    def _check_raise(self) -> None:
        """Raises ReadError or configured error if should_raise is set."""
        if self.should_raise:
            if self.error_to_raise is not None:
                raise self.error_to_raise
            raise ReadError("Connection refused: InfluxDB is unreachable")

    def add_row(
        self,
        stream: str,
        site_id: str,
        time_utc: datetime,
        tags: Dict[str, Any],
        fields: Dict[str, Any],
    ) -> None:
        """Adds a test row to the fake storage."""
        key = (stream, site_id)
        if key not in self.rows:
            self.rows[key] = []
        self.rows[key].append({
            "time": time_utc,
            "tags": dict(tags),
            "fields": dict(fields),
        })

    def ping(self) -> bool:
        if self.should_raise:
            return False
        return True

    def query(self, flux: str) -> Any:
        self._check_raise()
        return []

    def get_latest(self, stream: str, site_id: str) -> Optional[Dict[str, Any]]:
        self._check_raise()
        self.queries_executed.append(f"get_latest({stream}, {site_id})")
        key = (stream, site_id)
        rows = self.rows.get(key, [])
        if not rows:
            return None

        # Pick the row with maximum time
        latest_row = max(rows, key=lambda r: r["time"])
        time_str = latest_row["time"].strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "stream": stream,
            "site_id": site_id,
            "time": time_str,
            "tags": latest_row["tags"],
            "fields": latest_row["fields"],
        }

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
        self._check_raise()
        self.queries_executed.append(
            f"get_series({stream}, {site_id}, {field}, {start}, {stop}, every={every}, fn={fn}, limit={limit})"
        )
        key = (stream, site_id)
        rows = self.rows.get(key, [])

        # Filter by start <= time < stop and field present
        filtered: List[Tuple[datetime, Any]] = []
        for r in rows:
            t = r["time"]
            if start <= t < stop and field in r["fields"]:
                filtered.append((t, r["fields"][field]))

        filtered.sort(key=lambda item: item[0])

        if not every or not fn:
            # Raw points limited to limit
            res = [
                {"time": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "value": val}
                for t, val in filtered[:limit]
            ]
            return res

        # Aggregation mock (e.g. mean, min, max, last, count)
        if not filtered:
            return []

        vals = [v for _, v in filtered if isinstance(v, (int, float))]
        if not vals:
            agg_val = filtered[-1][1]
        elif fn == "mean":
            agg_val = sum(vals) / len(vals)
        elif fn == "min":
            agg_val = min(vals)
        elif fn == "max":
            agg_val = max(vals)
        elif fn == "last":
            agg_val = vals[-1]
        elif fn == "count":
            agg_val = len(vals)
        else:
            agg_val = vals[0]

        return [{"time": filtered[0][0].strftime("%Y-%m-%dT%H:%M:%SZ"), "value": agg_val}]

    def count_distinct_rows(
        self,
        stream: str,
        sites: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        stop: Optional[datetime] = None,
    ) -> int:
        self._check_raise()
        self.queries_executed.append(f"count_distinct_rows({stream}, sites={sites})")
        distinct_pairs: Set[Tuple[str, datetime]] = set()
        for (s, sid), rows in self.rows.items():
            if s != stream:
                continue
            if sites and sid not in sites:
                continue
            for r in rows:
                t = r["time"]
                if start and t < start:
                    continue
                if stop and t > stop:
                    continue
                distinct_pairs.add((sid, t))
        return len(distinct_pairs)

    def get_stored_record_ids(
        self,
        stream: str,
        sites: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        stop: Optional[datetime] = None,
    ) -> Set[str]:
        self._check_raise()
        self.queries_executed.append(f"get_stored_record_ids({stream})")
        rids: Set[str] = set()
        for (s, sid), rows in self.rows.items():
            if s != stream:
                continue
            if sites and sid not in sites:
                continue
            for r in rows:
                t = r["time"]
                if start and t < start:
                    continue
                if stop and t > stop:
                    continue
                rid = r["fields"].get("record_id")
                if rid:
                    rids.add(str(rid))
        return rids

    def count_field_values(
        self,
        measurement: str,
        start: str = "2026-09-01T00:00:00Z",
        stop: str = "2026-09-20T00:00:00Z",
    ) -> int:
        self._check_raise()
        return self.reserved_counts.get(measurement, 0)

    def close(self) -> None:
        pass

"""InfluxDB 2.7 writer client for project-backend.
Connects via synchronous API with WritePrecision.MS.
Preserves exact field types, omits missing values, handles batch writes,
and enforces safety guards without logging or exposing secrets.
"""
from datetime import datetime
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.rest import ApiException
from influxdb_client.client.write_api import SYNCHRONOUS

from backend.config import verify_startup_safety
from backend.points import PointSpec

load_dotenv()


def _sanitize_error(error_str: str, secrets: List[Optional[str]]) -> str:
    """Removes any sensitive tokens or keys from error messages."""
    res = error_str
    for s in secrets:
        if s and len(s) >= 4 and s in res:
            res = res.replace(s, "***")
    return res


class InfluxWriter:
    """Synchronous InfluxDB writer with type preservation and safety guards."""

    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
    ) -> None:
        # Enforce startup safety guard: prefix cannot be empty unless ALLOW_REAL_MEASUREMENTS == '1'
        verify_startup_safety()

        self.url = url
        self._token = token
        self.org = org
        self.bucket = bucket

        self.client = InfluxDBClient(
            url=self.url,
            token=self._token,
            org=self.org,
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()

    @classmethod
    def from_env(cls) -> "InfluxWriter":
        """Instantiates InfluxWriter from environment variables."""
        url = os.environ.get("INFLUX_URL", "http://localhost:8087")
        token = os.environ.get("INFLUX_TOKEN", "")
        org = os.environ.get("INFLUX_ORG", "j26")
        bucket = os.environ.get("INFLUX_BUCKET", "solar_pv")
        return cls(url=url, token=token, org=org, bucket=bucket)

    def ping(self) -> bool:
        """Pings the InfluxDB server. Returns True if alive."""
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def write(self, points: List[PointSpec]) -> Tuple[int, int, Optional[str]]:
        """Batched write of PointSpec objects to InfluxDB.
        Returns: (written_count, failed_count, error_text)
        Preserves field types: float stays float, int stays int, bool stays bool.
        Omitted/missing values are never filled with zero.
        """
        if not points:
            return 0, 0, None

        influx_points: List[Point] = []
        for pt in points:
            p = Point(pt.measurement).time(pt.time, WritePrecision.MS)
            # Tags are strictly strings
            for k, v in pt.tags.items():
                p.tag(k, str(v))

            # Fields keep their types exactly
            for k, v in pt.fields.items():
                if v is None:
                    continue
                if isinstance(v, bool):
                    p.field(k, v)
                elif isinstance(v, float):
                    p.field(k, float(v))
                elif isinstance(v, int):
                    p.field(k, int(v))
                else:
                    p.field(k, str(v))
            influx_points.append(p)

        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=influx_points)
            return len(influx_points), 0, None
        except ApiException as e:
            raw_err = f"InfluxDB ApiException: status={e.status}, reason={e.reason}"
            safe_err = _sanitize_error(raw_err, [self._token])
            return 0, len(influx_points), safe_err
        except Exception as e:
            raw_err = f"InfluxDB write error ({type(e).__name__}): {e}"
            safe_err = _sanitize_error(raw_err, [self._token])
            return 0, len(influx_points), safe_err

    def count_points(
        self,
        measurement: str,
        start: str = "2026-09-01T00:00:00Z",
        stop: str = "2026-09-20T00:00:00Z",
    ) -> int:
        """Read-only Flux count over all fields, grouped."""
        flux = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start}, stop: {stop})
          |> filter(fn: (r) => r._measurement == "{measurement}")
          |> group()
          |> count()
        '''
        try:
            tables = self.query_api.query(flux, org=self.org)
            if tables and tables[0].records:
                val = tables[0].records[0].get_value()
                return int(val) if val is not None else 0
            return 0
        except Exception:
            return 0

    def close(self) -> None:
        """Closes the InfluxDB client connection."""
        try:
            self.client.close()
        except Exception:
            pass

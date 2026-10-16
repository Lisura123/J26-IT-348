"""Unit and functional tests for Stage 2: InfluxWriter, Ingest Pipeline, MQTT Service, and HTTP API.
Pure in-process tests; no external network connection required.
"""
from datetime import datetime, timezone
import json
import os
import secrets
from typing import Any, Dict, List, Optional, Tuple
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.config import get_final_measurement, load_schema, verify_startup_safety
from backend.mongo_writer import MongoWriter, point_to_document
from backend.ingest.mqtt_service import handle_message
from backend.ingest.pipeline import process_records, reset_counters
from backend.ingest.rejects import REJECTS_FILE, log_reject
from backend.points import PointSpec


class FakeWriter:
    """Mock writer to test pipeline and HTTP without connecting to InfluxDB."""

    def __init__(self, should_fail: bool = False):
        self.points: List[PointSpec] = []
        self.should_fail = should_fail

    def ping(self) -> bool:
        return True

    def write(self, points: List[PointSpec]) -> Tuple[int, int, Optional[str]]:
        if self.should_fail:
            return 0, len(points), "Simulated write error"
        self.points.extend(points)
        return len(points), 0, None

    def count_points(self, measurement: str, start: str = "", stop: str = "") -> int:
        return len([p for p in self.points if p.measurement == measurement])

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def setup_env():
    """Sets up safe testing environment with test_ prefix and API key."""
    old_prefix = os.environ.get("MEASUREMENT_PREFIX")
    old_allow = os.environ.get("ALLOW_REAL_MEASUREMENTS")
    old_key = os.environ.get("API_KEY")

    os.environ["MEASUREMENT_PREFIX"] = "test_"
    os.environ["ALLOW_REAL_MEASUREMENTS"] = "0"
    os.environ["API_KEY"] = "test_secret_api_key_12345"
    load_schema()
    reset_counters()

    yield

    if old_prefix is not None:
        os.environ["MEASUREMENT_PREFIX"] = old_prefix
    elif "MEASUREMENT_PREFIX" in os.environ:
        del os.environ["MEASUREMENT_PREFIX"]

    if old_allow is not None:
        os.environ["ALLOW_REAL_MEASUREMENTS"] = old_allow
    elif "ALLOW_REAL_MEASUREMENTS" in os.environ:
        del os.environ["ALLOW_REAL_MEASUREMENTS"]

    if old_key is not None:
        os.environ["API_KEY"] = old_key
    elif "API_KEY" in os.environ:
        del os.environ["API_KEY"]
    load_schema()


def valid_record(sid="SITE-A", rec_id="REC-001"):
    return {
        "record_id": rec_id,
        "timestamp_utc": "2026-09-10T07:26:00Z",
        "site_id": sid,
        "voltage_v": 2.27,
        "current_ma": 70.0,
        "reading": 1,
    }


# =========================================================================
# 1. Pipeline Tests
# =========================================================================

def test_pipeline_valid_records():
    writer = FakeWriter()
    records = [valid_record("SITE-A", "REC-01"), valid_record("SITE-B", "REC-02")]
    res = process_records("readings", records, transport="http", writer=writer)

    assert res["accepted"] == 2
    assert res["rejected"] == 0
    assert res["written"] == 2
    assert len(writer.points) == 2


def test_pipeline_topic_mismatch_mqtt():
    writer = FakeWriter()
    # Topic claims SITE-A, but record body says SITE-B
    rec = valid_record("SITE-B", "REC-01")
    res = process_records("readings", [rec], transport="mqtt", topic_site="SITE-A", writer=writer)

    assert res["accepted"] == 0
    assert res["rejected"] == 1
    assert res["reasons"].get("TOPIC_MISMATCH") == 1


def test_pipeline_topic_site_auto_fill():
    writer = FakeWriter()
    # Record has no site_id; topic_site is SITE-A
    rec = {
        "record_id": "REC-01",
        "timestamp_utc": "2026-09-10T07:26:00Z",
        "voltage_v": 2.27,
        "reading": 1,
    }
    res = process_records("readings", [rec], transport="mqtt", topic_site="SITE-A", writer=writer)
    assert res["accepted"] == 1
    assert writer.points[0].tags["site_id"] == "SITE-A"


def test_pipeline_write_failure_captured():
    writer = FakeWriter(should_fail=True)
    res = process_records("readings", [valid_record()], transport="http", writer=writer)
    assert res["accepted"] == 1
    assert res["write_failed"] == 1
    assert res["reasons"].get("WRITE_FAILED") == 1


# =========================================================================
# 2. HTTP API Tests
# =========================================================================

def test_http_api_health():
    writer = FakeWriter()
    app.state.writer = writer
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["storage"] == "mongodb"
    assert data["connected"] is True
    assert "influx_connected" not in data
    assert "mongo_connected" not in data
    assert "database_connected" not in data
    assert "counters" in data


def test_http_api_auth_success():
    writer = FakeWriter()
    app.state.writer = writer
    client = TestClient(app)
    resp = client.post(
        "/api/v1/ingest/readings",
        headers={"X-API-Key": "test_secret_api_key_12345"},
        json=valid_record(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] == 1
    assert data["written"] == 1


def test_http_api_auth_unauthorized():
    client = TestClient(app)
    resp = client.post(
        "/api/v1/ingest/readings",
        headers={"X-API-Key": "wrong_key"},
        json=valid_record(),
    )
    assert resp.status_code == 401


def test_http_api_missing_key_header():
    client = TestClient(app)
    resp = client.post("/api/v1/ingest/readings", json=valid_record())
    assert resp.status_code == 401


def test_http_api_503_when_api_key_not_configured():
    os.environ["API_KEY"] = ""
    client = TestClient(app)
    resp = client.post(
        "/api/v1/ingest/readings",
        headers={"X-API-Key": "any_key"},
        json=valid_record(),
    )
    assert resp.status_code == 503


def test_http_api_unknown_stream_404():
    client = TestClient(app)
    resp = client.post(
        "/api/v1/ingest/unknown_stream",
        headers={"X-API-Key": "test_secret_api_key_12345"},
        json=valid_record(),
    )
    assert resp.status_code == 404


def test_http_api_batch_with_partial_rejection():
    writer = FakeWriter()
    app.state.writer = writer
    client = TestClient(app)

    good = valid_record("SITE-A", "REC-01")
    bad = valid_record("SITE-A", "REC-02")
    bad["timestamp_utc"] = "2026-09-10T07:26:00"  # Naive time -> TIME_NOT_UTC

    resp = client.post(
        "/api/v1/ingest/readings",
        headers={"X-API-Key": "test_secret_api_key_12345"},
        json=[good, bad],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] == 1
    assert data["rejected"] == 1
    assert data["reasons"].get("TIME_NOT_UTC") == 1


def test_http_api_too_many_records_413():
    client = TestClient(app)
    # 5001 records exceeds 5000 limit
    oversized = [valid_record() for _ in range(5001)]
    resp = client.post(
        "/api/v1/ingest/readings",
        headers={"X-API-Key": "test_secret_api_key_12345"},
        json=oversized,
    )
    assert resp.status_code == 413


# =========================================================================
# 3. MQTT handle_message Tests
# =========================================================================

def test_mqtt_handle_message_valid():
    writer = FakeWriter()
    rec = valid_record("SITE-A")
    payload = json.dumps(rec).encode("utf-8")
    res = handle_message("solar/SITE-A/readings", payload, writer=writer)
    assert res["accepted"] == 1
    assert res["written"] == 1


def test_mqtt_handle_message_topic_mismatch():
    writer = FakeWriter()
    rec = valid_record("SITE-B")
    payload = json.dumps(rec).encode("utf-8")
    res = handle_message("solar/SITE-A/readings", payload, writer=writer)
    assert res["accepted"] == 0
    assert res["rejected"] == 1
    assert res["reasons"].get("TOPIC_MISMATCH") == 1


def test_mqtt_handle_message_bad_json():
    writer = FakeWriter()
    bad_payload = b"not a json object"
    res = handle_message("solar/SITE-A/readings", bad_payload, writer=writer)
    assert res["rejected"] == 1
    assert res["reasons"].get("BAD_JSON") == 1


def test_mqtt_handle_message_too_large():
    writer = FakeWriter()
    # Payload over 1 MB
    huge_bytes = b" " * (1024 * 1024 + 10)
    res = handle_message("solar/SITE-A/readings", huge_bytes, writer=writer)
    assert res["rejected"] == 1
    assert res["reasons"].get("TOO_LARGE") == 1


# =========================================================================
# 4. Influx Point Line Protocol & Exact Types
# =========================================================================

def test_writer_preserves_exact_field_types():
    pt = PointSpec(
        measurement="test_sensor_readings",
        time=datetime(2026, 9, 10, 7, 26, 0, tzinfo=timezone.utc),
        tags={"site_id": "SITE-A", "device_id": "ESP32-A01", "protocol": "WIFI"},
        fields={
            "voltage_v": 2.27,       # float
            "current_ma": 70.0,      # float ending in .0 (must not become int)
            "reading": 10,           # int
            "is_active": True,       # bool
            "record_id": "REC-01",   # str
        },
    )

    doc = point_to_document(pt)
    # Verify exact types in MongoDB document
    assert isinstance(doc["voltage_v"], float)
    assert doc["voltage_v"] == 2.27
    assert isinstance(doc["current_ma"], float)
    assert doc["current_ma"] == 70.0
    assert isinstance(doc["reading"], int)
    assert doc["reading"] == 10
    assert isinstance(doc["is_active"], bool)
    assert doc["is_active"] is True
    assert isinstance(doc["record_id"], str)
    assert doc["record_id"] == "REC-01"
    assert doc["site_id"] == "SITE-A"
    assert doc["timestamp_utc"] == datetime(2026, 9, 10, 7, 26, 0, tzinfo=timezone.utc)


# =========================================================================
# 5. Startup Guard Tests
# =========================================================================

def test_startup_guard_empty_prefix_refused():
    os.environ["MEASUREMENT_PREFIX"] = ""
    os.environ["ALLOW_REAL_MEASUREMENTS"] = "0"
    load_schema()
    with pytest.raises(RuntimeError, match="SAFETY GUARD"):
        verify_startup_safety()


def test_startup_guard_allow_real_measurements():
    os.environ["MEASUREMENT_PREFIX"] = ""
    os.environ["ALLOW_REAL_MEASUREMENTS"] = "1"
    load_schema()
    # Should not raise
    verify_startup_safety()


def test_startup_guard_test_prefix_allowed():
    os.environ["MEASUREMENT_PREFIX"] = "test_"
    os.environ["ALLOW_REAL_MEASUREMENTS"] = "0"
    load_schema()
    # Should not raise
    verify_startup_safety()


# =========================================================================
# 6. Secret Scrubbing Test
# =========================================================================

def test_secrets_never_appear_in_reject_log(tmp_path, monkeypatch):
    secret_token = "SUPER_SECRET_INFLUX_TOKEN_XYZ_987"
    monkeypatch.setenv("INFLUX_TOKEN", secret_token)
    temp_rejects = tmp_path / "scrub_test_rejects.jsonl"
    monkeypatch.setenv("REJECTS_PATH", str(temp_rejects))

    log_reject(
        transport="http",
        stream="readings",
        code="TIME_NOT_UTC",
        detail=f"Error occurred while processing request with token {secret_token}",
        record={"auth": secret_token, "val": 123},
    )

    # Read rejects file and assert secret_token is not present
    assert temp_rejects.exists()
    content = temp_rejects.read_text(encoding="utf-8")
    assert secret_token not in content
    assert "***" in content




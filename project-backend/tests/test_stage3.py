"""Unit tests for Stage 3: Read API, Replay Tool, Verify Tool.
Tests run purely in-process with FakeReader (no external services required).
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from backend.api.main import app
from backend.config import verify_startup_safety
from backend.mongo_reader import FakeReader, ReadError
from backend.tools.replay import (
    is_pv_fault_dataset,
    load_and_filter_records,
    run_http_replay,
    run_mqtt_replay,
)
from backend.tools.verify_backend import verify_dataset


def resolve_dataset_path() -> Path:
    r"""Finds pv_fault_dataset.csv by checking:
    1. Environment variable PV_DATASET
    2. ..\pv_fault_dataset.csv relative to project-backend
    3. ..\component3-federated-backend\data\shared\pv_fault_dataset.csv relative to project-backend
    """
    env_val = os.environ.get("PV_DATASET", "").strip()
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p.resolve()

    project_backend = Path(__file__).resolve().parent.parent
    p2 = (project_backend / ".." / "pv_fault_dataset.csv").resolve()
    if p2.exists():
        return p2

    p3 = (project_backend / ".." / "component3-federated-backend" / "data" / "shared" / "pv_fault_dataset.csv").resolve()
    if p3.exists():
        return p3

    return p2


DATASET_CSV_PATH = resolve_dataset_path()
TEST_READ_KEY = "test-dashboard-read-key-1234"
TEST_INGEST_KEY = "test-ingest-key-5678"


@pytest.fixture(autouse=True)
def setup_env_keys(monkeypatch):
    """Sets standard distinct test keys in environment for all stage 3 tests."""
    monkeypatch.setenv("READ_API_KEY", TEST_READ_KEY)
    monkeypatch.setenv("API_KEY", TEST_INGEST_KEY)
    monkeypatch.setenv("MEASUREMENT_PREFIX", "test_")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")


@pytest.fixture
def fake_reader():
    """Provides a FakeReader populated with sample test points."""
    reader = FakeReader()
    # Add sample readings for SITE-A
    t1 = datetime(2026, 9, 10, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 10, 10, 1, 0, tzinfo=timezone.utc)
    reader.add_row(
        stream="readings",
        site_id="SITE-A",
        time_utc=t1,
        tags={"site_id": "SITE-A", "device_id": "ESP32-A01", "protocol": "WIFI"},
        fields={"voltage_v": 18.5, "current_ma": 420.0, "record_id": "REC-TEST-1"},
    )
    reader.add_row(
        stream="readings",
        site_id="SITE-A",
        time_utc=t2,
        tags={"site_id": "SITE-A", "device_id": "ESP32-A01", "protocol": "WIFI"},
        fields={"voltage_v": 18.7, "current_ma": 430.0, "record_id": "REC-TEST-2"},
    )
    return reader


@pytest.fixture
def test_client(fake_reader):
    """Provides a TestClient with injected FakeReader."""
    app.state.reader = fake_reader
    with TestClient(app) as client:
        yield client


# ============================================================================
# 1. READ API: AUTHENTICATION TESTS
# ============================================================================

def test_read_api_missing_key_401(test_client):
    """Requesting read route without X-API-Key returns 401."""
    resp = test_client.get("/api/v1/sites")
    assert resp.status_code == 401


def test_read_api_wrong_key_401(test_client):
    """Requesting read route with incorrect key returns 401."""
    resp = test_client.get("/api/v1/sites", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


def test_read_api_ingest_key_fails_read_route_401(test_client):
    """The ingest API_KEY must NOT open read routes (returns 401)."""
    resp = test_client.get("/api/v1/sites", headers={"X-API-Key": TEST_INGEST_KEY})
    assert resp.status_code == 401


def test_ingest_api_read_key_fails_ingest_route_401(test_client):
    """The READ_API_KEY must NOT open ingest routes (returns 401)."""
    payload = {
        "timestamp_utc": "2026-09-10T07:26:00Z",
        "site_id": "SITE-A",
        "record_id": "REC-9999",
        "voltage_v": 18.0,
    }
    resp = test_client.post(
        "/api/v1/ingest/readings",
        headers={"X-API-Key": TEST_READ_KEY},
        json=payload,
    )
    assert resp.status_code == 401


def test_read_api_empty_read_key_503(monkeypatch, test_client):
    """If READ_API_KEY is empty on the server, returns 503."""
    monkeypatch.setenv("READ_API_KEY", "")
    resp = test_client.get("/api/v1/sites", headers={"X-API-Key": "any-key"})
    assert resp.status_code == 503


def test_read_api_valid_key_200(test_client):
    """Valid READ_API_KEY succeeds."""
    resp = test_client.get("/api/v1/sites", headers={"X-API-Key": TEST_READ_KEY})
    assert resp.status_code == 200


# ============================================================================
# 2. READ API: CORS HEADERS
# ============================================================================

def test_cors_preflight_options(test_client):
    """OPTIONS preflight request from allowed origin returns CORS headers."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "X-API-Key",
    }
    resp = test_client.options("/api/v1/sites", headers=headers)
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
    methods = resp.headers.get("access-control-allow-methods", "")
    assert "GET" in methods
    assert "OPTIONS" in methods


def test_cors_get_origin_header(test_client):
    """GET request with Origin receives Access-Control-Allow-Origin header."""
    headers = {
        "X-API-Key": TEST_READ_KEY,
        "Origin": "http://localhost:5173",
    }
    resp = test_client.get("/api/v1/sites", headers=headers)
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


# ============================================================================
# 3. READ API: SITES ENDPOINT
# ============================================================================

def test_get_sites(test_client):
    """GET /api/v1/sites returns all canonical sites with exact required fields."""
    resp = test_client.get("/api/v1/sites", headers={"X-API-Key": TEST_READ_KEY})
    assert resp.status_code == 200
    sites = resp.json()
    assert len(sites) == 6
    expected_fields = {"site_id", "device_id", "panel_id", "display_name", "protocol", "gateway_id", "link_profile"}
    for site in sites:
        assert set(site.keys()) == expected_fields
    sids = [s["site_id"] for s in sites]
    assert sids == ["SITE-A", "SITE-B", "SITE-C", "SITE-D", "SITE-E", "SITE-F"]


# ============================================================================
# 4. READ API: LATEST QUERY
# ============================================================================

def test_get_latest_success(test_client):
    """GET /api/v1/query/latest returns pivoted fields of newest row."""
    resp = test_client.get(
        "/api/v1/query/latest?stream=readings&site_id=SITE-A",
        headers={"X-API-Key": TEST_READ_KEY},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["stream"] == "readings"
    assert data["site_id"] == "SITE-A"
    assert data["time"] == "2026-09-10T10:01:00Z"
    assert data["tags"]["device_id"] == "ESP32-A01"
    assert data["fields"]["voltage_v"] == 18.7
    assert data["fields"]["record_id"] == "REC-TEST-2"


def test_get_latest_404_when_no_data(test_client):
    """GET /api/v1/query/latest returns 404 when site has no rows."""
    resp = test_client.get(
        "/api/v1/query/latest?stream=readings&site_id=SITE-B",
        headers={"X-API-Key": TEST_READ_KEY},
    )
    assert resp.status_code == 404


def test_get_latest_invalid_stream_422(test_client):
    """GET /api/v1/query/latest with unknown stream returns 422."""
    resp = test_client.get(
        "/api/v1/query/latest?stream=unknown_stream&site_id=SITE-A",
        headers={"X-API-Key": TEST_READ_KEY},
    )
    assert resp.status_code == 422


def test_get_latest_invalid_site_422(test_client):
    """GET /api/v1/query/latest with unknown site_id returns 422."""
    resp = test_client.get(
        "/api/v1/query/latest?stream=readings&site_id=SITE-Z",
        headers={"X-API-Key": TEST_READ_KEY},
    )
    assert resp.status_code == 422


# ============================================================================
# 5. READ API: SERIES QUERY & VALIDATION
# ============================================================================

def test_get_series_raw_points(test_client):
    """GET /api/v1/query/series returns raw points without aggregation."""
    url = (
        "/api/v1/query/series?stream=readings&site_id=SITE-A&field=voltage_v"
        "&start=2026-09-10T09:00:00Z&stop=2026-09-10T11:00:00Z"
    )
    resp = test_client.get(url, headers={"X-API-Key": TEST_READ_KEY})
    assert resp.status_code == 200
    data = resp.json()
    assert data["stream"] == "readings"
    assert data["site_id"] == "SITE-A"
    assert data["field"] == "voltage_v"
    assert len(data["points"]) == 2
    assert data["points"][0]["value"] == 18.5
    assert data["points"][1]["value"] == 18.7


def test_get_series_aggregation_mean(test_client):
    """GET /api/v1/query/series with every=1h&fn=mean returns aggregated point."""
    url = (
        "/api/v1/query/series?stream=readings&site_id=SITE-A&field=voltage_v"
        "&start=2026-09-10T09:00:00Z&stop=2026-09-10T11:00:00Z&every=1h&fn=mean"
    )
    resp = test_client.get(url, headers={"X-API-Key": TEST_READ_KEY})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["points"]) == 1
    assert data["points"][0]["value"] == pytest.approx(18.6)


def test_get_series_aggregation_disallowed_every_422(test_client):
    """Invalid every parameter returns 422."""
    url = (
        "/api/v1/query/series?stream=readings&site_id=SITE-A&field=voltage_v"
        "&start=2026-09-10T09:00:00Z&stop=2026-09-10T11:00:00Z&every=10m&fn=mean"
    )
    resp = test_client.get(url, headers={"X-API-Key": TEST_READ_KEY})
    assert resp.status_code == 422


def test_get_series_aggregation_disallowed_fn_422(test_client):
    """Invalid aggregation function returns 422."""
    url = (
        "/api/v1/query/series?stream=readings&site_id=SITE-A&field=voltage_v"
        "&start=2026-09-10T09:00:00Z&stop=2026-09-10T11:00:00Z&every=1h&fn=stddev"
    )
    resp = test_client.get(url, headers={"X-API-Key": TEST_READ_KEY})
    assert resp.status_code == 422


def test_get_series_every_without_fn_422(test_client):
    """Providing every without fn returns 422."""
    url = (
        "/api/v1/query/series?stream=readings&site_id=SITE-A&field=voltage_v"
        "&start=2026-09-10T09:00:00Z&stop=2026-09-10T11:00:00Z&every=1h"
    )
    resp = test_client.get(url, headers={"X-API-Key": TEST_READ_KEY})
    assert resp.status_code == 422


def test_get_series_time_range_exceeds_31_days_422(test_client):
    """Query range exceeding 31 days returns 422."""
    url = (
        "/api/v1/query/series?stream=readings&site_id=SITE-A&field=voltage_v"
        "&start=2026-08-01T00:00:00Z&stop=2026-09-05T00:00:00Z"
    )
    resp = test_client.get(url, headers={"X-API-Key": TEST_READ_KEY})
    assert resp.status_code == 422


def test_get_series_start_after_stop_422(test_client):
    """start >= stop returns 422."""
    url = (
        "/api/v1/query/series?stream=readings&site_id=SITE-A&field=voltage_v"
        "&start=2026-09-10T12:00:00Z&stop=2026-09-10T10:00:00Z"
    )
    resp = test_client.get(url, headers={"X-API-Key": TEST_READ_KEY})
    assert resp.status_code == 422


def test_get_series_naive_timestamp_422(test_client):
    """Naive timestamp without Z returns 422."""
    url = (
        "/api/v1/query/series?stream=readings&site_id=SITE-A&field=voltage_v"
        "&start=2026-09-10T10:00:00&stop=2026-09-10T11:00:00Z"
    )
    resp = test_client.get(url, headers={"X-API-Key": TEST_READ_KEY})
    assert resp.status_code == 422


def test_get_series_limit_caps_422(test_client):
    """limit < 1 or limit > 10000 returns 422."""
    url_zero = (
        "/api/v1/query/series?stream=readings&site_id=SITE-A&field=voltage_v"
        "&start=2026-09-10T10:00:00Z&stop=2026-09-10T11:00:00Z&limit=0"
    )
    assert test_client.get(url_zero, headers={"X-API-Key": TEST_READ_KEY}).status_code == 422

    url_too_high = (
        "/api/v1/query/series?stream=readings&site_id=SITE-A&field=voltage_v"
        "&start=2026-09-10T10:00:00Z&stop=2026-09-10T11:00:00Z&limit=10001"
    )
    assert test_client.get(url_too_high, headers={"X-API-Key": TEST_READ_KEY}).status_code == 422


def test_get_series_strict_stream_undeclared_field_422(test_client):
    """Field not declared in schema.yaml for strict stream returns 422."""
    url = (
        "/api/v1/query/series?stream=readings&site_id=SITE-A&field=custom_magic_field"
        "&start=2026-09-10T10:00:00Z&stop=2026-09-10T11:00:00Z"
    )
    resp = test_client.get(url, headers={"X-API-Key": TEST_READ_KEY})
    assert resp.status_code == 422


# ============================================================================
# 6. INPUT SAFETY & INJECTION ATTEMPTS (ALL MUST RETURN 422 WITH NO QUERY SENT)
# ============================================================================

@pytest.mark.parametrize(
    "bad_field",
    [
        'voltage_v"',
        'voltage_v |> drop()',
        'r._measurement',
        'voltage_v") |> drop()',
        'voltage_v; drop',
        'voltage-v',
        '1voltage',
    ],
)
def test_flux_injection_attempts_in_field(test_client, fake_reader, bad_field):
    """Injection attempts in field parameter return 422 without executing any query."""
    url = (
        f"/api/v1/query/series?stream=readings&site_id=SITE-A&field={bad_field}"
        "&start=2026-09-10T10:00:00Z&stop=2026-09-10T11:00:00Z"
    )
    resp = test_client.get(url, headers={"X-API-Key": TEST_READ_KEY})
    assert resp.status_code == 422
    assert len(fake_reader.queries_executed) == 0


@pytest.mark.parametrize(
    "bad_site",
    [
        'SITE-A"',
        'SITE-A" or "1"="1',
        'SITE-A |> drop()',
        'SITE-UNKNOWN',
    ],
)
def test_flux_injection_attempts_in_site(test_client, fake_reader, bad_site):
    """Injection attempts in site_id parameter return 422 without executing any query."""
    url = (
        f"/api/v1/query/series?stream=readings&site_id={bad_site}&field=voltage_v"
        "&start=2026-09-10T10:00:00Z&stop=2026-09-10T11:00:00Z"
    )
    resp = test_client.get(url, headers={"X-API-Key": TEST_READ_KEY})
    assert resp.status_code == 422
    assert len(fake_reader.queries_executed) == 0


# ============================================================================
# 7. REPLAY TOOL TESTS
# ============================================================================

def test_replay_dry_run_selection():
    """Replay dry-run loads exactly 3000 OPERATION rows from dataset."""
    if not DATASET_CSV_PATH.exists():
        pytest.skip("Dataset file not found")

    records = load_and_filter_records(
        file_path=DATASET_CSV_PATH,
        stream="readings",
        window="OPERATION",
    )
    assert len(records) == 3000

    # Ensure metadata columns are dropped from readings payloads
    sample = records[0]
    for drop_col in ("window", "run_id", "label", "health_score"):
        assert drop_col not in sample
    assert "voltage_v" in sample
    assert "record_id" in sample


def test_replay_site_and_limit_filter():
    """Replay respects --site SITE-C and --limit filters."""
    if not DATASET_CSV_PATH.exists():
        pytest.skip("Dataset file not found")

    records_site_c = load_and_filter_records(
        file_path=DATASET_CSV_PATH,
        stream="readings",
        window="OPERATION",
        sites=["SITE-C"],
    )
    assert len(records_site_c) == 660

    records_limit = load_and_filter_records(
        file_path=DATASET_CSV_PATH,
        stream="readings",
        window="OPERATION",
        sites=["SITE-C"],
        limit=50,
    )
    assert len(records_limit) == 50


def test_replay_health_demo_fixture():
    """Replay health stream with dataset produces demo fixture records."""
    if not DATASET_CSV_PATH.exists():
        pytest.skip("Dataset file not found")

    records = load_and_filter_records(
        file_path=DATASET_CSV_PATH,
        stream="health",
        window="OPERATION",
        limit=5,
    )
    assert len(records) == 5
    sample = records[0]
    assert set(sample.keys()) == {"record_id", "timestamp_utc", "site_id", "device_id", "health_score"}


def test_replay_refusal_prediction_twin(capsys):
    """Replay prediction or twin streams refuse pv_fault_dataset.csv."""
    if not DATASET_CSV_PATH.exists():
        pytest.skip("Dataset file not found")

    with pytest.raises(SystemExit):
        load_and_filter_records(
            file_path=DATASET_CSV_PATH,
            stream="prediction",
        )
    captured = capsys.readouterr()
    assert "Refusal Error" in captured.err

    with pytest.raises(SystemExit):
        load_and_filter_records(
            file_path=DATASET_CSV_PATH,
            stream="twin",
        )


def test_replay_startup_guard(monkeypatch):
    """Safety guard refuses startup when prefix is empty and ALLOW_REAL_MEASUREMENTS != 1."""
    monkeypatch.setenv("MEASUREMENT_PREFIX", "")
    monkeypatch.delenv("ALLOW_REAL_MEASUREMENTS", raising=False)
    with pytest.raises(RuntimeError) as exc_info:
        verify_startup_safety()
    assert "SAFETY GUARD" in str(exc_info.value)


# ============================================================================
# 8. VERIFY BACKEND TOOL TESTS
# ============================================================================

def test_verify_backend_ok(fake_reader, tmp_path):
    """verify_dataset returns 0 when expected equals found."""
    # Create temporary CSV matching FakeReader data
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(
        "record_id,timestamp_utc,site_id,device_id,reading,voltage_v\n"
        "REC-TEST-1,2026-09-10T10:00:00Z,SITE-A,ESP32-A01,1,18.5\n"
        "REC-TEST-2,2026-09-10T10:01:00Z,SITE-A,ESP32-A01,2,18.7\n"
    )

    ret = verify_dataset(
        stream="readings",
        file_path=csv_file,
        reader=fake_reader,
    )
    assert ret == 0


def test_verify_backend_mismatch(fake_reader, tmp_path, capsys):
    """verify_dataset returns 1 (MISMATCH) and reports missing record_id when count differs."""
    # CSV has 3 records, fake_reader only has 2
    csv_file = tmp_path / "test_mismatch.csv"
    csv_file.write_text(
        "record_id,timestamp_utc,site_id,device_id,reading,voltage_v\n"
        "REC-TEST-1,2026-09-10T10:00:00Z,SITE-A,ESP32-A01,1,18.5\n"
        "REC-TEST-2,2026-09-10T10:01:00Z,SITE-A,ESP32-A01,2,18.7\n"
        "REC-TEST-3,2026-09-10T10:02:00Z,SITE-A,ESP32-A01,3,18.9\n"
    )

    ret = verify_dataset(
        stream="readings",
        file_path=csv_file,
        reader=fake_reader,
    )
    assert ret == 1
    captured = capsys.readouterr()
    assert "Verification:       MISMATCH" in captured.out
    assert "REC-TEST-3" in captured.out


# ============================================================================
# 9. TEST HYGIENE & REJECT LOG FORMAT (ITEM H & G)
# ============================================================================

def test_rejects_path_configurable_and_format(tmp_path, monkeypatch):
    """REJECTS_PATH is configurable; stream is stream name or null; topic is separate."""
    custom_rejects = tmp_path / "custom_rejects.jsonl"
    monkeypatch.setenv("REJECTS_PATH", str(custom_rejects))

    from backend.ingest.rejects import log_reject

    # HTTP reject
    log_reject(
        transport="http",
        stream="readings",
        code="TIME_NOT_UTC",
        detail="Missing Z offset",
        record={"test": 1},
    )

    # MQTT reject with valid stream and topic
    log_reject(
        transport="mqtt",
        stream="readings",
        code="BAD_TYPE",
        detail="Not an object",
        record="bad string",
        topic="solar/SITE-A/readings",
    )

    # MQTT reject with invalid stream (must be null) and topic
    log_reject(
        transport="mqtt",
        stream=None,
        code="BAD_TOPIC",
        detail="Invalid topic",
        record="",
        topic="bad/topic/here",
    )

    assert custom_rejects.exists()
    lines = [json.loads(line) for line in custom_rejects.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3

    # Check HTTP line
    assert lines[0]["transport"] == "http"
    assert lines[0]["stream"] == "readings"
    assert lines[0]["topic"] is None
    assert lines[0]["code"] == "TIME_NOT_UTC"

    # Check MQTT valid stream line
    assert lines[1]["transport"] == "mqtt"
    assert lines[1]["stream"] == "readings"
    assert lines[1]["topic"] == "solar/SITE-A/readings"
    assert lines[1]["code"] == "BAD_TYPE"

    # Check MQTT invalid stream line (stream must be null)
    assert lines[2]["transport"] == "mqtt"
    assert lines[2]["stream"] is None
    assert lines[2]["topic"] == "bad/topic/here"
    assert lines[2]["code"] == "BAD_TOPIC"


def test_mqtt_service_reject_log_stream_and_topic(tmp_path, monkeypatch):
    """mqtt_service handle_message sets stream to null/stream and topic to topic."""
    custom_rejects = tmp_path / "mqtt_rejects.jsonl"
    monkeypatch.setenv("REJECTS_PATH", str(custom_rejects))

    from backend.ingest.mqtt_service import handle_message

    # BAD_TOPIC
    handle_message("invalid/topic", b'{"a": 1}')
    # UNKNOWN_STREAM
    handle_message("solar/SITE-A/unknown_stream", b'{"a": 1}')
    # BAD_JSON on valid topic
    handle_message("solar/SITE-A/readings", b'bad json string')

    lines = [json.loads(line) for line in custom_rejects.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3

    assert lines[0]["code"] == "BAD_TOPIC"
    assert lines[0]["stream"] is None
    assert lines[0]["topic"] == "invalid/topic"

    assert lines[1]["code"] == "UNKNOWN_STREAM"
    assert lines[1]["stream"] is None
    assert lines[1]["topic"] == "solar/SITE-A/unknown_stream"

    assert lines[2]["code"] == "BAD_JSON"
    assert lines[2]["stream"] == "readings"
    assert lines[2]["topic"] == "solar/SITE-A/readings"


def test_smoke_test_run_marker_and_expected_voltages():
    """smoke_test extract_smoke_records applies run marker delta to voltage_v."""
    if not DATASET_CSV_PATH.exists():
        pytest.skip("Dataset file not found")

    from backend.tools.smoke_test import extract_smoke_records

    marker1 = 100
    valid1, bad1, exp1 = extract_smoke_records(DATASET_CSV_PATH, marker1)

    marker2 = 200
    valid2, bad2, exp2 = extract_smoke_records(DATASET_CSV_PATH, marker2)

    assert len(valid1) == 6
    assert len(bad1) == 3
    assert len(exp1) == 6

    # Verify that different markers produce different voltage_v values
    for (sid, t) in exp1:
        assert exp1[(sid, t)] != exp2[(sid, t)]
        assert abs(exp2[(sid, t)] - exp1[(sid, t)] - 0.1) < 1e-4


# ============================================================================
# 10. READ ERROR & RESILIENCE TESTS
# ============================================================================

def test_read_api_query_latest_503_on_read_error(test_client, fake_reader):
    """GET /api/v1/query/latest returns 503 with database unreachable when ReadError is raised."""
    fake_reader.should_raise = True
    resp = test_client.get(
        "/api/v1/query/latest?stream=readings&site_id=SITE-A",
        headers={"X-API-Key": TEST_READ_KEY},
    )
    assert resp.status_code == 503
    assert resp.json() == {"detail": "database unreachable"}


def test_read_api_query_series_503_on_read_error(test_client, fake_reader):
    """GET /api/v1/query/series returns 503 with database unreachable when ReadError is raised."""
    fake_reader.should_raise = True
    url = (
        "/api/v1/query/series?stream=readings&site_id=SITE-A&field=voltage_v"
        "&start=2026-09-10T09:00:00Z&stop=2026-09-10T11:00:00Z"
    )
    resp = test_client.get(url, headers={"X-API-Key": TEST_READ_KEY})
    assert resp.status_code == 503
    assert resp.json() == {"detail": "database unreachable"}


def test_verify_backend_read_error_exits_2_no_counts(tmp_path, capsys):
    """verify_dataset returns code 2 on ReadError and never prints counts or MISMATCH."""
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(
        "record_id,timestamp_utc,site_id,device_id,reading,voltage_v\n"
        "REC-TEST-1,2026-09-10T10:00:00Z,SITE-A,ESP32-A01,1,18.5\n"
    )
    failing_reader = FakeReader(should_raise=True)
    ret = verify_dataset(
        stream="readings",
        file_path=csv_file,
        reader=failing_reader,
    )
    assert ret == 2
    captured = capsys.readouterr()
    assert "CANNOT READ MONGODB:" in captured.out
    assert "Expected rows:" not in captured.out
    assert "Found rows:" not in captured.out
    assert "MISMATCH" not in captured.out
    assert "field values" not in captured.out


def test_smoke_test_read_error_exits_2(capsys, monkeypatch):
    """smoke_test prints CANNOT READ MONGODB and exits code 2 when ping/read fails."""
    from backend.tools.smoke_test import main as smoke_main

    failing_reader = FakeReader(should_raise=True)
    monkeypatch.setattr("backend.tools.smoke_test.MongoReader.from_env", lambda: failing_reader)
    test_args = ["smoke_test.py", "--transport", "http", "--file", "dummy.csv"]
    monkeypatch.setattr("sys.argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        smoke_main()
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "CANNOT READ MONGODB:" in captured.out
    assert "field values" not in captured.out


def test_replay_http_totals_and_exit_nonzero_on_write_failed(monkeypatch, capsys):
    """run_http_replay prints totals and exits non-zero if write_failed > 0."""
    class FakeResponse:
        status_code = 200
        def json(self):
            return {
                "accepted": 1,
                "rejected": 0,
                "written": 0,
                "write_failed": 1,
                "reasons": {"INFLUX_ERROR": 1},
            }

    monkeypatch.setattr(
        "backend.tools.replay._send_http_request_with_retries",
        lambda client, url, headers, payload: FakeResponse(),
    )

    records = [
        {"record_id": "R1", "timestamp_utc": "2026-09-10T10:00:00Z", "site_id": "SITE-A"}
    ]

    with pytest.raises(SystemExit) as exc_info:
        run_http_replay(
            records=records,
            stream="readings",
            api_url="http://localhost:8000",
            api_key="test-key",
            batch_size=10,
            speed=0,
        )
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "=== HTTP Replay Totals ===" in captured.out
    assert "Accepted:     1" in captured.out
    assert "Write Failed: 1" in captured.out
    assert "Reasons:      {'INFLUX_ERROR': 1}" in captured.out


def test_replay_mqtt_reminder_printed(monkeypatch, capsys):
    """run_mqtt_replay prints published count and reminder to run verify_backend."""
    class FakeMQTTClient:
        def __init__(self, *args, **kwargs):
            pass
        def connect(self, *args, **kwargs):
            pass
        def loop_start(self):
            pass
        def loop_stop(self):
            pass
        def disconnect(self):
            pass
        def publish(self, *args, **kwargs):
            class Info:
                def wait_for_publish(self):
                    pass
            return Info()

    monkeypatch.setattr("paho.mqtt.client.Client", FakeMQTTClient)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr("backend.tools.replay.check_mqtt_broker", lambda *a, **k: True)

    records = [
        {"record_id": "R1", "timestamp_utc": "2026-09-10T10:00:00Z", "site_id": "SITE-A"}
    ]
    run_mqtt_replay(
        records=records,
        stream="readings",
        mqtt_host="localhost",
        mqtt_port=1883,
        batch_size=10,
        speed=0,
        file_path=Path("dummy_path.csv"),
    )
    captured = capsys.readouterr()
    assert "1 rows published" in captured.out
    assert "Reminder: Run verify_backend" in captured.out
    assert "--stream readings --file dummy_path.csv" in captured.out



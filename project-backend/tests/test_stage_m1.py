"""Unit tests for Stage M1: MongoDB Atlas Migration (No Docker).
Tests run 100% in-process with mocks and fake clients, without Docker or live Atlas.
Tests all requirements:
  - Storage factory and role-based credential selection.
  - Document shape and UTC conversion.
  - Type preservation and missing-field omission.
  - Upsert behaviour and unique indexes.
  - Query allow-lists and injection rejection ($where, $gt, dotted paths).
  - Error mapping to HTTP 503 with {"detail": "database unreachable"}.
  - Safe error logging (no secrets/passwords exposed).
  - Health behaviour when MongoDB is reachable vs unreachable.
  - Integration test gating with RUN_INTEGRATION_MONGO=1.
"""
from datetime import datetime, timedelta, timezone
import math
import os
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

from backend.api.main import app
from backend.mongo_reader import (
    FakeReader,
    MongoReader,
    ReadError,
)
from backend.mongo_writer import MongoWriter, point_to_document
from backend.points import PointSpec
from backend.storage import (
    build_role_uri,
    get_database_name,
    get_mongo_client,
    sanitize_error,
    sanitize_uri,
)

TEST_READ_KEY = "test-dashboard-read-key-m1"
TEST_INGEST_KEY = "test-ingest-key-m1"


@pytest.fixture(autouse=True)
def setup_m1_env(monkeypatch):
    """Sets safe mock environment for M1 unit tests."""
    monkeypatch.setenv("READ_API_KEY", TEST_READ_KEY)
    monkeypatch.setenv("API_KEY", TEST_INGEST_KEY)
    monkeypatch.setenv("MEASUREMENT_PREFIX", "test_")
    monkeypatch.setenv("MONGO_DB", "solar_pv_test")
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://mockcluster.mongodb.net/testdb?retryWrites=true")


# ============================================================================
# 1. STORAGE FACTORY SELECTION & SANITIZATION
# ============================================================================

def test_database_name_resolution(monkeypatch):
    """MONGO_DB env var is respected; defaults to solar_pv if unset."""
    monkeypatch.setenv("MONGO_DB", "custom_pv")
    assert get_database_name() == "custom_pv"

    monkeypatch.delenv("MONGO_DB", raising=False)
    assert get_database_name() == "solar_pv"


def test_build_role_uri_credentials(monkeypatch):
    """build_role_uri injects reader or app credentials into URI netloc."""
    base = "mongodb+srv://cluster0.abc.mongodb.net/solar_pv?retryWrites=true"

    # 1. Reader role
    monkeypatch.setenv("MONGO_READER_USER", "reader_usr")
    monkeypatch.setenv("MONGO_READER_PASSWORD", "reader_pwd")
    reader_uri = build_role_uri(base, role="reader")
    assert "reader_usr:reader_pwd@cluster0.abc.mongodb.net" in reader_uri

    # 2. App role
    monkeypatch.setenv("MONGO_APP_USER", "app_writer")
    monkeypatch.setenv("MONGO_APP_PASSWORD", "app_secret")
    app_uri = build_role_uri(base, role="app")
    assert "app_writer:app_secret@cluster0.abc.mongodb.net" in app_uri

    # 3. No role env set returns base URI
    monkeypatch.delenv("MONGO_READER_USER", raising=False)
    monkeypatch.delenv("MONGO_READER_PASSWORD", raising=False)
    assert build_role_uri(base, role="reader") == base


def test_get_mongo_client_timeout_and_missing_uri(monkeypatch):
    """MongoClient uses 10000ms selection/connect and 60000ms socket timeouts; raises ValueError when MONGO_URI missing."""
    monkeypatch.setenv("MONGO_URI", "")
    with pytest.raises(ValueError, match="MONGO_URI"):
        get_mongo_client()

    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    with patch("backend.storage.MongoClient") as mock_client:
        get_mongo_client(role="app")
        mock_client.assert_called_once()
        _, kwargs = mock_client.call_args
        assert kwargs["serverSelectionTimeoutMS"] == 10000
        assert kwargs["connectTimeoutMS"] == 10000
        assert kwargs["socketTimeoutMS"] == 60000
        assert kwargs["retryWrites"] is True


# ============================================================================
# 2. DOCUMENT SHAPE AND UTC CONVERSION
# ============================================================================

def test_document_shape_and_utc_conversion():
    """Documents contain site_id, device_id, protocol, timestamp_utc as BSON UTC datetime."""
    t_naive = datetime(2026, 9, 10, 8, 30, 0)
    pt = PointSpec(
        measurement="test_sensor_readings",
        time=t_naive,
        tags={"site_id": "SITE-A", "device_id": "ESP32-A01", "protocol": "WIFI"},
        fields={"voltage_v": 2.34, "reading": 5},
    )
    doc = point_to_document(pt)

    assert doc["site_id"] == "SITE-A"
    assert doc["device_id"] == "ESP32-A01"
    assert doc["protocol"] == "WIFI"
    assert isinstance(doc["timestamp_utc"], datetime)
    assert doc["timestamp_utc"].tzinfo is not None
    assert doc["timestamp_utc"] == datetime(2026, 9, 10, 8, 30, 0, tzinfo=timezone.utc)
    assert doc["voltage_v"] == 2.34
    assert doc["reading"] == 5


# ============================================================================
# 3. TYPE PRESERVATION AND MISSING-FIELD OMISSION
# ============================================================================

def test_type_preservation_and_missing_field_omission():
    """Preserves int, float, string, boolean types; omits missing/NaN fields without inventing 0."""
    pt = PointSpec(
        measurement="test_sensor_readings",
        time=datetime(2026, 9, 10, 9, 0, 0, tzinfo=timezone.utc),
        tags={"site_id": "SITE-B", "device_id": "ESP32-B01", "protocol": "WIFI"},
        fields={
            "voltage_v": 12.0,       # float ending in .0 (must remain float)
            "reading": 42,           # int
            "is_valid": False,       # boolean False must not be omitted
            "real_zero": 0.0,        # real numeric 0.0 must be preserved
            "record_id": "REC-01",   # string
            "missing_none": None,    # must be omitted
            "missing_nan": float("nan"),  # must be omitted
            "missing_empty": "",     # empty string must be omitted
        },
    )
    doc = point_to_document(pt)

    assert type(doc["voltage_v"]) is float
    assert doc["voltage_v"] == 12.0
    assert type(doc["reading"]) is int
    assert doc["reading"] == 42
    assert type(doc["is_valid"]) is bool
    assert doc["is_valid"] is False
    assert type(doc["real_zero"]) is float
    assert doc["real_zero"] == 0.0
    assert type(doc["record_id"]) is str

    assert "missing_none" not in doc
    assert "missing_nan" not in doc
    assert "missing_empty" not in doc


# ============================================================================
# 4. UPSERT BEHAVIOUR AND UNIQUE INDEXES
# ============================================================================

def test_mongo_writer_indexes_and_replace_one():
    """MongoWriter ensures unique index on (site_id, timestamp_utc) and ReplaceOne upsert."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    writer = MongoWriter(client=MagicMock(), db=mock_db)

    # 1. Test ensure_indexes
    writer.ensure_indexes("test_sensor_readings")
    assert mock_collection.create_index.call_count == 2
    # Check unique index call
    first_call_args, first_call_kwargs = mock_collection.create_index.call_args_list[0]
    assert first_call_args[0] == [("site_id", 1), ("timestamp_utc", 1)]
    assert first_call_kwargs.get("unique") is True

    # Check timestamp index call
    second_call_args, _ = mock_collection.create_index.call_args_list[1]
    assert second_call_args[0] == [("timestamp_utc", 1)]

    # 2. Test bulk_write with ReplaceOne
    pt = PointSpec(
        measurement="test_sensor_readings",
        time=datetime(2026, 9, 10, 10, 0, 0, tzinfo=timezone.utc),
        tags={"site_id": "SITE-A", "device_id": "ESP32-A01", "protocol": "WIFI"},
        fields={"voltage_v": 2.5},
    )
    mock_collection.bulk_write.return_value = MagicMock(upserted_count=1, modified_count=0, matched_count=0)

    written, failed, err = writer.write([pt])
    assert written == 1
    assert failed == 0
    assert err is None

    mock_collection.bulk_write.assert_called_once()
    ops, kwargs = mock_collection.bulk_write.call_args
    assert kwargs.get("ordered") is False
    assert len(ops[0]) == 1
    replace_op = ops[0][0]
    assert replace_op._filter == {
        "site_id": "SITE-A",
        "timestamp_utc": datetime(2026, 9, 10, 10, 0, 0, tzinfo=timezone.utc),
    }
    assert replace_op._upsert is True


# ============================================================================
# 5. QUERY ALLOW-LISTS AND INJECTION REJECTION
# ============================================================================

def test_reader_query_injection_rejection():
    """MongoReader rejects $where, $gt, dotted paths, and arbitrary collection names."""
    reader = MongoReader(client=MagicMock(), db=MagicMock())

    # Invalid stream
    with pytest.raises(ValueError, match="Invalid stream"):
        reader.validate_stream("arbitrary_collection")

    # Invalid site
    with pytest.raises(ValueError, match="Invalid site_id"):
        reader.validate_site_id("SITE-UNKNOWN")

    # Dotted path
    with pytest.raises(ValueError, match="Invalid field name"):
        reader.validate_field_name("readings", "voltage.nested")

    # Injections
    with pytest.raises(ValueError, match="Invalid field name"):
        reader.validate_field_name("readings", "$where")

    with pytest.raises(ValueError, match="Invalid field name"):
        reader.validate_field_name("readings", "field $gt 5")

    with pytest.raises(ValueError, match="Invalid field name"):
        reader.validate_field_name("readings", "1_starts_with_number")

    # Undeclared field in strict stream
    with pytest.raises(ValueError, match="not a declared field"):
        reader.validate_field_name("readings", "random_unregistered_field")

    # Range exceeding 31 days
    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    t_too_far = t0 + timedelta(days=32)
    with pytest.raises(ValueError, match="exceeds maximum"):
        reader.validate_time_range(t0, t_too_far)

    # Start >= Stop
    with pytest.raises(ValueError, match="strictly before"):
        reader.validate_time_range(t0, t0)


# ============================================================================
# 6. ERROR MAPPING TO HTTP 503
# ============================================================================

def test_read_error_mapping_to_http_503():
    """Database failure maps to safe ReadError and HTTP 503 with detail 'database unreachable'."""
    fake_reader = FakeReader(should_raise=True)
    app.state.reader = fake_reader

    with TestClient(app) as client:
        # Latest endpoint
        resp_latest = client.get(
            "/api/v1/query/latest?stream=readings&site_id=SITE-A",
            headers={"X-API-Key": TEST_READ_KEY},
        )
        assert resp_latest.status_code == 503
        assert resp_latest.json() == {"detail": "database unreachable"}

        # Series endpoint
        resp_series = client.get(
            "/api/v1/query/series?stream=readings&site_id=SITE-A&field=voltage_v"
            "&start=2026-09-10T10:00:00Z&stop=2026-09-10T11:00:00Z",
            headers={"X-API-Key": TEST_READ_KEY},
        )
        assert resp_series.status_code == 503
        assert resp_series.json() == {"detail": "database unreachable"}


# ============================================================================
# 7. SAFE ERROR LOGGING (NO SECRETS EXPOSED)
# ============================================================================

def test_sanitize_error_and_uri_scrubbing(monkeypatch):
    """Credentials in URIs and error strings are stripped completely."""
    secret_pass = "super_secret_atlas_pass_1234"
    monkeypatch.setenv("MONGO_APP_PASSWORD", secret_pass)

    raw_uri = f"mongodb+srv://admin_user:{secret_pass}@cluster0.solar.mongodb.net/solar_pv"
    masked = sanitize_uri(raw_uri)
    assert secret_pass not in masked
    assert "admin_user" not in masked or "***" in masked
    assert "***:***@cluster0.solar.mongodb.net" in masked

    error_with_creds = f"AuthenticationFailed: Bad auth on URI mongodb+srv://admin:{secret_pass}@cluster0"
    cleaned_err = sanitize_error(error_with_creds)
    assert secret_pass not in cleaned_err
    assert "***" in cleaned_err


# ============================================================================
# 8. HEALTH BEHAVIOUR WHEN MONGODB REACHABLE VS UNREACHABLE
# ============================================================================

def test_health_check_mongodb_status():
    """Health endpoint identifies MongoDB as storage backend and reports connectivity."""
    class MockWriter:
        def __init__(self, healthy: bool):
            self.healthy = healthy

        def ping(self) -> bool:
            return self.healthy

    with TestClient(app) as client:
        # 1. Unhealthy/Unreachable
        app.state.writer = MockWriter(healthy=False)
        resp_deg = client.get("/health")
        assert resp_deg.status_code == 200
        data_deg = resp_deg.json()
        assert data_deg["status"] == "degraded"
        assert data_deg["storage"] == "mongodb"
        assert data_deg["connected"] is False
        assert "mongo_connected" not in data_deg
        assert "database_connected" not in data_deg
        assert "influx_connected" not in data_deg
        assert "counters" in data_deg

        # 2. Healthy/Connected
        app.state.writer = MockWriter(healthy=True)
        resp_ok = client.get("/health")
        assert resp_ok.status_code == 200
        data_ok = resp_ok.json()
        assert data_ok["status"] == "ok"
        assert data_ok["storage"] == "mongodb"
        assert data_ok["connected"] is True
        assert "mongo_connected" not in data_ok
        assert "database_connected" not in data_ok
        assert "influx_connected" not in data_ok
        assert "counters" in data_ok


# ============================================================================
# 9. READER URI DERIVATION TESTS
# ============================================================================

def test_build_role_uri_reader_derivation():
    """Tests that build_role_uri correctly derives reader account URI for mongodb+srv
    by replacing only the username and password while preserving host, path, and query.
    """
    base_uri = "mongodb+srv://writer_app:writer_secret@cluster0.abcde.mongodb.net/solar_pv?retryWrites=true&w=majority"
    reader_uri = build_role_uri(
        base_uri=base_uri,
        role="reader",
        username="solar_reader",
        password="reader_secret_pass",
    )
    assert reader_uri.startswith("mongodb+srv://solar_reader:reader_secret_pass@cluster0.abcde.mongodb.net/solar_pv")
    assert "retryWrites=true" in reader_uri
    assert "w=majority" in reader_uri
    assert "writer_app" not in reader_uri
    assert "writer_secret" not in reader_uri

    # Test special characters in credentials properly quoted
    special_uri = build_role_uri(
        base_uri=base_uri,
        role="reader",
        username="solar@reader",
        password="p@ss:word#123",
    )
    assert "solar%40reader" in special_uri
    assert "p%40ss%3Aword%23123" in special_uri
    assert "@cluster0.abcde.mongodb.net" in special_uri


# ============================================================================
# 10. MONGO WRITER CHUNKING & TRANSIENT RETRY TESTS
# ============================================================================

def test_mongo_writer_batch_chunking():
    """Tests that MongoWriter chunks bulk_write operations according to MONGO_WRITE_BATCH."""
    from unittest.mock import MagicMock
    from backend.mongo_writer import MongoWriter
    from backend.points import PointSpec

    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    writer = MongoWriter(client=MagicMock(), db=mock_db, write_batch=50, write_pause_s=0.0)

    # 125 test points -> 3 chunks: 50, 50, 25
    points = [
        PointSpec(
            measurement="readings",
            tags={"site_id": "SITE-A", "device_id": "D1", "protocol": "HTTP"},
            fields={"voltage_v": float(i)},
            time=datetime.now(timezone.utc),
        )
        for i in range(125)
    ]

    written, failed, err = writer.write(points)
    assert written == 125
    assert failed == 0
    assert err is None
    assert mock_col.bulk_write.call_count == 3
    # Check chunk sizes passed to bulk_write
    call_args_list = mock_col.bulk_write.call_args_list
    assert len(call_args_list[0][0][0]) == 50
    assert len(call_args_list[1][0][0]) == 50
    assert len(call_args_list[2][0][0]) == 25


def test_mongo_writer_transient_retry_success():
    """Tests that MongoWriter retries up to 3 times on transient errors and succeeds."""
    from unittest.mock import MagicMock
    from pymongo.errors import AutoReconnect
    from backend.mongo_writer import MongoWriter
    from backend.points import PointSpec

    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    # Fail twice with AutoReconnect, succeed on 3rd attempt
    mock_col.bulk_write.side_effect = [
        AutoReconnect("transient socket connection reset"),
        AutoReconnect("transient handshake timeout"),
        MagicMock(),
    ]

    writer = MongoWriter(client=MagicMock(), db=mock_db, write_batch=100, write_pause_s=0.0)
    point = PointSpec(
        measurement="readings",
        tags={"site_id": "SITE-A", "device_id": "D1", "protocol": "HTTP"},
        fields={"voltage_v": 230.0},
        time=datetime.now(timezone.utc),
    )

    written, failed, err = writer.write([point])
    assert written == 1
    assert failed == 0
    assert err is None
    assert mock_col.bulk_write.call_count == 3


def test_mongo_writer_transient_retry_exhausted():
    """Tests that MongoWriter marks batch failed after exhausting 3 transient retries."""
    from unittest.mock import MagicMock
    from pymongo.errors import NetworkTimeout
    from backend.mongo_writer import MongoWriter
    from backend.points import PointSpec

    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    # Fail 4 times (1 initial + 3 retries) with NetworkTimeout
    mock_col.bulk_write.side_effect = NetworkTimeout("timed out waiting for socket")

    writer = MongoWriter(client=MagicMock(), db=mock_db, write_batch=100, write_pause_s=0.0)
    point = PointSpec(
        measurement="readings",
        tags={"site_id": "SITE-A", "device_id": "D1", "protocol": "HTTP"},
        fields={"voltage_v": 230.0},
        time=datetime.now(timezone.utc),
    )

    written, failed, err = writer.write([point])
    assert written == 0
    assert failed == 1
    assert err is not None
    assert mock_col.bulk_write.call_count == 4


# ============================================================================
# 11. INTEGRATION TEST GATING (RUN_INTEGRATION_MONGO=1 ONLY)
# ============================================================================

@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_MONGO") != "1",
    reason="Integration test skipped unless RUN_INTEGRATION_MONGO=1",
)
def test_atlas_live_integration():
    """Optional live Atlas integration test; strictly gated."""
    reader = MongoReader.from_env()
    assert reader.ping() is True

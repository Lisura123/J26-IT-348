"""Unit tests for Component 3 MongoDB Atlas integration.
Runs 100% in-process without requiring live Atlas or external network connections.
Tests:
- Document shape, types preserved, NaN/None omitted
- Timezone-aware UTC BSON datetime conversion
- Unique keys and ReplaceOne upsert filters per collection
- All 126 OTA rows kept (missing deployed_at_utc row preserved)
- Chunking, pause, and transient error backoff retry
- verify_mongo error handling with exit code 2
- Secret and credential scrubbing in error messages
"""
from datetime import datetime, timezone
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from pymongo.errors import AutoReconnect, NetworkTimeout, ServerSelectionTimeoutError
import pytest

from src import data_loader as dl
from src.mongo_client import (
    BSON_DATETIME_FIELDS,
    COLLECTION_UNIQUE_KEYS,
    ensure_unique_index,
    row_to_document,
    scrub_secrets,
    to_utc_datetime,
    write_documents,
)
from src.verify_mongo import count_mongo


# ============================================================================
# 1. DOCUMENT SHAPE & TYPE PRESERVATION
# ============================================================================

def test_document_shape_and_type_preservation():
    """Exact types are kept: float stays float, int stays int, str stays str, bool stays bool."""
    row = {
        "site_id": "SITE-A",
        "round_id": 5,
        "delta_l2_norm": 0.4285,
        "online": True,
        "selected": False,
        "zero_val": 0.0,
        "zero_int": 0,
        "np_int": np.int64(42),
        "np_float": np.float64(3.1415),
    }
    doc = row_to_document(row)

    assert isinstance(doc["site_id"], str)
    assert doc["site_id"] == "SITE-A"

    assert isinstance(doc["round_id"], int)
    assert doc["round_id"] == 5

    assert isinstance(doc["delta_l2_norm"], float)
    assert abs(doc["delta_l2_norm"] - 0.4285) < 1e-6

    assert isinstance(doc["online"], bool)
    assert doc["online"] is True

    assert isinstance(doc["selected"], bool)
    assert doc["selected"] is False

    assert isinstance(doc["zero_val"], float)
    assert doc["zero_val"] == 0.0

    assert isinstance(doc["zero_int"], int)
    assert doc["zero_int"] == 0

    assert isinstance(doc["np_int"], int)
    assert doc["np_int"] == 42

    assert isinstance(doc["np_float"], float)
    assert abs(doc["np_float"] - 3.1415) < 1e-4


# ============================================================================
# 2. MISSING VALUES AND NAN OMISSION (NEVER ZERO)
# ============================================================================

def test_nan_and_none_omitted():
    """NaN, None, and pd.NA are omitted from the document, never stored as 0 or null."""
    row = {
        "valid_metric": 12.5,
        "missing_none": None,
        "missing_nan": float("nan"),
        "missing_np_nan": np.nan,
        "missing_pd_na": pd.NA,
        "real_zero": 0.0,
    }
    doc = row_to_document(row)

    assert "valid_metric" in doc
    assert doc["valid_metric"] == 12.5

    assert "real_zero" in doc
    assert doc["real_zero"] == 0.0

    assert "missing_none" not in doc
    assert "missing_nan" not in doc
    assert "missing_np_nan" not in doc
    assert "missing_pd_na" not in doc


# ============================================================================
# 3. TIMEZONE-AWARE UTC BSON DATES
# ============================================================================

def test_datetime_utc_bson_conversion():
    """Primary datetime fields are converted to timezone-aware UTC BSON datetimes."""
    for field in [
        "timestamp_utc",
        "start_utc",
        "sent_at_utc",
        "published_at_utc",
        "deployed_at_utc",
        "round_opened_utc",
    ]:
        row = {field: "2026-09-10T05:07:00Z"}
        doc = row_to_document(row)
        assert field in doc
        dt = doc[field]
        assert isinstance(dt, datetime)
        assert dt.tzinfo is not None
        assert dt.tzinfo == timezone.utc
        assert dt.year == 2026
        assert dt.month == 9
        assert dt.day == 10

    # Missing datetime is omitted
    row_missing = {"deployed_at_utc": None}
    doc_missing = row_to_document(row_missing)
    assert "deployed_at_utc" not in doc_missing


# ============================================================================
# 4. UNIQUE KEYS AND UPSERT FILTERS PER COLLECTION
# ============================================================================

def test_unique_keys_and_upsert_filters():
    """Verify ReplaceOne upsert filters match the expected compound unique keys."""
    expected_keys = {
        "network_link_metrics": ["run_id", "site_id", "timestamp_utc"],
        "network_outages": ["run_id", "site_id", "start_utc"],
        "fl_client_updates": ["run_id", "update_id"],
        "fl_global_models": ["run_id", "round_id"],
        "fl_site_state": ["run_id", "round_id", "site_id"],
        "ota_deployments": ["run_id", "site_id", "round_id"],
    }
    assert COLLECTION_UNIQUE_KEYS == expected_keys

    # Test ReplaceOne filter construction with a mock collection
    fake_col = MagicMock()
    now_dt = datetime.now(timezone.utc)
    docs = [
        {
            "run_id": "shared",
            "site_id": "SITE-A",
            "timestamp_utc": now_dt,
            "rssi_dbm": -72.0,
        }
    ]
    write_documents(fake_col, "network_link_metrics", docs)
    assert fake_col.bulk_write.called
    ops = fake_col.bulk_write.call_args[0][0]
    assert len(ops) == 1
    op = ops[0]
    assert op._filter == {
        "run_id": "shared",
        "site_id": "SITE-A",
        "timestamp_utc": now_dt,
    }
    assert op._upsert is True


# ============================================================================
# 5. ALL 126 OTA ROWS KEPT
# ============================================================================

def test_ota_all_126_rows_kept():
    """All 126 OTA status rows are preserved; row without deployed_at_utc retains all other fields."""
    st = dl.load_backend_status()
    assert len(st) == 126

    docs = []
    missing_deployed_count = 0
    for r in st.to_dict("records"):
        doc = row_to_document(r)
        docs.append(doc)
        if "deployed_at_utc" not in doc:
            missing_deployed_count += 1
            assert "site_id" in doc
            assert "round_id" in doc

    assert len(docs) == 126
    assert missing_deployed_count == 1  # Exactly 1 row had missing deployment timestamp


# ============================================================================
# 6. BATCH CHUNKING & TRANSIENT ERROR RETRY
# ============================================================================

def test_batch_chunking_and_transient_retry(monkeypatch):
    """write_documents chunks operations by batch_size and retries on transient errors."""
    fake_col = MagicMock()
    # First chunk fails once with AutoReconnect then succeeds
    attempts = [0]
    def fake_bulk_write(ops, ordered=False):
        if attempts[0] == 0:
            attempts[0] += 1
            raise AutoReconnect("connection reset by peer")
        return MagicMock()

    fake_col.bulk_write.side_effect = fake_bulk_write
    monkeypatch.setattr("time.sleep", lambda s: None)

    docs = [{"run_id": "shared", "update_id": f"U{i}"} for i in range(15)]
    count = write_documents(fake_col, "fl_client_updates", docs, batch_size=5)

    assert count == 15
    # 1 retry on first chunk + 3 chunks = 4 calls total
    assert fake_col.bulk_write.call_count == 4


def test_write_documents_retries_exhausted_raises(monkeypatch):
    """write_documents raises when transient error persists past max_retries."""
    fake_col = MagicMock()
    fake_col.bulk_write.side_effect = AutoReconnect("connection reset by peer")
    monkeypatch.setattr("time.sleep", lambda s: None)

    docs = [{"run_id": "shared", "update_id": "U1"}]
    with pytest.raises(AutoReconnect) as exc:
        write_documents(fake_col, "fl_client_updates", docs, batch_size=5, max_retries=2)
    assert "connection reset by peer" in str(exc.value)
    # attempt 0 + 2 retries = 3 calls
    assert fake_col.bulk_write.call_count == 3


# ============================================================================
# 7. READ ERROR HANDLING AND EXIT CODE 2
# ============================================================================

def test_verify_mongo_read_failure_exit_2(capsys):
    """count_mongo prints 'CANNOT READ MONGODB: ...' and exits code 2 on read error."""
    mock_db = MagicMock()
    mock_db["fl_client_updates"].count_documents.side_effect = ServerSelectionTimeoutError(
        "No replica set members found"
    )

    with pytest.raises(SystemExit) as exc:
        count_mongo(mock_db, "fl_client_updates", {"run_id": "shared"})

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "CANNOT READ MONGODB:" in captured.err
    assert "No replica set members found" in captured.err
    assert "MISMATCH" not in captured.out
    assert "mongo=" not in captured.out


# ============================================================================
# 8. SECRET SCRUBBING
# ============================================================================

def test_secret_scrubbing_no_leak():
    """Credentials and passwords are scrubbed from URIs and error strings."""
    secret_uri = "mongodb+srv://admin_user:P@ssw0rd123!@cluster0.mongodb.net/solar_pv?retryWrites=true"
    scrubbed = scrub_secrets(secret_uri)

    assert "P@ssw0rd123!" not in scrubbed
    assert "admin_user" not in scrubbed
    assert "mongodb+srv://***:***@cluster0.mongodb.net/solar_pv" in scrubbed

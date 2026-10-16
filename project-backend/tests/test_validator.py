"""Test suite for project-backend validation and data contract rules.
Pure Python in-process tests.
"""
from datetime import datetime, timezone
import os
import pytest

from backend.config import get_final_measurement, load_schema
from backend.validate import (
    REJECT_BAD_FIELD_NAME,
    REJECT_BAD_TYPE,
    REJECT_BAD_VALUE,
    REJECT_DEVICE_MISMATCH,
    REJECT_EMPTY_POINT,
    REJECT_MISSING_REQUIRED,
    REJECT_PROTOCOL_MISMATCH,
    REJECT_RESERVED_MEASUREMENT,
    REJECT_TIME_FORMAT,
    REJECT_TIME_NOT_UTC,
    REJECT_UNKNOWN_SITE,
    validate_record,
)


@pytest.fixture(autouse=True)
def reset_env_and_schema():
    """Ensure environment variable overrides are clean between tests."""
    old_prefix = os.environ.get("MEASUREMENT_PREFIX")
    if "MEASUREMENT_PREFIX" in os.environ:
        del os.environ["MEASUREMENT_PREFIX"]
    load_schema()
    yield
    if old_prefix is not None:
        os.environ["MEASUREMENT_PREFIX"] = old_prefix
    elif "MEASUREMENT_PREFIX" in os.environ:
        del os.environ["MEASUREMENT_PREFIX"]
    load_schema()


def valid_readings_row():
    return {
        "record_id": "REC-00001",
        "timestamp_utc": "2026-09-10T07:26:00Z",
        "site_id": "SITE-A",
        "voltage_v": 2.27,
        "current_ma": 70.4,
        "reading": 1,
    }


def test_missing_z_rejected():
    """Missing Z or timezone indicator must be rejected with TIME_NOT_UTC."""
    row = valid_readings_row()
    row["timestamp_utc"] = "2026-09-10T07:26:00"  # Naive
    res = validate_record("readings", row)
    assert not res.success
    assert res.reject_code == REJECT_TIME_NOT_UTC


def test_plus_0000_offset_accepted():
    """Explicit UTC offset +00:00 must be accepted."""
    row = valid_readings_row()
    row["timestamp_utc"] = "2026-09-10T07:26:00+00:00"
    res = validate_record("readings", row)
    assert res.success
    assert res.point is not None
    assert res.point.time == datetime(2026, 9, 10, 7, 26, 0, tzinfo=timezone.utc)


def test_naive_time_rejected():
    """Naive timestamp without timezone is rejected."""
    row = valid_readings_row()
    row["timestamp_utc"] = "2026-09-10 07:26:00"
    res = validate_record("readings", row)
    assert not res.success
    assert res.reject_code == REJECT_TIME_NOT_UTC


def test_non_utc_offset_rejected():
    """Non-UTC offsets (e.g. +05:30) must be rejected, never converted."""
    row = valid_readings_row()
    row["timestamp_utc"] = "2026-09-10T12:56:00+05:30"
    res = validate_record("readings", row)
    assert not res.success
    assert res.reject_code == REJECT_TIME_NOT_UTC


def test_unknown_site_rejected():
    """Sites not in sites.csv must be rejected with UNKNOWN_SITE."""
    row = valid_readings_row()
    row["site_id"] = "SITE-Z"
    res = validate_record("readings", row)
    assert not res.success
    assert res.reject_code == REJECT_UNKNOWN_SITE


def test_device_from_another_site_rejected():
    """If payload contains device_id that belongs to another site, reject with DEVICE_MISMATCH."""
    row = valid_readings_row()
    row["site_id"] = "SITE-A"  # expected ESP32-A01
    row["device_id"] = "ESP32-B01"
    res = validate_record("readings", row)
    assert not res.success
    assert res.reject_code == REJECT_DEVICE_MISMATCH


def test_wrong_protocol_rejected():
    """If payload specifies a protocol different from sites.csv, reject with PROTOCOL_MISMATCH."""
    row = valid_readings_row()
    row["site_id"] = "SITE-A"  # expected WIFI
    row["protocol"] = "LORA"
    res = validate_record("readings", row)
    assert not res.success
    assert res.reject_code == REJECT_PROTOCOL_MISMATCH


def test_nan_and_empty_string_omitted():
    """Missing values (None, '', NaN) must be omitted, never written as zero."""
    row = valid_readings_row()
    row["power_mw"] = float("nan")
    row["humidity_pct"] = ""
    row["dust_vo_v"] = None
    res = validate_record("readings", row)
    assert res.success
    assert res.point is not None
    assert "power_mw" not in res.point.fields
    assert "humidity_pct" not in res.point.fields
    assert "dust_vo_v" not in res.point.fields


def test_real_numeric_zero_kept():
    """A real numeric zero (0, 0.0) must be preserved in fields, not dropped."""
    row = valid_readings_row()
    row["power_mw"] = 0.0
    row["reading"] = 0
    res = validate_record("readings", row)
    assert res.success
    assert res.point is not None
    assert res.point.fields["power_mw"] == 0.0
    assert res.point.fields["reading"] == 0


def test_ints_converted_to_float_for_float_fields():
    """Integer values given for float_fields must be cast to Python float."""
    row = valid_readings_row()
    row["voltage_v"] = 2  # integer
    res = validate_record("readings", row)
    assert res.success
    assert res.point is not None
    assert isinstance(res.point.fields["voltage_v"], float)
    assert res.point.fields["voltage_v"] == 2.0


def test_type_stable_across_rows():
    """Float fields must always produce float type across multiple rows."""
    r1 = valid_readings_row()
    r1["voltage_v"] = 2.27
    r2 = valid_readings_row()
    r2["voltage_v"] = 3
    res1 = validate_record("readings", r1)
    res2 = validate_record("readings", r2)
    assert isinstance(res1.point.fields["voltage_v"], float)
    assert isinstance(res2.point.fields["voltage_v"], float)


def test_predicted_label_outside_allowed_values_rejected():
    """In prediction stream, predicted_label not in allowed list is rejected with BAD_VALUE."""
    row = {
        "record_id": "REC-00001",
        "timestamp_utc": "2026-09-10T07:26:00Z",
        "site_id": "SITE-A",
        "predicted_label": "UNKNOWN_FAULT",
    }
    res = validate_record("prediction", row)
    assert not res.success
    assert res.reject_code == REJECT_BAD_VALUE


def test_predicted_label_allowed_value_accepted():
    """In prediction stream, valid predicted_label is accepted."""
    row = {
        "record_id": "REC-00001",
        "timestamp_utc": "2026-09-10T07:26:00Z",
        "site_id": "SITE-A",
        "predicted_label": "PARTIAL_SHADING",
        "confidence": 0.95,
    }
    res = validate_record("prediction", row)
    assert res.success
    assert res.point.fields["predicted_label"] == "PARTIAL_SHADING"
    assert res.point.fields["confidence"] == 0.95


def test_strict_mode_drops_unknown_columns_and_counts_them():
    """In strict mode (readings), unknown columns and dataset metadata are dropped and tracked."""
    row = valid_readings_row()
    row["window"] = "OPERATION"
    row["run_id"] = "RUN-1"
    row["unknown_extra_col"] = "some_value"
    res = validate_record("readings", row)
    assert res.success
    assert "window" not in res.point.fields
    assert "run_id" not in res.point.fields
    assert "unknown_extra_col" not in res.point.fields
    assert "window" in res.dropped_columns
    assert "run_id" in res.dropped_columns
    assert "unknown_extra_col" in res.dropped_columns


def test_open_mode_accepts_new_numeric_field_as_float_and_rejects_list():
    """In open mode (health), new numeric field becomes float, and lists/dicts are rejected with BAD_TYPE."""
    # New valid numeric field
    row_valid = {
        "record_id": "REC-00001",
        "timestamp_utc": "2026-09-10T07:26:00Z",
        "site_id": "SITE-A",
        "health_score": 95.0,
        "new_metric_val": 42,  # int becomes float in open mode
    }
    res_valid = validate_record("health", row_valid)
    assert res_valid.success
    assert isinstance(res_valid.point.fields["new_metric_val"], float)
    assert res_valid.point.fields["new_metric_val"] == 42.0

    # Complex type (list) must be rejected
    row_invalid = {
        "record_id": "REC-00001",
        "timestamp_utc": "2026-09-10T07:26:00Z",
        "site_id": "SITE-A",
        "invalid_list_field": [1, 2, 3],
    }
    res_invalid = validate_record("health", row_invalid)
    assert not res_invalid.success
    assert res_invalid.reject_code == REJECT_BAD_TYPE


def test_bad_field_name_in_open_mode():
    """Field name not matching ^[a-z][a-z0-9_]*$ in open mode is rejected with BAD_FIELD_NAME."""
    row = {
        "record_id": "REC-00001",
        "timestamp_utc": "2026-09-10T07:26:00Z",
        "site_id": "SITE-A",
        "InvalidFieldName!": 12.3,
    }
    res = validate_record("health", row)
    assert not res.success
    assert res.reject_code == REJECT_BAD_FIELD_NAME


def test_site_device_timestamp_never_in_fields():
    """site_id, device_id, timestamp_utc must be tags / time, NEVER stored in fields."""
    row = valid_readings_row()
    row["device_id"] = "ESP32-A01"
    row["protocol"] = "WIFI"
    res = validate_record("readings", row)
    assert res.success
    assert "site_id" not in res.point.fields
    assert "device_id" not in res.point.fields
    assert "protocol" not in res.point.fields
    assert "timestamp_utc" not in res.point.fields
    # record_id IS stored as a string field
    assert res.point.fields["record_id"] == "REC-00001"
    # Tags check
    assert res.point.tags == {
        "site_id": "SITE-A",
        "device_id": "ESP32-A01",
        "protocol": "WIFI",
    }


def test_empty_point_rejected():
    """If all fields are missing or omitted, point is rejected with EMPTY_POINT."""
    row = {
        "record_id": "",  # missing required record_id will trigger missing required
        "timestamp_utc": "2026-09-10T07:26:00Z",
        "site_id": "SITE-A",
    }
    # For twin stream: requires only timestamp_utc and site_id, but has no fields
    res = validate_record("twin", row)
    assert not res.success
    assert res.reject_code == REJECT_EMPTY_POINT


def test_reserved_measurement_refused():
    """Streams resulting in reserved measurement names (even via prefix) must be refused."""
    # Temporarily set MEASUREMENT_PREFIX such that final measurement collides with reserved
    os.environ["MEASUREMENT_PREFIX"] = "fl_global_"
    load_schema()
    # If base is 'models' -> fl_global_models which is reserved
    # Let's test with a direct match by setting prefix to produce network_link_metrics
    os.environ["MEASUREMENT_PREFIX"] = "network_"
    load_schema()
    # sensor_readings -> network_sensor_readings (not reserved)
    # But if prefix creates an exact reserved measurement name:
    # Say prefix + sensor_readings == network_link_metrics
    # Let's verify with stream config
    assert not validate_record("readings", valid_readings_row()).reject_code == REJECT_RESERVED_MEASUREMENT

    # Now make it collide:
    # If final name is 'network_link_metrics':
    # Since base name is 'sensor_readings', let's set prefix that matches a reserved name
    # Or let's test is_measurement_reserved directly and via validate_record:
    # If we set MEASUREMENT_PREFIX to something that when combined with a stream makes a reserved name:
    from backend.config import is_measurement_reserved
    assert is_measurement_reserved("network_link_metrics")
    assert is_measurement_reserved("ota_deployments")
    assert not is_measurement_reserved("sensor_readings")


def test_reserved_measurement_via_prefix_rejected():
    """Prefix causing a final measurement to match a reserved measurement triggers RESERVED_MEASUREMENT."""
    # Suppose a stream's base is sensor_readings. If prefix makes final name == network_link_metrics
    # To test validate_record returning REJECT_RESERVED_MEASUREMENT:
    # We can temporarily patch get_final_measurement in validate module or test the prefix
    import backend.validate as val_mod
    orig_get_final = val_mod.get_final_measurement
    try:
        val_mod.get_final_measurement = lambda s: "network_link_metrics"
        res = val_mod.validate_record("readings", valid_readings_row())
        assert not res.success
        assert res.reject_code == REJECT_RESERVED_MEASUREMENT
    finally:
        val_mod.get_final_measurement = orig_get_final


def test_prefix_applied():
    """Environment variable MEASUREMENT_PREFIX is applied to the final measurement name."""
    os.environ["MEASUREMENT_PREFIX"] = "test_"
    load_schema()
    assert get_final_measurement("readings") == "test_sensor_readings"
    res = validate_record("readings", valid_readings_row())
    assert res.success
    assert res.point.measurement == "test_sensor_readings"

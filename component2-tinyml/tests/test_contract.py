from tinyml.contract import (
    FL_RAW_FEATURES, LABELS, LEAKAGE_COLUMNS, NEVER_MODEL_INPUTS, SENSOR_COLUMNS,
    classify_columns,
)

# Header of pv_fault_dataset.csv (component3-federated-backend/data/shared, commit a1b36f0)
PV_FAULT_HEADER = (
    "record_id,timestamp_utc,site_id,device_id,panel_id,window,run_id,reading,"
    "voltage_v,current_ma,power_mw,panel_temp1_c,panel_temp2_c,ambient_temp_c,"
    "humidity_pct,pressure_hpa,irradiance_lux,dust_vo_v,health_score,label"
).split(",")


def test_labels_match_backend_allowed_values():
    assert LABELS == ("NORMAL", "PARTIAL_SHADING", "DUST", "OPEN_CIRCUIT", "SHORT_CIRCUIT")


def test_fl_features_are_sensor_columns():
    assert set(FL_RAW_FEATURES) <= set(SENSOR_COLUMNS)


def test_no_sensor_column_is_forbidden_as_input():
    assert not set(SENSOR_COLUMNS) & set(NEVER_MODEL_INPUTS)


def test_health_score_is_flagged_as_leakage():
    assert "health_score" in LEAKAGE_COLUMNS
    assert "health_score" in NEVER_MODEL_INPUTS


def test_pv_fault_header_matches_contract():
    roles = classify_columns(PV_FAULT_HEADER)
    assert roles["missing"] == []
    assert roles["unknown"] == []
    assert roles["sensor"] == list(SENSOR_COLUMNS)
    assert roles["leakage"] == ["health_score"]


def test_unknown_and_missing_columns_are_reported():
    header = [c for c in PV_FAULT_HEADER if c != "dust_vo_v"] + ["wind_ms"]
    roles = classify_columns(header)
    assert roles["missing"] == ["dust_vo_v"]
    assert roles["unknown"] == ["wind_ms"]

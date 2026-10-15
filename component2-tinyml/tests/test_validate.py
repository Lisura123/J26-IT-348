import pandas as pd
import pytest

from tinyml.contract import LABELS
from tinyml.validate import (
    assert_not_predictions, check_power, compare_part, label_for_part_file, validate_dataset,
)

CODES = {"NORMAL": "NOR", "PARTIAL_SHADING": "PAR", "DUST": "DUS",
         "OPEN_CIRCUIT": "OPE", "SHORT_CIRCUIT": "SHO"}


def make_dataset(runs_per_label=2, readings=4):
    """Small dataset that passes every check: 5 labels, 5 s spacing, P = V*I."""
    rows, rec = [], 0
    start = pd.Timestamp("2026-09-01T07:00:00Z")
    for li, label in enumerate(LABELS):
        for r in range(runs_per_label):
            site = "SITE-A" if r % 2 == 0 else "SITE-B"
            run_start = start + pd.Timedelta(hours=li * 10 + r)
            for k in range(readings):
                rec += 1
                v, i = 2.5 + 0.01 * k, 40.0 + li + k
                rows.append({
                    "record_id": f"REC-{rec:05d}",
                    "timestamp_utc": (run_start + pd.Timedelta(seconds=5 * k)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "site_id": site, "device_id": f"ESP32-{site[-1]}01", "panel_id": f"PV-{site[-1]}01",
                    "window": "TRAINING" if r == 0 else "OPERATION",
                    "run_id": f"{site}-{CODES[label]}-{r + 1:02d}", "reading": k + 1,
                    "voltage_v": v, "current_ma": i, "power_mw": round(v * i, 2),
                    "panel_temp1_c": 35.0 + k, "panel_temp2_c": 34.5 + k, "ambient_temp_c": 28.0,
                    "humidity_pct": 80.0, "pressure_hpa": 1009.0, "irradiance_lux": 40000.0 + 10 * k,
                    "dust_vo_v": 0.9 if label == "DUST" else 0.1,
                    "health_score": 90.0 if label == "NORMAL" else 10.0 * li,
                    "label": label,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def df():
    return make_dataset()


def test_valid_dataset_has_no_errors(df):
    r = validate_dataset(df)
    assert r["errors"] == []
    assert r["labels"]["absent"] == []
    assert r["runs"]["runs"] == 10
    assert r["timestamps"]["intervals_off_expected"] == 0
    assert r["power"]["rows_over_tol"] == 0


def test_valid_dataset_still_reports_leakage(df):
    r = validate_dataset(df)
    assert "health_score" in r["leakage"]["leakage_columns_present"]
    assert any("health_score" in w for w in r["warnings"])
    assert r["runs"]["run_id_encodes_label_share"] == 1.0


def test_missing_column_is_an_error(df):
    r = validate_dataset(df.drop(columns=["dust_vo_v"]))
    assert any("dust_vo_v" in e for e in r["errors"])


def test_invalid_label_is_an_error(df):
    df.loc[df.index[:4], "label"] = "SOILING"
    r = validate_dataset(df)
    assert any("SOILING" in e for e in r["errors"])


def test_absent_label_is_an_error(df):
    r = validate_dataset(df[df.label != "DUST"])
    assert any("DUST" in e and "no rows" in e for e in r["errors"])


def test_duplicate_records_are_errors(df):
    r = validate_dataset(pd.concat([df, df.iloc[[0]]], ignore_index=True))
    assert r["duplicates"]["record_id"] == 1
    assert r["duplicates"]["device_timestamp"] == 1
    assert any("duplicate record_id" in e for e in r["errors"])


def test_missing_sensor_value_is_an_error(df):
    df.loc[3, "current_ma"] = float("nan")
    r = validate_dataset(df)
    assert r["missing"]["nan_by_column"] == {"current_ma": 1}
    assert any("missing values" in e for e in r["errors"])


def test_temperature_sentinel_is_reported(df):
    df.loc[0, "panel_temp1_c"] = -127.0
    r = validate_dataset(df)
    assert r["missing"]["temp_sentinels"] == {"panel_temp1_c": 1}
    assert r["ranges"]["panel_temp1_c"]["below"] == 0      # sentinel excluded from range check


def test_wrong_timestamp_format_is_an_error(df):
    df.loc[0, "timestamp_utc"] = "2026-09-01 07:00:00"
    r = validate_dataset(df)
    assert any("not UTC" in e for e in r["errors"])


def test_out_of_order_timestamp_is_an_error(df):
    df.loc[1, "timestamp_utc"] = df.loc[0, "timestamp_utc"]   # reading 2 not after reading 1
    r = validate_dataset(df)
    assert df.loc[0, "run_id"] in r["timestamps"]["non_increasing_runs"]
    assert any("not increasing" in e for e in r["errors"])


def test_irregular_interval_is_a_warning(df):
    run = df.run_id == df.loc[0, "run_id"]
    t = pd.to_datetime(df.loc[run, "timestamp_utc"]) + pd.Timedelta(seconds=7)
    df.loc[df.index[run][-1], "timestamp_utc"] = t.iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ")
    r = validate_dataset(df)
    assert r["timestamps"]["intervals_off_expected"] == 1
    assert r["errors"] == []
    assert any("intervals differ" in w for w in r["warnings"])


def test_inconsistent_run_label_is_an_error(df):
    df.loc[0, "label"] = "DUST"
    r = validate_dataset(df)
    assert df.loc[0, "run_id"] in r["runs"]["mixed_label_runs"]
    assert any("mixed_label_runs" in e for e in r["errors"])


def test_incorrect_power_is_an_error(df):
    df.loc[5, "power_mw"] += 10.0
    assert check_power(df)["rows_over_tol"] == 1
    r = validate_dataset(df)
    assert any("power_mw != voltage_v * current_ma" in e for e in r["errors"])


def test_out_of_range_voltage_and_saturation_are_warnings(df):
    df.loc[0, "voltage_v"] = 5.0
    df.loc[0, "power_mw"] = round(5.0 * df.loc[0, "current_ma"], 2)
    df.loc[1, "irradiance_lux"] = 100000.0
    r = validate_dataset(df)
    assert r["ranges"]["voltage_v"]["above"] == 1
    assert r["ranges"]["irradiance_lux"]["at_saturation"] == 1
    assert r["ranges"]["irradiance_lux"]["above_bh1750_max"] == 1


def test_prediction_file_is_rejected(df):
    preds = df[["record_id", "timestamp_utc", "site_id"]].assign(predicted_fault="NORMAL",
                                                                 confidence=0.9)
    with pytest.raises(ValueError, match="never ground-truth"):
        assert_not_predictions(preds)
    with pytest.raises(ValueError):
        validate_dataset(preds.assign(label="NORMAL"))


def test_prediction_file_is_described_not_used(df):
    preds = df[["record_id"]].assign(predicted_fault=df.label.where(df.index % 2 == 0, "NORMAL"))
    r = validate_dataset(df, predictions=preds)
    assert r["predictions"]["rows"] == len(df)
    assert 0 < r["predictions"]["agreement_with_labels"] < 1


def test_exact_part_file_matches(df):
    part = df[df.label == "DUST"].copy()
    p = compare_part(df, part, "DUST")
    assert p["exact_subset"] and p["identical_to_combined_slice"]


def test_part_file_with_changed_value_fails(df):
    part = df[df.label == "DUST"].copy()
    part.iloc[0, part.columns.get_loc("voltage_v")] += 0.1
    p = compare_part(df, part, "DUST")
    assert not p["exact_subset"]
    assert p["columns_with_mismatch"] == ["voltage_v"]
    r = validate_dataset(df, parts={"DUST": part})
    assert any("DUST" in e and "not an exact subset" in e for e in r["errors"])


def test_part_file_with_foreign_rows_and_wrong_label(df):
    part = df[df.label == "DUST"].copy()
    extra = part.iloc[[0]].assign(record_id="REC-99999")
    p = compare_part(df, pd.concat([part, df[df.label == "NORMAL"].iloc[[0]], extra]), "DUST")
    assert p["not_in_combined"] == 1
    assert p["wrong_label_rows"] == 1
    assert not p["exact_subset"]


def test_cross_site_overlap_is_a_provenance_warning(df):
    a = df.run_id == df.run_id.unique()[0]
    b = df.run_id == df.run_id.unique()[1]          # other site
    df.loc[b, "timestamp_utc"] = df.loc[a, "timestamp_utc"].values
    r = validate_dataset(df)
    assert r["overlaps"]["cross_site"]
    assert any("overlap in time" in w for w in r["provenance"])


def test_provenance_note_is_always_present(df):
    r = validate_dataset(df)
    assert "UNRESOLVED" in r["provenance"][0]


def test_unknown_site_ids_are_errors(df):
    sites = pd.DataFrame({"site_id": ["SITE-A"], "device_id": ["ESP32-A01"], "panel_id": ["PV-A01"]})
    r = validate_dataset(df, sites=sites)
    assert r["unknown_ids"]["site_id"] == ["SITE-B"]
    assert any("sites.csv" in e for e in r["errors"])


@pytest.mark.parametrize("name,label", [
    ("pv_dust.csv", "DUST"), ("44761136-pv_dust.csv", "DUST"),
    ("pv_partial_shading.csv", "PARTIAL_SHADING"), ("pv_fault_dataset.csv", None),
    ("member2_fault_predictions.csv", None),
])
def test_part_file_names(name, label):
    assert label_for_part_file(name) == label

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"


@lru_cache
def users() -> pd.DataFrame:
    return pd.read_csv(DATA / "dashboard_users.csv")


@lru_cache
def permissions() -> dict[str, set[str]]:
    """{'ADMIN': {'view_shap', ...}, 'MANAGER': {...}} - only permissions with allowed=True."""
    p = pd.read_csv(DATA / "roles_permissions.csv")
    p = p[p.allowed.astype(str).str.lower() == "true"]
    return p.groupby("role").permission.apply(set).to_dict()




#  twin outputs
def _ts(df: pd.DataFrame, col="timestamp_utc") -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], utc=True)
    return df


@lru_cache
def sites() -> pd.DataFrame:
    return pd.read_csv(DATA / "sites.csv")


@lru_cache
def twin_output() -> pd.DataFrame:
    """One row per live reading: actual vs expected, band, physics_cause, explanation.

    digital_twin_output.csv (data contract) holds only the twin's own columns, so the
    actual sensor values are joined back from the input on record_id.
    """
    out = pd.read_csv(OUT / "digital_twin_output.csv")
    actual = pd.read_csv(DATA / "digital_twin_input.csv",
                         usecols=["record_id", "voltage_v", "current_ma", "power_mw", "dust_vo_v"])
    return _ts(out.merge(actual, on="record_id", how="left")).sort_values("timestamp_utc")


@lru_cache
def model_status() -> pd.DataFrame:
    """Latest TinyML model / link status per site (Members 2 + 3, from the joined input)."""
    cols = ["site_id", "timestamp_utc", "model_version", "latest_published_version",
            "model_up_to_date", "last_ota_status", "protocol", "link_quality"]
    d = _ts(pd.read_csv(DATA / "digital_twin_input.csv", usecols=cols))
    return d.sort_values("timestamp_utc").groupby("site_id").tail(1).set_index("site_id")


@lru_cache
def alerts() -> pd.DataFrame:
    return pd.read_csv(OUT / "alerts.csv")


@lru_cache
def shap_values() -> pd.DataFrame:
    return pd.read_csv(OUT / "shap_values.csv").set_index("record_id")


@lru_cache
def metrics() -> dict:
    """Headline evaluation numbers for the reports page."""
    import json
    read = lambda name: json.loads((OUT / name).read_text())   # noqa: E731
    cal, val = read("twin_calibration.json"), read("validation_metrics.json")
    con, shp = read("conformal_report.json"), read("shap_vs_member2.json")
    return {
        "thermal": {"u0": cal["faiman_u0"], "u1": cal["faiman_u1"],
                    "rmse_with_wind_c": val["temp_with_wind"]["rmse"],
                    "rmse_no_wind_c": val["temp_no_wind"]["rmse"],
                    "wind_improvement_pct": val["temp_improvement_from_wind_pct"]},
        "power": {"rmse_mw": val["power_normal_all"]["rmse"],
                  "mape_pct": val["power_normal_all"]["mape_pct"],
                  "r2": val["power_normal_all"]["r2"],
                  "ablation_rmse_mw": val["ablation_true_inputs"]["rmse"]},
        "conformal": {"target": 1 - con["alpha"],
                      "random_split_coverage": con["random_split"]["test_coverage"],
                      "temporal_split_coverage": con["temporal_split"]["test_coverage"]},
        "shap_vs_member2": shp,
    }
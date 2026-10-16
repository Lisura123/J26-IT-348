import math

import pandas as pd
import requests

TWIN_FIELDS = [
    "record_id", "irradiance_wm2_est", "irradiance_source", "wind_speed_ms_used",
    "expected_power_mw", "expected_voltage_v", "expected_current_ma", "expected_voc_v",
    "expected_isc_ma", "expected_panel_temp_c", "voltage_residual_pct", "current_residual_pct",
    "power_residual_pct", "panel_temp_residual_c", "panel_temp_spread_c",
    "power_band_low_mw", "power_band_high_mw", "within_expected_band",
    "physics_cause", "ml_physics_agree", "dashboard_explanation",
]
BATCH = 500


def to_records(out: pd.DataFrame) -> list[dict]:
    cols = ["timestamp_utc", "site_id", "device_id"] + TWIN_FIELDS
    records = []
    for row in out[cols].to_dict(orient="records"):
        records.append({k: v for k, v in row.items() if not (isinstance(v, float) and math.isnan(v))})
    return records


def publish(out: pd.DataFrame, base_url: str, api_key: str, session=None) -> dict:
    http = session or requests.Session()
    total = {"sent": 0, "accepted": 0, "rejected": 0, "written": 0, "write_failed": 0, "reasons": {}}
    records = to_records(out)
    for i in range(0, len(records), BATCH):
        batch = records[i:i + BATCH]
        r = http.post(f"{base_url}/api/v1/ingest/twin", json=batch,
                      headers={"X-API-Key": api_key}, timeout=120)
        r.raise_for_status()
        res = r.json()
        total["sent"] += len(batch)
        for k in ("accepted", "rejected", "written", "write_failed"):
            total[k] += res.get(k, 0)
        for code, n in res.get("reasons", {}).items():
            total["reasons"][code] = total["reasons"].get(code, 0) + n
    return total
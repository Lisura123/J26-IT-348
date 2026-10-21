import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from twin import device_health as dh

from . import auth
from . import data_source as ds

app = FastAPI(title="Member 4 - Digital Twin API", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:4200"],
                   allow_methods=["*"], allow_headers=["*"])

PRIORITY_STATUS = {"CRITICAL": "FAULT", "HIGH": "FAULT", "MEDIUM": "WARNING"}


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def has(user: dict, permission: str) -> bool:
    return permission in ds.permissions().get(user["role"], set())


class LoginRequest(BaseModel):
    email: str
    password: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/login")
def login(body: LoginRequest):
    return auth.login(body.email, body.password)


@app.get("/auth/me")
def me(user: dict = Depends(auth.current_user)):
    return auth.public_user(user)


@app.get("/sites")
def list_sites(user: dict = Depends(auth.require("view_live_dashboard"))):
    out, al, ms = ds.twin_output(), ds.alerts(), ds.model_status()
    latest = out.groupby("site_id").tail(1).set_index("site_id")
    result = []
    for _, s in ds.sites().iterrows():
        site_alerts = al[al.site_id == s.site_id]
        worst = next((p for p in ("CRITICAL", "HIGH", "MEDIUM") if (site_alerts.priority == p).any()), None)
        row = {"site_id": s.site_id, "panel_id": s.panel_id, "display_name": s.display_name,
               "protocol": s.protocol, "status": PRIORITY_STATUS.get(worst, "OK"),
               "open_alerts": int(len(site_alerts)), "worst_priority": worst}
        if s.site_id in latest.index:
            r = latest.loc[s.site_id]
            row["latest"] = {k: r[k] for k in ("timestamp_utc", "power_mw", "expected_power_mw",
                                               "power_band_low_mw", "power_band_high_mw",
                                               "physics_cause", "within_expected_band")}
        if has(user, "view_model_status") and s.site_id in ms.index:
            m = ms.loc[s.site_id]
            row["model"] = {k: m[k] for k in ("model_version", "latest_published_version",
                                              "model_up_to_date", "link_quality")}
        result.append(row)
    return clean(result)


@app.get("/sites/{site_id}/timeseries")
def site_timeseries(site_id: str, user: dict = Depends(auth.require("view_live_dashboard"))):
    out = ds.twin_output()
    d = out[out.site_id == site_id]
    if d.empty:
        raise HTTPException(404, f"No readings for {site_id}")
    cols = ["record_id", "timestamp_utc", "power_mw", "expected_power_mw", "power_band_low_mw",
            "power_band_high_mw", "within_expected_band", "physics_cause", "irradiance_wm2_est"]
    return clean(d[cols].to_dict(orient="records"))


@app.get("/alerts")
def list_alerts(user: dict = Depends(auth.require("view_live_dashboard"))):
    cols = ["alert_id", "site_id", "physics_cause", "priority", "start_utc", "end_utc",
            "n_readings", "mean_power_loss_pct", "rule_id", "recommended_action",
            "plain_summary", "plain_action",
            "assigned_role", "respond_by_utc", "tinyml_majority", "tinyml_agree_frac"]
    return clean(ds.alerts()[cols].to_dict(orient="records"))


@app.get("/alerts/{alert_id}")
def alert_detail(alert_id: str, user: dict = Depends(auth.require("view_explanations"))):
    al = ds.alerts()
    match = al[al.alert_id == alert_id]
    if match.empty:
        raise HTTPException(404, f"Unknown alert {alert_id}")
    alert = match.iloc[0].to_dict()
    alert["shap"] = shap_for(alert["first_record_id"]) if has(user, "view_shap") else None
    alert["shap_visible"] = has(user, "view_shap")
    return clean(alert)


@app.get("/readings/{record_id}/shap")
def reading_shap(record_id: str, user: dict = Depends(auth.require("view_shap"))):
    return clean(shap_for(record_id))


def shap_for(record_id: str) -> dict | None:
    sv = ds.shap_values()
    if record_id not in sv.index:
        return None
    r = sv.loc[record_id]
    feats = {c.removeprefix("shap_"): r[c] for c in sv.columns if c.startswith("shap_")}
    return {"record_id": record_id, "model_version": r.model_version,
            "predicted_fault": r.predicted_fault, "confidence": r.confidence,
            "base_value": r.base_value,
            "features": sorted(({"feature": k, "value": v} for k, v in feats.items()),
                               key=lambda f: -abs(f["value"]))}


@app.get("/metrics")
def metrics(user: dict = Depends(auth.require("view_reports"))):
    return clean(ds.metrics())


VALUES = ["voltage_v", "current_ma", "power_mw", "irradiance_lux", "panel_temp1_c",
          "ambient_temp_c", "dust_lpo_pct", "health_score"]


@app.get("/devices/{site_id}/health")
def device_health(site_id: str, limit: int = 500,
                  user: dict = Depends(auth.require("log_maintenance"))):
    try:
        docs = ds.device_health(site_id, min(limit, 2000))
    except Exception as e:
        raise HTTPException(503, f"Device health database unavailable ({type(e).__name__})")
    if not docs:
        raise HTTPException(404, f"No device health data for {site_id} in {ds.HEALTH_COLLECTION}")

    latest = docs[0]
    counts, last_seen = Counter(), {}
    for d in docs:
        for code in dh.split_flags(d.get("quality_flag")):
            if code != "OK":
                counts[code] += 1
                last_seen.setdefault(code, d["timestamp_utc"])
    issues = [{**dh.explain_flag(c), "count": n, "last_seen_utc": last_seen[c]} for c, n in counts.items()]
    issues.sort(key=lambda i: (-dh.RANK.get(i["severity"], 0), -i["count"]))

    return clean({
        "site_id": site_id,
        "device_id": latest.get("device_id"),
        "collection": ds.HEALTH_COLLECTION,
        "n_readings": len(docs),
        "from_utc": docs[-1]["timestamp_utc"],
        "to_utc": latest["timestamp_utc"],
        "latest": {"timestamp_utc": latest["timestamp_utc"], "record_id": latest.get("record_id"),
                   "values": {k: latest.get(k) for k in VALUES if k in latest},
                   **dh.explain_reading(latest)},
        "status_counts": dict(Counter(d.get("health_status") or "MISSING" for d in docs)),
        "issues": issues,
    })
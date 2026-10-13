import math

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import auth
from . import data_source as ds

app = FastAPI(title="Member 4 - Digital Twin API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:4200"],
                   allow_methods=["*"], allow_headers=["*"])

PRIORITY_STATUS = {"CRITICAL": "FAULT", "HIGH": "FAULT", "MEDIUM": "WARNING"}


def clean(value):
    """Make pandas/numpy values JSON-safe (NaN -> None, Timestamp -> ISO string)."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):          # numpy scalar -> python scalar
        return clean(value.item())
    return value


def has(user: dict, permission: str) -> bool:
    return permission in ds.permissions().get(user["role"], set())


class LoginRequest(BaseModel):
    email: str
    password: str


#  auth
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/login")
def login(body: LoginRequest):
    return auth.login(body.email, body.password)


@app.get("/auth/me")
def me(user: dict = Depends(auth.current_user)):
    return auth.public_user(user)


#  sites
@app.get("/sites")
def list_sites(user: dict = Depends(auth.require("view_live_dashboard"))):
    """Site overview: latest twin reading, open alerts and model status for all 6 panels."""
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
    """Actual vs twin-expected power with the conformal band, for the panel detail chart."""
    out = ds.twin_output()
    d = out[out.site_id == site_id]
    if d.empty:
        raise HTTPException(404, f"No readings for {site_id}")
    cols = ["record_id", "timestamp_utc", "power_mw", "expected_power_mw", "power_band_low_mw",
            "power_band_high_mw", "within_expected_band", "physics_cause", "irradiance_wm2_est"]
    return clean(d[cols].to_dict(orient="records"))


#  alerts
@app.get("/alerts")
def list_alerts(user: dict = Depends(auth.require("view_live_dashboard"))):
    cols = ["alert_id", "site_id", "physics_cause", "priority", "start_utc", "end_utc",
            "n_readings", "mean_power_loss_pct", "rule_id", "recommended_action",
            "assigned_role", "respond_by_utc", "tinyml_majority", "tinyml_agree_frac"]
    return clean(ds.alerts()[cols].to_dict(orient="records"))


@app.get("/alerts/{alert_id}")
def alert_detail(alert_id: str, user: dict = Depends(auth.require("view_explanations"))):
    """Alert + explanation text; SHAP is attached ONLY for roles with view_shap."""
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


# reports
@app.get("/metrics")
def metrics(user: dict = Depends(auth.require("view_reports"))):
    return clean(ds.metrics())
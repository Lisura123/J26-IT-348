import pandas as pd

from .config import DATA

EPISODE_GAP_S = 60
DUST_LOSS_PCT = 8.0
UNEXPLAINED_MIN_N = 3

CAUSE_TO_RULE = {"SHORT_CIRCUIT": "R01", "OPEN_CIRCUIT": "R02", "PARTIAL_SHADING": "R03",
                 "DUST": "R04", "UNEXPLAINED_LOSS": "R05"}


def load_catalog() -> pd.DataFrame:
    return pd.read_csv(DATA / "maintenance_action_catalog.csv").set_index("rule_id")


def episodes(out: pd.DataFrame) -> pd.DataFrame:
    d = out.copy()
    d["t"] = pd.to_datetime(d.timestamp_utc, utc=True)
    d = d.sort_values(["device_id", "t"])
    new = ((d.device_id != d.device_id.shift())
           | (d.physics_cause != d.physics_cause.shift())
           | ((d.t - d.t.shift()).dt.total_seconds() > EPISODE_GAP_S))
    d["episode"] = new.cumsum()
    return d.groupby("episode").agg(
        site_id=("site_id", "first"), device_id=("device_id", "first"),
        physics_cause=("physics_cause", "first"),
        start_utc=("t", "min"), end_utc=("t", "max"), n_readings=("record_id", "size"),
        first_record_id=("record_id", "first"),
        mean_power_loss_pct=("power_residual_pct", lambda s: -s.mean()),
        tinyml_majority=("predicted_fault", lambda s: s.mode()[0]),
        tinyml_agree_frac=("ml_physics_agree", "mean"),
        explanation=("dashboard_explanation", "first"),
        plain_summary=("plain_summary", "first"),
        plain_action=("plain_action", "first"))


def recommend(out: pd.DataFrame) -> pd.DataFrame:
    cat = load_catalog()
    ep = episodes(out)
    ep = ep[ep.physics_cause != "NORMAL"].copy()
    ep["rule_id"] = ep.physics_cause.map(CAUSE_TO_RULE)
    keep = ~(((ep.rule_id == "R04") & (ep.mean_power_loss_pct <= DUST_LOSS_PCT))
             | ((ep.rule_id == "R05") & (ep.n_readings <= UNEXPLAINED_MIN_N)))
    ep = ep[keep].join(cat, on="rule_id")
    ep["respond_by_utc"] = ep.start_utc + pd.to_timedelta(ep.target_response_h, unit="h")
    prio = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    ep = ep.sort_values(["priority", "start_utc"], key=lambda s: s.map(prio) if s.name == "priority" else s)
    ep.insert(0, "alert_id", [f"ALR-{k:04d}" for k in range(1, len(ep) + 1)])
    for c in ("start_utc", "end_utc", "respond_by_utc"):
        ep[c] = ep[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    ep["mean_power_loss_pct"] = ep.mean_power_loss_pct.round(1)
    ep["tinyml_agree_frac"] = ep.tinyml_agree_frac.round(2)
    return ep.reset_index(drop=True)
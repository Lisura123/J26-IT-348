import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
from twin.config import OUT, CONFORMAL_MIN_G  # noqa: E402

week = pd.Timestamp("2026-09-10", tz="UTC")

lt = pd.read_parquet(OUT / "longterm_twin.parquet")
lt = lt[(lt.timestamp_utc >= week) & (lt.irradiance_wm2_est > CONFORMAL_MIN_G)]
lt["resid_pct"] = 100 * (lt.power_mw - lt.expected_power_mw) / lt.expected_power_mw
history = lt.groupby("site_id").agg(
    longterm_condition=("condition", lambda s: s.mode()[0]),
    longterm_resid_pct=("resid_pct", "mean"))

op = pd.read_csv(OUT / "digital_twin_output.csv")
op = op[op.ground_truth_label == "NORMAL"]
live = op.groupby("site_id").agg(
    op_normal_rows=("record_id", "size"),
    op_resid_pct=("power_residual_pct", "mean"),
    op_inside_band=("within_expected_band", "mean"))

print(history.join(live).round(2).to_string())
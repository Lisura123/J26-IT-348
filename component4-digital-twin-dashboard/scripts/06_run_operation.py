import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from twin import conformal
from twin.causes import classify, explain, plain_text, residuals
from twin.config import DATA, OUT
from twin.core import run_twin
from twin.inputs import prepare

COLUMNS = ["record_id", "timestamp_utc", "site_id", "device_id", "ground_truth_label",
           "predicted_fault", "confidence", "model_version", "relay_status",
           "irradiance_wm2_est", "irradiance_source", "wind_speed_ms_used", "wind_source",
           "expected_voltage_v", "expected_current_ma", "expected_power_mw", "expected_voc_v",
           "expected_isc_ma", "expected_panel_temp_c", "voltage_residual_pct", "current_residual_pct",
           "power_residual_pct", "panel_temp_residual_c", "panel_temp_spread_c",
           "power_band_low_mw", "power_band_high_mw", "within_expected_band", "calibration_row",
           "physics_cause", "ml_physics_agree", "dashboard_explanation",
           "plain_summary", "plain_action"]

inp = pd.read_csv(DATA / "digital_twin_input.csv")
band = conformal.load(OUT / "conformal_band.json")

df = run_twin(prepare(inp))
df = df.join(residuals(df))
df["power_band_low_mw"], df["power_band_high_mw"] = conformal.apply_band(df.expected_power_mw, band)
df["within_expected_band"] = df.power_mw.between(df.power_band_low_mw, df.power_band_high_mw)
df["calibration_row"] = False
df["physics_cause"] = classify(df)
df["ml_physics_agree"] = df.predicted_fault == df.physics_cause
df["dashboard_explanation"] = df.apply(explain, axis=1)
df["plain_summary"], df["plain_action"] = zip(*df.apply(plain_text, axis=1))

df["timestamp_utc"] = df.timestamp_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
num = df.select_dtypes("float").columns
df[num] = df[num].round(2)
df[COLUMNS].to_csv(OUT / "digital_twin_output.csv", index=False)

faults = df[df.ground_truth_label != "NORMAL"]
normal = df[df.ground_truth_label == "NORMAL"]
print(pd.crosstab(df.ground_truth_label, df.physics_cause))
print(f"\nFault-cause accuracy : {(faults.ground_truth_label == faults.physics_cause).mean():.4f}")
print(f"NORMAL inside band   : {normal.within_expected_band.mean():.3f}")
print(f"TinyML-physics agree : {df.ml_physics_agree.mean():.4f}")
print("\nExample explanation:\n", df.loc[df.physics_cause != "NORMAL", "dashboard_explanation"].iloc[0])
print("\nPlain version:\n", df.loc[df.physics_cause != "NORMAL", "plain_summary"].iloc[0],
      "\n", df.loc[df.physics_cause != "NORMAL", "plain_action"].iloc[0])
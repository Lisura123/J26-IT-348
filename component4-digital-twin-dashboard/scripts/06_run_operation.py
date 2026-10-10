
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
from twin import conformal  # noqa: E402
from twin.causes import classify, explain, residuals  # noqa: E402
from twin.config import DATA, OUT  # noqa: E402
from twin.core import run_twin  # noqa: E402
from twin.inputs import prepare  # noqa: E402

COLUMNS = ["record_id", "timestamp_utc", "site_id", "device_id", "ground_truth_label",
           "predicted_fault", "confidence", "model_version", "relay_status",
           "irradiance_wm2_est", "irradiance_source", "wind_speed_ms_used", "wind_source",
           "expected_voltage_v", "expected_current_ma", "expected_power_mw", "expected_voc_v",
           "expected_isc_ma", "expected_panel_temp_c", "voltage_residual_pct", "current_residual_pct",
           "power_residual_pct", "panel_temp_residual_c", "panel_temp_spread_c",
           "power_band_low_mw", "power_band_high_mw", "within_expected_band", "calibration_row",
           "physics_cause", "ml_physics_agree", "dashboard_explanation"]

band = conformal.load(OUT / "conformal_band.json")
df = run_twin(prepare(pd.read_csv(DATA / "digital_twin_input.csv")))
df = df.join(residuals(df))
df["power_band_low_mw"], df["power_band_high_mw"] = conformal.apply_band(df.expected_power_mw, band)
df["within_expected_band"] = df.power_mw.between(df.power_band_low_mw, df.power_band_high_mw)
df["calibration_row"] = False                    # live rows are never used for calibration
df["physics_cause"] = classify(df)
df["ml_physics_agree"] = df.predicted_fault == df.physics_cause
df["dashboard_explanation"] = df.apply(explain, axis=1)
df["timestamp_utc"] = df.timestamp_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")   # data contract format
df[df.select_dtypes("float").columns] = df.select_dtypes("float").round(2)
df[COLUMNS].to_csv(OUT / "digital_twin_output.csv", index=False)

print(pd.crosstab(df.ground_truth_label, df.physics_cause).to_string())
faults = df[df.ground_truth_label != "NORMAL"]
print("\nFault-cause accuracy :", round((faults.ground_truth_label == faults.physics_cause).mean(), 4))
print("NORMAL inside band   :", round(df[df.ground_truth_label == "NORMAL"].within_expected_band.mean(), 4))
print("TinyML-physics agree :", round(df.ml_physics_agree.mean(), 4))
print("\nExample explanation:\n", df.dashboard_explanation.iloc[0])
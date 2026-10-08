from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

lt = pd.read_csv(DATA / "member4_longterm_5min.csv", parse_dates=["timestamp_utc"])
print("LONG-TERM rows:", len(lt))
print("Time range:", lt.timestamp_utc.min(), "->", lt.timestamp_utc.max())
print("Readings per site:\n", lt.site_id.value_counts().sort_index())
print("\nColumns:", list(lt.columns))


saturated = (lt.irradiance_lux >= 100_000).sum()
print(f"\nLux sensor saturated: {saturated} rows ({100 * saturated / len(lt):.2f} %)")


for col in ["panel_temp1_c", "panel_temp2_c"]:
    bad = lt[col].isin([-127, -999]).sum()
    print(f"{col}: {bad} error readings")

truth = pd.read_csv(DATA / "member4_longterm_truth.csv")
print("\nTruth conditions:\n", truth.condition.value_counts())

wx = pd.read_csv(DATA / "weather_hourly.csv", parse_dates=["timestamp_utc"])
print("\nWEATHER rows:", len(wx), "| wind range:",
      wx.wind_speed_10m_ms.min(), "-", wx.wind_speed_10m_ms.max(), "m/s")

op = pd.read_csv(DATA / "digital_twin_input.csv")
print("\nOPERATION rows:", len(op))
print("Fault labels:\n", op.ground_truth_label.value_counts())
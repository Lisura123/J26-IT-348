import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
from twin.config import DATA  # noqa: E402
from twin.inputs import prepare  # noqa: E402

lt = prepare(pd.read_csv(DATA / "member4_longterm_5min.csv"))
cols = ["timestamp_utc", "site_id", "irradiance_lux", "irradiance_wm2_est",
        "irradiance_source", "wind_speed_ms_used", "panel_temp_spread_c"]
print(lt[cols].head(3).to_string())
print("\nSaturated rows (should use weather GHI):")
print(lt[lt.irradiance_source.str.startswith("WEATHER")][cols].head(3).to_string())
print("\n", lt[["irradiance_wm2_est", "wind_speed_ms_used", "panel_temp_spread_c"]].describe().round(2))
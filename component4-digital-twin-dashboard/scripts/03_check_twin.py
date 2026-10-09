import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  
from twin.config import DATA, load_datasheet  
from twin.core import run_twin  
from twin.electrical import expected_operating_point  
from twin.inputs import prepare  

ds = load_datasheet()
tests = pd.DataFrame({"g_wm2": [1000, 500, 1000, 200], "t_cell_c": [25, 25, 50, 25]})
result = expected_operating_point(tests.g_wm2, tests.t_cell_c, ds)
print("Model under fixed conditions:")
print(pd.concat([tests, result], axis=1).round(2).to_string())


lt = prepare(pd.read_csv(DATA / "member4_longterm_5min.csv"))
sample = run_twin(lt[lt.irradiance_wm2_est > 600].head(5))
cols = ["site_id", "irradiance_wm2_est", "ambient_temp_c", "expected_panel_temp_c",
        "panel_temp_mean_c", "power_mw", "expected_power_mw", "voltage_v", "expected_voltage_v"]
print("\nTwin vs actual on sunny readings:")
print(sample[cols].round(2).to_string())
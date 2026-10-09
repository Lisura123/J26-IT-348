import pandas as pd

from .config import load_calibration, load_datasheet
from .electrical import expected_operating_point
from .thermal import faiman


def run_twin(df: pd.DataFrame, ds: dict | None = None, cal: dict | None = None,
             use_wind: bool = True) -> pd.DataFrame:
    
    ds = load_datasheet() if ds is None else ds
    cal = load_calibration() if cal is None else cal
    if use_wind:
        u0, u1 = cal["faiman_u0"], cal["faiman_u1"]
    else:
        u0, u1 = cal["faiman_u0_no_wind"], 0.0

    out = df.copy()
    out["expected_panel_temp_c"] = faiman(out.irradiance_wm2_est, out.ambient_temp_c,
                                          out.wind_speed_ms_used, u0, u1)
    elec = expected_operating_point(out.irradiance_wm2_est, out.expected_panel_temp_c, ds)
    elec.index = out.index
    return out.join(elec)
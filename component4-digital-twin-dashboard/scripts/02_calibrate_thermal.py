"""Step 2 - Fit the Faiman coefficients on the long-term history.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
from twin.config import DATA, OUT, FIT_MIN_G, LUX_SATURATION, LUX_PER_WM2  # noqa: E402
from twin.inputs import prepare  # noqa: E402
from twin.thermal import fit_faiman, faiman  # noqa: E402

lt = prepare(pd.read_csv(DATA / "member4_longterm_5min.csv"))

fit = lt[(lt.irradiance_lux < LUX_SATURATION)
         & (lt.irradiance_wm2_est > FIT_MIN_G)
         & lt.panel_temp_mean_c.notna()]
g, t_amb, wind, t_mod = (fit.irradiance_wm2_est, fit.ambient_temp_c,
                         fit.wind_speed_ms_used, fit.panel_temp_mean_c)

u0, u1, rmse_wind = fit_faiman(g, t_amb, wind, t_mod, use_wind=True)
u0_nw, _, rmse_no_wind = fit_faiman(g, t_amb, wind, t_mod, use_wind=False)
rmse_default = float((((faiman(g, t_amb, wind, 25.0, 6.84) - t_mod) ** 2).mean()) ** 0.5)

cal = {
    "faiman_u0": round(u0, 3),
    "faiman_u1": round(u1, 3),
    "rmse_with_wind_c": round(rmse_wind, 3),
    "faiman_u0_no_wind": round(u0_nw, 3),
    "rmse_no_wind_c": round(rmse_no_wind, 3),
    "rmse_improvement_from_wind_pct": round(100 * (1 - rmse_wind / rmse_no_wind), 1),
    "pvlib_default_u0": 25.0,
    "pvlib_default_u1": 6.84,
    "rmse_pvlib_default_c": round(rmse_default, 3),
    "n_fit_rows": int(len(fit)),
    "fitted_on": f"member4_longterm_5min.csv, unsaturated readings with G>{FIT_MIN_G:g} W/m2 "
                 f"(lux/{LUX_PER_WM2:g}), target = mean of the two panel probes",
    "wind_source": "weather_hourly.csv wind_speed_10m_ms, linearly interpolated (no anemometer)",
}
(OUT / "twin_calibration.json").write_text(json.dumps(cal, indent=1))
print(json.dumps(cal, indent=1))
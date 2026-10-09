
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from twin.config import DATA, OUT, load_calibration, load_datasheet  # noqa: E402
from twin.core import run_twin  # noqa: E402
from twin.electrical import expected_operating_point  # noqa: E402
from twin.inputs import prepare  # noqa: E402
from twin.thermal import faiman  # noqa: E402

P_FLOOR = 20.0   


def metrics(pred, true, mape_floor=None):
    
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    e = pred - true
    m = {"n": int(len(e)),
         "rmse": float(np.sqrt(np.mean(e ** 2))),
         "mae": float(np.mean(np.abs(e))),
         "bias": float(np.mean(e)),
         "r2": float(1 - np.sum(e ** 2) / np.sum((true - true.mean()) ** 2))}
    if mape_floor is not None:
        k = true > mape_floor
        m["mape_pct"] = float(100 * np.mean(np.abs(e[k] / true[k])))
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}


ds, cal = load_datasheet(), load_calibration()
lt = prepare(pd.read_csv(DATA / "member4_longterm_5min.csv"))
truth = pd.read_csv(DATA / "member4_longterm_truth.csv").drop(
    columns=["timestamp_utc", "site_id", "device_id", "panel_id"])
tw = run_twin(lt, ds, cal).merge(truth, on="record_id", how="left")
tw["expected_panel_temp_nowind_c"] = run_twin(lt, ds, cal, use_wind=False)["expected_panel_temp_c"].to_numpy()

normal = tw.condition == "NORMAL"
sunny = tw.irradiance_wm2_est > 200
res = {}

res["power_normal_all"] = metrics(tw.loc[normal, "expected_power_mw"],
                                  tw.loc[normal, "p_mp_new_clean_mw"], P_FLOOR)
res["power_normal_sunny"] = metrics(tw.loc[normal & sunny, "expected_power_mw"],
                                    tw.loc[normal & sunny, "p_mp_new_clean_mw"], P_FLOOR)

true_g = tw.loc[normal, "g_effective_wm2"].clip(lower=0) / (1 - tw.loc[normal, "soiling_loss_frac"]).clip(lower=0.5)
abl = expected_operating_point(true_g, tw.loc[normal, "cell_temp_true_c"], ds)
res["ablation_true_inputs"] = metrics(abl["expected_power_mw"], tw.loc[normal, "p_mp_new_clean_mw"], P_FLOOR)

res["irradiance_sunny"] = metrics(tw.loc[sunny, "irradiance_wm2_est"], tw.loc[sunny, "poa_true_wm2"])

t_true = tw.loc[sunny, "module_temp_true_c"]
res["temp_with_wind"] = metrics(tw.loc[sunny, "expected_panel_temp_c"], t_true)
res["temp_no_wind"] = metrics(tw.loc[sunny, "expected_panel_temp_nowind_c"], t_true)
res["temp_pvlib_default"] = metrics(faiman(tw.loc[sunny, "irradiance_wm2_est"], tw.loc[sunny, "ambient_temp_c"],
                                           tw.loc[sunny, "wind_speed_ms_used"], 25.0, 6.84), t_true)
res["temp_improvement_from_wind_pct"] = round(
    100 * (1 - res["temp_with_wind"]["rmse"] / res["temp_no_wind"]["rmse"]), 1)

res["power_rmse_by_site"] = {
    site: round(float(np.sqrt(((g.expected_power_mw - g.p_mp_new_clean_mw) ** 2).mean())), 2)
    for site, g in tw[normal & sunny].groupby("site_id")}

(OUT / "validation_metrics.json").write_text(json.dumps(res, indent=1))
tw.to_parquet(OUT / "longterm_twin.parquet", index=False)   
print(json.dumps(res, indent=1))
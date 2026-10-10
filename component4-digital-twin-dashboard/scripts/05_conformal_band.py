
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from twin import conformal  # noqa: E402
from twin.config import OUT, CONFORMAL_ALPHA, CONFORMAL_MIN_G  # noqa: E402

tw = pd.read_parquet(OUT / "longterm_twin.parquet")
ok = (tw.condition == "NORMAL") & (tw.irradiance_wm2_est > CONFORMAL_MIN_G) & (tw.expected_power_mw > 0)
d = tw[ok].copy()
d["ratio"] = d.power_mw / d.expected_power_mw
rep = {"alpha": CONFORMAL_ALPHA, "n_normal_rows": int(len(d))}

rng = np.random.default_rng(42)
is_cal = rng.random(len(d)) < 0.5
band_r = conformal.fit_band(d.ratio[is_cal], CONFORMAL_ALPHA)
test = d[~is_cal]
rep["random_split"] = {**band_r, "n_test": int(len(test)),
                       "test_coverage": round(conformal.coverage(test.power_mw, test.expected_power_mw, band_r), 4)}
rep["random_split_coverage_by_site"] = {
    site: round(conformal.coverage(g.power_mw, g.expected_power_mw, band_r), 4)
    for site, g in test.groupby("site_id")}

ts = pd.to_datetime(d.timestamp_utc, utc=True)
early = ts < pd.Timestamp("2026-07-01", tz="UTC")
band_t = conformal.fit_band(d.ratio[early], CONFORMAL_ALPHA)
late = d[~early]
rep["temporal_split"] = {**band_t, "n_test": int(len(late)),
                         "test_coverage": round(conformal.coverage(late.power_mw, late.expected_power_mw, band_t), 4)}

band = conformal.fit_band(d.ratio, CONFORMAL_ALPHA)
band["calibrated_on"] = f"member4_longterm_5min.csv NORMAL rows, G > {CONFORMAL_MIN_G:g} W/m2"
conformal.save(band, OUT / "conformal_band.json")
rep["deployed_band"] = band

(OUT / "conformal_report.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
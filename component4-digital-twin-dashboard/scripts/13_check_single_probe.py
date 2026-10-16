import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from twin import conformal
from twin.causes import classify, residuals
from twin.config import DATA, OUT
from twin.core import run_twin
from twin.inputs import prepare


def causes(inp):
    band = conformal.load(OUT / "conformal_band.json")
    df = run_twin(prepare(inp))
    df = df.join(residuals(df))
    df["power_band_low_mw"], df["power_band_high_mw"] = conformal.apply_band(df.expected_power_mw, band)
    df["within_expected_band"] = df.power_mw.between(df.power_band_low_mw, df.power_band_high_mw)
    df["physics_cause"] = classify(df)
    return df


inp = pd.read_csv(DATA / "digital_twin_input.csv")
one = inp.copy()
one["panel_temp2_c"] = np.nan

for name, data in [("Two panel probes", inp), ("One panel probe (Member 1 hardware)", one)]:
    d = causes(data)
    f = d[d.ground_truth_label != "NORMAL"]
    n = d[d.ground_truth_label == "NORMAL"]
    print(f"\n=== {name} ===")
    print(f"Fault-cause accuracy : {(f.ground_truth_label == f.physics_cause).mean():.4f}")
    print(f"NORMAL -> PARTIAL_SHADING : {(n.physics_cause == 'PARTIAL_SHADING').sum()}"
          f"  {n[n.physics_cause == 'PARTIAL_SHADING'].site_id.value_counts().to_dict()}")
    print(pd.crosstab(d.ground_truth_label, d.physics_cause))
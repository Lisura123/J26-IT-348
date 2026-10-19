"""Check the PV feature layer on the TRAINING window and write
results/feature_validation.md.

Usage:
    python scripts/03_validate_features.py --data-dir data/raw

Only TRAINING runs are used (the OPERATION window stays untouched until the
final test). For every Phase 4 CV fold the feature pipeline is fitted on the
fold's training runs and applied, without refitting, to its validation runs.
No model is trained.
"""
# ---- Pipeline step 3 of 8: check the pv8-v1 feature layer (no model is trained) ----
# When:   after step 2, and again whenever features.py changes
# Reads:  pv_fault_dataset.csv (TRAINING rows only), results/splits_runwise_v1.json
# Writes: results/feature_validation.md
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyml.contract import LABELS  # noqa: E402
from tinyml.features import (  # noqa: E402
    FEATURES, FEATURE_UNITS, FeatureConfig, fold_matrices, input_status, physical_features,
)
from tinyml.splits import iter_folds, load_assignments  # noqa: E402
from tinyml.validate import file_sha256  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--splits", type=Path, default=ROOT / "results" / "splits_runwise_v1.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "feature_validation.md")
    a = ap.parse_args()

    # 1) Load the dataset and the saved run-wise splits from step 2.
    path = next((p for p in sorted(a.data_dir.glob("*.csv"))
                 if p.name == "pv_fault_dataset.csv" or p.name.endswith("-pv_fault_dataset.csv")), None)
    if path is None:
        sys.exit(f"pv_fault_dataset.csv not found in {a.data_dir}")
    full = pd.read_csv(path)
    assignments, meta = load_assignments(a.splits, full, dataset_sha256=file_sha256(path))
    # 2) Keep TRAINING rows only; OPERATION stays untouched for the final test.
    df = full[full.window == "TRAINING"]               # OPERATION is not used here

    # 3) Quality gates: count OK / LOW_LIGHT / LUX_SATURATED / INVALID_INPUT rows per class.
    base = FeatureConfig()
    status = input_status(df, base)
    L = ["# Feature layer check (TRAINING window only)", "",
         f"Feature version `{base.version}`. Splits `{meta['split_version']}` (seed {meta['seed']}). "
         "The OPERATION window was not read for this report. No model was trained.", "",
         "## Input status on TRAINING rows", "",
         "| Label | OK | LOW_LIGHT | LUX_SATURATED | INVALID_INPUT |", "|---|---|---|---|---|"]
    tab = pd.crosstab(df.label, status).reindex(index=list(LABELS), fill_value=0)
    for lab, row in tab.iterrows():
        L.append(f"| {lab} | " + " | ".join(str(int(row.get(s, 0))) for s in
                                           ("OK", "LOW_LIGHT", "LUX_SATURATED", "INVALID_INPUT")) + " |")
    L += ["", f"Gated rows are kept in the dataset and excluded from model inputs "
              f"(low-light threshold {base.low_light_lux:g} lx, saturation {base.lux_saturation:g} lx; "
              "both placeholders).", "",
          "## Per-fold fit (fit on fold-train runs, applied to fold-validation runs)", "",
          "| Fold | V_ref a (V) | b (V/ln klx) | c (V/°C) | Train OK rows | Val OK rows | Val gated | Finite |",
          "|---|---|---|---|---|---|---|---|"]
    # 4) Per CV fold: fit V_ref and the scaler on the fold's training runs only, apply them to its
    #    validation runs, and check that every feature value is finite.
    oof = []
    configs = {}
    for name, tr, te in iter_folds(df, assignments.loc[sorted(set(df.run_id))], "cv"):
        m = fold_matrices(df, tr, te, fold=name)
        c = m["config"]
        configs[name] = c
        finite = bool(np.isfinite(m["train"]["X"]).all() and np.isfinite(m["test"]["X"]).all())
        gated = int((m["test"]["status"] != "OK").sum())
        L.append(f"| {name} | {c.vref['a']:.4f} | {c.vref['b']:.4f} | {c.vref['c']:.4f} | "
                 f"{len(m['train']['X'])} | {len(m['test']['X'])} | {gated} | {finite} |")
        f = physical_features(df.loc[te], c).dropna()
        oof.append(f.assign(label=df.loc[f.index, "label"]))
    # 5) Median feature value per class: shows which features separate the faults.
    oof = pd.concat(oof)
    L += ["", "## Out-of-fold physical features per class (median, before scaling)", "",
          "Each value comes from the fold where that row was in the validation part.", "",
          "| Label | " + " | ".join(f"{f} ({u})" for f, u in zip(FEATURES, FEATURE_UNITS)) + " |",
          "|---|" + "---|" * len(FEATURES)]
    med = oof.groupby("label")[list(FEATURES)].median().reindex(list(LABELS))
    for lab, row in med.iterrows():
        L.append(f"| {lab} | " + " | ".join(f"{v:.3f}" for v in row) + " |")

    # 6) Worked example: one NORMAL reading, from raw sensor values to scaled model inputs.
    ex_idx = df.index[(status == "OK") & (df.label == "NORMAL")][0]
    name = next(n for n, _, te in iter_folds(df, assignments.loc[sorted(set(df.run_id))], "cv")
                if ex_idx in te)
    c = configs[name]
    raw = df.loc[[ex_idx]]
    phys = physical_features(raw, c).iloc[0]
    z = (phys.to_numpy() - np.asarray(c.scaler["mean"])) / np.asarray(c.scaler["std"])
    L += ["", f"## Example: record `{raw.record_id.iloc[0]}` (validation row of {name})", "",
          "Raw: " + ", ".join(f"{k}={raw[k].iloc[0]}" for k in
                              ("voltage_v", "current_ma", "power_mw", "panel_temp1_c", "panel_temp2_c",
                               "ambient_temp_c", "irradiance_lux", "dust_vo_v")), "",
          "| Feature | Physical value | Unit | Scaled (float32) |", "|---|---|---|---|"]
    for f, u, v, s in zip(FEATURES, FEATURE_UNITS, phys, z.astype(np.float32)):
        L.append(f"| {f} | {v:.4f} | {u} | {s:.4f} |")
    L += ["", "These are descriptive checks only. Whether the features improve classification "
              "is evaluated in the baseline comparison (Phase 7), not here.", ""]
    # 7) Write the report.
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

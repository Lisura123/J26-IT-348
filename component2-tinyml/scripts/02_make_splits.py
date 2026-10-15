"""Make run-wise split assignments and a class-coverage summary.

Usage:
    python scripts/02_make_splits.py --data-dir data/raw

Writes results/splits_runwise_v1.json (run -> fold assignments + metadata,
loaded by training code with tinyml.splits.load_assignments) and
results/split_summary.md. Only pv_fault_dataset.csv is read.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyml.contract import LABELS  # noqa: E402
from tinyml.splits import (  # noqa: E402
    DEFAULT_FOLDS, DEFAULT_SEED, SPLIT_VERSION, make_assignments, summarize, to_json,
)
from tinyml.validate import file_sha256  # noqa: E402

SHORT = {"NORMAL": "NOR", "PARTIAL_SHADING": "PAR", "DUST": "DUS",
         "OPEN_CIRCUIT": "OPE", "SHORT_CIRCUIT": "SHO"}


def find_dataset(data_dir):
    hits = sorted(p for p in data_dir.glob("*.csv")
                  if p.name == "pv_fault_dataset.csv" or p.name.endswith("-pv_fault_dataset.csv"))
    if not hits:
        sys.exit(f"pv_fault_dataset.csv not found in {data_dir}")
    return hits[0]


def render(summary, meta):
    L = ["# Run-wise split summary", "",
         f"Split version `{meta['split_version']}`, seed {meta['seed']}, "
         f"{meta['n_splits']}-fold CV over {meta['cv_window']} runs. "
         f"Dataset SHA-256 `{meta['dataset_sha256']}`.", "",
         "Every run is kept whole: no `run_id` appears in both train and test of any split. "
         "`run_id` and `site_id` are grouping keys only, never model features.", "",
         "Cells show runs per class as NOR/PAR/DUS/OPE/SHO.", ""]
    titles = {"cv": "Protocol 1: run-wise stratified CV (model selection, TRAINING window only)",
              "temporal": "Protocol 2: TRAINING → OPERATION (final held-out test, used once)",
              "loso": "Protocol 3: leave-one-site-out (generalisation to an unseen site, all runs)"}
    for protocol, folds in summary.items():
        L += [f"## {titles[protocol]}", "",
              "| Fold | Train runs | Train rows | Test runs | Test rows | Classes missing in test |",
              "|---|---|---|---|---|---|"]
        for f in folds:
            tr = "/".join(str(f["train_runs"][l]) for l in LABELS)
            te = "/".join(str(f["test_runs"][l]) for l in LABELS)
            miss = ", ".join(SHORT[l] for l in f["test_missing"]) or "none"
            L.append(f"| {f['fold']} | {tr} | {f['train_rows']} | {te} | {f['test_rows']} | {miss} |")
        L.append("")
    L += ["## Limitations", "",
          "- **CV** uses only the 50 TRAINING runs (10 per class, 2 per class per fold), so the "
          "OPERATION runs stay unseen until the final test.",
          "- **Temporal** has one split only; its result is a single estimate (report per-class "
          "recall and run-level results, not only accuracy).",
          "- **Leave-one-site-out:** some sites do not contain every class, so their test folds "
          "cannot measure recall for those classes. Report macro-F1 over the classes present in "
          "each fold, list the missing classes, and give the pooled result over all six folds. "
          "Site sizes differ (6 to 26 runs), so folds are unequal; SITE-F (6 runs) is the "
          "least reliable fold.",
          "- LOSO mixes TRAINING and OPERATION runs and is a separate generalisation study; "
          "it is not used for model selection.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    a = ap.parse_args()

    path = find_dataset(a.data_dir)
    df = pd.read_csv(path)
    assignments = make_assignments(df, n_splits=a.folds, seed=a.seed)
    meta = {"split_version": SPLIT_VERSION, "seed": a.seed, "n_splits": a.folds,
            "cv_window": "TRAINING", "dataset_file": "pv_fault_dataset.csv",
            "dataset_sha256": file_sha256(path), "dataset_rows": int(len(df)),
            "runs": int(len(assignments)), "sklearn_version": sklearn.__version__}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "splits_runwise_v1.json").write_text(to_json(assignments, meta) + "\n", encoding="utf-8")
    (a.out_dir / "split_summary.md").write_text(render(summarize(df, assignments), meta), encoding="utf-8")
    print(f"wrote {a.out_dir / 'splits_runwise_v1.json'} and {a.out_dir / 'split_summary.md'}")


if __name__ == "__main__":
    main()

"""Final model training and export (TRAINING window only).

Usage (in Colab; see notebooks/colab_final_training.ipynb):
    python scripts/05_train_final.py --data-dir data/raw --out-dir models

Fits the pv8-v1 features and the 8-16-5 tanh MLP on the TRAINING window,
then writes model_contract.json, feature_config.json, training_metadata.json
and manifest.json, and verifies them. The OPERATION window is not read for
fitting or evaluation.
"""
import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyml.export import build_contract, training_metadata, verify_artifacts, write_artifacts  # noqa: E402
from tinyml.splits import load_assignments  # noqa: E402
from tinyml.train import (FINAL_SEED, check_dataset_sha256, environment, git_commit,  # noqa: E402
                          train_final)
from tinyml.validate import file_sha256  # noqa: E402


def find_dataset(data_dir):
    for p in sorted(Path(data_dir).glob("*.csv")):
        if p.name == "pv_fault_dataset.csv" or p.name.endswith("-pv_fault_dataset.csv"):
            return p
    sys.exit(f"pv_fault_dataset.csv not found in {data_dir}")


def run(data_path, splits_path, out_dir, seed=FINAL_SEED, code_commit=None):
    """Train, export and verify. Returns (result, contract, metadata, paths)."""
    sha = check_dataset_sha256(file_sha256(data_path))
    df = pd.read_csv(data_path)
    assignments, meta = load_assignments(splits_path, df, dataset_sha256=sha)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = train_final(df, seed=seed, assignments=assignments)
    result["facts"]["warnings"] = sorted({f"{w.category.__name__}: {str(w.message)[:160]}"
                                          for w in caught})
    commit = code_commit or git_commit(ROOT)
    contract = build_contract(result, sha, meta["split_version"], commit, environment())
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata = training_metadata(result, contract, created_utc=created)
    paths = write_artifacts(out_dir, contract, result["config"], metadata)
    verify_artifacts(out_dir)
    return result, contract, metadata, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--splits", default=str(ROOT / "results" / "splits_runwise_v1.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "models"))
    ap.add_argument("--seed", type=int, default=FINAL_SEED)
    a = ap.parse_args()
    result, contract, _, paths = run(find_dataset(a.data_dir), a.splits, a.out_dir, a.seed)
    f = result["facts"]
    print(f"Trained on {f['rows']} TRAINING rows ({f['runs']} runs), {f['ok_rows_used']} used, "
          f"{f['skipped_rows']} skipped {f['skip_reasons']}")
    print(f"MLP seed {f['seed']}: {f['n_iter']} iterations, converged {f['converged']}, "
          f"{contract['n_params']} parameters, weights SHA-256 {contract['weights_sha256'][:16]}...")
    print("Wrote and verified:", *paths, sep="\n  ")
    if f["warnings"]:
        print("Warnings during training:", *f["warnings"], sep="\n  ")


if __name__ == "__main__":
    main()

"""Validate the PV fault dataset and write results/dataset_validation.md.

Usage:
    python scripts/01_validate_dataset.py --data-dir data/raw
    python scripts/01_validate_dataset.py --data-dir data/raw --out results/dataset_validation.md

The data directory should contain pv_fault_dataset.csv and, if available,
pv_<label>.csv, sites.csv and member2_fault_predictions.csv (file names may
carry a prefix such as an upload id). Files are only read.
"""
# ---- Pipeline step 1 of 8: check the dataset before anything else uses it ----
# When:   once, every time a new copy of pv_fault_dataset.csv is received
# Reads:  the CSV files in --data-dir (never changed)
# Writes: results/dataset_validation.md
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Make the tinyml package in src/ importable when this script is run directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyml.validate import (  # noqa: E402
    file_sha256, label_for_part_file, render_markdown, validate_dataset,
)


# Find a CSV by name; also accepts names with an upload prefix (e.g. 'abc123-sites.csv').
def find(data_dir, name):
    hits = sorted(p for p in data_dir.glob("*.csv")
                  if p.name == name or p.name.endswith("-" + name) or p.name.endswith("_" + name))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "dataset_validation.md")
    a = ap.parse_args()

    # 1) Locate the main dataset; stop if it is missing.
    combined_path = find(a.data_dir, "pv_fault_dataset.csv")
    if combined_path is None:
        sys.exit(f"pv_fault_dataset.csv not found in {a.data_dir}")
    files = []

    # Read one CSV and remember its name, row count and SHA-256 fingerprint for the report.
    def load(path):
        df = pd.read_csv(path)
        name = re.sub(r"^[0-9a-f]{8}-", "", path.name)      # drop upload-id prefixes
        files.append({"name": name, "rows": len(df), "sha256": file_sha256(path)})
        return df

    # 2) Load the main file, then the optional per-class, sites and prediction files.
    combined = load(combined_path)
    parts = {}
    for p in sorted(a.data_dir.glob("*.csv")):
        label = label_for_part_file(p)
        if label:
            parts[label] = load(p)
    sites_path = find(a.data_dir, "sites.csv")
    pred_path = find(a.data_dir, "member2_fault_predictions.csv")
    sites = load(sites_path) if sites_path else None
    preds = load(pred_path) if pred_path else None

    # 3) Run all checks: columns, units, ranges, labels, runs, duplicates, label leakage.
    result = validate_dataset(combined, parts, sites, preds)
    # 4) Write the report and print a summary; exit code 1 means errors were found.
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render_markdown(result, files), encoding="utf-8")
    print(json.dumps({"errors": result["errors"], "warnings": len(result["warnings"]),
                      "report": str(a.out)}, indent=1))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())

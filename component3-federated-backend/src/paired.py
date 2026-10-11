"""Paired comparison across seeds: every method saw the same seeds (same init, same network luck)."""
import argparse
import sys
import numpy as np
import pandas as pd
from src.fl_baseline import RESULTS_DIR

T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26,
       10: 2.23, 11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13}
METRICS = ["final_acc", "worst_class", "lost", "used", "lora_s", "bytes"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="harsh", choices=["trace", "stochastic", "harsh"])
    ap.add_argument("--ref", default="baseline", help="config to compare against")
    ap.add_argument("--tag", default="", help="same tag as used with evaluate.py")
    a = ap.parse_args()

    path = RESULTS_DIR / f"eval_{a.scenario}{'_' + a.tag if a.tag else ''}_runs.csv"
    if not path.exists():
        sys.exit(f"file not found: {path}\nrun evaluate.py first (with the same --scenario and --tag)")
    runs = pd.read_csv(path)

    if a.ref not in set(runs.config):
        sys.exit(f"'{a.ref}' is not in {path.name}. Configs there: {sorted(set(runs.config))}")
    ref = runs[runs.config == a.ref].set_index("seed")
    n = len(ref)
    crit = T95.get(n - 1, 2.0)
    print(f"{a.scenario}: each config minus '{a.ref}', per seed (n={n} seeds, |t| > {crit} = clearly not noise)\n")
    for cfg in [c for c in runs.config.unique() if c != a.ref]:
        cur = runs[runs.config == cfg].set_index("seed")
        parts = []
        for m in METRICS:
            if m not in runs.columns:
                continue
            d = (cur[m] - ref[m]).dropna()
            sd = d.std(ddof=1)
            t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else (float("inf") if d.mean() else 0.0)
            mark = "*" if abs(t) > crit else " "
            nd = 4 if m in ("final_acc", "worst_class") else 0
            parts.append(f"{m} {d.mean():+.{nd}f}{mark}")
        print(f"{cfg:24s} " + " | ".join(parts))
    print("\n* = difference larger than seed-to-seed noise")

if __name__ == "__main__":
    main()
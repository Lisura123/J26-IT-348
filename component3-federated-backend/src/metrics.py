"""Time to X% accuracy, counted from the first round that STAYS above X for `sustain` rounds."""
import sys
from glob import glob
from pathlib import Path
import pandas as pd
from src.fl_baseline import START

def time_to(df, target, sustain=3):
    acc = df.global_val_accuracy.tolist()
    for i in range(len(acc) - sustain + 1):
        if all(a >= target for a in acc[i:i + sustain]):
            return (pd.Timestamp(df.published_at_utc.iloc[i]) - START).total_seconds() / 3600
    return None

def summarize(path):
    df = pd.read_csv(path)
    if "global_val_accuracy" not in df.columns:        # e.g. site_state files
        return
    f = lambda v: "not reached" if v is None else f"{v:.1f} h"
    print(f"{Path(path).name:62s} final {df.global_val_accuracy.iloc[-1]:.4f} "
          f"| 90%: {f(time_to(df, .90))} | 95%: {f(time_to(df, .95))}")

if __name__ == "__main__":
    for pattern in sys.argv[1:]:
        for p in sorted(glob(pattern)) or [pattern]:   # PowerShell does not expand wildcards for Python
            summarize(p)
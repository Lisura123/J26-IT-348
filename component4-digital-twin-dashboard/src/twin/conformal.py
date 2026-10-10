
import json
from pathlib import Path

import numpy as np


def fit_band(ratio, alpha=0.10) -> dict:
    """Finite-sample split-conformal quantiles of the ratio (two-sided, equal tails)."""
    r = np.sort(np.asarray(ratio, float))
    n = len(r)
    k_hi = int(np.ceil((n + 1) * (1 - alpha / 2)))    # conformal rank, upper tail
    k_lo = int(np.floor((n + 1) * (alpha / 2)))       # conformal rank, lower tail
    return {"alpha": alpha, "n_calibration": n,
            "ratio_low": float(r[max(k_lo, 1) - 1]),
            "ratio_high": float(r[min(k_hi, n) - 1])}


def apply_band(expected, band: dict):
    e = np.asarray(expected, float)
    return e * band["ratio_low"], e * band["ratio_high"]


def coverage(actual, expected, band: dict) -> float:
    lo, hi = apply_band(expected, band)
    a = np.asarray(actual, float)
    return float(np.mean((a >= lo) & (a <= hi)))


def save(band: dict, path: Path):
    Path(path).write_text(json.dumps(band, indent=1))


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())
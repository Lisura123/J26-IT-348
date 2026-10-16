"""Final model training on the TRAINING window (Phase 8).

Fits the approved deployment model from Phase 7 (docs/PHASE7_RESULTS.md):
the pv8-v1 feature pipeline plus the 8-16-5 tanh MLP, with exactly the
hyperparameters used in the Phase 7 comparison (models.make_model("mlp")).

Only TRAINING-window rows are used for every learned value: V_ref, the
scaler and the MLP weights. OPERATION rows are never read for fitting,
hyperparameter choice, thresholds or exploratory evaluation; they stay
reserved for the single final test.

Seed policy (FINAL_SEED_POLICY): Phase 7 trained the MLP with seeds 0, 1
and 2 but did not define which seed the final model uses. The final model
uses seed 0, fixed before training by rule (the lowest Phase 7 seed). No
seed is selected by score, and OPERATION data is never used to choose one.
"""
import platform
import subprocess

import numpy as np
import pandas as pd
import sklearn

from .contract import GROUP_COLUMN, LABEL_COLUMN, LABELS, WINDOW_COLUMN
from .features import STATUS_OK, fit_pipeline, transform
from .models import make_model, model_size

FINAL_WINDOW = "TRAINING"
FINAL_FEATURE_SET = "physics8"
FINAL_MODEL = "mlp"
FINAL_SEED = 0
FINAL_SEED_POLICY = ("fixed a priori for reproducibility: seed 0, the lowest of the Phase 7 "
                     "seeds (0, 1, 2), declared before final training; not chosen by score, "
                     "no seed is selected by score and OPERATION data is never used")
FINAL_FIT_TAG = "final_TRAINING"
# pv_fault_dataset.csv validated in Phase 2 (results/dataset_validation.md)
DATASET_SHA256 = "5fbb93471c056c274cbf81e8c317934e379b2484013147e4c2cacfa93421595b"


def check_dataset_sha256(sha):
    if sha != DATASET_SHA256:
        raise ValueError(f"dataset SHA-256 {sha} is not the validated pv_fault_dataset.csv "
                         f"({DATASET_SHA256})")
    return sha


def final_training_rows(df, assignments=None):
    """Index of the rows used for final fitting: the TRAINING window only.

    If the saved run-wise assignments are given, the rows must be exactly the
    'train' side of the temporal split, so the final model and the Phase 4
    split file agree."""
    idx = df.index[df[WINDOW_COLUMN] == FINAL_WINDOW]
    if not len(idx):
        raise ValueError(f"no {FINAL_WINDOW} rows in the dataset")
    if assignments is not None:
        side = df[GROUP_COLUMN].map(assignments["temporal"])
        expected = df.index[(side == "train").to_numpy()]
        if not idx.equals(expected):
            raise ValueError("TRAINING rows do not match the temporal 'train' side of the split file")
    return idx


def train_final(df, seed=FINAL_SEED, assignments=None):
    """Fit the feature pipeline and the MLP on TRAINING rows only.

    Returns a dict with the fitted FeatureConfig, the sklearn model and the
    facts about the fit (rows, runs, coverage, iterations). The model is
    evaluated only on its own training rows as a fit diagnostic; that is
    not a performance estimate (Phase 7 cross-validation is)."""
    idx = final_training_rows(df, assignments)
    train = df.loc[idx]
    assert (train[WINDOW_COLUMN] == FINAL_WINDOW).all(), "non-TRAINING rows in final training"
    cfg = fit_pipeline(train, fold=FINAL_FIT_TAG)
    X, status = transform(train, cfg)
    keep = train.index[(status == STATUS_OK).to_numpy()]
    y = train.loc[keep, LABEL_COLUMN].to_numpy()
    missing = [l for l in LABELS if l not in set(y)]
    if missing:
        raise ValueError(f"classes missing from the training rows: {missing}")
    model = make_model(FINAL_MODEL, seed=seed)
    model.fit(X, y)
    fit_pred = model.predict(X)
    skipped = status[status != STATUS_OK]
    return {
        "config": cfg,
        "model": model,
        "seed": seed,
        "X": X,
        "y": y,
        "index": keep,
        "facts": {
            "window": FINAL_WINDOW,
            "feature_set": FINAL_FEATURE_SET,
            "model": FINAL_MODEL,
            "seed": seed,
            "seed_policy": FINAL_SEED_POLICY,
            "rows": int(len(train)),
            "runs": int(train[GROUP_COLUMN].nunique()),
            "sites": sorted(train["site_id"].unique().tolist()) if "site_id" in train else None,
            "ok_rows_used": int(len(keep)),
            "skipped_rows": int(len(skipped)),
            "skip_reasons": {k: int(v) for k, v in skipped.value_counts().sort_index().items()},
            "class_counts": {l: int((y == l).sum()) for l in LABELS},
            "hyperparameters": _json_params(model.get_params()),
            "n_iter": int(model.n_iter_),
            "max_iter": int(model.max_iter),
            "converged": bool(model.n_iter_ < model.max_iter),
            "final_loss": float(model.loss_),
            "out_activation": model.out_activation_,
            "size": model_size(model),
            "training_fit_accuracy": float((fit_pred == y).mean()),
            "training_fit_note": "accuracy on the training rows themselves; a fit check, "
                                 "not a performance estimate",
        },
    }


def _json_params(params):
    return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else
                list(v) if isinstance(v, tuple) else repr(v)) for k, v in sorted(params.items())}


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__}


def git_commit(repo_dir=None):
    """Short commit hash of the code that ran, or None outside a git checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None

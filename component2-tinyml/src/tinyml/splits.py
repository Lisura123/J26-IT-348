"""Run-wise, leakage-free split protocols (Phase 4).

All splitting is done on whole runs: a run_id is never divided between
train and test. Three protocols (docs/ARCHITECTURE.md section 8.1):

  cv        stratified 5-fold over TRAINING-window runs (model selection)
  temporal  TRAINING runs -> OPERATION runs (final held-out test, used once)
  loso      leave-one-site-out over all runs (generalisation to an unseen device)

run_id is used only as the grouping key and site_id only as the LOSO
grouping key; neither is a model feature. The same functions are used
locally and in Colab, so there is one split implementation.
"""
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .contract import GROUP_COLUMN, LABEL_COLUMN, LABELS, WINDOW_COLUMN

SPLIT_VERSION = "runwise-v1"
DEFAULT_SEED = 42
DEFAULT_FOLDS = 5
PROTOCOLS = ("cv", "temporal", "loso")


def run_table(df):
    """One row per run with its label, site and window; sorted by run_id."""
    g = df.groupby(GROUP_COLUMN)
    for col in (LABEL_COLUMN, "site_id", WINDOW_COLUMN):
        mixed = g[col].nunique()
        if (mixed > 1).any():
            raise ValueError(f"runs with more than one {col}: {sorted(mixed[mixed > 1].index)}")
    runs = g.agg(label=(LABEL_COLUMN, "first"), site_id=("site_id", "first"),
                 window=(WINDOW_COLUMN, "first"), rows=(LABEL_COLUMN, "size"))
    return runs.sort_index()


# [GUIDE] B. Dataset splitting: where the 50/50 division comes from
# WHAT: gives every run (run_id) a 5-fold CV fold, a temporal side and a LOSO fold.
# IMPORTANT: there is NO random `train_test_split(test_size=0.5)` anywhere in this code.
#       The dataset itself already has a `window` column: 3,000 TRAINING readings
#       (50 runs, 1-9 Sept) and 3,000 OPERATION readings (50 runs, 10-15 Sept).
#       This function only KEEPS those two windows (lines with runs["temporal"] below),
#       so the train/test sizes are 3,000 / 3,000 = 50 % / 50 % because of the data,
#       not because 50/50 was chosen as the best ratio.
# HOW THE COUNTS ARE VERIFIED: results/split_summary.md (temporal: 3000 train / 3000 test
#       rows, 10 runs per class on each side); holdout.EXPECTED_ROWS = 3000 and
#       EXPECTED_RUNS = 50 are checked again before the final test (holdout.run_evaluation).
# INPUT: the full DataFrame.  OUTPUT: one row per run with cv_fold, temporal, loso_fold.
# EVIDENCE: results/splits_runwise_v1.json, results/split_summary.md.
def make_assignments(df, n_splits=DEFAULT_FOLDS, seed=DEFAULT_SEED, cv_window="TRAINING"):
    """Assign every run to a CV fold, a temporal partition and a LOSO fold.

    Returns a DataFrame indexed by run_id with columns
    label, site_id, window, rows, cv_fold (-1 = not in CV), temporal, loso_fold.
    """
    runs = run_table(df)
    cv_runs = runs[runs.window == cv_window]
    per_class = cv_runs.label.value_counts()
    if per_class.min() < n_splits:
        raise ValueError(f"need >= {n_splits} {cv_window} runs per class for {n_splits}-fold CV, "
                         f"got {per_class.to_dict()}")
    runs["cv_fold"] = -1
    # [GUIDE] B. Run-wise 5-fold cross-validation (model selection, TRAINING window only)
    # WHY:  one train/test split gives one noisy number. 5 folds give 5 estimates, so models
    #       can be compared fairly without ever touching the OPERATION test data.
    # HOW:  the folds are made over RUNS, not rows (cv_runs has one row per run), stratified
    #       by label: each fold tests 2 runs per class (600 readings) and trains on 8 runs per class.
    # WHY RUNS: readings of one run are 5 s apart and nearly identical; if some went to train
    #       and some to validation, the model would be tested on near-copies of its training data.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(skf.split(cv_runs.index, cv_runs.label)):
        runs.loc[cv_runs.index[test], "cv_fold"] = fold
    # [GUIDE] B. THIS is the 50/50 line: TRAINING window -> "train", OPERATION window -> "test".
    # The split follows the dataset's own time windows; nothing is shuffled or resampled here.
    runs["temporal"] = np.where(runs.window == "TRAINING", "train",
                                np.where(runs.window == "OPERATION", "test", "unused"))
    runs["loso_fold"] = runs.site_id
    return runs


# [GUIDE] B. Turns the saved assignments into (train rows, test rows) for one protocol
# ("cv" = 5 folds, "temporal" = TRAINING->OPERATION, "loso" = leave one site out).
# Every split passes through _checked(), which stops if a run is on both sides.
def iter_folds(df, assignments, protocol):
    """Yield (fold_name, train_row_index, test_row_index) for one protocol.

    Row indices refer to `df`, so training code (local or Colab) can do
    df.loc[train_idx]. Every yielded split is checked for run leakage.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    missing = set(df[GROUP_COLUMN]) - set(assignments.index)
    if missing:
        raise ValueError(f"{len(missing)} runs in the data have no split assignment")
    a = assignments.reindex(df[GROUP_COLUMN]).set_index(df.index)
    if protocol == "cv":
        in_cv = a.cv_fold >= 0
        for k in sorted(a.loc[in_cv, "cv_fold"].unique()):
            yield _checked(df, f"cv{k}", df.index[in_cv & (a.cv_fold != k)],
                           df.index[a.cv_fold == k])
    elif protocol == "temporal":
        yield _checked(df, "temporal", df.index[a.temporal == "train"], df.index[a.temporal == "test"])
    else:
        for site in sorted(a.loso_fold.unique()):
            yield _checked(df, f"loso_{site}", df.index[a.loso_fold != site],
                           df.index[a.loso_fold == site])


# [GUIDE] B. Leakage guard: raises an error if any run_id is in both train and test.
def _checked(df, name, train_idx, test_idx):
    shared = set(df.loc[train_idx, GROUP_COLUMN]) & set(df.loc[test_idx, GROUP_COLUMN])
    if shared:
        raise AssertionError(f"{name}: runs in both train and test: {sorted(shared)[:5]}")
    return name, train_idx, test_idx


def summarize(df, assignments):
    """Per protocol and fold: runs/rows per class in train and test, and
    the classes missing from each side."""
    out = {}
    for protocol in PROTOCOLS:
        folds = []
        for name, tr, te in iter_folds(df, assignments, protocol):
            row = {"fold": name}
            for side, idx in (("train", tr), ("test", te)):
                part = df.loc[idx]
                row[f"{side}_runs"] = {l: int(part.loc[part[LABEL_COLUMN] == l, GROUP_COLUMN].nunique())
                                       for l in LABELS}
                row[f"{side}_rows"] = int(len(part))
                row[f"{side}_missing"] = [l for l in LABELS if row[f"{side}_runs"][l] == 0]
            folds.append(row)
        out[protocol] = folds
    return out


def to_json(assignments, meta):
    runs = {rid: {"label": r.label, "site_id": r.site_id, "window": r.window, "rows": int(r.rows),
                  "cv_fold": int(r.cv_fold), "temporal": r.temporal, "loso_fold": r.loso_fold}
            for rid, r in assignments.iterrows()}
    return json.dumps({"meta": meta, "runs": runs}, indent=1, sort_keys=True)


def load_assignments(path, df=None, dataset_sha256=None):
    """Load a saved split file. If a hash is given it must match the one the
    splits were made from; if df is given every run must be covered."""
    import pandas as pd
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if dataset_sha256 and data["meta"].get("dataset_sha256") != dataset_sha256:
        raise ValueError("split file was made from a different dataset (SHA-256 mismatch)")
    a = pd.DataFrame.from_dict(data["runs"], orient="index").sort_index()
    a.index.name = GROUP_COLUMN
    if df is not None:
        missing = set(df[GROUP_COLUMN]) - set(a.index)
        if missing:
            raise ValueError(f"{len(missing)} runs in the data are missing from the split file")
    return a, data["meta"]

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
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(skf.split(cv_runs.index, cv_runs.label)):
        runs.loc[cv_runs.index[test], "cv_fold"] = fold
    runs["temporal"] = np.where(runs.window == "TRAINING", "train",
                                np.where(runs.window == "OPERATION", "test", "unused"))
    runs["loso_fold"] = runs.site_id
    return runs


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

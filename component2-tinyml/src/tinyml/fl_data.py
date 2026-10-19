"""pv8-v1 data for Member 3's federated learning (integration adapter).

Member 3's FL (component3-federated-backend/src/pv_data.get_data) feeds raw8
sensor columns with a scaler refitted at every run, and validates on the
OPERATION window. This module gives the same shape of data in the input
space of the frozen model c2-local-v1 instead:

  - only TRAINING rows; OPERATION rows are dropped before any feature is
    computed, so they are never transformed, trained on or validated on
  - features from features.transform with the frozen FeatureConfig
    (models/feature_config.json, verified against the manifest): same
    V_ref, scaler, quality gates and feature order; nothing is refitted
  - only status-OK readings (the same gates the model was trained with)
  - development validation = the TRAINING runs of one cv_fold of the
    committed run-wise split (results/splits_runwise_v1.json); a run is
    never split between training and validation
  - site_id partitions kept; labels in LABELS order (= Member 3's order)

Return value of member3_get_data() matches pv_data.get_data():
(site_data, (X_val, y_val), cfg) with site_data[site] = (X, y).

Validation rows were also used to train c2-local-v1, so validation numbers
on them are a development check of the FL runs, not independent accuracy.
Nothing here trains a model or changes a file.
"""
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .contract import GROUP_COLUMN, LABEL_COLUMN, LABELS, WINDOW_COLUMN
from .export import contract_feature_config, contract_weights, verify_artifacts
from .features import FEATURE_VERSION, FEATURES, STATUS_OK, transform
from .splits import load_assignments
from .train import check_dataset_sha256
from .validate import file_sha256

COMPONENT = Path(__file__).resolve().parents[2]
MODELS_DIR = COMPONENT / "models"
SPLITS_PATH = COMPONENT / "results" / "splits_runwise_v1.json"
MEMBER3_ROOT = COMPONENT.parent / "component3-federated-backend"

FL_DATA_VERSION = "pv8-fl-data-1"
TRAIN_WINDOW = "TRAINING"
VALIDATION_FOLD = 0      # fixed a priori (lowest fold number); not chosen from results
LABEL_ID = {l: i for i, l in enumerate(LABELS)}


@dataclass
class FLData:
    site_data: dict                       # site -> (X float64 [n, 8], y int64 [n])
    val: tuple                            # pooled validation (X, y)
    site_val: dict                        # site -> (X, y); empty arrays when a site has no validation run
    cfg: object                           # the frozen FeatureConfig (not refitted)
    meta: dict = field(default_factory=dict)


def load_frozen_model(models_dir=MODELS_DIR):
    """(contract, FeatureConfig, weights) of c2-local-v1, after the manifest
    hashes and the contract self-checks (export.verify_artifacts)."""
    contract = verify_artifacts(models_dir)
    return contract, contract_feature_config(contract), contract_weights(contract)


def load_dataset(path):
    """Read the dataset CSV after checking it is the validated file."""
    check_dataset_sha256(file_sha256(path))
    return pd.read_csv(path)


def training_rows(df):
    """TRAINING rows only (copy). OPERATION and any other window are dropped here."""
    for col in (WINDOW_COLUMN, GROUP_COLUMN, LABEL_COLUMN, "site_id"):
        if col not in df.columns:
            raise KeyError(f"missing column {col!r}")
    tr = df[df[WINDOW_COLUMN] == TRAIN_WINDOW].copy()
    if not len(tr):
        raise ValueError(f"no {TRAIN_WINDOW} rows")
    unknown = sorted(set(tr[LABEL_COLUMN]) - set(LABELS))
    if unknown:
        raise ValueError(f"unknown labels {unknown}")
    return tr


def validation_runs(assignments, fold=VALIDATION_FOLD):
    """TRAINING runs whose committed cv_fold is `fold`."""
    a = assignments[(assignments["window"] == TRAIN_WINDOW) & (assignments["cv_fold"] == fold)]
    if not len(a):
        raise ValueError(f"no {TRAIN_WINDOW} runs in cv_fold {fold}")
    return set(a.index)


def _xy(df, cfg):
    X, status = transform(df, cfg)
    ok = (status == STATUS_OK).to_numpy()
    y = df.loc[ok, LABEL_COLUMN].map(LABEL_ID).to_numpy(np.int64)
    return X.astype(np.float64), y, status.value_counts().to_dict()


def build_fl_data(df, cfg, assignments, fold=VALIDATION_FOLD):
    """Per-site training data and a run-wise validation split, pv8-v1 inputs."""
    if not cfg.is_fitted or tuple(cfg.features) != FEATURES or cfg.version != FEATURE_VERSION:
        raise ValueError("needs the fitted, frozen pv8-v1 FeatureConfig")
    tr = training_rows(df)
    missing = set(tr[GROUP_COLUMN]) - set(assignments.index)
    if missing:
        raise ValueError(f"{len(missing)} TRAINING runs are not in the split file")
    val_runs = validation_runs(assignments, fold)
    is_val = tr[GROUP_COLUMN].isin(val_runs)
    sites = sorted(tr["site_id"].unique())
    site_data, site_val, counts = {}, {}, {}
    for s in sites:
        part = tr[tr["site_id"] == s]
        Xt, yt, st = _xy(part[~is_val.loc[part.index]], cfg)
        v = part[is_val.loc[part.index]]
        Xv, yv, sv = _xy(v, cfg) if len(v) else (np.empty((0, len(FEATURES))), np.empty(0, np.int64), {})
        site_data[s], site_val[s] = (Xt, yt), (Xv, yv)
        counts[s] = {"train_rows": int(len(yt)), "val_rows": int(len(yv)),
                     "train_runs": int(part.loc[~is_val.loc[part.index], GROUP_COLUMN].nunique()),
                     "val_runs": int(v[GROUP_COLUMN].nunique()),
                     "input_status": {k: int(n) for k, n in sorted({**st}.items())},
                     "val_input_status": {k: int(n) for k, n in sorted(sv.items())},
                     "class_counts": {l: int((yt == i).sum()) for i, l in enumerate(LABELS)}}
    val = (np.concatenate([site_val[s][0] for s in sites]), np.concatenate([site_val[s][1] for s in sites]))
    meta = {"fl_data_version": FL_DATA_VERSION, "feature_version": cfg.version, "features": list(cfg.features),
            "labels": list(LABELS), "window": TRAIN_WINDOW, "operation_rows_used": 0,
            "validation": {"split_file": SPLITS_PATH.name, "cv_fold": fold, "runs": sorted(val_runs),
                           "note": "TRAINING runs also seen by c2-local-v1; development check only"},
            "gates": "status-OK readings only", "sites": counts}
    return FLData(site_data, val, site_val, cfg, meta)


def member3_get_data(dataset_path, models_dir=MODELS_DIR, splits_path=SPLITS_PATH, fold=VALIDATION_FOLD):
    """Drop-in for Member 3's pv_data.get_data(): (site_data, (X_val, y_val), cfg)."""
    _, cfg, _ = load_frozen_model(models_dir)
    df = load_dataset(dataset_path)
    assignments, _ = load_assignments(splits_path, dataset_sha256=file_sha256(dataset_path))
    d = build_fl_data(df, cfg, assignments, fold)
    return d.site_data, d.val, d.cfg


def load_member3_module(name, root=MEMBER3_ROOT):
    """Import one of Member 3's src/<name>.py files read-only, under a private
    module name (both components have a top-level package called src)."""
    path = Path(root) / "src" / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"_member3_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

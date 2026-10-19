"""Run Member 3's FL engine from the frozen Model v1 (integration wrapper).

  verified c2-local-v1 (fl_data.load_frozen_model)
  + pv8-v1 TRAINING-only site data (fl_data.member3_get_data)
  -> component3-federated-backend/src/engine.run_fl(activation="tanh",
     data=..., init_weights=..., save=False)        [engine change PENDING SAMUDI REVIEW]
  -> candidate global model, in memory only

Nothing is written: save=False keeps Member 3's committed results/ files
(engine_<tag>_tanh_seed<seed>.csv) from being overwritten, and the candidate
is returned, not stored. The engine's "global_val_*" numbers are computed on
TRAINING runs that Model v1 was also trained on: a development check, not
independent validation and not an accuracy claim.
"""
import sys

import numpy as np

from . import fl_data as fd
from .export import N_PARAMS, flatten, weights_sha256
from .features import FEATURES
from .fl_adapter import feature_config_sha256

CANDIDATE_STATUS = "CANDIDATE_NOT_VALIDATED"
VALIDATION_NOTE = ("engine global_val_* metrics use TRAINING runs also used to train c2-local-v1: "
                   "development check only, not independent validation")
FIXED_ARGS = ("activation", "data", "init_weights", "save")


def member3_engine(root=fd.MEMBER3_ROOT):
    """Member 3's src.engine (needs flwr). Refuses another package called src."""
    root = str(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    import src.engine as engine
    if not engine.__file__.startswith(root):
        raise ImportError(f"src.engine resolved to {engine.__file__}, not {root}")
    return engine


def check_data(data, sites):
    """(site_data, (Xv, yv), _) with every engine site and 8 pv8-v1 columns."""
    site_data, (Xv, yv), _ = data
    if sorted(site_data) != sorted(sites):
        raise ValueError(f"sites {sorted(site_data)} != engine sites {sorted(sites)}")
    for s, (X, y) in [*site_data.items(), ("validation", (Xv, yv))]:
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 2 or X.shape[1] != len(FEATURES) or len(X) != len(y):
            raise ValueError(f"{s}: X {X.shape} / y {y.shape} is not pv8-v1 data")
        if s != "validation" and not len(y):
            raise ValueError(f"{s}: no training rows")
        if len(y) and (y.min() < 0 or y.max() > 4 or not np.issubdtype(y.dtype, np.integer)):
            raise ValueError(f"{s}: labels must be integers 0..4")
    return data


def run_from_model_v1(dataset_path=None, data=None, seed=0, engine=None, models_dir=fd.MODELS_DIR, **engine_kwargs):
    """Start Member 3's FL from Model v1. Returns {candidate, history, updates}.
    Give dataset_path (TRAINING rows are taken by fl_data) or ready data."""
    fixed = sorted(set(engine_kwargs) & set(FIXED_ARGS))
    if fixed:
        raise ValueError(f"{fixed} are fixed by the wrapper (tanh, Member 2 data and weights, save=False)")
    contract, _, w_v1 = fd.load_frozen_model(models_dir)
    base_sha = weights_sha256(w_v1)
    source = "caller-supplied (OPERATION exclusion not checked here)"
    if data is None:
        if dataset_path is None:
            raise ValueError("give dataset_path or data")
        data, source = fd.member3_get_data(dataset_path, models_dir), "fl_data.member3_get_data (TRAINING only)"
    engine = engine or member3_engine()
    check_data(data, engine.pv_data.SITES)
    engine_kwargs.setdefault("verbose", False)
    df = engine.run_fl(seed=seed, activation="tanh", data=data, init_weights=w_v1, save=False, **engine_kwargs)
    if weights_sha256(w_v1) != base_sha:
        raise AssertionError("Model v1 weights changed during the FL run")
    models = df.attrs["models"]
    last = max(models, key=lambda v: int(v.split(".")[1]))
    final = {k: np.asarray(v, dtype=np.float64) for k, v in models[last].items()}
    if flatten(final).size != N_PARAMS or not np.isfinite(flatten(final)).all():
        raise ValueError("engine returned an invalid global model")
    candidate = {"status": CANDIDATE_STATUS, "engine_version": last, "engine_start_version": "v1.0",
                 "base_model_version": contract["model_version"], "base_weights_sha256": base_sha,
                 "feature_version": contract["feature_version"],
                 "feature_config_sha256": feature_config_sha256(contract), "labels": list(contract["labels"]),
                 "activation": "tanh", "param_layout": contract["param_layout"], "seed": seed,
                 "engine_options": dict(engine_kwargs, save=False), "data_source": source,
                 "operation_rows_used": 0 if source.startswith("fl_data") else None,
                 "validation_note": VALIDATION_NOTE, "weights": final, "weights_sha256": weights_sha256(final)}
    return {"candidate": candidate, "history": df, "updates": df.attrs["updates"]}

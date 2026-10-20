"""Run Member 3's FL engine from the frozen Model v1 (integration wrapper).

  verified c2-local-v1 (fl_data.load_frozen_model)
  + pv8-v1 TRAINING-only site data (fl_data.member3_get_data)
  -> component3-federated-backend/src/engine.run_fl(activation="tanh",
     data=..., init_weights=..., save=False)        [engine change PENDING SAMUDI REVIEW]
  -> candidate global model, in memory
  -> export_global_document / save_global_document (create-only, fl_candidates/)
  -> import_and_build_candidate: fl_adapter.import_global_weights, then
     global_update.build_candidate / write_candidate (never inside models/)

save=False keeps Member 3's committed results/ files (engine_<tag>_tanh_seed<seed>.csv)
from being overwritten; Member 3's export_results.py is not used because it
rewrites models/runs/<run_id>/. Files are written only under the git-ignored
fl_candidates/ folder, never over an existing file and never into a protected
directory. Candidates are NOT_VALIDATED and NOT_DEPLOYED. The engine's "global_val_*" numbers are computed on
TRAINING runs that Model v1 was also trained on: a development check, not
independent validation and not an accuracy claim.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import fl_data as fd
from .export import N_PARAMS, PARAM_LAYOUT, file_sha256, flatten, weights_sha256
from .features import FEATURES
from .fl_adapter import (COMPAT_KEYS, IncompatibleModelError, check_weights, feature_config_sha256,
                         import_global_weights, local_model_metadata)
from .global_update import build_candidate, candidate_version, write_candidate

CANDIDATE_STATUS = "CANDIDATE_NOT_VALIDATED"
NOT_VALIDATED = "NOT_VALIDATED"
NOT_DEPLOYED = "NOT_DEPLOYED"
GLOBAL_DOC_FORMAT = "c2-fl-global-1"
CANDIDATES_DIR = fd.COMPONENT / "fl_candidates"          # git-ignored; integration output only
PROTECTED_DIRS = (fd.MODELS_DIR, fd.COMPONENT / "results", fd.COMPONENT / "notebooks", fd.COMPONENT / "data",
                  fd.COMPONENT / "firmware", fd.MEMBER3_ROOT)


class UnsafePathError(ValueError):
    """A write target outside the allowed folder or inside a protected one."""
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


def run_engine(init_weights, data, seed=0, engine=None, **engine_kwargs):
    """Member 3's run_fl from the given starting weights with the fixed
    integration settings (tanh, save=False). Returns the engine DataFrame."""
    fixed = sorted(set(engine_kwargs) & set(FIXED_ARGS))
    if fixed:
        raise ValueError(f"{fixed} are fixed by the wrapper (tanh, Member 2 data and weights, save=False)")
    base_sha = weights_sha256(init_weights)
    engine = engine or member3_engine()
    check_data(data, engine.pv_data.SITES)
    engine_kwargs.setdefault("verbose", False)
    df = engine.run_fl(seed=seed, activation="tanh", data=data, init_weights=init_weights, save=False, **engine_kwargs)
    if weights_sha256(init_weights) != base_sha:
        raise AssertionError("starting weights changed during the FL run")
    return df


def run_from_model_v1(dataset_path=None, data=None, seed=0, engine=None, models_dir=fd.MODELS_DIR, **engine_kwargs):
    """Start Member 3's FL from Model v1. Returns {candidate, history, updates}.
    Give dataset_path (TRAINING rows are taken by fl_data) or ready data."""
    fixed = sorted(set(engine_kwargs) & set(FIXED_ARGS))
    if fixed:
        raise ValueError(f"{fixed} are fixed by the wrapper (tanh, Member 2 data and weights, save=False)")
    contract, _, w_v1 = fd.load_frozen_model(models_dir)
    source = "caller-supplied (OPERATION exclusion not checked here)"
    if data is None:
        if dataset_path is None:
            raise ValueError("give dataset_path or data")
        data, source = fd.member3_get_data(dataset_path, models_dir), "fl_data.member3_get_data (TRAINING only)"
    df = run_engine(w_v1, data, seed, engine, **engine_kwargs)
    models = df.attrs["models"]
    last = max(models, key=lambda v: int(v.split(".")[1]))
    candidate = make_candidate(contract, models[last], last, seed, dict(engine_kwargs, save=False), source)
    return {"candidate": candidate, "history": df, "updates": df.attrs["updates"]}


def make_candidate(contract, weights, engine_version, seed, engine_options, source):
    """In-memory candidate for the engine's global weights (lineage to the base contract)."""
    final = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
    if flatten(final).size != N_PARAMS or not np.isfinite(flatten(final)).all():
        raise ValueError("engine returned an invalid global model")
    return {"status": CANDIDATE_STATUS, "engine_version": engine_version, "engine_start_version": "v1.0",
            "base_model_version": contract["model_version"], "base_weights_sha256": contract["weights_sha256"],
            "feature_version": contract["feature_version"],
            "feature_config_sha256": feature_config_sha256(contract), "labels": list(contract["labels"]),
            "activation": "tanh", "param_layout": contract["param_layout"], "seed": seed,
            "engine_options": engine_options, "data_source": source,
            "operation_rows_used": 0 if source.startswith("fl_data") else None,
            "validation_note": VALIDATION_NOTE, "weights": final, "weights_sha256": weights_sha256(final)}


# ---------------------------------------------------------------- Task 3: export / save / import

def global_version(candidate, run_id):
    """Unique, path-safe version: <run_id>-<engine version>-<first 8 hex of the weights SHA>."""
    v = f"{run_id}-{candidate['engine_version']}-{candidate['weights_sha256'][:8]}"
    candidate_version(v)                                  # same character rule as candidate names
    return v


def export_global_document(candidate, run_id, models_dir=fd.MODELS_DIR):
    """Global-model JSON document: Member 3's {"version", "weights"} plus the
    compatibility metadata of the frozen contract (fl_adapter.local_model_metadata),
    lineage and status. Weights are the 229 values in W1, b1, W2, b2 row-major."""
    contract, _, _ = fd.load_frozen_model(models_dir)
    meta = local_model_metadata(contract, file_sha256(Path(models_dir) / "model_contract.json"))
    for k in ("base_model_version", "base_weights_sha256", "feature_version", "feature_config_sha256",
              "activation", "labels", "param_layout"):
        expected = {"base_model_version": meta["model_version"],
                    "base_weights_sha256": meta["weights_sha256"]}.get(k, meta.get(k))
        if candidate.get(k) != expected:
            raise ValueError(f"candidate {k} does not match the frozen base model")
    w = {k: np.asarray(v, dtype=np.float64) for k, v in candidate["weights"].items()}
    check_weights(w)
    if weights_sha256(w) != candidate["weights_sha256"]:
        raise ValueError("candidate weights_sha256 does not match its weights")
    doc = {"document_format": GLOBAL_DOC_FORMAT, "version": global_version(candidate, run_id),
           "weights": {k: w[k].tolist() for k, _ in PARAM_LAYOUT}, "weights_sha256": candidate["weights_sha256"],
           "base_model_version": meta["model_version"], "base_weights_sha256": meta["weights_sha256"],
           **{k: meta[k] for k in COMPAT_KEYS},
           "status": NOT_VALIDATED, "deployment_status": NOT_DEPLOYED,
           "lineage": {k: candidate[k] for k in ("engine_version", "engine_start_version", "seed", "engine_options",
                                                 "data_source", "operation_rows_used", "validation_note")},
           "review_note": "engine.py integration change PENDING SAMUDI REVIEW"}
    return doc


def safe_target(path, allowed_root=CANDIDATES_DIR):
    """Resolved path (symlinks followed) that is inside allowed_root and not
    inside any protected directory; raises UnsafePathError otherwise."""
    root, target = Path(allowed_root).resolve(), Path(path).resolve()
    for p in (q.resolve() for q in PROTECTED_DIRS):
        for x in (root, target):
            if x == p or p in x.parents:
                raise UnsafePathError(f"{x} is inside the protected directory {p}")
    if target != root and root not in target.parents:
        raise UnsafePathError(f"{target} is outside {root}")
    return target


def save_global_document(doc, allowed_root=CANDIDATES_DIR):
    """Write <allowed_root>/global_<version>.json create-only (never overwrites,
    never follows a symlink at the target). Returns (path, file SHA-256)."""
    candidate_version(doc["version"])
    root = safe_target(allowed_root, allowed_root)
    root.mkdir(parents=True, exist_ok=True)
    path = safe_target(root / f"global_{doc['version']}.json", root)
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    fd_ = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
    with os.fdopen(fd_, "w", encoding="utf-8") as f:
        f.write(text)
    return path, file_sha256(path)


def import_and_build_candidate(doc_or_path, allowed_root=CANDIDATES_DIR, models_dir=fd.MODELS_DIR):
    """Exported document -> fl_adapter.import_global_weights (all compatibility
    checks, base version c2-local-v1) -> global_update.build_candidate /
    write_candidate under allowed_root. Returns (candidate contract, dir, report)."""
    doc = json.loads(Path(doc_or_path).read_text(encoding="utf-8")) if isinstance(doc_or_path, (str, Path)) \
        else doc_or_path
    contract, _, _ = fd.load_frozen_model(models_dir)
    contract_sha = file_sha256(Path(models_dir) / "model_contract.json")
    meta = local_model_metadata(contract, contract_sha)
    w, report = import_global_weights(doc, meta, expected_base_version=meta["model_version"])
    if not report["sha256_supplied"]:
        raise IncompatibleModelError("MISSING_HASH", "weights_sha256 is required for integration documents")
    if doc.get("base_weights_sha256") != meta["weights_sha256"]:
        raise IncompatibleModelError("BASE_MISMATCH", "document was not produced from the frozen Model v1 weights")
    source = {"document_format": doc.get("document_format"), "global_version": doc["version"],
              "lineage": doc.get("lineage"), "deployment_status": NOT_DEPLOYED,
              "review_note": doc.get("review_note")}
    cand = build_candidate(contract, contract_sha, w, doc["version"], source)
    d = write_candidate(safe_target(allowed_root, allowed_root), cand)
    return cand, d, report

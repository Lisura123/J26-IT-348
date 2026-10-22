"""Member 2 side of federated-learning integration (adapters only).

Reference: Member 3's branch feature/mongodb-migration (audited read-only,
496f5be): component3-federated-backend/src/mlp.py (LAYOUT, flatten),
src/compression.py (int8_dense) and data/shared/fl_weight_updates.jsonl
(update record format). Member 3's code is not changed.

What this module does:
  - exports the local model's compatibility metadata (feature version,
    FeatureConfig hash, labels, activation, layout, hashes),
  - flattens / unflattens the 229 parameters in Member 3's layout,
  - computes weight deltas and builds a local update record in Member 3's
    fl_weight_updates.jsonl format, with sample counts and compatibility
    fields added,
  - verifies update records and IMPORTS GLOBAL WEIGHTS ONLY WITH COMPLETE
    COMPATIBILITY METADATA. Member 3's current global files
    ({"version", "weights"}) carry no feature version, preprocessing hash or
    activation, and their FL trains on raw8 inputs, so they are REJECTED:
    loading them would give silently wrong predictions (TRAINING_EXPORT.md 7).

Nothing here trains, aggregates or contacts the FL server.
"""
import hashlib
import json
import math

import numpy as np

from .contract import LABELS
from .export import N_PARAMS, PARAM_LAYOUT, contract_weights, flatten, unflatten, weights_sha256

ADAPTER_VERSION = "c2-fl-adapter-1"
MEMBER3_LAYOUT = [["W1", [8, 16]], ["b1", [16]], ["W2", [16, 5]], ["b2", [5]]]   # src/mlp.py LAYOUT
RAW8_FEATURE_SPACE = "raw8"
MAX_ABS_WEIGHT = 64.0      # sanity guard against corrupt values, not a tuned limit (frozen max |w| = 1.34)
COMPAT_KEYS = ("feature_version", "feature_config_sha256", "activation", "output_activation", "labels", "features",
               "param_layout", "n_params")


class IncompatibleModelError(ValueError):
    """Raised with a machine-readable code when weights or updates are refused."""

    def __init__(self, code, detail):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def feature_config_sha256(contract):
    """SHA-256 of the canonical JSON of the frozen FeatureConfig (scaler,
    V_ref, gates, units, order). Identifies the preprocessing."""
    return hashlib.sha256(_canonical(contract["feature_config"])).hexdigest()


def layout():
    return [[n, list(s)] for n, s in PARAM_LAYOUT]


def local_model_metadata(contract, contract_sha256):
    """Compatibility metadata of the local model for Member 3 (no weights)."""
    if layout() != MEMBER3_LAYOUT:
        raise AssertionError("Component 2 layout differs from Member 3's LAYOUT")
    return {"adapter_version": ADAPTER_VERSION, "model_version": contract["model_version"],
            "contract_version": contract["contract_version"], "contract_sha256": contract_sha256,
            "weights_sha256": contract["weights_sha256"], "feature_version": contract["feature_version"],
            "feature_config_sha256": feature_config_sha256(contract), "features": list(contract["features"]),
            "labels": list(contract["labels"]), "activation": contract["activation"],
            "output_activation": contract["output_activation"], "param_layout": contract["param_layout"],
            "n_params": contract["n_params"], "flatten_order": contract["flatten_order"],
            "input_space": "pv8-v1 features from the frozen FeatureConfig (not raw8)",
            "training_rows": "status-OK readings only (quality gates applied)"}


def check_weights(weights, max_abs=MAX_ABS_WEIGHT):
    """Shapes, finiteness and a magnitude sanity bound. Returns the flat vector."""
    try:
        vec = flatten({k: np.asarray(v, dtype=np.float64) for k, v in weights.items()})
    except (ValueError, TypeError) as e:
        raise IncompatibleModelError("BAD_LAYOUT", str(e)) from None
    if not np.isfinite(vec).all():
        raise IncompatibleModelError("NON_FINITE_WEIGHTS", "NaN or inf in the weights")
    if np.abs(vec).max() > max_abs:
        raise IncompatibleModelError("WEIGHT_OUT_OF_RANGE", f"max |w| {np.abs(vec).max():.3g} > {max_abs}")
    return vec


# [GUIDE] H. Federated learning: a site sends only the CHANGE of its weights
# (local - base, 229 values), never its raw data. Same convention as Member 3's train_local.
# STATUS: Member 2-side adapters are host-tested only; no end-to-end FL round with
# Member 3's server has run (docs/FL_INTEGRATION.md).
def compute_delta(local_weights, base_weights):
    """Flattened local - base (Member 3's train_local convention: new - old)."""
    return check_weights(local_weights) - check_weights(base_weights)


def int8_dense(vec):
    """Member 3's compression.int8_dense, reproduced exactly (np.round = half
    to even). Returns (int8 values, scale)."""
    vec = np.asarray(vec, dtype=np.float64)
    scale = np.abs(vec).max() / 127.0 or 1.0
    return np.clip(np.round(vec / scale), -127, 127).astype(np.int64), float(scale)


def update_sha256(update):
    body = {k: v for k, v in update.items() if k != "update_sha256"}
    return hashlib.sha256(_canonical(body)).hexdigest()


# [GUIDE] H. Packs the 229-value delta (int8 + scale) with compatibility metadata
# (feature version, feature-config SHA-256, activation, label order) in Member 3's update format.
def build_update(*, local_weights, base_contract, base_contract_sha256, base_model_version, update_id, round_id,
                 site_id, device_id, num_samples, class_counts):
    """A local update in Member 3's fl_weight_updates.jsonl format (update_id,
    round_id, site_id, device_id, base_model_version, param_layout,
    delta_int8, int8_scale) plus Component 2 fields: sample counts and the
    compatibility metadata a server needs to refuse mixed preprocessing.
    num_samples counts status-OK readings used for local training."""
    if int(num_samples) <= 0 or sum(int(v) for v in class_counts.values()) != int(num_samples):
        raise ValueError("num_samples must be > 0 and equal the sum of class_counts")
    if set(class_counts) - set(LABELS):
        raise ValueError(f"unknown classes {sorted(set(class_counts) - set(LABELS))}")
    meta = local_model_metadata(base_contract, base_contract_sha256)
    q, scale = int8_dense(compute_delta(local_weights, contract_weights(base_contract)))
    upd = {"update_id": str(update_id), "round_id": int(round_id), "site_id": str(site_id),
           "device_id": str(device_id), "base_model_version": str(base_model_version),
           "param_layout": MEMBER3_LAYOUT, "delta_int8": q.tolist(), "int8_scale": scale,
           "num_samples": int(num_samples), "class_counts": {k: int(v) for k, v in class_counts.items()},
           "adapter_version": ADAPTER_VERSION, "base_weights_sha256": base_contract["weights_sha256"],
           **{k: meta[k] for k in COMPAT_KEYS}}
    upd["update_sha256"] = update_sha256(upd)
    return upd


def _compat_errors(doc, expected):
    """List of (code, detail) for metadata that is missing or differs."""
    missing = [k for k in COMPAT_KEYS if k not in doc]
    if missing:
        return [("MISSING_COMPATIBILITY_METADATA",
                 f"missing {missing}; a file without them may come from raw8 FL training and is not accepted")]
    errs = []
    if doc["feature_version"] != expected["feature_version"]:
        code = "INCOMPATIBLE_PREPROCESSING" if doc["feature_version"] == RAW8_FEATURE_SPACE else "FEATURE_VERSION"
        errs.append((code, f"feature_version {doc['feature_version']!r} != {expected['feature_version']!r}"))
    for k in COMPAT_KEYS[1:]:
        if doc[k] != expected[k]:
            errs.append(("INCOMPATIBLE_" + k.upper(), f"{k} differs"))
    return errs


def verify_update(update, base_metadata):
    """Raise IncompatibleModelError unless the update record is complete,
    intact and computed against the expected base model."""
    for code, detail in _compat_errors(update, base_metadata):
        raise IncompatibleModelError(code, detail)
    need = {"update_id", "round_id", "site_id", "device_id", "base_model_version", "param_layout", "delta_int8",
            "int8_scale", "num_samples", "class_counts", "base_weights_sha256", "update_sha256"}
    if need - set(update):
        raise IncompatibleModelError("MISSING_FIELDS", f"{sorted(need - set(update))}")
    if update["update_sha256"] != update_sha256(update):
        raise IncompatibleModelError("INTEGRITY", "update_sha256 does not match the record")
    if update["param_layout"] != MEMBER3_LAYOUT:
        raise IncompatibleModelError("BAD_LAYOUT", "param_layout differs from W1, b1, W2, b2")
    if update["base_weights_sha256"] != base_metadata["weights_sha256"]:
        raise IncompatibleModelError("BASE_MISMATCH", "update was computed against different base weights")
    q = update["delta_int8"]
    if len(q) != N_PARAMS or not all(isinstance(v, int) and -127 <= v <= 127 for v in q):
        raise IncompatibleModelError("BAD_DELTA", f"delta_int8 must be {N_PARAMS} integers in -127..127")
    s = update["int8_scale"]
    if not (isinstance(s, (int, float)) and math.isfinite(s) and s > 0):
        raise IncompatibleModelError("BAD_SCALE", "int8_scale must be finite and > 0")
    if update["num_samples"] <= 0 or sum(update["class_counts"].values()) != update["num_samples"]:
        raise IncompatibleModelError("BAD_SAMPLE_COUNT", "num_samples must equal the sum of class_counts")
    return True


def import_global_weights(doc, expected_metadata, expected_base_version=None, max_abs=MAX_ABS_WEIGHT):
    """Validate a global model document before any use. Returns
    (weights dict, report). The document must carry the complete
    compatibility metadata of expected_metadata (local_model_metadata) and
    'version' and 'weights'; optional 'weights_sha256' and
    'base_model_version' are checked when present (base_model_version is
    required when expected_base_version is given). Raises
    IncompatibleModelError otherwise. Nothing is written."""
    if not isinstance(doc, dict) or "weights" not in doc or "version" not in doc:
        raise IncompatibleModelError("BAD_DOCUMENT", "expected an object with 'version' and 'weights'")
    for code, detail in _compat_errors(doc, expected_metadata):
        raise IncompatibleModelError(code, detail)
    if expected_base_version is not None and doc.get("base_model_version") != expected_base_version:
        raise IncompatibleModelError("BASE_VERSION",
                                     f"base_model_version {doc.get('base_model_version')!r} != {expected_base_version!r}")
    if set(doc["weights"]) != {n for n, _ in PARAM_LAYOUT}:
        raise IncompatibleModelError("BAD_LAYOUT", f"weights keys {sorted(doc['weights'])}")
    vec = check_weights(doc["weights"], max_abs)
    w = unflatten(vec)
    sha = weights_sha256(w)
    if "weights_sha256" in doc and doc["weights_sha256"] != sha:
        raise IncompatibleModelError("INTEGRITY", "weights_sha256 does not match the weights")
    report = {"accepted": True, "global_version": str(doc["version"]), "weights_sha256": sha,
              "feature_version": doc["feature_version"], "feature_config_sha256": doc["feature_config_sha256"],
              "max_abs_weight": float(np.abs(vec).max()), "sha256_supplied": "weights_sha256" in doc}
    return w, report


def apply_update(base_weights, update):
    """Base + dequantised delta (int8 * scale), as Member 3's server sees it.
    For checks only; aggregation stays Member 3's responsibility."""
    delta = np.asarray(update["delta_int8"], dtype=np.float64) * float(update["int8_scale"])
    return unflatten(check_weights(base_weights) + delta)


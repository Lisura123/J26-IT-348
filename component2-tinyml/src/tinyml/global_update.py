"""Member 2 side of the future global-model update workflow.

  compatible global Float32 weights (fl_adapter.import_global_weights)
    -> versioned candidate contract (frozen FeatureConfig copied unchanged)
    -> candidate validation (layout, labels, preprocessing, hashes, probes)
    -> INT8 quantisation with the int8-v1 rule (needs approved calibration rows)
    -> reference-prediction verification (Float32 vs INT8)
    -> candidate directory with a manifest (input for ota.build_package)

Rules: the frozen model in models/ is never written; every candidate gets a
new version and its own directory and is never overwritten; the
preprocessing (scaler, V_ref, gates) is never changed; quantisation follows
the existing int8-v1 calibration rule (max-abs over status-OK TRAINING rows)
and refuses to run without explicitly supplied calibration rows, so no data
use happens implicitly. Running it on real data is a separate, approved step.
A candidate is CANDIDATE_NOT_VALIDATED: no accuracy is measured here.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .contract import LABELS
from .export import contract_weights, file_sha256, forward, weights_sha256
from .fl_adapter import check_weights, feature_config_sha256
from .quantize import compare_predictions, float_reference_labels, int8_predict_many, params_sha256, quantize_model

PIPELINE_VERSION = "c2-global-update-1"
CANDIDATE_KIND = "FL_GLOBAL_CANDIDATE"
CANDIDATE_STATUS = "CANDIDATE_NOT_VALIDATED"
FROZEN_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
UNCHANGED_KEYS = ("contract_version", "feature_version", "features", "feature_units", "feature_config", "scaler",
                  "labels", "activation", "output_activation", "param_layout", "n_params", "flatten_order",
                  "architecture", "rules")


def _dump(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def candidate_version(global_version):
    v = str(global_version)
    if not v or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in v):
        raise ValueError(f"bad global version {global_version!r}")
    return f"c2-fl-{v}"


def build_candidate(base_contract, base_contract_sha256, weights, global_version, source):
    """New contract: base contract with the global weights. Preprocessing,
    labels, layout and activation are copied unchanged; reference-probe
    probabilities are recomputed for the new weights."""
    vec = check_weights(weights)
    w = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
    cand = copy.deepcopy(base_contract)
    cand["model_version"] = candidate_version(global_version)
    if cand["model_version"] == base_contract["model_version"]:
        raise ValueError("candidate version equals the base version")
    cand["status"] = CANDIDATE_STATUS
    cand["artifact_kind"] = CANDIDATE_KIND
    cand["weights"] = {k: w[k].tolist() for k in ("W1", "b1", "W2", "b2")}
    cand["weights_sha256"] = weights_sha256(w)
    _, probs = forward(w, base_contract["reference_probes"]["inputs_scaled"])
    cand["reference_probes"] = dict(base_contract["reference_probes"], probabilities=probs.tolist(),
                                    labels=[LABELS[i] for i in probs.argmax(axis=1)])
    cand["candidate"] = {"pipeline_version": PIPELINE_VERSION, "global_version": str(global_version),
                         "base_model_version": base_contract["model_version"],
                         "base_contract_sha256": base_contract_sha256,
                         "base_weights_sha256": base_contract["weights_sha256"],
                         "feature_config_sha256": feature_config_sha256(base_contract),
                         "source": source, "max_abs_weight": float(np.abs(vec).max()),
                         "evaluation": "NOT_EVALUATED. No accuracy is measured by this pipeline."}
    cand.pop("trained_on", None)
    cand.pop("training", None)
    validate_candidate(cand, base_contract)
    return cand


def validate_candidate(cand, base_contract):
    def need(cond, msg):
        if not cond:
            raise ValueError(f"candidate: {msg}")
    need(cand.get("status") == CANDIDATE_STATUS and cand.get("artifact_kind") == CANDIDATE_KIND, "status/kind")
    for k in UNCHANGED_KEYS:
        need(cand.get(k) == base_contract.get(k), f"{k} differs from the base contract (must stay frozen)")
    need(feature_config_sha256(cand) == cand["candidate"]["feature_config_sha256"], "feature_config hash")
    w = contract_weights(cand)
    check_weights(w)
    need(weights_sha256(w) == cand["weights_sha256"], "weights_sha256 does not match the weights")
    _, probs = forward(w, cand["reference_probes"]["inputs_scaled"])
    need(np.allclose(probs, cand["reference_probes"]["probabilities"], atol=1e-9), "reference probes")
    need(cand["model_version"] != base_contract["model_version"], "version must be new")
    return True


def quantize_candidate(cand, cand_sha256, z_calib, calibration_info):
    """int8-v1 rule on explicitly supplied calibration rows (float32 scaled
    status-OK TRAINING rows). Refuses without rows or with another window."""
    if z_calib is None or len(z_calib) == 0:
        raise ValueError("INT8 quantisation needs explicitly supplied, approved TRAINING calibration rows")
    if (calibration_info or {}).get("window") != "TRAINING":
        raise ValueError("calibration rows must come from the TRAINING window (OPERATION is never used)")
    return quantize_model(cand, cand_sha256, z_calib, calibration_info)


def verify_candidate(cand, q, z64, z32):
    """Float32 reference vs INT8 on the given rows (agreement and OPEN/SHORT
    changes). Not an accuracy measurement."""
    ref = float_reference_labels(z64, cand)
    int8 = int8_predict_many(z32, q)
    cmp = compare_predictions(ref, int8["label_index"])
    return {"rows": cmp["rows"], "agreement": cmp["agreement"], "changed": cmp["changed"],
            "safety_changes_total": cmp["safety_changes_total"], "change_pairs": cmp["change_pairs"],
            "input_clips": int8["input_clips"], "lut_clips": int8["lut_clips"],
            "note": "INT8 vs Float32 agreement of the candidate; not independent accuracy"}


def write_candidate(out_root, cand, q=None, verification=None):
    """candidates/<model_version>/ under out_root, never inside models/ and
    never over an existing candidate. Returns the directory."""
    out_root = Path(out_root).resolve()
    if out_root == FROZEN_MODELS_DIR or FROZEN_MODELS_DIR in out_root.parents:
        raise ValueError("candidates are never written inside the frozen models/ directory")
    d = out_root / "candidates" / cand["model_version"]
    if d.exists():
        raise FileExistsError(f"{d} exists; candidates are never overwritten")
    d.mkdir(parents=True)
    (d / "model_contract.json").write_text(_dump(cand), encoding="utf-8")
    files = ["model_contract.json"]
    if q is not None:
        if q.contract_sha256 != file_sha256(d / "model_contract.json"):
            raise ValueError("INT8 parameters were not derived from this candidate contract file")
        (d / "quant_params.json").write_text(q.to_json(), encoding="utf-8")
        files.append("quant_params.json")
    if verification is not None:
        (d / "verification.json").write_text(_dump(verification), encoding="utf-8")
        files.append("verification.json")
    manifest = {"pipeline_version": PIPELINE_VERSION, "model_version": cand["model_version"],
                "status": CANDIDATE_STATUS, "weights_sha256": cand["weights_sha256"],
                "base_model_version": cand["candidate"]["base_model_version"],
                "quant_params_sha256": params_sha256(q) if q is not None else None,
                "files": {f: file_sha256(d / f) for f in files}}
    (d / "manifest.json").write_text(_dump(manifest), encoding="utf-8")
    return d


def contract_text_sha256(cand):
    """SHA-256 of the contract exactly as write_candidate stores it."""
    return hashlib.sha256(_dump(cand).encode("utf-8")).hexdigest()

"""Export of the locally trained model (Phase 8).

Writes the model contract (ARCHITECTURE.md section 7.2) and its companion
files, and reads them back with integrity checks:

  model_contract.json     labels, features, complete FeatureConfig, activation,
                          parameter layout, float weights, provenance
  feature_config.json     the complete fitted FeatureConfig on its own
  training_metadata.json  training facts, library versions, run time
  manifest.json           SHA-256 of every file above

Three kinds of model artifact are kept apart:

  LOCAL_TRAINED_WEIGHTS   produced here: float weights of one centrally
                          trained model (TRAINING window only)
  FL weight update        NOT produced here. A delta against a published
                          global model in Member 3's update format
                          (fl_weight_updates.jsonl: base_model_version,
                          param_layout, delta_int8, int8_scale)
  INT8 deployment build   NOT produced here. A quantised ESP32 build, chosen
                          later from on-device measurements (ARCHITECTURE 9.1)

The parameter layout copies Member 3's FL layout (component3-federated-backend
src/mlp.py, LAYOUT): W1 (8, 16), b1 (16,), W2 (16, 5), b2 (5,), flattened
row-major in that order, 229 values. Forward pass: softmax(tanh(x W1 + b1) W2 + b2)
with class index i = LABELS[i].
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contract import LABELS
from .features import (FEATURE_UNITS, FEATURE_VERSION, FEATURES, PREDICTION_STATUS, STATUS_OK,
                       FeatureConfig, transform)

CONTRACT_VERSION = "1.0"
MODEL_VERSION = "c2-local-v1"
ARTIFACT_KIND = "LOCAL_TRAINED_WEIGHTS"
STATUS_TRAINED = "TRAINED"
PARAM_LAYOUT = (("W1", (8, 16)), ("b1", (16,)), ("W2", (16, 5)), ("b2", (5,)))
N_PARAMS = sum(int(np.prod(s)) for _, s in PARAM_LAYOUT)            # 229
ACTIVATION = "tanh"
OUTPUT_ACTIVATION = "softmax"
LAYOUT_SOURCE = "component3-federated-backend/src/mlp.py LAYOUT (Member 3 federated learning)"
N_PROBES = 5
FILES = ("model_contract.json", "feature_config.json", "training_metadata.json")

INTEGRATION_STATUS = {
    "federated_learning": "NOT_SUBMITTED. Parameter layout matches Member 3's FL layout. "
                          "Member 3's current FL pipeline uses the 8 raw sensor columns with its "
                          "own scaler, not pv8-v1 features, so these weights are layout-compatible "
                          "only. Not accepted by, or tested with, the FL server.",
    "fl_weight_update": "NOT_PRODUCED. Updates are deltas in Member 3's update format, created "
                        "during FL rounds.",
    "int8_deployment": "NOT_PRODUCED. Float32 vs INT8 is decided from ESP32 measurements.",
    "esp32_deployment": "NOT_DEPLOYED",
    "field_validation": "NOT_VALIDATED",
    "relay_control": "NOT_IMPLEMENTED. relay_recommendation is an advisory proposal only "
                     "(ARCHITECTURE.md 7.5); nothing is triggered automatically.",
}

RULES_PLACEHOLDER = {
    "status": "PLACEHOLDER_NOT_IMPLEMENTED",
    "open_i_ma": None, "short_v": None, "k_oc": None, "i_sc_min": None, "persist_n": None,
    "note": "Planned electrical confirmation layer for OPEN_CIRCUIT / SHORT_CIRCUIT "
            "(PHASE7_RESULTS.md decision 3). Thresholds need Member 1's hardware requirements.",
}


# ---------------------------------------------------------------- weights

# [GUIDE] E. Export of the trained parameters
# WHAT: reads W1, b1, W2, b2 out of the scikit-learn model.
# WHY:  the ESP32, Member 3's FL and the digital twin cannot run scikit-learn; they need
#       plain numbers in a fixed layout and a fixed class order.
# HOW:  scikit-learn sorts classes alphabetically, so W2/b2 columns are reordered here to
#       LABELS order; otherwise every exported class would be mislabelled.
def extract_weights(model):
    """Weights of a fitted 8-16-5 tanh MLPClassifier in PARAM_LAYOUT order.

    scikit-learn orders its output units by sorted class name. The columns
    of W2 and b2 are reordered here so that output i means LABELS[i], the
    order used by Member 3's FL, the ESP32 and the digital twin."""
    if model.activation != ACTIVATION or model.out_activation_ != OUTPUT_ACTIVATION:
        raise ValueError(f"expected {ACTIVATION}/{OUTPUT_ACTIVATION}, got "
                         f"{model.activation}/{model.out_activation_}")
    classes = list(model.classes_)
    if sorted(classes) != sorted(LABELS):
        raise ValueError(f"model classes {classes} differ from {list(LABELS)}")
    order = [classes.index(l) for l in LABELS]
    w = {"W1": model.coefs_[0], "b1": model.intercepts_[0],
         "W2": model.coefs_[1][:, order], "b2": model.intercepts_[1][order]}
    w = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}
    check_shapes(w)
    return w


def check_shapes(weights):
    if set(weights) != {n for n, _ in PARAM_LAYOUT}:
        raise ValueError(f"weights must be {[n for n, _ in PARAM_LAYOUT]}, got {sorted(weights)}")
    for name, shape in PARAM_LAYOUT:
        if np.shape(weights[name]) != shape:
            raise ValueError(f"{name} has shape {np.shape(weights[name])}, expected {shape}")


# [GUIDE] H. The 229 parameters as one vector: W1 (8x16=128) + b1 (16) + W2 (16x5=80) + b2 (5),
# each row-major. This is the same layout as Member 3's FL code (PARAM_LAYOUT above).
def flatten(weights):
    """229-vector: W1, b1, W2, b2, each row-major (same as Member 3's mlp.flatten)."""
    check_shapes(weights)
    return np.concatenate([np.asarray(weights[n], dtype=np.float64).ravel() for n, _ in PARAM_LAYOUT])


def unflatten(vec):
    vec = np.asarray(vec, dtype=np.float64)
    if vec.shape != (N_PARAMS,):
        raise ValueError(f"expected {N_PARAMS} values, got shape {vec.shape}")
    out, i = {}, 0
    for name, shape in PARAM_LAYOUT:
        size = int(np.prod(shape))
        out[name] = vec[i:i + size].reshape(shape).copy()
        i += size
    return out


def weights_sha256(weights):
    """SHA-256 of the flattened weights as little-endian float64 bytes."""
    return hashlib.sha256(flatten(weights).astype("<f8").tobytes()).hexdigest()


# [GUIDE] E. Inference maths (training vs inference): training CHANGES the weights;
# inference only USES them. p = softmax(tanh(x W1 + b1) W2 + b2); label = argmax(p),
# confidence = max(p).
def forward(weights, X):
    """Reference forward pass. Returns (logits, probabilities), LABELS order."""
    X = np.asarray(X, dtype=np.float64)
    h = np.tanh(X @ weights["W1"] + weights["b1"])
    logits = h @ weights["W2"] + weights["b2"]
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return logits, e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- contract

def build_contract(result, dataset_sha256, split_version, code_commit, env):
    """Model contract for a train.train_final() result."""
    cfg, model = result["config"], result["model"]
    if not cfg.is_fitted:
        raise ValueError("feature config is not fitted")
    weights = extract_weights(model)
    probes = np.asarray(result["X"][:N_PROBES], dtype=np.float64)
    _, probs = forward(weights, probes)
    facts = result["facts"]
    return {
        "status": STATUS_TRAINED,
        "artifact_kind": ARTIFACT_KIND,
        "contract_version": CONTRACT_VERSION,
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "labels": list(LABELS),
        "features": list(FEATURES),
        "feature_units": list(FEATURE_UNITS),
        "feature_config": json.loads(cfg.to_json()),
        "scaler": cfg.scaler,
        "architecture": {"inputs": 8, "hidden": 16, "outputs": 5},
        "activation": ACTIVATION,
        "output_activation": OUTPUT_ACTIVATION,
        "param_layout": [[n, list(s)] for n, s in PARAM_LAYOUT],
        "param_layout_source": LAYOUT_SOURCE,
        "flatten_order": "W1, b1, W2, b2; each row-major (C order)",
        "n_params": N_PARAMS,
        "weights": {k: v.tolist() for k, v in weights.items()},
        "weights_sha256": weights_sha256(weights),
        "weights_dtype_trained": str(model.coefs_[0].dtype),
        "rules": RULES_PLACEHOLDER,
        "trained_on": {
            "dataset": "pv_fault_dataset.csv",
            "dataset_sha256": dataset_sha256,
            "split": split_version,
            "window": facts["window"],
            "rows": facts["rows"],
            "runs": facts["runs"],
            "ok_rows_used": facts["ok_rows_used"],
            "git_commit": code_commit,
        },
        "training": {k: facts[k] for k in ("model", "feature_set", "seed", "seed_policy",
                                            "hyperparameters", "n_iter", "converged")},
        "environment": env,
        "reference_probes": {
            "note": "first scaled TRAINING inputs and the expected probabilities, for "
                    "checking other implementations (C, FL, twin) of the forward pass",
            "inputs_scaled": probes.tolist(),
            "probabilities": probs.tolist(),
            "labels": [LABELS[i] for i in probs.argmax(axis=1)],
        },
        "integration_status": INTEGRATION_STATUS,
    }


def training_metadata(result, contract, created_utc=None):
    meta = {"model_version": contract["model_version"], "artifact_kind": ARTIFACT_KIND,
            **result["facts"], "dataset_sha256": contract["trained_on"]["dataset_sha256"],
            "split": contract["trained_on"]["split"], "git_commit": contract["trained_on"]["git_commit"],
            "environment": contract["environment"], "weights_sha256": contract["weights_sha256"],
            "operation_window_used": False}
    if created_utc:
        meta["created_utc"] = created_utc
    return meta


def _dump(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# [GUIDE] E/H. Where the trained model is SAVED: models/model_contract.json (weights, layout,
# feature config, provenance), feature_config.json, training_metadata.json and
# manifest.json (SHA-256 of each file, so any change is detected).
def write_artifacts(out_dir, contract, cfg, metadata):
    """Write the three files and a manifest of their SHA-256 hashes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    validate_contract(contract)
    (out / "model_contract.json").write_text(_dump(contract), encoding="utf-8")
    cfg.save(out / "feature_config.json")
    (out / "training_metadata.json").write_text(_dump(metadata), encoding="utf-8")
    manifest = {"model_version": contract["model_version"], "artifact_kind": ARTIFACT_KIND,
                "weights_sha256": contract["weights_sha256"],
                "files": {name: file_sha256(out / name) for name in FILES}}
    (out / "manifest.json").write_text(_dump(manifest), encoding="utf-8")
    return [out / n for n in (*FILES, "manifest.json")]


def load_contract(path):
    with open(path, encoding="utf-8") as f:
        contract = json.load(f)
    validate_contract(contract)
    return contract


def contract_weights(contract):
    w = {k: np.asarray(v, dtype=np.float64) for k, v in contract["weights"].items()}
    check_shapes(w)
    return w


def contract_feature_config(contract):
    return FeatureConfig.from_json(json.dumps(contract["feature_config"]))


def validate_contract(contract):
    """Raise ValueError if the contract is inconsistent with this code or itself."""
    def need(cond, msg):
        if not cond:
            raise ValueError(f"model contract: {msg}")
    need(contract.get("status") == STATUS_TRAINED, f"status is {contract.get('status')!r}, not TRAINED")
    need(contract.get("artifact_kind") == ARTIFACT_KIND, "artifact_kind")
    need(contract["labels"] == list(LABELS), f"labels {contract['labels']} != {list(LABELS)}")
    need(contract["features"] == list(FEATURES), f"features {contract['features']} != {list(FEATURES)}")
    need(contract["feature_version"] == FEATURE_VERSION, "feature_version")
    need(contract["activation"] == ACTIVATION and contract["output_activation"] == OUTPUT_ACTIVATION,
         "activation")
    need(contract["param_layout"] == [[n, list(s)] for n, s in PARAM_LAYOUT], "param_layout")
    need(contract["n_params"] == N_PARAMS, f"n_params {contract['n_params']} != {N_PARAMS}")
    w = contract_weights(contract)
    need(flatten(w).size == N_PARAMS and np.isfinite(flatten(w)).all(), "weights not finite")
    need(weights_sha256(w) == contract["weights_sha256"], "weights_sha256 does not match the weights")
    cfg = contract_feature_config(contract)
    need(cfg.is_fitted, "feature_config is not fitted")
    need(list(cfg.features) == contract["features"], "feature_config order differs from features")
    need(cfg.scaler == contract["scaler"], "scaler differs from feature_config.scaler")
    need(len(cfg.scaler["mean"]) == len(cfg.scaler["std"]) == len(FEATURES), "scaler length")
    need(contract["trained_on"]["window"] == "TRAINING", "model must be trained on the TRAINING window")
    need((cfg.fitted_on or {}).get("fold") == "final_TRAINING", "feature_config not fitted on final_TRAINING")
    p = contract["reference_probes"]
    _, probs = forward(w, p["inputs_scaled"])
    need(np.allclose(probs, p["probabilities"], atol=1e-9), "reference probes do not reproduce")
    need([LABELS[i] for i in probs.argmax(axis=1)] == p["labels"], "reference probe labels")
    return True


def verify_artifacts(out_dir):
    """Check the manifest hashes and the contract; return the loaded contract."""
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for name, sha in manifest["files"].items():
        actual = file_sha256(out / name)
        if actual != sha:
            raise ValueError(f"{name}: SHA-256 {actual} does not match manifest {sha}")
    contract = load_contract(out / "model_contract.json")
    if manifest["weights_sha256"] != contract["weights_sha256"]:
        raise ValueError("manifest weights_sha256 differs from the contract")
    cfg = FeatureConfig.load(out / "feature_config.json")
    if json.loads(cfg.to_json()) != contract["feature_config"]:
        raise ValueError("feature_config.json differs from the contract's feature_config")
    meta = json.loads((out / "training_metadata.json").read_text(encoding="utf-8"))
    if meta["weights_sha256"] != contract["weights_sha256"] or meta["operation_window_used"]:
        raise ValueError("training_metadata.json is inconsistent with the contract")
    return contract


# ---------------------------------------------------------------- inference from the contract

# [GUIDE] E. Inference from the saved contract only (no scikit-learn): raw sensor rows ->
# gates -> 8 features -> forward pass. Skipped readings get no label (never NORMAL by default).
# Used for the final OPERATION test (holdout.predict_frozen).
def predict_raw(df, contract):
    """Classify raw sensor rows with only the contract (no scikit-learn).

    Every row gets a prediction_status; label, confidence and probabilities
    are None for skipped rows (never NORMAL by default)."""
    cfg = contract_feature_config(contract)
    w = contract_weights(contract)
    X, status = transform(df, cfg)
    ok = (status == STATUS_OK).to_numpy()
    out = pd.DataFrame(index=df.index)
    out["prediction_status"] = status.map(PREDICTION_STATUS)
    out["predicted_label"] = None
    out["confidence"] = None
    for l in LABELS:
        out[f"prob_{l.lower()}"] = None
    if ok.any():
        _, probs = forward(w, X)
        rows = df.index[ok]
        out.loc[rows, "predicted_label"] = [LABELS[i] for i in probs.argmax(axis=1)]
        out.loc[rows, "confidence"] = probs.max(axis=1)
        for i, l in enumerate(LABELS):
            out.loc[rows, f"prob_{l.lower()}"] = probs[:, i]
    return out

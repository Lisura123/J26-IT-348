"""Independent OPERATION evaluation of the frozen model (Phase 9).

The protocol is fixed in docs/OPERATION_EVAL.md before the official run.
This module only reads the committed artifacts in models/ and never fits
anything: no V_ref or scaler refit, no training, no threshold or
hyperparameter change. It does not import fit_pipeline, train_final or
make_model.

  1. verify_frozen_model   committed artifacts, pinned SHA-256 values
  2. evaluation_rows       OPERATION rows = temporal 'test' side of the split
  3. inference_frame       drops label and health_score before inference
  4. export.predict_raw    contract-only NumPy inference with the quality gates
  5. metrics               classification, coverage, coverage-adjusted recall,
                           run level, per site, OPEN/SHORT safety analysis,
                           run-level bootstrap confidence intervals
  6. latency               host-CPU NumPy timing on TRAINING rows (not ESP32)

run_evaluation() refuses the OPERATION window unless confirm_final_test=True.
A dry run on TRAINING rows (dry_run_training=True) checks the plumbing and is
labelled DRY_RUN; it is not a result.
"""
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .contract import GROUP_COLUMN, LABEL_COLUMN, LABELS, LEAKAGE_COLUMNS, WINDOW_COLUMN
from .evaluate import SAFETY_CLASSES, classification_metrics, coverage, run_level
from .export import contract_feature_config, contract_weights, file_sha256, forward, predict_raw, verify_artifacts
from .features import STATUS_OK, input_status, transform
from .train import DATASET_SHA256, environment

PROTOCOL_VERSION = "operation-eval-v1"
EVAL_WINDOW = "OPERATION"
DRY_RUN_WINDOW = "TRAINING"
STATUS_FINAL = "FINAL_TEST_OPERATION"
STATUS_DRY_RUN = "DRY_RUN_TRAINING_NOT_A_RESULT"

# Artifacts committed in 49b96b5 (docs/TRAINING_EXPORT.md section 10)
PINNED_FILES = {
    "model_contract.json": "752f62285cca62b59a629686fd5b941b748a811c78808751c04595ee3452cb2a",
    "feature_config.json": "421b5d4ed7e2092b6da3e2f2f48e000a0ecda5ef0542e5a660c65fd607128a86",
    "training_metadata.json": "76389fc889c9b0097ba0d944c32e535930c835c7d15beef018b1d00f48f5501c",
    "manifest.json": "989cf6ece67ad8be044dd981ae1d8a9ba830798b5b56dbed5c45d7b71303a526",
}
PINNED_WEIGHTS_SHA256 = "94e7ab9384a1906ec2cf0d3176b435e700f15cd99552fbc6c71519c648ecc3a7"
EXPECTED_ROWS = 3000
EXPECTED_RUNS = 50

FORBIDDEN_INPUTS = (LABEL_COLUMN, *LEAKAGE_COLUMNS)       # never reach inference
READING_INTERVAL_S = 5.0
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 42
LATENCY_N = 1000
LATENCY_WARMUP = 100
ELECTRICAL_COLUMNS = ("current_ma", "voltage_v", "irradiance_lux")
# Phase 7 pooled results for physics8 + MLP (docs/PHASE7_RESULTS.md); context only
PHASE7_CONTEXT = {"cv_pooled_macro_f1": 0.981, "loso_training_pooled_macro_f1": 0.980,
                  "note": "TRAINING-window model-selection estimates, shown for context only"}


# ---------------------------------------------------------------- 1. frozen model

def verify_frozen_model(model_dir, pinned=PINNED_FILES, pinned_weights=PINNED_WEIGHTS_SHA256):
    """verify_artifacts() plus the SHA-256 values committed with the model.
    Returns the contract; raises ValueError on any difference."""
    model_dir = Path(model_dir)
    contract = verify_artifacts(model_dir)
    if pinned:
        for name, sha in pinned.items():
            actual = file_sha256(model_dir / name)
            if actual != sha:
                raise ValueError(f"{name}: SHA-256 {actual} is not the committed {sha}")
    if pinned_weights and contract["weights_sha256"] != pinned_weights:
        raise ValueError(f"weights SHA-256 {contract['weights_sha256']} is not the committed {pinned_weights}")
    return contract


# ---------------------------------------------------------------- 2-3. rows and inputs

# [GUIDE] F. Selects the final test rows: the OPERATION window (= the "test" side of the
# 50/50 temporal split). It checks that the window column and the split file agree.
def evaluation_rows(df, assignments, window=EVAL_WINDOW):
    """Index of the evaluation rows. OPERATION = temporal 'test' side of the
    split file; TRAINING (dry run only) = temporal 'train' side. The window
    column and the split file must agree, and no run may be on both sides."""
    side = df[GROUP_COLUMN].map(assignments["temporal"])
    if side.isna().any():
        raise ValueError("runs missing from the split file")
    want = {EVAL_WINDOW: "test", DRY_RUN_WINDOW: "train"}[window]
    by_split = df.index[(side == want).to_numpy()]
    by_window = df.index[(df[WINDOW_COLUMN] == window).to_numpy()]
    if not by_split.equals(by_window):
        raise ValueError(f"{window} rows do not match the temporal '{want}' side of the split file")
    other = df.index.difference(by_split)
    shared = set(df.loc[by_split, GROUP_COLUMN]) & set(df.loc[other, GROUP_COLUMN])
    if shared:
        raise ValueError(f"runs on both sides of the temporal split: {sorted(shared)[:5]}")
    return by_split


def inference_frame(df):
    """Copy of the rows without label and health_score, so they can never
    reach inference."""
    out = df.drop(columns=[c for c in FORBIDDEN_INPUTS if c in df.columns])
    assert not set(FORBIDDEN_INPUTS) & set(out.columns)
    return out


# ---------------------------------------------------------------- 4. inference

def predict_frozen(df_part, contract):
    """Contract-only inference on label-free inputs. Returns the predictions
    and the input status of every row (gating only; nothing is fitted)."""
    frame = inference_frame(df_part)
    pred = predict_raw(frame, contract)
    status = input_status(frame, contract_feature_config(contract))
    return pred, status


# ---------------------------------------------------------------- 5. metrics

def _ordered(part):
    """Rows of one run in time order (reading number, else timestamp)."""
    for col in ("reading", "timestamp_utc"):
        if col in part.columns:
            return part.sort_values(col, kind="stable")
    return part


def max_streak(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return int(best)


def coverage_adjusted_recall(labels, predicted):
    """Correct predictions / all readings of the class, skipped ones included
    (a skipped reading is a reading that was not detected)."""
    labels, predicted = np.asarray(labels, dtype=object), np.asarray(predicted, dtype=object)
    out = {}
    for l in LABELS:
        n = int((labels == l).sum())
        hit = int(((labels == l) & (predicted == l)).sum())
        out[l] = {"total": n, "correct": hit, "recall": hit / n if n else None}
    return out


def per_site(df_cls, y, p):
    d = pd.DataFrame({"site": df_cls["site_id"].to_numpy(), "ok": y == p})
    return {s: {"classified": int(len(g)), "accuracy": float(g.ok.mean())} for s, g in d.groupby("site")}


def _median(series):
    return float(series.median()) if len(series) else None


def safety_analysis(df_part, pred):
    """False and missed OPEN_CIRCUIT / SHORT_CIRCUIT predictions.

    Labels are dataset labels (provenance unresolved), and every OPEN/SHORT
    output is an ML prediction that has not been electrically confirmed: the
    confirmation rules are placeholders, so confirmed electrical faults are
    not assessed. Streak lengths are descriptive and must not be used to set
    persist_n."""
    labels = df_part[LABEL_COLUMN].to_numpy(dtype=object)
    predicted = pred["predicted_label"].to_numpy(dtype=object)
    classified = (pred["prediction_status"] == "CLASSIFIED").to_numpy()
    out = {}
    for fault in SAFETY_CLASSES:
        is_fault, said_fault = labels == fault, classified & (predicted == fault)
        false_pos = said_fault & ~is_fault
        non_fault_classified = int((classified & ~is_fault).sum())
        fp_runs = []
        for run, g in df_part.loc[false_pos].groupby(GROUP_COLUMN):
            whole = _ordered(df_part[df_part[GROUP_COLUMN] == run])
            flags = (pred.loc[whole.index, "predicted_label"] == fault).to_numpy()
            fp_runs.append({"run_id": run, "site_id": str(g["site_id"].iloc[0]) if "site_id" in g else None,
                            "true_label": str(g[LABEL_COLUMN].iloc[0]), "false_predictions": int(len(g)),
                            "run_readings": int(len(whole)), "max_consecutive": max_streak(flags)})
        true_hits = df_part.loc[classified & is_fault & (predicted == fault)]
        fp_rows = df_part.loc[false_pos]
        missed_runs = []
        for run, g in df_part.loc[is_fault].groupby(GROUP_COLUMN):
            g = _ordered(g)
            p = pred.loc[g.index]
            cls = (p["prediction_status"] == "CLASSIFIED").to_numpy()
            hit = cls & (p["predicted_label"] == fault).to_numpy()
            first = int(np.argmax(hit)) if hit.any() else None
            missed_as = p.loc[cls & ~hit, "predicted_label"].value_counts().sort_index()
            missed_runs.append({
                "run_id": run, "site_id": str(g["site_id"].iloc[0]) if "site_id" in g else None,
                "readings": int(len(g)), "classified": int(cls.sum()), "correct": int(hit.sum()),
                "missed": int((cls & ~hit).sum()), "missed_as": {k: int(v) for k, v in missed_as.items()},
                "skipped": int((~cls).sum()),
                "recall": float(hit.sum() / cls.sum()) if cls.sum() else None,
                "coverage_adjusted_recall": float(hit.sum() / len(g)),
                "first_detection_reading": None if first is None else first + 1,
                "time_to_first_detection_s": None if first is None else first * READING_INTERVAL_S})
        total_fault = int(is_fault.sum())
        cls_fault = int((classified & is_fault).sum())
        hits = int((classified & is_fault & (predicted == fault)).sum())
        out[fault] = {
            "false_predictions": {
                "count": int(false_pos.sum()),
                "rate_per_1000_classified_non_fault": (1000.0 * false_pos.sum() / non_fault_classified
                                                       if non_fault_classified else None),
                "by_true_label": {k: int(v) for k, v in
                                  pd.Series(labels[false_pos]).value_counts().sort_index().items()},
                "runs": fp_runs,
                "electrical_context_median": {
                    "false_predictions": {c: _median(fp_rows[c]) for c in ELECTRICAL_COLUMNS if c in fp_rows},
                    "true_fault_detections": {c: _median(true_hits[c]) for c in ELECTRICAL_COLUMNS if c in true_hits},
                    "note": "descriptive only; no threshold is derived from these values"},
            },
            "missed_faults": {
                "fault_readings": total_fault, "classified": cls_fault, "correct": hits,
                "missed": cls_fault - hits, "skipped": total_fault - cls_fault,
                "recall": hits / cls_fault if cls_fault else None,
                "coverage_adjusted_recall": hits / total_fault if total_fault else None,
                "runs": missed_runs},
            "electrical_confirmation": "NOT_ASSESSED: confirmation rules are placeholders "
                                       "(PHASE7_RESULTS.md decision 3); predictions are unconfirmed",
        }
    return out


def _confusion(y, p):
    idx = {l: i for i, l in enumerate(LABELS)}
    m = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for a, b in zip(y, p):
        m[idx[a], idx[b]] += 1
    return m


def confusion_scores(m):
    """Accuracy, macro-F1 (classes present only, zero_division=0) and
    per-class recall from a confusion matrix (rows = true)."""
    m = np.asarray(m, dtype=float)
    tp, rows, cols = np.diag(m), m.sum(axis=1), m.sum(axis=0)
    present = rows > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(cols > 0, tp / cols, 0.0)
        rec = np.where(rows > 0, tp / rows, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    total = m.sum()
    return {"accuracy": float(tp.sum() / total) if total else None,
            "macro_f1": float(f1[present].mean()) if present.any() else None,
            "recall": {l: (float(rec[i]) if present[i] else None) for i, l in enumerate(LABELS)}}


def bootstrap_ci(y, p, runs, run_labels, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED, level=0.95):
    """Run-level (cluster) bootstrap on classified readings. Runs are
    resampled with replacement within each true class, so every class keeps
    its number of runs. Deterministic for a given seed."""
    y, p, runs = np.asarray(y, dtype=object), np.asarray(p, dtype=object), np.asarray(runs, dtype=object)
    per_run = {r: _confusion(y[runs == r], p[runs == r]) for r in pd.unique(runs)}
    strata = {}
    for r in per_run:
        strata.setdefault(run_labels[r], []).append(r)
    strata = {k: sorted(v) for k, v in sorted(strata.items())}
    rng = np.random.default_rng(seed)
    keys = ("accuracy", "macro_f1", "OPEN_CIRCUIT_recall", "SHORT_CIRCUIT_recall")
    samples = {k: [] for k in keys}
    for _ in range(b):
        m = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
        for runs_in in strata.values():
            for i in rng.integers(0, len(runs_in), len(runs_in)):
                m += per_run[runs_in[i]]
        s = confusion_scores(m)
        samples["accuracy"].append(s["accuracy"])
        samples["macro_f1"].append(s["macro_f1"])
        for f in SAFETY_CLASSES:
            samples[f"{f}_recall"].append(s["recall"][f])
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out = {"method": "run-level bootstrap, stratified by true class, percentile interval",
           "resamples": b, "seed": seed, "level": level,
           "runs_per_class": {k: len(v) for k, v in strata.items()}}
    for k, v in samples.items():
        v = np.array([x for x in v if x is not None], dtype=float)
        out[k] = {"low": float(np.percentile(v, lo)), "high": float(np.percentile(v, hi))} if len(v) else None
    return out


# ---------------------------------------------------------------- 6. latency

def _stats_us(ns):
    us = np.asarray(ns, dtype=float) / 1000.0
    return {"n": int(len(us)), "mean_us": float(us.mean()), "median_us": float(np.median(us)),
            "p95_us": float(np.percentile(us, 95)), "p99_us": float(np.percentile(us, 99))}


def host_latency(raw_rows, contract, n=LATENCY_N, warmup=LATENCY_WARMUP):
    """Host-CPU timing of NumPy inference on raw TRAINING rows (labels
    removed). Not an ESP32 measurement."""
    frame = inference_frame(raw_rows)
    cfg, w = contract_feature_config(contract), contract_weights(contract)
    X, _ = transform(frame, cfg)
    ok_rows = frame.loc[(input_status(frame, cfg) == STATUS_OK).to_numpy()]
    if not len(X):
        raise ValueError("no status-OK rows for the latency measurement")
    for i in range(warmup):
        forward(w, X[i % len(X):i % len(X) + 1])
    fwd = []
    for i in range(n):
        x = X[i % len(X):i % len(X) + 1]
        t0 = time.perf_counter_ns(); forward(w, x); fwd.append(time.perf_counter_ns() - t0)
    single = ok_rows.iloc[[0]]
    for _ in range(min(warmup, 10)):
        predict_raw(single, contract)
    pipe = []
    for i in range(min(n, 200)):
        row = ok_rows.iloc[[i % len(ok_rows)]]
        t0 = time.perf_counter_ns(); predict_raw(row, contract); pipe.append(time.perf_counter_ns() - t0)
    t0 = time.perf_counter_ns(); predict_raw(frame, contract); batch_ns = time.perf_counter_ns() - t0
    return {
        "scope": "HOST_CPU_NOT_ESP32",
        "note": "Python/NumPy on the evaluation host. ESP32 flash, RAM and latency must be measured "
                "on the device and are not estimated from these numbers.",
        "rows": "TRAINING rows (labels and health_score removed); OPERATION is not used for timing",
        "forward_pass_single_reading": _stats_us(fwd),
        "pandas_reference_pipeline_single_reading": _stats_us(pipe),
        "batched_pipeline": {"readings": int(len(frame)), "total_ms": batch_ns / 1e6,
                             "per_reading_us": batch_ns / 1000.0 / len(frame)},
        "host": {"machine": platform.machine(), "processor": platform.processor() or None,
                 "cpu_count": os.cpu_count(), "platform": platform.platform()},
    }


# ---------------------------------------------------------------- runner

# [GUIDE] F. FINAL INDEPENDENT TEST on the OPERATION window (scripts/06_evaluate_operation.py)
# WHAT: the frozen, already-trained model predicts 3,000 unseen OPERATION readings.
# WHY:  this is the honest performance estimate: the model never saw these runs, and no
#       threshold, seed or hyperparameter was chosen with them. It was run ONCE
#       (2026-10-16) and the script refuses to run it again.
# EVIDENCE: results/operation_eval.md / .json (2,940 classified, 60 LOW_LIGHT skipped,
#           accuracy 0.979, macro-F1 0.978, confusion matrix, OPEN/SHORT recall 1.000).
def run_evaluation(df, assignments, model_dir, dataset_sha256, split_meta=None,
                   confirm_final_test=False, dry_run_training=False, code_commit=None,
                   expected_dataset_sha256=DATASET_SHA256, pinned=PINNED_FILES,
                   pinned_weights=PINNED_WEIGHTS_SHA256, expected_rows=EXPECTED_ROWS,
                   expected_runs=EXPECTED_RUNS, bootstrap_b=BOOTSTRAP_B, latency_n=LATENCY_N):
    """Evaluate the frozen model once. OPERATION needs confirm_final_test=True;
    dry_run_training=True evaluates TRAINING rows instead (plumbing check)."""
    if dry_run_training and confirm_final_test:
        raise ValueError("choose either the dry run or the final test, not both")
    if not dry_run_training and not confirm_final_test:
        raise PermissionError("the OPERATION window is the independent final test; "
                              "pass confirm_final_test=True to run it once")
    if expected_dataset_sha256 and dataset_sha256 != expected_dataset_sha256:
        raise ValueError(f"dataset SHA-256 {dataset_sha256} is not the validated {expected_dataset_sha256}")
    contract = verify_frozen_model(model_dir, pinned, pinned_weights)
    window = DRY_RUN_WINDOW if dry_run_training else EVAL_WINDOW
    idx = evaluation_rows(df, assignments, window)
    part = df.loc[idx]
    if expected_rows is not None and len(part) != expected_rows:
        raise ValueError(f"{window}: {len(part)} rows, expected {expected_rows}")
    if expected_runs is not None and part[GROUP_COLUMN].nunique() != expected_runs:
        raise ValueError(f"{window}: {part[GROUP_COLUMN].nunique()} runs, expected {expected_runs}")

    # [GUIDE] F. Inference on the OPERATION rows (labels removed first; nothing is fitted).
    pred, status = predict_frozen(part, contract)
    cls = (pred["prediction_status"] == "CLASSIFIED").to_numpy()
    y = part.loc[cls, LABEL_COLUMN].to_numpy(dtype=object)
    p = pred.loc[cls, "predicted_label"].to_numpy(dtype=object)
    runs = part.loc[cls, GROUP_COLUMN].to_numpy(dtype=object)
    # [GUIDE] F. Accuracy, precision, recall, macro-F1 and confusion matrix of the final test.
    metrics = classification_metrics(y, p)
    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics["confusion_matrix_row_normalised"] = np.where(
            cm.sum(axis=1, keepdims=True) > 0, cm / cm.sum(axis=1, keepdims=True), 0.0).round(4).tolist()
    run_labels = part.groupby(GROUP_COLUMN)[LABEL_COLUMN].first().to_dict()

    latency_rows = df.loc[evaluation_rows(df, assignments, DRY_RUN_WINDOW)]
    files = {n: file_sha256(Path(model_dir) / n) for n in PINNED_FILES}
    return {
        "status": STATUS_DRY_RUN if dry_run_training else STATUS_FINAL,
        "protocol_version": PROTOCOL_VERSION,
        "window": window,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "dataset_sha256": dataset_sha256,
            "split_version": (split_meta or {}).get("split_version"),
            "model_version": contract["model_version"],
            "weights_sha256": contract["weights_sha256"],
            "artifact_sha256": files,
            "training_commit": contract["trained_on"]["git_commit"],
            "evaluation_code_commit": code_commit,
            "environment": environment(),
        },
        "frozen": {"refit": False, "retrained": False, "thresholds_changed": False,
                   "inference": "contract-only NumPy (export.predict_raw)",
                   "inputs_excluded": list(FORBIDDEN_INPUTS)},
        "rows": {"total": int(len(part)), "runs": int(part[GROUP_COLUMN].nunique()),
                 "runs_per_class": {l: int(sum(v == l for v in run_labels.values())) for l in LABELS}},
        "coverage": coverage(part, status),
        "classification": metrics,
        "coverage_adjusted_recall": coverage_adjusted_recall(
            part[LABEL_COLUMN].to_numpy(dtype=object),
            pred["predicted_label"].where(pred["prediction_status"] == "CLASSIFIED").to_numpy(dtype=object)),
        "run_level": run_level(y, p, runs),
        "per_site": per_site(part.loc[cls], y, p) if "site_id" in part else None,
        "safety": safety_analysis(part, pred),
        "bootstrap": bootstrap_ci(y, p, runs, run_labels, b=bootstrap_b),
        "latency": host_latency(latency_rows, contract, n=latency_n),
        "phase7_context": PHASE7_CONTEXT,
        "caveats": [
            "OPEN/SHORT outputs are ML predictions; electrical confirmation is not implemented.",
            "Relay isolation is not implemented; nothing is actuated.",
            "Labels are dataset labels; dataset provenance is UNRESOLVED (DATA_PROVENANCE.md).",
            "Latency is host CPU, not ESP32.",
            "Low-light and saturation thresholds are placeholders and were not tuned.",
        ],
    }

"""Pruning feasibility study E1 (no retraining, no fine-tuning).

Derives pruned variants of the frozen 8-16-5 tanh MLP and compares them
with the frozen model on TRAINING rows. Nothing here changes the frozen
Float32 contract, the INT8 parameters or any deployable artifact; the
pruned weights exist only in memory and in the study report.

  unstructured  global magnitude pruning of the 208 weights (W1, W2) at
                10 / 20 / 30 %; biases are kept; the 229-value layout stays,
                pruned weights become 0
  structured    removal of the k = 2 / 3 / 5 hidden units with the lowest
                importance |W1[:, j]|_1 * |W2[j, :]|_1; the architecture
                becomes 8-(16-k)-5 (experimental: breaks the 229-value FL layout)

Pruned INT8 variants are derived from the frozen INT8 parameters
(models/int8/quant_params.json) without re-quantisation: the same scales,
rescaling constants (M1, SH1) and tanh table; pruned weights are set to 0
and removed hidden units drop their qW1 column, qb1 entry and qW2 row. Storage and multiply-add figures are
analytical ESTIMATES, not hardware measurements.
"""
import math

import numpy as np

from .export import contract_weights, forward
from .quantize import F32, K_MAX, K_MIN, compare_predictions, quantize_input

STUDY_VERSION = "prune-e1-v1"
UNSTRUCTURED_FRACTIONS = (0.10, 0.20, 0.30)
STRUCTURED_UNITS = (2, 3, 5)
N_IN, N_HID, N_OUT = 8, 16, 5
INT8_CONSTANT_BYTES = 13                  # INV_SX 4 + OUT_SCALE 4 + M1 4 + SH1 1
TANH_LUT_BYTES = 256
ACCEPT_MIN_AGREEMENT = 0.995
ACCEPT_MAX_SAFETY_CHANGES = 0


# ---------------------------------------------------------------- pruning methods

def _copy(w):
    return {k: np.array(v, dtype=np.float64, copy=True) for k, v in w.items()}


# [GUIDE] G. Pruning study E1 (docs/PRUNING.md): can weights be removed WITHOUT retraining?
# Unstructured = set the 10/20/30 % smallest weights to 0 (shape stays 8-16-5, 229 values).
# RESULT: predictions unchanged on TRAINING rows for every variant, but no measured ESP32
# benefit (estimated saving at most 36-85 B) -> "evaluated, not adopted"
# (results/pruning_study.md, docs/PRUNING.md section 9.4).
def unstructured_prune(weights, fraction):
    """Zero the round(fraction * 208) smallest-magnitude weights of W1 and W2
    together. Ties are broken by layer (W1 first) and row-major position,
    so the result is deterministic. Biases are not pruned. Returns
    (pruned weights, boolean keep-masks for W1 and W2)."""
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0, 1)")
    w = _copy(weights)
    flat = np.concatenate([np.abs(w["W1"]).ravel(), np.abs(w["W2"]).ravel()])
    k = int(round(fraction * flat.size))
    order = np.lexsort((np.arange(flat.size), flat))      # by |w|, then position
    keep = np.ones(flat.size, dtype=bool)
    keep[order[:k]] = False
    n1 = w["W1"].size
    m1, m2 = keep[:n1].reshape(w["W1"].shape), keep[n1:].reshape(w["W2"].shape)
    w["W1"] = np.where(m1, w["W1"], 0.0)
    w["W2"] = np.where(m2, w["W2"], 0.0)
    return w, {"W1": m1, "W2": m2, "removed": k}


def unit_importance(weights):
    """|W1[:, j]|_1 * |W2[j, :]|_1 for each hidden unit j."""
    return np.abs(weights["W1"]).sum(axis=0) * np.abs(weights["W2"]).sum(axis=1)


# [GUIDE] G. Structured = remove whole hidden neurons (16 -> 14/13/11). This changes the
# 229-parameter layout, so it would break the FL layout (PRUNING.md criterion 4).
def structured_prune(weights, n_units):
    """Remove the n_units hidden units with the lowest importance (ties: lower
    index first). Returns (reduced weights, sorted removed unit indices)."""
    if not 0 <= n_units < weights["W1"].shape[1]:
        raise ValueError("n_units out of range")
    imp = unit_importance(weights)
    removed = sorted(np.lexsort((np.arange(imp.size), imp))[:n_units].tolist())
    keep = [j for j in range(imp.size) if j not in removed]
    w = _copy(weights)
    return {"W1": w["W1"][:, keep], "b1": w["b1"][keep], "W2": w["W2"][keep, :], "b2": w["b2"]}, removed


# ---------------------------------------------------------------- INT8 for any hidden size

def int8_params_for(q, masks=None, removed_units=()):
    """INT8 arrays of a pruned variant, taken from the frozen INT8 parameters q:
    weights pruned in float are set to 0 in qW1 / qW2 (masks from
    unstructured_prune), and removed hidden units drop their qW1 column, qb1
    entry and qW2 row. No value is re-quantised, so masks=None and no removed
    units give exactly the frozen INT8 model."""
    qw1, qb1, qw2, qb2, _ = q.arrays()
    if masks is not None:
        qw1, qw2 = np.where(masks["W1"], qw1, 0), np.where(masks["W2"], qw2, 0)
    keep = [j for j in range(qw1.shape[1]) if j not in set(removed_units)]
    return {"qw1": qw1[:, keep], "qb1": qb1[keep], "qw2": qw2[keep, :], "qb2": qb2}


def _rshift_rnd_vec(v, n):
    half = np.int64(1) << np.int64(n - 1)
    return np.where(v >= 0, (v + half) >> np.int64(n), -((-v + half) >> np.int64(n)))


def int8_labels(z32, p, q):
    """Vectorised INT8 path for any hidden size; same rules as
    quantize.int8_predict. Returns (label indices, input clips, lut clips)."""
    z32 = np.asarray(z32, dtype=F32)
    qx = np.empty(z32.shape, dtype=np.int64)
    in_clips = 0
    for r, z in enumerate(z32):
        qx[r], c = quantize_input(z, q.inv_sx)
        in_clips += c
    acc1 = qx @ p["qw1"] + p["qb1"]
    assert np.abs(acc1).max() < 2 ** 31
    k = _rshift_rnd_vec(acc1 * np.int64(q.m1), q.sh1)
    lut_clips = int(((k < K_MIN) | (k > K_MAX)).sum())
    hq = np.asarray(q.tanh_lut, dtype=np.int64)[np.clip(k, K_MIN, K_MAX) - K_MIN]
    acc2 = hq @ p["qw2"] + p["qb2"]
    assert np.abs(acc2).max() < 2 ** 31
    return acc2.argmax(axis=1), in_clips, lut_clips             # argmax: first maximum


# ---------------------------------------------------------------- estimates

def storage_estimate(n_hidden, nnz_weights, n_dense_weights=None):
    """Parameter storage in bytes (ESTIMATE). Dense stores every weight;
    bitmap stores 1 bit per weight position plus the non-zero values; the
    index list stores a 1-byte position plus the value per non-zero weight.
    Decoding code is NOT included."""
    n_dense = n_dense_weights if n_dense_weights is not None else N_IN * n_hidden + n_hidden * N_OUT
    n_bias = n_hidden + N_OUT
    bitmap = math.ceil(n_dense / 8)
    out = {}
    for name, elem, bias, extra in (("float32", 4, 4, 0), ("int8", 1, 4, TANH_LUT_BYTES + INT8_CONSTANT_BYTES)):
        out[name] = {"dense": n_dense * elem + n_bias * bias + extra,
                     "bitmap_sparse": bitmap + nnz_weights * elem + n_bias * bias + extra,
                     "index_list_sparse": nnz_weights * (1 + elem) + n_bias * bias + extra}
    return out


def mac_estimate(n_hidden, nnz_weights):
    return {"dense_multiply_adds": N_IN * n_hidden + n_hidden * N_OUT,
            "zero_skipping_multiply_adds": int(nnz_weights), "tanh_evaluations": n_hidden}


# ---------------------------------------------------------------- study

def _variant(name, method, level, w, masks, removed_units, z64, z32, ref, ref8, y, q, fl_layout):
    n_hidden = w["W1"].shape[1]
    nnz = int((w["W1"] != 0).sum() + (w["W2"] != 0).sum())
    _, p = forward(w, z64)
    float_cmp = compare_predictions(ref, p.argmax(axis=1), y)
    p8 = int8_params_for(q, masks, removed_units)
    lab8, in_c, lut_c = int8_labels(z32, p8, q)
    int8_cmp = compare_predictions(ref, lab8, y)
    int8_vs_frozen_int8 = compare_predictions(ref8, lab8)
    nnz8 = int((p8["qw1"] != 0).sum() + (p8["qw2"] != 0).sum())
    return {
        "name": name, "method": method, "level": level, "hidden_units": n_hidden,
        "removed_units": list(removed_units), "parameters_stored_dense": N_IN * n_hidden + n_hidden * N_OUT + n_hidden + N_OUT,
        "nonzero_weights_float": nnz, "nonzero_weights_int8": nnz8,
        "fl_layout": fl_layout,
        "storage_estimate_bytes": {"float32": storage_estimate(n_hidden, nnz)["float32"],
                                   "int8": storage_estimate(n_hidden, nnz8)["int8"]},
        "multiply_add_estimate": {"float32": mac_estimate(n_hidden, nnz), "int8": mac_estimate(n_hidden, nnz8)},
        "pruned_float_vs_frozen_float": float_cmp,
        "pruned_int8_vs_frozen_float": int8_cmp,
        "pruned_int8_vs_frozen_int8": int8_vs_frozen_int8,
        "int8_input_clips": in_c, "int8_lut_clips": lut_c,
        "e1_checks": e1_checks(float_cmp, int8_cmp),
    }


def e1_checks(float_cmp, int8_cmp):
    """E1 criteria 1-2 (docs/PRUNING.md section 5). Criteria 3 (measured
    on-device benefit) and 4 (FL layout) are not decided by E1."""
    ok = lambda c: (c["safety_changes_total"] <= ACCEPT_MAX_SAFETY_CHANGES       # noqa: E731
                    and c["agreement"] is not None and c["agreement"] >= ACCEPT_MIN_AGREEMENT)
    return {"float32": {"safety_changes_zero": float_cmp["safety_changes_total"] == 0,
                        "agreement_at_least_0.995": float_cmp["agreement"] >= ACCEPT_MIN_AGREEMENT,
                        "passed": ok(float_cmp)},
            "int8": {"safety_changes_zero": int8_cmp["safety_changes_total"] == 0,
                     "agreement_at_least_0.995": int8_cmp["agreement"] >= ACCEPT_MIN_AGREEMENT,
                     "passed": ok(int8_cmp)}}


def run_study(contract, q, z64, z32, y):
    """All variants on status-OK TRAINING rows. z64: float64 scaled inputs
    (features.transform); z32: float32 device inputs (device_reference)."""
    if len(z64) != len(z32) or len(z64) != len(y) or not len(y):
        raise ValueError("z64, z32 and y must describe the same non-empty rows")
    w0 = contract_weights(contract)
    _, p0 = forward(w0, z64)
    ref = p0.argmax(axis=1)
    ref8, _, _ = int8_labels(z32, int8_params_for(q), q)
    variants = [_variant("baseline", "none", 0, w0, None, (), z64, z32, ref, ref8, y, q, "unchanged (229 values)")]
    for f in UNSTRUCTURED_FRACTIONS:
        w, mask = unstructured_prune(w0, f)
        variants.append(_variant(f"unstructured_{round(f * 100)}pct", "unstructured_magnitude", f, w, mask, (),
                                 z64, z32, ref, ref8, y, q,
                                 "unchanged shape (229 values); FL would need a shared fixed mask"))
    for k in STRUCTURED_UNITS:
        w, removed = structured_prune(w0, k)
        variants.append(_variant(f"structured_minus{k}", "structured_hidden_units", k, w, None, removed,
                                 z64, z32, ref, ref8, y, q,
                                 f"BROKEN: 8-{N_HID - k}-5 has {8 * (N_HID - k) + (N_HID - k) * 5 + N_HID - k + 5} "
                                 "values, not 229 (experimental only)"))
    return {"study_version": STUDY_VERSION, "rows": int(len(y)), "variants": variants,
            "baseline_int8_agreement_with_float": compare_predictions(ref, ref8)["agreement"],
            "conclusion_rule": "E1 checks 1-2 only. Adoption also needs a measured on-device benefit "
                               "(criterion 3) and an unchanged FL layout (criterion 4); neither is "
                               "decided by E1."}

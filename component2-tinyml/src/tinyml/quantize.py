"""Custom INT8 quantisation of the frozen 8-16-5 tanh MLP (Phase 10).

System: Float32 preprocessing (device_reference.py, pv_fault.c) followed by
INT8 neural inference (this module, pv_mlp_int8.c). The float contract in
models/ stays the source of truth; the INT8 parameters are a deployment
derivative that records the contract and weights SHA-256 it came from.

Arithmetic (docs/QUANTIZATION.md section 3), identical in Python and C:

  rnd(v)            round half away from zero
  rshift_rnd(v, n)  v / 2^n, round half away from zero (integers only)
  sat(v, lo, hi)    clamp, counted

  s_x  = A_x / 127      A_x  = max |z| over status-OK TRAINING rows
  s_W1 = max|W1| / 127  s_W2 = max|W2| / 127
  s_z1 = A_z1 / 127     A_z1 = max |z W1 + b1| over the same rows
  s_h  = 1 / 127

  qx   = sat(rnd(float32(z * INV_SX)), -127, 127)            INV_SX = float32(1/s_x)
  acc1 = qb1 + sum qx*qW1                                    int32
  k    = sat(rshift_rnd(int64(acc1) * M1, SH1), -128, 127)  M1*2^-SH1 = s_x*s_W1/s_z1
  hq   = TANH_LUT[k + 128]                                   int8
  acc2 = qb2 + sum hq*qW2                                    int32
  label = argmax acc2 (lowest index on ties); logits = float32(acc2) * OUT_SCALE

Calibration uses TRAINING rows only. Nothing here trains or refits.
"""
import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from .contract import LABELS
from .evaluate import SAFETY_CLASSES, classification_metrics
from .export import PARAM_LAYOUT, contract_weights, forward
from .features import FEATURES

QUANT_VERSION = "int8-v1"
QMAX = 127                       # symmetric int8: -127..127 (never -128)
K_MIN, K_MAX = -128, 127         # tanh lookup index range
INT32_MIN, INT32_MAX = -(2 ** 31), 2 ** 31 - 1
M_MIN, M_MAX = 2 ** 30, 2 ** 31 - 1
ACCEPT_MIN_AGREEMENT = 0.995
ACCEPT_MAX_MACRO_F1_DROP = 0.01
ACCEPT_MAX_SAFETY_CHANGES = 0
F32 = np.float32


# ---------------------------------------------------------------- arithmetic rules

def rnd(v):
    """Round half away from zero (float64). Exact for float32 inputs, which
    are converted to float64 first (|v| + 0.5 is then exact)."""
    v = np.asarray(v, dtype=np.float64)
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def rnd_int(v):
    return int(rnd(v))


def rshift_rnd(v, n):
    """Integer v / 2^n rounded half away from zero; n >= 1. Matches
    pv_rshift_rnd() in pv_quant.h for |v| < 2^62."""
    v, n = int(v), int(n)
    if n < 1:
        raise ValueError("shift must be >= 1")
    half = 1 << (n - 1)
    return (v + half) >> n if v >= 0 else -((-v + half) >> n)


def sat(v, lo, hi):
    """(clamped value, clipped flag)."""
    return (lo, True) if v < lo else (hi, True) if v > hi else (v, False)


def quantize_multiplier(m_real):
    """Fixed-point form m_real ~= M * 2^-SH with M in [2^30, 2^31) and SH >= 1."""
    if not (m_real > 0 and math.isfinite(m_real)):
        raise ValueError(f"multiplier must be positive and finite, got {m_real}")
    f, e = math.frexp(m_real)                       # m_real = f * 2^e, f in [0.5, 1)
    m = rnd_int(f * 2 ** 31)
    if m == 2 ** 31:
        m, e = m // 2, e + 1
    sh = 31 - e
    if not (M_MIN <= m <= M_MAX) or not (1 <= sh <= 62):
        raise ValueError(f"multiplier {m_real} gives M={m}, SH={sh} outside the supported range")
    return m, sh


def f32_hex(x):
    """Exact C99 hex literal of a float32 value."""
    return float(F32(x)).hex()


# ---------------------------------------------------------------- parameters

@dataclass
class QuantParams:
    """Everything the INT8 core needs, plus provenance. Stored as JSON."""
    version: str
    contract_sha256: str
    weights_sha256: str
    model_version: str
    feature_version: str
    features: list
    labels: list
    param_layout: list
    activation: str
    a_x: float
    a_z1: float
    s_x: float
    s_w1: float
    s_w2: float
    s_z1: float
    s_h: float
    inv_sx_f32: str          # float32 as exact hex
    out_scale_f32: str
    m1: int
    sh1: int
    qw1: list                # [8][16] int8
    qb1: list                # [16] int32
    qw2: list                # [16][5] int8
    qb2: list                # [5] int32
    tanh_lut: list           # [256] int8, index k + 128
    calibration: dict = field(default_factory=dict)

    def to_json(self):
        return json.dumps(self.__dict__, indent=1, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text):
        q = cls(**json.loads(text))
        validate_params(q)
        return q

    @property
    def inv_sx(self):
        return F32(float.fromhex(self.inv_sx_f32))

    @property
    def out_scale(self):
        return F32(float.fromhex(self.out_scale_f32))

    def arrays(self):
        return (np.array(self.qw1, dtype=np.int64), np.array(self.qb1, dtype=np.int64),
                np.array(self.qw2, dtype=np.int64), np.array(self.qb2, dtype=np.int64),
                np.array(self.tanh_lut, dtype=np.int64))


def accumulator_bounds(q):
    """Worst-case |acc1|, |acc2| and |acc1 * M1| for any int8 input."""
    qw1, qb1, qw2, qb2, _ = q.arrays()
    acc1 = int((np.abs(qw1).sum(axis=0) * QMAX + np.abs(qb1)).max())
    acc2 = int((np.abs(qw2).sum(axis=0) * QMAX + np.abs(qb2)).max())
    return {"acc1": acc1, "acc2": acc2, "acc1_times_m1": acc1 * q.m1}


def validate_params(q):
    """Ranges, shapes, order and overflow bounds; raises ValueError."""
    def need(c, m):
        if not c:
            raise ValueError(f"quant params: {m}")
    need(q.version == QUANT_VERSION, f"version {q.version}")
    need(list(q.features) == list(FEATURES), "feature order differs from pv8-v1")
    need(list(q.labels) == list(LABELS), "label order differs from the contract")
    need(q.param_layout == [[n, list(s)] for n, s in PARAM_LAYOUT], "parameter layout")
    need(q.activation == "tanh", "activation must be tanh")
    qw1, qb1, qw2, qb2, lut = q.arrays()
    need(qw1.shape == (8, 16) and qw2.shape == (16, 5) and qb1.shape == (16,) and qb2.shape == (5,), "shapes")
    need(lut.shape == (256,), "tanh lookup table must have 256 entries")
    for name, a in (("qw1", qw1), ("qw2", qw2), ("tanh_lut", lut)):
        need(np.abs(a).max() <= QMAX, f"{name} outside -127..127")
    for name, a in (("qb1", qb1), ("qb2", qb2)):
        need(a.min() >= INT32_MIN and a.max() <= INT32_MAX, f"{name} outside int32")
    need(M_MIN <= q.m1 <= M_MAX and 1 <= q.sh1 <= 62, "M1/SH1 out of range")
    b = accumulator_bounds(q)
    need(b["acc1"] <= INT32_MAX and b["acc2"] <= INT32_MAX, "int32 accumulator could overflow")
    need(b["acc1_times_m1"] < 2 ** 62, "int64 rescaling could overflow")
    need(np.isfinite(q.inv_sx) and q.inv_sx > 0 and np.isfinite(q.out_scale) and q.out_scale > 0, "scales")
    return True


def build_tanh_lut(s_z1):
    t = np.arange(K_MIN, K_MAX + 1, dtype=np.float64)
    return np.clip(rnd(QMAX * np.tanh(t * s_z1)), -QMAX, QMAX).astype(np.int64)


# [GUIDE] G. INT8 quantisation (docs/QUANTIZATION.md)
# WHAT: converts the float weights to 8-bit integers (int8) with fixed scales, plus a
#       256-entry int8 tanh lookup table, so the ESP32 can run the MLP with integer maths.
# WHY:  smaller weights and faster integer arithmetic on a microcontroller.
# HOW:  scales from max-abs values over status-OK TRAINING rows only (no OPERATION data).
# RESULT: agreement with the float model 1.000 on 2,924 TRAINING rows, 0 changed
#       predictions, 0 OPEN/SHORT changes (results/quantization_report.md).
# STILL UNVERIFIED IN THE REPO: no ESP32 flash/RAM/latency measurement is committed
#       (docs/ESP32_INTEGRATION.md, checklist H3-H7 NOT_EXECUTED).
def quantize_model(contract, contract_sha256, z_calib, calibration_info=None):
    """Derive INT8 parameters from the float contract and calibration inputs
    z_calib (scaled status-OK TRAINING rows, float32 device values)."""
    z = np.asarray(z_calib, dtype=F32)
    if z.ndim != 2 or z.shape[1] != len(FEATURES) or not len(z) or not np.isfinite(z).all():
        raise ValueError("calibration inputs must be finite, shape (n, 8), n > 0")
    w = contract_weights(contract)                      # float64, LABELS order
    a_x = float(np.abs(z).max())
    pre = z.astype(np.float64) @ w["W1"] + w["b1"]
    a_z1 = float(np.abs(pre).max())
    s_x, s_z1, s_h = a_x / QMAX, a_z1 / QMAX, 1.0 / QMAX
    s_w1, s_w2 = float(np.abs(w["W1"]).max()) / QMAX, float(np.abs(w["W2"]).max()) / QMAX
    qw1 = np.clip(rnd(w["W1"] / s_w1), -QMAX, QMAX).astype(np.int64)
    qw2 = np.clip(rnd(w["W2"] / s_w2), -QMAX, QMAX).astype(np.int64)
    qb1 = rnd(w["b1"] / (s_x * s_w1)).astype(np.int64)
    qb2 = rnd(w["b2"] / (s_h * s_w2)).astype(np.int64)
    m1, sh1 = quantize_multiplier(s_x * s_w1 / s_z1)
    q = QuantParams(
        version=QUANT_VERSION, contract_sha256=contract_sha256, weights_sha256=contract["weights_sha256"],
        model_version=contract["model_version"], feature_version=contract["feature_version"],
        features=list(contract["features"]), labels=list(contract["labels"]),
        param_layout=contract["param_layout"], activation=contract["activation"],
        a_x=a_x, a_z1=a_z1, s_x=s_x, s_w1=s_w1, s_w2=s_w2, s_z1=s_z1, s_h=s_h,
        inv_sx_f32=f32_hex(1.0 / s_x), out_scale_f32=f32_hex(s_h * s_w2), m1=m1, sh1=sh1,
        qw1=qw1.tolist(), qb1=qb1.tolist(), qw2=qw2.tolist(), qb2=qb2.tolist(),
        tanh_lut=build_tanh_lut(s_z1).tolist(),
        calibration=dict(calibration_info or {}, rows=int(len(z)), rule="max-abs over the calibration rows"))
    validate_params(q)
    return q


def params_sha256(q):
    return hashlib.sha256(q.to_json().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- INT8 emulator

def quantize_input(z, inv_sx):
    """qx = sat(rnd(float32(z * INV_SX)), -127, 127). Returns (qx, clipped)
    for one float32 vector; the product is taken in float32 as in C."""
    z = np.asarray(z, dtype=F32)
    p = (z * F32(inv_sx)).astype(F32)
    out, clips = [], 0
    for v in p:
        if not np.isfinite(v):
            raise ValueError("non-finite model input")
        r, c = sat(int(rnd(float(v))), -QMAX, QMAX)
        out.append(r); clips += c
    return out, clips


# [GUIDE] G. The integer forward pass (same arithmetic as firmware/pv_fault/pv_mlp_int8.c):
# int8 inputs x int8 weights + int32 bias -> fixed-point rescale -> tanh lookup -> layer 2.
def int8_core(qx, q):
    """Integer core for one reading: int8 inputs -> int32 output accumulators.
    Returns (acc2, k, hq, lut_clips)."""
    qw1, qb1, qw2, qb2, lut = q.arrays()
    k, hq, lut_clips = [], [], 0
    for j in range(16):
        acc1 = int(qb1[j]) + sum(int(qx[i]) * int(qw1[i][j]) for i in range(8))
        assert INT32_MIN <= acc1 <= INT32_MAX
        kj, c = sat(rshift_rnd(acc1 * q.m1, q.sh1), K_MIN, K_MAX)
        lut_clips += c
        k.append(kj); hq.append(int(lut[kj - K_MIN]))
    acc2 = [int(qb2[c]) + sum(hq[j] * int(qw2[j][c]) for j in range(16)) for c in range(5)]
    assert all(INT32_MIN <= a <= INT32_MAX for a in acc2)
    return acc2, k, hq, lut_clips


def argmax_first(values):
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def int8_predict(z, q):
    """Full INT8 path for one scaled float32 input vector."""
    qx, in_clips = quantize_input(z, q.inv_sx)
    acc2, k, hq, lut_clips = int8_core(qx, q)
    logits = (np.asarray(acc2, dtype=np.float64).astype(F32) * q.out_scale).astype(F32)
    return {"qx": qx, "k": k, "hq": hq, "acc2": acc2, "logits": logits,
            "label_index": argmax_first(acc2), "input_clips": in_clips, "lut_clips": lut_clips}


def int8_predict_many(z_rows, q):
    res = [int8_predict(z, q) for z in np.asarray(z_rows, dtype=F32)]
    return {"label_index": np.array([r["label_index"] for r in res], dtype=np.int64),
            "logits": np.stack([r["logits"] for r in res]) if res else np.zeros((0, 5), F32),
            "input_clips": int(sum(r["input_clips"] for r in res)),
            "lut_clips": int(sum(r["lut_clips"] for r in res)),
            "rows_with_input_clip": int(sum(r["input_clips"] > 0 for r in res))}


# ---------------------------------------------------------------- comparison

def float_reference_labels(z64, contract):
    """Official float reference: the Phase 8 contract forward pass (float64)."""
    _, p = forward(contract_weights(contract), np.asarray(z64, dtype=np.float64))
    return p.argmax(axis=1)


def compare_predictions(ref_idx, int8_idx, y_true=None):
    """Agreement between the float reference and INT8, with every prediction
    change that involves OPEN_CIRCUIT or SHORT_CIRCUIT, in either direction."""
    ref_idx, int8_idx = np.asarray(ref_idx), np.asarray(int8_idx)
    ref_l = np.array([LABELS[i] for i in ref_idx], dtype=object)
    q_l = np.array([LABELS[i] for i in int8_idx], dtype=object)
    changed = ref_l != q_l
    safety = {}
    for f in SAFETY_CLASSES:
        to_f = changed & (q_l == f)
        from_f = changed & (ref_l == f)
        safety[f] = {"changed_to": int(to_f.sum()), "changed_from": int(from_f.sum())}
    out = {
        "rows": int(len(ref_l)),
        "agreement": float((~changed).mean()) if len(ref_l) else None,
        "changed": int(changed.sum()),
        "per_class_agreement": {l: (float((~changed[ref_l == l]).mean()) if (ref_l == l).any() else None)
                                for l in LABELS},
        "change_pairs": {f"{a}->{b}": int(((ref_l == a) & (q_l == b)).sum())
                         for a in LABELS for b in LABELS if a != b and ((ref_l == a) & (q_l == b)).any()},
        "safety_changes": safety,
        # each changed row counted once if either side is OPEN_CIRCUIT or SHORT_CIRCUIT
        "safety_changes_total": int((changed & (np.isin(ref_l, SAFETY_CLASSES)
                                                | np.isin(q_l, SAFETY_CLASSES))).sum()),
    }
    if y_true is not None:
        mr, mq = classification_metrics(y_true, ref_l), classification_metrics(y_true, q_l)
        out["training_fit_macro_f1"] = {"float_reference": mr["macro_f1"], "int8": mq["macro_f1"],
                                        "drop": mr["macro_f1"] - mq["macro_f1"]}
    return out


def acceptance(comparison, golden_passed):
    """TRAINING-only acceptance rule fixed in docs/QUANTIZATION.md section 6."""
    checks = {
        "safety_changes_zero": comparison["safety_changes_total"] <= ACCEPT_MAX_SAFETY_CHANGES,
        "agreement_at_least_0.995": comparison["agreement"] is not None
        and comparison["agreement"] >= ACCEPT_MIN_AGREEMENT,
        "macro_f1_drop_at_most_1pp": comparison.get("training_fit_macro_f1", {}).get("drop", 1.0)
        <= ACCEPT_MAX_MACRO_F1_DROP,
        "golden_tests_passed": bool(golden_passed),
    }
    return {"checks": checks, "int8_accepted": all(checks.values()),
            "deployment_candidate": "INT8 and Float32 (final choice by on-device measurement)"
            if all(checks.values()) else "Float32 only",
            "note": "Acceptance is measured on TRAINING rows the model was fitted on; it is not a "
                    "held-out estimate and does not establish electrical safety."}

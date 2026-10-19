"""Float32 device reference for the frozen preprocessing (Phase 10).

A NumPy float32 mirror of firmware/pv_fault/pv_fault.c: quality gates,
pv8-v1 features and the frozen scaler, computed in the same order of
operations and in IEEE single precision, as on the ESP32. It reads every
constant from the frozen FeatureConfig and never fits anything.

Raw inputs are arrays of shape (n, 8) in features.RAW_INPUTS order.
The float64 pipeline in features.py stays the reference for training and
evaluation; tests check that this float32 copy agrees with it.
"""
import numpy as np

from .features import FEATURES, RAW_INPUTS, STATUS_CODES

F32 = np.float32
RAW_INDEX = {c: i for i, c in enumerate(RAW_INPUTS)}
STATUS_OK, STATUS_LOW_LIGHT = STATUS_CODES["OK"], STATUS_CODES["LOW_LIGHT"]
STATUS_SATURATED, STATUS_INVALID = STATUS_CODES["LUX_SATURATED"], STATUS_CODES["INVALID_INPUT"]


def device_constants(cfg):
    """The float32 constants the firmware uses, in a fixed layout."""
    if not cfg.is_fitted:
        raise ValueError("feature config is not fitted")
    if tuple(cfg.features) != FEATURES:
        raise ValueError(f"feature order {cfg.features} differs from {FEATURES}")
    lo = np.full(len(RAW_INPUTS), -np.inf, dtype=F32)
    hi = np.full(len(RAW_INPUTS), np.inf, dtype=F32)
    for col, (a, b) in cfg.input_limits.items():
        lo[RAW_INDEX[col]], hi[RAW_INDEX[col]] = F32(a), F32(b)
    return {
        "lux_per_unit": F32(cfg.lux_per_unit), "low_light_lux": F32(cfg.low_light_lux),
        "lux_saturation": F32(cfg.lux_saturation), "t_ref_c": F32(cfg.t_ref_c),
        "vref_a": F32(cfg.vref["a"]), "vref_b": F32(cfg.vref["b"]), "vref_c": F32(cfg.vref["c"]),
        "mean": np.asarray(cfg.scaler["mean"], dtype=F32), "std": np.asarray(cfg.scaler["std"], dtype=F32),
        "temp_sentinels": np.asarray(cfg.temp_sentinels, dtype=F32), "limit_lo": lo, "limit_hi": hi,
    }


def raw_matrix(df):
    """(n, 8) float32 raw inputs in RAW_INPUTS order (label columns are never read)."""
    return df[list(RAW_INPUTS)].to_numpy(dtype=np.float64).astype(F32)


def gate_f32(raw, k):
    """Status code per reading. Precedence: INVALID > SATURATED > LOW_LIGHT > OK."""
    raw = np.asarray(raw, dtype=F32)
    invalid = ~np.isfinite(raw).all(axis=1)
    with np.errstate(invalid="ignore"):
        for t in (RAW_INDEX["panel_temp1_c"], RAW_INDEX["panel_temp2_c"]):
            invalid |= np.isin(raw[:, t], k["temp_sentinels"])
        invalid |= ((raw < k["limit_lo"]) | (raw > k["limit_hi"])).any(axis=1)
        lux = raw[:, RAW_INDEX["irradiance_lux"]]
        status = np.where(invalid, STATUS_INVALID,
                          np.where(lux >= k["lux_saturation"], STATUS_SATURATED,
                                   np.where(lux < k["low_light_lux"], STATUS_LOW_LIGHT, STATUS_OK)))
    return status.astype(np.int64)


def features_f32(raw, k):
    """pv8-v1 features in FEATURES order for status-OK readings (float32,
    same operation order as pv_fault.c)."""
    r = np.asarray(raw, dtype=F32)
    v, i, p = r[:, 0], r[:, 1], r[:, 2]
    t1, t2, tamb, lux, dust = r[:, 3], r[:, 4], r[:, 5], r[:, 6], r[:, 7]
    klx = lux / k["lux_per_unit"]
    tp = (t1 + t2) / F32(2.0)
    vref = (k["vref_a"] + k["vref_b"] * np.log(klx)) + k["vref_c"] * (tp - k["t_ref_c"])
    out = np.stack([vref - v, i / klx, p / klx, dust, np.abs(t1 - t2), tp - tamb, v, i], axis=1)
    assert out.dtype == F32
    return out


def scale_f32(feat, k):
    return ((np.asarray(feat, dtype=F32) - k["mean"]) / k["std"]).astype(F32)


def preprocess_f32(raw, cfg):
    """Gate, features and scaling. Returns (status, z) where z has one row
    per reading and is NaN for readings that are not status OK."""
    k = device_constants(cfg)
    raw = np.asarray(raw, dtype=F32)
    status = gate_f32(raw, k)
    z = np.full((len(raw), len(FEATURES)), np.nan, dtype=F32)
    ok = status == STATUS_OK
    if ok.any():
        z[ok] = scale_f32(features_f32(raw[ok], k), k)
    return status, z


def forward_f32(z, w):
    """Float32 MLP forward pass, as in pv_mlp_f32.c. Returns float32 logits."""
    z = np.asarray(z, dtype=F32)
    h = np.tanh(z @ w["W1"].astype(F32) + w["b1"].astype(F32))
    return (h @ w["W2"].astype(F32) + w["b2"].astype(F32)).astype(F32)


def softmax_f32(logits):
    logits = np.asarray(logits, dtype=F32)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(F32)

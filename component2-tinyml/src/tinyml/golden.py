"""Golden test vectors for the host C tests (Phase 10).

Every case carries an origin tag:
  TRAINING_DERIVED                     scaled inputs computed from TRAINING rows
                                       (no raw readings are stored)
  TRAINING_DERIVED_REFERENCE_PROBE     the 5 reference probes from the contract
  SYNTHETIC_PROTOTYPE_FROM_TRAINING_MEDIANS
                                       synthetic raw readings built from per-class
                                       median TRAINING values; not measurements
  SYNTHETIC_EDGE_CASE                  synthetic raw readings for gates and limits
  SYNTHETIC_TEST_FIXTURE               arithmetic and core vectors, synthetic models
None of these are physical sensor measurements. Expected values come from the
Python float32 device reference and the INT8 emulator.
"""
import numpy as np

from .contract import LABEL_COLUMN, LABELS
from .device_reference import (RAW_INDEX, STATUS_OK, device_constants, forward_f32, gate_f32,
                               preprocess_f32, raw_matrix)
from .export import contract_weights
from .features import RAW_INPUTS
from .quantize import F32, K_MAX, QMAX, argmax_first, int8_core, int8_predict, quantize_input, rshift_rnd

ORIGIN_TRAINING = "TRAINING_DERIVED"
ORIGIN_PROBE = "TRAINING_DERIVED_REFERENCE_PROBE"
ORIGIN_PROTOTYPE = "SYNTHETIC_PROTOTYPE_FROM_TRAINING_MEDIANS"
ORIGIN_EDGE = "SYNTHETIC_EDGE_CASE"
ORIGIN_FIXTURE = "SYNTHETIC_TEST_FIXTURE"
EXACT_ACC2_GAP = 2          # raw cases need a clear top-2 gap for an exact label check
EXACT_LOGIT_GAP = 1e-3


# ---------------------------------------------------------------- selections

def training_representatives(train_df, cfg, runs_per_class=5, positions=(1, 20, 40, 60)):
    """Scaled float32 inputs of fixed reading positions in the first runs (by
    run_id) of each class, status-OK rows only. Returns [(id, origin, z)]."""
    if set(train_df["window"]) != {"TRAINING"}:
        raise ValueError("representatives must come from TRAINING rows only")
    status, z = preprocess_f32(raw_matrix(train_df), cfg)
    d = train_df.assign(_ok=status == STATUS_OK, _row=np.arange(len(train_df)))
    out = []
    for label in LABELS:
        runs = sorted(d.loc[d[LABEL_COLUMN] == label, "run_id"].unique())
        taken = 0
        for run in runs:
            g = d[(d.run_id == run) & d.reading.isin(positions) & d._ok]
            if len(g) < len(positions):
                continue
            for _, r in g.sort_values("reading").iterrows():
                out.append((f"{run}#r{int(r.reading)}", ORIGIN_TRAINING, z[int(r._row)]))
            taken += 1
            if taken == runs_per_class:
                break
    return out


def reference_probes(contract):
    return [(f"probe{i}", ORIGIN_PROBE, np.asarray(z, dtype=F32))
            for i, z in enumerate(contract["reference_probes"]["inputs_scaled"])]


def class_prototypes(train_df, cfg, factors=(0.85, 0.95, 1.0, 1.05, 1.15), temp_offsets=(0.0, 0.5)):
    """Synthetic raw readings per class from median status-OK TRAINING values.
    Irradiance, current and power are scaled together; both panel
    temperatures are offset. Returns [(id, origin, raw32)]."""
    if set(train_df["window"]) != {"TRAINING"}:
        raise ValueError("prototypes must come from TRAINING rows only")
    k = device_constants(cfg)
    raw = raw_matrix(train_df)
    ok = gate_f32(raw, k) == STATUS_OK
    out = []
    for label in LABELS:
        med = np.median(raw[ok & (train_df[LABEL_COLUMN] == label).to_numpy()].astype(np.float64), axis=0)
        for f in factors:
            for dt in temp_offsets:
                r = med.copy()
                for c in ("irradiance_lux", "current_ma", "power_mw"):
                    r[RAW_INDEX[c]] *= f
                for c in ("panel_temp1_c", "panel_temp2_c"):
                    r[RAW_INDEX[c]] += dt
                out.append((f"{label}_f{f}_dt{dt}", ORIGIN_PROTOTYPE, r.astype(F32)))
    return out


def edge_cases(base_raw, cfg):
    """Synthetic gate and limit cases around one status-OK reading."""
    k = device_constants(cfg)
    base = np.asarray(base_raw, dtype=F32)
    if gate_f32(base[None], k)[0] != STATUS_OK:
        raise ValueError("edge cases need a status-OK base reading")
    up = lambda x: np.nextafter(F32(x), F32(np.inf))       # noqa: E731
    down = lambda x: np.nextafter(F32(x), F32(-np.inf))    # noqa: E731
    cases = []

    def add(name, **changes):
        r = base.copy()
        for col, v in changes.items():
            r[RAW_INDEX[col]] = F32(v)
        cases.append((name, ORIGIN_EDGE, r))
    lux = "irradiance_lux"
    for v in (k["low_light_lux"] - 1, down(k["low_light_lux"]), k["low_light_lux"],
              k["lux_saturation"] - 1, down(k["lux_saturation"]), k["lux_saturation"],
              up(k["lux_saturation"]), 0.0, -1.0):
        add(f"lux={float(v)!r}", **{lux: v})
    for col in RAW_INPUTS:
        for v, tag in ((np.nan, "nan"), (np.inf, "inf"), (-np.inf, "-inf")):
            add(f"{col}={tag}", **{col: v})
        lo, hi = k["limit_lo"][RAW_INDEX[col]], k["limit_hi"][RAW_INDEX[col]]
        if np.isfinite(lo):
            add(f"{col}=lo", **{col: lo}); add(f"{col}=below_lo", **{col: down(lo)})
        if np.isfinite(hi):
            add(f"{col}=hi", **{col: hi}); add(f"{col}=above_hi", **{col: up(hi)})
    for s in k["temp_sentinels"]:
        add(f"panel_temp1_c=sentinel{float(s)!r}", panel_temp1_c=s)
        add(f"panel_temp2_c=sentinel{float(s)!r}", panel_temp2_c=s)
    add("invalid_and_saturated", **{lux: k["lux_saturation"], "voltage_v": np.nan})
    add("invalid_and_low_light", **{lux: 1000.0, "panel_temp1_c": -127.0})
    add("valid_extreme_temp_spread", panel_temp1_c=85.0, panel_temp2_c=-10.0)
    add("valid_extreme_dust", dust_vo_v=3.3)
    return cases


# ---------------------------------------------------------------- vectors

def rshift_vectors(seed=11):
    rng = np.random.default_rng(seed)
    out = []
    for n in (1, 2, 5, 16, 31, 37, 45, 61, 62):
        half = 1 << (n - 1)
        vals = [0, 1, -1, half, -half, half - 1, -(half - 1), half + 1, -(half + 1),
                3 * half, -3 * half, (1 << 62) - 1, -((1 << 62) - 1)]
        vals += [int(x) for x in rng.integers(-(1 << 50), 1 << 50, 4)]
        for v in vals:
            if abs(v) < (1 << 62):
                out.append((v, n, rshift_rnd(v, n)))
    return out


def quantize_input_vectors(inv_sx):
    zs = [0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, float(np.nextafter(F32(0.5), F32(0))),
          -float(np.nextafter(F32(0.5), F32(0))), 126.5, -126.5, 127.4, 127.5, -127.5, 128.0, 1e30, -1e30]
    out = []
    for z in zs:
        (q,), c = quantize_input([z], F32(1.0))
        out.append((z, 1.0, q, c))
    for z in (0.0, 0.3, -0.3, 1.0, -1.0, 4.0, -4.0, 1e3, -1e3,
              float(F32(0.5) / F32(inv_sx)), -float(F32(0.5) / F32(inv_sx))):
        (q,), c = quantize_input([z], F32(inv_sx))
        out.append((z, float(F32(inv_sx)), q, c))
    return out


ARGMAX_CASES = ([5, 5, 3, 1, 0], [1, 7, 7, 7, 0], [0, 0, 0, 0, 0], [-3, -1, -1, -2, -5], [1, 2, 3, 4, 4],
                [9, 1, 2, 3, 4])


def core_vectors(q, seed=7):
    rng = np.random.default_rng(seed)
    sets = [[0] * 8, [QMAX] * 8, [-QMAX] * 8, [QMAX, -QMAX] * 4, [-QMAX, QMAX] * 4]
    sets += [list(map(int, rng.integers(-QMAX, QMAX + 1, 8))) for _ in range(16)]
    out = []
    for qx in sets:
        acc2, k, hq, clips = int8_core(qx, q)
        out.append({"qx": qx, "acc2": acc2, "hq": hq, "k": k, "lut_clips": clips})
    return out


def z_case(cid, origin, z, q, w):
    p = int8_predict(z, q)
    lf = forward_f32(np.asarray(z, dtype=F32)[None], w)[0]
    return {"id": cid, "origin": origin, "z": np.asarray(z, dtype=F32), "qx": p["qx"], "acc2": p["acc2"],
            "label_int8": p["label_index"], "label_f32": argmax_first(list(lf)), "logits_f32": lf,
            "input_clips": p["input_clips"]}


def _gap(values):
    s = sorted(values, reverse=True)
    return s[0] - s[1]


def raw_case(cid, origin, raw, cfg, q, w):
    status, z = preprocess_f32(np.asarray(raw, dtype=F32)[None], cfg)
    c = {"id": cid, "origin": origin, "raw": np.asarray(raw, dtype=F32), "status": int(status[0]),
         "z": z[0], "label_int8": -1, "label_f32": -1, "qx": [0] * 8, "exact_label": True}
    if status[0] == STATUS_OK:
        p = int8_predict(z[0], q)
        lf = forward_f32(z, w)[0]
        c.update(label_int8=p["label_index"], label_f32=argmax_first(list(lf)), qx=p["qx"],
                 exact_label=_gap(p["acc2"]) >= EXACT_ACC2_GAP and _gap(list(lf)) >= EXACT_LOGIT_GAP)
    return c


def build_golden(contract, cfg, q, z_inputs, raw_inputs):
    """z_inputs, raw_inputs: lists of (id, origin, vector)."""
    w = contract_weights(contract)
    g = {"model": q.model_version,
         "rshift": rshift_vectors(), "quantize_input": quantize_input_vectors(q.inv_sx),
         "argmax": [(a, [float(x) for x in a], argmax_first(a)) for a in ARGMAX_CASES],
         "core": core_vectors(q),
         "z_cases": [z_case(i, o, z, q, w) for i, o, z in z_inputs],
         "raw_cases": [raw_case(i, o, r, cfg, q, w) for i, o, r in raw_inputs]}
    if not g["z_cases"] or not g["raw_cases"]:
        raise ValueError("golden vectors need at least one z case and one raw case")
    g["origins"] = {ORIGIN_FIXTURE} | {c["origin"] for c in g["z_cases"] + g["raw_cases"]}
    assert all(abs(x) <= K_MAX for c in g["core"] for x in c["hq"])
    return g


def golden_summary(g):
    def count(cases):
        out = {}
        for c in cases:
            out[c["origin"]] = out.get(c["origin"], 0) + 1
        return out
    return {"z_cases": count(g["z_cases"]), "raw_cases": count(g["raw_cases"]),
            "raw_status_counts": {s: sum(c["status"] == s for c in g["raw_cases"]) for s in range(4)},
            "rshift": len(g["rshift"]), "quantize_input": len(g["quantize_input"]),
            "argmax": len(g["argmax"]), "core": len(g["core"]),
            "raw_cases_without_exact_label_check": [c["id"] for c in g["raw_cases"] if not c["exact_label"]]}

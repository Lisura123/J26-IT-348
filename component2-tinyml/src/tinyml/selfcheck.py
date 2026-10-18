"""Initialisation self-check vectors for the firmware API (pv_api.c).

A small generated header, pv_selfcheck.h, lets pv_api_init() confirm on the
device that the compiled Float32 and INT8 models still reproduce the Python
reference before any reading is classified. It is generated from the frozen
contract and the frozen INT8 parameters only (no dataset); it does not
change any existing generated header.

Vectors: the contract's reference probes (scaled TRAINING inputs, stored in
the contract) plus seeded synthetic scaled inputs. Expected values:
Float32 logits from device_reference.forward_f32 (checked with a tolerance)
and INT8 output accumulators from quantize.int8_predict (checked bit for bit).
"""
from pathlib import Path

import numpy as np

from .codegen import GENERATOR, _guard, c_floats, c_ints
from .device_reference import forward_f32
from .export import contract_weights
from .quantize import F32, int8_predict, validate_params

SELFCHECK_VERSION = "selfcheck-v1"
SYNTHETIC_SEED = 2026
N_SYNTHETIC = 11
HEADER = "pv_selfcheck.h"


def selfcheck_vectors(contract, q, n_synthetic=N_SYNTHETIC, seed=SYNTHETIC_SEED):
    probes = np.asarray(contract["reference_probes"]["inputs_scaled"], dtype=F32)
    rng = np.random.default_rng(seed)
    synth = np.clip(rng.normal(0.0, 1.0, size=(n_synthetic, probes.shape[1])), -3.0, 3.0).astype(F32)
    z = np.concatenate([probes, synth]).astype(F32)
    origins = ["CONTRACT_REFERENCE_PROBE"] * len(probes) + ["SYNTHETIC_SEEDED"] * n_synthetic
    logits = forward_f32(z, contract_weights(contract))
    int8 = [int8_predict(r, q) for r in z]
    return {"z": z, "origins": origins, "logits_f32": logits,
            "label_f32": [int(np.argmax(r)) for r in logits],
            "acc2": [r["acc2"] for r in int8], "label_int8": [r["label_index"] for r in int8]}


def selfcheck_header(contract, contract_sha256, q):
    validate_params(q)
    if q.contract_sha256 != contract_sha256:
        raise ValueError("INT8 parameters do not belong to this contract")
    v = selfcheck_vectors(contract, q)
    n = len(v["z"])
    rows = lambda m, fn: ",\n".join("    {" + fn(r) + "}" for r in m)   # noqa: E731
    body = ["#include <stdint.h>", "",
            f'#define PV_SC_VERSION "{SELFCHECK_VERSION}"',
            f'#define PV_SC_MODEL_VERSION "{contract["model_version"]}"',
            f'#define PV_SC_FEATURE_VERSION "{contract["feature_version"]}"',
            f'#define PV_SC_QUANT_VERSION "{q.version}"',
            f'#define PV_SC_CONTRACT_SHA256 "{contract_sha256}"',
            f'#define PV_SC_WEIGHTS_SHA256 "{contract["weights_sha256"]}"',
            "#define PV_SC_LABEL_NAMES_INIT {" + ", ".join(f'"{l}"' for l in contract["labels"]) + "}",
            f"#define PV_SC_N {n}", "",
            "/* origins: " + ", ".join(f"{i}={o}" for i, o in enumerate(v["origins"])) + " */",
            f"static const float PV_SC_Z[{n}][8] = {{\n{rows(v['z'], c_floats)}\n}};",
            f"static const float PV_SC_LOGITS_F32[{n}][5] = {{\n{rows(v['logits_f32'], c_floats)}\n}};",
            f"static const int PV_SC_LABEL_F32[{n}] = {{{c_ints(v['label_f32'])}}};",
            f"static const int32_t PV_SC_ACC2[{n}][5] = {{\n{rows(v['acc2'], c_ints)}\n}};",
            f"static const int PV_SC_LABEL_INT8[{n}] = {{{c_ints(v['label_int8'])}}};"]
    prov = [f"self-check vectors ({SELFCHECK_VERSION}) for model {contract['model_version']} and "
            f"INT8 {q.version}", f"contract SHA-256 {contract_sha256}",
            f"weights SHA-256 {contract['weights_sha256']}", "no dataset rows are used"]
    return _guard(HEADER, body, prov).replace(GENERATOR, f"src/tinyml/selfcheck.py ({SELFCHECK_VERSION})")


def write_selfcheck(out_dir, contract, contract_sha256, q):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / HEADER
    path.write_text(selfcheck_header(contract, contract_sha256, q), encoding="utf-8")
    return path

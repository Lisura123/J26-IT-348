"""INT8 quantisation of the frozen model (TRAINING window only).

Usage (in Colab; see notebooks/colab_quantization.ipynb):
    python scripts/07_quantize_model.py --data-dir data/raw --out-root /content/quant_out

Steps: verify the frozen model files (pinned SHA-256) and the dataset hash;
drop OPERATION rows straight after loading; check the float32 device
preprocessing against features.py; calibrate on status-OK TRAINING rows;
quantise; compare INT8 with the float reference on TRAINING rows; build the
golden vectors; generate the C headers; run the host C tests; write the
INT8 parameters, manifest and report. Nothing is trained or refitted, and
the float contract is not changed.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyml import codegen, golden  # noqa: E402
from tinyml.contract import LABEL_COLUMN, LABELS, WINDOW_COLUMN  # noqa: E402
from tinyml.device_reference import forward_f32, preprocess_f32, raw_matrix  # noqa: E402
from tinyml.export import contract_feature_config, contract_weights, file_sha256  # noqa: E402
from tinyml.features import STATUS_CODES, transform  # noqa: E402
from tinyml.holdout import PINNED_FILES, PINNED_WEIGHTS_SHA256, inference_frame, verify_frozen_model  # noqa: E402
from tinyml.hostc import compile_and_run  # noqa: E402
from tinyml.quantize import (QUANT_VERSION, acceptance, accumulator_bounds, argmax_first,  # noqa: E402
                             compare_predictions, float_reference_labels, int8_predict_many, params_sha256,
                             quantize_model)
from tinyml.splits import load_assignments  # noqa: E402
from tinyml.train import DATASET_SHA256, environment, final_training_rows, git_commit  # noqa: E402

GEN_REL = Path("firmware/pv_fault/generated")
GOLDEN_REL = Path("tests/host_c")
INT8_REL = Path("models/int8")
REPORT_REL = Path("results")


def find_dataset(data_dir):
    for p in sorted(Path(data_dir).glob("*.csv")):
        if p.name == "pv_fault_dataset.csv" or p.name.endswith("-pv_fault_dataset.csv"):
            return p
    sys.exit(f"pv_fault_dataset.csv not found in {data_dir}")


def training_only(df):
    """Drop every OPERATION row before anything else is computed."""
    out = df[df[WINDOW_COLUMN] == "TRAINING"].reset_index(drop=True)
    assert set(out[WINDOW_COLUMN]) == {"TRAINING"}
    return out


def run(data_path, splits_path, model_dir, out_root, code_commit=None, run_c_tests=True,
        expected_dataset_sha256=DATASET_SHA256, pinned=PINNED_FILES, pinned_weights=PINNED_WEIGHTS_SHA256,
        expected_rows=3000, representatives=(5, (1, 20, 40, 60))):
    out_root = Path(out_root)
    if (out_root / INT8_REL / "quant_params.json").exists():
        raise FileExistsError(f"{out_root / INT8_REL / 'quant_params.json'} exists; quantisation outputs "
                              "are never overwritten")
    sha = file_sha256(data_path)
    if expected_dataset_sha256 and sha != expected_dataset_sha256:
        raise ValueError(f"dataset SHA-256 {sha} is not the validated {expected_dataset_sha256}")
    contract = verify_frozen_model(model_dir, pinned, pinned_weights)
    contract_sha = file_sha256(Path(model_dir) / "model_contract.json")
    cfg = contract_feature_config(contract)

    df = training_only(pd.read_csv(data_path))                # OPERATION rows dropped here
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"{len(df)} TRAINING rows, expected {expected_rows}")
    assignments, meta = load_assignments(splits_path, df, dataset_sha256=sha)
    final_training_rows(df, assignments)                      # split file agrees

    # float32 device preprocessing vs the float64 reference pipeline
    status32, z32 = preprocess_f32(raw_matrix(df), cfg)
    x64, status64 = transform(inference_frame(df), cfg)
    status64 = status64.map(STATUS_CODES).to_numpy()
    ok = status32 == STATUS_CODES["OK"]
    device_check = {"status_identical": bool((status32 == status64).all()),
                    "max_abs_z_difference": float(np.abs(z32[ok].astype(np.float64) - x64).max()),
                    "ok_rows": int(ok.sum()), "skipped_rows": int((~ok).sum())}
    if not device_check["status_identical"]:
        raise ValueError("float32 device gates differ from features.input_status")

    q = quantize_model(contract, contract_sha, z32[ok], calibration_info={
        "window": "TRAINING", "rows": int(len(df)), "dataset_sha256": sha,
        "split_version": meta["split_version"]})
    y = df.loc[ok, LABEL_COLUMN].to_numpy()
    ref = float_reference_labels(x64, contract)
    int8 = int8_predict_many(z32[ok], q)
    dev = np.array([argmax_first(list(r)) for r in forward_f32(z32[ok], contract_weights(contract))])
    comparison = {"int8_vs_float_reference": compare_predictions(ref, int8["label_index"], y),
                  "float32_device_vs_float_reference": compare_predictions(ref, dev, y),
                  "int8_input_clips": int8["input_clips"], "int8_rows_with_input_clip": int8["rows_with_input_clip"],
                  "int8_lut_clips": int8["lut_clips"]}

    n_runs, positions = representatives
    z_inputs = golden.training_representatives(df, cfg, n_runs, positions) + golden.reference_probes(contract)
    protos = golden.class_prototypes(df, cfg)
    base = next(r for i, _, r in protos if i.startswith("NORMAL_f1.0_dt0.0"))
    g = golden.build_golden(contract, cfg, q, z_inputs, protos + golden.edge_cases(base, cfg))

    gen_dir, golden_dir, int8_dir = out_root / GEN_REL, out_root / GOLDEN_REL, out_root / INT8_REL
    headers = codegen.write_headers(gen_dir, contract, contract_sha, cfg, q)
    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_path = golden_dir / "golden_vectors.h"
    golden_path.write_text(codegen.golden_header(g), encoding="utf-8")
    c_tests = (compile_and_run(gen_dir, golden_dir, out_dir=out_root) if run_c_tests
               else {"ran": False, "passed": False, "output": "not run"})
    exe = out_root / "pv_host_test"
    if exe.exists():
        exe.unlink()

    int8_dir.mkdir(parents=True, exist_ok=True)
    (int8_dir / "quant_params.json").write_text(q.to_json(), encoding="utf-8")
    files = {str(INT8_REL / "quant_params.json"): int8_dir / "quant_params.json",
             **{str(GEN_REL / p.name): p for p in headers}, str(GOLDEN_REL / "golden_vectors.h"): golden_path}
    manifest = {"quant_version": QUANT_VERSION, "kind": "INT8_DEPLOYMENT_DERIVATIVE",
                "contract_sha256": contract_sha, "weights_sha256": contract["weights_sha256"],
                "quant_params_sha256": params_sha256(q),
                "files": {k: file_sha256(v) for k, v in sorted(files.items())}}
    (int8_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "status": "QUANTIZATION_TRAINING_ONLY", "quant_version": QUANT_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {"dataset_sha256": sha, "split_version": meta["split_version"],
                       "contract_sha256": contract_sha, "weights_sha256": contract["weights_sha256"],
                       "model_version": contract["model_version"], "code_commit": code_commit or git_commit(ROOT),
                       "environment": environment()},
        "operation_window_used": False, "window": "TRAINING",
        "device_preprocessing_check": device_check,
        "scales": {k: getattr(q, k) for k in ("a_x", "a_z1", "s_x", "s_w1", "s_w2", "s_z1", "s_h",
                                              "inv_sx_f32", "out_scale_f32", "m1", "sh1")},
        "accumulator_bounds": accumulator_bounds(q),
        "comparison": comparison, "golden": golden.golden_summary(g), "host_c_tests": c_tests,
        "acceptance": acceptance(comparison["int8_vs_float_reference"], c_tests["passed"]),
        "sizes_host_estimate": {"int8_weights_bytes": 8 * 16 + 16 * 5, "int32_bias_bytes": (16 + 5) * 4,
                                "tanh_lut_bytes": 256, "float32_weight_and_bias_bytes": 229 * 4,
                                "note": "parameter storage only; ESP32 flash, RAM and latency are not measured"},
        "caveats": [
            "Calibration and comparison use TRAINING rows only; the model was fitted on these rows.",
            "OPERATION was not used. INT8 behaviour on held-out or field data is not measured.",
            "Prediction agreement does not establish electrical safety; OPEN/SHORT outputs stay unconfirmed "
            "ML predictions and no relay isolation is implemented.",
            "No ESP32 memory, latency or energy result is claimed.",
        ],
    }
    out_report = out_root / REPORT_REL
    out_report.mkdir(parents=True, exist_ok=True)
    (out_report / "quantization_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out_report / "quantization_report.md").write_text(render(report), encoding="utf-8")
    return report, q, g


def fmt(x, nd=4):
    return "–" if x is None else f"{x:.{nd}f}"


def render(r):
    c = r["comparison"]["int8_vs_float_reference"]
    d = r["comparison"]["float32_device_vs_float_reference"]
    a = r["acceptance"]
    s = r["scales"]
    L = ["# INT8 quantisation report (TRAINING window only)", "",
         f"Quantisation `{r['quant_version']}`, created {r['created_utc']}. Model `{r['provenance']['model_version']}`, "
         f"contract SHA-256 `{r['provenance']['contract_sha256']}`, weights SHA-256 `{r['provenance']['weights_sha256']}`. "
         f"Code `{r['provenance']['code_commit']}`. Environment {r['provenance']['environment']}.", "",
         "System: **Float32 preprocessing + INT8 neural inference**. The float contract is unchanged; the INT8 "
         "parameters are a deployment derivative. **OPERATION was not used.**", "",
         "## Scales", "",
         f"A_x = {s['a_x']:.6f}, s_x = {s['s_x']:.8f}, s_W1 = {s['s_w1']:.8f}, s_W2 = {s['s_w2']:.8f}, "
         f"A_z1 = {s['a_z1']:.6f}, s_z1 = {s['s_z1']:.8f}, s_h = 1/127; M1 = {s['m1']}, SH1 = {s['sh1']}; "
         f"INV_SX = {s['inv_sx_f32']}, OUT_SCALE = {s['out_scale_f32']}.", "",
         f"Accumulator bounds: {r['accumulator_bounds']} (int32 limit 2147483647; int64 rescaling limit 2^62).", "",
         "## Float32 device preprocessing check", "", f"{r['device_preprocessing_check']}", "",
         "## INT8 vs float reference (TRAINING status-OK rows)", "",
         f"Rows {c['rows']}, agreement {fmt(c['agreement'])}, changed predictions {c['changed']}, "
         f"changes involving OPEN/SHORT {c['safety_changes_total']} {c['safety_changes']}.", "",
         f"Change pairs: {c['change_pairs'] or 'none'}. Per-class agreement: {c['per_class_agreement']}.", "",
         f"Training-fit macro-F1: float {fmt(c['training_fit_macro_f1']['float_reference'])}, INT8 "
         f"{fmt(c['training_fit_macro_f1']['int8'])}, drop {fmt(c['training_fit_macro_f1']['drop'])}.", "",
         f"Input clips {r['comparison']['int8_input_clips']} (rows {r['comparison']['int8_rows_with_input_clip']}); "
         f"tanh lookup clips {r['comparison']['int8_lut_clips']}.", "",
         f"Float32 device path vs float reference: agreement {fmt(d['agreement'])}, changes {d['changed']}.", "",
         "## Golden vectors and host C tests", "", f"{r['golden']}", "",
         f"Host C tests: ran {r['host_c_tests']['ran']}, passed {r['host_c_tests']['passed']}; "
         f"compiler {r['host_c_tests'].get('compiler')}; output: `{r['host_c_tests']['output'][-200:]}`.", "",
         "## Acceptance (TRAINING only)", "",
         *[f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in a["checks"].items()], "",
         f"INT8 accepted: **{a['int8_accepted']}**. Deployment candidates: {a['deployment_candidate']}. {a['note']}", "",
         f"Sizes (parameter storage only): {r['sizes_host_estimate']}.", "",
         "## Caveats", "", *[f"- {x}" for x in r["caveats"]], ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--splits", default=str(ROOT / "results" / "splits_runwise_v1.json"))
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--out-root", required=True, help="output root (mirrors the repository layout)")
    ap.add_argument("--skip-c-tests", action="store_true")
    a = ap.parse_args()
    r, _, _ = run(find_dataset(a.data_dir), a.splits, a.models, a.out_root, run_c_tests=not a.skip_c_tests)
    print(json.dumps(r["acceptance"], indent=1))
    print("labels:", LABELS)


if __name__ == "__main__":
    main()

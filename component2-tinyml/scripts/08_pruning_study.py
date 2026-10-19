"""Pruning feasibility study E1 (TRAINING window only, no retraining).

Usage (in Colab; see notebooks/colab_pruning.ipynb and docs/PRUNING.md):
    python scripts/08_pruning_study.py --data-dir data/raw --out-root /content/prune_out

Steps: verify the frozen model files and the frozen INT8 parameters (pinned
SHA-256) and the dataset hash; drop OPERATION rows straight after loading;
check the split file; preprocess with the float64 reference pipeline and the
float32 device pipeline; derive the pruned variants in memory (prune.py);
compare them with the frozen Float32 and INT8 models on status-OK TRAINING
rows; write the study report. Nothing is trained, refitted or tuned, and no
model artifact is written or changed.
"""
# ---- Pipeline step 8 of 8: pruning feasibility study E1 (no retraining) ----
# When:   already done once (docs/PRUNING.md section 9); result: evaluated, not adopted
# Reads:  pv_fault_dataset.csv (TRAINING rows only), the frozen Float32 and INT8 models
# Writes: results/pruning_study.json and .md (models/ is never written)
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyml.contract import LABEL_COLUMN, WINDOW_COLUMN  # noqa: E402
from tinyml.device_reference import preprocess_f32, raw_matrix  # noqa: E402
from tinyml.export import contract_feature_config, file_sha256  # noqa: E402
from tinyml.features import STATUS_CODES, transform  # noqa: E402
from tinyml.holdout import PINNED_FILES, PINNED_WEIGHTS_SHA256, inference_frame, verify_frozen_model  # noqa: E402
from tinyml.prune import STUDY_VERSION, run_study  # noqa: E402
from tinyml.quantize import QuantParams  # noqa: E402
from tinyml.splits import load_assignments  # noqa: E402
from tinyml.train import DATASET_SHA256, environment, final_training_rows, git_commit  # noqa: E402

REPORT_REL = Path("results")
PINNED_INT8_FILES = {
    "quant_params.json": "5daa66d21098b20845ea7b9d9d891a7a43fb62fd4d225b9b348936f5de62ab25",
    "manifest.json": "30d073f7b8952bd730a399a37eccf9c57c34dc0c426bf6c8b854a507eaab58d4",
}
ESTIMATE_NOTE = ("ESTIMATE, not a hardware measurement: analytical parameter-storage bytes and multiply-add "
                 "counts. Sparse formats exclude decoding code; no ESP32 flash, RAM, latency or energy "
                 "was measured.")


def find_dataset(data_dir):
    for p in sorted(Path(data_dir).glob("*.csv")):
        if p.name == "pv_fault_dataset.csv" or p.name.endswith("-pv_fault_dataset.csv"):
            return p
    sys.exit(f"pv_fault_dataset.csv not found in {data_dir}")


# Keep TRAINING rows only; OPERATION is never used for this study.
def training_only(df):
    """Drop every OPERATION row before anything else is computed."""
    out = df[df[WINDOW_COLUMN] == "TRAINING"].reset_index(drop=True)
    assert set(out[WINDOW_COLUMN]) == {"TRAINING"}
    return out


# Load the frozen INT8 parameters after checking their pinned hashes.
def load_frozen_int8(int8_dir, pinned=PINNED_INT8_FILES):
    """Frozen INT8 parameters; refuses any file whose SHA-256 is not pinned."""
    int8_dir = Path(int8_dir)
    for name, want in (pinned or {}).items():
        got = file_sha256(int8_dir / name)
        if got != want:
            raise ValueError(f"{name} SHA-256 {got} is not the frozen {want}")
    return QuantParams.from_json((int8_dir / "quant_params.json").read_text(encoding="utf-8"))


def run(data_path, splits_path, model_dir, out_root, code_commit=None,
        expected_dataset_sha256=DATASET_SHA256, pinned=PINNED_FILES, pinned_weights=PINNED_WEIGHTS_SHA256,
        pinned_int8=PINNED_INT8_FILES, expected_rows=3000):
    # 1) Refuse to overwrite; check the dataset, the frozen model and the INT8 file hashes.
    out_report = Path(out_root) / REPORT_REL
    for name in ("pruning_study.json", "pruning_study.md"):
        if (out_report / name).exists():
            raise FileExistsError(f"{out_report / name} exists; study outputs are never overwritten")
    sha = file_sha256(data_path)
    if expected_dataset_sha256 and sha != expected_dataset_sha256:
        raise ValueError(f"dataset SHA-256 {sha} is not the validated {expected_dataset_sha256}")
    contract = verify_frozen_model(model_dir, pinned, pinned_weights)
    contract_sha = file_sha256(Path(model_dir) / "model_contract.json")
    q = load_frozen_int8(Path(model_dir) / "int8", pinned_int8)
    if q.contract_sha256 != contract_sha:
        raise ValueError("frozen INT8 parameters do not belong to this model contract")
    cfg = contract_feature_config(contract)

    # 2) Load TRAINING rows and check the split file agrees.
    df = training_only(pd.read_csv(data_path))                # OPERATION rows dropped here
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"{len(df)} TRAINING rows, expected {expected_rows}")
    assignments, meta = load_assignments(splits_path, df, dataset_sha256=sha)
    final_training_rows(df, assignments)                      # split file agrees

    # 3) Preprocess with both the Python reference and the device float32 pipeline (gates must agree).
    status32, z32 = preprocess_f32(raw_matrix(df), cfg)
    x64, status64 = transform(inference_frame(df), cfg)
    status64 = status64.map(STATUS_CODES).to_numpy()
    if not (status32 == status64).all():
        raise ValueError("float32 device gates differ from features.input_status")
    ok = status32 == STATUS_CODES["OK"]
    y = df.loc[ok, LABEL_COLUMN].to_numpy()

    # 4) Prune in memory (10/20/30 % of weights; 2/3/5 hidden units) and compare with the frozen model.
    study = run_study(contract, q, x64, z32[ok], y)
    report = {
        "status": "PRUNING_STUDY_E1_TRAINING_ONLY", "study_version": STUDY_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {"dataset_sha256": sha, "split_version": meta["split_version"],
                       "contract_sha256": contract_sha, "weights_sha256": contract["weights_sha256"],
                       "model_version": contract["model_version"], "quant_version": q.version,
                       "quant_params_sha256": file_sha256(Path(model_dir) / "int8" / "quant_params.json"),
                       "code_commit": code_commit or git_commit(ROOT), "environment": environment()},
        "operation_window_used": False, "window": "TRAINING",
        "training_rows": int(len(df)), "status_ok_rows": int(ok.sum()), "skipped_rows": int((~ok).sum()),
        "retrained_or_fine_tuned": False, "model_artifacts_written": False,
        "estimate_note": ESTIMATE_NOTE,
        **study,
        "caveats": [
            "All comparisons use TRAINING rows, on which the model was fitted; agreement and training-fit "
            "macro-F1 are not independent accuracy.",
            "OPERATION was not used and is not used to choose a pruning level.",
            "Storage and multiply-add figures are estimates; no ESP32 measurement was made.",
            "Structured variants change the architecture and break the 229-value FL layout (experimental only).",
            "Prediction agreement does not establish electrical safety; OPEN/SHORT outputs stay unconfirmed "
            "ML predictions.",
        ],
    }
    # 5) Write the report; size numbers are estimates, not ESP32 measurements.
    out_report.mkdir(parents=True, exist_ok=True)
    (out_report / "pruning_study.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out_report / "pruning_study.md").write_text(render(report), encoding="utf-8")
    return report


def fmt(x, nd=4):
    return "–" if x is None else f"{x:.{nd}f}"


def render(r):
    p = r["provenance"]
    L = ["# Pruning feasibility study E1 (TRAINING window only)", "",
         f"Study `{r['study_version']}`, created {r['created_utc']}. Model `{p['model_version']}`, weights SHA-256 "
         f"`{p['weights_sha256']}`, INT8 `{p['quant_version']}`. Code `{p['code_commit']}`. "
         f"Environment {p['environment']}.", "",
         f"Rows: {r['training_rows']} TRAINING, {r['status_ok_rows']} status-OK compared, {r['skipped_rows']} "
         "skipped by the input gates. **OPERATION was not used. No retraining or fine-tuning. "
         "No model artifact was written.**", "",
         f"Frozen INT8 vs frozen Float32 agreement: {fmt(r['baseline_int8_agreement_with_float'])}.", "",
         "## Prediction agreement with the frozen Float32 model", "",
         "| variant | hidden | non-zero W (f32 / int8) | f32 agree | f32 changes | f32 OPEN/SHORT | "
         "int8 agree | int8 changes | int8 OPEN/SHORT | int8 vs frozen int8 | E1 f32 | E1 int8 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in r["variants"]:
        f, i, ii = v["pruned_float_vs_frozen_float"], v["pruned_int8_vs_frozen_float"], v["pruned_int8_vs_frozen_int8"]
        L.append(f"| {v['name']} | {v['hidden_units']} | {v['nonzero_weights_float']} / {v['nonzero_weights_int8']} "
                 f"| {fmt(f['agreement'])} | {f['changed']} | {f['safety_changes_total']} | {fmt(i['agreement'])} "
                 f"| {i['changed']} | {i['safety_changes_total']} | {fmt(ii['agreement'])} "
                 f"| {'PASS' if v['e1_checks']['float32']['passed'] else 'FAIL'} "
                 f"| {'PASS' if v['e1_checks']['int8']['passed'] else 'FAIL'} |")
    L += ["", "## Training-fit macro-F1 (frozen Float32 vs variant; not independent accuracy)", "",
          "| variant | frozen f32 | pruned f32 | drop | pruned int8 | drop |", "|---|---|---|---|---|---|"]
    for v in r["variants"]:
        f = v["pruned_float_vs_frozen_float"]["training_fit_macro_f1"]
        i = v["pruned_int8_vs_frozen_float"]["training_fit_macro_f1"]
        L.append(f"| {v['name']} | {fmt(f['float_reference'])} | {fmt(f['int8'])} | {fmt(f['drop'])} "
                 f"| {fmt(i['int8'])} | {fmt(i['drop'])} |")
    L += ["", "## Changed OPEN/SHORT predictions and change pairs", ""]
    for v in r["variants"]:
        for key in ("pruned_float_vs_frozen_float", "pruned_int8_vs_frozen_float"):
            c = v[key]
            L.append(f"- {v['name']} ({key}): OPEN/SHORT {c['safety_changes']}; pairs {c['change_pairs'] or 'none'}")
    L += ["", "## Size and compute ESTIMATES (not hardware measurements)", "", f"_{r['estimate_note']}_", "",
          "| variant | stored params (dense) | f32 dense B | f32 bitmap B | f32 index B | int8 dense B "
          "| int8 bitmap B | int8 index B | MACs dense | MACs zero-skip (int8) | FL layout |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in r["variants"]:
        s, m = v["storage_estimate_bytes"], v["multiply_add_estimate"]
        L.append(f"| {v['name']} | {v['parameters_stored_dense']} | {s['float32']['dense']} | "
                 f"{s['float32']['bitmap_sparse']} | {s['float32']['index_list_sparse']} | {s['int8']['dense']} | "
                 f"{s['int8']['bitmap_sparse']} | {s['int8']['index_list_sparse']} | "
                 f"{m['int8']['dense_multiply_adds']} | {m['int8']['zero_skipping_multiply_adds']} | {v['fl_layout']} |")
    L += ["", "INT8 bytes include the 256-byte tanh table and 13 bytes of constants; biases are int32 / float32.",
          "", "## Conclusion rule", "", r["conclusion_rule"], "",
          "## Caveats", "", *[f"- {x}" for x in r["caveats"]], ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--splits", default=str(ROOT / "results" / "splits_runwise_v1.json"))
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--out-root", required=True, help="output root (report goes to <out-root>/results/)")
    a = ap.parse_args()
    r = run(find_dataset(a.data_dir), a.splits, a.models, a.out_root)
    for v in r["variants"]:
        print(v["name"], v["e1_checks"])


if __name__ == "__main__":
    main()

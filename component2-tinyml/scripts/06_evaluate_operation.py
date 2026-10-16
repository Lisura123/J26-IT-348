"""Independent OPERATION evaluation of the frozen model (run once).

Usage:
    # official final test, once (docs/OPERATION_EVAL.md)
    python scripts/06_evaluate_operation.py --data-dir data/raw --confirm-final-test

    # plumbing check on TRAINING rows; labelled DRY_RUN, not a result
    python scripts/06_evaluate_operation.py --data-dir data/raw --dry-run-training --out-dir /tmp/dry

The final test writes results/operation_eval.json and .md and refuses to run
if either file already exists. Nothing is fitted or changed.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyml.contract import LABELS  # noqa: E402
from tinyml.holdout import STATUS_DRY_RUN, run_evaluation  # noqa: E402
from tinyml.splits import load_assignments  # noqa: E402
from tinyml.train import git_commit  # noqa: E402
from tinyml.validate import file_sha256  # noqa: E402

FINAL_NAMES = ("operation_eval.json", "operation_eval.md")
DRY_RUN_NAMES = ("dry_run_training_eval.json", "dry_run_training_eval.md")
SHORT = {"NORMAL": "NOR", "PARTIAL_SHADING": "PAR", "DUST": "DUS", "OPEN_CIRCUIT": "OPE", "SHORT_CIRCUIT": "SHO"}


def find_dataset(data_dir):
    for p in sorted(Path(data_dir).glob("*.csv")):
        if p.name == "pv_fault_dataset.csv" or p.name.endswith("-pv_fault_dataset.csv"):
            return p
    sys.exit(f"pv_fault_dataset.csv not found in {data_dir}")


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def render(r):
    dry = r["status"] == STATUS_DRY_RUN
    m, cov = r["classification"], r["coverage"]
    title = ("DRY RUN on TRAINING rows — NOT A RESULT" if dry
             else "Independent OPERATION evaluation (final test)")
    pv = r["provenance"]
    L = [f"# {title}", "",
         f"Status `{r['status']}`, protocol `{r['protocol_version']}`, window `{r['window']}`, "
         f"created {r['created_utc']}.", "",
         f"Model `{pv['model_version']}`, weights SHA-256 `{pv['weights_sha256']}`, trained at `{pv['training_commit']}`; "
         f"evaluation code `{pv['evaluation_code_commit']}`. Dataset SHA-256 `{pv['dataset_sha256']}`, split "
         f"`{pv['split_version']}`. Environment {pv['environment']}.", "",
         "Frozen model: no refit, no retraining, no threshold change. Inference uses the model contract "
         "only (NumPy); `label` and `health_score` are removed before inference.", ""]
    if dry:
        L += ["> **This is a plumbing check on the rows the model was trained on. These numbers are "
              "training-fit values and must not be reported as results.**", ""]
    L += ["## Coverage (quality gates)", "",
          f"{cov['total']} readings in {r['rows']['runs']} runs; {cov['classified']} classified, "
          f"{cov['skipped']} skipped {cov['skip_reasons'] or ''}.", "",
          "| Class | Readings | Classified | Coverage | Correct | Coverage-adjusted recall |", "|---|---|---|---|---|---|"]
    for l in LABELS:
        pc, ca = cov["per_class"][l], r["coverage_adjusted_recall"][l]
        L.append(f"| {l} | {pc['total']} | {pc['classified']} | {fmt(pc['coverage'])} | {ca['correct']} | "
                 f"{fmt(ca['recall'])} |")
    runs = ", ".join(f"{k} ({v['classified']}/{v['total']})" for k, v in cov["runs_not_fully_classified"].items())
    L += ["", f"Runs not fully classified: {runs or 'none'}.", "",
          "## Classification (classified readings)", "",
          f"Accuracy {fmt(m['accuracy'])}, macro-F1 {fmt(m['macro_f1'])}, balanced accuracy "
          f"{fmt(m['balanced_accuracy'])}, n = {m['n']}.", "",
          "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for l in LABELS:
        pc = m["per_class"][l]
        L.append(f"| {l} | {fmt(pc['precision'])} | {fmt(pc['recall'])} | {fmt(pc['f1'])} | {pc['support']} |")
    L += ["", "Confusion matrix (rows = true, columns = predicted):", "",
          "| true \\ pred | " + " | ".join(SHORT[l] for l in LABELS) + " |", "|---" * (len(LABELS) + 1) + "|"]
    for l, row in zip(LABELS, m["confusion_matrix"]):
        L.append(f"| {SHORT[l]} | " + " | ".join(str(v) for v in row) + " |")
    b, rl = r["bootstrap"], r["run_level"]
    L += ["", f"Run level: majority vote {fmt(rl['majority_vote_accuracy'])} over {rl['runs']} runs; "
          f"worst run `{rl['worst_run']}` ({fmt(rl['worst_run_accuracy'])}).", "",
          f"95% intervals ({b['method']}, {b['resamples']} resamples, seed {b['seed']}): "
          + "; ".join(f"{k} [{fmt(b[k]['low'])}, {fmt(b[k]['high'])}]" for k in
                      ("accuracy", "macro_f1", "OPEN_CIRCUIT_recall", "SHORT_CIRCUIT_recall") if b[k]) + ".", "",
          "Per site: " + ", ".join(f"{s} {fmt(v['accuracy'])} (n={v['classified']})"
                                   for s, v in (r["per_site"] or {}).items()) + ".", "",
          "## OPEN_CIRCUIT and SHORT_CIRCUIT safety analysis", "",
          "Every OPEN/SHORT output is an **ML prediction, not electrically confirmed** (confirmation rules are "
          "placeholders). Labels are dataset labels. No relay isolation is implemented.", ""]
    for fault, s in r["safety"].items():
        fp, mf = s["false_predictions"], s["missed_faults"]
        L += [f"### {fault}", "",
              f"- False {fault} predictions: {fp['count']} "
              f"({fmt(fp['rate_per_1000_classified_non_fault'], 2)} per 1,000 classified non-{fault} readings); "
              f"by true class {fp['by_true_label'] or 'none'}.",
              f"- Missed {fault} faults: {mf['missed']} of {mf['classified']} classified fault readings; "
              f"{mf['skipped']} skipped; recall {fmt(mf['recall'])}, coverage-adjusted recall "
              f"{fmt(mf['coverage_adjusted_recall'])}.",
              f"- Electrical confirmation: {s['electrical_confirmation']}.", ""]
        if fp["runs"]:
            L += ["| Run with false predictions | Site | True class | False | Readings | Max consecutive |",
                  "|---|---|---|---|---|---|"]
            L += [f"| {x['run_id']} | {x['site_id']} | {x['true_label']} | {x['false_predictions']} | "
                  f"{x['run_readings']} | {x['max_consecutive']} |" for x in fp["runs"]]
            ctx = fp["electrical_context_median"]
            L += ["", f"Median electrical context (descriptive only): false predictions {ctx['false_predictions']}; "
                  f"true detections {ctx['true_fault_detections']}.", ""]
        L += ["| Fault run | Site | Classified | Correct | Missed (as) | Skipped | First detection (s) |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {x['run_id']} | {x['site_id']} | {x['classified']} | {x['correct']} | {x['missed']} "
              f"{x['missed_as'] or ''} | {x['skipped']} | {fmt(x['time_to_first_detection_s'], 0)} |"
              for x in mf["runs"]]
        L.append("")
    lt = r["latency"]
    L += ["## Host CPU latency (not ESP32)", "", lt["note"], f"Rows: {lt['rows']}. Host: {lt['host']}.", "",
          "| Measurement | n | mean µs | median µs | p95 µs | p99 µs |", "|---|---|---|---|---|---|"]
    for k in ("forward_pass_single_reading", "pandas_reference_pipeline_single_reading"):
        s = lt[k]
        L.append(f"| {k} | {s['n']} | {s['mean_us']:.1f} | {s['median_us']:.1f} | {s['p95_us']:.1f} | {s['p99_us']:.1f} |")
    bp = lt["batched_pipeline"]
    L += ["", f"Batched pipeline: {bp['readings']} readings in {bp['total_ms']:.1f} ms "
          f"({bp['per_reading_us']:.1f} µs per reading).", "",
          f"Phase 7 context (TRAINING model selection, not comparable as a test result): {r['phase7_context']}.", "",
          "## Caveats", ""] + [f"- {x}" for x in r["caveats"]] + [""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--splits", default=str(ROOT / "results" / "splits_runwise_v1.json"))
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--confirm-final-test", action="store_true",
                      help="run the one-time OPERATION evaluation")
    mode.add_argument("--dry-run-training", action="store_true",
                      help="plumbing check on TRAINING rows (not a result)")
    a = ap.parse_args()
    run(find_dataset(a.data_dir), a.splits, a.models, a.out_dir,
        confirm_final_test=a.confirm_final_test, dry_run_training=a.dry_run_training)


def run(data_path, splits_path, model_dir, out_dir, confirm_final_test=False, dry_run_training=False,
        code_commit=None, **kw):
    """Evaluate and write the JSON and Markdown report. The final test never
    overwrites existing results."""
    out = Path(out_dir)
    names = DRY_RUN_NAMES if dry_run_training else FINAL_NAMES
    if not dry_run_training:
        existing = [n for n in names if (out / n).exists()]
        if existing:
            raise FileExistsError(f"{existing} already exist in {out}; the OPERATION final test is run "
                                  "once and its results are never overwritten")
    sha = file_sha256(data_path)
    df = pd.read_csv(data_path)
    assignments, meta = load_assignments(splits_path, df, dataset_sha256=sha)
    result = run_evaluation(df, assignments, model_dir, sha, split_meta=meta,
                            confirm_final_test=confirm_final_test, dry_run_training=dry_run_training,
                            code_commit=code_commit or git_commit(ROOT), **kw)
    out.mkdir(parents=True, exist_ok=True)
    mode = "w" if dry_run_training else "x"          # "x": fail if the file appeared meanwhile
    with open(out / names[0], mode, encoding="utf-8") as f:
        f.write(json.dumps(result, indent=1) + "\n")
    with open(out / names[1], mode, encoding="utf-8") as f:
        f.write(render(result))
    print(f"{result['status']}: wrote {out / names[0]} and {out / names[1]}")
    return result, [out / n for n in names]


if __name__ == "__main__":
    main()

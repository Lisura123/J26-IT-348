"""Phase 7: compare baseline models and feature sets on TRAINING-window folds.

Usage (locally or in Colab):
    python scripts/04_compare_baselines.py --data-dir data/raw

Writes results/baselines_phase7.json (every fit) and
results/baselines_phase7.md (summary, v_deficit ablation, coverage).
The OPERATION window is not used.
"""
# ---- Pipeline step 4 of 8: compare candidate models and feature sets (model selection) ----
# When:   after step 3; its results chose the final model (physics8 + 8-16-5 tanh MLP)
# Reads:  pv_fault_dataset.csv (TRAINING folds only), results/splits_runwise_v1.json
# Writes: results/baselines_phase7.json, results/baselines_phase7.md
import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyml.experiment import (  # noqa: E402
    DEFAULT_SEEDS, FEATURE_SETS, PROTOCOLS, ablation_table, run_comparison, summary_table,
)
from tinyml.models import MODEL_NAMES  # noqa: E402
from tinyml.splits import load_assignments  # noqa: E402
from tinyml.validate import file_sha256  # noqa: E402


def find_dataset(data_dir):
    for p in sorted(Path(data_dir).glob("*.csv")):
        if p.name == "pv_fault_dataset.csv" or p.name.endswith("-pv_fault_dataset.csv"):
            return p
    sys.exit(f"pv_fault_dataset.csv not found in {data_dir}")


# Format a number for the report ('–' when missing).
def fmt(x, nd=3):
    return "–" if x is None or pd.isna(x) else f"{x:.{nd}f}"


# Build the Markdown report: metrics per protocol, the v_deficit ablation and coverage.
def render(result, table, sha):
    env = result["environment"]
    L = ["# Phase 7: baseline comparison (TRAINING window only)", "",
         f"Dataset SHA-256 `{sha}`. Python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, "
         f"scikit-learn {env['sklearn']}. Seeds for stochastic models: {result['seeds']}.", "",
         "The OPERATION window was not used. Reading-level metrics are computed on classified readings; "
         "coverage is reported separately. **Pooled** = one metric over the combined held-out "
         "predictions of all folds; **fold mean/min** = statistics over per-fold values. Recall is "
         "not reported for classes absent from a held-out partition. Size = learned parameters or "
         "tree nodes (an indicator only, not measured flash/RAM).", ""]
    for protocol in result["protocols"]:
        t = table[table.protocol == protocol].sort_values(["feature_set", "pooled_macro_f1"],
                                                          ascending=[True, False])
        L += [f"## {protocol}", "", "### Reading-level metrics", "",
              "| Feature set | Model | pooled macro-F1 | fold mean | fold std | fold min | "
              "pooled OPEN recall | pooled SHORT recall | size | fits |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            L.append(f"| {r.feature_set} | {r.model} | {fmt(r.pooled_macro_f1)} | {fmt(r.macro_f1_mean)} | "
                     f"{fmt(r.macro_f1_std)} | {fmt(r.macro_f1_min)} | {fmt(r.pooled_open_recall)} | "
                     f"{fmt(r.pooled_short_recall)} | {int(r['size'])} | {int(r.fits)} |")
        L += ["", "### Run-level metrics (majority vote per run, pooled over folds)", "",
              "| Feature set | Model | runs | majority-vote accuracy | worst single-run reading accuracy |",
              "|---|---|---|---|---|"]
        for _, r in t.iterrows():
            L.append(f"| {r.feature_set} | {r.model} | {int(r.runs)} | {fmt(r.run_vote_acc)} | "
                     f"{fmt(r.worst_run_acc)} |")
        for metric, title in (("pooled_macro_f1", "pooled"), ("macro_f1_mean", "fold mean")):
            ab = ablation_table(table, protocol, metric)
            L += ["", f"### v_deficit ablation ({title} macro-F1)", "",
                  "| Model | with v_deficit | without | difference |", "|---|---|---|---|"]
            for model, r in ab.iterrows():
                L.append(f"| {model} | {fmt(r.with_v_deficit)} | {fmt(r.without_v_deficit)} | "
                         f"{fmt(r.difference, 4)} |")
        short = {"NORMAL": "NOR", "PARTIAL_SHADING": "PAR", "DUST": "DUS",
                 "OPEN_CIRCUIT": "OPE", "SHORT_CIRCUIT": "SHO"}
        cov = {}
        for rec in result["records"]:
            if rec["protocol"] == protocol:
                cov.setdefault(rec["fold"], rec["coverage"])
        L += ["", "### Held-out partitions: classes present and coverage", "",
              "| Fold | Runs | Classes in partition | Classes classified | Total | Classified | Skipped | "
              "Skip reasons | Runs not fully classified |", "|---|---|---|---|---|---|---|---|---|"]
        for tc in result["test_classes"]:
            if tc["protocol"] != protocol:
                continue
            c = cov[tc["fold"]]
            runs = ", ".join(f"{k} ({v['classified']}/{v['total']})"
                             for k, v in c["runs_not_fully_classified"].items()) or "none"
            L.append(f"| {tc['fold']} | {tc['runs']} | {'/'.join(short[x] for x in tc['classes_in_partition'])} | "
                     f"{'/'.join(short[x] for x in tc['classes_classified'])} | {c['total']} | "
                     f"{c['classified']} | {c['skipped']} | {c['skip_reasons'] or 'none'} | {runs} |")
        L.append("")
    L += ["These are model-selection results on TRAINING folds. They are not the final test "
          "result (TRAINING → OPERATION), which is run once after the model is chosen.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--splits", default=str(ROOT / "results" / "splits_runwise_v1.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    ap.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=list(FEATURE_SETS))
    a = ap.parse_args()

    # 1) Load the dataset and the saved splits; the dataset fingerprint must match the split file.
    path = find_dataset(a.data_dir)
    sha = file_sha256(path)
    df = pd.read_csv(path)
    assignments, meta = load_assignments(a.splits, df, dataset_sha256=sha)
    # 2) Train and score every model x feature set x seed on the TRAINING folds (OPERATION not used).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_comparison(df, assignments, protocols=a.protocols, models=a.models,
                                feature_sets=a.feature_sets, seeds=tuple(a.seeds))
    # 3) Add provenance and any training warnings to the result.
    result["dataset_sha256"] = sha
    result["split_version"] = meta["split_version"]
    result["warnings"] = sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught})
    # 4) Summarise, then write the JSON (every fit) and Markdown reports.
    table = summary_table(result)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines_phase7.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out / "baselines_phase7.md").write_text(render(result, table, sha), encoding="utf-8")
    print(f"{len(result['records'])} fits; wrote {out / 'baselines_phase7.json'} and {out / 'baselines_phase7.md'}")
    if result["warnings"]:
        print("Warnings during training:", *result["warnings"], sep="\n  ")


if __name__ == "__main__":
    main()

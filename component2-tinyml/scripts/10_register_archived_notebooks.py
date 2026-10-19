"""Register the executed Colab notebooks as archived runs (byte-exact).

Usage:
    python scripts/10_register_archived_notebooks.py           # write notebooks/ARCHIVED_RUNS.json
    python scripts/10_register_archived_notebooks.py --check   # verify only (exit 1 on any difference)

Nothing in a notebook is changed. For every archived notebook the registry
records its exact SHA-256, the commit that saved it, its saved settings
(APPROVED_COMMIT, CONFIRM_*), the official record it documents, what a
re-run would do, and the SHA-256 of its normalised code taken from the
reviewed template commit (REVIEWED_TEMPLATE_COMMIT), which the saved code must
still equal. Writing refuses to replace an existing registry entry whose
SHA-256 differs: a changed archived notebook needs a new, reviewed decision.
"""
# ---- Pipeline step support: register executed Colab notebooks as archived runs ----
# When:   after a notebook is saved from Colab and reviewed (--check verifies the registry)
# Reads:  the saved notebooks (never changed)
# Writes: notebooks/ARCHIVED_RUNS.json
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyml import notebook_archive as na  # noqa: E402

REVIEWED_TEMPLATE_COMMIT = "95af2d25e1eab1e7471aa977ae674ee6b74121d2"   # last reviewed template versions

# One entry per archived notebook: what it documents and what a re-run would do.
ENTRIES = {
    "component2-tinyml/notebooks/colab_operation_eval.ipynb": {
        "documents": "Phase 9 one-time OPERATION evaluation: docs/OPERATION_RESULTS.md, results/operation_eval.json",
        "official_run_commit": None,
        "rerun_effect": "BLOCKED: the notebook clones the latest branch, and scripts/06_evaluate_operation.py "
                        "refuses the final test while results/operation_eval.json is committed.",
    },
    "component2-tinyml/notebooks/colab_quantization.ipynb": {
        "documents": "int8-v1 quantisation run: docs/QUANTIZATION.md section 12, models/int8/, "
                     "results/quantization_report.*",
        "official_run_commit": "ef6e9ae9badeeb06b149bc1177d546d014d54ed2",
        "rerun_effect": "NOT BLOCKED: pins ef6e9ae; Run all repeats the deterministic TRAINING-only quantisation "
                        "into /content/quant_out. Committed artifacts are not changed. Accepted by Member 2.",
    },
    "component2-tinyml/notebooks/colab_pruning.ipynb": {
        "documents": "Pruning study E1: docs/PRUNING.md section 9, results/pruning_study.*",
        "official_run_commit": "4f9effeb121fbf86abd8e4eb2de2735457007eea",
        "rerun_effect": "NOT BLOCKED: pins 4f9effe; Run all repeats the deterministic TRAINING-only study into "
                        "/content/prune_out. Committed results are not changed. Accepted by Member 2.",
    },
    "component2-tinyml/notebooks/colab_final_training.ipynb": {
        "documents": "Phase 8 final training: docs/TRAINING_EXPORT.md section 10, models/",
        "official_run_commit": None,
        "rerun_effect": "NOT BLOCKED: Run all retrains on TRAINING into /content/final_model; models/ in the "
                        "repository is not changed (frozen by pinned SHA-256). Accepted by Member 2.",
    },
    "component2-tinyml/notebooks/colab_phase7_baselines.ipynb": {
        "documents": "Phase 7 baseline comparison: docs/PHASE7_RESULTS.md, results/baselines_phase7.*",
        "official_run_commit": None,
        "rerun_effect": "NOT BLOCKED: Run all repeats the TRAINING-only cross-validation in Colab; committed "
                        "results are not changed. Accepted by Member 2.",
    },
    "component2-tinyml/notebooks/colab_train.ipynb": {
        "documents": "Colab environment check: results/colab_env_check.json",
        "official_run_commit": None,
        "rerun_effect": "NOT BLOCKED: repeats the environment and test check; no training, no OPERATION.",
    },
    "colab_train.ipynb": {
        "documents": "Personal Colab environment-check session saved at the repository root (not a repository "
                     "template; outside component2-tinyml)",
        "official_run_commit": None,
        "rerun_effect": "NOT BLOCKED: repeats the environment check (dataset hash, TRAINING-only feature smoke "
                        "test, pytest); no training, no OPERATION.",
        "template_path": None,
    },
}


def git(*args):
    return subprocess.run(["git", "-C", str(na.REPO), *args], capture_output=True, text=True, check=True).stdout


def template_code_sha256(rel):
    path = ENTRIES[rel].get("template_path", rel)
    if path is None:
        return None
    return na.normalised_code_sha256(json.loads(git("show", f"{REVIEWED_TEMPLATE_COMMIT}:{path}")))


# Build the registry: exact SHA-256, saved settings and the reviewed-code check for each notebook.
def build(existing=None):
    old = {e["path"]: e for e in (existing or {}).get("notebooks", [])}
    entries = []
    for rel, info in ENTRIES.items():
        p = na.REPO / rel
        nb = na.load(p)
        sha = na.file_sha256(p)
        if rel in old and old[rel]["sha256"] != sha:
            raise SystemExit(f"{rel}: SHA-256 differs from the registered archive; not re-registered")
        tsha = template_code_sha256(rel)
        if tsha is not None and tsha != na.normalised_code_sha256(nb):
            raise SystemExit(f"{rel}: code differs from the reviewed template {REVIEWED_TEMPLATE_COMMIT[:7]}")
        entries.append({"path": rel, "state": na.ARCHIVED, "sha256": sha,
                        "saved_in_commit": git("log", "-1", "--format=%H", "--", rel).strip(),
                        "settings": na.settings(nb), "has_outputs": na.has_outputs(nb),
                        "template_code_sha256": tsha, **{k: v for k, v in info.items() if k != "template_path"}})
    return {"registry_version": na.REGISTRY_VERSION,
            "reviewed_template_commit": REVIEWED_TEMPLATE_COMMIT,
            "note": "Executed Colab notebooks kept exactly as saved (outputs and settings, including CONFIRM_* = "
                    "True). Do not use Run all on them. New runs start from a clean template.",
            "notebooks": entries}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    # Build the registry, then either compare (--check) or write it.
    existing = na.load(na.REGISTRY) if na.REGISTRY.exists() else None
    reg = build(existing)
    text = json.dumps(reg, indent=1, sort_keys=True) + "\n"
    if a.check:
        same = na.REGISTRY.exists() and na.REGISTRY.read_text(encoding="utf-8") == text
        print("registry up to date" if same else "registry differs")
        sys.exit(0 if same else 1)
    na.REGISTRY.write_text(text, encoding="utf-8")
    print(f"registered {len(reg['notebooks'])} archived notebooks")


if __name__ == "__main__":
    main()

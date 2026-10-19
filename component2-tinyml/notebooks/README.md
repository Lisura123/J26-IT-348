# Colab notebooks: archived runs

The notebooks in this folder (and `colab_train.ipynb` at the repository root) were saved
from Google Colab after the official runs ("Save a copy in GitHub"). They are kept
**exactly as saved**, with their outputs, approved commit hashes and `CONFIRM_* = True`,
so GitHub shows the executed runs in its Preview. They are **archived executed
notebooks**, not templates.

**Do not use "Run all" on them.** Each one is registered with its exact SHA-256 in
`ARCHIVED_RUNS.json`, together with the official record it documents and what a re-run
would do:

| Notebook | Re-run effect |
|---|---|
| `colab_operation_eval.ipynb` | **blocked**: `scripts/06_evaluate_operation.py` refuses the OPERATION final test while `results/operation_eval.json` is committed |
| `colab_quantization.ipynb`, `colab_pruning.ipynb` | not blocked: they pin the original commits and would repeat the TRAINING-only computation in `/content`; committed results are not changed |
| `colab_final_training.ipynb`, `colab_phase7_baselines.ipynb`, `colab_train.ipynb`, `../colab_train.ipynb` | not blocked: TRAINING-only work in `/content`; committed models and results are not changed |

Rules checked by `tests/test_notebook_archive.py` and the notebook tests:

- every notebook is either a **clean template** (guards `False`, no hash, no outputs) or a
  **registered archive** whose bytes match its SHA-256;
- archived notebooks keep the reviewed safety code (only setting values and outputs differ);
- the guard logic is still tested with explicitly set values (refuses without
  confirmation, pins the exact approved commit, refuses other branches).

A future run should start from a clean template (the reviewed versions at commit
`95af2d2`), and the executed result is archived with
`python scripts/10_register_archived_notebooks.py` after review.

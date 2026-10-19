# Phase 7: Baseline Comparison Design

Code: `src/tinyml/models.py`, `src/tinyml/evaluate.py`, `src/tinyml/experiment.py`. Script: `scripts/04_compare_baselines.py`. Colab: `notebooks/colab_phase7_baselines.ipynb`. Tests: `tests/test_experiment.py`.

Purpose: decide, on TRAINING data only, whether the planned 8-16-5 MLP and the PV-physics features are justified, compared with simpler models and with the raw sensor features.

## What is compared

| Axis | Values |
|---|---|
| Models | `rules` (one learned threshold per fault: current, voltage, dust, temp_spread), `logreg` (softmax regression), `tree` (depth ≤ 4), `forest` (20 trees, depth ≤ 6), `mlp` (8-16-5 tanh, the ESP32/FL architecture) |
| Feature sets | `physics8` (pv8-v1), `physics7_no_v_deficit` (ablation for Finding B), `raw8` (Member 3's raw sensor columns, standardised) |
| Protocols | `cv`: Phase 4 run-wise 5-fold CV over TRAINING runs. `loso_training`: leave-one-site-out within TRAINING runs |
| Seeds | 0, 1, 2 for `forest` and `mlp`; deterministic models run once |

`rules` is skipped for `raw8`, because it needs `temp_spread`.

## Rules that keep the comparison fair
- Every learned quantity (V_ref, scalers, rule thresholds, model weights) is fitted on the training part of each fold only.
- All feature sets use the same quality gate, so every model is evaluated on exactly the same readings.
- **The OPERATION window is not used.** `protocol_folds()` refuses the TRAINING → OPERATION protocol unless `allow_final_test=True`. That final test is run once, after the model is chosen.
- `health_score`, `run_id`, `site_id`, `record_id`, timestamps and `window` are never inputs.
- Library versions are stored with the results (Colab and local scikit-learn versions differ).

## Model sizes (learned parameters)

| Model | physics8 (8 inputs) | physics7_no_v_deficit (7 inputs) | raw8 (8 inputs) |
|---|---|---|---|
| mlp 16 hidden, 5 outputs | 8·16 + 16 + 16·5 + 5 = **229** | 7·16 + 16 + 16·5 + 5 = **213** | **229** |
| logreg | 8·5 + 5 = 45 | 7·5 + 5 = 40 | 45 |
| rules | 4 thresholds | 4 thresholds | not applicable |
| tree / forest | node count, reported per fit | | |

The 213-parameter ablation MLP is used only for the comparison. It does not fit Member 3's fixed 229-parameter FL layout.

## Pooled and fold-level metrics
- **Pooled:** the held-out predictions of all folds for one (protocol, feature set, model, seed) are concatenated, and one metric is computed over them. Pooled values are averaged over seeds for stochastic models.
- **Fold mean / std / min:** statistics over the per-fold values.
- **Reading-level** (macro-F1, per-class recall) and **run-level** (majority vote per run, worst single-run accuracy) metrics are reported in separate tables.
- **Classes per held-out partition:** each fold lists the classes present in it. Recall is `None` (not reported) for classes absent from a partition. Macro-F1 uses only the classes present.

## Reported per fit
- macro-F1 (primary), balanced accuracy, accuracy, per-class precision/recall/F1, confusion matrix.
- OPEN/SHORT recall.
- Run-level majority-vote accuracy and worst-run accuracy.
- Coverage: total, classified and skipped readings, skip reasons, per class and per run (Finding A).
- Model size as parameter or tree-node count. This is an indicator only, not measured flash or RAM.

## How results are read
- The **v_deficit ablation** compares `physics8` with `physics7_no_v_deficit`, per model and protocol (Finding B).
- `raw8` vs `physics8` shows whether the PV-specific features help.
- If `rules` or `logreg` match the MLP, that is reported. The MLP is then justified by FL compatibility, not by accuracy (ARCHITECTURE §6.2).
- `loso_training` folds lack some classes. Macro-F1 is computed over the classes present in each fold.
- These are model-selection results. They are not final-test results.

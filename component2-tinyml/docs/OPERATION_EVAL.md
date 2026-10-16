# Independent OPERATION Evaluation: Protocol

Code: `src/tinyml/holdout.py`. Script: `scripts/06_evaluate_operation.py`. Colab: `notebooks/colab_operation_eval.ipynb`. Tests: `tests/test_operation_eval.py` (synthetic data only).

**Status:** protocol fixed and committed **before** the official run. The OPERATION evaluation has **not** been run. This document is not changed after the run; results and their interpretation go into separate files.

Protocol version: `operation-eval-v1`.

## 1. Purpose

Measure, once, how the frozen final model (`docs/TRAINING_EXPORT.md`) performs on the OPERATION window: 50 runs that were not used for training, feature fitting, model selection, seed choice or threshold setting.

## 2. Frozen inputs (checked before evaluation)

| Item | Pinned value |
|---|---|
| `models/model_contract.json` | SHA-256 `752f62285cca62b59a629686fd5b941b748a811c78808751c04595ee3452cb2a` |
| `models/feature_config.json` | SHA-256 `421b5d4ed7e2092b6da3e2f2f48e000a0ecda5ef0542e5a660c65fd607128a86` |
| `models/training_metadata.json` | SHA-256 `76389fc889c9b0097ba0d944c32e535930c835c7d15beef018b1d00f48f5501c` |
| `models/manifest.json` | SHA-256 `989cf6ece67ad8be044dd981ae1d8a9ba830798b5b56dbed5c45d7b71303a526` |
| Weights | SHA-256 `94e7ab9384a1906ec2cf0d3176b435e700f15cd99552fbc6c71519c648ecc3a7` |
| `pv_fault_dataset.csv` | SHA-256 `5fbb93471c056c274cbf81e8c317934e379b2484013147e4c2cacfa93421595b` |
| Split | `results/splits_runwise_v1.json` (`runwise-v1`), temporal `test` side |

`verify_frozen_model()` runs `export.verify_artifacts()` and compares every pinned hash. Any difference stops the evaluation.

## 3. Procedure

1. Check the dataset hash and the frozen artifacts (section 2).
2. Select the rows whose `window` is OPERATION **and** whose run is on the temporal `test` side of the split file. The two must agree, no run may also appear in TRAINING, and there must be 3,000 rows in 50 runs.
3. Remove `label` and `health_score`, then run `export.predict_raw()`. This is contract-only NumPy inference: quality gates, pv8-v1 features with the frozen V_ref, the frozen scaler, and the 8-16-5 tanh forward pass.
4. Join the labels back **only for scoring**.
5. Compute every output in section 4 in the same run and write `results/operation_eval.json` and `.md`.

**Never done:** refitting V_ref or the scaler, training or changing weights, changing the low-light (2,400 lx) or saturation (100,000 lx) thresholds, changing reference probes, seed or hyperparameters. The evaluation module does not import `fit_pipeline`, `train_final` or `make_model`, and a test fails if any fitting function is called.

## 4. Outputs (all defined before the run)

**Classification** (on classified readings, always shown next to coverage):
- accuracy, macro-F1 over the classes present, balanced accuracy
- per-class precision, recall, F1 and support
- confusion matrix as counts and row-normalised (rows = true, LABELS order)
- run-level majority vote and the worst run
- accuracy per site (descriptive)

**Quality gates and coverage:**
- total, classified and skipped readings; skip reasons; coverage per class and per run
- **coverage-adjusted recall** = correct predictions ÷ all readings of the class, skipped readings included. A skipped reading counts as not detected.

**Uncertainty:** 95% percentile intervals from a **run-level bootstrap**: whole runs are resampled with replacement within each true class, 1,000 resamples, seed 42. Reported for accuracy, macro-F1, OPEN_CIRCUIT recall and SHORT_CIRCUIT recall. A run whose readings are all skipped has no classified readings and does not enter the bootstrap.

**OPEN_CIRCUIT / SHORT_CIRCUIT safety analysis** (separately for each):
- **False predictions:** count; rate per 1,000 classified readings of other classes; true class of each; per run: run, site, true class, number of false predictions and the longest run of consecutive false predictions; median `current_ma`, `voltage_v` and `irradiance_lux` of the false predictions compared with correct detections (descriptive only, no threshold is derived).
- **Missed faults:** fault readings, classified, correct, missed (and what they were predicted as), skipped; recall and coverage-adjusted recall; per fault run, the time to the first correct detection (reading index × 5 s).
- Phase 7 Finding 5 (a shading run predicted as OPEN_CIRCUIT) is checked through the same false-prediction table.

**Predictions vs confirmed electrical faults:** every OPEN/SHORT output is an **ML prediction that has not been electrically confirmed**. The electrical confirmation layer is a placeholder (`rules.status = PLACEHOLDER_NOT_IMPLEMENTED`), so confirmed electrical faults are reported as **not assessed**. Ground truth is the dataset label, not a measured electrical fault. **No relay isolation is implemented**, and the streak lengths must not be used to set `persist_n`.

**Host CPU latency (not ESP32):** timed on TRAINING rows (labels removed), so OPERATION is not needed for it: the NumPy forward pass for one reading (1,000 timings after 100 warm-up calls), the pandas reference pipeline for one reading (200 timings), and the batched pipeline. Mean, median, p95 and p99 in µs, with host CPU details. These figures say nothing about ESP32 flash, RAM or latency, which must be measured on the device.

**Context only:** the Phase 7 TRAINING-window estimates (CV pooled macro-F1 0.981, LOSO 0.980) are shown next to the results. They are not used for any decision.

## 5. Known before the run

Counted in earlier phases without any model (gate status and counts only):
- OPERATION has 3,000 readings in 50 runs, 10 runs per class.
- Run `SITE-D-DUS-12` (DUST, 60 readings, 683–1,306 lx) is entirely below the low-light threshold, so 60 readings are expected to be skipped, and only 9 DUST runs can be classified (`FEATURES.md`, Finding A). No saturated or invalid OPERATION readings were counted.

## 6. Prior OPERATION exposure (disclosure)

| Phase | What touched the OPERATION window | Effect on the model |
|---|---|---|
| Dataset validation | Row, label and schema counts per window; the stand-in file `member2_fault_predictions.csv` (OPERATION rows only, not produced by this component) was compared with the labels: 72.07% agreement, and it was rejected as ground truth | none |
| Run-wise splits | Runs and rows per class per window | none |
| Feature review (Finding A) | Input-gate status was **counted** for OPERATION rows. No feature values, V_ref residuals or model inputs were computed | none; thresholds stayed as placeholders and were not tuned |
| Member 3 FL (other component) | Member 3's FL code validates a different, raw-input model on OPERATION in every round | none; not our model |

No Component 2 model has ever produced predictions on OPERATION rows, and no Component 2 decision (features, thresholds, model, seed, hyperparameters, probes) used OPERATION results.

## 7. Run once: guards and re-run policy

- `run_evaluation()` refuses the OPERATION window unless `confirm_final_test=True`. The script needs `--confirm-final-test`, and the Colab notebook needs `CONFIRM_FINAL_TEST = True` to be set by hand.
- The script refuses to run if `operation_eval.json` or `operation_eval.md` already exists, and writes with exclusive file creation.
- **Crash before any metric is shown:** fix the plumbing, run again, and record it in the results document.
- **Bug found after results are shown:** only reporting or plumbing bugs may be fixed. Both runs are kept and documented. The model, preprocessing and thresholds are never changed.
- **Any later change to the model, features or thresholds** makes a new model version. OPERATION is then no longer independent for it, and any result on it is reported as post-hoc.
- A dry run on **TRAINING rows** (`--dry-run-training`, status `DRY_RUN_TRAINING_NOT_A_RESULT`) may be used to check the plumbing. Its numbers are training-fit values and are not reported.

## 8. Reproducibility

- Inference is deterministic NumPy from the contract. The bootstrap has a fixed seed. Latency varies between hosts and runs and is not reproducible exactly.
- The result records the dataset hash, split version, artifact hashes, training commit, evaluation code commit and library versions.
- Metrics use the same functions as Phase 7 (`evaluate.classification_metrics`, `run_level`, `coverage`).

## 9. What is not claimed

- No ESP32 deployment, INT8 build or on-device measurement.
- No electrical confirmation of faults and no relay isolation.
- No federated learning integration (`TRAINING_EXPORT.md` §7).
- No field validation. Dataset provenance is UNRESOLVED (`DATA_PROVENANCE.md`), so results describe performance on the provided dataset only.

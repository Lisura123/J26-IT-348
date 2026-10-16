# Independent OPERATION Evaluation: Results

**Status:** the frozen final model was evaluated **once** on the OPERATION window, following the protocol fixed beforehand in `docs/OPERATION_EVAL.md` (`operation-eval-v1`, committed in `a2a85d6`). These are results **on the provided dataset**. They are not real-world or hardware validation (section 8).

Raw results: `results/operation_eval.json` and `results/operation_eval.md`, the unmodified Colab output.

## 1. Run record

| Item | Value |
|---|---|
| Run | Google Colab, 2026-10-16T09:29:54Z, status `FINAL_TEST_OPERATION` |
| Evaluation code | `a2a85d6` (Add Independent OPERATION Evaluation Pipeline) |
| Model | `c2-local-v1`, trained at `9a641d5`, weights SHA-256 `94e7ab9384a1906ec2cf0d3176b435e700f15cd99552fbc6c71519c648ecc3a7` |
| Model files | `model_contract.json`, `feature_config.json`, `training_metadata.json`, `manifest.json`: all four SHA-256 values equal the committed ones (checked before evaluation and recorded in the result) |
| Dataset | `pv_fault_dataset.csv`, SHA-256 `5fbb93471c056c274cbf81e8c317934e379b2484013147e4c2cacfa93421595b`, split `runwise-v1` |
| Environment | Python 3.13.16, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1 |
| Frozen | no refit, no retraining, no threshold change; contract-only NumPy inference; `label` and `health_score` removed before inference |

**Run history.** The OPERATION evaluation ran exactly once. Before it, the notebook was started with "Run all" while `CONFIRM_FINAL_TEST` was `False`. Step 6 then stopped at the confirmation guard **before any OPERATION row was read**, and no result was produced. The dataset was also uploaded a second time in the same session (saved as `pv_fault_dataset (1).csv`); its SHA-256 matched the validated file. After the confirmation was set by hand, the evaluation cell ran once without errors.

**Checks after download:** status, window and protocol; dataset, model-file and weights hashes; frozen flags; 3,000 rows in 50 runs; coverage totals; confusion-matrix totals and accuracy; coverage-adjusted recall and OPEN/SHORT counts against the confusion matrix; bootstrap settings; and the Markdown report equal to one regenerated from the JSON. All passed. The evaluation was not run again for these checks.

## 2. Coverage (quality gates)

| Class | Readings | Classified | Skipped | Coverage |
|---|---|---|---|---|
| NORMAL | 600 | 600 | 0 | 1.000 |
| PARTIAL_SHADING | 600 | 600 | 0 | 1.000 |
| DUST | 600 | 540 | 60 | 0.900 |
| OPEN_CIRCUIT | 600 | 600 | 0 | 1.000 |
| SHORT_CIRCUIT | 600 | 600 | 0 | 1.000 |
| **Total** | **3,000** | **2,940** | **60** | **0.980** |

All 60 skipped readings are `LOW_LIGHT` and come from one run, `SITE-D-DUS-12` (0 of 60 classified). This was known and written down before the run (`OPERATION_EVAL.md` §5, `FEATURES.md` Finding A). No saturated or invalid readings occurred. The low-light threshold (2,400 lx) is a placeholder and was not changed.

## 3. Classification results (2,940 classified readings)

| Metric | Value | 95% interval |
|---|---|---|
| Accuracy | **0.979** | 0.941 – 0.999 |
| Macro-F1 | **0.978** | 0.937 – 0.999 |
| Balanced accuracy | 0.977 | – |
| Run-level majority vote | 0.980 (48 of 49 classifiable runs) | – |

Intervals: run-level bootstrap, runs resampled within each true class, 1,000 resamples, seed 42, percentile method. DUST enters with 9 runs, because `SITE-D-DUS-12` has no classified readings.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| NORMAL | 0.910 | 1.000 | 0.953 | 600 |
| PARTIAL_SHADING | 0.997 | 1.000 | 0.998 | 600 |
| DUST | 1.000 | 0.887 | 0.940 | 540 |
| OPEN_CIRCUIT | 1.000 | 1.000 | 1.000 | 600 |
| SHORT_CIRCUIT | 1.000 | 1.000 | 1.000 | 600 |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | NORMAL | PARTIAL_SHADING | DUST | OPEN_CIRCUIT | SHORT_CIRCUIT |
|---|---|---|---|---|---|
| NORMAL | 600 | 0 | 0 | 0 | 0 |
| PARTIAL_SHADING | 0 | 600 | 0 | 0 | 0 |
| DUST | 59 | 2 | 479 | 0 | 0 |
| OPEN_CIRCUIT | 0 | 0 | 0 | 600 | 0 |
| SHORT_CIRCUIT | 0 | 0 | 0 | 0 | 600 |

**Coverage-adjusted recall** (correct ÷ all readings of the class, skipped readings counted as not detected): NORMAL 1.000, PARTIAL_SHADING 1.000, **DUST 0.798** (479 / 600), OPEN_CIRCUIT 1.000, SHORT_CIRCUIT 1.000.

## 4. DUST weakness and `SITE-C-DUS-06`

- DUST is the only class with errors. Of the 600 DUST readings, 60 were skipped (`SITE-D-DUS-12`), 59 were predicted as NORMAL and 2 as PARTIAL_SHADING. All 61 classification errors in the whole evaluation are DUST readings.
- **`SITE-C-DUS-06`** is the worst run: only 5 of its 60 readings (0.083) were classified as DUST, and 55 were misclassified. It is the only run whose majority vote is wrong. All 55 SITE-C errors come from this run (SITE-C accuracy 0.917 over 660 readings). The other 6 DUST errors are at SITE-D (4) and SITE-E (2).
- The DUST errors also explain the lower NORMAL precision (0.910): every reading predicted as NORMAL that was not NORMAL was a DUST reading.
- **What this means:** in this evaluation, dust soiling is the class most likely to be missed, either because the reading is skipped in low light or because the model calls it NORMAL. A missed DUST reading means maintenance would not be flagged; it does not cause a false fault alarm.
- **What is not done:** the cause of `SITE-C-DUS-06` was not investigated during the evaluation, and nothing (features, thresholds, model) is changed because of it. Any later analysis of this run is post-hoc and must be reported as such. A model changed because of it would no longer have an independent OPERATION result (`OPERATION_EVAL.md` §7).

## 5. OPEN_CIRCUIT and SHORT_CIRCUIT safety results

| | OPEN_CIRCUIT | SHORT_CIRCUIT |
|---|---|---|
| False predictions | **0** (0.00 per 1,000 classified other readings) | **0** (0.00 per 1,000) |
| Missed faults | **0** of 600 classified fault readings | **0** of 600 |
| Skipped fault readings | 0 | 0 |
| Recall / coverage-adjusted recall | 1.000 / 1.000 | 1.000 / 1.000 |
| 95% interval for recall | 1.000 – 1.000 | 1.000 – 1.000 |
| First correct detection | at the first reading (0 s) in all 10 fault runs | at the first reading (0 s) in all 10 fault runs |

- The Phase 7 false-alarm pattern (a PARTIAL_SHADING run predicted as OPEN_CIRCUIT, Finding 5) did **not** occur on OPERATION.
- **Predictions, not confirmed faults:** these OPEN/SHORT outputs are ML predictions against the dataset labels. The electrical confirmation layer is a placeholder, so confirmed electrical faults were **not assessed**. The labels are dataset labels, not measured electrical faults.
- **No relay isolation is implemented**, and nothing is actuated. These results do not show that automatic isolation would be safe.
- The bootstrap interval for OPEN/SHORT recall is 1.000 – 1.000 because no fault reading was missed in any run. With 10 runs per class, this does not prove a zero miss rate on other panels.

## 6. Comparison with Phase 7 (descriptive only)

| | Phase 7 (TRAINING window, model selection) | OPERATION (this evaluation) |
|---|---|---|
| Pooled macro-F1, physics8 + MLP | CV 0.981; leave-one-site-out 0.980 | 0.978 (0.937 – 0.999) |
| OPEN/SHORT recall | 1.000 | 1.000 |
| Main error class | PARTIAL_SHADING (recall about 0.90) | DUST (recall 0.887; coverage-adjusted 0.798) |
| False OPEN on a shading run | yes (`SITE-A-PAR-01`) | none |

The OPERATION macro-F1 (0.978) is close to the Phase 7 cross-validation estimate (0.981), indicating consistent performance across these dataset partitions. However, this does not establish generalization to independently collected field data. The error pattern is different: PARTIAL_SHADING was the weak class in Phase 7 and is error-free here, while DUST is the weak class here. With 10 runs per class, a single unusual run (such as `SITE-C-DUS-06`) can decide which class looks weakest. The comparison is descriptive: Phase 7 numbers are model-selection estimates on a different window and are not a second test.

## 7. Host CPU latency (not ESP32)

Measured on TRAINING rows in Colab (x86_64, 2 CPUs); OPERATION was not used for timing.

| Measurement | Median | Mean | p95 |
|---|---|---|---|
| NumPy forward pass, one reading | 20.8 µs | 21.6 µs | 26.2 µs |
| Batched pipeline (3,000 readings in 25.1 ms) | 8.4 µs per reading | – | – |
| pandas reference pipeline, one reading | 13.6 ms | 14.2 ms | 16.9 ms |

The per-reading pandas time is dominated by pandas overhead and does not represent an embedded implementation. **None of these numbers estimates ESP32 flash, RAM or latency**, which must be measured on the device.

## 8. Dataset evaluation vs real-world validation

- These results describe the model on **`pv_fault_dataset.csv` only**. They are **not** real-world, field or hardware validation.
- **Dataset provenance is UNRESOLVED** (`DATA_PROVENANCE.md`). The dataset is the PV sensor dataset provided by Member 1; the group data contracts describe its origin and must be reconciled with Member 1, and firmware, raw logs and run records have not yet been provided. Until then, no result is described as validated on real laboratory or field data.
- Scores near 1.0 point to classes that are easy to separate in this dataset. Real panels, sensors and weather may be harder.
- Not done: ESP32 deployment, INT8 build, on-device measurement, electrical fault confirmation, relay isolation, federated learning integration (`TRAINING_EXPORT.md` §7).

## 9. Limitations

- One temporal split, 10 runs per class (9 classifiable DUST runs): the 95% intervals (macro-F1 0.937 – 0.999) are wide, and single runs have a large effect.
- The low-light and saturation thresholds are placeholders. Their effect is reported as coverage and was not tuned.
- Prior OPERATION exposure (gate counts, the stand-in prediction file, Member 3's FL validation) is disclosed in `OPERATION_EVAL.md` §6. None of it touched this model.
- The OPERATION window has now been used. Any change to the model, features or thresholds makes a new model version without an independent OPERATION result; results for it on this window are post-hoc.

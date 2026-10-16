# Independent OPERATION evaluation (final test)

Status `FINAL_TEST_OPERATION`, protocol `operation-eval-v1`, window `OPERATION`, created 2026-10-16T09:29:54Z.

Model `c2-local-v1`, weights SHA-256 `94e7ab9384a1906ec2cf0d3176b435e700f15cd99552fbc6c71519c648ecc3a7`, trained at `9a641d5`; evaluation code `a2a85d6`. Dataset SHA-256 `5fbb93471c056c274cbf81e8c317934e379b2484013147e4c2cacfa93421595b`, split `runwise-v1`. Environment {'python': '3.13.16', 'numpy': '2.1.3', 'pandas': '2.2.3', 'sklearn': '1.6.1'}.

Frozen model: no refit, no retraining, no threshold change. Inference uses the model contract only (NumPy); `label` and `health_score` are removed before inference.

## Coverage (quality gates)

3000 readings in 50 runs; 2940 classified, 60 skipped {'LOW_LIGHT': 60}.

| Class | Readings | Classified | Coverage | Correct | Coverage-adjusted recall |
|---|---|---|---|---|---|
| NORMAL | 600 | 600 | 1.000 | 600 | 1.000 |
| PARTIAL_SHADING | 600 | 600 | 1.000 | 600 | 1.000 |
| DUST | 600 | 540 | 0.900 | 479 | 0.798 |
| OPEN_CIRCUIT | 600 | 600 | 1.000 | 600 | 1.000 |
| SHORT_CIRCUIT | 600 | 600 | 1.000 | 600 | 1.000 |

Runs not fully classified: SITE-D-DUS-12 (0/60).

## Classification (classified readings)

Accuracy 0.979, macro-F1 0.978, balanced accuracy 0.977, n = 2940.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| NORMAL | 0.910 | 1.000 | 0.953 | 600 |
| PARTIAL_SHADING | 0.997 | 1.000 | 0.998 | 600 |
| DUST | 1.000 | 0.887 | 0.940 | 540 |
| OPEN_CIRCUIT | 1.000 | 1.000 | 1.000 | 600 |
| SHORT_CIRCUIT | 1.000 | 1.000 | 1.000 | 600 |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | NOR | PAR | DUS | OPE | SHO |
|---|---|---|---|---|---|
| NOR | 600 | 0 | 0 | 0 | 0 |
| PAR | 0 | 600 | 0 | 0 | 0 |
| DUS | 59 | 2 | 479 | 0 | 0 |
| OPE | 0 | 0 | 0 | 600 | 0 |
| SHO | 0 | 0 | 0 | 0 | 600 |

Run level: majority vote 0.980 over 49 runs; worst run `SITE-C-DUS-06` (0.083).

95% intervals (run-level bootstrap, stratified by true class, percentile interval, 1000 resamples, seed 42): accuracy [0.941, 0.999]; macro_f1 [0.937, 0.999]; OPEN_CIRCUIT_recall [1.000, 1.000]; SHORT_CIRCUIT_recall [1.000, 1.000].

Per site: SITE-A 1.000 (n=480), SITE-B 1.000 (n=480), SITE-C 0.917 (n=660), SITE-D 0.989 (n=360), SITE-E 0.997 (n=780), SITE-F 1.000 (n=180).

## OPEN_CIRCUIT and SHORT_CIRCUIT safety analysis

Every OPEN/SHORT output is an **ML prediction, not electrically confirmed** (confirmation rules are placeholders). Labels are dataset labels. No relay isolation is implemented.

### OPEN_CIRCUIT

- False OPEN_CIRCUIT predictions: 0 (0.00 per 1,000 classified non-OPEN_CIRCUIT readings); by true class none.
- Missed OPEN_CIRCUIT faults: 0 of 600 classified fault readings; 0 skipped; recall 1.000, coverage-adjusted recall 1.000.
- Electrical confirmation: NOT_ASSESSED: confirmation rules are placeholders (PHASE7_RESULTS.md decision 3); predictions are unconfirmed.

| Fault run | Site | Classified | Correct | Missed (as) | Skipped | First detection (s) |
|---|---|---|---|---|---|---|
| SITE-B-OPE-18 | SITE-B | 60 | 60 | 0  | 0 | 0 |
| SITE-C-OPE-02 | SITE-C | 60 | 60 | 0  | 0 | 0 |
| SITE-C-OPE-06 | SITE-C | 60 | 60 | 0  | 0 | 0 |
| SITE-C-OPE-08 | SITE-C | 60 | 60 | 0  | 0 | 0 |
| SITE-C-OPE-10 | SITE-C | 60 | 60 | 0  | 0 | 0 |
| SITE-C-OPE-20 | SITE-C | 60 | 60 | 0  | 0 | 0 |
| SITE-E-OPE-12 | SITE-E | 60 | 60 | 0  | 0 | 0 |
| SITE-E-OPE-14 | SITE-E | 60 | 60 | 0  | 0 | 0 |
| SITE-F-OPE-04 | SITE-F | 60 | 60 | 0  | 0 | 0 |
| SITE-F-OPE-16 | SITE-F | 60 | 60 | 0  | 0 | 0 |

### SHORT_CIRCUIT

- False SHORT_CIRCUIT predictions: 0 (0.00 per 1,000 classified non-SHORT_CIRCUIT readings); by true class none.
- Missed SHORT_CIRCUIT faults: 0 of 600 classified fault readings; 0 skipped; recall 1.000, coverage-adjusted recall 1.000.
- Electrical confirmation: NOT_ASSESSED: confirmation rules are placeholders (PHASE7_RESULTS.md decision 3); predictions are unconfirmed.

| Fault run | Site | Classified | Correct | Missed (as) | Skipped | First detection (s) |
|---|---|---|---|---|---|---|
| SITE-A-SHO-14 | SITE-A | 60 | 60 | 0  | 0 | 0 |
| SITE-B-SHO-16 | SITE-B | 60 | 60 | 0  | 0 | 0 |
| SITE-C-SHO-04 | SITE-C | 60 | 60 | 0  | 0 | 0 |
| SITE-C-SHO-08 | SITE-C | 60 | 60 | 0  | 0 | 0 |
| SITE-C-SHO-18 | SITE-C | 60 | 60 | 0  | 0 | 0 |
| SITE-E-SHO-02 | SITE-E | 60 | 60 | 0  | 0 | 0 |
| SITE-E-SHO-06 | SITE-E | 60 | 60 | 0  | 0 | 0 |
| SITE-E-SHO-10 | SITE-E | 60 | 60 | 0  | 0 | 0 |
| SITE-E-SHO-12 | SITE-E | 60 | 60 | 0  | 0 | 0 |
| SITE-E-SHO-20 | SITE-E | 60 | 60 | 0  | 0 | 0 |

## Host CPU latency (not ESP32)

Python/NumPy on the evaluation host. ESP32 flash, RAM and latency must be measured on the device and are not estimated from these numbers.
Rows: TRAINING rows (labels and health_score removed); OPERATION is not used for timing. Host: {'machine': 'x86_64', 'processor': 'x86_64', 'cpu_count': 2, 'platform': 'Linux-6.6.122+-x86_64-with-glibc2.39'}.

| Measurement | n | mean µs | median µs | p95 µs | p99 µs |
|---|---|---|---|---|---|
| forward_pass_single_reading | 1000 | 21.6 | 20.8 | 26.2 | 35.8 |
| pandas_reference_pipeline_single_reading | 200 | 14180.0 | 13567.4 | 16884.7 | 22894.5 |

Batched pipeline: 3000 readings in 25.1 ms (8.4 µs per reading).

Phase 7 context (TRAINING model selection, not comparable as a test result): {'cv_pooled_macro_f1': 0.981, 'loso_training_pooled_macro_f1': 0.98, 'note': 'TRAINING-window model-selection estimates, shown for context only'}.

## Caveats

- OPEN/SHORT outputs are ML predictions; electrical confirmation is not implemented.
- Relay isolation is not implemented; nothing is actuated.
- Labels are dataset labels; dataset provenance is UNRESOLVED (DATA_PROVENANCE.md).
- Latency is host CPU, not ESP32.
- Low-light and saturation thresholds are placeholders and were not tuned.

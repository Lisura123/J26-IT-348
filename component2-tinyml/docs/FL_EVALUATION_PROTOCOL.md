# FL Evaluation Protocol (pre-registration)

**Status: PROTOCOL ONLY. Nothing in this document has been run.** No development model has
been trained, no FL experiment has been run and no result exists. This file fixes the design
**before** any training, so that no choice can be adjusted to the results.

Branch: `integration/member2-member3-fl`. Written after Integration Tasks 1–5 (commits
`630481c`, `ed4da2e`, `0d88840`, `f217b50`).

---

## 1. Question

Does Member 3's federated learning (FL) give a better fault classifier than the alternatives,
when every model is judged on the **same unseen readings**? Four models are compared:

| Name | Model |
|---|---|
| **M0** | Initial TinyML model: the official Model v1 recipe trained on initial central data |
| **LOCAL-s** | Separately trained solar-site models: M0 trained further on one site's data only (one per site) |
| **FL** | Member 3's combined global model: M0 trained further with Member 3's FL engine over all sites |
| **CENT** | Centralised training baseline: the official recipe on all training data pooled (raw data shared) |

One extra control, **CENT-FT** (M0 trained further on all sites' data pooled, with the same
optimiser and budget as FL), separates "the FL algorithm" from "more data".

## 2. Why the official Model v1 cannot be one of the compared models

Checked from the code and the training record (`models/training_metadata.json`):

- Model v1 (`c2-local-v1`) was trained on **all 2,924 status-OK TRAINING rows** (50 runs,
  `early_stopping: false`, so no internal hold-out). Its FeatureConfig (V_ref and scaler) was
  fitted on the same rows (`fitted_on.fold = final_TRAINING`).
- Every row the FL adapter gives Member 3's engine (2,324 site-training rows) and every row of
  the adapter's development validation (600 rows, cv_fold 0) is one of those 2,924 rows:
  **100 % overlap**.

Any score of Model v1, or of FL started from it, on TRAINING rows is therefore a score on seen
data. Model v1 stays the official model; the development models below are research models only
and never replace it.

## 3. Data

**PV Research Dataset:** `pv_fault_dataset.csv`, SHA-256
`5fbb93471c056c274cbf81e8c317934e379b2484013147e4c2cacfa93421595b`, checked on every load
(`fl_data.load_dataset`). Provenance exactly as already documented, not restated here:

- `docs/DATA_PROVENANCE.md`: provided by Member 1; status **UNRESOLVED** until firmware, raw logs
  and run records are provided.
- `component3-federated-backend/data/shared/DATA_CONTRACT.md` (Member 3's copy of the shared
  contract): "All data is SYNTHETIC. Use it for development and integration, not for final results."

The dataset is read only and never changed.

**Window used: TRAINING only** (50 runs, 3,000 rows, 2,924 status-OK). OPERATION rows are
dropped immediately after loading (`fl_data.training_rows`) and are **never** transformed,
trained on, evaluated on or used for tuning in this protocol (Section 9).

**Sites:** SITE-A … SITE-F are the dataset's `site_id` values. In FL they are **simulated FL
sites**: partitions of the dataset, run in one Python process with replayed network traces.
They are not six physical devices and none of them is our ESP32.

## 4. Separation of training and unseen evaluation runs (fixed before training)

### 4.1 Outer folds

The outer folds are the **committed** run-wise split `results/splits_runwise_v1.json`
(`split_version runwise-v1`, seed 42, made in Phase 4 before any of this work). Each TRAINING run
has one `cv_fold` (0–4); a run's 60 readings are never split. Every fold holds exactly
**2 runs per class** (10 runs).

For fold k:

| Set | Content | Use |
|---|---|---|
| **EVAL_k** | the 10 TRAINING runs with `cv_fold == k` | evaluation only: never fitted, tuned or used to choose anything |
| **POOL_k** | the other 40 TRAINING runs | all fitting |
| **INIT_k** | per site, the earliest ⌊n/2⌋ runs of that site in POOL_k (n = the site's runs in POOL_k), ordered by first timestamp; ties by `run_id` | "initial central data" |
| **FL_k** | per site, the remaining runs of that site in POOL_k | "new data at the sites" |

The INIT/FL rule is fixed here and is not changed after any result is seen.

### 4.2 Pre-flight check of the folds (done 2026-10-20, counting only, TRAINING rows)

| Fold | EVAL runs | INIT runs | FL runs | FL runs per site (A, B, C, D, E, F) | INIT classes (N, PS, D, OC, SC) | NORMAL OK rows in INIT |
|---|---|---|---|---|---|---|
| 0 | 10 | 18 | 22 | 5, 2, 3, 4, 7, 1 | 2, 3, 5, 5, 3 | 120 |
| 1 | 10 | 18 | 22 | 6, 3, 3, 4, 4, 2 | 3, 3, 4, 4, 4 | 180 |
| 2 | 10 | 19 | 21 | 6, 2, 2, 4, 6, 1 | 3, 3, 4, 5, 4 | 180 |
| 3 | 10 | 18 | 22 | 6, 3, 3, 3, 6, 1 | 2, 4, 4, 4, 4 | 120 |
| 4 | 10 | 19 | 21 | 5, 3, 2, 4, 5, 2 | 4, 3, 5, 3, 4 | 240 |

Result: in every fold **every site has at least one FL run**, INIT contains **all five classes**,
and INIT has enough NORMAL rows to fit V_ref (at least 3 are needed). This check is repeated
automatically before any training (Section 4.3) and its table is stored with the results.

Classes are **not** spread evenly over the sites (non-IID). In the whole TRAINING window, only
SITE-A and SITE-E have all five classes; SITE-F has only DUST and OPEN_CIRCUIT (3 runs). This is
part of the experiment, not something to correct.

### 4.3 Empty-site and invalid-fold rule (no silent change)

Before any model of fold k is trained, a pre-flight step checks:

1. every site has at least one FL_k run with at least one status-OK reading;
2. INIT_k contains all five classes and at least 3 status-OK NORMAL rows;
3. EVAL_k, INIT_k and FL_k are disjoint and together equal the 50 TRAINING runs.

If a check fails, the fold is **not repaired**. Runs are never moved between EVAL, INIT and FL,
the rule in 4.1 is never changed, and no other fold is put in its place. Instead:

- a site without FL_k data is recorded as `NOT_PARTICIPATING` for fold k. It takes part in no
  round, gets no LOCAL model and is reported as such (Member 3's engine currently expects all six
  sites; supporting a non-participating site is a change to Member 3's code and is
  **PENDING SAMUDI REVIEW**; until then such a fold stops at pre-flight with status
  `BLOCKED_EMPTY_SITE`);
- a fold failing check 2 or 3 is reported as `INVALID_FOLD` with the reason, and all summary
  results state how many folds were used.

With the committed split and the rule in 4.1, no fold fails (Section 4.2).

## 5. Leakage prevention

| Setting | Fitted on | Never on |
|---|---|---|
| V_ref coefficients and z-score scaler (`features.fit_pipeline`) | INIT_k only, then **frozen** for M0, LOCAL, FL, CENT-FT and CENT (primary analysis) | EVAL_k, FL_k |
| Quality-gate thresholds (LOW_LIGHT, LUX_SATURATED, INVALID_INPUT) | not fitted: fixed constants of the frozen FeatureConfig | — |
| Model weights | as listed in Section 6 | EVAL_k |
| INT8 calibration (if INT8 is evaluated) | status-OK INIT_k rows (int8-v1 rule) | EVAL_k |
| Hyperparameters | fixed in Section 6 before any run; if tuning is ever needed, only on an inner split of POOL_k | EVAL_k |
| Number of FL rounds | fixed: 20 | chosen from any curve |

Consequences:

- **Same feature-processing rules for every model of a fold.** All models of fold k use the same
  FeatureConfig_k, so the same EVAL_k readings pass the gates and every model sees identical
  inputs. Only the weights differ.
- A **sensitivity analysis** (reported separately, never replacing the primary one) refits the
  FeatureConfig on POOL_k for CENT, because a real central system would fit its scaler on all its
  data.
- Results are written outside `models/` and `results/` (proposed folder
  `component2-tinyml/fl_dev_runs/`, git-ignored, create-only). Model v1 and its artifacts are
  never written.

## 6. Models and fixed settings

| Model | Starting point | Data | Training procedure |
|---|---|---|---|
| M0_k | — | INIT_k (status-OK) | the official Model v1 recipe (`train.py`: MLP 8-16-5, tanh, Adam lr 0.001, α 1e-4, max_iter 2000, early stopping off) |
| LOCAL_k,s | M0_k | FL_k of site s only | Member 3's `mlp.train_local` (SGD, lr 0.05, batch 32), 20 × 3 = 60 epochs: the same local work an FL site does |
| FL_k | M0_k | FL_k of all sites | Member 3's engine `run_fl(activation="tanh", data=…, init_weights=M0_k, save=False)`, 20 rounds, its existing baseline configuration (int8 compression, network trace replay), unchanged Delta W and FedAvg |
| CENT-FT_k | M0_k | FL_k of all sites pooled | `mlp.train_local`, 60 epochs, same lr and batch |
| CENT_k | — | INIT_k ∪ FL_k pooled | the official Model v1 recipe |

- **Seeds:** 0, 1 and 2 for every model, fixed now. No seed is chosen by score.
- Every TRAINING run is in EVAL exactly once per seed (five folds).
- **Engine dependency:** the `run_fl(data=..., init_weights=...)` change in
  `component3-federated-backend/src/engine.py` exists only on this integration branch and is
  **PENDING SAMUDI REVIEW**. The FL arm cannot run until Member 3 agrees.

## 7. Measurements

All on EVAL_k status-OK readings, after the **final, fixed round**. Every model of a fold is
scored on exactly the same readings.

### 7.1 Classification

- **Accuracy** and **macro-F1** (five classes).
- Per-class recall, precision and F1; confusion matrix.
- **Run-level accuracy:** majority vote of each EVAL run's readings, because 60 readings of one
  run are not independent.

### 7.2 OPEN/SHORT fault detection

- Recall and precision of OPEN_CIRCUIT and SHORT_CIRCUIT. Missed OPEN/SHORT readings and false
  OPEN/SHORT alarms are counted separately.
- Prediction changes involving OPEN/SHORT between models (`quantize.compare_predictions`
  safety counts), in both directions.

### 7.3 Confidence threshold (0.90, as `pv_confirm.c`)

- Share of readings with confidence ≥ 0.90.
- Readings crossing 0.90 compared with M0 (`fl_deploy.confidence_crossings`: fell below, rose above).
- A model with more readings below 0.90 confirms faults later on the device, even if its labels are right.

### 7.4 Communication (FL only)

Taken from Member 3's engine records:

- bytes attempted and received, payload per update;
- updates received / late / failed per round;
- LoRa airtime and radio energy.

These come from **replayed network traces in a simulation**, not radio measurements.

### 7.5 Per-round records (monitoring only)

- local loss before/after and ‖Delta W‖ per site;
- aggregation weights;
- drift ‖w − w(M0)‖ and label-change rate against M0 on POOL_k;
- inner POOL_k validation loss.

EVAL_k is scored **only** at the fixed final round. A curve may be shown but is never used to
choose a round.

### 7.6 Site differences

- LOCAL_k,s scores per class on EVAL_k, including the classes site s never saw.
- Per-site EVAL scores where EVAL_k contains that site's runs (sparse: reported as counts, not
  rates, when fewer than 2 runs).

Accuracy (7.1–7.3), communication (7.4) and site differences (7.6) are reported in separate
tables and never combined into one score.

## 8. Analysis and what counts as evidence

- **Primary comparison:** FL − M0 in macro-F1, pooled out-of-fold over five folds × three seeds.
  The uncertainty is a 95 % run-level paired bootstrap interval: whole EVAL runs are resampled,
  2,000 resamples, seed 0.
- **Secondary comparisons:** FL vs the mean of LOCAL-s; FL vs CENT; FL vs CENT-FT; OPEN/SHORT
  recall; readings below 0.90.
- Fold-to-fold and seed-to-seed spread is reported next to every mean.

**Evidence that FL helps** (all must hold):

1. The FL − M0 macro-F1 interval is above zero.
2. FL is no worse than the LOCAL-s average, and better on classes a site does not have.
3. OPEN/SHORT recall does not decrease and false OPEN/SHORT alarms do not increase.
4. The share of readings below 0.90 does not increase.

FL close to CENT means FL reaches central performance without sharing raw readings, at the
communication cost of 7.4.

**Not evidence:**

- scores on rows a model was trained on (including any score of Model v1 on TRAINING rows);
- one fold or one seed;
- a round chosen from a curve;
- differences inside the fold-to-fold spread;
- any OPERATION number produced during development.

With 2 runs per class per fold, each EVAL fold is small; only the pooled five-fold result is
interpreted.

## 9. OPERATION stays sealed

The OPERATION window (50 runs) was used **once**, for the official Model v1 evaluation
(`docs/OPERATION_RESULTS.md`). This protocol does not read, transform, evaluate or tune on it.

Any later use (for example a single final comparison of Model v1 with an FL model) needs its own
written pre-registration and supervisor approval, and would still be one comparison on the same
PV Research Dataset.

## 10. Limitations

- **Data provenance:** synthetic according to Member 3's contract; provenance UNRESOLVED according
  to Member 2's record. Results are results on this dataset, not field performance.
- **Simulated sites:** six partitions of one dataset in one process; the network is replayed from
  traces. No device trains, and our single physical ESP32 is not involved.
- **Small, unbalanced sites:** SITE-F has 3 runs and 2 classes; fold sizes are 2 runs per class.
  Run-level effects dominate.
- **Time split inside an 8-day window:** "new site data" is simulated by the per-site time order,
  not by data collected after deployment.
- **Optimiser difference:** M0 and CENT use Adam (official recipe); LOCAL, FL and CENT-FT use
  Member 3's SGD. CENT-FT is included to separate this from the FL effect.
- **No hardware measurements:** flash, RAM, latency and energy on the ESP32 are outside this
  protocol.

## 11. Status of each part

| Part | Status |
|---|---|
| This protocol | WRITTEN; awaiting approval |
| Pre-flight fold check (Section 4.2) | DONE (counting only) |
| `run_fl(data, init_weights)` in Member 3's `engine.py` | IMPLEMENTED on the integration branch, **PENDING SAMUDI REVIEW** |
| Non-participating site support in the engine | NOT IMPLEMENTED, **PENDING SAMUDI REVIEW** (not needed with the committed split) |
| Development training and evaluation code (proposed `src/tinyml/fl_dev.py`) | NOT IMPLEMENTED |
| Development models, FL runs, results | NOT RUN |
| OPERATION | SEALED |

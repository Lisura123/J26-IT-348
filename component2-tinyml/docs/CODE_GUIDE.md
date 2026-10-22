# Code Navigation Guide (Member 2, Model v1 `c2-local-v1`)

**Purpose:** answer "show me the exact line" questions quickly during a research evaluation.
Each row below gives the question, the file, the exact line, what to highlight, a short
explanation and the result file that proves it.

**How to jump to a line in VS Code:** press `Cmd+P` (Windows: `Ctrl+P`) and type
`train.py:98`, then Enter. To see every explained location, search the whole folder
(`Cmd+Shift+F`) for **`[GUIDE]`**: each important function has a `# [GUIDE]` comment block
(WHAT / WHY / HOW / INPUT / OUTPUT / EVIDENCE) directly above it.

**Paths** are relative to `component2-tinyml/`. **Line numbers** are for the current
branch `component2-tinyml` with the `[GUIDE]` comments added. If the code changes, search
for the function name instead.

**Scope:** this guide covers the official Model v1 pipeline (8-16-5 tanh MLP, 229
parameters, physics8 features, `pv_fault_dataset.csv`). Model v2/v3 experiments are not
part of this branch.

---

## 1. The pipeline in one picture

```
pv_fault_dataset.csv (6,000 readings, 100 runs)
  │  step 1  scripts/01_validate_dataset.py  → src/tinyml/validate.py     → results/dataset_validation.md
  │  step 2  scripts/02_make_splits.py       → src/tinyml/splits.py       → results/splits_runwise_v1.json, split_summary.md
  │          TRAINING window: 3,000 rows / 50 runs      OPERATION window: 3,000 rows / 50 runs
  │  step 3  scripts/03_validate_features.py → src/tinyml/features.py     → results/feature_validation.md
  │  step 4  scripts/04_compare_baselines.py → models.py + experiment.py  → results/baselines_phase7.md  (5-fold CV, TRAINING only)
  │  step 5  scripts/05_train_final.py       → train.py + export.py       → models/model_contract.json   (MLP trained, TRAINING only)
  │  step 6  scripts/06_evaluate_operation.py→ holdout.py + evaluate.py   → results/operation_eval.md    (final test, OPERATION, run once)
  │  step 7  scripts/07_quantize_model.py    → quantize.py + codegen.py   → models/int8/, firmware/pv_fault/generated/*.h
  └  step 8  scripts/08_pruning_study.py     → prune.py                   → results/pruning_study.md     (evaluated, not adopted)
```

---

## 2. Navigation table (35 questions)

### A. Dataset and validation

| # | Panel Question | Source File | Function / Exact Line | What to Highlight | Simple English Explanation | Evidence File |
|---|---|---|---|---|---|---|
| 1 | Where did your dataset come from? | `docs/DATA_PROVENANCE.md` | §1 "Source and ownership" (line 11); `src/tinyml/train.py:37` `DATASET_SHA256` | the source row and the SHA-256 | The PV sensor dataset was provided by Member 1. Its SHA-256 is fixed in code, so training stops if a different file is used. Provenance is still marked **UNRESOLVED** in the doc (how the faults were physically produced is not documented). | `docs/DATA_PROVENANCE.md`, `results/dataset_validation.md` |
| 2 | Where do you validate the dataset? | `src/tinyml/validate.py` | `validate_dataset()` line 382; called by `scripts/01_validate_dataset.py:71` | the list of `check_*` calls and the errors/warnings sorting | One function runs every check. Errors mean "do not use the data"; warnings mean "usable, but report it". | `results/dataset_validation.md` line 18: **PASS (with warnings)**, 0 errors, 4 warnings |
| 3 | Where do you check missing values? | `src/tinyml/validate.py` | `check_missing()` line 128 | `df.isna().sum()` and the DS18B20 error codes −127/−999 | Counts empty values per column and sensor error codes. Any NaN is an error. | `results/dataset_validation.md` (0 errors) |
| 4 | Where do you check duplicates? | `src/tinyml/validate.py` | `check_duplicates()` line 142 | `record_id` duplicates and `(device_id, timestamp_utc)` duplicates | Duplicate readings could end up in both train and test and inflate the score. | `results/dataset_validation.md` |
| 5 | Where do you check the fault labels? | `src/tinyml/validate.py` | `check_labels()` line 116; label names in `src/tinyml/contract.py:11` | `unknown` and `absent` lists | All five classes must exist with exactly the names used on the ESP32 and in FL. | `results/dataset_validation.md` "Classes and runs" (1,200 rows, 20 runs per class) |
| 6 | Where do you check invalid sensor values / physical ranges? | `src/tinyml/validate.py` | limits `PHYSICAL_LIMITS` line 27; `check_ranges()` line 221; `check_power()` line 240 | `below` / `above` counts; P = V × I check | Values outside what the panel and sensors can produce are reported, never changed. The limits come from an **assumed** datasheet. | `results/dataset_validation.md` (warning: 76 readings at lux saturation) |
| 7 | How do you prevent data leakage? | `src/tinyml/validate.py`; `src/tinyml/contract.py` | `check_leakage()` line 256; `LEAKAGE_COLUMNS` line 33 and `NEVER_MODEL_INPUTS` line 35 in `contract.py`; `splits._checked()` line 112 | `health_score` accuracy; `random_row_split_unsafe`; `raise AssertionError` at `splits.py:115` | `health_score` alone predicts the label with 74.6 % run-wise accuracy, so it is never an input. `run_id`/`site_id` are only used for grouping. No run is allowed on both sides of a split. | `results/dataset_validation.md` "Warnings" (line 22) |

### B. Splitting (the 50/50 question)

| # | Panel Question | Source File | Function / Exact Line | What to Highlight | Simple English Explanation | Evidence File |
|---|---|---|---|---|---|---|
| 8 | Where is your 50/50 division implemented? | `src/tinyml/splits.py` | `make_assignments()` line 52; **lines 77–78** `runs["temporal"] = np.where(...)` | `"TRAINING" → "train"`, `"OPERATION" → "test"` | The split keeps the dataset's own time windows: TRAINING (1–9 Sept) is the training side, OPERATION (10–15 Sept) is the test side. | `results/split_summary.md` "Protocol 2" (line 19) |
| 9 | Where does the 50 % actually come from? | `src/tinyml/splits.py` | comment above `make_assignments()` (line 52) | "there is NO `train_test_split(test_size=0.5)`" | The dataset already contains 3,000 TRAINING and 3,000 OPERATION readings (50 runs each). Keeping the windows gives 50/50; the ratio is a property of the data, not a tuning choice. | `results/split_summary.md`: temporal 3000 train / 3000 test rows |
| 10 | How do you verify the counts? | `src/tinyml/holdout.py`; `src/tinyml/train.py` | `EXPECTED_ROWS = 3000` line 51, `EXPECTED_RUNS = 50` line 52 (checked inside `run_evaluation()`); `final_training_rows()` line 49 | the row/run checks | Before the final test the code checks 3,000 rows in 50 runs. Training refuses rows that are not exactly the "train" side of the split file. | `results/split_summary.md`, `results/operation_eval.md` "3000 readings in 50 runs" |
| 11 | Is 50/50 the best split? | — | — | — | No such claim is made. 50/50 is simply how the provided dataset was delivered (two time windows). Model selection did not depend on it: it used 5-fold CV inside the TRAINING half. | `docs/ARCHITECTURE.md` §8.1, `results/split_summary.md` |
| 12 | Why not a random row split? | `src/tinyml/validate.py` | `check_leakage()` line 256 (`random_row_split_unsafe`) | comment block above `StratifiedKFold` in `splits.py:72` | Each run has 60 readings taken 5 s apart, so neighbours are nearly identical. A random row split would test the model on near-copies of its training data and give a falsely high score. | `results/dataset_validation.md` |
| 13 | Why did you use cross-validation? | `src/tinyml/splits.py` | **line 72** `skf = StratifiedKFold(n_splits=n_splits, ...)` | "folds over RUNS, stratified by label" | Five folds give five estimates instead of one, so models can be compared fairly without touching the OPERATION test data. Each fold tests 2 runs per class (600 readings). | `results/split_summary.md` "Protocol 1" (cv0–cv4: 2400 / 600) |
| 14 | Where are train/validation folds produced for training code? | `src/tinyml/splits.py` | `iter_folds()` line 86 | `"cv"`, `"temporal"`, `"loso"` branches | Converts the saved fold file into row indices; every split goes through the leakage guard `_checked()` (line 112). | `results/splits_runwise_v1.json` |

### C. Features

| # | Panel Question | Source File | Function / Exact Line | What to Highlight | Simple English Explanation | Evidence File |
|---|---|---|---|---|---|---|
| 15 | Where are your eight features calculated? | `src/tinyml/features.py` | order `FEATURES` line 35; `physical_features()`, **lines 173–181** | the `[GUIDE] C` comment block listing all eight | v_deficit, i_norm, p_norm, dust_vo_v, temp_spread, temp_rise, voltage_v, current_ma: each one is a physical signature of a fault (see table in section 4). | `docs/FEATURES.md` §2 (line 20), `results/feature_validation.md` |
| 16 | How is the voltage reference (V_ref) learned? | `src/tinyml/features.py` | `fit_pipeline()` line 195; **line 211** `np.linalg.lstsq(...)` | NORMAL, status-OK **training** rows only | A healthy-panel voltage is fitted from light and temperature; `v_deficit` is how far the measured voltage is below it. | `results/feature_validation.md` |
| 17 | Where is the scaler fitted, and on which data? | `src/tinyml/features.py` | **line 215** `mean = feats.mean()` (inside `fit_pipeline`) | "training rows only" in the `[GUIDE] C` block above `fit_pipeline` | Mean and std come only from training rows. The test side reuses them via `transform()` (line 227, scaling at line 235), so test information never leaks into training. | `models/feature_config.json` (`scaler`, `fitted_on: final_TRAINING`) |
| 18 | What happens to dark or saturated readings? | `src/tinyml/features.py` | `input_status()` line 112; thresholds lines 63–64 | OK / LOW_LIGHT / LUX_SATURATED / INVALID_INPUT | Readings the model cannot interpret are skipped and counted, never guessed. 2,400 lx and 100,000 lx are **placeholders**. | `results/operation_eval.md` "Coverage" (60 LOW_LIGHT skipped) |

### D. Algorithm selection

| # | Panel Question | Source File | Function / Exact Line | What to Highlight | Simple English Explanation | Evidence File |
|---|---|---|---|---|---|---|
| 19 | Which algorithms did you compare and where are they created? | `src/tinyml/models.py` | `make_model()` line 96: LogisticRegression line 100, DecisionTree line 102, RandomForest line 104, **MLP lines 109–110**; `ThresholdRules` class line 29 | the five `return` lines | Rules ("is ML needed?"), logistic regression, decision tree, random forest and the 8-16-5 MLP, all with the same `fit`/`predict` interface. | `results/baselines_phase7.md` "## cv" (line 7) |
| 20 | Where is the comparison run? | `src/tinyml/experiment.py` | `run_comparison()` line 70; **lines 102–104** `make_model` / `model.fit(X_tr, y_tr)` / `model.predict(X_te)` | fit on train fold, predict held-out runs | 286 fits: every model × feature set × fold (5-fold CV and leave-one-site-out), TRAINING window only. Called by `scripts/04_compare_baselines.py:123`. | `results/baselines_phase7.json` (286 records) |
| 21 | Why did you choose the MLP? | `docs/PHASE7_RESULTS.md` | "Decisions" (line 53), decision 2; `[GUIDE] D` block above `make_model()` | MLP 0.981 vs logistic regression 0.982 | Not because it is more accurate (it is within 0.001 of logistic regression). It is chosen because its macro-F1 is competitive, it is compact (229 parameters for the ESP32), and its 8-16-5 layout matches Member 3's federated learning model. | `results/baselines_phase7.md` lines 18–22 |
| 22 | Do the physics features help? | `docs/PHASE7_RESULTS.md` | Finding 1 | physics8 vs raw8 | MLP macro-F1 0.981 (physics8) vs 0.958 (raw8) in CV; the gain is larger on unseen sites. `v_deficit` alone shows **no meaningful benefit** for the MLP (Finding 3). | `results/baselines_phase7.md` |

### E. MLP training

| # | Panel Question | Source File | Function / Exact Line | What to Highlight | Simple English Explanation | Evidence File |
|---|---|---|---|---|---|---|
| 23 | Where is your MLP defined? | `src/tinyml/models.py` | **lines 109–110** `MLPClassifier(hidden_layer_sizes=(16,), activation="tanh", solver="adam", max_iter=2000, random_state=seed)` | `(16,)` and `"tanh"` | 8 inputs → 16 hidden neurons (tanh) → 5 outputs (softmax) = 229 learned numbers. | `models/model_contract.json`, `models/training_metadata.json` |
| 24 | Where are the eight inputs prepared for training? | `src/tinyml/train.py` | **line 82** `cfg = fit_pipeline(train, ...)`; **line 84** `X, status = transform(train, cfg)` | X = scaled features | Learn V_ref and the scaler on TRAINING rows, then build the 8-column input matrix X. | `models/feature_config.json` |
| 25 | Where are the correct labels assigned? | `src/tinyml/train.py` | **line 88** `y = train.loc[keep, LABEL_COLUMN].to_numpy()` | same rows as X | y holds the dataset's `label` for exactly the rows in X, in the same order. | `models/training_metadata.json` (`class_counts`) |
| 26 | Which exact line trains your model? | `src/tinyml/train.py` | **line 98** `model.fit(X, y)` (model created on line 93; called from `scripts/05_train_final.py:50`) | the `[GUIDE] E. Step 5` comment | X = 8 PV features, y = correct labels. `fit` starts from random weights and adjusts all 229 weights (adam optimiser) until predictions match y. | `models/training_metadata.json`: 2,924 rows, 142 iterations, converged |
| 27 | Where does the model predict? Training vs inference? | `src/tinyml/train.py`; `src/tinyml/export.py` | **line 104** `fit_pred = model.predict(X)`; `forward()` line 137; `predict_raw()` line 314 | "fit check, NOT performance" | Training changes the weights; inference only uses them. Line 104 predicts the training rows (accuracy 1.0, a fit check only). Real inference on new data uses `predict_raw()`. | `models/training_metadata.json` (`training_fit_accuracy: 1.0`, with its note) |
| 28 | Where are the 229 parameters saved? | `src/tinyml/export.py` | layout `PARAM_LAYOUT` line 43, `N_PARAMS` line 44; `extract_weights()` line 81 (class reorder at line 93); `flatten()` line 111; `write_artifacts()` line 226 | W1 8×16 + b1 16 + W2 16×5 + b2 5 = 229 | Weights are read out of scikit-learn, put in the fixed order used by Member 3's FL, and saved with SHA-256 hashes. Called from `scripts/05_train_final.py:59`. | `models/model_contract.json`, `models/manifest.json`, `docs/TRAINING_EXPORT.md` §5 (line 49) |

### F. Evaluation

| # | Panel Question | Source File | Function / Exact Line | What to Highlight | Simple English Explanation | Evidence File |
|---|---|---|---|---|---|---|
| 29 | Where do you calculate accuracy and F1-score? | `src/tinyml/evaluate.py` | `classification_metrics()` line 26: precision/recall **line 35**, macro-F1 **line 45**, accuracy **line 50** | the three lines | One function computes every metric for every experiment, so CV and final-test numbers are calculated the same way. | `results/operation_eval.md` line 25 |
| 30 | Where is the confusion matrix? | `src/tinyml/evaluate.py` | **line 55** `"confusion_matrix": confusion_matrix(...)` | rows = true, columns = predicted | Shows exactly which classes get confused. In the final test the only errors are DUST readings (59 predicted NORMAL, 2 predicted PARTIAL_SHADING). | `results/operation_eval.md` line 35 (table) |
| 31 | Where is the final test run, and why only once? | `src/tinyml/holdout.py` | `run_evaluation()` line 345; refusal `raise PermissionError` line 355; inference **line 369**; metrics **line 375** | `confirm_final_test` guard | The frozen model predicts the unseen OPERATION half once. Nothing was tuned on it, so it is an honest estimate. Labels and `health_score` are removed before inference (`inference_frame()` line 106). | `results/operation_eval.md`: accuracy 0.979, macro-F1 0.978, OPEN/SHORT recall 1.000 |
| 32 | What is the difference between your three kinds of results? | `src/tinyml/evaluate.py` | `[GUIDE] F` block above `classification_metrics()` | 1 = CV, 2 = training fit, 3 = final test | (1) 5-fold CV on TRAINING: model selection (MLP 0.981). (2) Training-fit accuracy: 1.0, not a performance claim. (3) Final OPERATION test: 0.978 macro-F1, the reported result. | `results/baselines_phase7.md`, `models/training_metadata.json`, `results/operation_eval.md` |

### G–H. Quantization, pruning, ESP32, FL

| # | Panel Question | Source File | Function / Exact Line | What to Highlight | Simple English Explanation | Evidence File |
|---|---|---|---|---|---|---|
| 33 | How is the model prepared for the ESP32 (INT8)? | `src/tinyml/quantize.py`; `src/tinyml/codegen.py` | `quantize_model()` line 200; `build_tanh_lut()` line 186; `int8_core()` line 253; `write_headers()` (C export) line 165 | int8 weights, scales, tanh lookup table | Float weights become 8-bit integers with fixed scales from TRAINING rows; C header files are generated from the saved model. | `results/quantization_report.md` line 19: agreement 1.0000 on 2,924 TRAINING rows, 0 changed predictions; `firmware/pv_fault/generated/pv_model_int8.h` |
| 34 | Did you try pruning? | `src/tinyml/prune.py` | `unstructured_prune()` line 49; `structured_prune()` line 76 | "evaluated, not adopted" | Removing 10–30 % of the weights did not change any prediction on TRAINING rows, but there is no measured ESP32 benefit (estimated saving 36–85 bytes), so the deployed model is unchanged. | `results/pruning_study.md`, `docs/PRUNING.md` §9.4 (line 222) |
| 35 | What is still pending? | `docs/ESP32_INTEGRATION.md`; `docs/FL_INTEGRATION.md`; `src/tinyml/fl_adapter.py` | checklist H3–H11 (line 144 onward); `compute_delta()` line 95, `build_update()` line 115 | `NOT_EXECUTED` / `BLOCKED_BY_HARDWARE` rows | **Pending in the repository:** ESP32 measurements (H3–H7), real sensor inputs (second panel DS18B20, lux and dust calibration), electrical fault confirmation, relay (disabled), and any end-to-end FL round with Member 3's server (adapters are host-tested only). | `docs/ESP32_INTEGRATION.md` §5, `docs/FL_INTEGRATION.md` §1 (line 9) |

---

## 3. Beginner explanations of the key lines

**`model = make_model("mlp", seed=0)`** (`train.py:93`): builds an *empty* neural network
with 8 inputs, 16 hidden neurons and 5 outputs. Its 229 weights start as small random numbers.

**`model.fit(X, y)`** (`train.py:98`):
- **X** = the eight PV input features (one row per reading, already scaled).
- **y** = the correct fault label for each row.
- **fit** = the model repeatedly compares its predictions with y and adjusts its weights
  (adam optimiser) to reduce the error. Here it stopped after 142 iterations.

**`model.predict(X)`** (`train.py:104`): uses the learned weights to output one label per
row. No learning happens here. That is **inference**.

**`fit_pipeline(train)`** (`features.py:195`): learns the scaler mean/std (and V_ref) from
**training rows only**. If test rows were included, the test score would be optimistic
(data leakage).

**Precision / recall / F1** (`evaluate.py:35`): precision = how many alarms were real;
recall = how many real faults were found; F1 combines both; macro-F1 (`evaluate.py:45`)
averages F1 over the five classes equally.

---

## 4. The eight features in one table

| # | Feature | Formula | Why it helps (docs/FEATURES.md §2) |
|---|---|---|---|
| 0 | `v_deficit` | V_ref(lux, T) − V | voltage below a healthy panel: high for SHORT/SHADING |
| 1 | `i_norm` | I / klx | current corrected for light: low for DUST/SHADING, ~0 for OPEN |
| 2 | `p_norm` | P / klx | power per unit light: drops for every fault |
| 3 | `dust_vo_v` | raw | dust sensor voltage: high for DUST (not a calibrated concentration) |
| 4 | `temp_spread` | \|T1 − T2\| | hotspot under PARTIAL_SHADING |
| 5 | `temp_rise` | mean(T1, T2) − T_ambient | heat not converted to power |
| 6 | `voltage_v` | raw | ≈ 0 for SHORT_CIRCUIT |
| 7 | `current_ma` | raw | ≈ 0 for OPEN_CIRCUIT |

---

## 5. Known documentation inconsistencies (reported, not changed)

1. **Stale status lines.** `docs/PHASE7_RESULTS.md` ("The final TRAINING → OPERATION test
   has **not** been run") and `docs/TRAINING_EXPORT.md` ("The OPERATION final test has
   **not** been run") were written before Phase 9. The test **was** run once on 2026-10-16
   (`results/operation_eval.md`, `docs/OPERATION_RESULTS.md`). Say so if asked.
2. **`docs/FEATURES.md` status** says the benefit of the features "has not been evaluated
   yet". It was evaluated later in Phase 7 (`docs/PHASE7_RESULTS.md`, Findings 1 and 3).
3. **ESP32 status.** `docs/ESP32_INTEGRATION.md` lists every on-device step as
   `NOT_EXECUTED` and the board model as unknown. A self-test run of the INT8 model on an
   ESP32-D0WD-V3 with an Arduino sketch was done by Member 2 on 2026-10-22, but its log is
   **not yet committed**, so in the repository it is still unverified. Commit the serial
   log before citing it as evidence.
4. **Hardware vs dataset.** The dataset has two panel temperatures; Member 1's stated
   hardware has one panel probe (`docs/DATA_PROVENANCE.md` §3). With one probe every real
   reading would be `INVALID_INPUT` (`docs/ESP32_INTEGRATION.md` §4).
5. **Quantization and pruning results** are measured on TRAINING rows (the rows the model
   was fitted on): they show agreement with the float model, not independent accuracy.

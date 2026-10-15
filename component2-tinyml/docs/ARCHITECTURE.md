# Component 2 Architecture: PV-Specific On-Device TinyML Fault Detection

**Member 2 · Group J26-IT-348 · Branch `component2-tinyml` · Status: initial design (v1), accepted for development; items marked PROPOSED / PLACEHOLDER / UNRESOLVED are not final**

Based on `Lisura123/J26-IT-348` as of 2026-10-15: `component2-tinyml@6bf0049`, `component1-sensing-grading@c78753e`, `component3-federated-backend@375a9e9`, `feature/project-backend@b853d2e`, `component4-digital-twin-dashboard@6ca5773`.

---

## 1. Scope and design principles

The component classifies every PV sensor reading on the ESP32 into one of five classes and reports the result to the group backend. It also outputs a **proposed** relay recommendation (§7.5); actual relay control is out of scope until Member 1's hardware and protection requirements are confirmed. The model is trained centrally (Colab) and updated through Member 3's federated learning.

Design principles:
1. **One source of truth.** Feature order, scaling, labels and weights live in a single `model_contract.json`. Python, the ESP32 C header, Member 3's FL and Member 4's SHAP are all generated from or validated against that file. No logic is re-typed by hand.
2. **Run-wise evaluation only.** Readings from the same experiment run never appear in both train and test.
3. **ML never actuates hardware by itself.** Any future relay action comes only from deterministic electrical rules that Member 1 validates against the hardware. ML is advisory.
4. **Simplest model that meets integration constraints.** Every model is compared against threshold rules and logistic regression.
5. **No claim without measurement.** Size, RAM and latency claims come from on-device measurements, not estimates.
6. **Data honesty.** The dataset is the PV sensor dataset provided by Member 1. Dataset provenance is **UNRESOLVED**. Until Member 1 provides primary evidence (firmware, raw logs, run records), no result is described as validated on real laboratory data (§12, R1).
7. **Never hide a non-decision.** A reading that is not classified gets an explicit status. It is never silently reported as `NORMAL`.

---

## 2. Evidence this design is based on

| Fact | Evidence |
|---|---|
| Combined dataset: 6,000 rows, 100 runs × 60 readings, 5 s interval, 1,200 rows/class, TRAINING 3,000 / OPERATION 3,000 | `pv_fault_dataset.csv` (SHA-256 `5fbb9347…5595b`; same bytes in the upload and in `component3-…/data/shared`, commit `a1b36f0`) |
| The 5 per-class files are **exact copies** of the combined rows for their class (same rows, values, order) | read-only check of uploaded `pv_<label>.csv` vs combined: 0 missing, 0 mismatched, 0 extra |
| `member2_fault_predictions.csv` is a **stand-in**, not ground truth: OPERATION rows only, 72.07% agreement with labels, identical to the prediction columns of Member 4's `digital_twin_input.csv` (`0cb3777`, yohan317) | computed |
| `health_score` is a label proxy (74.6% run-wise accuracy on its own) | computed |
| Dataset: PV sensor dataset provided by Member 1; group data contracts describe its origin and must be reconciled with Member 1 | `docs/DATA_CONTRACT.md` (`665de11`), `component3-…/data/shared/DATA_CONTRACT.md` |
| No firmware exists on any branch; sensor models and resolution are undocumented | `git ls-tree` over all branches |
| Member 3 FL model: **8-16-5 MLP, tanh by default, 229 parameters hard-coded** (`N_PARAMS = 229`; LoRa payload sized as 68 B header + 229 int8) | `component3-…/src/mlp.py`, `src/compression.py`, `src/engine.py --activation default tanh` |
| Member 4 runs the FL model itself with **ReLU** (hard-coded `np.maximum(0, …)`) and an 8-feature scaler from `fl_global_model_history.json` (see §6.4) | `component4-…/src/twin/tinyml_model.py` |
| Backend prediction stream: `fault_predictions`, MQTT `solar/{site_id}/prediction`, fields `predicted_label`, `confidence`, `prob_*` (PROPOSED, Member 2 to confirm) | `project-backend/schema.yaml`, `handover/member2_tinyml.md` |
| Member 4 expects `relay_status` (`CONNECTED`/`ISOLATED`) and `alert_level` (`NONE`/`INFO`/`WARNING`/`CRITICAL`); in the stand-in, `ISOLATED` only ever occurs with OPEN/SHORT predictions | `digital_twin_input.csv` |

**Key consequence:** Member 3's FL payload and LoRa budget are built around exactly 229 parameters (8 inputs → 16 hidden → 5 outputs). If Component 2 keeps an **8-16-5 shape with 8 inputs**, FL works with **no change to Member 3's code**. Only the meaning of the 8 inputs and the scaler change.

---

## 3. System architecture

```
                         ┌──────────────── ESP32 (per site) ────────────────────────────────┐
 Member 1 sensors        │                                                                   │
 V, I, P, T1, T2, Tamb,  │  [1] Input guard ──► [2] Feature layer ──► [3] Scaler ──► [4] MLP │
 lux, dust, RH, hPa ───► │   ranges, NaN,        8 PV-physics         z=(x-μ)/σ    8-16-5    │
 (one reading / 5 s)     │   sentinels, low-     features (§5)        from contract tanh,    │
                         │   light gate                                             softmax  │
                         │        │                                                   │      │
                         │        │ electrical facts / skip status           probs[5] │      │
                         │        ▼                                                   ▼      │
                         │  [5] Rule guards (OPEN / SHORT) ──► [6] Decision fusion           │
                         │                                 status (CLASSIFIED / SKIPPED_*),  │
                         │                                 label|null, confidence, alert     │
                         │                                            │                      │
                         │                    ┌───────────────────────┼──────────────┐       │
                         │                    ▼                       ▼              │       │
                         │   [7] Relay recommendation         [8] Publisher (MQTT)   │       │
                         │   PROPOSED, advisory only;        solar/{site}/prediction │       │
                         │   no GPIO until Member 1                                  │       │
                         │   confirms hardware/protection                            │       │
                         └───────────────────────────────────────────┬───────────────────────┘
                                                                     ▼
                                                     Backend (fault_predictions)
                                                                     │
                                         ┌───────────────────────────┼────────────────────┐
                                         ▼                           ▼                    ▼
                                Member 4 twin/SHAP          Member 3 FL server    Dashboard alerts
                                                                     │
       Offline (Colab / laptop)                                       │ new global weights
       dataset ─► validate ─► run-wise split ─► features ─► train ─► │ (same 8-16-5 layout)
       ─► evaluate ─► export model_contract.json ─► gen model_params.h ◄──┘ (OTA)
```

---

## 4. Data pipeline (offline)

| Stage | Module | Input → Output | Key rules |
|---|---|---|---|
| Validation | `src/tinyml/validate.py` | raw CSV → validation report | read-only; schema, labels, missing, duplicates, run structure, 5 s sampling, physical ranges, P = V·I, leakage, subset check of per-class files, refuse prediction files |
| Provenance | `docs/DATA_PROVENANCE.md` | evidence → document | file hashes, commits, contradictions, open questions for Member 1 |
| Splitting | `src/tinyml/splits.py` | dataset → fold assignments | `run_id` groups; 3 protocols (§8); seeds fixed; split file saved |
| Features | `src/tinyml/features.py` | sensor columns → 8 features | stateless per reading; any fitted constant (e.g. Voc curve a, b) is fit on **train fold only** |

Datasets are **never committed** (`data/raw/` is git-ignored). The README records SHA-256 hashes so anyone can confirm they use the same files.

---

## 5. PV-specific feature layer (8 inputs, keeps the FL shape)

| # | Feature | Formula | PV meaning |
|---|---|---|---|
| f1 | `v_deficit` | `a + b·ln(klx) + c·(T_panel − 25) − V`, with `klx = lux/1000` | voltage drop vs the fitted healthy-panel voltage at this light and temperature (shading, short) |
| f2 | `i_norm` | `I / klx` (mA per klx) | current per unit of light (dust, shading, open) |
| f3 | `p_norm` | `P / klx` (mW per klx) | performance ratio proxy |
| f4 | `dust_vo_v` | raw | dust sensor output |
| f5 | `temp_spread` | `|T1 − T2|` | hotspot from partial shading |
| f6 | `temp_rise` | `(T1+T2)/2 − Tamb` | thermal load |
| f7 | `voltage_v` | raw | needed for short-circuit detection near 0 V |
| f8 | `current_ma` | raw | needed for open-circuit detection near 0 mA |

- Irradiance is used in lux (kilolux). The group convention `lux/120` (Member 1 `entropy_weighting.py`, Member 4 `inputs.py`) is not applied, because it is not a calibration. Implemented as `pv8-v1` in `src/tinyml/features.py`; full definitions, gating and placeholders are in `docs/FEATURES.md`.
- **Low-light gate:** if `G < G_min` (placeholder 20 W/m², i.e. 2,400 lx; to be tuned on data), no ML inference is run. The reading gets `prediction_status = SKIPPED_LOW_LIGHT` and `predicted_label = null`. It is **not** reported as `NORMAL`, because "no decision" is not "healthy". The ratio features become unstable at very low irradiance, so classifying there is unreliable. Electrical rule guards (§7.4) still run, because a short or open circuit can be detected without irradiance.
- **Lux saturation:** if lux ≥ the sensor maximum, `G` is unknown. f1–f3 are unreliable, so the reading gets `prediction_status = SKIPPED_LUX_SATURATED` unless a rule guard fires. A degraded ML fallback without f1–f3 is a possible later extension.
- **Invalid input** (NaN, DS18B20 sentinels −127/−999, out-of-range values): `prediction_status = SKIPPED_INVALID_INPUT`.
- Humidity and pressure are not used. There is no physical fault mechanism for them, and they risk site/weather shortcuts.
- Final feature selection is confirmed by the Phase 6 ablation (raw-8 vs physics-8). If physics features do not help on run-wise and leave-one-site-out tests, we say so.

---

## 6. Model architecture and selection

### 6.1 Candidates (all evaluated in Phase 6, run-wise)
| Model | Role | FL-compatible with Member 3 as-is | On-device form |
|---|---|---|---|
| Threshold rules | mandatory baseline + safety guards | n/a | C if-else |
| Softmax regression (8→5, 45 params) | baseline | needs Member 3 change (N_PARAMS) | C |
| Decision tree (d ≤ 4) | interpretable baseline | no | C if-else |
| Random forest (≤ 20 trees, d ≤ 6) | upper-bound baseline | no | C (emlearn/m2cgen) |
| **MLP 8-16-5, tanh** | **deployed model** | **yes, unchanged** | hand C float32; TFLM int8 for ablation |
| Windowed MLP (6 × 5 s mean/std) | only if per-reading results are unstable | needs Member 3 change | C + ring buffer |

### 6.2 Recommendation
**Deploy an 8-16-5 MLP (tanh) on the 8 PV-physics features, behind the rule guards.**

Why the MLP is chosen over simpler models, stated honestly:
- On the current data, **accuracy does not justify an MLP**. Rules reach ~97–98%, and logistic regression and the MLP both reach ~97–99% run-wise. That must be stated in the thesis.
- The MLP is chosen because it (a) is the exact model Member 3 federates (229 params, payload already designed), (b) is what Member 4 already explains with SHAP, and (c) can learn non-linear class boundaries that may matter on real-panel data, which is still unproven.
- If real-lab validation shows softmax regression is as good, the thesis reports that. Switching FL to 45 params would then be a joint decision with Member 3.

### 6.3 Exact model definition
```
input  x ∈ R^8                       (feature order = contract.features)
z      = (x − μ) / σ                 (μ, σ from contract.scaler, fitted on train fold)
h      = tanh(W1ᵀ z + b1)            W1: 8×16, b1: 16
logits = W2ᵀ h + b2                  W2: 16×5, b2: 5
p      = softmax(logits)             order = contract.labels
params = 8·16 + 16 + 16·5 + 5 = 229
```
Weight layout `W1 (8,16), b1, W2 (16,5), b2` matches `component3-…/src/mlp.py LAYOUT`.

### 6.4 Known integration issue: activation mismatch (UNRESOLVED)
| Component | Activation used | Evidence |
|---|---|---|
| Member 3 FL training | **tanh** (default `--activation tanh`; reported runs `member3-full-tanh-*`, `member3-baseline-tanh-*`) | `component3-…/src/engine.py`, `src/evaluate.py`, `src/export_results.py`, `models/runs/` |
| Member 4 twin / SHAP | **ReLU**, hard-coded: `h = np.maximum(0, z @ W1 + b1)` | `component4-…/src/twin/tinyml_model.py` |
| Shared weight files | no activation field stored | `fl_global_model_history.json`, `models/global_v*.json` |

**Impact:** weights trained with tanh but evaluated with ReLU give different hidden values, so different predictions and SHAP explanations. Nothing raises an error, so the bug is silent.

**Decision for Component 2:** use **tanh** to match the FL trainer, and add an `"activation"` field to `model_contract.json`. **Requested from the team:** Member 4 reads the activation from the contract instead of hard-coding ReLU, and Member 3 writes it into published models. Until both confirm, this stays open (R3).

---

## 7. Contracts (exact)

### 7.1 Input: one reading from Member 1 (backend `readings` stream)
| Field | Type | Unit | Required |
|---|---|---|---|
| `record_id` | string | — | yes |
| `timestamp_utc` | string `YYYY-MM-DDTHH:MM:SSZ` | UTC | yes |
| `site_id`, `device_id`, `panel_id` | string | `sites.csv` IDs | yes |
| `voltage_v` | float | V | yes |
| `current_ma` | float | mA (≥ 0 after sign fix) | yes |
| `power_mw` | float | mW (= V·mA) | yes |
| `panel_temp1_c`, `panel_temp2_c` | float | °C (−127/−999 = sensor error) | yes |
| `ambient_temp_c` | float | °C | yes |
| `irradiance_lux` | float | lx | yes |
| `dust_vo_v` | float | V | yes |
| `humidity_pct`, `pressure_hpa` | float | %, hPa | no (not used by the model) |

### 7.2 Model contract file: `models/model_contract.json`

> **⚠ SCHEMA EXAMPLE ONLY. NO VALUE BELOW IS A TRAINED PARAMETER.**
> Every number here is a placeholder that shows the format. Real values are produced only by the training pipeline (Phase 7) and carry `"status": "TRAINED"` plus the dataset hash and git commit. Firmware, FL and the twin must refuse a contract whose `status` is not `TRAINED`.

| Field | Kind | Source of the real value |
|---|---|---|
| `labels`, `features`, `activation` | **fixed by design** (this document) | §5, §6 |
| `feature_params.lux_per_unit` | **fixed by design** (1000, kilolux scale) | `docs/FEATURES.md` |
| `feature_params.vref` (a, b, c) | **trained** (fitted on train fold) | Phase 5 code, fitted during training |
| `feature_params.low_light_lux`, `lux_saturation` | **placeholder**, tuned on data / sensor spec | `docs/FEATURES.md` §7 |
| `scaler.mean`, `scaler.std` | **trained** | Phase 7 |
| `weights.*` | **trained** | Phase 7 / FL rounds |
| `rules.*` | **placeholder**, needs Member 1 hardware confirmation | §7.4, R6 |

```json
{
  "status": "PLACEHOLDER",
  "contract_version": "1.0",
  "model_version": "c2-v0.0-placeholder",
  "feature_version": "pv8-v1",
  "labels": ["NORMAL","PARTIAL_SHADING","DUST","OPEN_CIRCUIT","SHORT_CIRCUIT"],
  "features": ["v_deficit","i_norm","p_norm","dust_vo_v","temp_spread","temp_rise","voltage_v","current_ma"],
  "feature_params": {"lux_per_unit": 1000.0, "vref": {"a": "<TRAINED>", "b": "<TRAINED>", "c": "<TRAINED>"}, "low_light_lux": "<PLACEHOLDER 2400>", "lux_saturation": "<PLACEHOLDER 100000>"},
  "scaler": {"mean": "<TRAINED: 8 floats>", "std": "<TRAINED: 8 floats>"},
  "activation": "tanh",
  "weights": {"W1": "<TRAINED: 8x16>", "b1": "<TRAINED: 16>", "W2": "<TRAINED: 16x5>", "b2": "<TRAINED: 5>"},
  "rules": {"open_i_ma": "<PLACEHOLDER>", "short_v": "<PLACEHOLDER>", "persist_n": "<PLACEHOLDER>"},
  "trained_on": {"dataset_sha256": "<set by training>", "split": "<set by training>", "git_commit": "<set by training>"}
}
```
`scripts/export_c_header.py` turns a **TRAINED** contract into `firmware/.../model_params.h`. A host-side C test checks that C and Python give the same outputs (tolerance 1e-5).

### 7.3 Output: prediction message (backend `prediction` stream, MQTT `solar/{site_id}/prediction`)
| Field | Type | Notes |
|---|---|---|
| `record_id` | string | same `record_id` as the input reading (join key) |
| `timestamp_utc`, `site_id`, `device_id` | string | copied from input |
| `prediction_status` | enum `CLASSIFIED` / `SKIPPED_LOW_LIGHT` / `SKIPPED_LUX_SATURATED` / `SKIPPED_INVALID_INPUT` | **new field**: whether a classification exists |
| `predicted_label` | enum or `null` | one of the 5 labels when `CLASSIFIED`; `null` when skipped (never `NORMAL` by default) |
| `confidence` | float 0–1 or `null` | max probability; `null` when skipped |
| `prob_normal … prob_short_circuit` | float ×5 or `null` | **recommendation: send all 5**. 5 × 4 B is small, and Member 4 SHAP needs them |
| `decision_source` | enum `ML` / `RULE` / `NONE` | **new field**: who decided (`NONE` when skipped) |
| `alert_level` | enum `NONE`/`INFO`/`WARNING`/`CRITICAL` | mapping §7.4 |
| `relay_recommendation` | enum `KEEP_CONNECTED` / `RECOMMEND_ISOLATE` | **PROPOSED**, advisory only, see §7.5. Replaces a real `relay_status` until hardware exists |
| `model_version` | string | e.g. `v1.20` from FL or `c2-v0.1` |
| `inference_latency_ms` | float | **measured** on device (`esp_timer`) |

### 7.4 Decision fusion
```
if invalid_input:                       status=SKIPPED_INVALID_INPUT, label=null, source=NONE, alert=NONE
elif I < open_i_ma and V > k_oc·V_ref:  status=CLASSIFIED, label=OPEN_CIRCUIT,  source=RULE, alert=CRITICAL
elif V < short_v and I > i_sc_min:      status=CLASSIFIED, label=SHORT_CIRCUIT, source=RULE, alert=CRITICAL
elif low_light:                         status=SKIPPED_LOW_LIGHT,     label=null, source=NONE, alert=NONE
elif lux_saturated:                     status=SKIPPED_LUX_SATURATED, label=null, source=NONE, alert=NONE
else:                                   status=CLASSIFIED, label=argmax(p), source=ML,
                                        alert = WARNING if label≠NORMAL and conf≥c_hi
                                                INFO    if label≠NORMAL and conf<c_hi
                                                NONE    if label=NORMAL
```
All thresholds (`open_i_ma`, `short_v`, `k_oc`, `i_sc_min`, `c_hi`) are **placeholders**. They are set from validation data and confirmed against Member 1's hardware.

Skipped readings are counted and reported in evaluation (coverage = classified / total). They are never mixed into accuracy as `NORMAL`.

### 7.5 Relay control: PROPOSED safety subsystem (not implemented until confirmed)
**Status:** design proposal only. **No GPIO or relay code is written** until Member 1 confirms:
1. whether the device has a relay/MOSFET, and its rating (voltage, current, DC breaking capability);
2. which faults require isolation, and the **maximum allowed time-to-isolate** for this panel and wiring;
3. existing hardware protection (fuse, current limit), and whether software isolation is needed at all;
4. who owns the actuator code (Member 1 firmware or Member 2 module).

**Proposed logic, for discussion only:**
- Isolation is recommended only from **RULE** detections of OPEN or SHORT that persist for `persist_n` consecutive readings. **`persist_n` and therefore the delay are placeholders.** The earlier "3 × 5 s = 15 s" is **not** claimed to be electrically safe. A short circuit on a PV panel is current-limited by the cell (≈ Isc), but wiring, connectors and any battery/load side may need faster action. That decision belongs to Member 1's protection design, not to the ML model.
- Reconnection only on manual reset; no automatic reconnection.
- ML-only labels (DUST, SHADING) never trigger isolation.
- Until confirmed, Component 2 outputs only `relay_recommendation` (advisory). Hardware-level protection, if any, remains Member 1's responsibility.

---

## 8. Training and evaluation

### 8.1 Split protocols (all three reported)
1. **Run-wise 5-fold StratifiedGroupKFold** (group = `run_id`), used for model selection.
2. **Temporal hold-out:** TRAINING window → OPERATION window, matching the group data contract.
3. **Leave-one-site-out** (6 folds), for generalisation to an unseen device. Note: some sites lack some classes (e.g. SITE-F has no NORMAL), so per-fold class coverage is reported.

The final test set (OPERATION) is touched once, after model selection.

### 8.2 Metrics
| Group | Metric |
|---|---|
| Classification | macro-F1 (primary), per-class precision/recall, confusion matrix, balanced accuracy |
| Safety | OPEN/SHORT recall (target ≥ 0.99), **false isolation-recommendation rate** (per 1,000 normal readings), time-to-detect (s) |
| Coverage | share of readings `CLASSIFIED` vs each `SKIPPED_*` status; accuracy reported on classified readings only, with coverage shown next to it |
| Run-level | per-run majority-vote accuracy, worst-run accuracy |
| Calibration | expected calibration error (confidence is sent to the dashboard) |
| Robustness | leave-one-site-out macro-F1; sensitivity to ±5% sensor noise |
| Agreement | agreement with Member 4 physics rules (`causes.py`) |
| On-device | flash bytes, RAM bytes (heap before/after), latency µs (mean/p95 over 1,000 runs), optional energy |
| Quantisation | Δ macro-F1 and Δ OPEN/SHORT recall: float32 vs INT8 |

### 8.3 Colab workflow
1. `!git clone -b component2-tinyml https://github.com/Lisura123/J26-IT-348` and `pip install -r component2-tinyml/requirements.txt`.
2. Upload the CSVs to the Colab session (not to git). The notebook checks SHA-256 against the README.
3. The notebook **only calls repo modules** (`validate`, `splits`, `features`, `train`, `evaluate`, `export`). No preprocessing code lives in the notebook.
4. Outputs: `model_contract.json`, `metrics.json`, plots. These are downloaded and committed by the normal reviewed workflow.

The dataset is small (6k rows) and trains on CPU in seconds. Colab is for convenience, not compute.

---

## 9. Quantisation and ESP32 deployment

### 9.1 Deployment candidates (final choice made from ESP32 measurements)
| Variant | Description | Status |
|---|---|---|
| A. Float32 | hand-written C (FPU), weights from `model_params.h`; optionally TFLM float32 for comparison | **candidate** |
| B. INT8 | TFLM, full-integer per-tensor quantisation (per-channel Dense was unsupported in the earlier `tflm_esp32` test) | **candidate** |

Both are built, run on the same ESP32 board and the same test readings, and measured on:
- flash (bytes, from build map), RAM (heap before/after init + arena size), latency (µs, mean and p95 over ≥ 1,000 inferences, `esp_timer_get_time()`);
- macro-F1 and OPEN/SHORT recall on the held-out set (on-device outputs vs Python reference);
- optional: energy per inference.

**Selection rule (fixed before measuring):** choose INT8 only if its macro-F1 drop is ≤ 1 percentage point, OPEN/SHORT recall does not drop, and it gives a measurable flash, RAM or latency benefit. Otherwise choose Float32. Either outcome is reported as a result. No improvement is claimed before measurement. The TF/TFLM versions used are recorded.

### 9.2 Pruning decision vs the original research proposal
| | |
|---|---|
| Proposal wording | **To be quoted from the approved proposal by Member 2** (exact sentence on pruning/optimisation) |
| Decision | Pruning is **not part of the deployment pipeline**. It is kept as an **optional measured experiment**, not dropped silently |
| Technical reason | The model has 229 parameters (~0.9 KB float32, ~0.3 KB int8). Unstructured pruning saves memory only with sparse storage, and sparse indices cost more than the removed weights at this size. TFLM does not speed up unstructured sparsity on ESP32. Structured pruning (removing hidden units, e.g. 16 → 8) changes the 229-param layout that Member 3's FL relies on |
| What will be done instead | (1) Optional experiment: magnitude pruning at 25/50/75% sparsity, reporting macro-F1 and stored size, to show the effect empirically. (2) A hidden-size sensitivity study (4/8/16 units) offline, as evidence of the size-accuracy trade-off, without changing the FL layout |
| Thesis statement | "Pruning was evaluated/considered and not adopted because …", with the measured numbers. This is a documented, justified deviation from the proposal, to be confirmed with the supervisor |

### 9.3 Firmware module (proposed; firmware owner to agree)
```c
typedef struct { float v, i_ma, p_mw, t1, t2, tamb, lux, dust; } pv_reading_t;
typedef struct { uint8_t status;      /* CLASSIFIED / SKIPPED_*        */
                 int8_t  label;       /* -1 when not classified        */
                 float   conf; float prob[5];
                 uint8_t source, alert, relay_recommendation;
                 uint32_t latency_us; } pv_result_t;
void pv_fault_init(void);
pv_result_t pv_fault_classify(const pv_reading_t *r);  // called every 5 s by Member 1's loop
```
- Feature, scaler and MLP code are generated or parameterised from `model_params.h`.
- Unit-tested on the host with gcc against Python outputs before flashing.
- OTA update: Member 3 publishes new weights (same 229 layout). The device swaps them, and the `model_version` field reports which is active.

---

## 10. Folder structure (target)
```
component2-tinyml/
  README.md  requirements.txt  pytest.ini  .gitignore
  docs/
    ARCHITECTURE.md          this document
    DATA_PROVENANCE.md       Phase 3
    EVALUATION.md            results (later)
  src/tinyml/
    contract.py              column roles, labels           (done, 6bf0049)
    validate.py              dataset checks                 Phase 2
    splits.py                run-wise splits                Phase 4
    features.py              PV-physics features            Phase 5
    models.py                MLP + baselines (numpy/sklearn) Phase 6–7
    train.py  evaluate.py    training / metrics              Phase 7–8
    export.py                model_contract.json, C header  Phase 9
  scripts/                   01_validate_dataset.py, 02_make_splits.py, ...
  notebooks/                 colab_train.ipynb (calls src only)
  models/                    model_contract.json (small, committed)
  results/                   metrics json/csv, plots (small, committed)
  firmware/pv_fault/         C module, model_params.h, host tests
  tests/                     pytest (+ C host test)
  data/raw/                  git-ignored; datasets live here locally
```

---

## 11. Roadmap (one reviewed commit + push per step)
| Phase | Deliverable | Depends on |
|---|---|---|
| 1 ✅ | Component init + column contract (`6bf0049`) | — |
| 1b | **This architecture document** | approval |
| 2 | Dataset validation module, tests, validation report | uploaded CSVs |
| 3 | `DATA_PROVENANCE.md` + questions to Member 1 | Phase 2 |
| 4 | Run-wise split module (3 protocols), saved split file | Phase 2 |
| 5 | Feature layer (8 PV features, train-only fitting, low-light gate) | Phase 4 |
| 6 | Baseline comparison (rules, LR, DT, RF, MLP; raw-8 vs physics-8) | Phase 5 |
| 7 | MLP training + `model_contract.json` export; Colab notebook | Phase 6 |
| 8 | Evaluation report (all metrics §8.2 except on-device) | Phase 7 |
| 9 | C export + host-side C test (C == Python) | Phase 7 |
| 10 | Decision fusion with prediction status + advisory relay recommendation (Python reference + C), tests | Phase 9 |
| 11 | Float32 and INT8 builds (both candidates); optional pruning experiment | Phase 9 |
| 12 | ESP32 integration + on-device measurements → **select Float32 or INT8** by §9.1 rule | hardware, Member 1 |
| 12b | Relay actuation, **only if** Member 1 confirms hardware + protection requirements | R6 |
| 13 | FL integration with Member 3 (features, scaler, activation, OTA) | Phase 7, Member 3 |
| 14 | Backend publishing + Member 4 SHAP alignment | Phase 10, Members 3/4 |
| 15 | Real-lab validation runs (2–3 per class) | Member 1 / lab access |
| 16 | Final documentation | all |

---

## 12. Unresolved risks and questions
| # | Risk / question | Impact | Owner |
|---|---|---|---|
| R1 | **Dataset provenance: UNRESOLVED.** PV sensor dataset provided by Member 1; group contracts to be reconciled with Member 1; 4 cross-site run pairs overlap in time; no firmware or raw logs in the repo. Stays unresolved until Member 1 provides firmware, raw logs and run records | results may not transfer to real panels; academic integrity | Member 1 → evidence; Member 2 → document in `DATA_PROVENANCE.md` |
| R2 | **Task too easy on current data** (rules ≈ 97–98%) | weak justification for ML; must be addressed in the thesis | Member 2 |
| R3 | **Activation mismatch** (Member 3 tanh, Member 4 ReLU), see §6.4 | silent wrong predictions/SHAP in the twin | Members 3/4; contract field proposed |
| R4 | **Feature change needs Member 3 agreement** (same dimension, new meaning + scaler) | FL trains the wrong function if not aligned | Members 2/3 |
| R5 | **Sensor models unknown** (lux > 65,535 in data) | feature formulas, saturation and low-light thresholds may be wrong | Member 1 |
| R6 | **Relay hardware, protection requirements and ownership unknown**; safe time-to-isolate unknown | relay stays a proposal; no actuation code | Member 1 |
| R11 | Pruning in the original proposal is not adopted for deployment (§9.2) | must be justified to the supervisor | Member 2 |
| R12 | Skipped readings (low light, saturation) reduce coverage | dashboard has gaps at dawn/dusk | report coverage; tune `g_min` |
| R7 | Uneven class mix per site (some sites miss classes) | non-IID FL; LOSO folds miss classes | Member 3 aware; report it |
| R8 | INT8/TFLM version issues on ESP32 | ablation may be partial | Member 2 |
| R9 | No ESP32 measurements yet | all cost numbers are estimates until Phase 12 | Member 2 |
| R10 | `member2_fault_predictions.csv` stand-in could be mistaken for real results | invalid conclusions | guard in `validate.py`; documented |

**Questions to confirm before Phase 5:** (1) sensor part numbers and ADC/resolution; (2) is there a relay on the device, its rating, the required time-to-isolate, and who controls it; (3) can Member 3 accept physics-8 features with the same 229-param layout; (4) tanh as the agreed activation, with Member 4 reading it from the contract; (5) is lab access available for real validation runs; (6) can the backend accept `prediction_status`, a `null` `predicted_label`, and `relay_recommendation`; (7) the exact pruning wording in the approved proposal.

# Final Model Training and Export

Code: `src/tinyml/train.py`, `src/tinyml/export.py`. Script: `scripts/05_train_final.py`. Colab: `notebooks/colab_final_training.ipynb`. Tests: `tests/test_train_export.py`.

**Status:** code ready. The official training run has **not** been done; it is run manually by Member 2 in Google Colab. No trained weights or results exist yet.

## 1. What is trained

The model approved in `PHASE7_RESULTS.md` (decisions 1 and 2):

| Item | Value | Source |
|---|---|---|
| Features | physics8, `pv8-v1`, in the order of `features.FEATURES` | `features.py`, unchanged |
| Model | 8-16-5 MLP, tanh hidden layer, softmax output, 229 parameters | `models.make_model("mlp")`, unchanged |
| Hyperparameters | the Phase 7 settings: `hidden_layer_sizes=(16,)`, `activation="tanh"`, `solver="adam"`, `max_iter=2000`, other values scikit-learn defaults | no new hyperparameters; the full `get_params()` is stored |
| Input gates | low light 2,400 lx and lux saturation 100,000 lx, still **placeholders** | `FeatureConfig`, unchanged |

## 2. Data used

- Only `pv_fault_dataset.csv`. Its SHA-256 must be `5fbb9347…421595b` (`train.DATASET_SHA256`), or training stops. The per-class files are not used or concatenated.
- **Only the TRAINING window.** These rows must equal the `train` side of the temporal split in `results/splits_runwise_v1.json`.
- V_ref, the scaler and the MLP weights are all fitted on these rows. The low-light, saturation and rule thresholds are not tuned.
- **The OPERATION window is not used** for training, hyperparameter or seed choice, feature fitting, scaling, threshold tuning or exploratory evaluation. Tests check that changing every OPERATION row leaves the trained model bit-for-bit identical.
- The only metric computed is training-fit accuracy on the training rows. It is a fit check, **not a performance result**. Performance estimates are the Phase 7 cross-validation results; the TRAINING → OPERATION final test is a separate later step.

## 3. Seed policy

**Issue:** Phase 7 trained the MLP with seeds 0, 1 and 2 and reported their mean, but it did not decide which seed the final model uses.

**Policy (approved by Member 2):** the final model uses **seed 0**, fixed before training for reproducibility: the lowest of the Phase 7 seeds. Seed 0 is **not** chosen because of its score. No seed is selected by score, and OPERATION data is never used. Seeds 1 and 2 are not trained for the final model.

Why a fixed rule instead of picking the best seed: Phase 7 shows the seed has little effect. The pooled macro-F1 for physics8 + MLP varies by at most 0.002 across seeds (CV 0.9818 / 0.9798 / 0.9818; LOSO 0.9798 / 0.9794 / 0.9794), and OPEN/SHORT recall is 1.000 for all three. Picking the highest-scoring seed would add a selection step with no real gain. Seed 0 happens to share the highest CV value, but that played no part in the choice.

The run is reproducible only with the same library versions. The versions used are stored in the contract (Colab and local scikit-learn versions differ).

## 4. Files produced

| File | Content |
|---|---|
| `model_contract.json` | status `TRAINED`, artifact kind, labels, features and units, the complete fitted `FeatureConfig`, scaler, activation, parameter layout, weights, weights SHA-256, rule placeholders, provenance (dataset hash, split, window, git commit), library versions, reference probes, integration status |
| `feature_config.json` | the complete fitted `FeatureConfig` (`FeatureConfig.save`) |
| `training_metadata.json` | rows, runs, skipped readings, class counts, hyperparameters, iterations, convergence, loss, warnings, time created, `operation_window_used: false` |
| `manifest.json` | SHA-256 of the three files above, plus the weights SHA-256 |

`export.verify_artifacts()` checks every hash, the label and feature order, the 229-value layout, that the contract's `feature_config` matches `feature_config.json`, that the model was fitted on `final_TRAINING`, and that the reference probes reproduce. A contract whose status is not `TRAINED` is refused.

The reference probes are the first 5 scaled TRAINING inputs and their expected probabilities. Other implementations of the forward pass (C on the ESP32, Member 3's FL, Member 4's twin) can test against them.

## 5. Weight layout and class order

- Layout copied from Member 3's FL code (`component3-federated-backend/src/mlp.py`, `LAYOUT`): `W1` (8, 16), `b1` (16), `W2` (16, 5), `b2` (5). The flattened vector is these four in order, each row-major: 128 + 16 + 80 + 5 = 229 values.
- Forward pass: `p = softmax(tanh(x · W1 + b1) · W2 + b2)`, where `x` is the scaled pv8-v1 input.
- **Class order fix.** scikit-learn orders its output units alphabetically (DUST, NORMAL, OPEN_CIRCUIT, PARTIAL_SHADING, SHORT_CIRCUIT). The export reorders the columns of `W2` and `b2` so that output `i` is `LABELS[i]` (NORMAL, PARTIAL_SHADING, DUST, OPEN_CIRCUIT, SHORT_CIRCUIT), the order Member 3 and Member 4 use. Without this step the exported weights would mislabel every class. Tests check that the exported forward pass gives the same probabilities as scikit-learn (tolerance 1e-5) and the same labels.
- `export.predict_raw()` classifies raw sensor rows from the contract alone, without scikit-learn. Skipped readings get a `SKIPPED_*` status and no label (never `NORMAL` by default).

## 6. Three kinds of model artifact

| Kind | Produced here? | What it is |
|---|---|---|
| **Locally trained model weights** (`LOCAL_TRAINED_WEIGHTS`) | **yes** | Float weights of one model trained centrally on the TRAINING window |
| **Future federated learning weight updates** | **no** | Per-site deltas against a published global model, in Member 3's format (`fl_weight_updates.jsonl`: `update_id`, `base_model_version`, `param_layout`, `delta_int8`, `int8_scale`). Created during FL rounds by Member 3's system |
| **Future INT8 deployment artifacts** | **no** | A quantised ESP32 build. Float32 or INT8 is chosen later from on-device measurements (`ARCHITECTURE.md` §9.1) |

## 7. Federated learning integration: verified limitations

**Reference branch:** `origin/feature/mongodb-migration` (Member 3's official FL integration branch, audited read-only at `496f5be`, 2026-10-16). Its FL core (`mlp.py`, `pv_data.py`, `engine.py`, `fl_baseline.py`, `compression.py`) is identical to `component3-federated-backend@375a9e9`. No FL interface specification from Member 1 was found, so none is assumed here. Member 3's code has not been changed.

**Compatible:** 8-16-5 MLP, 229 parameters, layout `W1, b1, W2, b2` flattened row-major, forward `x · W1`, softmax output, label order, and the nested-list weight format (`src/mlp.py:3,27-36`, `src/compression.py:5`, `src/pv_data.py:7`). These match only the **shape** of the model. They do not make the models functionally compatible.

**Not compatible yet (verified):**

| # | Limitation | Evidence (Member 3 branch) |
|---|---|---|
| 1 | FL currently uses **raw8 inputs**: `voltage_v, current_ma, power_mw, panel_temp1_c, panel_temp2_c, ambient_temp_c, irradiance_lux, dust_vo_v` | `src/pv_data.py:5-6` |
| 2 | Our TinyML uses **pv8-v1 features** (`v_deficit, i_norm, p_norm, dust_vo_v, temp_spread, temp_rise, voltage_v, current_ma`). Input 0 means `voltage_v` there and `v_deficit` here, so loading our weights would give silently wrong predictions | `src/tinyml/features.py` |
| 3 | **Preprocessing configurations differ.** FL fits a z-score scaler centrally on all TRAINING rows at every run and does not save it; there is no V_ref and no low-light, saturation or invalid-input gating. Ours is the saved, fitted `FeatureConfig` | `src/pv_data.py:16-24,31-34` |
| 4 | **Initial model weights cannot yet be loaded.** Every FL run starts from random weights | `src/engine.py:43`, `src/fl_baseline.py:61` |
| 5 | **Global model delivery is not implemented.** The backend API has no model endpoint; OTA flashing is simulated and `ota_deployments` holds the shared synthetic log | `project-backend/backend/api/main.py:77,102`, `routes_query.py:75,97,140`, `DEMO.md:103`, `src/load_to_mongo.py:159-187` |
| 6 | **OPERATION is used for repeated FL validation**: global accuracy on the OPERATION window is computed every round and used to compare configurations. Component 2 keeps OPERATION for a single final test | `src/pv_data.py:27-30`, `src/engine.py:62`, `src/evaluate.py:31-39` |
| 7 | **MongoDB stores model paths, not model weights.** `fl_global_models` has a local `model_path`; weight files hold only `{"version", "weights"}`, with no features, scaler, activation or hash | `docs/DATABASE.md:83,95`, `src/export_results.py:96-103` |

Activation: FL runs default to tanh (`src/engine.py:29`), but `mlp.py` and `fl_baseline.py` default to ReLU (`src/mlp.py:10,32`, `src/fl_baseline.py:54,120`), and saved model files do not record the activation. The contract states `tanh` explicitly (risk R3).

**What integration would need (to agree with Member 3; nothing is changed here):** FL clients compute pv8-v1 with our `FeatureConfig` and train only on status-OK readings; preprocessing stays frozen while weights are federated; FL can start from a validated `model_contract.json`; global models carry feature version, labels, activation and hashes; run-wise TRAINING validation replaces per-round OPERATION validation; and a delivery path for the model and its `FeatureConfig` is defined. The contract's `reference_probes` can confirm that both forward passes agree.

Until then the exported model is **layout-compatible only**. It has **not** been submitted to, accepted by, or tested with the FL server. No aggregation or OTA update is implemented.

## 8. What is not claimed

- No federated aggregation is implemented, and the FL server has not accepted the model.
- No OTA update, no ESP32 deployment, no INT8 build.
- No relay is triggered. `relay_recommendation` stays an advisory proposal (`ARCHITECTURE.md` §7.5). The electrical confirmation rules stay placeholders (`PHASE7_RESULTS.md` decision 3).
- No field validation. Dataset provenance stays UNRESOLVED (`DATA_PROVENANCE.md`).

## 9. How to run (Colab)

1. Push the reviewed code to `component2-tinyml`, because the notebook clones that branch.
2. Open `notebooks/colab_final_training.ipynb` in Colab and run Steps 1–7 in order. Upload `pv_fault_dataset.csv` at Step 3.
3. Download `final_model.zip` and send it back for review. The files are committed only after review and approval.

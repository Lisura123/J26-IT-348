# Federated Learning Compatibility, Global-Model Update and OTA Package

**Status:** Member 2-side adapters implemented and tested on the host. **FL integration is
NOT complete**: no end-to-end round has run, Member 3's server has not accepted any Component
2 update or model, and no global model has been delivered to a device. Member 3's code is
unchanged. Reference: `origin/feature/mongodb-migration` at `496f5be` (re-checked
2026-10-16, unchanged since the audit in `TRAINING_EXPORT.md` §7).

## 1. Compatibility matrix (Member 3's FL vs the frozen contract)

| Item | Member 3 (actual code) | Component 2 | Result |
|---|---|---|---|
| Architecture | 8-16-5 MLP (`src/mlp.py`) | 8-16-5 MLP | compatible |
| 229-parameter layout | `LAYOUT` W1(8,16), b1(16), W2(16,5), b2(5), row-major | same | **compatible** (tested against `mlp.py` source) |
| Activation | `engine.py` default tanh; `mlp.py`/`fl_baseline.py` default ReLU; not stored in files | tanh, stored in the contract | **risk**: files do not say which was used |
| Feature order / input space | raw8: `voltage_v … dust_vo_v` (`pv_data.py`) | pv8-v1: `v_deficit … current_ma` | **incompatible** |
| Normalisation | z-score fitted at every run on all TRAINING rows, not saved | frozen FeatureConfig scaler | **incompatible** |
| V_ref | none | fitted V_ref(lux, T) in FeatureConfig | **missing in FL** |
| Quality gates | none (all rows used) | LOW_LIGHT, LUX_SATURATED, INVALID_INPUT; train on status-OK rows | **missing in FL** |
| Label order | NORMAL, PARTIAL_SHADING, DUST, OPEN_CIRCUIT, SHORT_CIRCUIT | same | compatible |
| Initial weights | random init every run (`engine.py`, `fl_baseline.py`) | trained contract | **cannot load ours** |
| Model versions | `v1.0 … v1.20`, file `{"version", "weights"}` | `c2-local-v1`, contract + hashes | not linked |
| FL updates | `fl_weight_updates.jsonl`: `update_id, round_id, site_id, device_id, base_model_version, param_layout, delta_int8, int8_scale` | same fields + compatibility fields | format compatible; semantics not |
| Aggregation | FedAvg (`fl_global_models.csv`) | not Member 2's | Member 3 |
| Global-weight delivery | not implemented; OTA simulated | package format pv-ota-1 (Section 4) | **gap** |
| Model integrity | no hashes in weight files | SHA-256 of weights, contract, FeatureConfig | **gap** |
| Validation data | OPERATION used every round | OPERATION used once (Phase 9), never again | **conflict** |

Consequence: Member 3's current global models (raw8, unknown activation) **must not** be
loaded into Component 2. `import_global_weights` rejects them (`MISSING_COMPATIBILITY_METADATA`;
`INCOMPATIBLE_PREPROCESSING` for `feature_version = "raw8"`); `tests/test_member3_compat.py`
checks this with Member 3's real `global_v1.20.json`.

## 2. Member 2-side adapters (`src/tinyml/fl_adapter.py`)

| Function | Purpose |
|---|---|
| `local_model_metadata(contract, sha)` | model/contract versions, hashes, `feature_version`, `feature_config_sha256` (canonical JSON of the frozen FeatureConfig), features, labels, activation, layout, `n_params` |
| `flatten` / `unflatten` (export.py) | 229-vector in Member 3's order (round-trip tested) |
| `check_weights` | shapes, finite, sanity bound \|w\| ≤ 64 (corruption guard, not a tuned limit) |
| `compute_delta`, `int8_dense` | local − base; Member 3's `int8_dense` reproduced exactly (tested against their source) |
| `build_update` | record in Member 3's JSONL format + `num_samples`, `class_counts`, `base_weights_sha256`, compatibility fields, `update_sha256` |
| `verify_update` | completeness, integrity hash, layout, base weights, int8 range, scale, sample counts |
| `import_global_weights(doc, metadata, base_version)` | accepts only documents with complete compatibility metadata equal to ours; checks layout, values, optional `weights_sha256`, base version |
| `apply_update` | base + dequantised delta, for checks only (aggregation stays with Member 3) |

## 3. Global-model update workflow (`src/tinyml/global_update.py`)

```
global Float32 weights + compatibility metadata
  -> fl_adapter.import_global_weights          version, contract, layout, values, integrity
  -> build_candidate                           new version c2-fl-<global>, status CANDIDATE_NOT_VALIDATED,
                                               FeatureConfig/scaler/gates/labels copied unchanged,
                                               reference probes recomputed
  -> validate_candidate                        frozen keys unchanged, hashes, probes
  -> quantize_candidate                        int8-v1 rule; REFUSES without explicitly supplied
                                               TRAINING calibration rows (never OPERATION)
  -> verify_candidate                          Float32 vs INT8 agreement and OPEN/SHORT changes
  -> write_candidate(out_root)                 candidates/<version>/ + manifest; never inside models/,
                                               never overwritten
  -> ota.build_package                         pv-ota-1 package
```

- The frozen model is immutable: nothing writes to `models/`; tests snapshot `models/`
  before and after.
- Calibration policy is unchanged (max-abs over status-OK TRAINING rows). Running the
  pipeline on real data, any recalibration and any accuracy evaluation of a candidate need
  a separate approval. A candidate is not evaluated by this pipeline.

## 4. OTA package format `pv-ota-1` (`src/tinyml/ota.py`)

ZIP with `ota_manifest.json`, `ota_manifest.sha256`, `model_contract.json` and, for INT8,
`quant_params.json`. Deterministic (fixed timestamps).

| Manifest field | Content |
|---|---|
| `package_format`, `package_id`, `created_utc` | `pv-ota-1`, hash-derived id |
| `model_version`, `base_model_version`, `contract_version` | new and base versions |
| `feature_version`, `feature_config_sha256` | preprocessing identity (must equal the device's) |
| `backend`, `quant_version`, `quant_params_sha256` | FLOAT32 or INT8 |
| `model_hash` | weights SHA-256 (recomputed on validation) |
| `files` | SHA-256 per payload file |
| `compatibility` | labels, features, activation, output activation, layout, `n_params`, `api_version` |
| `rollback` | `rollback_to_model_version`, `rollback_weights_sha256`, policy |
| `signature` | `null`: **signing not implemented**; validation warns `UNSIGNED_PACKAGE` |
| `deployment_status` | `NOT_DEPLOYED` |

`validate_package(package, device_state)` rejects: missing/changed manifest, unexpected or
changed files, other feature version or FeatureConfig, other model interface, wrong base
version, already-active model, wrong rollback metadata, model-hash mismatch, invalid
weights, INT8 parameters not derived from the contract. `MockDevice` simulates A/B slots:
`RECEIVED → VERIFIED → STAGED → ACTIVE_PENDING_CONFIRMATION → CONFIRMED | ROLLED_BACK`, or
`REJECTED`.

**Not done:** no package has been sent to a device and no OTA flashing occurred. The
firmware compiles the model into the image (generated headers), so a real model update needs
either a full firmware OTA (`esp_ota_*`, Member 1's partition table) or runtime model loading,
plus package signing. Both are open design decisions.

## 5. Handover specification for Member 3

Needed before FL can train or deliver models usable by Component 2:

1. **Input space:** FL clients compute pv8-v1 features with the frozen FeatureConfig
   (`models/feature_config.json`, `fl_adapter.feature_config_sha256`) and train only on
   status-OK readings. The scaler is not refitted per run.
2. **Activation:** tanh everywhere (`mlp.py` and `fl_baseline.py` defaults are ReLU).
3. **Initial model:** start round 0 from `models/model_contract.json` weights (same 229 layout).
4. **Global model file:** add `feature_version`, `feature_config_sha256`, `activation`,
   `output_activation`, `labels`, `features`, `param_layout`, `n_params`, `weights_sha256`
   and `base_model_version` to every global model (values from `local_model_metadata`).
5. **Updates:** accept the extra fields of `build_update` (`num_samples`, `class_counts`,
   compatibility fields, hashes) or tell Member 2 which to drop.
6. **Validation:** run-wise TRAINING validation instead of per-round OPERATION accuracy.
7. **Delivery:** an API endpoint or MQTT topic for a pv-ota-1 package, the device-side
   update mechanism (firmware OTA or runtime loading) and who signs packages.
8. **End-to-end test:** one round with the adapters on both sides; only after it passes may
   FL integration be called complete.

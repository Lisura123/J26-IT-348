# Member 2 Software Architecture (Component 2: PV-Specific On-Device TinyML Fault Detection)

This document ties the parts together. Details: `ESP32_INTEGRATION.md` (device software,
sensor adapter), `FAULT_CONFIRMATION_AND_RELAY.md`, `MESSAGING.md`, `FL_INTEGRATION.md`
(FL, global update, OTA), `COMPLETION_REPORT.md` (status). Model research: `ARCHITECTURE.md`,
`TRAINING_EXPORT.md`, `OPERATION_RESULTS.md`, `QUANTIZATION.md`, `PRUNING.md`.

## 1. Frozen research results (unchanged by the software work)

| Item | Value | Source |
|---|---|---|
| Model | `c2-local-v1`: pv8-v1 features → 8-16-5 tanh MLP → softmax, 229 parameters | `models/` |
| Independent OPERATION test (once) | accuracy 0.979, macro-F1 0.978 (95% CI 0.937–0.999), coverage 0.980 | `OPERATION_RESULTS.md` |
| INT8 `int8-v1` | agrees with Float32 on all 2,924 TRAINING rows (0 changes); not independent accuracy | `QUANTIZATION.md` §12 |
| Pruning E1 | no TRAINING prediction changed; evaluated, not adopted (provisional) | `PRUNING.md` §9 |

No component of the software layer retrains, re-tunes, re-calibrates or re-evaluates the
model, and none reads OPERATION rows.

## 2. Runtime architecture (device, shadow mode)

```
Member 1 drivers (INA219, DS18B20 x2 (+1 needed), TEMT6000, GP2Y10), calibration, clock
        |  pv_sensor_sample_t: calibrated values + per-channel state + UTC ms
        v
pv_sensor_adapt          NaN for anything not VALID; power_mw = V x mA; no filling or copying
        |  raw[8] + invalid_mask
        v
pv_api_classify          quality gates -> pv8-v1 features -> frozen scaler -> MLP
  (FLOAT32 | INT8)       skipped reading: status, no label (never NORMAL, never a fault)
        |  pv_prediction_t: status, label, confidence, prob[5], clip counts
        v
pv_confirm_update        non-actuating diagnostic state machine (provisional defaults)
        |  NO_FAULT / ML_PREDICTED / CONFIRMATION_PENDING / CONFIRMED_BY_CONFIGURED_RULES /
        |  INSUFFICIENT_EVIDENCE
        v
pv_message_json          pv-msg-1 JSON -> MQTT solar/{site_id}/prediction (Member 1's transport)

pv_relay (mock only, physical actuation disabled) is NOT in this path.
```

`pv_pipeline_step()` performs these steps for one reading. `pv_api_init()` checks the
compiled model against generated reference vectors before the first classification.

## 3. Fault detection workflow and quality gates

1. **Gates (frozen, precedence INVALID_INPUT > LUX_SATURATED > LOW_LIGHT > OK):** NaN/±inf,
   DS18B20 error codes −127/−999 in a panel temperature, or a value outside the physical
   limits → INVALID_INPUT; lux ≥ 100,000 → LUX_SATURATED; lux < 2,400 → LOW_LIGHT. Both lux
   thresholds are placeholders. Skipped readings are counted (coverage) and reported with
   `prediction_status = SKIPPED_*`.
2. **Features:** V_ref(lux, T) deficit, current and power per kilolux, dust voltage, panel
   temperature spread and rise, voltage, current; frozen scaler.
3. **Classification:** Float32 or INT8 (below). Label = argmax; confidence = max softmax
   probability.
4. **Confirmation:** software rule over consecutive readings; never electrical confirmation.
5. **Message:** every reading produces a message (also skipped ones) when a UTC timestamp exists.

## 4. Float32 and INT8 deployment options

| | Float32 (`pv_mlp_f32.c`) | INT8 (`pv_mlp_int8.c`) |
|---|---|---|
| Role | reference, source of truth, default | deployment derivative |
| Preprocessing | Float32 | Float32 (same) |
| Core | float MACs + `tanhf` | int8 × int8 → int32, fixed-point M1/SH1, 256-entry tanh table |
| Equivalence | logits within tolerance of Python | bit-exact with the Python emulator |
| Parameter storage (estimate) | 916 B | 561 B incl. tanh table and constants |
| Decision | from ESP32 flash/RAM/latency measurements (`ARCHITECTURE.md` §9.1) | **not decided: no measurement exists** |

Both are linked and selectable per call; the benchmark application measures both.

## 5. Host-side software (Python, `src/tinyml/`)

| Module | Purpose |
|---|---|
| `selfcheck.py` | generates `pv_selfcheck.h` from the frozen contract and INT8 parameters (no data) |
| `messaging.py` | pv-msg-1 builder and validator (mirrors the backend rules) |
| `fl_adapter.py` | FL metadata, layout, deltas, update records, global-weight import validation |
| `global_update.py` | versioned candidate pipeline (never writes `models/`) |
| `ota.py` | pv-ota-1 package build/validation, mock A/B device |
| `benchmark.py` | parses device logs; NOT_EXECUTED / HOST_ONLY / INVALID / ESP32_MEASURED |
| `hostc.py` | builds and runs the C tests and the apps against host stubs |

## 6. Testing instructions

```bash
cd component2-tinyml
pip install -r requirements.txt
pytest                                   # all Python and host C tests
python scripts/09_generate_selfcheck.py --check
cmake -S firmware/pv_fault -B /tmp/pvb && cmake --build /tmp/pvb      # host library build
```

`tests/test_member3_compat.py` additionally needs `pyyaml` and `python-dotenv` and the
fetched branch `origin/feature/mongodb-migration`; it skips otherwise. C tests need gcc.
No test reads the dataset, except the existing scripts' synthetic or TRAINING-only checks.

## 7. Known limitations

- Nothing has run on an ESP32; no ESP-IDF build has been made; no memory or timing figure
  exists (`NOT_EXECUTED`).
- With the stated sensors (one panel DS18B20) every reading is INVALID_INPUT; TEMT6000 lux
  conversion and the GP2Y10 `dust_vo_v` definition are unresolved (`ARCHITECTURE.md` §7.6).
- Dataset provenance unresolved; OPERATION results are dataset results, not field validation.
- Confirmation defaults and electrical thresholds are provisional; no physical isolation.
- FL is layout-compatible only; Member 3's pipeline uses raw8 inputs; no end-to-end round.
- OTA packages are unsigned; no device-side update mechanism; no OTA performed.
- The prediction field list awaits Member 3's confirmation of the backend schema.

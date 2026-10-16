# INT8 Quantisation: Protocol and Arithmetic

Code: `src/tinyml/quantize.py` (arithmetic, calibration, emulator, comparison), `src/tinyml/device_reference.py` (Float32 preprocessing reference), `src/tinyml/golden.py` (golden vectors), `src/tinyml/codegen.py` (C headers), `src/tinyml/hostc.py` (gcc host tests). Script: `scripts/07_quantize_model.py`. Colab: `notebooks/colab_quantization.ipynb`. Firmware: `firmware/pv_fault/` (portable C99). Tests: `tests/test_quantize.py`, `tests/test_codegen.py`, `tests/host_c/test_main.c`.

**Status:** protocol fixed before the official run (`int8-v1`). The official quantisation has **not** been run; generated headers, INT8 parameters and the report are produced by that run and committed after review. No ESP32 deployment or measurement has been done.

## 1. System

**Float32 preprocessing + INT8 neural inference.**

```
raw reading -> [Float32] quality gates -> [Float32] pv8-v1 features -> [Float32] frozen scaler -> z
z -> [INT8] quantise -> int8 x int8 + int32 bias -> fixed-point rescale -> tanh lookup -> int8 x int8 + int32 bias
  -> [Float32] 5 logits -> argmax (label), softmax (confidence)
```

- The Float32 fallback (`pv_mlp_f32.c`) uses the same preprocessing and a Float32 MLP.
- The float contract (`models/model_contract.json`, `feature_config.json`) is **not changed**. It stays the source of truth and keeps the 229-value FL layout. The INT8 parameters (`models/int8/quant_params.json`) are a **deployment derivative** that records the contract and weights SHA-256 they were made from.
- Feature order (pv8-v1), label order (NORMAL, PARTIAL_SHADING, DUST, OPEN_CIRCUIT, SHORT_CIRCUIT), gate thresholds, V_ref and scaler come from the frozen FeatureConfig. `validate_params` and the code generator refuse any other order.

## 2. Arithmetic rules (identical in Python and C)

| Rule | Python | C |
|---|---|---|
| Round half away from zero (floats) | `rnd(v) = sign(v)·floor(\|v\| + 0.5)` in float64 (exact for float32 values) | `roundf()` |
| Integer division by 2^n, half away from zero | `rshift_rnd(v, n)` | `pv_rshift_rnd()`: negative values are negated first, so there is no right shift of a negative number |
| Saturation | `sat(v, lo, hi)` with a clip flag | `pv_sat()` with a clip counter |
| argmax ties | lowest index (LABELS order) | lowest index |

## 3. Quantisation mathematics

**Scales (fixed rule, computed once):**

| Symbol | Definition |
|---|---|
| `A_x` | max \|z\| over the status-OK TRAINING rows (Float32 device values) |
| `s_x = A_x / 127` | input scale; one scale for all 8 inputs; zero point 0 |
| `s_W1 = max\|W1\| / 127`, `s_W2 = max\|W2\| / 127` | symmetric per-tensor weight scales |
| `A_z1` | max \|z·W1 + b1\| over the same rows (float64) |
| `s_z1 = A_z1 / 127` | scale of the tanh lookup index |
| `s_h = 1 / 127` | scale of the tanh output |

**Parameters (float64, in Python):**
```
qW1 = sat(rnd(W1 / s_W1), -127, 127)        int8, never -128
qW2 = sat(rnd(W2 / s_W2), -127, 127)
qb1 = rnd(b1 / (s_x * s_W1))                 int32
qb2 = rnd(b2 / (s_h * s_W2))                 int32
M1, SH1:  M1 * 2^-SH1 = s_x * s_W1 / s_z1,  M1 in [2^30, 2^31), 1 <= SH1 <= 62
TANH_LUT[k + 128] = sat(rnd(127 * tanh(k * s_z1)), -127, 127),  k = -128..127
INV_SX = float32(1 / s_x),  OUT_SCALE = float32(s_h * s_W2)
```

**Inference:**
```
1. qx[i]   = sat(rnd(float32(z[i] * INV_SX)), -127, 127)          input clip counted
2. acc1[j] = qb1[j] + sum_i qx[i] * qW1[i][j]                        int32
3. k[j]    = sat(rshift_rnd(int64(acc1[j]) * M1, SH1), -128, 127)    lookup clip counted
4. hq[j]   = TANH_LUT[k[j] + 128]                                    int8
5. acc2[c] = qb2[c] + sum_j hq[j] * qW2[j][c]                        int32
6. label   = argmax acc2 (ties: lowest index)
7. logits  = float32(acc2) * OUT_SCALE;  prob = softmax(logits) in float32
```

**Overflow bounds** (checked by `validate_params` and the generator):
- \|acc1\| ≤ max_j(Σ_i \|qW1\|·127 + \|qb1\|) ≤ 2^31 − 1, and the same for acc2.
- \|acc1·M1\| < 2^31·2^31 = 2^62. With the rounding half (at most 2^61), the int64 rescaling stays below 2^63.
- Inputs beyond ±A_x clip to ±127. Pre-activations beyond ±A_z1 clip to lookup index −128 or 127, where tanh is already about ±1.

Float32 constants are written into the C headers as exact hexadecimal literals, so C and Python use bit-identical values.

## 4. Calibration (TRAINING only)

- The script drops every OPERATION row straight after loading. It checks the dataset SHA-256, the pinned model-file hashes and the split file (TRAINING rows = temporal `train` side).
- The Float32 device preprocessing is checked against the float64 pipeline (`features.py`): the gate status must be identical for every row, and the z differences are reported.
- `A_x` and `A_z1` are the max-abs values over the status-OK TRAINING rows. No percentile clipping and no tuning.
- Nothing is trained or refitted. The low-light (2,400 lx) and saturation (100,000 lx) thresholds stay placeholders.

## 5. Python–C equivalence

| Part | Requirement |
|---|---|
| INT8 core (steps 2–6), argmax, tie handling | **bit-exact** (acc2, k, hq, lookup clips) |
| Input quantisation for a given float32 z (step 1) | **exact** (same float32 multiply, same rounding) |
| Float32 preprocessing (uses `logf`) | \|Δz\| ≤ 1e-5 relative + 1e-5 absolute; gate status exact |
| Raw-reading cases | qx within ±1; labels exact when the top-2 margin is clear (acc2 gap ≥ 2 and Float32 logit gap ≥ 1e-3) |
| Float32 MLP | logits within 1e-5 relative + 1e-4 absolute; labels exact |

The host tests are compiled with `gcc -std=c99 -Wall -Wextra -Werror -pedantic -ffp-contract=off -fsanitize=undefined,address`. Firmware builds should also use `-ffp-contract=off`, so that the compiler does not fuse multiply-adds in the Float32 preprocessing.

## 6. Acceptance rule (TRAINING only, fixed before the run)

INT8 is compared with the official float reference (the Phase 8 contract forward pass in float64) on every status-OK TRAINING row. **INT8 passes only if all of these hold:**

1. **0** prediction changes involving OPEN_CIRCUIT or SHORT_CIRCUIT, in **either direction** (to or from);
2. agreement ≥ **99.5%**;
3. training-fit macro-F1 drop ≤ **1 percentage point**;
4. the golden vectors pass in the host C tests.

Otherwise the deployment stays Float32. The Float32 device path is also compared with the reference and reported.

**Limitations:**
- These are the rows the model was fitted on. On TRAINING, the gap between the top two class probabilities is at least 0.85, so the check is optimistic. It is **not** a held-out estimate.
- OPERATION was already used for the single final test and is not used again. INT8 behaviour on OPERATION and on real panels is **not measured**.
- Agreement between INT8 and Float32 says nothing about electrical safety. OPEN/SHORT outputs remain **unconfirmed ML predictions**, and no relay isolation is implemented.
- Passing this rule only makes INT8 a candidate. ARCHITECTURE §9.1 still requires a measured on-device benefit before INT8 is chosen.

## 7. Golden vectors and data origins

| Origin tag | What it is |
|---|---|
| `TRAINING_DERIVED` | scaled Float32 inputs (z) from fixed reading positions (1, 20, 40, 60) of the first 5 runs (by run_id) of each class. Derived values only; no raw readings are stored |
| `TRAINING_DERIVED_REFERENCE_PROBE` | the 5 reference probes from the contract |
| `SYNTHETIC_PROTOTYPE_FROM_TRAINING_MEDIANS` | synthetic raw readings from per-class **median** TRAINING sensor values, with irradiance, current and power scaled by 0.85–1.15 and panel temperatures offset by 0/0.5 °C. Expected labels come from the Python references, not from ground truth |
| `SYNTHETIC_EDGE_CASE` | synthetic raw readings for the gates and limits (section 8) |
| `SYNTHETIC_TEST_FIXTURE` | arithmetic vectors, core vectors and the synthetic models used in automated tests |

**None of these are physical sensor measurements.** Real sensor readings from Member 1's hardware have not been used.

## 8. Edge, clipping and overflow tests

- **Lux:** 2,399 and the float32 value just below 2,400 (LOW_LIGHT), 2,400 (OK), 99,999 (OK), 100,000 (LUX_SATURATED), just above 100,000 (INVALID_INPUT), 0 (LOW_LIGHT), −1 (INVALID_INPUT). `logf` is never called for a skipped reading.
- **Non-finite:** NaN, +inf and −inf in each of the 8 inputs give INVALID_INPUT. A non-finite model input is rejected (C returns −1; Python raises).
- **Sensor errors:** −127 and −999 in either panel temperature give INVALID_INPUT.
- **Limits:** each input exactly at its lower and upper physical limit, and one float32 step beyond.
- **Precedence:** INVALID_INPUT wins over LUX_SATURATED and LOW_LIGHT.
- **Clipping:** a valid reading with a large temperature spread (85 °C / −10 °C) is classified, and its input clip is counted.
- **Arithmetic:** `rshift_rnd` for n = 1…62 on positive, negative, tie and near-2^62 values, checked against exact fractions; `roundf`-style ties (±0.5, ±1.5, ±2.5, the float32 value just below 0.5, ±126.5, ±127.5); argmax ties.
- **Extreme parameters:** all weights ±127, biases at the int32 limit, M1 = 2^31 − 1, SH1 = 62. C (under UBSan) and Python agree, with no overflow.
- **Error detection:** a deliberately wrong expected value makes the C test fail, which shows the harness catches mismatches.

## 9. Federated learning compatibility

- The float contract and its 229-value layout (W1, b1, W2, b2) are unchanged. FL keeps working on float weights.
- An INT8 build is produced from any TRAINED float contract by the same generator. The manifest links it to the contract and weights SHA-256.
- Member 3's int8 **update deltas** are transport compression, not this deployment quantisation.
- **Open point:** an FL server cannot recalibrate on raw TRAINING data. Options for later are fixed scales stored with the contract, data-free scales derived from the weights, or range statistics shared by the sites.

## 10. Not claimed

- No ESP32 flash, RAM, latency or energy figure, and no "improvement", until measured on the device. The ESP32 variant is still to be confirmed by Member 1, so the firmware stays portable C99 with no ESP-IDF code.
- No electrical fault confirmation and no relay isolation.
- No field validation; dataset provenance is UNRESOLVED (`DATA_PROVENANCE.md`).

## 11. Workflow

1. **Commit 1:** code, tests, firmware sources, this protocol, the architecture update. No official run.
2. **Official run (Colab, once):** `notebooks/colab_quantization.ipynb` writes `models/int8/quant_params.json` and its manifest, `firmware/pv_fault/generated/*.h`, `tests/host_c/golden_vectors.h` and `results/quantization_report.{json,md}`. Existing output is never overwritten.
3. **Review**, then commit those files unchanged. After that, `tests/test_codegen.py::test_official_int8_artifacts_and_c_tests` checks the manifest hashes and runs the host C tests on the committed headers.

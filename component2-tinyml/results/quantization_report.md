# INT8 quantisation report (TRAINING window only)

Quantisation `int8-v1`, created 2026-10-16T19:46:32Z. Model `c2-local-v1`, contract SHA-256 `752f62285cca62b59a629686fd5b941b748a811c78808751c04595ee3452cb2a`, weights SHA-256 `94e7ab9384a1906ec2cf0d3176b435e700f15cd99552fbc6c71519c648ecc3a7`. Code `ef6e9ae`. Environment {'python': '3.13.16', 'numpy': '2.1.3', 'pandas': '2.2.3', 'sklearn': '1.6.1'}.

System: **Float32 preprocessing + INT8 neural inference**. The float contract is unchanged; the INT8 parameters are a deployment derivative. **OPERATION was not used.**

## Scales

A_x = 4.488935, s_x = 0.03534594, s_W1 = 0.01053759, s_W2 = 0.00879529, A_z1 = 6.152358, s_z1 = 0.04844377, s_h = 1/127; M1 = 2113405489, SH1 = 38; INV_SX = 0x1.c4ab2e0000000p+4, OUT_SCALE = 0x1.2279260000000p-14.

Accumulator bounds: {'acc1': 71507, 'acc2': 165764, 'acc1_times_m1': 151123286301923} (int32 limit 2147483647; int64 rescaling limit 2^62).

## Float32 device preprocessing check

{'status_identical': True, 'max_abs_z_difference': 3.039836883544922e-06, 'ok_rows': 2924, 'skipped_rows': 76}

## INT8 vs float reference (TRAINING status-OK rows)

Rows 2924, agreement 1.0000, changed predictions 0, changes involving OPEN/SHORT 0 {'OPEN_CIRCUIT': {'changed_to': 0, 'changed_from': 0}, 'SHORT_CIRCUIT': {'changed_to': 0, 'changed_from': 0}}.

Change pairs: none. Per-class agreement: {'NORMAL': 1.0, 'PARTIAL_SHADING': 1.0, 'DUST': 1.0, 'OPEN_CIRCUIT': 1.0, 'SHORT_CIRCUIT': 1.0}.

Training-fit macro-F1: float 1.0000, INT8 1.0000, drop 0.0000.

Input clips 0 (rows 0); tanh lookup clips 0.

Float32 device path vs float reference: agreement 1.0000, changes 0.

## Golden vectors and host C tests

{'z_cases': {'TRAINING_DERIVED': 100, 'TRAINING_DERIVED_REFERENCE_PROBE': 5}, 'raw_cases': {'SYNTHETIC_PROTOTYPE_FROM_TRAINING_MEDIANS': 50, 'SYNTHETIC_EDGE_CASE': 73}, 'raw_status_counts': {0: 69, 1: 4, 2: 2, 3: 48}, 'rshift': 151, 'quantize_input': 28, 'argmax': 6, 'core': 21, 'raw_cases_without_exact_label_check': []}

Host C tests: ran True, passed True; compiler gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0; output: `ALL PASSED 2920`.

## Acceptance (TRAINING only)

- safety_changes_zero: PASS
- agreement_at_least_0.995: PASS
- macro_f1_drop_at_most_1pp: PASS
- golden_tests_passed: PASS

INT8 accepted: **True**. Deployment candidates: INT8 and Float32 (final choice by on-device measurement). Acceptance is measured on TRAINING rows the model was fitted on; it is not a held-out estimate and does not establish electrical safety.

Sizes (parameter storage only): {'int8_weights_bytes': 208, 'int32_bias_bytes': 84, 'tanh_lut_bytes': 256, 'float32_weight_and_bias_bytes': 916, 'note': 'parameter storage only; ESP32 flash, RAM and latency are not measured'}.

## Caveats

- Calibration and comparison use TRAINING rows only; the model was fitted on these rows.
- OPERATION was not used. INT8 behaviour on held-out or field data is not measured.
- Prediction agreement does not establish electrical safety; OPEN/SHORT outputs stay unconfirmed ML predictions and no relay isolation is implemented.
- No ESP32 memory, latency or energy result is claimed.

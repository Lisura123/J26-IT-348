# Pruning feasibility study E1 (TRAINING window only)

Study `prune-e1-v1`, created 2026-10-16T20:39:55Z. Model `c2-local-v1`, weights SHA-256 `94e7ab9384a1906ec2cf0d3176b435e700f15cd99552fbc6c71519c648ecc3a7`, INT8 `int8-v1`. Code `4f9effe`. Environment {'python': '3.13.16', 'numpy': '2.1.3', 'pandas': '2.2.3', 'sklearn': '1.6.1'}.

Rows: 3000 TRAINING, 2924 status-OK compared, 76 skipped by the input gates. **OPERATION was not used. No retraining or fine-tuning. No model artifact was written.**

Frozen INT8 vs frozen Float32 agreement: 1.0000.

## Prediction agreement with the frozen Float32 model

| variant | hidden | non-zero W (f32 / int8) | f32 agree | f32 changes | f32 OPEN/SHORT | int8 agree | int8 changes | int8 OPEN/SHORT | int8 vs frozen int8 | E1 f32 | E1 int8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 16 | 208 / 206 | 1.0000 | 0 | 0 | 1.0000 | 0 | 0 | 1.0000 | PASS | PASS |
| unstructured_10pct | 16 | 187 / 187 | 1.0000 | 0 | 0 | 1.0000 | 0 | 0 | 1.0000 | PASS | PASS |
| unstructured_20pct | 16 | 166 / 166 | 1.0000 | 0 | 0 | 1.0000 | 0 | 0 | 1.0000 | PASS | PASS |
| unstructured_30pct | 16 | 146 / 146 | 1.0000 | 0 | 0 | 1.0000 | 0 | 0 | 1.0000 | PASS | PASS |
| structured_minus2 | 14 | 182 / 180 | 1.0000 | 0 | 0 | 1.0000 | 0 | 0 | 1.0000 | PASS | PASS |
| structured_minus3 | 13 | 169 / 167 | 1.0000 | 0 | 0 | 1.0000 | 0 | 0 | 1.0000 | PASS | PASS |
| structured_minus5 | 11 | 143 / 142 | 1.0000 | 0 | 0 | 1.0000 | 0 | 0 | 1.0000 | PASS | PASS |

## Training-fit macro-F1 (frozen Float32 vs variant; not independent accuracy)

| variant | frozen f32 | pruned f32 | drop | pruned int8 | drop |
|---|---|---|---|---|---|
| baseline | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| unstructured_10pct | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| unstructured_20pct | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| unstructured_30pct | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| structured_minus2 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| structured_minus3 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| structured_minus5 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |

## Changed OPEN/SHORT predictions and change pairs

- baseline (pruned_float_vs_frozen_float): OPEN/SHORT {'OPEN_CIRCUIT': {'changed_to': 0, 'changed_from': 0}, 'SHORT_CIRCUIT': {'changed_to': 0, 'changed_from': 0}}; pairs none
- baseline (pruned_int8_vs_frozen_float): OPEN/SHORT {'OPEN_CIRCUIT': {'changed_to': 0, 'changed_from': 0}, 'SHORT_CIRCUIT': {'changed_to': 0, 'changed_from': 0}}; pairs none
- unstructured_10pct (pruned_float_vs_frozen_float): OPEN/SHORT {'OPEN_CIRCUIT': {'changed_to': 0, 'changed_from': 0}, 'SHORT_CIRCUIT': {'changed_to': 0, 'changed_from': 0}}; pairs none
- unstructured_10pct (pruned_int8_vs_frozen_float): OPEN/SHORT {'OPEN_CIRCUIT': {'changed_to': 0, 'changed_from': 0}, 'SHORT_CIRCUIT': {'changed_to': 0, 'changed_from': 0}}; pairs none
- unstructured_20pct (pruned_float_vs_frozen_float): OPEN/SHORT {'OPEN_CIRCUIT': {'changed_to': 0, 'changed_from': 0}, 'SHORT_CIRCUIT': {'changed_to': 0, 'changed_from': 0}}; pairs none
- unstructured_20pct (pruned_int8_vs_frozen_float): OPEN/SHORT {'OPEN_CIRCUIT': {'changed_to': 0, 'changed_from': 0}, 'SHORT_CIRCUIT': {'changed_to': 0, 'changed_from': 0}}; pairs none
- unstructured_30pct (pruned_float_vs_frozen_float): OPEN/SHORT {'OPEN_CIRCUIT': {'changed_to': 0, 'changed_from': 0}, 'SHORT_CIRCUIT': {'changed_to': 0, 'changed_from': 0}}; pairs none
- unstructured_30pct (pruned_int8_vs_frozen_float): OPEN/SHORT {'OPEN_CIRCUIT': {'changed_to': 0, 'changed_from': 0}, 'SHORT_CIRCUIT': {'changed_to': 0, 'changed_from': 0}}; pairs none
- structured_minus2 (pruned_float_vs_frozen_float): OPEN/SHORT {'OPEN_CIRCUIT': {'changed_to': 0, 'changed_from': 0}, 'SHORT_CIRCUIT': {'changed_to': 0, 'changed_from': 0}}; pairs none
- structured_minus2 (pruned_int8_vs_frozen_float): OPEN/SHORT {'OPEN_CIRCUIT': {'changed_to': 0, 'changed_from': 0}, 'SHORT_CIRCUIT': {'changed_to': 0, 'changed_from': 0}}; pairs none
- structured_minus3 (pruned_float_vs_frozen_float): OPEN/SHORT {'OPEN_CIRCUIT': {'changed_to': 0, 'changed_from': 0}, 'SHORT_CIRCUIT': {'changed_to': 0, 'changed_from': 0}}; pairs none
- structured_minus3 (pruned_int8_vs_frozen_float): OPEN/SHORT {'OPEN_CIRCUIT': {'changed_to': 0, 'changed_from': 0}, 'SHORT_CIRCUIT': {'changed_to': 0, 'changed_from': 0}}; pairs none
- structured_minus5 (pruned_float_vs_frozen_float): OPEN/SHORT {'OPEN_CIRCUIT': {'changed_to': 0, 'changed_from': 0}, 'SHORT_CIRCUIT': {'changed_to': 0, 'changed_from': 0}}; pairs none
- structured_minus5 (pruned_int8_vs_frozen_float): OPEN/SHORT {'OPEN_CIRCUIT': {'changed_to': 0, 'changed_from': 0}, 'SHORT_CIRCUIT': {'changed_to': 0, 'changed_from': 0}}; pairs none

## Size and compute ESTIMATES (not hardware measurements)

_ESTIMATE, not a hardware measurement: analytical parameter-storage bytes and multiply-add counts. Sparse formats exclude decoding code; no ESP32 flash, RAM, latency or energy was measured._

| variant | stored params (dense) | f32 dense B | f32 bitmap B | f32 index B | int8 dense B | int8 bitmap B | int8 index B | MACs dense | MACs zero-skip (int8) | FL layout |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 229 | 916 | 942 | 1124 | 561 | 585 | 765 | 208 | 206 | unchanged (229 values) |
| unstructured_10pct | 229 | 916 | 858 | 1019 | 561 | 566 | 727 | 208 | 187 | unchanged shape (229 values); FL would need a shared fixed mask |
| unstructured_20pct | 229 | 916 | 774 | 914 | 561 | 545 | 685 | 208 | 166 | unchanged shape (229 values); FL would need a shared fixed mask |
| unstructured_30pct | 229 | 916 | 694 | 814 | 561 | 525 | 645 | 208 | 146 | unchanged shape (229 values); FL would need a shared fixed mask |
| structured_minus2 | 201 | 804 | 827 | 986 | 527 | 548 | 705 | 182 | 180 | BROKEN: 8-14-5 has 201 values, not 229 (experimental only) |
| structured_minus3 | 187 | 748 | 770 | 917 | 510 | 530 | 675 | 169 | 167 | BROKEN: 8-13-5 has 187 values, not 229 (experimental only) |
| structured_minus5 | 159 | 636 | 654 | 779 | 476 | 493 | 617 | 143 | 142 | BROKEN: 8-11-5 has 159 values, not 229 (experimental only) |

INT8 bytes include the 256-byte tanh table and 13 bytes of constants; biases are int32 / float32.

## Conclusion rule

E1 checks 1-2 only. Adoption also needs a measured on-device benefit (criterion 3) and an unchanged FL layout (criterion 4); neither is decided by E1.

## Caveats

- All comparisons use TRAINING rows, on which the model was fitted; agreement and training-fit macro-F1 are not independent accuracy.
- OPERATION was not used and is not used to choose a pruning level.
- Storage and multiply-add figures are estimates; no ESP32 measurement was made.
- Structured variants change the architecture and break the 229-value FL layout (experimental only).
- Prediction agreement does not establish electrical safety; OPEN/SHORT outputs stay unconfirmed ML predictions.

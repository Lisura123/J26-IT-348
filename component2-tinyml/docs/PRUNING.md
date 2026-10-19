# Pruning feasibility study E1 (protocol)

Study version `prune-e1-v1`. This protocol is fixed **before** the official run. Code:
`src/tinyml/prune.py`, `scripts/08_pruning_study.py`, `notebooks/colab_pruning.ipynb`;
tests: `tests/test_prune.py`, `tests/test_notebook_pruning.py` (synthetic data only).

## 1. Question

Can the frozen 8-16-5 tanh MLP (`c2-local-v1`, 229 parameters) be pruned **without
retraining** while keeping its predictions, and would that give a worthwhile saving on
the ESP32? E1 answers the first part on TRAINING rows and gives **estimates** for the
second. It does not decide adoption on its own (section 5).

## 2. Fixed rules

- No retraining, fine-tuning, re-calibration or re-quantisation. Weights, preprocessing,
  quality-gate thresholds, labels and the INT8 parameters (`int8-v1`) are unchanged.
- The frozen artifacts are only read and are verified by pinned SHA-256 before use:
  `models/` (contract, feature config, metadata, manifest, weights) and
  `models/int8/quant_params.json` (`5daa66d2…`) and `models/int8/manifest.json` (`30d073f7…`).
  Pruned variants exist only in memory and in the study report. Nothing is written to `models/`,
  `firmware/` or `tests/host_c/`.
- **TRAINING rows only.** OPERATION rows are dropped straight after loading. OPERATION
  was used once in Phase 9 and is not used to evaluate or choose a pruning level.
- Dataset SHA-256 must be the validated `5fbb9347…421595b`; 3,000 TRAINING rows; the split
  file must agree (`final_training_rows`). Rows that fail the input gates (status not OK)
  are skipped exactly as on the device.
- The levels in section 3 are the only levels run. No level is added or changed after
  results are seen.
- The run is done once, from the approved commit, with the Colab guards (section 7).
  Output is never overwritten.

## 3. Methods and levels

**Unstructured magnitude pruning** (10 %, 20 %, 30 %). The 208 weights of W1 (8×16) and
W2 (16×5) are ranked together by |w|; the `round(fraction × 208)` smallest are set to 0
(21, 42, 62 weights). Ties are broken by position (W1 before W2, row-major), so the
result is deterministic. The 21 biases are kept. The shape stays 8-16-5, so the
229-value parameter layout is unchanged; only some values become 0.

**Structured hidden-unit pruning** (remove 2, 3, 5 units). Unit importance is
`|W1[:, j]|₁ × |W2[j, :]|₁` (input weight mass times output weight mass). The units with the
lowest importance are removed (ties: lower index first) together with their W1 column, b1
entry and W2 row. The model becomes 8-14-5, 8-13-5 or 8-11-5 (201, 187, 159 values).
**Experimental only:** this breaks the 229-value FL layout.

Facts computed from the frozen weights only (no data), recorded before the run:

| item | value |
|---|---|
| \|w\| quartiles (min / Q1 / median / Q3 / max) | 0.003 / 0.162 / 0.369 / 0.740 / 1.338 |
| largest removed \|w\| at 10 / 20 / 30 % | 0.060 / 0.112 / 0.194 |
| unit importance range | 3.28 (unit 14) … 21.19 (unit 15); no unit is unused |
| units removed for −2 / −3 / −5 | {6, 14} / {0, 6, 14} / {0, 1, 6, 7, 14} |
| weights already 0 in frozen INT8 | 2 |

**INT8 variants.** Each pruned variant is also run in INT8, derived from the frozen INT8
parameters **without re-quantisation**: the same scales, M1 = 2113405489, SH1 = 38 and
the same 256-entry tanh table; weights pruned in float are set to 0 in qW1 / qW2, and a
removed unit drops its qW1 column, qb1 entry and qW2 row. The INT8 path is the same
integer arithmetic as `quantize.int8_predict` (tested to give identical labels).

## 4. Measurements (all on status-OK TRAINING rows)

For every variant (baseline, three unstructured, three structured):

- prediction agreement of pruned Float32 vs frozen Float32;
- agreement of pruned INT8 vs frozen Float32, and vs frozen INT8;
- changed predictions involving OPEN_CIRCUIT or SHORT_CIRCUIT (either direction), with
  every change pair;
- training-fit macro-F1 of the frozen model and the variant, and the drop
  (**not** independent accuracy: the model was fitted on these rows);
- parameter count (stored dense) and non-zero weights (float and INT8);
- INT8 input and tanh-table clip counts.

**Estimates, not measurements.** The report also gives analytical numbers, labelled
"ESTIMATE, not a hardware measurement":

- parameter storage in bytes, Float32 and INT8, as dense, bitmap-sparse (1 bit per weight
  position + non-zero values) and index-list-sparse (1-byte position + value per non-zero).
  INT8 includes int32 biases, the 256-byte tanh table and 13 bytes of constants. Decoding
  code is **not** included;
- multiply-adds: dense, and with zero skipping; number of tanh evaluations.

Expected size estimates (from the formulas, independent of data):

| variant | Float32 dense / bitmap B | INT8 dense / bitmap B |
|---|---|---|
| baseline | 916 / – | 561 / – |
| unstructured 30 % | 916 / 694 | 561 / 525 (−36 B at most) |
| structured −2 / −3 / −5 | 804 / 748 / 636 | 527 / 510 / 476 (−17 B per unit) |

No ESP32 flash, RAM, latency or energy is measured in E1. Any saving claimed for the
device needs a later on-device measurement.

## 5. Decision criteria

A pruning variant can be **adopted** for deployment only if all four hold:

1. 0 changed OPEN_CIRCUIT / SHORT_CIRCUIT predictions vs the frozen Float32 model;
2. agreement ≥ 99.5 % vs the frozen Float32 model;
3. a **measured** on-device benefit (flash, RAM, latency or energy) that matters for the
   ESP32 build;
4. the FL parameter layout (229 values, 8-16-5) is unchanged.

E1 checks criteria 1–2 for the Float32 and INT8 forms of each variant (`e1_checks`).
Criteria 3 and 4 are not decided by E1: criterion 3 needs hardware, and structured
variants fail criterion 4 by construction. If E1 fails criteria 1–2 for every variant,
or the estimated savings are too small to matter, the outcome is recorded as
**"evaluated, not adopted"**. A pass in E1 only makes a variant a candidate for a later
on-device check; it does not change the deployed model.

## 6. FL contract

The FL contract is unchanged: `c2-local-v1`, 229 values in `param_layout` order. Unstructured
pruning keeps the layout, but FL rounds would re-fill pruned weights unless every client
shared one fixed mask; this is not implemented. Structured pruning changes the layout and
is experimental only.

## 7. Workflow

1. This protocol and the code are committed (after approval) before the run.
2. In Colab (`notebooks/colab_pruning.ipynb`), run cell by cell:
   Step 1 checks out the pasted `APPROVED_COMMIT` exactly (40 hex characters; must be on
   `component2-tinyml`); Step 2 verifies the frozen Float32 and INT8 files; Step 3 uploads
   the dataset; Step 4 checks its SHA-256; Step 5 runs only after `CONFIRM_PRUNING = True`
   is set by hand and the commit is re-checked; Step 6 shows the report; Step 7 downloads
   `prune_out.zip`.
3. The zip is reviewed; `results/pruning_study.{json,md}` and a run record are committed only
   after approval.

## 8. Caveats

- TRAINING rows only; agreement and training-fit macro-F1 are not independent accuracy.
- Dataset provenance is still unresolved (`docs/DATA_PROVENANCE.md`).
- Storage and multiply-add numbers are estimates; no ESP32 measurement.
- Prediction agreement does not establish electrical safety; OPEN/SHORT outputs stay
  unconfirmed ML predictions.

## 9. Official run record

Sections 1–8 were committed in `4f9effe` before the run and are unchanged. The official
study was run once in Google Colab, cell by cell, on 2026-10-16 (report created
`2026-10-16T20:39:55Z`), from the approved commit `4f9effeb121fbf86abd8e4eb2de2735457007eea`
with `CONFIRM_PRUNING = True` set by hand. The downloaded `prune_out.zip` contained only the
two report files, which are committed byte-identically:

| file | SHA-256 |
|---|---|
| `results/pruning_study.json` | `38fe7c3079bbefaa969e27d210c613e109a70accb6854b9a2e06f198ab8e0c21` |
| `results/pruning_study.md` | `57206d800e66718bca92b5286c3810d6bd47ad0c321ace029143100e4ab3dbf6` |

The Markdown report was checked to be exactly the rendering of the JSON report.

**Provenance.** Study `prune-e1-v1`; code commit `4f9effe`; model `c2-local-v1`, contract
SHA-256 `752f6228…`, weights `94e7ab93…`; INT8 `int8-v1`, `quant_params.json` `5daa66d2…`;
dataset `5fbb9347…421595b`, split `runwise-v1`; Python 3.13.16, numpy 2.1.3, pandas 2.2.3,
scikit-learn 1.6.1. OPERATION was not used (`operation_window_used: false`); no retraining
or fine-tuning; no model artifact was written. Rows: 3,000 TRAINING, 2,924 status-OK
compared, 76 skipped by the input gates.

### 9.1 Results (status-OK TRAINING rows only)

All measurements below are on TRAINING rows, on which the model was fitted. Agreement and
training-fit macro-F1 describe how far the pruned variants reproduce the frozen model on
these rows; **they are not independent classification accuracy.**

| variant | hidden units | stored params (dense) | non-zero W float / INT8 | agreement vs frozen Float32: pruned Float32 / pruned INT8 | pruned INT8 vs frozen INT8 | changed predictions | OPEN/SHORT changes | training-fit macro-F1 (frozen / pruned Float32 / pruned INT8) | E1 criteria 1–2 |
|---|---|---|---|---|---|---|---|---|---|
| original (baseline) | 16 | 229 | 208 / 206 | 1.000 / 1.000 | 1.000 | 0 | 0 | 1.000 / 1.000 / 1.000 | PASS |
| unstructured 10 % | 16 | 229 | 187 / 187 | 1.000 / 1.000 | 1.000 | 0 | 0 | 1.000 / 1.000 / 1.000 | PASS |
| unstructured 20 % | 16 | 229 | 166 / 166 | 1.000 / 1.000 | 1.000 | 0 | 0 | 1.000 / 1.000 / 1.000 | PASS |
| unstructured 30 % | 16 | 229 | 146 / 146 | 1.000 / 1.000 | 1.000 | 0 | 0 | 1.000 / 1.000 / 1.000 | PASS |
| structured −2 (units 6, 14) | 14 | 201 | 182 / 180 | 1.000 / 1.000 | 1.000 | 0 | 0 | 1.000 / 1.000 / 1.000 | PASS |
| structured −3 (units 0, 6, 14) | 13 | 187 | 169 / 167 | 1.000 / 1.000 | 1.000 | 0 | 0 | 1.000 / 1.000 / 1.000 | PASS |
| structured −5 (units 0, 1, 6, 7, 14) | 11 | 159 | 143 / 142 | 1.000 / 1.000 | 1.000 | 0 | 0 | 1.000 / 1.000 / 1.000 | PASS |

No prediction changed for any variant, in Float32 or INT8, so there were no changes to or
from OPEN_CIRCUIT or SHORT_CIRCUIT. INT8 input clips and tanh-table clips were 0 for every
variant. The frozen INT8 model agreed with the frozen Float32 model on all 2,924 rows, as
in the quantisation run (`docs/QUANTIZATION.md` §12). The removed units and the number of
removed weights are those recorded in section 3 before the run.

The model reproduces its own TRAINING predictions perfectly even after these cuts. This
shows spare capacity on the rows it was fitted on; it does not show how the pruned
variants behave on readings near class boundaries or on new field data (DUST was the
weakest class in Phase 9).

### 9.2 Size and compute ESTIMATES (not hardware measurements)

Analytical parameter storage in bytes. INT8 includes int32 biases, the 256-byte tanh table
and 13 bytes of constants. **Sparse figures exclude the decoder implementation (code and
any working buffer) needed to read the bitmap or index list**, so they are lower bounds
for a sparse build. Multiply-adds are per inference; "zero skip" assumes the kernel skips
zero weights, which the current firmware does not do.

| variant | Float32 dense / bitmap / index B | INT8 dense / bitmap / index B | multiply-adds dense / zero skip (INT8) | tanh evaluations |
|---|---|---|---|---|
| original | 916 / 942 / 1124 | 561 / 585 / 765 | 208 / 206 | 16 |
| unstructured 10 % | 916 / 858 / 1019 | 561 / 566 / 727 | 208 / 187 | 16 |
| unstructured 20 % | 916 / 774 / 914 | 561 / 545 / 685 | 208 / 166 | 16 |
| unstructured 30 % | 916 / 694 / 814 | 561 / 525 / 645 | 208 / 146 | 16 |
| structured −2 | 804 / 827 / 986 | 527 / 548 / 705 | 182 / 180 | 14 |
| structured −3 | 748 / 770 / 917 | 510 / 530 / 675 | 169 / 167 | 13 |
| structured −5 | 636 / 654 / 779 | 476 / 493 / 617 | 143 / 142 | 11 |

Largest estimated INT8 parameter saving against the 561-byte dense INT8 model: 36 B for
unstructured 30 % (bitmap, before decoder overhead) and 85 B for structured −5 (dense).
No ESP32 flash, RAM, latency or energy was measured.

### 9.3 FL contract

The FL contract is a flat vector of 229 values in `param_layout` order for an 8-16-5
network (W1 128, b1 16, W2 80, b2 5). Structured pruning removes hidden units, so W1, b1
and W2 shrink (8-14-5: 201 values, 8-13-5: 187, 8-11-5: 159). The vector then has a
different length and the positions of the remaining values move, so it can no longer be
averaged with 229-value client updates or loaded as `c2-local-v1`; it breaks the contract.
Unstructured pruning keeps the 229-value shape, but FL averaging would re-fill pruned
weights unless all clients shared one fixed mask, which is not implemented. The FL
contract is unchanged by this study.

### 9.4 Decision

Against section 5: criteria 1–2 pass for every variant on TRAINING rows; criterion 3 (a
measured on-device benefit) is not shown, as no ESP32 measurement exists and the estimated
savings are at most 36–85 B of parameter storage; criterion 4 fails for every structured
variant and holds for unstructured variants only without FL mask support.

**Pruning was experimentally evaluated but not adopted for deployment.** This is a
provisional deployment decision because no ESP32 measurements have been performed. It can
be revisited if on-device measurements show a real flash, RAM, latency or energy limit.
The frozen Float32 model (`models/`, `c2-local-v1`) and the frozen INT8 model
(`models/int8/`, `int8-v1`) remain the deployment references; neither was changed.

### 9.5 Limitations

- TRAINING rows only; the model was fitted on them. Agreement and training-fit macro-F1
  are not independent accuracy, and robustness of pruned variants on unseen data is not
  measured. OPERATION was not used and will not be used to choose a pruning level.
- Storage and multiply-add figures are analytical estimates; sparse figures exclude decoder
  overhead; no ESP32 measurement was made.
- Dataset provenance is still unresolved (`docs/DATA_PROVENANCE.md`); results describe this
  dataset, not verified lab or field data.
- Prediction agreement does not establish electrical safety; OPEN/SHORT outputs stay
  unconfirmed ML predictions, and no relay or actuation is implemented.

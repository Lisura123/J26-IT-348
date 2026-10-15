# PV Feature Layer (`pv8-v1`)

Implementation: `src/tinyml/features.py` · Tests: `tests/test_features.py` · Check: `scripts/03_validate_features.py` → `results/feature_validation.md`

The feature layer turns one raw reading of the PV sensor dataset provided by Member 1 into the 8 inputs of the 8-16-5 TinyML model. The same code is imported by local scripts and by the Colab notebook. The ESP32 implementation reproduces the same arithmetic from the saved configuration.

**Status:** implemented and unit-tested. Whether these features improve fault classification has **not** been evaluated yet; that is the baseline comparison (raw-8 vs physics-8) in the training phase.

---

## 1. Inputs

Raw columns used (the same 8 raw columns Member 3's FL model uses, `contract.FL_RAW_FEATURES`):
`voltage_v` (V), `current_ma` (mA), `power_mw` (mW), `panel_temp1_c`, `panel_temp2_c`, `ambient_temp_c` (°C), `irradiance_lux` (lx), `dust_vo_v` (V).

Never used as features: `label`, `run_id`, `site_id`, `record_id`, `timestamp_utc`, `window`, `health_score`, `humidity_pct`, `pressure_hpa`.

---

## 2. Features (fixed order)

Notation: `klx = irradiance_lux / 1000`, `T_p = (panel_temp1_c + panel_temp2_c) / 2`. Ranges are the observed min–max on the TRAINING window (status-OK rows), not design limits.

| # | Feature | Formula | Unit | Observed range | Physical meaning | Fault relevance |
|---|---|---|---|---|---|---|
| 0 | `v_deficit` | `V_ref − voltage_v`, with `V_ref = a + b·ln(klx) + c·(T_p − 25)` | V | −0.60 – 2.72 | how far the voltage is below the voltage of a healthy panel at the same light and temperature | high for SHORT_CIRCUIT and PARTIAL_SHADING, negative for OPEN_CIRCUIT |
| 1 | `i_norm` | `current_ma / klx` | mA/klx | 0 – 1.87 | current per unit of light | low for DUST, PARTIAL_SHADING, ~0 for OPEN_CIRCUIT |
| 2 | `p_norm` | `power_mw / klx` | mW/klx | 0 – 4.52 | power per unit of light (performance-ratio proxy) | drops for every fault class |
| 3 | `dust_vo_v` | raw | V | 0.035 – 1.73 | dust sensor output voltage | high for DUST |
| 4 | `temp_spread` | `|panel_temp1_c − panel_temp2_c|` | °C | 0 – 6.51 | temperature difference across the panel | hotspot under PARTIAL_SHADING |
| 5 | `temp_rise` | `T_p − ambient_temp_c` | °C | 1.13 – 24.07 | panel heating above ambient | heat not converted to power |
| 6 | `voltage_v` | raw | V | 0.02 – 3.30 | measured panel voltage | ≈ 0 for SHORT_CIRCUIT |
| 7 | `current_ma` | raw | mA | 0 – 173.6 | measured panel current | ≈ 0 for OPEN_CIRCUIT |

### Design notes per feature
- **`v_deficit`:** no panel rating is assumed. The reference voltage is learned from the data. `V_ref` is a least-squares fit of `voltage_v` on `[1, ln(klx), T_p − 25]`, using only **NORMAL, status-OK rows of the training part of the current fold**. It represents the operating voltage of a healthy panel under this dataset's load conditions. The log term follows the logarithmic dependence of PV voltage on irradiance (single-diode model). The temperature term follows the negative voltage temperature coefficient. On the CV folds the fitted `c` is between −0.0095 and −0.0154 V/°C, i.e. negative as physics expects (`results/feature_validation.md`). This is an empirical reference, not a datasheet Voc.
- **`i_norm`, `p_norm`:** irradiance stays in **lux**. Kilolux is only a scale change for readable numbers. No lux → W/m² conversion is applied: the group convention `lux / 120` is not a calibration, and its constant would cancel in the scaler anyway.
- **`dust_vo_v`:** used as the raw sensor output voltage. It is **not** a dust concentration: no calibration curve is available.
- **`voltage_v`, `current_ma`:** kept raw so that open/short circuits remain obvious to the model even when ratios are small.

---

## 3. Input gating (applied identically in training, evaluation and on the ESP32)

| Status | Code | Condition (checked in this order) | Model input? | Prediction status |
|---|---|---|---|---|
| `INVALID_INPUT` | 3 | any of the 8 raw values missing, non-numeric, NaN/±inf; DS18B20 code −127/−999; value outside `input_limits` | no | `SKIPPED_INVALID_INPUT` |
| `LUX_SATURATED` | 2 | `irradiance_lux ≥ lux_saturation` (100,000) | no | `SKIPPED_LUX_SATURATED` |
| `LOW_LIGHT` | 1 | `irradiance_lux < low_light_lux` (2,400) | no | `SKIPPED_LOW_LIGHT` |
| `OK` | 0 | otherwise | yes | `CLASSIFIED` |

- **Rows are never deleted.** `input_status()` labels every row. Gated rows get NaN features and are excluded from `X`, and the counts are reported.
- **Division by zero and log(0) cannot happen** for OK rows, because `klx ≥ 2.4` after gating.
- After scaling, `transform()` asserts that every accepted input is finite.
- **Effect on TRAINING:** 76 rows are `LUX_SATURATED` (49 DUST, 27 PARTIAL_SHADING); no LOW_LIGHT or INVALID rows. Those 76 readings are not used to fit or train. In deployment, the decision layer (architecture §7.4) still applies the OPEN/SHORT rule guards to gated readings.
- `input_limits` reuse the physical limits from `validate.py`. They come from an **assumed** panel datasheet and are a sanity check, not a specification.

---

## 4. Scaling and leakage prevention

- Stage A (physical features) and stage B (scaling `z = (x − mean) / std`, population std, std < 1e-9 → 1) are separate.
- **Everything learned (`V_ref` a, b, c and the scaler mean/std) is fitted on the training rows of the current fold only.** `fold_matrices(df, train_idx, test_idx)` fits on `train_idx` and applies the result to `test_idx` without refitting. A test proves that changing the test rows does not change the fitted values.
- Folds come from the Phase 4 split file (`results/splits_runwise_v1.json`) via `iter_folds()`. No new splitting is done here.
- `scripts/03_validate_features.py` reads only the TRAINING window. **The OPERATION window is not used during feature development.**
- `V_ref` uses the label (NORMAL) of training rows to choose which rows define "healthy". This is allowed because it uses training labels only, inside the fold.

---

## 5. Saved configuration (`FeatureConfig`, JSON)

```json
{
  "version": "pv8-v1",
  "features": ["v_deficit","i_norm","p_norm","dust_vo_v","temp_spread","temp_rise","voltage_v","current_ma"],
  "units": ["V","mA/klx","mW/klx","V","degC","degC","V","mA"],
  "dtype": "float32",
  "lux_per_unit": 1000.0,
  "low_light_lux": 2400.0,
  "lux_saturation": 100000.0,
  "t_ref_c": 25.0,
  "temp_sentinels": [-127.0, -999.0],
  "input_limits": {"voltage_v": [0.0, 4.0], "...": "..."},
  "vref": {"a": "<fitted>", "b": "<fitted>", "c": "<fitted>"},
  "scaler": {"mean": ["<8 fitted>"], "std": ["<8 fitted>"]},
  "fitted_on": {"fold": "...", "train_rows": 0, "ok_rows": 0, "normal_rows_for_vref": 0}
}
```
Loading a config whose version or feature order differs from the code raises an error. The fitted config becomes part of `model_contract.json` when the model is trained.

---

## 6. ESP32 implementation requirements
- float32 arithmetic; `logf()` once per reading; `fabsf()`; 8 multiply-adds for scaling.
- Same gating order and thresholds as §3. Status codes 0–3 are passed to the decision layer.
- The constants and fitted values are generated from the saved JSON into a C header. They are never typed by hand.
- `power_mw` must be the measured (or V×I computed) power in mW, matching the dataset definition.

---

## 7. Unresolved calibration dependencies (placeholders)

| Item | Current value | Why it is a placeholder | Resolution |
|---|---|---|---|
| `low_light_lux` | 2,400 lx | from architecture §5; no TRAINING row is below it, so its effect is untested | tune with validation folds; confirm with Member 1's lux sensor |
| `lux_saturation` | 100,000 lx | value seen in the data; the lux sensor model is undocumented (a BH1750 would saturate at 65,535 lx) | sensor part number from Member 1 |
| `input_limits` | from the assumed panel datasheet | panel rating not confirmed | real datasheet |
| `dust_vo_v` meaning | raw voltage | no dust-sensor calibration | calibration curve, if needed |
| `V_ref` | fitted per fold | depends on the load/operating point of the experiments | document the load used in the experiments (Member 1) |

---

## 8. Use from Colab

```python
!git clone -b component2-tinyml https://github.com/Lisura123/J26-IT-348
import sys; sys.path.insert(0, "J26-IT-348/component2-tinyml/src")
import pandas as pd
from tinyml.splits import load_assignments, iter_folds
from tinyml.features import fold_matrices

df = pd.read_csv("pv_fault_dataset.csv")
assignments, meta = load_assignments("J26-IT-348/component2-tinyml/results/splits_runwise_v1.json", df)
for name, tr, te in iter_folds(df, assignments, "cv"):
    m = fold_matrices(df, tr, te, fold=name)   # m["train"]["X"], m["train"]["y"], m["config"]
```

---

## 9. Phase 5 Technical Review Findings

Review of the feature layer before the baseline comparison. All figures below were recomputed during the review. TRAINING figures use feature values. For OPERATION, only the input status (gate result) was counted: no feature values or residuals were computed on OPERATION rows.

### Finding A: a whole OPERATION DUST run is skipped by the low-light gate
- Run `SITE-D-DUS-12` (OPERATION window, DUST) has 60 readings with irradiance 683–1,306 lx. All 60 are below the 2,400 lx low-light threshold and get status `LOW_LIGHT`.
- In the TRAINING → OPERATION test, 9 of the 10 DUST test runs can therefore be classified. The same run is skipped in the leave-one-site-out fold for SITE-D.
- The threshold cannot be tuned on TRAINING data (no TRAINING reading is below 2,400 lx), and tuning it on OPERATION would leak the test set.
- **Requirement for the final evaluation report.** It must show, and must not silently drop skipped readings from:
  - total readings, classified readings and skipped readings;
  - coverage per class and per run;
  - the reason for every skipped prediction (`LOW_LIGHT`, `LUX_SATURATED`, `INVALID_INPUT`).

  Classification metrics are computed on classified readings and always reported next to coverage.

Readings excluded by the gates, per protocol (rows, by class):

| Protocol / fold | Train side excluded | Test side excluded |
|---|---|---|
| CV cv0, cv1, cv3 | PAR 27, DUS 49 (saturated) | none |
| CV cv2 | PAR 27 | DUS 49 (saturated) |
| CV cv4 | DUS 49 | PAR 27 (saturated) |
| Temporal | PAR 27, DUS 49 (saturated) | DUS 60 (low light, `SITE-D-DUS-12`) |
| LOSO SITE-A, B, C, F | PAR 27, DUS 109 | none |
| LOSO SITE-D | PAR 27, DUS 49 | DUS 60 (low light) |
| LOSO SITE-E | DUS 60 | PAR 27, DUS 49 (saturated) |

Excluded rows come from three runs only: `SITE-E-DUS-17` (49 saturated), `SITE-E-PAR-05` (27 saturated) and `SITE-D-DUS-12` (60 low light). No `INVALID_INPUT` rows occur. Every model sees the same gating, so model comparisons stay like-for-like. Because the excluded rows are all DUST or PARTIAL_SHADING, coverage must be reported per class.

### Finding B: the voltage reference generalises imperfectly across sites
Residual `v_deficit` on NORMAL rows (should be ≈ 0 for a healthy panel), TRAINING window only:

| Check | Fitted rows: RMSE | Held-out rows: bias / RMSE |
|---|---|---|
| CV folds (5) | 0.021 – 0.057 V | bias −0.051 … +0.077 V, RMSE 0.028 – 0.116 V |
| Leave-one-site-out within TRAINING, SITE-C held out | 0.020 V | **bias +0.163 V, RMSE 0.164 V** (60 rows) |
| Leave-one-site-out within TRAINING, other sites with NORMAL rows | 0.055 – 0.064 V | bias −0.038 … +0.015 V |

- For comparison, the out-of-fold median `v_deficit` of PARTIAL_SHADING is 0.335 V. A +0.16 V bias at an unseen site could therefore move NORMAL readings towards PARTIAL_SHADING.
- This points to panel/site differences that a single voltage reference does not capture. It does not show that the feature is wrong.
- **Requirement for the baseline comparison:** include a feature ablation that compares the same models **with and without `v_deficit`** (and raw-8 vs physics-8) under run-wise CV and leave-one-site-out. Report whether `v_deficit` improves or reduces macro-F1 and per-class recall. Keep or drop the feature based on that result.

### Finding C: lux sensor not confirmed; thresholds are provisional
- The lux sensor model is not documented. The **100,000 lx saturation threshold** and the **2,400 lx low-light threshold** are provisional assumptions, not sensor specifications.
- Near-saturation observations in the TRAINING window (3,000 rows):
  - 76 rows are at exactly 100,000 lx (gated).
  - **52 rows** are between 95,000 and 99,999 lx and are **not** gated (PARTIAL_SHADING 33, DUST 11, SHORT_CIRCUIT 8). They come from runs `SITE-E-PAR-05`, `SITE-E-DUS-17` and `SITE-E-SHO-15`, the same runs that also contain saturated readings. They may be partly affected by saturation.
  - 776 rows are above 65,535 lx, the maximum of a BH1750 sensor. They are only physically possible if a different sensor or range setting was used.
- **Needed from Member 1:** lux sensor part number, measurement range/mode, saturation behaviour (clipped value or error code), and the device's sampling settings. When received, revise `lux_saturation` and `low_light_lux` and re-run the gate counts.

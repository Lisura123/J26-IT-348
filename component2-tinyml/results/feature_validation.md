# Feature layer check (TRAINING window only)

Feature version `pv8-v1`. Splits `runwise-v1` (seed 42). The OPERATION window was not read for this report. No model was trained.

## Input status on TRAINING rows

| Label | OK | LOW_LIGHT | LUX_SATURATED | INVALID_INPUT |
|---|---|---|---|---|
| NORMAL | 600 | 0 | 0 | 0 |
| PARTIAL_SHADING | 573 | 0 | 27 | 0 |
| DUST | 551 | 0 | 49 | 0 |
| OPEN_CIRCUIT | 600 | 0 | 0 | 0 |
| SHORT_CIRCUIT | 600 | 0 | 0 | 0 |

Gated rows are kept in the dataset and excluded from model inputs (low-light threshold 2400 lx, saturation 100000 lx; both placeholders).

## Per-fold fit (fit on fold-train runs, applied to fold-validation runs)

| Fold | V_ref a (V) | b (V/ln klx) | c (V/°C) | Train OK rows | Val OK rows | Val gated | Finite |
|---|---|---|---|---|---|---|---|
| cv0 | 2.5645 | 0.0711 | -0.0095 | 2324 | 600 | 0 | True |
| cv1 | 2.5815 | 0.0769 | -0.0115 | 2324 | 600 | 0 | True |
| cv2 | 2.5175 | 0.0999 | -0.0138 | 2373 | 551 | 49 | True |
| cv3 | 2.5502 | 0.0809 | -0.0105 | 2324 | 600 | 0 | True |
| cv4 | 2.4911 | 0.1188 | -0.0154 | 2351 | 573 | 27 | True |

## Out-of-fold physical features per class (median, before scaling)

Each value comes from the fold where that row was in the validation part.

| Label | v_deficit (V) | i_norm (mA/klx) | p_norm (mW/klx) | dust_vo_v (V) | temp_spread (degC) | temp_rise (degC) | voltage_v (V) | current_ma (mA) |
|---|---|---|---|---|---|---|---|---|
| NORMAL | -0.016 | 1.454 | 3.794 | 0.141 | 0.500 | 8.160 | 2.737 | 53.760 |
| PARTIAL_SHADING | 0.335 | 0.767 | 1.750 | 0.149 | 2.280 | 11.435 | 2.346 | 29.570 |
| DUST | 0.013 | 1.001 | 2.704 | 1.006 | 0.540 | 8.435 | 2.689 | 38.290 |
| OPEN_CIRCUIT | -0.538 | 0.004 | 0.012 | 0.136 | 0.640 | 4.993 | 3.258 | 0.105 |
| SHORT_CIRCUIT | 2.605 | 1.655 | 0.103 | 0.113 | 0.490 | 13.410 | 0.062 | 88.120 |

## Example: record `REC-00301` (validation row of cv0)

Raw: voltage_v=2.777, current_ma=64.15, power_mw=178.18, panel_temp1_c=37.23, panel_temp2_c=36.6, ambient_temp_c=26.18, irradiance_lux=43867.0, dust_vo_v=0.062

| Feature | Physical value | Unit | Scaled (float32) |
|---|---|---|---|
| v_deficit | -0.0563 | V | -0.4927 |
| i_norm | 1.4624 | mA/klx | 0.8059 |
| p_norm | 4.0618 | mW/klx | 1.5970 |
| dust_vo_v | 0.0620 | V | -0.6320 |
| temp_spread | 0.6300 | degC | -0.3140 |
| temp_rise | 10.7350 | degC | 0.0550 |
| voltage_v | 2.7770 | V | 0.5167 |
| current_ma | 64.1500 | mA | 0.4133 |

These are descriptive checks only. Whether the features improve classification is evaluated in the baseline comparison (Phase 7), not here.

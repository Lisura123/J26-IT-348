"""PV-specific feature layer (Phase 5).

Turns one raw sensor reading into the 8 model inputs, in this order:

  0 v_deficit    V        V_ref(lux, T_panel) - V
  1 i_norm       mA/klx   I / (lux / 1000)
  2 p_norm       mW/klx   P / (lux / 1000)
  3 dust_vo_v    V        dust sensor output voltage (raw, uncalibrated)
  4 temp_spread  degC     |T1 - T2|
  5 temp_rise    degC     (T1 + T2) / 2 - T_amb
  6 voltage_v    V        measured panel voltage
  7 current_ma   mA       measured panel current

Two separate stages, both reproducible in C on the ESP32:
  A. physical features: closed-form arithmetic plus one logf()
  B. learned scaling:   z = (x - mean) / std

Learned values (the V_ref coefficients and the scaler) are fitted only on
the training rows of a fold and stored in a JSON config together with the
feature order, constants and input-gating policy.

Irradiance is used in lux (per kilolux) on purpose: no lux -> W/m^2
conversion is applied, because no calibration for the lux sensor exists.
"""
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .contract import FL_RAW_FEATURES, LABEL_COLUMN
from .validate import LUX_SATURATION, PHYSICAL_LIMITS, TEMP_SENTINELS

FEATURE_VERSION = "pv8-v1"
FEATURES = ("v_deficit", "i_norm", "p_norm", "dust_vo_v",
            "temp_spread", "temp_rise", "voltage_v", "current_ma")
FEATURE_UNITS = ("V", "mA/klx", "mW/klx", "V", "degC", "degC", "V", "mA")
RAW_INPUTS = FL_RAW_FEATURES          # the 8 raw sensor columns the features need

# Input status, in order of precedence. Integer codes are used on the ESP32.
STATUS_OK = "OK"
STATUS_INVALID = "INVALID_INPUT"
STATUS_SATURATED = "LUX_SATURATED"
STATUS_LOW_LIGHT = "LOW_LIGHT"
STATUS_CODES = {STATUS_OK: 0, STATUS_LOW_LIGHT: 1, STATUS_SATURATED: 2, STATUS_INVALID: 3}
# Architecture prediction_status for each input status
PREDICTION_STATUS = {STATUS_OK: "CLASSIFIED", STATUS_LOW_LIGHT: "SKIPPED_LOW_LIGHT",
                     STATUS_SATURATED: "SKIPPED_LUX_SATURATED",
                     STATUS_INVALID: "SKIPPED_INVALID_INPUT"}


@dataclass
class FeatureConfig:
    """Everything needed to reproduce the 8 inputs from raw readings.

    Constants marked PLACEHOLDER are not calibrated; see docs/FEATURES.md.
    """
    version: str = FEATURE_VERSION
    features: tuple = FEATURES
    units: tuple = FEATURE_UNITS
    dtype: str = "float32"
    lux_per_unit: float = 1000.0                     # klx: scaling only, not a calibration
    low_light_lux: float = 2400.0                    # PLACEHOLDER (architecture section 5)
    lux_saturation: float = float(LUX_SATURATION)    # PLACEHOLDER (sensor model unknown)
    t_ref_c: float = 25.0                            # reference temperature for V_ref
    temp_sentinels: tuple = TEMP_SENTINELS
    input_limits: dict = field(default_factory=lambda: {
        c: [lo, hi] for c, (lo, hi, _) in PHYSICAL_LIMITS.items() if c in RAW_INPUTS})
    # learned, fold-specific (None until fitted)
    vref: dict = None                                # {"a": V, "b": V per ln(klx), "c": V/degC}
    scaler: dict = None                              # {"mean": [8], "std": [8]}
    fitted_on: dict = None                           # provenance of the fit

    @property
    def is_fitted(self):
        return self.vref is not None and self.scaler is not None

    def to_json(self):
        d = asdict(self)
        return json.dumps(d, indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        for k in ("features", "units", "temp_sentinels"):
            d[k] = tuple(d[k])
        cfg = cls(**d)
        if cfg.features != FEATURES or cfg.version != FEATURE_VERSION:
            raise ValueError(f"config is {cfg.version} {cfg.features}; this code is "
                             f"{FEATURE_VERSION} {FEATURES}")
        return cfg

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json() + "\n")

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_json(f.read())


# ---------------------------------------------------------------- stage A

# [GUIDE] C. Input quality gates (same rules in training, evaluation and on the ESP32)
# WHAT: labels every reading OK / LOW_LIGHT / LUX_SATURATED / INVALID_INPUT.
# WHY:  the model must not guess on readings it cannot interpret (dark, saturated light
#       sensor, missing or impossible values). Rows are never deleted; they are skipped
#       and counted (coverage is reported next to every metric).
# NOTE: 2,400 lx and 100,000 lx are PLACEHOLDER thresholds (docs/FEATURES.md section 7).
# EVIDENCE: results/feature_validation.md; results/operation_eval.md (60 LOW_LIGHT readings skipped).
def input_status(df, cfg):
    """OK / LOW_LIGHT / LUX_SATURATED / INVALID_INPUT for every row.

    Rows are never dropped; callers use the status to decide what to do.
    Precedence: INVALID > SATURATED > LOW_LIGHT > OK.
    """
    missing = [c for c in RAW_INPUTS if c not in df.columns]
    if missing:
        raise KeyError(f"missing raw sensor columns: {missing}")
    raw = df[list(RAW_INPUTS)].apply(pd.to_numeric, errors="coerce").astype(float)
    invalid = ~np.isfinite(raw).all(axis=1)
    for c in ("panel_temp1_c", "panel_temp2_c"):
        invalid |= raw[c].isin(cfg.temp_sentinels)
    for c, (lo, hi) in cfg.input_limits.items():
        invalid |= (raw[c] < lo) | (raw[c] > hi)
    lux = raw["irradiance_lux"]
    status = np.select([invalid, lux >= cfg.lux_saturation, lux < cfg.low_light_lux],
                       [STATUS_INVALID, STATUS_SATURATED, STATUS_LOW_LIGHT], default=STATUS_OK)
    return pd.Series(status, index=df.index, name="input_status")


def _klux(df, cfg):
    return df["irradiance_lux"].astype(float) / cfg.lux_per_unit


def _panel_temp(df):
    return (df["panel_temp1_c"].astype(float) + df["panel_temp2_c"].astype(float)) / 2.0


def vref_design(df, cfg):
    """Columns [1, ln(klx), T_panel - T_ref] for the voltage reference."""
    return np.column_stack([np.ones(len(df)), np.log(_klux(df, cfg)),
                            _panel_temp(df) - cfg.t_ref_c])


def physical_features(df, cfg, status=None):
    """Stage A: the 8 features in FEATURES order (float64 DataFrame).

    Rows whose status is not OK get NaN, so they can never reach the model
    by accident. Needs cfg.vref.
    """
    if cfg.vref is None:
        raise ValueError("V_ref is not fitted; use fit_pipeline() on training rows first")
    status = input_status(df, cfg) if status is None else status
    ok = (status == STATUS_OK).to_numpy()
    out = pd.DataFrame(np.nan, index=df.index, columns=list(FEATURES))
    d = df.loc[ok]
    if len(d):
        klx = _klux(d, cfg)
        a, b, c = cfg.vref["a"], cfg.vref["b"], cfg.vref["c"]
        # [GUIDE] C. The 8 PV-specific features (docs/FEATURES.md section 2):
        #  0 v_deficit   = V_ref(light, temperature) - measured V: how far the voltage is below a
        #                healthy panel at the same light and temperature (high for SHORT/SHADING)
        #  1 i_norm      = current per kilolux: current corrected for light level (low for DUST/SHADING)
        #  2 p_norm      = power per kilolux: performance-ratio proxy (drops for every fault)
        #  3 dust_vo_v   = raw dust-sensor voltage (high for DUST; not a calibrated concentration)
        #  4 temp_spread = |T1 - T2| across the panel (hotspot under PARTIAL_SHADING)
        #  5 temp_rise   = mean panel temperature - ambient (heat not converted to power)
        #  6 voltage_v   = raw voltage (about 0 for SHORT_CIRCUIT)
        #  7 current_ma  = raw current (about 0 for OPEN_CIRCUIT)
        # Ratios use kilolux only as a scale; no lux -> W/m^2 calibration is assumed.
        v_ref = a + b * np.log(klx) + c * (_panel_temp(d) - cfg.t_ref_c)
        out.loc[ok, "v_deficit"] = v_ref - d["voltage_v"]
        out.loc[ok, "i_norm"] = d["current_ma"] / klx
        out.loc[ok, "p_norm"] = d["power_mw"] / klx
        out.loc[ok, "dust_vo_v"] = d["dust_vo_v"]
        out.loc[ok, "temp_spread"] = (d["panel_temp1_c"] - d["panel_temp2_c"]).abs()
        out.loc[ok, "temp_rise"] = _panel_temp(d) - d["ambient_temp_c"]
        out.loc[ok, "voltage_v"] = d["voltage_v"]
        out.loc[ok, "current_ma"] = d["current_ma"]
    return out


# ---------------------------------------------------------------- fitting

# [GUIDE] C. Fitting on TRAINING data only (leakage prevention)
# WHAT: learns V_ref (a, b, c) and the scaler mean/std from the rows it is given.
# WHY:  if the scaler saw test rows, test information would leak into training and the
#       test score would be too optimistic. So this is always called with training rows
#       only (fold_matrices below, train.train_final); the test side reuses the fitted values.
# HOW:  V_ref = least squares on NORMAL, status-OK training rows; scaler = mean and
#       population std of the 8 features over status-OK training rows.
# INPUT: training rows.  OUTPUT: a fitted FeatureConfig (saved as models/feature_config.json).
def fit_pipeline(train_df, cfg=None, fold=None):
    """Fit V_ref and the scaler on training rows only; return a fitted config.

    V_ref: least squares of voltage_v on [1, ln(klx), T_panel - 25] over the
    NORMAL, status-OK training rows (expected operating voltage of a healthy
    panel). Scaler: mean/std of the 8 features over status-OK training rows.
    """
    base = cfg or FeatureConfig()
    cfg = FeatureConfig.from_json(base.to_json())
    cfg.vref = cfg.scaler = cfg.fitted_on = None
    status = input_status(train_df, cfg)
    ok = status == STATUS_OK
    normal = ok & (train_df[LABEL_COLUMN] == "NORMAL")
    if normal.sum() < 3:
        raise ValueError("need at least 3 OK NORMAL training rows to fit V_ref")
    d = train_df.loc[normal]
    coef, *_ = np.linalg.lstsq(vref_design(d, cfg), d["voltage_v"].to_numpy(float), rcond=None)
    cfg.vref = {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2])}

    feats = physical_features(train_df, cfg, status).loc[ok]
    mean = feats.mean().to_numpy(float, copy=True)
    std = feats.std(ddof=0).to_numpy(float, copy=True)
    std[~np.isfinite(std) | (std < 1e-9)] = 1.0
    cfg.scaler = {"mean": mean.tolist(), "std": std.tolist()}
    cfg.fitted_on = {"fold": fold, "train_rows": int(len(train_df)), "ok_rows": int(ok.sum()),
                     "normal_rows_for_vref": int(normal.sum())}
    return cfg


# [GUIDE] C. Applying the fitted features: z = (x - mean) / std
# Used for training AND inference; it never refits anything.
# OUTPUT: X (float32, one row per OK reading, 8 columns) and the status of every row.
def transform(df, cfg):
    """Stage A + B. Returns (X, status): X is a float32 array with one row per
    status-OK reading, in df order; status covers every row of df."""
    if not cfg.is_fitted:
        raise ValueError("config is not fitted")
    status = input_status(df, cfg)
    ok = (status == STATUS_OK).to_numpy()
    feats = physical_features(df, cfg, status).to_numpy(float)[ok]
    z = (feats - np.asarray(cfg.scaler["mean"])) / np.asarray(cfg.scaler["std"])
    X = z.astype(cfg.dtype)
    if not np.isfinite(X).all():
        raise FloatingPointError("non-finite model input after gating; check the input limits")
    return X, status


# [GUIDE] C. One fold: fit_pipeline on train rows, then transform train and test with the
# SAME fitted values (no refit on the test side). Used by the 5-fold comparison.
def fold_matrices(df, train_idx, test_idx, cfg=None, fold=None):
    """Fit on df.loc[train_idx], apply to df.loc[test_idx] without refitting.

    Returns a dict with the fitted config and X/y/row index for train and test
    (status-OK rows only) plus the full status of both sides.
    """
    tr, te = df.loc[train_idx], df.loc[test_idx]
    fitted = fit_pipeline(tr, cfg, fold=fold)
    out = {"config": fitted}
    for side, part in (("train", tr), ("test", te)):
        X, status = transform(part, fitted)
        keep = part.index[(status == STATUS_OK).to_numpy()]
        out[side] = {"X": X, "y": part.loc[keep, LABEL_COLUMN].to_numpy(),
                     "index": keep, "status": status}
    return out

import math

import numpy as np
import pandas as pd
import pytest

from tinyml.contract import LABELS, NEVER_MODEL_INPUTS
from tinyml.features import (
    FEATURES, PREDICTION_STATUS, STATUS_CODES, FeatureConfig, fit_pipeline, fold_matrices,
    input_status, physical_features, transform,
)
from tinyml.splits import iter_folds, make_assignments

CODES = {"NORMAL": "NOR", "PARTIAL_SHADING": "PAR", "DUST": "DUS",
         "OPEN_CIRCUIT": "OPE", "SHORT_CIRCUIT": "SHO"}


def reading(**kw):
    r = {"voltage_v": 2.7, "current_ma": 50.0, "power_mw": 135.0, "panel_temp1_c": 36.0,
         "panel_temp2_c": 35.0, "ambient_temp_c": 28.0, "irradiance_lux": 40000.0,
         "dust_vo_v": 0.12, "humidity_pct": 80.0, "pressure_hpa": 1009.0,
         "label": "NORMAL", "health_score": 90.0}
    r.update(kw)
    return r


def make_dataset(runs_per_window=5, readings=4, seed=0):
    """Runs with a known NORMAL voltage law V = 2.6 + 0.08 ln(klx) - 0.01 (T - 25)."""
    rng = np.random.default_rng(seed)
    rows, rec = [], 0
    for label in LABELS:
        n = 0
        for window in ("TRAINING", "OPERATION"):
            for _ in range(runs_per_window):
                n += 1
                site = ("SITE-A", "SITE-B", "SITE-C")[n % 3]
                lux = float(rng.uniform(10000, 90000))
                for k in range(readings):
                    rec += 1
                    t1, t2 = 30.0 + k, 29.5 + k
                    v = 2.6 + 0.08 * math.log(lux / 1000) - 0.01 * ((t1 + t2) / 2 - 25)
                    i = 0.0015 * lux
                    if label == "OPEN_CIRCUIT":
                        v, i = v + 0.4, 0.1
                    elif label == "SHORT_CIRCUIT":
                        v, i = 0.05, 0.0018 * lux
                    elif label == "PARTIAL_SHADING":
                        v, t1 = v - 0.3, t1 + 2.0
                    elif label == "DUST":
                        i = 0.001 * lux
                    rows.append(reading(
                        record_id=f"REC-{rec:05d}", run_id=f"{site}-{CODES[label]}-{n:02d}",
                        site_id=site, window=window, reading=k + 1, label=label,
                        voltage_v=round(v, 3), current_ma=round(i, 2), power_mw=round(v * i, 2),
                        panel_temp1_c=t1, panel_temp2_c=t2, irradiance_lux=round(lux),
                        dust_vo_v=0.9 if label == "DUST" else 0.12))
    return pd.DataFrame(rows)


@pytest.fixture
def df():
    return make_dataset()


@pytest.fixture
def cfg(df):
    return fit_pipeline(df[df.window == "TRAINING"])


def test_feature_order_matches_architecture():
    assert FEATURES == ("v_deficit", "i_norm", "p_norm", "dust_vo_v",
                        "temp_spread", "temp_rise", "voltage_v", "current_ma")
    assert not set(FEATURES[3:]) & set(NEVER_MODEL_INPUTS)


def test_formulas_and_units():
    cfg = FeatureConfig(vref={"a": 2.6, "b": 0.08, "c": -0.01})
    row = pd.DataFrame([reading()])
    f = physical_features(row, cfg).iloc[0]
    klx, tp = 40.0, 35.5
    assert f.v_deficit == pytest.approx(2.6 + 0.08 * math.log(klx) - 0.01 * (tp - 25) - 2.7)
    assert f.i_norm == pytest.approx(50.0 / klx)          # mA per klx
    assert f.p_norm == pytest.approx(135.0 / klx)         # mW per klx
    assert f.dust_vo_v == pytest.approx(0.12)
    assert f.temp_spread == pytest.approx(1.0)
    assert f.temp_rise == pytest.approx(tp - 28.0)
    assert (f.voltage_v, f.current_ma) == (2.7, 50.0)
    assert list(f.index) == list(FEATURES)


def test_temp_spread_is_absolute():
    cfg = FeatureConfig(vref={"a": 0, "b": 0, "c": 0})
    f = physical_features(pd.DataFrame([reading(panel_temp1_c=30.0, panel_temp2_c=33.0)]), cfg)
    assert f.temp_spread.iloc[0] == pytest.approx(3.0)


@pytest.mark.parametrize("lux,expected", [
    (0.0, "LOW_LIGHT"), (100.0, "LOW_LIGHT"), (2399.0, "LOW_LIGHT"), (2400.0, "OK"),
    (99999.0, "OK"), (100000.0, "LUX_SATURATED"),
])
def test_irradiance_gating(lux, expected):
    s = input_status(pd.DataFrame([reading(irradiance_lux=lux)]), FeatureConfig())
    assert s.iloc[0] == expected


@pytest.mark.parametrize("change", [
    {"voltage_v": float("nan")}, {"current_ma": float("inf")}, {"irradiance_lux": -5.0},
    {"panel_temp1_c": -127.0}, {"panel_temp2_c": -999.0}, {"current_ma": -3.0},
    {"voltage_v": 9.0}, {"dust_vo_v": None}, {"ambient_temp_c": "n/a"},
])
def test_invalid_inputs_are_flagged(change):
    s = input_status(pd.DataFrame([reading(**change)]), FeatureConfig())
    assert s.iloc[0] == "INVALID_INPUT"


def test_invalid_takes_precedence_over_irradiance_status():
    s = input_status(pd.DataFrame([reading(irradiance_lux=100000.0, voltage_v=float("nan"))]),
                     FeatureConfig())
    assert s.iloc[0] == "INVALID_INPUT"


def test_gated_rows_are_kept_and_get_nan_features():
    cfg = FeatureConfig(vref={"a": 2.6, "b": 0.08, "c": -0.01})
    rows = pd.DataFrame([reading(), reading(irradiance_lux=0.0), reading(irradiance_lux=100000.0),
                         reading(voltage_v=float("nan"))])
    f = physical_features(rows, cfg)
    assert len(f) == 4                                      # nothing dropped
    assert f.iloc[0].notna().all()
    assert f.iloc[1:].isna().all().all()                    # no division by zero, no log(0)


def test_accepted_inputs_are_finite_float32(df, cfg):
    X, status = transform(df[df.window == "TRAINING"], cfg)
    assert X.dtype == np.float32 and X.shape[1] == 8
    assert np.isfinite(X).all()
    assert len(X) == (status == "OK").sum()


def test_zero_irradiance_never_reaches_model(cfg):
    X, status = transform(pd.DataFrame([reading(irradiance_lux=0.0), reading()]), cfg)
    assert list(status) == ["LOW_LIGHT", "OK"] and len(X) == 1


def test_missing_raw_column_raises():
    with pytest.raises(KeyError, match="dust_vo_v"):
        input_status(pd.DataFrame([reading()]).drop(columns="dust_vo_v"), FeatureConfig())


def test_vref_is_fitted_on_normal_training_rows(df, cfg):
    assert cfg.vref["a"] == pytest.approx(2.6, abs=0.02)
    assert cfg.vref["b"] == pytest.approx(0.08, abs=0.01)
    assert cfg.vref["c"] == pytest.approx(-0.01, abs=0.005)


def test_scaler_uses_training_rows_only(df):
    tr, te = df.index[df.window == "TRAINING"], df.index[df.window == "OPERATION"]
    a = fold_matrices(df, tr, te)["config"]
    changed = df.copy()
    changed.loc[te, "current_ma"] *= 3                       # change test rows drastically
    changed.loc[te, "voltage_v"] *= 0.5
    b = fold_matrices(changed, tr, te)["config"]
    assert a.scaler == b.scaler and a.vref == b.vref


def test_transform_does_not_refit(df, cfg):
    before = cfg.to_json()
    transform(df[df.window == "OPERATION"], cfg)
    assert cfg.to_json() == before


def test_training_features_are_standardised(df, cfg):
    X, _ = transform(df[df.window == "TRAINING"], cfg)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(X.std(axis=0)[X.std(axis=0) > 0], 1, atol=1e-3)


def test_fit_ignores_gated_rows(df):
    train = df[df.window == "TRAINING"].copy()
    clean = fit_pipeline(train)
    train.loc[train.index[:3], "irradiance_lux"] = 100000.0   # now saturated
    train.loc[train.index[3], "voltage_v"] = float("nan")     # now invalid
    gated = fit_pipeline(train)
    assert gated.fitted_on["ok_rows"] == clean.fitted_on["ok_rows"] - 4
    assert all(np.isfinite(gated.scaler["mean"]))


def test_unfitted_config_is_rejected(df):
    with pytest.raises(ValueError, match="not fitted"):
        transform(df, FeatureConfig())
    with pytest.raises(ValueError, match="V_ref"):
        physical_features(df, FeatureConfig())


def test_too_few_normal_rows_raises(df):
    with pytest.raises(ValueError, match="NORMAL"):
        fit_pipeline(df[df.label != "NORMAL"])


def test_config_round_trip(tmp_path, df, cfg):
    path = tmp_path / "features.json"
    cfg.save(path)
    loaded = FeatureConfig.load(path)
    assert loaded.to_json() == cfg.to_json()
    X1, _ = transform(df, cfg)
    X2, _ = transform(df, loaded)
    np.testing.assert_array_equal(X1, X2)


def test_config_records_everything_esp32_needs(cfg):
    d = FeatureConfig.from_json(cfg.to_json())
    assert d.features == FEATURES and d.version == "pv8-v1" and d.dtype == "float32"
    assert set(d.vref) == {"a", "b", "c"}
    assert len(d.scaler["mean"]) == len(d.scaler["std"]) == 8
    assert d.low_light_lux and d.lux_saturation and d.input_limits


def test_config_with_other_feature_order_is_rejected(cfg):
    bad = cfg.to_json().replace('"v_deficit"', '"v_deficitX"')
    with pytest.raises(ValueError):
        FeatureConfig.from_json(bad)


def test_repeated_transform_is_deterministic(df, cfg):
    X1, s1 = transform(df, cfg)
    X2, s2 = transform(df.copy(), cfg)
    np.testing.assert_array_equal(X1, X2)
    assert s1.equals(s2)
    assert fit_pipeline(df[df.window == "TRAINING"]).to_json() == cfg.to_json()


def test_features_ignore_metadata_columns(df, cfg):
    shuffled = df.copy()
    shuffled["health_score"] = 0.0
    shuffled["run_id"] = "X"
    shuffled["site_id"] = "SITE-Z"
    shuffled["record_id"] = "R"
    shuffled["timestamp_utc"] = "2000-01-01T00:00:00Z"
    np.testing.assert_array_equal(transform(df, cfg)[0], transform(shuffled, cfg)[0])


@pytest.mark.parametrize("protocol", ["cv", "temporal", "loso"])
def test_fold_matrices_with_phase4_splits(df, protocol):
    assignments = make_assignments(df, n_splits=5, seed=42)
    for name, tr, te in iter_folds(df, assignments, protocol):
        m = fold_matrices(df, tr, te, fold=name)
        assert m["config"].fitted_on["fold"] == name
        assert m["config"].fitted_on["train_rows"] == len(tr)        # fitted on train side only
        assert not set(m["train"]["index"]) & set(m["test"]["index"])
        assert len(m["test"]["X"]) == len(m["test"]["y"])
        assert m["train"]["X"].shape[1] == 8


def test_fold_fit_matches_direct_fit_on_train_rows(df):
    assignments = make_assignments(df, n_splits=5, seed=42)
    name, tr, te = next(iter_folds(df, assignments, "cv"))
    m = fold_matrices(df, tr, te, fold=name)
    assert m["config"].vref == fit_pipeline(df.loc[tr]).vref


def test_status_codes_map_to_architecture():
    assert STATUS_CODES == {"OK": 0, "LOW_LIGHT": 1, "LUX_SATURATED": 2, "INVALID_INPUT": 3}
    assert PREDICTION_STATUS["LOW_LIGHT"] == "SKIPPED_LOW_LIGHT"
    assert PREDICTION_STATUS["OK"] == "CLASSIFIED"

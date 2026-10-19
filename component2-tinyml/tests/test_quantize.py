"""Phase 10 INT8 arithmetic, quantisation and comparison (synthetic data only)."""
import json
import math
from fractions import Fraction

import numpy as np
import pytest

from tinyml.contract import LABELS
from tinyml.device_reference import (STATUS_INVALID, STATUS_LOW_LIGHT, STATUS_OK, STATUS_SATURATED,
                                     device_constants, gate_f32, preprocess_f32, raw_matrix)
from tinyml.export import build_contract, contract_weights, forward
from tinyml.features import FEATURES, STATUS_CODES, input_status, transform
from tinyml.quantize import (F32, QMAX, QuantParams, acceptance, accumulator_bounds, argmax_first,
                             build_tanh_lut, compare_predictions, int8_core, int8_predict, int8_predict_many,
                             quantize_input, quantize_model, quantize_multiplier, rnd, rshift_rnd, sat,
                             validate_params)
from tinyml.splits import make_assignments
from tinyml.train import train_final

from test_features import make_dataset

ROOT_SEED = 3


@pytest.fixture(scope="module")
def setup():
    df = make_dataset(runs_per_window=5, readings=6)
    a = make_assignments(df)
    res = train_final(df, seed=0, assignments=a)
    contract = build_contract(res, "0" * 64, "runwise-v1", "test", {"python": "t"})
    tr = df[df.window == "TRAINING"].reset_index(drop=True)
    status, z = preprocess_f32(raw_matrix(tr), res["config"])
    q = quantize_model(contract, "c" * 64, z[status == STATUS_OK], {"window": "TRAINING"})
    return {"df": df, "tr": tr, "contract": contract, "cfg": res["config"], "z": z, "status": status, "q": q}


def exact_round_half_away(fr):
    """Independent reference: Fraction rounded half away from zero."""
    s = -1 if fr < 0 else 1
    a = abs(fr)
    return s * int(math.floor(a + Fraction(1, 2)))


# ---------------------------------------------------------------- rounding / shifting / saturation

@pytest.mark.parametrize("v,expected", [(0.5, 1), (-0.5, -1), (1.5, 2), (-1.5, -2), (2.5, 3), (-2.5, -3),
                                        (0.49999997, 0), (-0.49999997, 0), (126.5, 127), (0.0, 0)])
def test_rnd_is_half_away_from_zero(v, expected):
    assert int(rnd(float(F32(v)))) == expected


def test_rnd_float32_tie_neighbour():
    below_half = float(np.nextafter(F32(0.5), F32(0)))           # float32 + 0.5 would round to 1.0
    assert int(rnd(below_half)) == 0 and int(rnd(-below_half)) == 0


def test_rshift_rnd_matches_exact_division_positive_and_negative():
    rng = np.random.default_rng(ROOT_SEED)
    for n in (1, 2, 7, 31, 37, 45, 62):
        half = 1 << (n - 1)
        vals = [0, 1, -1, half, -half, half - 1, -(half - 1), 3 * half, -3 * half]
        vals += [int(x) for x in rng.integers(-(1 << 61), 1 << 61, 50)]
        for v in vals:
            if abs(v) < (1 << 62):
                assert rshift_rnd(v, n) == exact_round_half_away(Fraction(v, 1 << n)), (v, n)
    with pytest.raises(ValueError):
        rshift_rnd(5, 0)


def test_sat_reports_clipping():
    assert sat(5, -127, 127) == (5, False)
    assert sat(200, -127, 127) == (127, True) and sat(-200, -127, 127) == (-127, True)
    assert sat(-128, -128, 127) == (-128, False)


@pytest.mark.parametrize("m", [1e-6, 0.0077, 0.25, 0.5, 0.9999999999, 1.0, 1.5, 123.456])
def test_quantize_multiplier(m):
    m1, sh = quantize_multiplier(m)
    assert 2 ** 30 <= m1 < 2 ** 31 and 1 <= sh <= 62
    assert abs(m1 / 2 ** sh - m) <= m * 2 ** -30


def test_quantize_multiplier_rejects_bad_values():
    for bad in (0.0, -1.0, float("inf"), float("nan"), 2.0 ** 40):
        with pytest.raises(ValueError):
            quantize_multiplier(bad)


def test_fixed_point_rescale_equals_exact_rational(setup):
    """rshift_rnd(acc * M1, SH1) is exact rounding of acc * M1 / 2^SH1 for
    positive and negative accumulators, including the int32 extremes."""
    q = setup["q"]
    rng = np.random.default_rng(1)
    accs = [0, 1, -1, 2 ** 31 - 1, -(2 ** 31 - 1)] + [int(x) for x in rng.integers(-300000, 300000, 300)]
    for acc in accs:
        assert rshift_rnd(acc * q.m1, q.sh1) == exact_round_half_away(Fraction(acc * q.m1, 2 ** q.sh1))
        assert abs(acc * q.m1) < 2 ** 62


# ---------------------------------------------------------------- quantised parameters

def test_quantize_model_ranges_and_order(setup):
    q = setup["q"]
    assert validate_params(q)
    qw1, qb1, qw2, qb2, lut = q.arrays()
    assert np.abs(qw1).max() == QMAX and np.abs(qw2).max() == QMAX       # max weight maps to +-127
    assert qw1.min() >= -QMAX and qw2.min() >= -QMAX                      # never -128
    assert q.features == list(FEATURES) and q.labels == list(LABELS)
    assert q.weights_sha256 == setup["contract"]["weights_sha256"]
    w = contract_weights(setup["contract"])
    np.testing.assert_allclose(qw1 * q.s_w1, w["W1"], atol=q.s_w1 / 2 + 1e-12)
    np.testing.assert_allclose(qb1 * q.s_x * q.s_w1, w["b1"], atol=q.s_x * q.s_w1 / 2 + 1e-12)
    np.testing.assert_allclose(qb2 * q.s_h * q.s_w2, w["b2"], atol=q.s_h * q.s_w2 / 2 + 1e-12)


def test_calibration_is_max_abs_of_the_given_rows(setup):
    q, z, ok = setup["q"], setup["z"], setup["status"] == STATUS_OK
    assert q.a_x == pytest.approx(float(np.abs(z[ok]).max()))
    w = contract_weights(setup["contract"])
    assert q.a_z1 == pytest.approx(float(np.abs(z[ok].astype(float) @ w["W1"] + w["b1"]).max()))
    assert q.calibration["rows"] == int(ok.sum())


def test_tanh_lut_shape_symmetry_and_monotonicity(setup):
    lut = np.array(setup["q"].tanh_lut)
    assert lut.shape == (256,) and lut[128] == 0 and np.abs(lut).max() <= QMAX
    assert (np.diff(lut) >= 0).all()
    assert all(lut[128 + k] == -lut[128 - k] for k in range(1, 128))
    np.testing.assert_array_equal(build_tanh_lut(setup["q"].s_z1), lut)


def test_quant_params_json_round_trip(setup):
    q = setup["q"]
    back = QuantParams.from_json(q.to_json())
    assert back.to_json() == q.to_json() and back.inv_sx == q.inv_sx


@pytest.mark.parametrize("field,value", [("labels", list(reversed(LABELS))), ("features", list(reversed(FEATURES))),
                                         ("qb1", [2 ** 31] + [0] * 15), ("m1", 5), ("sh1", 0),
                                         ("activation", "relu")])
def test_invalid_params_are_refused(setup, field, value):
    d = json.loads(setup["q"].to_json())
    d[field] = value
    with pytest.raises(ValueError):
        QuantParams.from_json(json.dumps(d))


def test_overflow_bounds_are_enforced(setup):
    d = json.loads(setup["q"].to_json())
    d["qw1"] = [[127] * 16 for _ in range(8)]
    d["qb1"] = [2 ** 31 - 1 - 8 * 127 * 127 + 1] * 16           # acc1 bound just above int32
    with pytest.raises(ValueError, match="int32"):
        QuantParams.from_json(json.dumps(d))
    # exactly at the int32 limit with the largest M1: accepted, and the int64
    # rescaling cannot overflow (|acc1 * M1| < 2^62, plus the rounding half 2^61 < 2^63)
    d = json.loads(setup["q"].to_json())
    d["qw1"] = [[127] * 16 for _ in range(8)]
    d["qb1"] = [2 ** 31 - 1 - 8 * 127 * 127] * 16
    d["m1"], d["sh1"] = 2 ** 31 - 1, 62
    q = QuantParams.from_json(json.dumps(d))
    b = accumulator_bounds(q)
    assert b["acc1"] == 2 ** 31 - 1 and b["acc1_times_m1"] + 2 ** 61 < 2 ** 63
    acc2, k, hq, _ = int8_core([127] * 8, q)
    assert rshift_rnd(b["acc1"] * q.m1, 62) == exact_round_half_away(Fraction(b["acc1"] * q.m1, 2 ** 62))


def test_extreme_core_inputs_stay_in_range(setup):
    q = setup["q"]
    b = accumulator_bounds(q)
    assert b["acc1"] < 2 ** 31 and b["acc2"] < 2 ** 31 and b["acc1_times_m1"] < 2 ** 62
    for qx in ([QMAX] * 8, [-QMAX] * 8, [QMAX, -QMAX] * 4):
        acc2, k, hq, _ = int8_core(qx, q)
        assert all(-128 <= x <= 127 for x in k) and all(abs(x) <= QMAX for x in hq)


# ---------------------------------------------------------------- emulator

def test_input_quantisation_clips_and_rejects_non_finite():
    assert quantize_input([1000.0, -1000.0, 0.4], F32(1.0)) == ([127, -127, 0], 2)
    with pytest.raises(ValueError):
        quantize_input([np.nan], F32(1.0))


def test_argmax_ties_go_to_lowest_index():
    assert argmax_first([5, 5, 3, 1, 0]) == 0
    assert argmax_first([1, 7, 7, 7, 0]) == 1
    assert argmax_first([0, 0, 0, 0, 0]) == 0


def test_int8_tracks_float_model(setup):
    q, z, ok = setup["q"], setup["z"], setup["status"] == STATUS_OK
    p = int8_predict_many(z[ok], q)
    _, probs = forward(contract_weights(setup["contract"]), z[ok].astype(float))
    lg64 = np.log(probs)
    assert (p["label_index"] == probs.argmax(axis=1)).all()
    assert p["input_clips"] == 0                                    # calibration rows never clip
    shift = lambda a: a - a.mean(axis=1, keepdims=True)              # noqa: E731 softmax-invariant
    assert np.abs(shift(p["logits"].astype(float)) - shift(lg64)).max() < 0.25


def test_out_of_range_input_is_clipped_and_counted(setup):
    q = setup["q"]
    z = np.zeros(8, dtype=F32)
    z[4] = F32(q.a_x * 3)
    r = int8_predict(z, q)
    assert r["input_clips"] == 1 and r["qx"][4] == QMAX


# ---------------------------------------------------------------- device reference

def test_device_reference_matches_float64_pipeline(setup):
    tr, cfg = setup["tr"], setup["cfg"]
    x64, st64 = transform(tr.drop(columns=["label", "health_score"]), cfg)
    assert (st64.map(STATUS_CODES).to_numpy() == setup["status"]).all()
    ok = setup["status"] == STATUS_OK
    np.testing.assert_allclose(setup["z"][ok], x64, rtol=1e-4, atol=1e-5)


def test_device_reference_reads_no_labels(setup):
    tr = setup["tr"].drop(columns=["label", "health_score"])
    status, _ = preprocess_f32(raw_matrix(tr), setup["cfg"])
    assert (status == setup["status"]).all()


def test_gate_boundaries_and_precedence(setup):
    k = device_constants(setup["cfg"])
    base = raw_matrix(setup["tr"])[0].copy()
    def st(**kw):
        r = base.copy()
        for col, v in kw.items():
            r[["voltage_v", "current_ma", "power_mw", "panel_temp1_c", "panel_temp2_c", "ambient_temp_c",
               "irradiance_lux", "dust_vo_v"].index(col)] = F32(v)
        return int(gate_f32(r[None], k)[0])
    assert st(irradiance_lux=2399.0) == STATUS_LOW_LIGHT and st(irradiance_lux=2400.0) == STATUS_OK
    assert st(irradiance_lux=99999.0) == STATUS_OK and st(irradiance_lux=100000.0) == STATUS_SATURATED
    assert st(irradiance_lux=float(np.nextafter(F32(100000), F32(np.inf)))) == STATUS_INVALID
    assert st(irradiance_lux=-1.0) == STATUS_INVALID and st(irradiance_lux=0.0) == STATUS_LOW_LIGHT
    assert st(panel_temp1_c=-127.0) == STATUS_INVALID and st(panel_temp2_c=-999.0) == STATUS_INVALID
    assert st(voltage_v=float("nan")) == STATUS_INVALID and st(dust_vo_v=float("inf")) == STATUS_INVALID
    assert st(irradiance_lux=100000.0, voltage_v=float("nan")) == STATUS_INVALID
    assert st(dust_vo_v=3.3) == STATUS_OK
    assert st(dust_vo_v=float(np.nextafter(F32(3.3), F32(np.inf)))) == STATUS_INVALID


def test_gate_agrees_with_float64_input_status(setup):
    import pandas as pd
    k = device_constants(setup["cfg"])
    base = setup["tr"].iloc[[0] * 6].reset_index(drop=True).copy()
    base.loc[0, "irradiance_lux"] = 2399.0
    base.loc[1, "irradiance_lux"] = 100000.0
    base.loc[2, "panel_temp1_c"] = -127.0
    base.loc[3, "voltage_v"] = np.nan
    base.loc[4, "ambient_temp_c"] = 50.0
    expected = input_status(base, setup["cfg"]).map(STATUS_CODES).to_numpy()
    assert (gate_f32(raw_matrix(base), k) == expected).all()
    assert isinstance(base, pd.DataFrame)


# ---------------------------------------------------------------- comparison and acceptance

def test_comparison_detects_changes_to_and_from_open_and_short():
    i = {l: n for n, l in enumerate(LABELS)}
    ref = [i["NORMAL"], i["OPEN_CIRCUIT"], i["SHORT_CIRCUIT"], i["DUST"], i["OPEN_CIRCUIT"], i["NORMAL"]]
    q8 = [i["OPEN_CIRCUIT"], i["NORMAL"], i["SHORT_CIRCUIT"], i["NORMAL"], i["SHORT_CIRCUIT"], i["NORMAL"]]
    c = compare_predictions(ref, q8)
    assert c["safety_changes"]["OPEN_CIRCUIT"] == {"changed_to": 1, "changed_from": 2}
    assert c["safety_changes"]["SHORT_CIRCUIT"] == {"changed_to": 1, "changed_from": 0}
    assert c["safety_changes_total"] == 3                          # rows 0, 1 and 4, each counted once
    assert c["changed"] == 4 and c["change_pairs"]["DUST->NORMAL"] == 1


def test_acceptance_rule():
    good = {"safety_changes_total": 0, "agreement": 0.996, "training_fit_macro_f1": {"drop": 0.005}}
    assert acceptance(good, True)["int8_accepted"]
    assert not acceptance(dict(good, safety_changes_total=1), True)["int8_accepted"]
    assert not acceptance(dict(good, agreement=0.994), True)["int8_accepted"]
    assert not acceptance(dict(good, training_fit_macro_f1={"drop": 0.011}), True)["int8_accepted"]
    assert not acceptance(good, False)["int8_accepted"]
    assert "does not establish electrical safety" in acceptance(good, True)["note"]


def test_quantisation_does_not_change_the_float_contract(setup):
    before = json.dumps(setup["contract"], sort_keys=True)
    quantize_model(setup["contract"], "c" * 64, setup["z"][setup["status"] == STATUS_OK])
    assert json.dumps(setup["contract"], sort_keys=True) == before

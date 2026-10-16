"""Pruning feasibility study E1 (synthetic models and data only).

The official study is run later in Colab on TRAINING rows (docs/PRUNING.md);
nothing here reads the real dataset or OPERATION rows."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tinyml import prune
from tinyml.contract import LABELS
from tinyml.device_reference import STATUS_OK, preprocess_f32, raw_matrix
from tinyml.export import build_contract, contract_weights, file_sha256, forward, training_metadata, write_artifacts
from tinyml.quantize import QuantParams, compare_predictions, int8_core, int8_predict_many, quantize_input, quantize_model
from tinyml.splits import make_assignments, to_json
from tinyml.train import train_final

from test_features import make_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synth():
    df = make_dataset(runs_per_window=5, readings=6)
    a = make_assignments(df)
    res = train_final(df, seed=0, assignments=a)
    contract = build_contract(res, "0" * 64, "runwise-v1", "test", {"python": "t"})
    tr = df[df.window == "TRAINING"].reset_index(drop=True)
    status, z = preprocess_f32(raw_matrix(tr), res["config"])
    z = z[status == STATUS_OK]
    q = quantize_model(contract, "c" * 64, z, {"window": "TRAINING"})
    return {"df": df, "a": a, "res": res, "contract": contract, "w": contract_weights(contract),
            "z": z, "y": tr.loc[status == STATUS_OK, "label"].to_numpy(), "q": q}


def random_weights(seed=0, n_hid=16):
    r = np.random.default_rng(seed)
    return {"W1": r.normal(size=(8, n_hid)), "b1": r.normal(size=n_hid),
            "W2": r.normal(size=(n_hid, 5)), "b2": r.normal(size=5)}


# ---------------------------------------------------------------- unstructured

@pytest.mark.parametrize("fraction, removed", [(0.10, 21), (0.20, 42), (0.30, 62)])
def test_unstructured_removes_the_smallest_weights(fraction, removed):
    w = random_weights()
    p, m = prune.unstructured_prune(w, fraction)
    assert m["removed"] == removed
    zero = np.concatenate([(~m["W1"]).ravel(), (~m["W2"]).ravel()])
    mag = np.abs(np.concatenate([w["W1"].ravel(), w["W2"].ravel()]))
    assert zero.sum() == removed and mag[zero].max() <= mag[~zero].min()
    assert np.array_equal(p["W1"], np.where(m["W1"], w["W1"], 0)) and np.array_equal(p["W2"], np.where(m["W2"], w["W2"], 0))
    assert np.array_equal(p["b1"], w["b1"]) and np.array_equal(p["b2"], w["b2"])      # biases kept
    assert p["W1"].shape == (8, 16) and p["W2"].shape == (16, 5)                      # 229-value layout kept


def test_unstructured_is_deterministic_with_ties_and_does_not_mutate():
    w = random_weights()
    w["W1"][0, :4] = 1e-3                                       # four equal magnitudes
    w["W2"][0, 0] = -1e-3
    orig = {k: v.copy() for k, v in w.items()}
    p1, m1 = prune.unstructured_prune(w, 2 / 208)
    p2, _ = prune.unstructured_prune(w, 2 / 208)
    assert all(np.array_equal(p1[k], p2[k]) for k in w)
    assert not m1["W1"][0, 0] and not m1["W1"][0, 1] and m1["W1"][0, 2] and m1["W2"][0, 0]   # by position
    assert all(np.array_equal(w[k], orig[k]) for k in w)


def test_zero_fraction_is_the_original_and_bad_fractions_are_refused():
    w = random_weights()
    p, m = prune.unstructured_prune(w, 0)
    assert m["removed"] == 0 and all(np.array_equal(p[k], w[k]) for k in w)
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            prune.unstructured_prune(w, bad)


# ---------------------------------------------------------------- structured

def test_unit_importance_definition():
    w = random_weights()
    j = 5
    assert np.isclose(prune.unit_importance(w)[j], np.abs(w["W1"][:, j]).sum() * np.abs(w["W2"][j]).sum())


@pytest.mark.parametrize("k", prune.STRUCTURED_UNITS)
def test_structured_removes_lowest_importance_units(k):
    w = random_weights(1)
    imp = prune.unit_importance(w)
    p, removed = prune.structured_prune(w, k)
    assert len(removed) == k and removed == sorted(removed)
    assert imp[removed].max() <= np.delete(imp, removed).min()
    assert p["W1"].shape == (8, 16 - k) and p["b1"].shape == (16 - k,) and p["W2"].shape == (16 - k, 5)
    assert np.array_equal(p["b2"], w["b2"])


def test_structured_ties_go_to_the_lower_index_and_range_is_checked():
    w = random_weights(2)
    for j in (3, 9, 12):
        w["W1"][:, j] = 1e-4
        w["W2"][j] = 1e-4
    assert prune.structured_prune(w, 2)[1] == [3, 9]
    for bad in (-1, 16):
        with pytest.raises(ValueError):
            prune.structured_prune(w, bad)


def test_removing_units_equals_silencing_their_outgoing_weights():
    w = random_weights(3)
    z = np.random.default_rng(4).normal(size=(50, 8))
    p, removed = prune.structured_prune(w, 3)
    silenced = {k: v.copy() for k, v in w.items()}
    silenced["W2"][removed] = 0.0
    assert np.allclose(forward(p, z)[1], forward(silenced, z)[1])


# ---------------------------------------------------------------- INT8 variants

def test_vectorised_int8_equals_the_frozen_int8_path(synth):
    q, z = synth["q"], synth["z"]
    lab, in_c, lut_c = prune.int8_labels(z, prune.int8_params_for(q), q)
    ref = int8_predict_many(z, q)
    assert np.array_equal(lab, ref["label_index"])
    assert in_c == ref["input_clips"] and lut_c == ref["lut_clips"]


def test_int8_variant_zeroes_pruned_weights_without_requantising(synth):
    q, w = synth["q"], synth["w"]
    _, m = prune.unstructured_prune(w, 0.3)
    p8 = prune.int8_params_for(q, m)
    qw1, qb1, qw2, qb2, _ = q.arrays()
    assert np.array_equal(p8["qw1"], np.where(m["W1"], qw1, 0)) and np.array_equal(p8["qw2"], np.where(m["W2"], qw2, 0))
    assert np.array_equal(p8["qb1"], qb1) and np.array_equal(p8["qb2"], qb2)
    assert q.qw1 == qw1.tolist()                                     # frozen params not mutated


def test_int8_structured_variant_equals_int8_core_with_silenced_units(synth):
    q, z = synth["q"], synth["z"]
    removed = [1, 4]
    lab, _, _ = prune.int8_labels(z, prune.int8_params_for(q, removed_units=removed), q)
    d = json.loads(q.to_json())
    for j in removed:
        d["qw2"][j] = [0] * 5
    q0 = QuantParams(**d)
    expect = [int(np.argmax(int8_core(quantize_input(r, q.inv_sx)[0], q0)[0])) for r in z]
    assert lab.tolist() == expect


# ---------------------------------------------------------------- estimates

def test_storage_estimate_arithmetic():
    s = prune.storage_estimate(16, 208)
    assert s["int8"]["dense"] == 208 + 21 * 4 + 256 + 13 == 561
    assert s["float32"]["dense"] == 229 * 4 == 916
    s30 = prune.storage_estimate(16, 146)
    assert s30["int8"]["bitmap_sparse"] == 26 + 146 + 84 + 269 == 525          # saves 36 B
    assert s30["float32"]["bitmap_sparse"] == 26 + 146 * 4 + 84 == 694          # saves 222 B
    assert s30["int8"]["index_list_sparse"] == 146 * 2 + 84 + 269
    s11 = prune.storage_estimate(11, 11 * 13)
    assert s16_minus_s11(prune.storage_estimate(16, 208), s11) == (5 * 17, 5 * 56)


def s16_minus_s11(a, b):
    return a["int8"]["dense"] - b["int8"]["dense"], a["float32"]["dense"] - b["float32"]["dense"]


def test_mac_estimate():
    assert prune.mac_estimate(16, 146) == {"dense_multiply_adds": 208, "zero_skipping_multiply_adds": 146,
                                           "tanh_evaluations": 16}
    assert prune.mac_estimate(13, 13 * 13)["dense_multiply_adds"] == 169


# ---------------------------------------------------------------- checks and study

def test_e1_checks_detect_open_short_changes_and_low_agreement():
    ref = np.zeros(1000, dtype=int)
    same = compare_predictions(ref, ref)
    to_open = ref.copy(); to_open[0] = LABELS.index("OPEN_CIRCUIT")
    to_dust = ref.copy(); to_dust[:10] = LABELS.index("DUST")
    assert prune.e1_checks(same, same)["int8"]["passed"]
    c = prune.e1_checks(same, compare_predictions(ref, to_open))
    assert c["float32"]["passed"] and not c["int8"]["passed"] and not c["int8"]["safety_changes_zero"]
    c = prune.e1_checks(compare_predictions(ref, to_dust), same)            # agreement 0.990 < 0.995
    assert not c["float32"]["passed"] and c["float32"]["safety_changes_zero"]


def test_run_study_variants_and_baseline(synth):
    s = synth
    z64 = s["z"].astype(np.float64)
    r = prune.run_study(s["contract"], s["q"], z64, s["z"], s["y"])
    names = [v["name"] for v in r["variants"]]
    assert names == ["baseline", "unstructured_10pct", "unstructured_20pct", "unstructured_30pct",
                     "structured_minus2", "structured_minus3", "structured_minus5"]
    base = r["variants"][0]
    assert base["pruned_float_vs_frozen_float"]["agreement"] == 1.0
    assert base["pruned_int8_vs_frozen_int8"]["agreement"] == 1.0
    assert base["parameters_stored_dense"] == 229
    by = {v["name"]: v for v in r["variants"]}
    assert by["unstructured_30pct"]["nonzero_weights_float"] == 208 - 62
    assert by["unstructured_30pct"]["parameters_stored_dense"] == 229
    assert by["structured_minus5"]["hidden_units"] == 11 and by["structured_minus5"]["parameters_stored_dense"] == 159
    assert "BROKEN" in by["structured_minus2"]["fl_layout"]
    assert "criterion 3" in r["conclusion_rule"]
    with pytest.raises(ValueError):
        prune.run_study(s["contract"], s["q"], z64[:3], s["z"], s["y"])


def test_study_leaves_the_frozen_model_unchanged(synth):
    w_before = {k: v.copy() for k, v in synth["w"].items()}
    q_before = synth["q"].to_json()
    prune.run_study(synth["contract"], synth["q"], synth["z"].astype(np.float64), synth["z"], synth["y"])
    assert all(np.array_equal(contract_weights(synth["contract"])[k], w_before[k]) for k in w_before)
    assert synth["q"].to_json() == q_before


# ---------------------------------------------------------------- script

def _script():
    spec = importlib.util.spec_from_file_location("prune_script", ROOT / "scripts" / "08_pruning_study.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _model_dir(tmp_path, s):
    model_dir = tmp_path / "models"
    write_artifacts(model_dir, s["contract"], s["res"]["config"], training_metadata(s["res"], s["contract"]))
    q = quantize_model(s["contract"], file_sha256(model_dir / "model_contract.json"), s["z"], {"window": "TRAINING"})
    (model_dir / "int8").mkdir()
    (model_dir / "int8" / "quant_params.json").write_text(q.to_json(), encoding="utf-8")
    return model_dir


def _snapshot(d):
    return {str(p.relative_to(d)): file_sha256(p) for p in sorted(d.rglob("*")) if p.is_file()}


def test_script_uses_training_only_and_never_overwrites(tmp_path, synth):
    s = synth
    model_dir = _model_dir(tmp_path, s)
    before = _snapshot(model_dir)
    df = s["df"].copy()
    df.loc[df.window == "OPERATION", ["voltage_v", "irradiance_lux"]] = np.nan   # would break anything using it
    data = tmp_path / "pv_fault_dataset.csv"
    df.to_csv(data, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(to_json(s["a"], {"split_version": "runwise-v1", "dataset_sha256": file_sha256(data)}))
    m = _script()
    kw = dict(expected_dataset_sha256=None, pinned=None, pinned_weights=None, pinned_int8=None, expected_rows=None)
    out = tmp_path / "out"
    r = m.run(data, splits, model_dir, out, code_commit="test", **kw)
    assert r["operation_window_used"] is False and r["window"] == "TRAINING"
    assert r["training_rows"] == (df.window == "TRAINING").sum()
    assert r["retrained_or_fine_tuned"] is False and r["model_artifacts_written"] is False
    assert r["variants"][0]["pruned_int8_vs_frozen_int8"]["agreement"] == 1.0
    assert "not a hardware measurement" in r["estimate_note"]
    md = (out / "results" / "pruning_study.md").read_text()
    assert "OPERATION was not used" in md and "ESTIMATE" in md
    assert sorted(p.name for p in out.rglob("*") if p.is_file()) == ["pruning_study.json", "pruning_study.md"]
    assert _snapshot(model_dir) == before                                     # nothing in models/ changed
    with pytest.raises(FileExistsError):
        m.run(data, splits, model_dir, out, code_commit="test", **kw)


def test_script_refuses_unpinned_int8_parameters(tmp_path, synth):
    model_dir = _model_dir(tmp_path, synth)
    with pytest.raises(ValueError, match="quant_params.json SHA-256"):
        _script().load_frozen_int8(model_dir / "int8")                   # official pins, synthetic file


def test_official_int8_pins_match_the_committed_files():
    m = _script()
    q = m.load_frozen_int8(ROOT / "models" / "int8")
    assert q.contract_sha256 == file_sha256(ROOT / "models" / "model_contract.json")

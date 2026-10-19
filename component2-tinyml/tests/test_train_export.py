import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import tinyml.train as train_mod
from tinyml.contract import LABELS
from tinyml.export import (N_PARAMS, PARAM_LAYOUT, build_contract, contract_feature_config,
                           extract_weights, flatten, forward, load_contract, predict_raw,
                           training_metadata, unflatten, validate_contract, verify_artifacts,
                           weights_sha256, write_artifacts)
from tinyml.features import FEATURES, FeatureConfig, fit_pipeline, transform
from tinyml.models import model_size
from tinyml.splits import make_assignments, to_json
from tinyml.train import final_training_rows, train_final
from tinyml.validate import file_sha256

from test_features import make_dataset

ROOT = Path(__file__).resolve().parents[1]
# Member 3's FL layout (component3-federated-backend/src/mlp.py, LAYOUT), copied for the check
MEMBER3_LAYOUT = [("W1", (8, 16)), ("b1", (16,)), ("W2", (16, 5)), ("b2", (5,))]
ENV = {"python": "test", "numpy": "test", "pandas": "test", "sklearn": "test"}


@pytest.fixture(scope="module")
def df():
    return make_dataset(runs_per_window=5, readings=6)


@pytest.fixture(scope="module")
def assignments(df):
    return make_assignments(df, n_splits=5, seed=42)


@pytest.fixture(scope="module")
def result(df, assignments):
    return train_final(df, seed=0, assignments=assignments)


@pytest.fixture(scope="module")
def contract(result):
    return build_contract(result, "0" * 64, "runwise-v1", "abc1234", ENV)


# ---------------------------------------------------------------- training data

def test_final_rows_are_the_training_window(df, assignments):
    idx = final_training_rows(df, assignments)
    assert set(df.loc[idx, "window"]) == {"TRAINING"}
    assert len(idx) == (df.window == "TRAINING").sum()


def test_split_file_mismatch_is_refused(df, assignments):
    bad = assignments.copy()
    bad.iloc[0, bad.columns.get_loc("temporal")] = "test" if bad.iloc[0].temporal == "train" else "train"
    with pytest.raises(ValueError):
        final_training_rows(df, bad)


def test_feature_pipeline_is_fitted_on_training_rows_only(df, result):
    training = df[df.window == "TRAINING"]
    expected = fit_pipeline(training, fold="final_TRAINING")
    assert result["config"].vref == expected.vref
    assert result["config"].scaler == expected.scaler
    assert result["config"].fitted_on["train_rows"] == len(training)
    assert result["facts"]["rows"] == len(training)


def test_operation_rows_do_not_change_the_model(df, result):
    changed = df.copy()
    op = changed.window == "OPERATION"
    changed.loc[op, "voltage_v"] *= 3
    changed.loc[op, "current_ma"] += 40
    changed.loc[op, "label"] = "DUST"
    again = train_final(changed, seed=0)
    assert again["config"].to_json() == result["config"].to_json()
    np.testing.assert_array_equal(flatten(extract_weights(again["model"])),
                                  flatten(extract_weights(result["model"])))
    only_training = train_final(df[df.window == "TRAINING"], seed=0)
    np.testing.assert_array_equal(flatten(extract_weights(only_training["model"])),
                                  flatten(extract_weights(result["model"])))


def test_dataset_hash_check():
    assert train_mod.check_dataset_sha256(train_mod.DATASET_SHA256)
    with pytest.raises(ValueError):
        train_mod.check_dataset_sha256("0" * 64)


# ---------------------------------------------------------------- architecture and weights

def test_architecture_is_8_16_5_tanh_softmax(result):
    m = result["model"]
    assert [w.shape for w in m.coefs_] == [(8, 16), (16, 5)]
    assert m.activation == "tanh" and m.out_activation_ == "softmax"
    assert m.get_params()["hidden_layer_sizes"] == (16,)


def test_229_parameters(result, contract):
    assert N_PARAMS == 229 == model_size(result["model"])["count"]
    assert contract["n_params"] == 229 and len(flatten(extract_weights(result["model"]))) == 229


def test_layout_matches_member3(contract):
    assert [(n, tuple(s)) for n, s in PARAM_LAYOUT] == MEMBER3_LAYOUT
    assert contract["param_layout"] == [[n, list(s)] for n, s in MEMBER3_LAYOUT]


def test_weight_shapes(result):
    w = extract_weights(result["model"])
    assert {k: v.shape for k, v in w.items()} == dict(MEMBER3_LAYOUT)


def test_output_units_follow_contract_label_order(result):
    m = result["model"]
    assert list(m.classes_) == sorted(LABELS)          # scikit-learn sorts class names
    w = extract_weights(m)
    for i, label in enumerate(LABELS):
        j = list(m.classes_).index(label)
        np.testing.assert_array_equal(w["W2"][:, i], m.coefs_[1][:, j])
        assert w["b2"][i] == m.intercepts_[1][j]


def test_flatten_unflatten_round_trip(result):
    w = extract_weights(result["model"])
    vec = flatten(w)
    np.testing.assert_array_equal(vec[:128], w["W1"].ravel())          # W1 first, row-major
    np.testing.assert_array_equal(vec[128:144], w["b1"])
    np.testing.assert_array_equal(vec[144:224], w["W2"].ravel())
    np.testing.assert_array_equal(vec[224:], w["b2"])
    back = unflatten(vec)
    for k in w:
        np.testing.assert_array_equal(back[k], w[k])
    with pytest.raises(ValueError):
        unflatten(vec[:-1])


# ---------------------------------------------------------------- prediction equivalence

def test_exported_forward_matches_sklearn(result):
    m, X = result["model"], result["X"]
    w = extract_weights(m)
    _, probs = forward(w, X)
    order = [list(m.classes_).index(l) for l in LABELS]
    np.testing.assert_allclose(probs, m.predict_proba(X)[:, order], atol=1e-5)
    assert [LABELS[i] for i in probs.argmax(axis=1)] == list(m.predict(X))


def test_predict_raw_from_contract_matches_pipeline(df, result, contract):
    part = df[df.window == "TRAINING"].copy()
    part.loc[part.index[0], "irradiance_lux"] = 500        # LOW_LIGHT -> skipped
    part.loc[part.index[1], "panel_temp1_c"] = -127        # sensor error -> skipped
    out = predict_raw(part, contract)
    X, status = transform(part, result["config"])
    ok = (status == "OK").to_numpy()
    assert list(out.loc[ok, "predicted_label"]) == list(result["model"].predict(X))
    assert out.loc[part.index[0], "prediction_status"] == "SKIPPED_LOW_LIGHT"
    assert out.loc[part.index[1], "prediction_status"] == "SKIPPED_INVALID_INPUT"
    assert out.loc[~ok, "predicted_label"].isna().all()   # never NORMAL by default
    probs = out.loc[ok, [f"prob_{l.lower()}" for l in LABELS]].to_numpy(float)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


# ---------------------------------------------------------------- contract

def test_contract_order_and_status(contract):
    assert contract["labels"] == list(LABELS)
    assert contract["features"] == list(FEATURES) == list(contract["feature_config"]["features"])
    assert contract["status"] == "TRAINED" and contract["artifact_kind"] == "LOCAL_TRAINED_WEIGHTS"
    assert contract["activation"] == "tanh"
    assert contract["trained_on"]["window"] == "TRAINING"
    assert validate_contract(contract)


def test_contract_does_not_claim_integration(contract):
    s = contract["integration_status"]
    assert s["federated_learning"].startswith("NOT_SUBMITTED")
    assert s["fl_weight_update"].startswith("NOT_PRODUCED")
    assert s["int8_deployment"].startswith("NOT_PRODUCED")
    assert s["esp32_deployment"] == "NOT_DEPLOYED" and s["field_validation"] == "NOT_VALIDATED"
    assert s["relay_control"].startswith("NOT_IMPLEMENTED")
    assert contract["rules"]["status"] == "PLACEHOLDER_NOT_IMPLEMENTED"


@pytest.mark.parametrize("tamper", ["labels", "features", "weight", "status", "n_params",
                                    "activation", "scaler", "window"])
def test_inconsistent_contract_is_refused(contract, tamper):
    c = json.loads(json.dumps(contract))
    if tamper == "labels":
        c["labels"] = c["labels"][::-1]
    elif tamper == "features":
        c["features"] = c["features"][1:] + c["features"][:1]
    elif tamper == "weight":
        c["weights"]["W2"][0][0] += 1e-3
    elif tamper == "status":
        c["status"] = "PLACEHOLDER"
    elif tamper == "n_params":
        c["n_params"] = 213
    elif tamper == "activation":
        c["activation"] = "relu"
    elif tamper == "scaler":
        c["scaler"]["mean"][0] += 1.0
    elif tamper == "window":
        c["trained_on"]["window"] = "OPERATION"
    with pytest.raises(ValueError):
        validate_contract(c)


def test_feature_config_save_load(tmp_path, result, contract):
    cfg = result["config"]
    cfg.save(tmp_path / "fc.json")
    loaded = FeatureConfig.load(tmp_path / "fc.json")
    assert loaded.to_json() == cfg.to_json()
    assert contract_feature_config(contract).to_json() == cfg.to_json()
    assert loaded.low_light_lux == 2400.0 and loaded.lux_saturation == 100000.0   # unchanged


# ---------------------------------------------------------------- determinism and integrity

def test_training_is_deterministic(df, assignments, result, contract):
    again = train_final(df, seed=0, assignments=assignments)
    c2 = build_contract(again, "0" * 64, "runwise-v1", "abc1234", ENV)
    assert c2 == contract
    other = train_final(df, seed=1, assignments=assignments)
    assert weights_sha256(extract_weights(other["model"])) != contract["weights_sha256"]


def test_artifacts_are_written_and_verified(tmp_path, result, contract):
    meta = training_metadata(result, contract, created_utc="2026-01-01T00:00:00Z")
    paths = write_artifacts(tmp_path, contract, result["config"], meta)
    assert sorted(p.name for p in paths) == ["feature_config.json", "manifest.json",
                                             "model_contract.json", "training_metadata.json"]
    assert verify_artifacts(tmp_path)["weights_sha256"] == contract["weights_sha256"]
    assert load_contract(tmp_path / "model_contract.json") == contract
    assert meta["operation_window_used"] is False


def test_tampered_artifact_is_detected(tmp_path, result, contract):
    write_artifacts(tmp_path, contract, result["config"], training_metadata(result, contract))
    p = tmp_path / "feature_config.json"
    p.write_text(p.read_text().replace("2400.0", "2500.0"))
    with pytest.raises(ValueError, match="SHA-256"):
        verify_artifacts(tmp_path)


# ---------------------------------------------------------------- script

def test_script_trains_exports_and_verifies(tmp_path, df, assignments, monkeypatch):
    data = tmp_path / "pv_fault_dataset.csv"
    df.to_csv(data, index=False)
    sha = file_sha256(data)
    monkeypatch.setattr(train_mod, "DATASET_SHA256", sha)
    splits = tmp_path / "splits.json"
    splits.write_text(to_json(assignments, {"split_version": "runwise-v1", "dataset_sha256": sha}))
    spec = importlib.util.spec_from_file_location("train_final_script", ROOT / "scripts" / "05_train_final.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    result, contract, meta, _ = script.run(data, splits, tmp_path / "models", code_commit="abc1234")
    assert contract["trained_on"]["dataset_sha256"] == sha
    assert contract["trained_on"]["git_commit"] == "abc1234"
    assert meta["window"] == "TRAINING" and meta["seed"] == 0
    assert verify_artifacts(tmp_path / "models")

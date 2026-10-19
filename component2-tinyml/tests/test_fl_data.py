"""pv8-v1 FL data adapter (fl_data.py) and Model v1 through Member 3's MLP.

Unit tests use synthetic runs and the frozen FeatureConfig. The Member 3
checks import component3-federated-backend/src/mlp.py read-only from the
integration branch and parse pv_data.py / engine.py / fl_baseline.py without
running them; they skip when that folder is absent. The dataset checks use
TRAINING rows only and skip when the dataset file is absent. Nothing is
trained and no accuracy is measured."""
import ast
import json

import numpy as np
import pytest

from test_features import make_dataset
from tinyml import fl_data as fd
from tinyml.contract import FL_RAW_FEATURES, LABELS
from tinyml.export import N_PARAMS, PARAM_LAYOUT, file_sha256, flatten, forward, weights_sha256
from tinyml.features import FEATURES, STATUS_OK, transform
from tinyml.splits import load_assignments, make_assignments

WEIGHTS_SHA256 = "94e7ab9384a1906ec2cf0d3176b435e700f15cd99552fbc6c71519c648ecc3a7"
DATASET = fd.MEMBER3_ROOT / "data" / "shared" / "pv_fault_dataset.csv"
needs_member3 = pytest.mark.skipif(not (fd.MEMBER3_ROOT / "src" / "mlp.py").exists(),
                                   reason="Member 3 code not in this checkout")
needs_dataset = pytest.mark.skipif(not DATASET.exists(), reason="dataset not in this checkout")


@pytest.fixture(scope="module")
def frozen():
    contract, cfg, w = fd.load_frozen_model()
    return {"contract": contract, "cfg": cfg, "w": w}


@pytest.fixture(scope="module")
def synth(frozen):
    df = make_dataset(runs_per_window=5, readings=4)
    a = make_assignments(df, n_splits=5, seed=42)
    return df, a, fd.build_fl_data(df, frozen["cfg"], a)


def _m3_constants(name):
    tree = ast.parse((fd.MEMBER3_ROOT / "src" / f"{name}.py").read_text(encoding="utf-8"))
    return {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, (ast.List, ast.Tuple))}


def _m3_default(name, func, arg):
    tree = ast.parse((fd.MEMBER3_ROOT / "src" / f"{name}.py").read_text(encoding="utf-8"))
    f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func)
    names = [a.arg for a in f.args.args]
    return ast.literal_eval(f.args.defaults[names.index(arg) - (len(names) - len(f.args.defaults))])


# ---------------------------------------------------------------- frozen Model v1

def test_frozen_model_v1_is_loaded_with_its_hashes(frozen):
    c, cfg, w = frozen["contract"], frozen["cfg"], frozen["w"]
    assert c["model_version"] == "c2-local-v1" and weights_sha256(w) == WEIGHTS_SHA256
    assert {k: v.shape for k, v in w.items()} == dict(PARAM_LAYOUT) and flatten(w).size == N_PARAMS == 229
    assert c["activation"] == "tanh" and c["output_activation"] == "softmax"
    assert c["labels"] == list(LABELS) and c["features"] == list(FEATURES) == list(cfg.features)
    manifest = json.loads((fd.MODELS_DIR / "manifest.json").read_text())
    assert file_sha256(fd.MODELS_DIR / "feature_config.json") == manifest["files"]["feature_config.json"]
    assert cfg.scaler == c["scaler"] and cfg.fitted_on["fold"] == "final_TRAINING"


# ---------------------------------------------------------------- adapter on synthetic runs

def test_operation_rows_never_reach_the_feature_code(frozen, monkeypatch):
    df = make_dataset(runs_per_window=5, readings=4)
    seen = []

    def spy(part, cfg):
        seen.extend(part["window"])
        return transform(part, cfg)
    monkeypatch.setattr(fd, "transform", spy)
    d = fd.build_fl_data(df, frozen["cfg"], make_assignments(df, n_splits=5, seed=42))
    assert seen and set(seen) == {"TRAINING"} and d.meta["operation_rows_used"] == 0


def test_features_equal_the_existing_pipeline_and_scaler_is_not_refitted(frozen, synth):
    df, a, d = synth
    before = frozen["cfg"].to_json()
    tr = df[df.window == "TRAINING"]
    val = tr.run_id.isin(fd.validation_runs(a))
    for s, (X, y) in d.site_data.items():
        part = tr[(tr.site_id == s) & ~val]
        Xe, st = transform(part, frozen["cfg"])
        ok = (st == STATUS_OK).to_numpy()
        assert np.array_equal(X, Xe.astype(np.float64)) and X.dtype == np.float64
        assert list(y) == [LABELS.index(l) for l in part.label[ok]]
    assert d.cfg is frozen["cfg"] and frozen["cfg"].to_json() == before


def test_validation_is_run_wise_from_training_only(synth):
    df, a, d = synth
    tr = df[df.window == "TRAINING"]
    val_runs = fd.validation_runs(a)
    assert val_runs and set(a.loc[sorted(val_runs), "window"]) == {"TRAINING"}
    assert set(a.loc[sorted(val_runs), "cv_fold"]) == {fd.VALIDATION_FOLD}
    assert d.meta["validation"]["runs"] == sorted(val_runs)
    for s, c in d.meta["sites"].items():
        runs = tr[tr.site_id == s].run_id
        assert c["train_runs"] + c["val_runs"] == runs.nunique()          # no run in both
        assert c["val_runs"] == runs[runs.isin(val_runs)].nunique()
    n_ok = sum(len(y) for _, y in d.site_data.values()) + len(d.val[1])
    assert n_ok == (transform(tr, d.cfg)[1] == STATUS_OK).sum()            # every OK TRAINING row, once
    assert len(d.val[1]) == sum(len(v[1]) for v in d.site_val.values())


def test_site_partitions_and_label_order(synth):
    df, _, d = synth
    assert sorted(d.site_data) == sorted(df[df.window == "TRAINING"].site_id.unique())
    assert fd.LABEL_ID == {l: i for i, l in enumerate(LABELS)} and d.meta["labels"] == list(LABELS)
    for X, y in d.site_data.values():
        assert X.shape[1] == len(FEATURES) == 8 and np.isfinite(X).all() and set(y) <= set(range(5))


@pytest.mark.parametrize("change, error", [
    (lambda df, a: (df[df.window == "OPERATION"], a), "no TRAINING rows"),
    (lambda df, a: (df, a.drop(index=df[df.window == "TRAINING"].run_id.iloc[0])), "not in the split file"),
    (lambda df, a: (df.assign(label=df.label.replace("DUST", "SOILING")), a), "unknown labels"),
    (lambda df, a: (df.drop(columns="site_id"), a), "site_id"),
])
def test_bad_inputs_are_refused(frozen, change, error):
    df = make_dataset(runs_per_window=5, readings=4)
    df2, a2 = change(df, make_assignments(df, n_splits=5, seed=42))
    with pytest.raises((ValueError, KeyError), match=error):
        fd.build_fl_data(df2, frozen["cfg"], a2)


def test_unfitted_or_other_feature_config_is_refused(synth):
    from tinyml.features import FeatureConfig
    df, a, _ = synth
    with pytest.raises(ValueError, match="frozen pv8-v1"):
        fd.build_fl_data(df, FeatureConfig(), a)


# ---------------------------------------------------------------- Member 3's code (read-only)

@needs_member3
def test_member3_constants_match_the_contract():
    pv = _m3_constants("pv_data")
    assert pv["LABELS"] == list(LABELS)                                   # same label order
    assert pv["FEATURES"] == list(FL_RAW_FEATURES)                        # raw8 = the adapter's raw inputs
    assert pv["FEATURES"] != list(FEATURES)                               # but not the model's inputs
    assert [(n, tuple(s)) for n, s in _m3_constants("mlp")["LAYOUT"]] == list(PARAM_LAYOUT)


@needs_member3
def test_member3_activation_defaults_are_recorded():
    assert _m3_default("engine", "run_fl", "activation") == "tanh"
    assert _m3_default("fl_baseline", "run_baseline", "activation") == "relu"
    assert _m3_default("mlp", "forward", "activation") == "relu"


@needs_member3
def test_member3_flatten_order_equals_ours(frozen):
    mlp = fd.load_member3_module("mlp")
    w = frozen["w"]
    assert np.array_equal(mlp.flatten(w), flatten(w))
    back = mlp.unflatten(flatten(w))
    assert all(np.array_equal(back[k], w[k]) for k in w)


@needs_member3
def test_model_v1_through_member3_mlp_reproduces_the_reference_probes(frozen):
    mlp = fd.load_member3_module("mlp")
    p = frozen["contract"]["reference_probes"]
    _, _, probs = mlp.forward(frozen["w"], np.asarray(p["inputs_scaled"]), "tanh")
    assert np.allclose(probs, p["probabilities"], atol=1e-12)
    assert [LABELS[i] for i in probs.argmax(axis=1)] == p["labels"]


@needs_member3
def test_member3_default_relu_changes_model_v1_outputs(frozen):
    """Member 3's mlp default is relu: Model v1 must always be run with tanh."""
    mlp = fd.load_member3_module("mlp")
    X = np.asarray(frozen["contract"]["reference_probes"]["inputs_scaled"])
    _, _, relu = mlp.forward(frozen["w"], X)
    assert np.abs(relu - forward(frozen["w"], X)[1]).max() > 0.05


# ---------------------------------------------------------------- the real dataset (TRAINING rows only)

@pytest.fixture(scope="module")
def real():
    if not DATASET.exists():
        pytest.skip("dataset not in this checkout")
    site_data, val, cfg = fd.member3_get_data(DATASET)
    a, _ = load_assignments(fd.SPLITS_PATH, dataset_sha256=file_sha256(DATASET))
    return site_data, val, cfg, a


@needs_dataset
def test_real_dataset_partitions(real):
    site_data, val, _, a = real
    assert sorted(site_data) == ["SITE-A", "SITE-B", "SITE-C", "SITE-D", "SITE-E", "SITE-F"]
    rows = {s: len(y) for s, (_, y) in site_data.items()}
    assert rows == {"SITE-A": 540, "SITE-B": 240, "SITE-C": 300, "SITE-D": 420, "SITE-E": 704, "SITE-F": 120}
    assert len(val[1]) == 600 and np.bincount(val[1], minlength=5).tolist() == [120] * 5
    assert sum(rows.values()) + len(val[1]) == 2924                       # all status-OK TRAINING rows


@needs_member3
@needs_dataset
def test_model_v1_predictions_match_through_member3_mlp(frozen, real):
    """Same weights and inputs, two implementations: identical outputs.
    An implementation check on TRAINING rows, not an accuracy measurement."""
    site_data, val, _, _ = real
    mlp = fd.load_member3_module("mlp")
    X = np.concatenate([x for x, _ in site_data.values()] + [val[0]])
    _, _, m3 = mlp.forward(frozen["w"], X, "tanh")
    _, m2 = forward(frozen["w"], X)
    assert len(X) == 2924 and np.abs(m3 - m2).max() < 1e-12
    assert np.array_equal(m3.argmax(axis=1), m2.argmax(axis=1))

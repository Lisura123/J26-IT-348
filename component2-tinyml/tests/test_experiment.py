import numpy as np
import pandas as pd
import pytest

from tinyml.contract import LABELS
from tinyml.evaluate import classification_metrics, coverage, run_level
from tinyml.experiment import (ablation_table, feature_set_matrices, protocol_folds, run_comparison,
                               summary_table)
from tinyml.features import FEATURES, fold_matrices
from tinyml.models import ThresholdRules, is_applicable, make_model, model_size
from tinyml.splits import iter_folds, make_assignments

from test_features import make_dataset


@pytest.fixture(scope="module")
def df():
    return make_dataset(runs_per_window=5, readings=6)


@pytest.fixture(scope="module")
def assignments(df):
    return make_assignments(df, n_splits=5, seed=42)


@pytest.fixture(scope="module")
def fold(df, assignments):
    training = df[df.window == "TRAINING"]
    name, tr, te = next(iter_folds(training, assignments, "cv"))
    return fold_matrices(training, tr, te, fold=name)


# ---------------------------------------------------------------- metrics

def test_perfect_predictions():
    y = np.array(LABELS * 3)
    m = classification_metrics(y, y)
    assert m["macro_f1"] == 1.0 and m["accuracy"] == 1.0
    assert m["safety_recall"] == {"OPEN_CIRCUIT": 1.0, "SHORT_CIRCUIT": 1.0}
    assert np.trace(np.array(m["confusion_matrix"])) == len(y)


def test_absent_class_is_excluded_from_macro_f1():
    y = np.array(["NORMAL", "DUST", "NORMAL", "DUST"])
    p = np.array(["NORMAL", "DUST", "DUST", "DUST"])
    m = classification_metrics(y, p)
    assert m["classes_present"] == ["NORMAL", "DUST"]
    assert m["per_class"]["OPEN_CIRCUIT"]["recall"] is None
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 1.0) / 2)


def test_run_level_majority_vote():
    r = run_level(["DUST"] * 3 + ["NORMAL"] * 3, ["DUST", "DUST", "NORMAL", "DUST", "DUST", "NORMAL"],
                  ["a"] * 3 + ["b"] * 3)
    assert r["runs"] == 2 and r["majority_vote_accuracy"] == 0.5
    assert r["worst_run"] == "b" and r["worst_run_accuracy"] == pytest.approx(1 / 3)


def test_coverage_reports_skipped_readings(df):
    part = df.iloc[:12]
    status = pd.Series("OK", index=part.index)
    status.iloc[:6] = "LOW_LIGHT"
    c = coverage(part, status)
    assert (c["total"], c["classified"], c["skipped"]) == (12, 6, 6)
    assert c["skip_reasons"] == {"LOW_LIGHT": 6}
    assert part.run_id.iloc[0] in c["runs_not_fully_classified"]


# ---------------------------------------------------------------- models

def test_rules_learn_thresholds_on_training_data(fold):
    X, y = fold["train"]["X"], fold["train"]["y"]
    rules = ThresholdRules(FEATURES).fit(X, y)
    assert set(rules.thresholds_) == {"OPEN_CIRCUIT", "SHORT_CIRCUIT", "DUST", "PARTIAL_SHADING"}
    assert (rules.predict(X) == y).mean() > 0.9           # synthetic test data is separable


def test_rules_need_their_features():
    assert not is_applicable("rules", ["voltage_v", "current_ma"])
    assert is_applicable("rules", FEATURES)
    with pytest.raises(ValueError):
        ThresholdRules(["voltage_v"])


@pytest.mark.parametrize("name,kind", [("logreg", "parameters"), ("tree", "tree_nodes"),
                                       ("forest", "tree_nodes"), ("mlp", "parameters")])
def test_models_fit_predict_and_report_size(fold, name, kind):
    m = make_model(name, seed=0, feature_names=FEATURES).fit(fold["train"]["X"], fold["train"]["y"])
    pred = m.predict(fold["test"]["X"])
    assert len(pred) == len(fold["test"]["y"]) and set(pred) <= set(LABELS)
    assert model_size(m)["kind"] == kind


def test_mlp_is_8_16_5_tanh(fold):
    m = make_model("mlp", seed=0).fit(fold["train"]["X"], fold["train"]["y"])
    assert [w.shape for w in m.coefs_] == [(8, 16), (16, 5)]
    assert m.activation == "tanh"
    assert model_size(m)["count"] == 229


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        make_model("svm")


# ---------------------------------------------------------------- feature sets / protocols

def test_feature_sets_keep_the_same_rows(df, fold):
    training = df[df.window == "TRAINING"]
    shapes = {fs: feature_set_matrices(training, fold, fs) for fs in
              ("physics8", "physics7_no_v_deficit", "raw8")}
    assert shapes["physics8"][0].shape[1] == 8
    assert shapes["physics7_no_v_deficit"][0].shape[1] == 7
    assert "v_deficit" not in shapes["physics7_no_v_deficit"][2]
    assert "health_score" not in shapes["raw8"][2]
    assert len({v[0].shape[0] for v in shapes.values()}) == 1   # same training rows
    assert len({v[1].shape[0] for v in shapes.values()}) == 1   # same test rows


def test_raw8_scaler_uses_training_rows_only(df, fold):
    training = df[df.window == "TRAINING"]
    Xtr, _, _ = feature_set_matrices(training, fold, "raw8")
    assert np.allclose(Xtr.mean(axis=0), 0, atol=1e-4)
    changed = training.copy()
    changed.loc[fold["test"]["index"], "voltage_v"] += 5
    Xtr2, _, _ = feature_set_matrices(changed, fold, "raw8")
    np.testing.assert_array_equal(Xtr, Xtr2)


def test_final_test_is_refused_without_explicit_permission(df, assignments):
    with pytest.raises(PermissionError):
        protocol_folds(df, assignments, "temporal")
    assert len(protocol_folds(df, assignments, "temporal", allow_final_test=True)) == 1


@pytest.mark.parametrize("protocol", ["cv", "loso_training"])
def test_model_selection_protocols_never_use_operation(df, assignments, protocol):
    folds = protocol_folds(df, assignments, protocol)
    assert folds
    for _, tr, te in folds:
        assert set(df.loc[list(tr) + list(te), "window"]) == {"TRAINING"}
        assert not set(df.loc[tr, "run_id"]) & set(df.loc[te, "run_id"])


# ---------------------------------------------------------------- runner

@pytest.fixture(scope="module")
def result(df, assignments):
    return run_comparison(df, assignments, protocols=("cv",), models=("rules", "logreg", "mlp"),
                          feature_sets=("physics8", "physics7_no_v_deficit", "raw8"),
                          seeds=(0, 1), progress=lambda s: None)


def test_runner_records(result):
    recs = result["records"]
    combos = {(r["feature_set"], r["model"]) for r in recs}
    assert ("raw8", "rules") not in combos                  # not applicable, skipped
    assert {r["seed"] for r in recs if r["model"] == "logreg"} == {0}      # deterministic
    assert {r["seed"] for r in recs if r["model"] == "mlp"} == {0, 1}
    assert result["environment"]["sklearn"]
    assert all(r["coverage"]["total"] >= r["test_rows"] for r in recs)


def test_runner_is_reproducible(df, assignments, result):
    again = run_comparison(df, assignments, protocols=("cv",), models=("rules", "logreg", "mlp"),
                           feature_sets=("physics8", "physics7_no_v_deficit", "raw8"),
                           seeds=(0, 1), progress=lambda s: None)
    a = [(r["fold"], r["model"], r["seed"], r["metrics"]["macro_f1"]) for r in result["records"]]
    b = [(r["fold"], r["model"], r["seed"], r["metrics"]["macro_f1"]) for r in again["records"]]
    assert a == b


def test_summary_and_ablation_tables(result):
    t = summary_table(result)
    assert {"macro_f1_mean", "open_recall_mean", "size"} <= set(t.columns)
    ab = ablation_table(t, "cv")
    assert set(ab.index) == {"rules", "logreg", "mlp"}
    assert np.allclose(ab.difference, ab.with_v_deficit - ab.without_v_deficit)


# ---------------------------------------------------------------- pooled / run-level / class presence

def test_mlp_parameter_counts_per_feature_set(fold, df):
    training = df[df.window == "TRAINING"]
    counts = {}
    for fs in ("physics8", "physics7_no_v_deficit", "raw8"):
        X_tr, _, _ = feature_set_matrices(training, fold, fs)
        m = make_model("mlp", seed=0).fit(X_tr, fold["train"]["y"])
        counts[fs] = model_size(m)["count"]
    assert counts == {"physics8": 229, "physics7_no_v_deficit": 213, "raw8": 229}


def test_pooled_metrics_use_combined_held_out_predictions(df, assignments, result):
    pooled = {(p["feature_set"], p["model"], p["seed"]): p for p in result["pooled"]}
    key = ("physics8", "logreg", 0)
    folds = [r for r in result["records"] if (r["feature_set"], r["model"], r["seed"]) == key]
    assert pooled[key]["folds"] == len(folds) == 5
    assert pooled[key]["metrics"]["n"] == sum(r["metrics"]["n"] for r in folds)
    # recompute from scratch: refit each fold and concatenate predictions
    training = df[df.window == "TRAINING"]
    ys, ps = [], []
    for name, tr, te in iter_folds(training, assignments, "cv"):
        f = fold_matrices(training, tr, te, fold=name)
        m = make_model("logreg").fit(f["train"]["X"], f["train"]["y"])
        ys.append(f["test"]["y"]); ps.append(m.predict(f["test"]["X"]))
    expected = classification_metrics(np.concatenate(ys), np.concatenate(ps))
    assert pooled[key]["metrics"]["macro_f1"] == pytest.approx(expected["macro_f1"])


def test_run_level_is_reported_separately(result):
    p = result["pooled"][0]
    assert {"majority_vote_accuracy", "worst_run_accuracy", "runs"} <= set(p["run_level"])
    t = summary_table(result)
    assert {"pooled_macro_f1", "run_vote_acc", "worst_run_acc"} <= set(t.columns)


def test_loso_reports_classes_and_no_recall_for_absent_classes():
    data = make_dataset(runs_per_window=5, readings=4)
    # remove all DUST runs from SITE-B so its held-out partition lacks DUST
    data = data[~((data.site_id == "SITE-B") & (data.label == "DUST"))].reset_index(drop=True)
    a = make_assignments(data, n_splits=2, seed=1)
    res = run_comparison(data, a, protocols=("loso_training",), models=("logreg",),
                         feature_sets=("physics8",), progress=lambda s: None)
    tc = {t["fold"]: t for t in res["test_classes"]}
    assert "DUST" not in tc["loso_SITE-B"]["classes_in_partition"]
    rec = next(r for r in res["records"] if r["fold"] == "loso_SITE-B")
    assert rec["metrics"]["per_class"]["DUST"]["recall"] is None
    assert "DUST" not in rec["metrics"]["classes_present"]

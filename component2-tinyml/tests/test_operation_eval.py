"""Phase 9 evaluation pipeline, tested on synthetic data only.

The real OPERATION window is never read here. The one test that touches the
committed models/ directory only checks file hashes."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tinyml.holdout as holdout
from tinyml.contract import LABELS
from tinyml.evaluate import classification_metrics
from tinyml.export import build_contract, training_metadata, write_artifacts
from tinyml.holdout import (PINNED_FILES, bootstrap_ci, confusion_scores, coverage_adjusted_recall,
                            evaluation_rows, inference_frame, max_streak, run_evaluation, safety_analysis,
                            verify_frozen_model)
from tinyml.splits import make_assignments, to_json
from tinyml.train import train_final
from tinyml.validate import file_sha256

from test_features import make_dataset

ROOT = Path(__file__).resolve().parents[1]
SYNTH = dict(pinned=None, pinned_weights=None, expected_dataset_sha256=None, expected_rows=None,
             expected_runs=None, bootstrap_b=50, latency_n=20)


@pytest.fixture(scope="module")
def df():
    return make_dataset(runs_per_window=5, readings=6)


@pytest.fixture(scope="module")
def assignments(df):
    return make_assignments(df, n_splits=5, seed=42)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, df, assignments):
    """A synthetic frozen model trained on the synthetic TRAINING window."""
    out = tmp_path_factory.mktemp("models")
    result = train_final(df, seed=0, assignments=assignments)
    contract = build_contract(result, "0" * 64, "runwise-v1", "test", {"python": "t"})
    write_artifacts(out, contract, result["config"], training_metadata(result, contract))
    return out


def evaluate(df, assignments, model_dir, **kw):
    args = dict(SYNTH, **kw)
    return run_evaluation(df, assignments, model_dir, "0" * 64, **args)


# ---------------------------------------------------------------- guards

def test_final_test_requires_explicit_confirmation(df, assignments, model_dir):
    with pytest.raises(PermissionError):
        evaluate(df, assignments, model_dir)
    with pytest.raises(ValueError):
        evaluate(df, assignments, model_dir, confirm_final_test=True, dry_run_training=True)


def test_committed_artifacts_match_pinned_hashes():
    contract = verify_frozen_model(ROOT / "models")         # hashes only; no data is read
    assert contract["weights_sha256"] == holdout.PINNED_WEIGHTS_SHA256
    assert {n: file_sha256(ROOT / "models" / n) for n in PINNED_FILES} == PINNED_FILES


def test_changed_artifacts_are_refused(tmp_path, model_dir):
    for f in Path(model_dir).iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    with pytest.raises(ValueError, match="committed"):
        verify_frozen_model(tmp_path, pinned={"model_contract.json": "0" * 64}, pinned_weights=None)
    p = tmp_path / "feature_config.json"
    p.write_text(p.read_text().replace("2400.0", "2300.0"))
    with pytest.raises(ValueError, match="SHA-256"):
        verify_frozen_model(tmp_path, pinned=None, pinned_weights=None)


def test_dataset_hash_is_checked(df, assignments, model_dir):
    with pytest.raises(ValueError, match="dataset"):
        run_evaluation(df, assignments, model_dir, "1" * 64, confirm_final_test=True,
                       **dict(SYNTH, expected_dataset_sha256="0" * 64))


# ---------------------------------------------------------------- rows and inputs

def test_only_operation_rows_are_evaluated(df, assignments, model_dir):
    r = evaluate(df, assignments, model_dir, confirm_final_test=True)
    assert r["window"] == "OPERATION" and r["status"] == "FINAL_TEST_OPERATION"
    assert r["rows"]["total"] == (df.window == "OPERATION").sum()
    idx = evaluation_rows(df, assignments)
    assert set(df.loc[idx, "window"]) == {"OPERATION"}
    assert not set(df.loc[idx, "run_id"]) & set(df.loc[df.window == "TRAINING", "run_id"])


def test_split_and_window_must_agree(df, assignments):
    bad = assignments.copy()
    run = bad.index[bad.temporal == "test"][0]
    bad.loc[run, "temporal"] = "train"
    with pytest.raises(ValueError):
        evaluation_rows(df, bad)


def test_inference_never_receives_labels_or_health_score(df, assignments, model_dir, monkeypatch):
    seen = []
    real = holdout.predict_raw

    def spy(frame, contract):
        seen.append(set(frame.columns))
        return real(frame, contract)
    monkeypatch.setattr(holdout, "predict_raw", spy)
    evaluate(df, assignments, model_dir, confirm_final_test=True)
    assert seen and all(not {"label", "health_score"} & cols for cols in seen)
    assert "label" not in inference_frame(df).columns and "health_score" not in inference_frame(df).columns


def test_nothing_is_fitted_during_evaluation(df, assignments, model_dir, monkeypatch):
    import sklearn.neural_network
    import tinyml.features
    import tinyml.models
    import tinyml.train

    def forbidden(*a, **k):
        raise AssertionError("fitting is not allowed during evaluation")
    for mod, name in ((tinyml.features, "fit_pipeline"), (tinyml.train, "train_final"),
                      (tinyml.train, "fit_pipeline"), (tinyml.models, "make_model")):
        monkeypatch.setattr(mod, name, forbidden)
    monkeypatch.setattr(sklearn.neural_network.MLPClassifier, "fit", forbidden)
    r = evaluate(df, assignments, model_dir, confirm_final_test=True)
    assert r["frozen"]["refit"] is False
    assert not {"fit_pipeline", "train_final", "make_model"} & set(vars(holdout))


def test_evaluation_leaves_artifacts_unchanged(df, assignments, model_dir):
    before = {f.name: file_sha256(f) for f in Path(model_dir).iterdir()}
    evaluate(df, assignments, model_dir, confirm_final_test=True)
    assert {f.name: file_sha256(f) for f in Path(model_dir).iterdir()} == before


# ---------------------------------------------------------------- gates and coverage

def test_skipped_readings_are_never_counted_as_normal(df, assignments, model_dir):
    d = df.copy()
    op_normal = d.index[(d.window == "OPERATION") & (d.label == "NORMAL")][:4]
    d.loc[op_normal, "irradiance_lux"] = 500                        # LOW_LIGHT
    r = evaluate(d, assignments, model_dir, confirm_final_test=True)
    assert r["coverage"]["skip_reasons"] == {"LOW_LIGHT": 4}
    support = r["classification"]["per_class"]["NORMAL"]["support"]
    assert support == (d.loc[evaluation_rows(d, assignments), "label"] == "NORMAL").sum() - 4
    car = r["coverage_adjusted_recall"]["NORMAL"]
    assert car["total"] == support + 4                              # skipped stay in the denominator
    assert car["correct"] <= support                                # and are never counted as correct
    assert r["coverage"]["per_class"]["NORMAL"]["classified"] == support


def test_coverage_adjusted_recall_counts_skipped_as_missed():
    labels = ["DUST"] * 4 + ["NORMAL"] * 2
    pred = ["DUST", "DUST", None, None, "NORMAL", "DUST"]
    car = coverage_adjusted_recall(labels, pred)
    assert car["DUST"] == {"total": 4, "correct": 2, "recall": 0.5}
    assert car["NORMAL"]["recall"] == 0.5
    assert car["OPEN_CIRCUIT"]["recall"] is None


# ---------------------------------------------------------------- safety analysis

def _toy():
    """Hand-checked example: one OPEN run, one SHORT run, one shading run."""
    rows = []
    for run, label, preds in (("R-OPE", "OPEN_CIRCUIT", ["NORMAL", "OPEN_CIRCUIT", "OPEN_CIRCUIT", None]),
                              ("R-SHO", "SHORT_CIRCUIT", ["SHORT_CIRCUIT"] * 4),
                              ("R-PAR", "PARTIAL_SHADING", ["OPEN_CIRCUIT", "OPEN_CIRCUIT", "PARTIAL_SHADING",
                                                            "OPEN_CIRCUIT"])):
        for k, p in enumerate(preds):
            rows.append({"run_id": run, "site_id": "SITE-A", "label": label, "reading": k + 1,
                         "current_ma": 1.0 + k, "voltage_v": 2.0, "irradiance_lux": 30000.0, "pred": p})
    d = pd.DataFrame(rows)
    pred = pd.DataFrame({"prediction_status": np.where(d.pred.isna(), "SKIPPED_LOW_LIGHT", "CLASSIFIED"),
                         "predicted_label": d.pred}, index=d.index)
    return d.drop(columns="pred"), pred


def test_safety_analysis_hand_checked():
    d, pred = _toy()
    s = safety_analysis(d, pred)
    op = s["OPEN_CIRCUIT"]
    assert op["false_predictions"]["count"] == 3
    assert op["false_predictions"]["by_true_label"] == {"PARTIAL_SHADING": 3}
    assert op["false_predictions"]["rate_per_1000_classified_non_fault"] == pytest.approx(1000 * 3 / 8)
    (run,) = op["false_predictions"]["runs"]
    assert run["run_id"] == "R-PAR" and run["max_consecutive"] == 2
    mf = op["missed_faults"]
    assert (mf["fault_readings"], mf["classified"], mf["correct"], mf["missed"], mf["skipped"]) == (4, 3, 2, 1, 1)
    assert mf["recall"] == pytest.approx(2 / 3) and mf["coverage_adjusted_recall"] == 0.5
    (r,) = mf["runs"]
    assert r["missed_as"] == {"NORMAL": 1} and r["time_to_first_detection_s"] == 5.0
    sh = s["SHORT_CIRCUIT"]
    assert sh["false_predictions"]["count"] == 0 and sh["missed_faults"]["recall"] == 1.0
    assert sh["missed_faults"]["runs"][0]["time_to_first_detection_s"] == 0.0
    assert s["OPEN_CIRCUIT"]["electrical_confirmation"].startswith("NOT_ASSESSED")


def test_max_streak():
    assert max_streak([]) == 0
    assert max_streak([True, True, False, True, True, True]) == 3


# ---------------------------------------------------------------- bootstrap

def test_confusion_scores_match_classification_metrics():
    rng = np.random.default_rng(1)
    y = rng.choice(LABELS, 300)
    p = np.where(rng.random(300) < 0.8, y, rng.choice(LABELS, 300))
    m = classification_metrics(y, p)
    s = confusion_scores(m["confusion_matrix"])
    assert s["macro_f1"] == pytest.approx(m["macro_f1"])
    assert s["accuracy"] == pytest.approx(m["accuracy"])


def test_bootstrap_is_deterministic_and_stratified():
    rng = np.random.default_rng(2)
    runs = np.repeat([f"R{i}" for i in range(10)], 6)
    run_labels = {f"R{i}": LABELS[i % 5] for i in range(10)}
    y = np.array([run_labels[r] for r in runs])
    p = np.where(rng.random(60) < 0.85, y, "NORMAL")
    a = bootstrap_ci(y, p, runs, run_labels, b=200, seed=42)
    b = bootstrap_ci(y, p, runs, run_labels, b=200, seed=42)
    assert a == b
    assert a["runs_per_class"] == {l: 2 for l in sorted(LABELS)}
    assert 0 <= a["macro_f1"]["low"] <= a["macro_f1"]["high"] <= 1


# ---------------------------------------------------------------- latency and report

def test_latency_is_host_cpu_on_training_rows(df, assignments, model_dir, monkeypatch):
    seen = []
    real = holdout.host_latency

    def spy(rows, contract, **kw):
        seen.append(set(rows["window"]))
        return real(rows, contract, **kw)
    monkeypatch.setattr(holdout, "host_latency", spy)
    r = evaluate(df, assignments, model_dir, confirm_final_test=True)
    assert seen == [{"TRAINING"}]
    lt = r["latency"]
    assert lt["scope"] == "HOST_CPU_NOT_ESP32"
    assert {"forward_pass_single_reading", "pandas_reference_pipeline_single_reading",
            "batched_pipeline", "host"} <= set(lt)
    assert lt["forward_pass_single_reading"]["n"] == 20


def test_result_records_provenance_and_caveats(df, assignments, model_dir):
    r = evaluate(df, assignments, model_dir, confirm_final_test=True)
    assert r["provenance"]["weights_sha256"] and r["provenance"]["artifact_sha256"]
    assert r["frozen"] == {"refit": False, "retrained": False, "thresholds_changed": False,
                           "inference": "contract-only NumPy (export.predict_raw)",
                           "inputs_excluded": ["label", "health_score"]}
    assert any("Relay isolation is not implemented" in c for c in r["caveats"])
    assert len(r["classification"]["confusion_matrix_row_normalised"]) == 5
    json.dumps(r)                                                   # serialisable


# ---------------------------------------------------------------- script

def _script():
    spec = importlib.util.spec_from_file_location("op_eval", ROOT / "scripts" / "06_evaluate_operation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _files(tmp_path, df, assignments):
    data = tmp_path / "pv_fault_dataset.csv"
    df.to_csv(data, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(to_json(assignments, {"split_version": "runwise-v1", "dataset_sha256": file_sha256(data)}))
    return data, splits


def test_script_final_test_runs_once_and_never_overwrites(tmp_path, df, assignments, model_dir):
    data, splits = _files(tmp_path, df, assignments)
    s = _script()
    out = tmp_path / "results"
    kw = dict(SYNTH, official_result=tmp_path / "not_yet_recorded.json")   # a checkout before Phase 9
    r, paths = s.run(data, splits, model_dir, out, confirm_final_test=True, code_commit="abc", **kw)
    assert [p.name for p in paths] == ["operation_eval.json", "operation_eval.md"]
    first = paths[0].read_bytes()
    assert "Independent OPERATION evaluation" in paths[1].read_text()
    with pytest.raises(FileExistsError):
        s.run(data, splits, model_dir, out, confirm_final_test=True, code_commit="abc", **kw)
    assert paths[0].read_bytes() == first
    with pytest.raises(PermissionError):
        s.run(data, splits, model_dir, tmp_path / "other", code_commit="abc", **kw)


def test_script_dry_run_uses_training_and_is_labelled(tmp_path, df, assignments, model_dir):
    data, splits = _files(tmp_path, df, assignments)
    r, paths = _script().run(data, splits, model_dir, tmp_path / "dry", dry_run_training=True,
                             code_commit="abc", **SYNTH)
    assert r["status"] == "DRY_RUN_TRAINING_NOT_A_RESULT" and r["window"] == "TRAINING"
    assert [p.name for p in paths] == ["dry_run_training_eval.json", "dry_run_training_eval.md"]
    assert "NOT A RESULT" in paths[1].read_text()


def test_script_refuses_to_repeat_the_recorded_operation_evaluation(tmp_path):
    """The committed Phase 9 result blocks a second OPERATION evaluation before
    any data is read, also with confirm_final_test=True (as an executed
    notebook saved with CONFIRM_FINAL_TEST = True would call it)."""
    s = _script()
    assert s.OFFICIAL_RESULT == ROOT / "results" / "operation_eval.json" and s.OFFICIAL_RESULT.exists()
    missing = tmp_path / "no_such_dataset.csv"                     # never opened
    with pytest.raises(RuntimeError, match="already recorded"):
        s.run(missing, missing, ROOT / "models", tmp_path / "out", confirm_final_test=True, code_commit="abc")
    assert not (tmp_path / "out").exists()
    with pytest.raises(FileNotFoundError):                         # dry run on TRAINING is not blocked
        s.run(missing, missing, ROOT / "models", tmp_path / "dry", dry_run_training=True, code_commit="abc")

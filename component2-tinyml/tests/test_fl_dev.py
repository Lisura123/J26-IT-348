"""Task 7: development comparison of docs/FL_EVALUATION_PROTOCOL.md (fl_dev.py).

Test Data only: software-generated fixture runs (test_features.make_dataset)
spread over the six site ids with generated timestamps. Not the PV Research
Dataset and not hardware measurements. The engine is limited to ONE round
(engine.N_ROUNDS patched, rounds=1, save=False). Scores on Test Data only
check the plumbing; they say nothing about model quality."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from test_features import make_dataset
from tinyml import fl_data as fd
from tinyml import fl_dev, fl_run
from tinyml.contract import LABELS
from tinyml.export import flatten
from tinyml.features import FeatureConfig, fit_pipeline
from tinyml.splits import make_assignments

SITES = ["SITE-A", "SITE-B", "SITE-C", "SITE-D", "SITE-E", "SITE-F"]


def _tree_hashes(*dirs):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in dirs if d.exists() for p in sorted(d.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


PROTECTED = (fd.MODELS_DIR, fd.COMPONENT / "results", fd.COMPONENT / "firmware", fd.MEMBER3_ROOT / "models",
             fd.MEMBER3_ROOT / "results", fd.MEMBER3_ROOT / "src", fd.MEMBER3_ROOT / "data")
BEFORE = _tree_hashes(*PROTECTED)


def make_test_data():
    """Test Data: 50 TRAINING + 50 OPERATION fixture runs, six site ids, 5 s readings."""
    df = make_dataset(runs_per_window=10, readings=6)
    runs = sorted(df.run_id.unique())
    df["site_id"] = df.run_id.map({r: SITES[i % 6] for i, r in enumerate(runs)})
    start = {r: pd.Timestamp("2026-09-01T06:00:00Z") + pd.Timedelta(hours=3 * i) for i, r in enumerate(runs)}
    df["timestamp_utc"] = [(start[r] + pd.Timedelta(seconds=5 * (n - 1))).strftime("%Y-%m-%dT%H:%M:%SZ")
                           for r, n in zip(df.run_id, df.reading)]
    return df


@pytest.fixture(scope="module")
def data():
    df = make_test_data()
    return df, make_assignments(df, n_splits=5, seed=42)


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("flwr")
    return fl_run.member3_engine()


@pytest.fixture(scope="module")
def fold(engine, data):
    """One fold x seed with ONE engine round; every data access is recorded."""
    df, a = data
    log = {"phase": "train", "seen": [], "fit_pipeline": [], "engine": []}
    mp = pytest.MonkeyPatch()
    real_xy, real_tf, real_fit, real_eval, real_run = fd._xy, fl_dev.transform, fl_dev.fit_pipeline, \
        fl_dev.evaluate_models, fl_run.run_engine

    def rec(part):
        log["seen"].append((log["phase"], set(part["run_id"]), set(part["window"])))

    mp.setattr(fd, "_xy", lambda part, cfg: (rec(part), real_xy(part, cfg))[1])
    mp.setattr(fl_dev, "transform", lambda part, cfg: (rec(part), real_tf(part, cfg))[1])
    mp.setattr(fl_dev, "fit_pipeline", lambda part, **kw: (log["fit_pipeline"].append(set(part["run_id"])),
                                                           rec(part), real_fit(part, **kw))[2])

    def eval_spy(models, eval_df):
        log["phase"] = "eval"
        return real_eval(models, eval_df)
    mp.setattr(fl_dev, "evaluate_models", eval_spy)

    def run_spy(w, d, **kw):
        log["engine"].append({"w": {k: v.copy() for k, v in w.items()}, "data": d})
        return real_run(w, d, **kw)
    mp.setattr(fl_run, "run_engine", run_spy)
    mp.setattr(engine, "N_ROUNDS", 1)
    try:
        result = fl_dev.run_fold(df, a, 0, 0, engine=engine, rounds=1)
    finally:
        mp.undo()
    return {"r": result, "log": log, "df": df, "a": a}


# ---------------------------------------------------------------- groups and pre-flight

def test_fold_groups_follow_the_protocol(data):
    df, a = data
    tr = fd.training_rows(df)
    for k in range(5):
        g = fl_dev.fold_groups(tr, a, k)
        ev, ini = set(g["EVAL"]), set(g["INIT"])
        fl = {r for v in g["FL"].values() for r in v}
        assert ev == set(a.index[(a.cv_fold == k) & (a.window == "TRAINING")])
        assert not (ev & ini or ev & fl or ini & fl) and ev | ini | fl == set(tr.run_id)
        assert not ({r for r in ev | ini | fl} & set(df[df.window == "OPERATION"].run_id))
        start = tr.groupby("run_id").timestamp_utc.min()
        for s, fl_runs in g["FL"].items():
            pool = sorted((start[r], r) for r in set(tr[tr.site_id == s].run_id) - ev)
            m = len(pool) // 2
            assert sorted(ini & {r for _, r in pool}) == sorted(r for _, r in pool[:m])     # earliest floor(n/2)
            assert fl_runs == sorted(r for _, r in pool[m:])


def test_ties_in_start_time_are_broken_by_run_id(data):
    df, a = data
    tr = fd.training_rows(df).copy()
    s_runs = sorted(set(tr[tr.site_id == "SITE-A"].run_id) - set(fl_dev.fold_groups(tr, a, 0)["EVAL"]))
    tr.loc[tr.run_id.isin(s_runs), "timestamp_utc"] = "2026-09-01T06:00:00Z"            # all equal
    g = fl_dev.fold_groups(tr, a, 0)
    assert sorted(set(g["INIT"]) & set(s_runs)) == s_runs[:len(s_runs) // 2]


def test_preflight_passes_on_test_data(data):
    df, a = data
    tr = fd.training_rows(df)
    t = fl_dev.preflight(tr, fl_dev.fold_groups(tr, a, 0), SITES)
    assert t["status"] == "OK" and t["eval_runs"] == 10 and t["eval_runs"] + t["init_runs"] + t["fl_runs"] == 50
    assert all(n >= 1 for n in t["fl_runs_per_site"].values()) and all(n >= 1 for n in t["init_class_runs"].values())


def test_preflight_stops_without_repairing(data):
    df, a = data
    tr = fd.training_rows(df)
    g = fl_dev.fold_groups(tr, a, 0)
    no_f = dict(g, FL=dict(g["FL"], **{"SITE-F": []}))
    with pytest.raises(fl_dev.PreflightError) as e:                                   # empty site
        fl_dev.preflight(tr[~tr.run_id.isin(g["FL"]["SITE-F"])], no_f, SITES)
    assert e.value.code == "BLOCKED_EMPTY_SITE"
    dark = tr.copy()
    dark.loc[dark.run_id.isin(g["FL"]["SITE-B"]), "irradiance_lux"] = 100.0           # all LOW_LIGHT
    with pytest.raises(fl_dev.PreflightError) as e:
        fl_dev.preflight(dark, g, SITES)
    assert e.value.code == "BLOCKED_EMPTY_SITE"
    leak = dict(g, INIT=g["INIT"] + g["EVAL"][:1])                                    # overlap
    with pytest.raises(fl_dev.PreflightError) as e:
        fl_dev.preflight(tr, leak, SITES)
    assert e.value.code == "INVALID_FOLD"
    normal_init = [r for r in g["INIT"] if tr[tr.run_id == r].label.iloc[0] == "NORMAL"]
    no_normal = dict(g, INIT=[r for r in g["INIT"] if r not in normal_init])
    with pytest.raises(fl_dev.PreflightError) as e:                                   # INIT lacks a class
        fl_dev.preflight(tr[~tr.run_id.isin(normal_init)], no_normal, SITES)
    assert e.value.code == "INVALID_FOLD" and "NORMAL" in e.value.detail


def test_run_fold_stops_at_preflight_before_any_fitting(engine, data, monkeypatch):
    df, a = data
    calls = []
    monkeypatch.setattr(fl_dev, "fit_pipeline", lambda *x, **k: calls.append(1))
    monkeypatch.setattr(engine, "N_ROUNDS", 1)
    with pytest.raises(fl_dev.PreflightError):
        fl_dev.run_fold(df, a, 0, 0, engine=engine, rounds=1, sites=SITES + ["SITE-G"])
    assert calls == []


def test_engine_round_count_must_equal_the_fixed_final_round(engine, data, monkeypatch):
    monkeypatch.setattr(engine, "N_ROUNDS", 1)
    with pytest.raises(ValueError, match="fixed final round"):
        fl_dev.run_fold(*data, 0, 0, engine=engine)                                    # rounds=20 by default


# ---------------------------------------------------------------- leakage

def test_eval_runs_are_never_seen_before_evaluation(fold):
    ev = set(fold["r"]["groups"]["EVAL"])
    train = [s for p, s, _ in fold["log"]["seen"] if p == "train"]
    evals = [s for p, s, _ in fold["log"]["seen"] if p == "eval"]
    assert train and evals
    assert all(not (s & ev) for s in train)                                            # nothing of EVAL before
    assert all(s <= ev for s in evals)                                                 # only EVAL afterwards
    assert all(w == {"TRAINING"} for _, _, w in fold["log"]["seen"])                   # no OPERATION anywhere


def test_feature_config_is_fitted_on_init_only(fold):
    g = fold["r"]["groups"]
    fits = fold["log"]["fit_pipeline"]
    pool = set(g["INIT"]) | {r for v in g["FL"].values() for r in v}
    assert fits[0] == set(g["INIT"]) and fits[1] == pool and len(fits) == 2           # INIT; POOL sensitivity
    assert all(not (f & set(g["EVAL"])) for f in fits)


def test_data_use_record_matches_the_protocol(fold):
    u, g = fold["r"]["data_use"], fold["r"]["groups"]
    fl_runs = sorted(r for v in g["FL"].values() for r in v)
    assert u["fit_pipeline"] == u["M0"] == u["engine_monitor"] == g["INIT"]
    assert u["FL"] == u["CENT_FT"] == fl_runs and u["LOCAL"] == g["FL"]
    assert u["CENT"] == sorted(set(g["INIT"]) | set(fl_runs)) and u["EVAL_scored"] == g["EVAL"]


# ---------------------------------------------------------------- models

def test_every_site_starts_from_the_same_m0(fold):
    r, call = fold["r"], fold["log"]["engine"][0]
    m0 = {k: np.asarray(v) for k, v in r["weights"]["M0"].items()}
    assert all(np.array_equal(call["w"][k], m0[k]) for k in m0)                       # one M0 for all sites
    site_data, monitor, _ = call["data"]
    assert sorted(site_data) == SITES
    tr = fd.training_rows(fold["df"])
    g = r["groups"]
    for s in SITES:                                                                    # each site: its own FL rows
        assert len(site_data[s][1]) == len(fd._xy(tr[tr.run_id.isin(g["FL"][s])], _cfg(fold))[1])
    assert len(monitor[1]) == len(fd._xy(tr[tr.run_id.isin(g["INIT"])], _cfg(fold))[1])


def _cfg(fold):
    tr = fd.training_rows(fold["df"])
    return fit_pipeline(tr[tr.run_id.isin(fold["r"]["groups"]["INIT"])], fold="dev_fold0_INIT")


def test_local_models_train_only_on_their_own_site(fold):
    r, cfg = fold["r"], _cfg(fold)
    tr = fd.training_rows(fold["df"])
    m0 = {k: np.asarray(v) for k, v in r["weights"]["M0"].items()}
    for i, s in enumerate(SITES):
        w, _ = fl_dev.sgd_from(m0, tr[tr.run_id.isin(r["groups"]["FL"][s])], cfg, [0, 0, 1000 + i])
        assert np.array_equal(flatten(w), flatten({k: np.asarray(v) for k, v in r["weights"][f"LOCAL-{s}"].items()}))
    assert not np.array_equal(np.asarray(r["weights"]["LOCAL-SITE-A"]["W1"]), np.asarray(r["weights"]["LOCAL-SITE-B"]["W1"]))


def test_fl_model_is_the_sample_weighted_mean_of_received_updates(fold, engine):
    """Recompute round 1 exactly as Member 3's engine does (SiteClient.fit + int8 + FedAvg)."""
    r = fold["r"]
    site_data, _, _ = fold["log"]["engine"][0]["data"]
    m0 = {k: np.asarray(v) for k, v in r["weights"]["M0"].items()}
    assert r["communication"]["updates"] == {"RECEIVED": 6}
    n = {s: len(site_data[s][1]) for s in SITES}
    step = np.zeros(229)
    for i, s in enumerate(SITES):
        _, d = engine.mlp.train_local({k: v.copy() for k, v in m0.items()}, *site_data[s],
                                      np.random.default_rng([0, 1, i]), "tanh")
        step += n[s] / sum(n.values()) * engine.int8_dense(d)
    fl = {k: np.asarray(v) for k, v in r["weights"]["FL"].items()}
    assert np.allclose(flatten(fl), flatten(m0) + step, atol=1e-12)


def test_local_training_uses_member3_train_local(engine):
    assert fl_dev.member3_mlp().__file__ == engine.mlp.__file__


def test_budgets_are_recorded_not_claimed_equal(fold):
    r = fold["r"]
    assert "not equal compute" in r["budget_note"]
    assert set(r["sgd_steps"]) >= {f"LOCAL-{s}" for s in SITES} | {"CENT-FT", "FL_per_site_per_round"}


# ---------------------------------------------------------------- comparison

def test_all_models_are_scored_on_the_same_readings(fold):
    m = fold["r"]["metrics"]
    assert set(m) == {"M0", *(f"LOCAL-{s}" for s in SITES), "FL", "CENT-FT", "CENT", "CENT-POOLCFG"}
    n = {k: v["changes_vs_m0"]["rows"] for k, v in m.items()}
    assert len(set(n.values())) == 1 and sum(x["readings"] for x in m["M0"]["per_site"].values()) == n["M0"]
    for v in m.values():
        assert {"accuracy", "macro_f1", "open_circuit_recall", "short_circuit_recall", "run_level",
                "confidence_at_or_above_0_90", "confidence_crossings_vs_m0", "changes_vs_m0", "per_site"} <= set(v)
    assert m["M0"]["changes_vs_m0"]["changed"] == 0 and m["M0"]["confidence_crossings_vs_m0"]["crossings"] == 0


def test_different_input_processing_is_refused(data):
    df, a = data
    tr = fd.training_rows(df)
    cfg = fit_pipeline(tr)
    other = FeatureConfig.from_json(cfg.to_json())
    other.low_light_lux = 60000.0                                                       # other gates -> other rows
    w = {k: np.zeros(s) for k, s in (("W1", (8, 16)), ("b1", (16,)), ("W2", (16, 5)), ("b2", (5,)))}
    with pytest.raises(ValueError, match="different readings"):
        fl_dev.evaluate_models({"M0": (w, cfg), "X": (w, other)}, tr)


def test_fast_macro_f1_equals_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(3)
    for _ in range(50):
        y = rng.integers(0, 5, 40) if rng.random() < .5 else rng.choice([0, 2, 4], 40)
        p = rng.integers(0, 5, 40)
        present = sorted(set(y.tolist()))
        assert fl_dev.macro_f1_codes(y, p) == pytest.approx(
            f1_score(y, p, labels=present, average="macro", zero_division=0), abs=1e-12)


def test_paired_bootstrap_resamples_whole_runs():
    y = ["NORMAL"] * 6 + ["DUST"] * 6
    t = {"run_id": ["r1"] * 3 + ["r2"] * 3 + ["r3"] * 3 + ["r4"] * 3, "y": y,
         "A": y, "B": ["NORMAL"] * 12}
    same = fl_dev.paired_run_bootstrap([t], "A", "A", resamples=100)
    assert same["observed_difference"] == 0.0 and same["ci95"] == [0.0, 0.0] and same["runs"] == 4
    better = fl_dev.paired_run_bootstrap([t], "A", "B", resamples=100)
    assert better["observed_difference"] > 0 and better["ci95"][1] > 0
    assert better == fl_dev.paired_run_bootstrap([t], "A", "B", resamples=100)          # seeded


# ---------------------------------------------------------------- monitoring, output and safety

def test_monitoring_is_labelled_training_derived(fold):
    r = fold["r"]
    assert "TRAINING-DERIVED DIAGNOSTICS ONLY" in r["monitor_note"] and r["final_round"] == 1
    text = json.dumps(r, default=str)
    assert "global_val_accuracy" not in text and "validation_accuracy" not in text
    assert all("monitor_loss_training_derived" in x for x in r["rounds"])
    assert r["communication"]["note"].startswith("SIMULATED")
    assert r["label"] == "DEVELOPMENT_MODEL_NOT_OFFICIAL" and "PENDING SAMUDI REVIEW" in r["engine_change"]


def test_results_are_written_create_only_outside_protected_folders(fold, tmp_path):
    p = fl_dev.write_result(fold["r"], "dev-test-run", tmp_path)
    assert p == (tmp_path / "dev-test-run" / "fold0_seed0.json").resolve()
    assert json.loads(p.read_text())["label"] == "DEVELOPMENT_MODEL_NOT_OFFICIAL"
    with pytest.raises(FileExistsError):
        fl_dev.write_result(fold["r"], "dev-test-run", tmp_path)
    for root in (fd.MODELS_DIR, fd.COMPONENT / "results", fd.MEMBER3_ROOT):
        with pytest.raises(fl_run.UnsafePathError):
            fl_dev.write_result(fold["r"], "dev-test-run", root)
    for bad in ("../x", "a/b"):
        with pytest.raises(ValueError):
            fl_dev.write_result(fold["r"], bad, tmp_path)


def test_full_protocol_run_needs_explicit_approval(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("dataset read")
    monkeypatch.setattr(fd, "load_dataset", refuse)
    for flag in (False, None, "yes", 1):
        with pytest.raises(PermissionError):
            fl_dev.run_protocol(fd.MEMBER3_ROOT / "data" / "shared" / "pv_fault_dataset.csv", "x",
                                confirm_protocol_run=flag)


def test_fixed_settings_match_the_committed_protocol():
    text = (fd.COMPONENT / "docs" / "FL_EVALUATION_PROTOCOL.md").read_text()
    assert fl_dev.ROUNDS == 20 and "20 rounds" in text
    assert fl_dev.LOCAL_EPOCHS == 60 and "60 epochs" in text
    assert fl_dev.SEEDS == (0, 1, 2) and "0, 1 and 2" in text
    assert (fl_dev.BOOTSTRAP_RESAMPLES, fl_dev.BOOTSTRAP_SEED) == (2000, 0) and "2,000 resamples, seed 0" in text
    assert fl_dev.CONFIDENCE_THRESHOLD == 0.90 and fl_dev.SGD == {"batch": 32, "lr": 0.05}


# ---------------------------------------------------------------- Task 8a: G1 fault counts

def test_fault_counts_from_a_confusion_matrix():
    cm = np.zeros((5, 5), dtype=int)
    o, s, n = LABELS.index("OPEN_CIRCUIT"), LABELS.index("SHORT_CIRCUIT"), LABELS.index("NORMAL")
    cm[o, o], cm[o, n] = 7, 2          # 2 missed OPEN
    cm[n, o] = 3                       # 3 false OPEN alarms
    cm[s, s], cm[s, o] = 5, 1          # 1 missed SHORT (read as OPEN: also a false OPEN alarm)
    cm[n, n] = 10
    c = fl_dev.fault_counts(cm)
    assert c["OPEN_CIRCUIT"] == {"true": 9, "detected": 7, "missed": 2, "false_alarms": 4}
    assert c["SHORT_CIRCUIT"] == {"true": 6, "detected": 5, "missed": 1, "false_alarms": 0}


def test_every_model_records_open_short_counts(fold):
    for m in fold["r"]["metrics"].values():
        c = m["open_short_counts"]
        assert c == fl_dev.fault_counts(m["confusion_matrix"])
        for f, key in (("OPEN_CIRCUIT", "open_circuit_recall"), ("SHORT_CIRCUIT", "short_circuit_recall")):
            assert c[f]["detected"] + c[f]["missed"] == c[f]["true"]
            assert m[key] == pytest.approx(c[f]["detected"] / c[f]["true"])


# ---------------------------------------------------------------- G2 communication, G3 site records

def test_communication_records_are_simulated_trace_estimates(fold):
    c, recs = fold["r"]["communication"], fold["r"]["site_records"]
    assert c["note"].startswith("SIMULATED TRACE ESTIMATES") and "not physical radio measurements" in c["note"]
    payloads = [x["payload_bytes"] for x in recs]
    assert c["payload_bytes_total"] == sum(payloads) == c["bytes_attempted"]
    assert c["payload_bytes_per_update"] == {"min": min(payloads), "max": max(payloads),
                                             "mean": pytest.approx(np.mean(payloads))}
    assert c["updates"] == {st: sum(x["status"] == st for x in recs) for st in {x["status"] for x in recs}}
    assert c["lora_airtime_s"] >= 0 and "radio_energy_mj" in c


def test_site_records_cover_every_site_and_round(fold):
    r = fold["r"]
    recs, site_data = r["site_records"], fold["log"]["engine"][0]["data"][0]
    assert sorted((x["round_id"], x["site_id"]) for x in recs) == [(1, s) for s in SITES]
    for x in recs:
        assert set(x) == set(fl_dev.SITE_RECORD_COLUMNS)
        assert x["num_samples"] == len(site_data[x["site_id"]][1])
        assert x["local_loss_before"] is not None and x["local_loss_after"] is not None
        assert x["float_delta_l2_norm"] > 0 and x["status"] in ("RECEIVED", "LATE", "FAILED")
    used = [x for x in recs if x["used_in_round"] == 1]
    assert sum(x["aggregation_weight"] for x in used) == pytest.approx(1.0, abs=1e-5)
    n = sum(x["num_samples"] for x in used)
    assert all(x["aggregation_weight"] == pytest.approx(x["num_samples"] / n, abs=1e-6) for x in used)


# ---------------------------------------------------------------- G6 reproducibility

def test_reproducibility_record(fold):
    rp = fold["r"]["reproducibility"]
    assert set(rp["environment"]) >= {"python", "numpy", "sklearn", "flwr"} and rp["environment"]["flwr"]
    a = rp["architecture"]
    assert (a["inputs"], a["hidden"], a["outputs"], a["n_params"]) == (8, 16, 5, 229)
    assert a["activation"] == "tanh" and a["feature_version"] == "pv8-v1" and a["labels"] == list(LABELS)
    assert rp["protocol"]["sha256"] == fd.file_sha256(fd.COMPONENT / "docs" / "FL_EVALUATION_PROTOCOL.md")
    assert rp["split_file"]["sha256"] == fd.file_sha256(fd.SPLITS_PATH)
    assert rp["fixed"]["folds"] == [0, 1, 2, 3, 4] and rp["fixed"]["seeds"] == [0, 1, 2]
    assert rp["fixed"]["rounds"] == 20 and rp["dataset_sha256"] is None                # Test Data: no PV dataset


# ---------------------------------------------------------------- G4 summary, G5 failures

def _experiment(engine, data, tmp_path, monkeypatch, run_id="exp-test", folds=(0, 1)):
    monkeypatch.setattr(engine, "N_ROUNDS", 1)
    return fl_dev.run_experiment(*data, run_id, engine=engine, allowed_root=tmp_path, folds=folds, seeds=(0,),
                                 rounds=1)


def test_experiment_writes_results_and_a_complete_summary(engine, data, tmp_path, monkeypatch):
    s, paths = _experiment(engine, data, tmp_path, monkeypatch)
    assert [p.name for p in paths] == ["fold0_seed0.json", "fold1_seed0.json", "summary.json"]
    assert json.loads(paths[-1].read_text()) == json.loads(json.dumps(s, default=str))
    assert s["status"] == "OK" and s["counts"] == {"planned": 2, "succeeded": 2, "failed": 0}
    assert set(s["models"]) == {"M0", *(f"LOCAL-{x}" for x in SITES), "FL", "CENT-FT", "CENT", "CENT-POOLCFG"}
    for m in s["models"].values():
        assert {"readings", "accuracy", "macro_f1", "open_circuit_recall", "short_circuit_recall",
                "open_short_counts", "below_0_90_share", "crossings_vs_m0"} <= set(m["pooled"])
        assert set(m["spread"]["macro_f1"]["by_fold_seed"]) == {"fold0_seed0", "fold1_seed0"}
        assert set(m["spread"]) == set(fl_dev.SPREAD_KEYS) and m["spread"]["open_circuit_recall"]["undefined"] == []
        assert set(m["spread"]["macro_f1"]["by_fold"]) == {"0", "1"}
    assert set(s["comparisons"]) == {"FL_vs_M0", "FL_vs_LOCAL_mean", "FL_vs_CENT", "FL_vs_CENT-FT"}
    assert s["comparisons"]["FL_vs_LOCAL_mean"]["b"] == [f"LOCAL-{x}" for x in SITES]
    assert all(c["resamples"] == 2000 and c["seed"] == 0 for c in s["comparisons"].values())
    assert set(s["open_short_safety"]) == {"OPEN_CIRCUIT", "SHORT_CIRCUIT"}
    assert set(s["confidence_below_0_90"]) == set(s["models"]) and sorted(s["classes_missing_at_site"]) == SITES
    assert s["communication"]["note"].startswith("SIMULATED TRACE ESTIMATES")
    assert "need human review" in s["section8_criteria"]["note"] and len(s["section8_criteria"]) == 6
    n = sum(len(json.loads(p.read_text())["eval_table"]["y"]) for p in paths[:2])
    assert s["models"]["FL"]["pooled"]["readings"] == n


def test_pooled_numbers_are_computed_from_the_eval_tables(engine, data, tmp_path, monkeypatch):
    s, paths = _experiment(engine, data, tmp_path, monkeypatch)
    rs = [json.loads(p.read_text()) for p in paths[:2]]
    y = sum((r["eval_table"]["y"] for r in rs), [])
    fl = sum((r["eval_table"]["FL"] for r in rs), [])
    conf = sum((r["eval_table"]["conf:FL"] for r in rs), [])
    assert s["models"]["FL"]["pooled"]["accuracy"] == pytest.approx(np.mean(np.array(y) == np.array(fl)))
    assert s["models"]["FL"]["pooled"]["below_0_90_share"] == pytest.approx(np.mean(np.array(conf) < 0.90))
    f1s = [r["metrics"]["FL"]["macro_f1"] for r in rs]
    assert s["models"]["FL"]["spread"]["macro_f1"]["mean"] == pytest.approx(np.mean(f1s))


def test_a_failing_fold_is_recorded_and_others_continue(engine, data, tmp_path, monkeypatch):
    real = fl_dev.run_fold

    def flaky(df, a, k, seed, **kw):
        if k == 1:
            raise fl_dev.PreflightError("BLOCKED_EMPTY_SITE", "test: SITE-F has no FL readings")
        return real(df, a, k, seed, **kw)
    monkeypatch.setattr(fl_dev, "run_fold", flaky)
    s, paths = _experiment(engine, data, tmp_path, monkeypatch)
    assert [p.name for p in paths] == ["fold0_seed0.json", "fold1_seed0.FAILED.json", "summary.json"]
    rec = json.loads(paths[1].read_text())
    assert rec["status"] == "FAILED" and rec["error_code"] == "BLOCKED_EMPTY_SITE" and rec["metrics"] is None
    assert s["counts"] == {"planned": 2, "succeeded": 1, "failed": 1} and s["used"] == ["fold0_seed0"]
    assert s["folds_used"] == [0] and s["failed"][0]["error_code"] == "BLOCKED_EMPTY_SITE"
    assert list(s["models"]["FL"]["spread"]["macro_f1"]["by_fold_seed"]) == ["fold0_seed0"]   # nothing invented


def test_other_errors_are_recorded_with_their_type(engine, data, tmp_path, monkeypatch):
    real = fl_dev.run_fold
    monkeypatch.setattr(fl_dev, "run_fold", lambda df, a, k, seed, **kw:
                        (_ for _ in ()).throw(ValueError("test failure")) if k == 0 else real(df, a, k, seed, **kw))
    s, paths = _experiment(engine, data, tmp_path, monkeypatch)
    rec = json.loads(paths[0].read_text())
    assert paths[0].name == "fold0_seed0.FAILED.json" and rec["error_code"] == "ValueError"
    assert rec["error"] == "test failure" and s["counts"]["failed"] == 1


def test_all_failed_gives_no_metrics(engine, data, tmp_path, monkeypatch):
    monkeypatch.setattr(fl_dev, "run_fold", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    s, paths = _experiment(engine, data, tmp_path, monkeypatch)
    assert s["status"] == "NO_SUCCESSFUL_RESULTS" and "models" not in s and "comparisons" not in s
    assert s["counts"] == {"planned": 2, "succeeded": 0, "failed": 2}


@pytest.mark.parametrize("error", [fl_dev.IntegrityError("test"), AssertionError("engine did not start from M0")])
def test_integrity_failures_stop_everything(engine, data, tmp_path, monkeypatch, error):
    monkeypatch.setattr(fl_dev, "run_fold", lambda *a, **k: (_ for _ in ()).throw(error))
    with pytest.raises(type(error)):
        _experiment(engine, data, tmp_path, monkeypatch)
    assert not (tmp_path / "exp-test" / "summary.json").exists()


def test_integrity_is_rechecked_after_every_fold(engine, data, tmp_path, monkeypatch):
    calls = []
    real = fl_dev.verify_artifacts

    def check(d):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("model_contract.json changed")
        return real(d)
    monkeypatch.setattr(fl_dev, "verify_artifacts", check)
    with pytest.raises(fl_dev.IntegrityError, match="Model v1 artifacts"):
        _experiment(engine, data, tmp_path, monkeypatch)
    assert sorted(p.name for p in (tmp_path / "exp-test").iterdir()) == ["fold0_seed0.json"]   # stopped, no summary


def test_runs_are_never_overwritten_or_mixed(engine, data, tmp_path, monkeypatch):
    _experiment(engine, data, tmp_path, monkeypatch, folds=(0,))
    before = sorted(p.name for p in (tmp_path / "exp-test").iterdir())
    with pytest.raises(FileExistsError):
        _experiment(engine, data, tmp_path, monkeypatch, folds=(0,))
    assert sorted(p.name for p in (tmp_path / "exp-test").iterdir()) == before


def test_pv_research_dataset_needs_approval_even_through_run_experiment(engine, data, tmp_path):
    from tinyml.train import DATASET_SHA256
    with pytest.raises(PermissionError):
        fl_dev.run_experiment(*data, "x", dataset_sha256=DATASET_SHA256, engine=engine, allowed_root=tmp_path)
    assert not any(tmp_path.iterdir())


def test_dataset_validation_failure_stops_before_anything_is_written(tmp_path):
    csv = tmp_path / "pv_fault_dataset.csv"
    make_test_data().to_csv(csv, index=False)                                         # not the PV Research Dataset
    out = tmp_path / "runs"
    with pytest.raises(ValueError, match="SHA-256"):
        fl_dev.run_protocol(csv, "x", confirm_protocol_run=True, allowed_root=out)
    assert not out.exists()


# ---------------------------------------------------------------- hand-calculated values (no training)

def test_drift_and_label_change_rate_values():
    rng = np.random.default_rng(0)
    m0 = {"W1": rng.normal(0, .5, (8, 16)), "b1": rng.normal(0, .1, 16), "W2": rng.normal(0, .5, (16, 5)),
          "b2": rng.normal(0, .1, 5)}
    d = {k: rng.normal(0, .3, v.shape) for k, v in m0.items()}
    w1 = {k: m0[k] + d[k] for k in m0}
    X = rng.normal(0, 1, (200, 8))
    hist = pd.DataFrame([{"round_id": r, "global_model_version": f"v1.{r}", "global_val_loss": .1,
                          "clients_received": 6, "clients_late": 0, "clients_failed": 0, "mean_update_l2_norm": 0.0}
                         for r in (0, 1)])
    hist.attrs["models"] = {"v1.0": {k: v.tolist() for k, v in m0.items()},
                            "v1.1": {k: v.tolist() for k, v in w1.items()}}
    pr = fl_dev.per_round(hist, m0, X)
    from tinyml.export import forward
    assert pr[0]["drift_l2_from_m0"] == 0.0 and pr[0]["label_change_rate_vs_m0_on_pool"] == 0.0
    assert pr[1]["drift_l2_from_m0"] == pytest.approx(np.sqrt(sum((d[k] ** 2).sum() for k in d)), abs=1e-12)
    assert pr[1]["drift_l2_from_m0"] == pytest.approx(4.6587401233, abs=1e-9)
    changed = (forward(w1, X)[1].argmax(1) != forward(m0, X)[1].argmax(1)).mean()
    assert pr[1]["label_change_rate_vs_m0_on_pool"] == pytest.approx(changed) == pytest.approx(0.445)


# Hand-made EVAL tables: 10 runs of 4 readings (2 per class), two seeds. GOOD reads every
# reading correctly; BAD reads both OPEN runs as NORMAL and two SHORT readings as OPEN.
Y = (["NORMAL"] * 4 + ["OPEN_CIRCUIT"] * 4 + ["SHORT_CIRCUIT"] * 4 + ["DUST"] * 4 + ["PARTIAL_SHADING"] * 4) * 2
GOOD = list(Y)
BAD = list(Y)
BAD[4:8] = ["NORMAL"] * 4
BAD[8:10] = ["OPEN_CIRCUIT"] * 2
BAD[24:28] = ["NORMAL"] * 4
HI, LO = [0.99] * len(Y), [0.80] * len(Y)
SITE_FL = {"SITE-A": ["DUST", "NORMAL", "PARTIAL_SHADING"]}                 # SITE-A has no OPEN or SHORT data


def _fake_result(seed, preds, conf):
    from tinyml.evaluate import classification_metrics
    m = {}
    for n, p in preds.items():
        cm = classification_metrics(np.array(Y), np.array(p))
        c, c0 = np.array(conf[n]), np.array(conf["M0"])
        m[n] = {"macro_f1": cm["macro_f1"], "accuracy": cm["accuracy"],
                "open_circuit_recall": cm["safety_recall"]["OPEN_CIRCUIT"],
                "short_circuit_recall": cm["safety_recall"]["SHORT_CIRCUIT"],
                "confidence_at_or_above_0_90": float((c >= .9).mean()),
                "confidence_crossings_vs_m0": {"fell_below": int(((c0 >= .9) & (c < .9)).sum()),
                                               "rose_above": int(((c0 < .9) & (c >= .9)).sum()),
                                               "crossings": int(((c0 >= .9) != (c >= .9)).sum())}}
    t = {"run_id": [f"r{i // 4}" for i in range(len(Y))], "site_id": ["SITE-A"] * len(Y), "y": Y, **preds,
         **{f"conf:{n}": v for n, v in conf.items()}}
    return {"fold": 0, "seed": seed, "metrics": m, "eval_table": t, "site_classes_fl": SITE_FL,
            "communication": {"payload_bytes_total": 100, "lora_airtime_s": 1.0, "radio_energy_mj": 2.0,
                              "updates": {"RECEIVED": 6}}}


def _case(fl, m0, local, conf_fl, conf_m0):
    preds = {"M0": m0, "LOCAL-SITE-A": local, "FL": fl, "CENT": GOOD, "CENT-FT": GOOD}
    conf = {"M0": conf_m0, "LOCAL-SITE-A": HI, "FL": conf_fl, "CENT": HI, "CENT-FT": HI}
    return fl_dev.summarize([_fake_result(s, preds, conf) for s in (0, 1)], [], {}, planned=2)


def test_open_and_short_missed_and_false_alarm_values():
    s = _case(fl=GOOD, m0=BAD, local=BAD, conf_fl=HI, conf_m0=LO)
    # per seed: OPEN 8 true, all 8 read NORMAL; 2 SHORT read OPEN (2 missed SHORT, 2 false OPEN alarms)
    assert s["open_short_safety"]["OPEN_CIRCUIT"]["M0"] == {"true": 16, "detected": 0, "missed": 16, "false_alarms": 4}
    assert s["open_short_safety"]["SHORT_CIRCUIT"]["M0"] == {"true": 16, "detected": 12, "missed": 4, "false_alarms": 0}
    assert s["open_short_safety"]["OPEN_CIRCUIT"]["FL"] == {"true": 16, "detected": 16, "missed": 0, "false_alarms": 0}
    assert s["open_short_safety"]["OPEN_CIRCUIT"]["FL_minus_M0_missed"] == -16
    assert s["open_short_safety"]["OPEN_CIRCUIT"]["FL_minus_M0_false_alarms"] == -4


def test_confidence_threshold_values():
    s = _case(fl=GOOD, m0=BAD, local=BAD, conf_fl=HI, conf_m0=LO)
    assert s["confidence_below_0_90"]["FL"] == 0.0 and s["confidence_below_0_90"]["M0"] == 1.0
    assert s["models"]["FL"]["pooled"]["crossings_vs_m0"] == {"fell_below": 0, "rose_above": 80, "crossings": 80}


def test_fl_versus_local_average_values():
    from sklearn.metrics import f1_score
    s = _case(fl=GOOD, m0=BAD, local=BAD, conf_fl=HI, conf_m0=LO)
    expected = (f1_score(Y * 2, GOOD * 2, labels=list(LABELS), average="macro", zero_division=0)
                - f1_score(Y * 2, BAD * 2, labels=list(LABELS), average="macro", zero_division=0))
    assert s["comparisons"]["FL_vs_LOCAL_mean"]["observed_difference"] == pytest.approx(expected, abs=1e-12)
    assert s["comparisons"]["FL_vs_LOCAL_mean"]["observed_difference"] == pytest.approx(0.295238095238, abs=1e-9)
    assert s["classes_missing_at_site"]["SITE-A"] == {"readings": 32, "fl_correct": 32, "local_correct": 12}


def test_evidence_criteria_positive_case():
    s = _case(fl=GOOD, m0=BAD, local=BAD, conf_fl=HI, conf_m0=LO)            # FL clearly better everywhere
    c = s["section8_criteria"]
    assert all(c[k] is True for k in c if k != "note")
    lo, hi = s["comparisons"]["FL_vs_M0"]["ci95"]
    assert s["comparisons"]["FL_vs_M0"]["observed_difference"] == pytest.approx(0.2952380952, abs=1e-9) and 0 < lo < hi


def test_evidence_criteria_negative_case():
    s = _case(fl=BAD, m0=GOOD, local=GOOD, conf_fl=LO, conf_m0=HI)            # FL clearly worse everywhere
    c = s["section8_criteria"]
    assert all(c[k] is False for k in c if k != "note")
    lo, hi = s["comparisons"]["FL_vs_M0"]["ci95"]
    assert s["comparisons"]["FL_vs_M0"]["observed_difference"] == pytest.approx(-0.2952380952, abs=1e-9) and lo < hi < 0


def test_official_files_unchanged_and_no_dev_runs_folder():
    from tinyml.export import verify_artifacts
    verify_artifacts(fd.MODELS_DIR)
    assert _tree_hashes(*PROTECTED) == BEFORE
    assert not fl_dev.DEV_RUNS_DIR.exists()
    assert list(LABELS) == ["NORMAL", "PARTIAL_SHADING", "DUST", "OPEN_CIRCUIT", "SHORT_CIRCUIT"]

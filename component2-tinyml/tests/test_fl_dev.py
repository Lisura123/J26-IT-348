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


def test_official_files_unchanged_and_no_dev_runs_folder():
    from tinyml.export import verify_artifacts
    verify_artifacts(fd.MODELS_DIR)
    assert _tree_hashes(*PROTECTED) == BEFORE
    assert not fl_dev.DEV_RUNS_DIR.exists()
    assert list(LABELS) == ["NORMAL", "PARTIAL_SHADING", "DUST", "OPEN_CIRCUIT", "SHORT_CIRCUIT"]

"""Model v1 -> Member 3's FL engine (fl_run.py and the engine change PENDING
SAMUDI REVIEW). Synthetic runs only; the engine is limited to ONE round
(engine.N_ROUNDS patched) with save=False. Nothing is trained beyond that
one-round smoke run, no accuracy is measured and no file is written. Skipped
when Member 3's code or flwr is not available."""
import hashlib

import numpy as np
import pytest

from test_features import make_dataset
from tinyml import fl_data as fd
from tinyml import fl_run
from tinyml.contract import LABELS
from tinyml.export import flatten, weights_sha256
from tinyml.features import FEATURES, transform
from tinyml.splits import make_assignments

pytestmark = pytest.mark.skipif(not (fd.MEMBER3_ROOT / "src" / "engine.py").exists(),
                                reason="Member 3 code not in this checkout")
SITES = ["SITE-A", "SITE-B", "SITE-C", "SITE-D", "SITE-E", "SITE-F"]


def six_site_dataset():
    """Synthetic runs (test_features.make_dataset) spread over the six sites."""
    df = make_dataset(runs_per_window=6, readings=6)
    runs = sorted(df.run_id.unique())
    df["site_id"] = df.run_id.map({r: SITES[i % 6] for i, r in enumerate(runs)})
    return df


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("flwr")
    return fl_run.member3_engine()


@pytest.fixture(scope="module")
def v1():
    contract, cfg, w = fd.load_frozen_model()
    return {"contract": contract, "cfg": cfg, "w": w, "sha": weights_sha256(w)}


@pytest.fixture(scope="module")
def synth(v1):
    df = six_site_dataset()
    d = fd.build_fl_data(df, v1["cfg"], make_assignments(df, n_splits=5, seed=42))
    return df, (d.site_data, d.val, d.cfg)


@pytest.fixture
def one_round(engine, monkeypatch):
    monkeypatch.setattr(engine, "N_ROUNDS", 1)
    return engine


def _results_hashes():
    files = sorted(p for p in (fd.MEMBER3_ROOT / "results").rglob("*") if p.is_file())
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


@pytest.fixture(scope="module")
def run(engine, synth, v1):
    """The one-round smoke run shared by several tests."""
    mp = pytest.MonkeyPatch()
    mp.setattr(engine, "N_ROUNDS", 1)
    before = _results_hashes()
    try:
        out = fl_run.run_from_model_v1(data=synth[1], engine=engine)
    finally:
        mp.undo()
    return {"out": out, "results_before": before}


# ---------------------------------------------------------------- inputs

def test_model_v1_loads_verified(v1):
    assert v1["contract"]["model_version"] == "c2-local-v1" and v1["contract"]["activation"] == "tanh"
    assert v1["sha"] == "94e7ab9384a1906ec2cf0d3176b435e700f15cd99552fbc6c71519c648ecc3a7"


def test_synthetic_data_is_pv8_v1_and_covers_every_engine_site(engine, synth):
    site_data, (Xv, yv), cfg = fl_run.check_data(synth[1], engine.pv_data.SITES)
    assert tuple(cfg.features) == FEATURES and sorted(site_data) == SITES
    assert all(X.shape[1] == 8 and len(y) for X, y in site_data.values()) and len(yv)


def test_no_operation_rows_enter_the_fl_data(v1, monkeypatch):
    seen = []
    monkeypatch.setattr(fd, "transform", lambda part, cfg: (seen.extend(part["window"]), transform(part, cfg))[1])
    df = six_site_dataset()
    fd.build_fl_data(df, v1["cfg"], make_assignments(df, n_splits=5, seed=42))
    assert set(seen) == {"TRAINING"} and (df.window == "OPERATION").any()


@pytest.mark.parametrize("bad", [
    lambda d: ({k: v for k, v in d[0].items() if k != "SITE-F"}, d[1], d[2]),                 # missing site
    lambda d: ({**d[0], "SITE-A": (d[0]["SITE-A"][0][:, :7], d[0]["SITE-A"][1])}, d[1], d[2]),   # raw7 / wrong width
    lambda d: ({**d[0], "SITE-B": (d[0]["SITE-B"][0], d[0]["SITE-B"][1] + 5)}, d[1], d[2]),     # labels outside 0..4
])
def test_bad_data_is_refused(engine, synth, bad):
    with pytest.raises(ValueError):
        fl_run.check_data(bad(synth[1]), engine.pv_data.SITES)


def test_wrapper_fixes_activation_save_and_inputs(engine, synth):
    for k, v in (("activation", "relu"), ("save", True), ("init_weights", {})):
        with pytest.raises(ValueError, match="fixed by the wrapper"):
            fl_run.run_from_model_v1(data=synth[1], engine=engine, **{k: v})


# ---------------------------------------------------------------- engine change (PENDING SAMUDI REVIEW)

def test_engine_refuses_integration_inputs_without_tanh(engine, synth, v1):
    with pytest.raises(ValueError, match="tanh"):
        engine.run_fl(activation="relu", data=synth[1], init_weights=v1["w"], save=False, verbose=False)


@pytest.mark.parametrize("name, shape", [("W1", (16, 8)), ("b2", (4,))])
def test_engine_rejects_wrong_weight_shapes(one_round, synth, v1, name, shape):
    w = dict(v1["w"], **{name: np.zeros(shape)})
    with pytest.raises(ValueError, match="LAYOUT"):
        one_round.run_fl(data=synth[1], init_weights=w, save=False, verbose=False)


def test_default_behaviour_is_unchanged(one_round, synth, monkeypatch):
    """data=None and init_weights=None: pv_data.get_data() and a random
    mlp.init_params(default_rng(seed), activation), any activation allowed."""
    calls = []
    monkeypatch.setattr(one_round.pv_data, "get_data", lambda: (calls.append("get_data"), synth[1])[1])
    df = one_round.run_fl(seed=3, activation="relu", save=False, verbose=False)
    expected = one_round.mlp.init_params(np.random.default_rng(3), "relu")
    assert calls == ["get_data"]
    assert all(np.allclose(df.attrs["models"]["v1.0"][k], expected[k], rtol=0, atol=0) for k in expected)


# ---------------------------------------------------------------- one-round smoke run from Model v1

def test_round0_is_model_v1_and_v1_is_unchanged(run, v1):
    out = run["out"]
    v10 = {k: np.asarray(v) for k, v in out["history"].attrs["models"]["v1.0"].items()}
    assert all(np.array_equal(v10[k], v1["w"][k]) for k in v1["w"])
    assert weights_sha256(v1["w"]) == v1["sha"]                        # caller's arrays untouched
    _, _, w_again = fd.load_frozen_model()
    assert weights_sha256(w_again) == v1["sha"]                        # artifact on disk untouched


def test_candidate_is_marked_and_not_written(run, v1):
    c = run["out"]["candidate"]
    assert c["status"] == "CANDIDATE_NOT_VALIDATED" and c["engine_version"] == "v1.1"
    assert c["base_model_version"] == "c2-local-v1" and c["base_weights_sha256"] == v1["sha"]
    assert c["activation"] == "tanh" and c["labels"] == list(LABELS) and "not independent" in c["validation_note"]
    assert c["engine_options"]["save"] is False and c["data_source"].startswith("caller-supplied")
    assert c["operation_rows_used"] is None                             # only claimed when fl_data built the data
    assert flatten(c["weights"]).size == 229 and c["weights_sha256"] == weights_sha256(c["weights"])


def test_delta_w_is_new_minus_model_v1(run, engine, synth, v1):
    """Recompute each site's local training exactly as SiteClient.fit does
    and check the engine's float Delta W norm (engine.py: x_raw = new - w_vec)."""
    site_data = synth[1][0]
    ups = run["out"]["updates"].set_index("site_id")
    for i, s in enumerate(engine.pv_data.SITES):
        X, y = site_data[s]
        rng = np.random.default_rng([0, 1, i])
        new, delta = engine.mlp.train_local({k: v.copy() for k, v in v1["w"].items()}, X, y, rng, "tanh")
        assert np.allclose(delta, flatten(new) - flatten(v1["w"]))
        assert ups.loc[s, "float_delta_l2_norm"] == pytest.approx(round(float(np.linalg.norm(delta)), 5), abs=1e-5)
        assert ups.loc[s, "num_samples"] == len(y) and ups.loc[s, "base_model_version"] == "v1.0"


def test_aggregation_is_the_sample_weighted_mean_of_received_deltas(run, engine, synth, v1):
    """Round 1 global = Model v1 + sum(n_s / N * int8(delta_s)) over RECEIVED updates (FedAvg)."""
    site_data = synth[1][0]
    ups = run["out"]["updates"].set_index("site_id")
    got = [s for s in engine.pv_data.SITES if ups.loc[s, "status"] == "RECEIVED"]
    assert got
    n = {s: len(site_data[s][1]) for s in got}
    step = np.zeros(229)
    for s in got:
        i = engine.pv_data.SITES.index(s)
        _, delta = engine.mlp.train_local({k: v.copy() for k, v in v1["w"].items()}, *site_data[s],
                                          np.random.default_rng([0, 1, i]), "tanh")
        step += n[s] / sum(n.values()) * engine.int8_dense(delta)
    assert np.allclose(flatten(run["out"]["candidate"]["weights"]), flatten(v1["w"]) + step, atol=1e-12)


def test_no_committed_results_overwritten(run):
    assert _results_hashes() == run["results_before"]

"""Member 2 FL adapters: metadata, layout, deltas, update records, global
weight import validation. Uses the frozen contract read-only and synthetic
weight changes; nothing is trained."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tinyml import fl_adapter as fa
from tinyml.export import contract_weights, file_sha256, flatten, unflatten, weights_sha256

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "models" / "model_contract.json"


@pytest.fixture(scope="module")
def base():
    c = json.loads(CONTRACT_PATH.read_text())
    sha = file_sha256(CONTRACT_PATH)
    return {"contract": c, "sha": sha, "meta": fa.local_model_metadata(c, sha), "w": contract_weights(c)}


def perturbed(w, seed=1, scale=0.01):
    r = np.random.default_rng(seed)
    return {k: v + r.normal(0, scale, v.shape) for k, v in w.items()}


def compatible_doc(base, w, version="v2.0"):
    m = base["meta"]
    return {"version": version, "weights": {k: v.tolist() for k, v in w.items()},
            "weights_sha256": weights_sha256(w), "base_model_version": "c2-local-v1",
            **{k: m[k] for k in fa.COMPAT_KEYS}}


def test_metadata_describes_the_frozen_contract(base):
    m = base["meta"]
    assert m["feature_version"] == "pv8-v1" and m["activation"] == "tanh" and m["n_params"] == 229
    assert m["labels"] == ["NORMAL", "PARTIAL_SHADING", "DUST", "OPEN_CIRCUIT", "SHORT_CIRCUIT"]
    assert m["param_layout"] == fa.MEMBER3_LAYOUT and m["weights_sha256"].startswith("94e7ab93")
    assert len(m["feature_config_sha256"]) == 64 and "weights" not in m
    c2 = copy.deepcopy(base["contract"])
    c2["feature_config"]["scaler"]["mean"][0] += 1e-6          # any preprocessing change changes the hash
    assert fa.feature_config_sha256(c2) != m["feature_config_sha256"]


def test_flatten_round_trip_in_member3_order(base):
    vec = flatten(base["w"])
    assert vec.shape == (229,) and np.array_equal(vec[:128], base["w"]["W1"].ravel())
    assert np.array_equal(vec[128:144], base["w"]["b1"]) and np.array_equal(vec[224:], base["w"]["b2"])
    back = unflatten(vec)
    assert all(np.array_equal(back[k], base["w"][k]) for k in back)


def test_delta_and_int8_compression(base):
    w2 = perturbed(base["w"])
    d = fa.compute_delta(w2, base["w"])
    assert np.allclose(d, flatten(w2) - flatten(base["w"]))
    q, scale = fa.int8_dense(d)
    assert q.min() >= -127 and q.max() <= 127 and np.abs(q * scale - d).max() <= scale / 2 + 1e-12
    assert fa.int8_dense(np.zeros(229))[1] == 1.0


def test_update_record_round_trip(base):
    w2 = perturbed(base["w"])
    u = fa.build_update(local_weights=w2, base_contract=base["contract"], base_contract_sha256=base["sha"],
                        base_model_version="c2-local-v1", update_id="R01-SITE-A", round_id=1, site_id="SITE-A",
                        device_id="ESP32-A01", num_samples=30, class_counts={"NORMAL": 20, "DUST": 10})
    assert fa.verify_update(u, base["meta"])
    rec = fa.apply_update(base["w"], u)
    assert np.abs(flatten(rec) - flatten(w2)).max() <= u["int8_scale"] / 2 + 1e-12
    json.dumps(u)                                                       # serialisable as one JSONL line


@pytest.mark.parametrize("mutate, code", [
    (lambda u: u.update(delta_int8=u["delta_int8"][:-1]), "INTEGRITY"),
    (lambda u: u.update(feature_version="raw8"), "INCOMPATIBLE_PREPROCESSING"),
    (lambda u: u.update(activation="relu"), "INCOMPATIBLE_ACTIVATION"),
    (lambda u: u.pop("feature_config_sha256"), "MISSING_COMPATIBILITY_METADATA"),
    (lambda u: u.update(base_weights_sha256="0" * 64), "BASE_MISMATCH"),
])
def test_bad_updates_are_rejected(base, mutate, code):
    u = fa.build_update(local_weights=perturbed(base["w"]), base_contract=base["contract"],
                        base_contract_sha256=base["sha"], base_model_version="c2-local-v1", update_id="u", round_id=1,
                        site_id="SITE-A", device_id="ESP32-A01", num_samples=1, class_counts={"NORMAL": 1})
    mutate(u)
    if code in ("BASE_MISMATCH",):
        u["update_sha256"] = fa.update_sha256(u)                        # well-formed but wrong base
    with pytest.raises(fa.IncompatibleModelError) as e:
        fa.verify_update(u, base["meta"])
    assert e.value.code == code


def test_sample_counts_are_checked(base):
    with pytest.raises(ValueError):
        fa.build_update(local_weights=base["w"], base_contract=base["contract"], base_contract_sha256=base["sha"],
                        base_model_version="v", update_id="u", round_id=1, site_id="S", device_id="D",
                        num_samples=5, class_counts={"NORMAL": 4})
    with pytest.raises(ValueError):
        fa.build_update(local_weights=base["w"], base_contract=base["contract"], base_contract_sha256=base["sha"],
                        base_model_version="v", update_id="u", round_id=1, site_id="S", device_id="D",
                        num_samples=1, class_counts={"SOILING": 1})


def test_compatible_global_weights_are_accepted(base):
    w2 = perturbed(base["w"])
    w, rep = fa.import_global_weights(compatible_doc(base, w2), base["meta"], expected_base_version="c2-local-v1")
    assert rep["accepted"] and rep["sha256_supplied"] and all(np.allclose(w[k], w2[k]) for k in w)


@pytest.mark.parametrize("mutate, code", [
    (lambda d: [d.pop(k) for k in fa.COMPAT_KEYS], "MISSING_COMPATIBILITY_METADATA"),   # Member 3's current format
    (lambda d: d.update(feature_version="raw8"), "INCOMPATIBLE_PREPROCESSING"),
    (lambda d: d.update(feature_config_sha256="0" * 64), "INCOMPATIBLE_FEATURE_CONFIG_SHA256"),
    (lambda d: d.update(labels=list(reversed(d["labels"]))), "INCOMPATIBLE_LABELS"),
    (lambda d: d.update(activation="relu"), "INCOMPATIBLE_ACTIVATION"),
    (lambda d: d["weights"]["W1"][0].__setitem__(0, float("nan")), "NON_FINITE_WEIGHTS"),
    (lambda d: d["weights"]["b2"].__setitem__(0, 1e6), "WEIGHT_OUT_OF_RANGE"),
    (lambda d: d["weights"].update(b2=[0.0] * 4), "BAD_LAYOUT"),
    (lambda d: d.update(weights_sha256="0" * 64), "INTEGRITY"),
    (lambda d: d.update(base_model_version="v1.0"), "BASE_VERSION"),
    (lambda d: d.pop("weights"), "BAD_DOCUMENT"),
])
def test_incompatible_global_weights_are_rejected(base, mutate, code):
    d = compatible_doc(base, perturbed(base["w"]))
    mutate(d)
    with pytest.raises(fa.IncompatibleModelError) as e:
        fa.import_global_weights(d, base["meta"], expected_base_version="c2-local-v1")
    assert e.value.code == code


def test_frozen_contract_is_not_modified(base):
    before = file_sha256(CONTRACT_PATH)
    fa.import_global_weights(compatible_doc(base, perturbed(base["w"])), base["meta"])
    assert file_sha256(CONTRACT_PATH) == before and base["contract"]["weights_sha256"].startswith("94e7ab93")

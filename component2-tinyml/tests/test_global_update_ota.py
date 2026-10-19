"""Global-model update pipeline and OTA package format (mock only).

Uses the frozen contract read-only, synthetic "global" weights (frozen
weights plus noise) and synthetic calibration rows from the test data
generator; no real dataset, no OPERATION rows, no flashing."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tinyml import global_update as gu
from tinyml import ota
from tinyml.device_reference import STATUS_OK, preprocess_f32, raw_matrix
from tinyml.export import contract_feature_config, contract_weights, file_sha256
from tinyml.features import transform
from tinyml.holdout import inference_frame

from test_features import make_dataset

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def snapshot():
    return {str(p.relative_to(MODELS)): file_sha256(p) for p in sorted(MODELS.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def setup():
    before = snapshot()
    base = json.loads((MODELS / "model_contract.json").read_text())
    base_sha = file_sha256(MODELS / "model_contract.json")
    r = np.random.default_rng(5)
    w = {k: v + r.normal(0, 0.01, v.shape) for k, v in contract_weights(base).items()}
    cand = gu.build_candidate(base, base_sha, w, "v2.0", source="SYNTHETIC_TEST_WEIGHTS")
    df = make_dataset(runs_per_window=3, readings=6)
    tr = df[df.window == "TRAINING"].reset_index(drop=True)            # synthetic generator rows
    cfg = contract_feature_config(base)
    status, z32 = preprocess_f32(raw_matrix(tr), cfg)
    x64, _ = transform(inference_frame(tr), cfg)
    ok = status == STATUS_OK
    cand_sha = gu.contract_text_sha256(cand)
    q = gu.quantize_candidate(cand, cand_sha, z32[ok], {"window": "TRAINING", "source": "SYNTHETIC_TEST_ROWS"})
    ver = gu.verify_candidate(cand, q, x64, z32[ok])
    return {"base": base, "base_sha": base_sha, "w": w, "cand": cand, "q": q, "ver": ver, "z32": z32[ok],
            "before": before}


def test_candidate_keeps_preprocessing_and_gets_a_new_version(setup):
    c, b = setup["cand"], setup["base"]
    assert c["model_version"] == "c2-fl-v2.0" and c["status"] == "CANDIDATE_NOT_VALIDATED"
    for k in gu.UNCHANGED_KEYS:
        assert c[k] == b[k]
    assert c["weights_sha256"] != b["weights_sha256"] and c["candidate"]["base_model_version"] == "c2-local-v1"
    assert "NOT_EVALUATED" in c["candidate"]["evaluation"]


def test_candidate_validation_refuses_changed_preprocessing(setup):
    import copy
    c = copy.deepcopy(setup["cand"])
    c["feature_config"]["low_light_lux"] = 1000.0
    with pytest.raises(ValueError, match="feature_config"):
        gu.validate_candidate(c, setup["base"])
    c = copy.deepcopy(setup["cand"])
    c["weights"]["b2"][0] += 0.1
    with pytest.raises(ValueError, match="weights_sha256"):
        gu.validate_candidate(c, setup["base"])
    with pytest.raises(ValueError):
        gu.build_candidate(setup["base"], setup["base_sha"], setup["w"], "../x", source="t")


def test_quantisation_needs_explicit_training_rows(setup):
    sha = gu.contract_text_sha256(setup["cand"])
    with pytest.raises(ValueError, match="explicitly supplied"):
        gu.quantize_candidate(setup["cand"], sha, None, {"window": "TRAINING"})
    with pytest.raises(ValueError, match="TRAINING"):
        gu.quantize_candidate(setup["cand"], sha, setup["z32"], {"window": "OPERATION"})


def test_verification_reports_agreement_not_accuracy(setup):
    v = setup["ver"]
    assert v["rows"] > 0 and 0.0 <= v["agreement"] <= 1.0 and "not independent accuracy" in v["note"]


def test_candidate_directory_is_versioned_and_never_overwritten(tmp_path, setup):
    d = gu.write_candidate(tmp_path, setup["cand"], setup["q"], setup["ver"])
    m = json.loads((d / "manifest.json").read_text())
    assert d.name == "c2-fl-v2.0" and set(m["files"]) == {"model_contract.json", "quant_params.json",
                                                           "verification.json"}
    assert all(file_sha256(d / f) == s for f, s in m["files"].items())
    with pytest.raises(FileExistsError):
        gu.write_candidate(tmp_path, setup["cand"])
    with pytest.raises(ValueError, match="frozen models"):
        gu.write_candidate(MODELS, setup["cand"])
    with pytest.raises(ValueError, match="frozen models"):
        gu.write_candidate(MODELS / "int8", setup["cand"])


# ---------------------------------------------------------------- OTA package

@pytest.fixture
def package(tmp_path, setup):
    p = tmp_path / "pkg.zip"
    ota.build_package(p, setup["cand"], setup["base"], setup["q"], created_utc="2026-10-16T00:00:00Z")
    return p


def device(setup):
    return ota.MockDevice(setup["base"])


def test_package_is_valid_for_the_frozen_device(package, setup):
    res = ota.validate_package(package, ota.device_state_from_contract(setup["base"]))
    assert res["accepted"] and res["status"] == "VERIFIED", res["errors"]
    m = res["manifest"]
    assert m["package_format"] == "pv-ota-1" and m["backend"] == "INT8" and m["deployment_status"] == "NOT_DEPLOYED"
    assert m["model_version"] == "c2-fl-v2.0" and m["base_model_version"] == "c2-local-v1"
    assert m["rollback"]["rollback_to_model_version"] == "c2-local-v1" and m["signature"] is None
    assert any("UNSIGNED" in w for w in res["warnings"])


def test_package_build_is_deterministic_and_never_overwrites(tmp_path, setup, package):
    p2 = tmp_path / "pkg2.zip"
    ota.build_package(p2, setup["cand"], setup["base"], setup["q"], created_utc="2026-10-16T00:00:00Z")
    assert p2.read_bytes() == package.read_bytes()
    with pytest.raises(FileExistsError):
        ota.build_package(package, setup["cand"], setup["base"], setup["q"])


def rewrite(src, dst, change):
    files = ota.read_package(src)
    change(files)
    with zipfile.ZipFile(dst, "w") as z:
        for n, b in files.items():
            z.writestr(n, b)
    return dst


def remanifest(files, edit):
    m = json.loads(files[ota.MANIFEST])
    edit(m)
    files[ota.MANIFEST] = (json.dumps(m, indent=1, sort_keys=True) + "\n").encode()
    files[ota.MANIFEST_SHA] = (ota._sha(files[ota.MANIFEST]) + "\n").encode()


@pytest.mark.parametrize("change, error", [
    (lambda f: f.update({"model_contract.json": f["model_contract.json"].replace(b"c2-fl-v2.0", b"c2-fl-v2.1")}),
     "FILE_INTEGRITY"),
    (lambda f: f.update({ota.MANIFEST: f[ota.MANIFEST].replace(b"c2-local-v1", b"c2-local-v0")}), "MANIFEST_INTEGRITY"),
    (lambda f: f.update({"extra.bin": b"x"}), "UNEXPECTED_FILES"),
    (lambda f: f.pop(ota.MANIFEST_SHA), "MISSING_MANIFEST"),
    (lambda f: remanifest(f, lambda m: m.update(base_model_version="c2-other")), "BASE_VERSION_MISMATCH"),
    (lambda f: remanifest(f, lambda m: m.update(feature_version="raw8")), "INCOMPATIBLE_FEATURE_VERSION"),
    (lambda f: remanifest(f, lambda m: m.update(feature_config_sha256="0" * 64)), "INCOMPATIBLE_PREPROCESSING"),
    (lambda f: remanifest(f, lambda m: m["compatibility"].update(activation="relu")), "INCOMPATIBLE_MODEL_INTERFACE"),
    (lambda f: remanifest(f, lambda m: m.update(model_hash="0" * 64)), "MODEL_HASH"),
    (lambda f: remanifest(f, lambda m: m["rollback"].update(rollback_weights_sha256="0" * 64)), "ROLLBACK_METADATA"),
    (lambda f: remanifest(f, lambda m: m.update(quant_params_sha256="0" * 64)), "QUANT_PARAMS_MISMATCH"),
])
def test_tampered_or_incompatible_packages_are_rejected(tmp_path, package, setup, change, error):
    bad = rewrite(package, tmp_path / "bad.zip", change)
    res = ota.validate_package(bad, ota.device_state_from_contract(setup["base"]))
    assert not res["accepted"] and any(e.startswith(error) for e in res["errors"]), res["errors"]


def test_not_a_zip_is_rejected(tmp_path, setup):
    p = tmp_path / "x.zip"
    p.write_bytes(b"not a zip")
    assert ota.validate_package(p, ota.device_state_from_contract(setup["base"]))["status"] == "REJECTED"


def test_mock_update_confirms_or_rolls_back(package, setup):
    d = device(setup)
    assert d.apply(package) == "ACTIVE_PENDING_CONFIRMATION"
    assert d.state["active_model_version"] == "c2-fl-v2.0"
    assert d.apply(package) == "REJECTED"                                # one update at a time
    assert d.confirm(self_check_passed=False) == "ROLLED_BACK"
    assert d.state["active_model_version"] == "c2-local-v1"
    assert [h["status"] for h in d.history][:4] == ["RECEIVED", "VERIFIED", "STAGED", "ACTIVE_PENDING_CONFIRMATION"]
    assert d.apply(package) == "ACTIVE_PENDING_CONFIRMATION" and d.confirm(True) == "CONFIRMED"
    assert d.state["active_model_version"] == "c2-fl-v2.0"
    assert d.apply(package) == "REJECTED"                                # base no longer matches
    with pytest.raises(RuntimeError):
        d.confirm(True)


def test_frozen_artifacts_untouched_by_pipeline(setup):
    assert snapshot() == setup["before"]

"""Task 3: global-model export (fl_run.export_global_document /
save_global_document) and import into the Member 2 candidate pipeline
(fl_run.import_and_build_candidate -> fl_adapter.import_global_weights ->
global_update.build_candidate / write_candidate).

Global weights are Model v1 plus a small synthetic change, or the result of
ONE synthetic FL round (save=False). Files go to tmp_path only. No dataset is
read, nothing is trained beyond that one round and no accuracy is measured."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from tinyml import fl_data as fd
from tinyml import fl_run
from tinyml.contract import LABELS
from tinyml.export import file_sha256, flatten, verify_artifacts, weights_sha256
from tinyml.features import FEATURES
from tinyml.fl_adapter import IncompatibleModelError, import_global_weights, local_model_metadata

RUN = "m2init-test-s0"


def _tree_hashes(*dirs):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in dirs if d.exists() for p in sorted(d.rglob("*")) if p.is_file()}


PROTECTED = (fd.MODELS_DIR, fd.COMPONENT / "results", fd.MEMBER3_ROOT / "models", fd.MEMBER3_ROOT / "results")
BEFORE = _tree_hashes(*PROTECTED)                                   # taken when the module is imported


@pytest.fixture(scope="module")
def v1():
    contract, _, w = fd.load_frozen_model()
    meta = local_model_metadata(contract, file_sha256(fd.MODELS_DIR / "model_contract.json"))
    return {"contract": contract, "w": w, "meta": meta}


def candidate(v1, seed=1, scale=0.01, version="v1.1"):
    r = np.random.default_rng(seed)
    w = {k: v + r.normal(0, scale, v.shape) for k, v in v1["w"].items()}
    return fl_run.make_candidate(v1["contract"], w, version, 0, {"save": False},
                                 "fl_data.member3_get_data (TRAINING only)")


@pytest.fixture
def doc(v1):
    return fl_run.export_global_document(candidate(v1), RUN)


# ---------------------------------------------------------------- export

def test_export_has_member3_format_plus_compatibility_metadata(doc, v1):
    assert {"version", "weights"} <= set(doc)                          # Member 3's readers still work
    for k in ("feature_version", "feature_config_sha256", "features", "labels", "activation", "output_activation",
              "param_layout", "n_params"):
        assert doc[k] == v1["meta"][k]                                  # taken from the frozen contract
    assert doc["features"] == list(FEATURES) and doc["labels"] == list(LABELS)
    assert doc["activation"] == "tanh" and doc["output_activation"] == "softmax" and doc["n_params"] == 229
    assert doc["base_model_version"] == "c2-local-v1" and doc["base_weights_sha256"] == v1["contract"]["weights_sha256"]
    assert doc["version"].startswith(f"{RUN}-v1.1-") and doc["status"] == "NOT_VALIDATED"
    assert doc["deployment_status"] == "NOT_DEPLOYED" and "PENDING SAMUDI REVIEW" in doc["review_note"]


def test_export_refuses_a_candidate_that_does_not_match_model_v1(v1):
    for k, bad in (("feature_config_sha256", "0" * 64), ("base_model_version", "v1.0"), ("activation", "relu"),
                   ("base_weights_sha256", "0" * 64)):
        c = candidate(v1)
        c[k] = bad
        with pytest.raises(ValueError, match=k):
            fl_run.export_global_document(c, RUN)
    c = candidate(v1)
    c["weights"]["b2"] = c["weights"]["b2"] + 1e-6                     # weights no longer match their hash
    with pytest.raises(ValueError, match="weights_sha256"):
        fl_run.export_global_document(c, RUN)


# ---------------------------------------------------------------- import (existing fl_adapter checks)

def test_exported_document_imports_with_all_229_values_unchanged(doc, v1, tmp_path):
    c = candidate(v1)
    path, _ = fl_run.save_global_document(doc, tmp_path)
    loaded = json.loads(path.read_text())
    w, report = import_global_weights(loaded, v1["meta"], expected_base_version="c2-local-v1")
    assert np.array_equal(flatten(w), flatten(c["weights"])) and flatten(w).size == 229   # exact after JSON
    assert {k: v.shape for k, v in w.items()} == {"W1": (8, 16), "b1": (16,), "W2": (16, 5), "b2": (5,)}
    assert report["accepted"] and report["sha256_supplied"]
    assert report["weights_sha256"] == doc["weights_sha256"] == c["weights_sha256"] == weights_sha256(w)


@pytest.mark.parametrize("change, code", [
    (lambda d: d.update(features=d["features"][::-1]), "INCOMPATIBLE_FEATURES"),
    (lambda d: d.update(feature_config_sha256="0" * 64), "INCOMPATIBLE_FEATURE_CONFIG_SHA256"),
    (lambda d: d.update(activation="relu"), "INCOMPATIBLE_ACTIVATION"),
    (lambda d: d.update(output_activation="sigmoid"), "INCOMPATIBLE_OUTPUT_ACTIVATION"),
    (lambda d: d.update(labels=[d["labels"][1], d["labels"][0], *d["labels"][2:]]), "INCOMPATIBLE_LABELS"),
    (lambda d: d.update(feature_version="raw8"), "INCOMPATIBLE_PREPROCESSING"),
    (lambda d: d["weights"].update(W1=np.asarray(d["weights"]["W1"]).T.tolist()), "BAD_LAYOUT"),
    (lambda d: d["weights"].update(b2=d["weights"]["b2"][:4]), "BAD_LAYOUT"),
    (lambda d: d["weights"]["W2"][0].__setitem__(0, float("nan")), "NON_FINITE_WEIGHTS"),
    (lambda d: d.update(base_model_version="v1.0"), "BASE_VERSION"),
    (lambda d: d["weights"]["b1"].__setitem__(0, d["weights"]["b1"][0] + 1e-9), "INTEGRITY"),
    (lambda d: d.pop("weights_sha256"), "MISSING_HASH"),
    (lambda d: d.update(base_weights_sha256="0" * 64), "BASE_MISMATCH"),
    (lambda d: d.pop("labels"), "MISSING_COMPATIBILITY_METADATA"),
])
def test_incompatible_documents_are_rejected_with_a_code(doc, tmp_path, change, code):
    bad = copy.deepcopy(doc)
    change(bad)
    with pytest.raises(IncompatibleModelError) as e:
        fl_run.import_and_build_candidate(bad, tmp_path)
    assert e.value.code == code
    assert not (tmp_path / "candidates").exists()                       # nothing written for a rejected document


def test_member3_plain_global_files_are_still_rejected(v1, tmp_path):
    plain = json.loads((fd.MEMBER3_ROOT / "models" / "global_v1.20.json").read_text())
    with pytest.raises(IncompatibleModelError) as e:
        fl_run.import_and_build_candidate(plain, tmp_path)
    assert e.value.code == "MISSING_COMPATIBILITY_METADATA"


# ---------------------------------------------------------------- candidate

def test_candidate_is_new_versioned_not_validated_and_keeps_the_frozen_preprocessing(doc, v1, tmp_path):
    cand, d, _ = fl_run.import_and_build_candidate(doc, tmp_path)
    base = v1["contract"]
    assert cand["model_version"] == f"c2-fl-{doc['version']}" != base["model_version"]
    assert cand["status"] == "CANDIDATE_NOT_VALIDATED" and cand["candidate"]["evaluation"].startswith("NOT_EVALUATED")
    assert cand["candidate"]["source"]["deployment_status"] == "NOT_DEPLOYED"
    assert cand["candidate"]["base_model_version"] == "c2-local-v1"
    assert cand["candidate"]["base_weights_sha256"] == base["weights_sha256"]
    for k in ("feature_config", "scaler", "features", "labels", "activation", "param_layout"):
        assert cand[k] == base[k]
    assert cand["weights_sha256"] == doc["weights_sha256"]
    assert d == (tmp_path / "candidates" / cand["model_version"]).resolve()
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["files"]["model_contract.json"] == file_sha256(d / "model_contract.json")


def test_candidate_versions_are_unique(v1, tmp_path):
    a = fl_run.export_global_document(candidate(v1, seed=1), RUN)
    b = fl_run.export_global_document(candidate(v1, seed=2), RUN)          # other weights
    c = fl_run.export_global_document(candidate(v1, seed=1), "m2init-other")  # other run
    assert len({a["version"], b["version"], c["version"]}) == 3
    fl_run.import_and_build_candidate(a, tmp_path)
    with pytest.raises(FileExistsError):                                   # same candidate twice: never overwritten
        fl_run.import_and_build_candidate(a, tmp_path)


# ---------------------------------------------------------------- safe saving

def test_existing_files_are_never_overwritten(doc, tmp_path):
    path, sha = fl_run.save_global_document(doc, tmp_path)
    with pytest.raises(FileExistsError):
        fl_run.save_global_document(doc, tmp_path)
    assert file_sha256(path) == sha


def test_symlink_at_the_target_is_not_followed(doc, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    (tmp_path / f"global_{doc['version']}.json").symlink_to(victim)
    with pytest.raises(FileExistsError):
        fl_run.save_global_document(doc, tmp_path)
    assert victim.read_text() == "keep"


@pytest.mark.parametrize("root", [fd.MODELS_DIR, fd.COMPONENT / "results", fd.COMPONENT / "notebooks",
                                  fd.MEMBER3_ROOT, fd.MEMBER3_ROOT / "models" / "runs", fd.MEMBER3_ROOT / "results"])
def test_protected_directories_are_refused(doc, root):
    with pytest.raises(fl_run.UnsafePathError):
        fl_run.save_global_document(doc, root)
    with pytest.raises(fl_run.UnsafePathError):
        fl_run.import_and_build_candidate(doc, root)


def test_traversal_and_symlink_escapes_are_refused(doc, tmp_path):
    link = tmp_path / "escape"
    link.symlink_to(fd.MODELS_DIR, target_is_directory=True)
    with pytest.raises(fl_run.UnsafePathError):                         # allowed root that is really models/
        fl_run.save_global_document(doc, link)
    with pytest.raises(fl_run.UnsafePathError):
        fl_run.safe_target(tmp_path / ".." / "outside.json", tmp_path)
    with pytest.raises(fl_run.UnsafePathError):
        fl_run.safe_target(link / "model_contract.json", tmp_path)
    for bad in ("../../models/x", "a/b", ""):
        with pytest.raises(ValueError):
            fl_run.save_global_document(dict(doc, version=bad), tmp_path)


def test_default_folder_is_git_ignored():
    assert fl_run.CANDIDATES_DIR == fd.COMPONENT / "fl_candidates"
    assert "fl_candidates/" in (fd.COMPONENT / ".gitignore").read_text().splitlines()


# ---------------------------------------------------------------- data safety

def test_export_and_import_never_read_the_dataset(doc, tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("dataset read during export/import")
    monkeypatch.setattr(pd, "read_csv", refuse)
    monkeypatch.setattr(fd, "load_dataset", refuse)
    path, _ = fl_run.save_global_document(doc, tmp_path)
    cand, _, _ = fl_run.import_and_build_candidate(path, tmp_path)
    assert cand["candidate"]["source"]["lineage"]["operation_rows_used"] == 0


# ---------------------------------------------------------------- end to end: one synthetic FL round

def test_one_round_engine_output_exports_and_imports(v1, tmp_path, monkeypatch):
    pytest.importorskip("flwr")
    from test_fl_run import six_site_dataset
    from tinyml.splits import make_assignments
    engine = fl_run.member3_engine()
    monkeypatch.setattr(engine, "N_ROUNDS", 1)
    df = six_site_dataset()
    _, cfg, _ = fd.load_frozen_model()
    d = fd.build_fl_data(df, cfg, make_assignments(df, n_splits=5, seed=42))
    out = fl_run.run_from_model_v1(data=(d.site_data, d.val, d.cfg), engine=engine)
    doc = fl_run.export_global_document(out["candidate"], RUN)
    path, _ = fl_run.save_global_document(doc, tmp_path)
    cand, cdir, report = fl_run.import_and_build_candidate(path, tmp_path)
    assert doc["version"].startswith(f"{RUN}-v1.1-") and report["accepted"]
    assert np.array_equal(flatten({k: np.asarray(v) for k, v in cand["weights"].items()}),
                          flatten(out["candidate"]["weights"]))
    assert cdir.parent == (tmp_path / "candidates").resolve()


# ---------------------------------------------------------------- nothing protected changed (runs last)

def test_model_v1_and_saved_research_files_are_unchanged():
    verify_artifacts(fd.MODELS_DIR)
    assert _tree_hashes(*PROTECTED) == BEFORE
    assert not fl_run.CANDIDATES_DIR.exists() or not any(fl_run.CANDIDATES_DIR.iterdir())

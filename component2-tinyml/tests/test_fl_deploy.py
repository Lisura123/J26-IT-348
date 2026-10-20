"""Task 4: device preparation of an FL Global Model Candidate (fl_deploy.py), host only.

The FL Global Model Candidate here is software-generated (Model v1 plus a small
random change; not an FL result), exported and imported with Task 3 into a
temporary folder. Calibration uses either Test Data (software-generated
fixture runs, test_features.make_dataset) or the PV Research Dataset
(pv_fault_dataset.csv, TRAINING rows only, SHA-256 checked) through the frozen
FeatureConfig. Host gcc/g++ builds are not Arduino-ESP32 builds; nothing runs
on an ESP32 and no hardware measurement exists. Agreement numbers are
prediction agreement, never independent accuracy."""
import hashlib
import json
import re
import shutil

import numpy as np
import pytest

from test_features import make_dataset
from tinyml import fl_data as fd
from tinyml import fl_deploy, fl_run
from tinyml.contract import LABELS
from tinyml.export import contract_feature_config, file_sha256
from tinyml.features import STATUS_OK, transform
from tinyml.hostc import FIRMWARE, FIRMWARE_TEST, compiler
from tinyml.quantize import QUANT_VERSION, QuantParams, compare_predictions, validate_params
from tinyml.selfcheck import selfcheck_header

pytestmark = pytest.mark.skipif(compiler() is None or shutil.which("g++") is None, reason="needs gcc and g++")


def _tree_hashes(*dirs):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in dirs if d.exists() for p in sorted(d.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


PROTECTED = (fd.MODELS_DIR, fd.COMPONENT / "results", fd.COMPONENT / "firmware", fd.COMPONENT / "tests" / "host_c",
             fd.MEMBER3_ROOT / "models", fd.MEMBER3_ROOT / "results", fd.MEMBER3_ROOT / "src", fd.MEMBER3_ROOT / "data")
BEFORE = _tree_hashes(*PROTECTED)
POSITIONS = (1, 2, 3, 4)


TEST_CANDIDATE_SOURCE = "TEST_DATA: software-generated candidate (Model v1 + small random change); not an FL result"
DATASET = fd.MEMBER3_ROOT / "data" / "shared" / "pv_fault_dataset.csv"        # the PV Research Dataset
needs_dataset = pytest.mark.skipif(not DATASET.exists(), reason="PV Research Dataset not in this checkout")


def make_test_data():
    """Test Data: software-generated fixture runs (not the PV Research Dataset)."""
    return make_dataset(runs_per_window=6, readings=6)


def make_test_candidate(root, seed=1):
    base, _, w = fd.load_frozen_model()
    r = np.random.default_rng(seed)
    w2 = {k: v + r.normal(0, 0.01, v.shape) for k, v in w.items()}
    cand = fl_run.make_candidate(base, w2, "v1.1", 0, {"save": False}, TEST_CANDIDATE_SOURCE)
    _, cdir, _ = fl_run.import_and_build_candidate(fl_run.export_global_document(cand, "m2test-s0"), root)
    return base, cdir


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("fl_candidates")
    base, cdir = make_test_candidate(root)
    d, report = fl_deploy.prepare_candidate(cdir, test_data=make_test_data(), allowed_root=root, positions=POSITIONS)
    contract = json.loads((cdir / "model_contract.json").read_text())
    q = QuantParams.from_json((d / "quant_params.json").read_text())
    return {"root": root, "cdir": cdir, "d": d, "report": report, "cand": contract, "base": base, "q": q,
            "lib": d / "arduino" / "pv_fault"}


# ---------------------------------------------------------------- calibration and INT8

def test_calibration_uses_training_rows_only_with_the_frozen_config(monkeypatch):
    df = make_test_data()
    seen = []
    real = fl_deploy.raw_matrix
    monkeypatch.setattr(fl_deploy, "raw_matrix", lambda d: (seen.extend(d["window"]), real(d))[1])
    base, cfg, _ = fd.load_frozen_model()
    cal = fl_deploy.training_calibration(df, cfg)
    assert set(seen) == {"TRAINING"} and (df.window == "OPERATION").any()
    tr = df[df.window == "TRAINING"]
    x64, st = transform(tr, cfg)
    assert cal["info"]["window"] == "TRAINING" and len(cal["z32"]) == (st == STATUS_OK).sum() == len(cal["y"])
    assert np.allclose(cal["z32"], x64, atol=1e-5)                       # same frozen scaler, float32 vs float64
    assert cfg.to_json() == contract_feature_config(base).to_json()       # nothing refitted


def test_int8_parameters_are_linked_to_the_candidate(prepared):
    q, cand, base = prepared["q"], prepared["cand"], prepared["base"]
    validate_params(q)
    assert q.version == QUANT_VERSION == "int8-v1"
    assert q.contract_sha256 == file_sha256(prepared["cdir"] / "model_contract.json")
    assert q.weights_sha256 == cand["weights_sha256"] != base["weights_sha256"]
    assert q.model_version == cand["model_version"] and q.features == base["features"] and q.labels == list(LABELS)
    assert q.param_layout == base["param_layout"] and q.activation == "tanh"
    assert q.calibration["window"] == "TRAINING"


def test_calibration_scale_comes_from_the_training_rows(prepared):
    cal = fl_deploy.training_calibration(make_test_data(), contract_feature_config(prepared["cand"]), kind=fl_deploy.TEST_DATA)
    assert prepared["q"].a_x == pytest.approx(float(np.abs(cal["z32"]).max()))


# ---------------------------------------------------------------- comparisons and safety

def test_report_separates_agreement_from_accuracy(prepared):
    ch = prepared["report"]["checks"]
    assert "not independent accuracy" in ch["note"] and prepared["report"]["operation_window_used"] is False
    for k in ("float32_vs_int8", "v1_vs_candidate"):
        assert 0.0 <= ch[k]["agreement"] <= 1.0 and ch[k]["rows"] == ch["rows"] > 0
        assert set(ch[k]["safety_changes"]) == {"OPEN_CIRCUIT", "SHORT_CIRCUIT"}
        assert ch[k]["confidence"]["threshold"] == fl_deploy.CONFIDENCE_THRESHOLD == 0.90
    assert "accuracy" not in json.dumps(ch["int8_acceptance"]["checks"])
    assert prepared["report"]["status"] == "NOT_VALIDATED" and prepared["report"]["esp32"] == "NOT_EXECUTED"


def test_confidence_crossing_counts_both_directions():
    a = np.array([[0.95, .05, 0, 0, 0], [0.89, .11, 0, 0, 0], [0.90, .10, 0, 0, 0], [0.5, .5, 0, 0, 0]])
    b = np.array([[0.85, .15, 0, 0, 0], [0.91, .09, 0, 0, 0], [0.90, .10, 0, 0, 0], [0.6, .4, 0, 0, 0]])
    c = fl_deploy.confidence_crossings(a, b)
    assert (c["fell_below"], c["rose_above"], c["crossings"]) == (1, 1, 2)        # 0.90 itself counts as above
    assert c["a_at_or_above"] == 2 and c["b_at_or_above"] == 2
    assert c["max_abs_confidence_change"] == pytest.approx(0.10)


def test_open_and_short_changes_are_counted_in_both_directions():
    i = {l: n for n, l in enumerate(LABELS)}
    ref = [i["NORMAL"], i["OPEN_CIRCUIT"], i["SHORT_CIRCUIT"], i["DUST"]]
    new = [i["OPEN_CIRCUIT"], i["NORMAL"], i["SHORT_CIRCUIT"], i["PARTIAL_SHADING"]]
    c = compare_predictions(ref, new)
    assert c["safety_changes"]["OPEN_CIRCUIT"] == {"changed_to": 1, "changed_from": 1}
    assert c["safety_changes"]["SHORT_CIRCUIT"] == {"changed_to": 0, "changed_from": 0}
    assert c["safety_changes_total"] == 2 and c["changed"] == 3                 # DUST->SHADING is a label change only


def test_weight_drift_from_model_v1(prepared):
    from tinyml.export import contract_weights, flatten
    bw, cw = contract_weights(prepared["base"]), contract_weights(prepared["cand"])
    assert fl_deploy.weight_drift(bw, bw)["l2"] == 0.0
    d = prepared["report"]["checks"]["weight_drift_from_v1"]
    assert d["l2"] == pytest.approx(float(np.linalg.norm(flatten(cw) - flatten(bw)))) and d["l2"] > 0


# ---------------------------------------------------------------- headers

def test_candidate_headers_and_regenerated_selfcheck(prepared):
    d, cand, q = prepared["d"], prepared["cand"], prepared["q"]
    sha = file_sha256(prepared["cdir"] / "model_contract.json")
    gen = {p.name: p.read_text() for p in (d / "firmware" / "generated").iterdir()}
    assert sorted(gen) == sorted(fl_deploy.GENERATED)
    assert f'#define PV_MODEL_VERSION "{cand["model_version"]}"' in gen["pv_feature_params.h"]
    assert all(sha in t for t in (gen["pv_feature_params.h"], gen["pv_model_f32.h"]))
    sc = (d / "firmware" / "selfcheck" / "pv_selfcheck.h").read_text()
    assert sc == selfcheck_header(cand, sha, q)                                # regenerated for this candidate
    assert sc != (FIRMWARE / "selfcheck" / "pv_selfcheck.h").read_text()       # not Model v1's
    assert (d / "firmware" / "golden" / "golden_vectors.h").stat().st_size > 0
    manifest = json.loads((d / "manifest.json").read_text())
    assert all(file_sha256(d / rel) == h for rel, h in manifest["files"].items())
    assert manifest["deployment_status"] == "NOT_DEPLOYED" and manifest["arduino_compile"] == "NOT_EXECUTED"


def test_host_c_golden_tests_pass_against_the_candidate_headers(prepared):
    assert prepared["report"]["host_c_golden_tests"] == {"ran": True, "passed": True}
    assert fl_deploy.host_golden_check(prepared["d"])["passed"]


# ---------------------------------------------------------------- create-only and paths

def test_deploy_is_create_only(prepared):
    with pytest.raises(FileExistsError):
        fl_deploy.prepare_candidate(prepared["cdir"], test_data=make_test_data(), allowed_root=prepared["root"],
                                    positions=POSITIONS)
    p = prepared["d"] / "report.json"
    before = p.read_bytes()
    with pytest.raises(FileExistsError):
        fl_deploy._create(p, "x")
    assert p.read_bytes() == before


def test_symlink_target_is_not_followed(tmp_path):
    victim = tmp_path / "victim.h"
    victim.write_text("keep")
    (tmp_path / "pv_model_f32.h").symlink_to(victim)
    with pytest.raises(FileExistsError):
        fl_deploy._create(tmp_path / "pv_model_f32.h", "x")
    assert victim.read_text() == "keep"


@pytest.mark.parametrize("target", [fd.MODELS_DIR, fd.COMPONENT / "firmware" / "pv_fault" / "generated",
                                    fd.MEMBER3_ROOT / "models"])
def test_protected_and_outside_paths_are_refused(prepared, target):
    with pytest.raises(fl_run.UnsafePathError):
        fl_deploy.prepare_candidate(target, test_data=make_test_data(), allowed_root=prepared["root"], positions=POSITIONS)
    with pytest.raises(fl_run.UnsafePathError):
        fl_deploy.prepare_candidate(prepared["cdir"], test_data=make_test_data(), allowed_root=target, positions=POSITIONS)


# ---------------------------------------------------------------- Arduino library (host checks only)

def test_arduino_library_layout(prepared):
    lib = prepared["lib"]
    props = dict(l.split("=", 1) for l in (lib / "library.properties").read_text().splitlines() if "=" in l)
    assert props["name"] == "pv_fault" and props["architectures"] == "esp32" and "NOT_VALIDATED" in props["sentence"]
    src = lib / "src"
    expected = {*fl_deploy.LIB_SOURCES, *fl_deploy.LIB_HEADERS, *fl_deploy.TEST_FILES, *fl_deploy.GENERATED,
                "pv_selfcheck.h", "golden_vectors.h"}
    assert {p.name for p in src.iterdir()} == expected and not any(p.is_dir() for p in src.iterdir())   # flat
    for n in fl_deploy.LIB_SOURCES + fl_deploy.LIB_HEADERS:
        assert (src / n).read_bytes() == (FIRMWARE / n).read_bytes()          # unchanged copies
    for n in fl_deploy.TEST_FILES:
        assert (src / n).read_bytes() == (FIRMWARE_TEST / n).read_bytes()
    assert sorted(p.name for p in (lib / "examples" / "pv_selftest").iterdir()) == ["pv_selftest.ino"]


def test_sketch_is_self_test_only_with_c_linkage(prepared):
    ino = (prepared["lib"] / "examples" / "pv_selftest" / "pv_selftest.ino").read_text()
    assert re.search(r'extern "C" \{\s*#include "pv_api.h"\s*#include "pv_selftest.h"\s*\}', ino)
    assert "PVINFO" in ino and "PVTEST_DONE" in ino and "pv_selftest_run" in ino
    for banned in ("WiFi", "pv_relay", "digitalWrite", "pinMode", "Update.", "esp_ota", "HTTPClient", "analogRead"):
        assert banned not in ino
    assert not any(n in {p.name for p in (prepared["lib"] / "src").iterdir()}
                   for n in ("pv_relay.c", "pv_sensor.c", "pv_message.c", "pv_pipeline.c"))


def test_host_c_and_cpp_build_of_the_library_runs_the_self_test(prepared):
    r = fl_deploy.host_arduino_check(prepared["lib"])
    assert r["ran"] and r["passed"], r["output"][-2000:]
    assert '"target":"host-g++-arduino-stub"' in r["output"] and "arduino-esp32" not in r["output"]
    m = re.search(r"^PVTEST (\{.*\})$", r["output"], re.M)
    rep = json.loads(m.group(1))
    assert rep["result"] == "PASS" and rep["failures"] == 0 and rep["model_version"] == prepared["cand"]["model_version"]
    assert rep["int8_exact_ok"] == rep["int8_total"] > 0
    assert "HOST_ONLY" in r["status"]


def test_mismatched_selfcheck_is_rejected(prepared, tmp_path):
    lib = tmp_path / "pv_fault"
    shutil.copytree(prepared["lib"], lib)
    (lib / "src" / "pv_selfcheck.h").unlink()
    shutil.copyfile(FIRMWARE / "selfcheck" / "pv_selfcheck.h", lib / "src" / "pv_selfcheck.h")   # Model v1's
    r = fl_deploy.host_arduino_check(lib)
    assert r["ran"] and not r["passed"] and "PVTEST_DONE FAIL" in r["output"]


def test_missing_extern_c_fails_to_link(prepared, tmp_path):
    lib = tmp_path / "pv_fault"
    shutil.copytree(prepared["lib"], lib)
    ino = lib / "examples" / "pv_selftest" / "pv_selftest.ino"
    text = ino.read_text().replace('extern "C" {', "").replace('#include "pv_selftest.h"\n}', '#include "pv_selftest.h"')
    ino.unlink()
    ino.write_text(text)
    r = fl_deploy.host_arduino_check(lib)
    assert not r["ran"] and "undefined reference" in r["output"]


def test_real_arduino_compile_is_reported_honestly(prepared):
    s = prepared["report"]["arduino"]
    assert s["status"].startswith(("NOT_EXECUTED", "AVAILABLE"))
    if shutil.which("arduino-cli") is None:
        assert s["status"] == "NOT_EXECUTED: arduino-cli not installed"


# ---------------------------------------------------------------- terminology and data labels

def test_report_labels_data_candidate_and_hardware(prepared):
    rep, q = prepared["report"], prepared["q"]
    origin = rep["fl_global_model_candidate"]["origin"]
    assert origin["kind"] == "TEST_DATA_CANDIDATE" and "not an FL result" in origin["fl_sites"]
    cal = rep["calibration"]
    assert cal["data_kind"] == fl_deploy.TEST_DATA and cal["dataset_sha256"] is None
    assert "not the PV Research Dataset" in cal["data_description"] and "not hardware measurements" in cal["data_description"]
    assert rep["hardware_measurements"].startswith("NONE") and rep["golden"]["derived_from"].startswith("Test Data")
    m = json.loads((prepared["d"] / "manifest.json").read_text())
    assert m["artifact"] == "FL Global Model Candidate" and m["calibration_data_kind"] == "TEST_DATA"
    assert m["candidate_origin"] == "TEST_DATA_CANDIDATE" and m["hardware_measurements"].startswith("NONE")
    assert q.calibration["data_kind"] == "TEST_DATA"                             # recorded in quant_params.json too


def test_candidate_origin_keeps_simulated_sites_and_test_candidates_apart():
    def lineage(src):
        return {"candidate": {"source": {"lineage": {"data_source": src}}}}
    fl = fl_deploy.candidate_origin(lineage("fl_data.member3_get_data (TRAINING only)"))
    assert fl["kind"] == "FL_ENGINE_ON_PV_RESEARCH_DATASET" and "not physical devices" in fl["fl_sites"]
    assert fl_deploy.candidate_origin(lineage(TEST_CANDIDATE_SOURCE))["kind"] == "TEST_DATA_CANDIDATE"
    other = fl_deploy.candidate_origin(lineage("caller-supplied (OPERATION exclusion not checked here)"))
    assert other["kind"] == "FL_ENGINE_ON_CALLER_DATA"


def test_exactly_one_calibration_source_is_required(prepared):
    for kw in ({}, {"dataset_path": DATASET, "test_data": make_test_data()}):
        with pytest.raises(ValueError, match="exactly one"):
            fl_deploy.prepare_candidate(prepared["cdir"], allowed_root=prepared["root"], **kw)


# ---------------------------------------------------------------- PV Research Dataset (TRAINING rows only)

@pytest.fixture(scope="module")
def pv_cal():
    if not DATASET.exists():
        pytest.skip("PV Research Dataset not in this checkout")
    sha = file_sha256(DATASET)
    base, cfg, _ = fd.load_frozen_model()
    return {"sha": sha, "base": base, "cal": fl_deploy.training_calibration(fd.load_dataset(DATASET), cfg, sha)}


@needs_dataset
def test_pv_research_dataset_calibration_reproduces_the_official_int8_v1_parameters(pv_cal):
    """Model v1 through this calibration path gives exactly the committed
    models/int8/quant_params.json: same rows, same rule, nothing refitted."""
    from tinyml.quantize import quantize_model
    official = QuantParams.from_json((fd.MODELS_DIR / "int8" / "quant_params.json").read_text())
    info = pv_cal["cal"]["info"]
    assert info["data_kind"] == "PV_RESEARCH_DATASET" and info["rows"] == 3000 and info["ok_rows"] == 2924
    assert official.contract_sha256 == file_sha256(fd.MODELS_DIR / "model_contract.json")
    q = quantize_model(pv_cal["base"], official.contract_sha256, pv_cal["cal"]["z32"], info)
    for k in ("a_x", "a_z1", "s_x", "s_w1", "s_w2", "s_z1", "s_h", "inv_sx_f32", "out_scale_f32", "m1", "sh1",
              "qw1", "qb1", "qw2", "qb2", "tanh_lut"):
        assert getattr(q, k) == getattr(official, k), k


@needs_dataset
def test_candidate_prepared_with_pv_research_dataset_calibration(pv_cal, tmp_path):
    _, cdir = make_test_candidate(tmp_path)
    d, rep = fl_deploy.prepare_candidate(cdir, dataset_path=DATASET, allowed_root=tmp_path)
    cal = rep["calibration"]
    assert cal["data_kind"] == "PV_RESEARCH_DATASET" and cal["dataset_sha256"] == pv_cal["sha"]
    assert cal["rows"] == 2924 and rep["checks"]["rows"] == 2924 and rep["operation_window_used"] is False
    assert "Not hardware measurements" in cal["data_description"]
    assert rep["fl_global_model_candidate"]["origin"]["kind"] == "TEST_DATA_CANDIDATE"
    assert rep["host_c_golden_tests"] == {"ran": True, "passed": True} and rep["host_arduino_check"]["passed"]
    assert json.loads((d / "manifest.json").read_text())["calibration_data_kind"] == "PV_RESEARCH_DATASET"
    assert file_sha256(DATASET) == pv_cal["sha"]                                 # the dataset is never changed


# ---------------------------------------------------------------- nothing protected changed (runs last)

def test_official_model_v1_firmware_and_member3_files_unchanged(prepared):
    from tinyml.export import verify_artifacts
    verify_artifacts(fd.MODELS_DIR)
    assert _tree_hashes(*PROTECTED) == BEFORE
    assert not fl_run.CANDIDATES_DIR.exists() or not any(fl_run.CANDIDATES_DIR.iterdir())

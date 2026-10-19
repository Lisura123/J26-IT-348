"""Phase 10 code generation and host C tests (synthetic models and data).

The official generated headers and golden vectors are produced later by the
approved quantisation run; the last test checks them only if they exist."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tinyml import codegen, golden
from tinyml.device_reference import STATUS_OK, preprocess_f32, raw_matrix
from tinyml.export import build_contract, file_sha256, training_metadata, write_artifacts
from tinyml.hostc import compile_and_run, compiler
from tinyml.quantize import QuantParams, int8_predict, quantize_model
from tinyml.splits import make_assignments, to_json
from tinyml.train import train_final

from test_features import make_dataset

ROOT = Path(__file__).resolve().parents[1]
needs_cc = pytest.mark.skipif(compiler() is None, reason="no C compiler")


@pytest.fixture(scope="module")
def synth():
    df = make_dataset(runs_per_window=5, readings=6)
    a = make_assignments(df)
    res = train_final(df, seed=0, assignments=a)
    contract = build_contract(res, "0" * 64, "runwise-v1", "test", {"python": "t"})
    tr = df[df.window == "TRAINING"].reset_index(drop=True)
    cfg = res["config"]
    status, z = preprocess_f32(raw_matrix(tr), cfg)
    q = quantize_model(contract, "c" * 64, z[status == STATUS_OK], {"window": "TRAINING"})
    z_in = golden.training_representatives(tr, cfg, 2, (1, 3, 6)) + golden.reference_probes(contract)
    protos = golden.class_prototypes(tr, cfg)
    raw_in = protos + golden.edge_cases(protos[0][2], cfg)
    g = golden.build_golden(contract, cfg, q, z_in, raw_in)
    return {"df": df, "a": a, "res": res, "contract": contract, "tr": tr, "cfg": cfg, "q": q, "g": g}


def write(tmp, s, q=None, g=None):
    return codegen.write_headers(tmp, s["contract"], "c" * 64, s["cfg"], q or s["q"], g or s["g"])


# ---------------------------------------------------------------- headers

def test_headers_are_deterministic(tmp_path, synth):
    a = [p.read_bytes() for p in write(tmp_path / "a", synth)]
    b = [p.read_bytes() for p in write(tmp_path / "b", synth)]
    assert a == b


def test_headers_keep_feature_label_order_and_provenance(tmp_path, synth):
    paths = {p.name: p.read_text() for p in write(tmp_path, synth)}
    fp = paths["pv_feature_params.h"]
    assert ('#define PV_FEATURE_NAMES_INIT {"v_deficit", "i_norm", "p_norm", "dust_vo_v", "temp_spread", '
            '"temp_rise", "voltage_v", "current_ma"}') in fp
    assert ('#define PV_LABEL_NAMES_INIT {"NORMAL", "PARTIAL_SHADING", "DUST", "OPEN_CIRCUIT", '
            '"SHORT_CIRCUIT"}') in fp
    assert synth["contract"]["weights_sha256"] in fp and "c" * 64 in fp
    assert "placeholder threshold" in fp
    int8 = paths["pv_model_int8.h"]
    assert f"#define PV_M1 ((int32_t){synth['q'].m1})" in int8 and "PV_TANH_LUT[256]" in int8
    assert "deployment derivative" in int8


def test_float_constants_are_exact_hex(synth):
    v = np.float32(0.1)
    assert codegen.c_float(v) == f"{float(v).hex()}f" and float.fromhex(codegen.c_float(v)[:-1]) == float(v)
    assert codegen.c_float(float("nan")) == "NAN" and codegen.c_float(-np.inf) == "(-INFINITY)"


def test_codegen_refuses_wrong_order(tmp_path, synth):
    bad = json.loads(json.dumps(synth["contract"]))
    bad["labels"] = bad["labels"][::-1]
    with pytest.raises(ValueError):
        codegen.feature_params_header(bad, "c" * 64, synth["cfg"])


def test_golden_origins_are_labelled(synth):
    g = synth["g"]
    assert {c["origin"] for c in g["z_cases"]} == {golden.ORIGIN_TRAINING, golden.ORIGIN_PROBE}
    assert {c["origin"] for c in g["raw_cases"]} == {golden.ORIGIN_PROTOTYPE, golden.ORIGIN_EDGE}
    s = golden.golden_summary(g)
    assert all(s["raw_status_counts"][k] > 0 for k in range(4))       # every gate outcome covered
    labels = {c["label_int8"] for c in g["raw_cases"] if c["origin"] == golden.ORIGIN_PROTOTYPE}
    assert labels == {0, 1, 2, 3, 4}                                  # all five classes covered


def test_golden_selection_refuses_operation_rows(synth):
    with pytest.raises(ValueError):
        golden.training_representatives(synth["df"], synth["cfg"])
    with pytest.raises(ValueError):
        golden.class_prototypes(synth["df"], synth["cfg"])


def test_extreme_valid_reading_clips_inputs(synth):
    case = next(c for c in synth["g"]["raw_cases"] if c["id"] == "valid_extreme_temp_spread")
    assert case["status"] == STATUS_OK
    assert int8_predict(case["z"], synth["q"])["input_clips"] > 0


# ---------------------------------------------------------------- host C

@needs_cc
def test_c_matches_python_golden_vectors(tmp_path, synth):
    write(tmp_path, synth)
    r = compile_and_run(tmp_path)
    assert r["ran"] and r["passed"], r["output"]


@needs_cc
def test_c_detects_a_wrong_expectation(tmp_path, synth):
    g = json.loads(json.dumps(synth["g"], default=lambda o: o.tolist() if hasattr(o, "tolist") else sorted(o)))
    g["core"][1]["acc2"][0] += 1
    g["origins"] = set(g["origins"])
    write(tmp_path, synth, g=g)
    r = compile_and_run(tmp_path)
    assert r["ran"] and not r["passed"] and "core 1 acc2" in r["output"]


@needs_cc
def test_c_extreme_parameters_without_overflow(tmp_path, synth):
    """All weights +-127, biases at the int32 limit, largest M1 and SH1 = 62:
    C (with UBSan) and Python must agree on every core vector."""
    d = json.loads(synth["q"].to_json())
    d["qw1"] = [[127 if (i + j) % 2 else -127 for j in range(16)] for i in range(8)]
    d["qb1"] = [(2 ** 31 - 1 - 8 * 127 * 127) * (1 if j % 2 else -1) for j in range(16)]
    d["qw2"] = [[127 if (j + c) % 3 else -127 for c in range(5)] for j in range(16)]
    d["qb2"] = [(2 ** 31 - 1 - 16 * 127 * 127) * (1 if c % 2 else -1) for c in range(5)]
    d["m1"], d["sh1"] = 2 ** 31 - 1, 62
    q = QuantParams.from_json(json.dumps(d))
    s = synth
    g = golden.build_golden(s["contract"], s["cfg"], q, golden.reference_probes(s["contract"]),
                            golden.class_prototypes(s["tr"], s["cfg"])[:3])
    write(tmp_path, s, q=q, g=g)
    r = compile_and_run(tmp_path)
    assert r["ran"] and r["passed"], r["output"]


# ---------------------------------------------------------------- script

def _script():
    spec = importlib.util.spec_from_file_location("quant_script", ROOT / "scripts" / "07_quantize_model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_script_uses_training_only_and_never_overwrites(tmp_path, synth):
    s = synth
    model_dir = tmp_path / "models"
    write_artifacts(model_dir, s["contract"], s["cfg"], training_metadata(s["res"], s["contract"]))
    before = {p.name: file_sha256(p) for p in model_dir.iterdir()}
    df = s["df"].copy()
    df.loc[df.window == "OPERATION", ["voltage_v", "irradiance_lux"]] = np.nan   # would break anything using it
    data = tmp_path / "pv_fault_dataset.csv"
    df.to_csv(data, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(to_json(s["a"], {"split_version": "runwise-v1", "dataset_sha256": file_sha256(data)}))
    m = _script()
    kw = dict(expected_dataset_sha256=None, pinned=None, pinned_weights=None, expected_rows=None,
              representatives=(2, (1, 3, 6)), run_c_tests=compiler() is not None)
    report, q, g = m.run(data, splits, model_dir, tmp_path / "out", code_commit="test", **kw)
    assert report["operation_window_used"] is False and report["window"] == "TRAINING"
    assert report["device_preprocessing_check"]["status_identical"]
    assert report["comparison"]["int8_vs_float_reference"]["rows"] == (df.window == "TRAINING").sum()
    assert report["acceptance"]["checks"]["safety_changes_zero"]
    if compiler():
        assert report["host_c_tests"]["passed"], report["host_c_tests"]["output"]
    out = tmp_path / "out"
    manifest = json.loads((out / "models/int8/manifest.json").read_text())
    assert manifest["kind"] == "INT8_DEPLOYMENT_DERIVATIVE"
    assert all(file_sha256(out / k) == v for k, v in manifest["files"].items())
    assert {p.name: file_sha256(p) for p in model_dir.iterdir()} == before       # float model untouched
    with pytest.raises(FileExistsError):
        m.run(data, splits, model_dir, out, code_commit="test", **kw)


# ---------------------------------------------------------------- official artifacts (after the approved run)

GEN = ROOT / "firmware" / "pv_fault" / "generated"
OFFICIAL_MANIFEST = ROOT / "models" / "int8" / "manifest.json"


@pytest.mark.skipif(not OFFICIAL_MANIFEST.exists(), reason="official quantisation run not committed yet")
def test_official_int8_artifacts_and_c_tests(tmp_path):
    manifest = json.loads(OFFICIAL_MANIFEST.read_text())
    assert all(file_sha256(ROOT / k) == v for k, v in manifest["files"].items())
    assert manifest["contract_sha256"] == file_sha256(ROOT / "models" / "model_contract.json")
    if compiler():
        r = compile_and_run(GEN, ROOT / "tests" / "host_c", out_dir=tmp_path)
        assert r["passed"], r["output"]

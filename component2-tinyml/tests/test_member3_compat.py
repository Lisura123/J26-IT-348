"""Cross-checks against Member 3's ACTUAL code (read-only).

Files are extracted with `git archive` from origin/feature/mongodb-migration
into a temporary directory and imported from there; the teammate branch is
never modified. Skipped when the branch or its Python dependencies (pyyaml,
python-dotenv) are not available."""
import importlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from tinyml import fl_adapter, messaging
from tinyml.export import file_sha256

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
REF = "origin/feature/mongodb-migration"


def _extract(tmp, *paths):
    try:
        tar = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", REF, *paths],
                             capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"{REF} not available")
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(tmp, filter="data")
    return tmp


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    pytest.importorskip("yaml")
    pytest.importorskip("dotenv")
    root = _extract(tmp_path_factory.mktemp("m3"), "project-backend")
    pb = root / "project-backend"
    sys.path.insert(0, str(pb))
    try:
        for m in [m for m in sys.modules if m == "backend" or m.startswith("backend.")]:
            del sys.modules[m]
        validate = importlib.import_module("backend.validate")
        yield validate
    finally:
        sys.path.remove(str(pb))
        for m in [m for m in sys.modules if m == "backend" or m.startswith("backend.")]:
            del sys.modules[m]


def test_sample_messages_pass_member3_backend_validator(backend):
    for f in sorted((ROOT / "docs" / "examples").glob("prediction_*.json")):
        msg = json.loads(f.read_text(encoding="utf-8"))
        res = backend.validate_record("prediction", msg)
        assert res.success, (f.name, res.reject_code, res.reject_reason)
        assert res.point.fields["record_id"] == msg["record_id"]
        assert res.point.measurement == "fault_predictions"


def test_member3_backend_rejects_what_pv_msg_forbids(backend):
    msg = json.loads((ROOT / "docs" / "examples" / "prediction_classified_int8.json").read_text())
    bad = dict(msg, predicted_label="SOILING")
    assert backend.validate_record("prediction", bad).reject_code == "BAD_VALUE"
    bad = dict(msg, timestamp_utc="2026-09-10T00:26:40")
    assert backend.validate_record("prediction", bad).reject_code == "TIME_NOT_UTC"
    bad = dict(msg, device_id="ESP32-B01")                  # device must match the site registry
    assert backend.validate_record("prediction", bad).reject_code == "DEVICE_MISMATCH"


@pytest.fixture(scope="module")
def fl(tmp_path_factory):
    return _extract(tmp_path_factory.mktemp("m3fl"), "component3-federated-backend/src/mlp.py",
                    "component3-federated-backend/src/compression.py",
                    "component3-federated-backend/models/global_v1.20.json",
                    "component3-federated-backend/data/shared/fl_weight_updates.jsonl") / \
        "component3-federated-backend"


def test_layout_matches_member3_mlp_layout(fl):
    src = (fl / "src" / "mlp.py").read_text()
    assert 'LAYOUT = [("W1", (8, 16)), ("b1", (16,)), ("W2", (16, 5)), ("b2", (5,))]' in src
    assert fl_adapter.MEMBER3_LAYOUT == [["W1", [8, 16]], ["b1", [16]], ["W2", [16, 5]], ["b2", [5]]]


def test_int8_dense_matches_member3_compression(fl):
    src = (fl / "src" / "compression.py").read_text()
    body = src[src.index("def int8_dense"):src.index("def topk_int8")]
    ns = {"np": np}
    exec(body, ns)                                          # Member 3's function, unchanged
    rng = np.random.default_rng(0)
    for vec in (rng.normal(0, 0.05, 229), np.zeros(229), np.linspace(-1, 1, 229)):
        q, scale = fl_adapter.int8_dense(vec)
        assert np.array_equal(ns["int8_dense"](vec), q * scale)


def test_member3_global_model_file_is_rejected(fl):
    contract = json.loads((ROOT / "models" / "model_contract.json").read_text())
    meta = fl_adapter.local_model_metadata(contract, file_sha256(ROOT / "models" / "model_contract.json"))
    doc = json.loads((fl / "models" / "global_v1.20.json").read_text())
    assert set(doc) == {"version", "weights"}               # no feature version, hash or activation
    with pytest.raises(fl_adapter.IncompatibleModelError) as e:
        fl_adapter.import_global_weights(doc, meta)
    assert e.value.code == "MISSING_COMPATIBILITY_METADATA"


def test_member3_update_record_fields_are_kept(fl):
    rec = json.loads((fl / "data" / "shared" / "fl_weight_updates.jsonl").read_text().splitlines()[0])
    contract = json.loads((ROOT / "models" / "model_contract.json").read_text())
    from tinyml.export import contract_weights
    w = contract_weights(contract)
    ours = fl_adapter.build_update(local_weights=w, base_contract=contract,
                                   base_contract_sha256=file_sha256(ROOT / "models" / "model_contract.json"),
                                   base_model_version="v1.0", update_id="R01-SITE-A", round_id=1, site_id="SITE-A",
                                   device_id="ESP32-A01", num_samples=10, class_counts={"NORMAL": 10})
    shared = set(rec) - {"delta_float_device_only"}
    assert shared <= set(ours)                               # every Member 3 field is present
    assert ours["param_layout"] == rec["param_layout"]
    with pytest.raises(fl_adapter.IncompatibleModelError):
        fl_adapter.verify_update(rec, fl_adapter.local_model_metadata(contract, "x"))   # raw8 record refused

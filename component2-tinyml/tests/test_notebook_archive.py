"""Archived executed notebooks vs clean templates (notebooks/ARCHIVED_RUNS.json).

The saved Colab notebooks are kept exactly as saved, with outputs and with
CONFIRM_* = True. These tests make that a recorded, checked state instead of
an unnoticed one: every repository notebook must be a clean template or a
registered archive with its exact SHA-256, and archived notebooks must keep
the reviewed safety code."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tinyml import notebook_archive as na

ROOT = Path(__file__).resolve().parents[1]
REG = na.registry()
ENTRIES = {e["path"]: e for e in REG["notebooks"]}


def test_every_repository_notebook_is_a_template_or_a_registered_archive():
    states = {p.relative_to(na.REPO).as_posix(): na.classify(p, REG) for p in na.repository_notebooks()}
    assert states, "no notebooks found"
    bad = {k: v for k, v in states.items() if v not in (na.TEMPLATE, na.ARCHIVED)}
    assert bad == {}, f"unregistered executed or modified notebooks: {bad}"


@pytest.mark.parametrize("rel", sorted(ENTRIES))
def test_archived_notebook_is_byte_exact_and_matches_its_record(rel):
    e = ENTRIES[rel]
    path = na.REPO / rel
    assert path.exists() and na.file_sha256(path) == e["sha256"]
    nb = na.load(path)
    assert na.settings(nb) == e["settings"] and na.has_outputs(nb) == e["has_outputs"]
    assert e["state"] == na.ARCHIVED and e["rerun_effect"] and e["documents"]


@pytest.mark.parametrize("rel", sorted(r for r, e in ENTRIES.items() if e["template_code_sha256"]))
def test_archived_notebook_keeps_the_reviewed_safety_code(rel):
    """Only the setting values and outputs differ from the reviewed template."""
    assert na.normalised_code_sha256(na.load(na.REPO / rel)) == ENTRIES[rel]["template_code_sha256"]


@pytest.mark.parametrize("rel", sorted(r for r, e in ENTRIES.items() if e["official_run_commit"]))
def test_archived_approved_commit_is_the_documented_official_run(rel):
    e = ENTRIES[rel]
    assert e["settings"]["APPROVED_COMMIT"] == e["official_run_commit"]
    doc = {"colab_quantization.ipynb": "docs/QUANTIZATION.md", "colab_pruning.ipynb": "docs/PRUNING.md"}[
        Path(rel).name]
    assert e["official_run_commit"] in (ROOT / doc).read_text(encoding="utf-8")


def test_archived_operation_notebook_cannot_repeat_the_evaluation():
    """The registry says BLOCKED: the notebook runs the latest code, which
    refuses the final test while the official result is committed."""
    rel = "component2-tinyml/notebooks/colab_operation_eval.ipynb"
    e = ENTRIES[rel]
    assert e["settings"] == {"CONFIRM_FINAL_TEST": True} and e["rerun_effect"].startswith("BLOCKED")
    code = "\n".join(na.code_cells(na.load(na.REPO / rel)))
    assert 'reset", "--hard", f"origin/{BRANCH}"' in code            # always the latest branch code
    assert "op_eval.run(" in code and "allow_repeat_operation_eval" not in code
    assert (ROOT / "results" / "operation_eval.json").exists()
    assert "allow_repeat_operation_eval=False" in (ROOT / "scripts" / "06_evaluate_operation.py").read_text()


# ---------------------------------------------------------------- the classifier catches risky saves

@pytest.fixture
def archived_copy(tmp_path):
    src = na.REPO / "component2-tinyml/notebooks/colab_pruning.ipynb"
    dst = tmp_path / "colab_pruning.ipynb"
    shutil.copyfile(src, dst)
    return dst


def test_unregistered_executed_notebook_is_refused(archived_copy):
    assert na.classify(archived_copy, REG) == na.UNREGISTERED          # same bytes, but not registered here


def test_modified_archived_notebook_is_detected(tmp_path, monkeypatch):
    rel = "component2-tinyml/notebooks/colab_pruning.ipynb"
    fake_repo = tmp_path / "repo"
    (fake_repo / "component2-tinyml/notebooks").mkdir(parents=True)
    target = fake_repo / rel
    shutil.copyfile(na.REPO / rel, target)
    monkeypatch.setattr(na, "REPO", fake_repo)
    assert na.classify(target, REG) == na.ARCHIVED
    target.write_bytes(target.read_bytes().replace(b"CONFIRM_PRUNING = True", b"CONFIRM_PRUNING = 1==1"))
    assert na.classify(target, REG) == na.MODIFIED


def test_clean_template_is_recognised(archived_copy):
    nb = na.load(archived_copy)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            for name in na.settings({"cells": [c]}):
                src = na.set_setting(src, name, "" if name == "APPROVED_COMMIT" else False)
            c["source"], c["outputs"], c["execution_count"] = src, [], None
    archived_copy.write_text(json.dumps(nb), encoding="utf-8")
    assert na.is_template(na.load(archived_copy)) and na.classify(archived_copy, REG) == na.TEMPLATE
    first_code = next(c for c in nb["cells"] if c["cell_type"] == "code")
    first_code["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "x"}]
    archived_copy.write_text(json.dumps(nb), encoding="utf-8")
    assert na.classify(archived_copy, REG) == na.UNREGISTERED          # outputs alone make it executed


def test_set_setting_refuses_missing_or_ambiguous_assignments():
    with pytest.raises(ValueError):
        na.set_setting("x = 1\n", "CONFIRM_PRUNING", True)
    with pytest.raises(ValueError):
        na.set_setting("CONFIRM_PRUNING = False\nCONFIRM_PRUNING = False\n", "CONFIRM_PRUNING", True)
    assert na.set_setting("CONFIRM_PRUNING = True   # note\n", "CONFIRM_PRUNING", False) == \
        "CONFIRM_PRUNING = False   # note\n"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_registry_file_is_reproducible():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "10_register_archived_notebooks.py"), "--check"],
                       capture_output=True, text=True)
    if "unknown revision" in r.stderr or "bad object" in r.stderr:
        pytest.skip("reviewed template commit not in this clone")
    assert r.returncode == 0, r.stdout + r.stderr

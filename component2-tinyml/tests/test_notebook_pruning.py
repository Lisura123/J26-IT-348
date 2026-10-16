"""Safety guards in notebooks/colab_pruning.ipynb.

The cells are executed with local temporary git repositories and a stub study
function, so the official pruning study is never run and no network access is
needed."""
import json
import subprocess
import types
from pathlib import Path

import pytest

from test_notebooks import git, repos  # noqa: F401  (pytest fixture)

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "colab_pruning.ipynb"


def code_cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def step1_source(repo_dir, approved):
    src = code_cells()[0]
    src = src.split("subprocess.run([sys.executable")[0]            # stop before pip install
    src = src.replace('APPROVED_COMMIT = ""', f'APPROVED_COMMIT = "{approved}"')
    return src.replace('"/content/J26-IT-348"', repr(str(repo_dir)))


def test_seven_steps_guards_and_training_only():
    cells = code_cells()
    assert [c.splitlines()[0][:9] for c in cells] == [f"# STEP {i}:" for i in range(1, 8)]
    s1, s2, s4, s5 = cells[0], cells[1], cells[3], cells[4]
    assert 'APPROVED_COMMIT = ""' in s1 and "--is-ancestor" in s1 and "--detach" in s1
    assert "reset" not in s1
    assert "verify_frozen_model(MODEL_DIR)" in s2 and "load_frozen_int8" in s2
    assert "check_dataset_sha256" in s4
    assert "CONFIRM_PRUNING = False" in s5
    assert s5.index("if not CONFIRM_PRUNING") < s5.index("prune_study.run(")
    assert s5.index("head_now != APPROVED_COMMIT") < s5.index("prune_study.run(")
    text = "\n".join(cells)
    assert '"OPERATION"' not in text and "read_csv" not in text        # rows are selected only in the script
    assert "/content/prune_out" in s5 and "MODEL_DIR, OUT_ROOT" in s5    # output outside models/


def test_step1_refuses_missing_or_malformed_hash(monkeypatch):
    def no_git(*a, **k):
        raise AssertionError("git must not run before the hash is validated")
    monkeypatch.setattr(subprocess, "run", no_git)
    for bad in ("", "cf7cf49", "Z" * 40):
        with pytest.raises(SystemExit, match="40-character"):
            exec(step1_source("/nonexistent", bad), {})


def test_step1_checks_out_exactly_the_approved_older_revision(repos):  # noqa: F811
    g = {}
    exec(step1_source(repos["clone"], repos["c1"]), g)
    assert git(repos["clone"], "rev-parse", "HEAD") == repos["c1"]
    assert g["COMMIT"] == repos["c1"][:7]


def test_step1_refuses_commit_from_another_branch(repos):  # noqa: F811
    with pytest.raises(SystemExit, match="not part of origin/component2-tinyml"):
        exec(step1_source(repos["clone"], repos["other"]), {})


def step5_globals(repo_dir, approved, calls):
    def fake_run(*args, **kw):
        calls.append(kw)
        return {"training_rows": 1, "status_ok_rows": 1, "operation_window_used": False,
                "variants": [{"name": "baseline", "e1_checks": {"float32": {"passed": True},
                                                                "int8": {"passed": True}}}],
                "estimate_note": "ESTIMATE"}
    return {"subprocess": subprocess, "REPO_DIR": str(repo_dir), "APPROVED_COMMIT": approved,
            "COMMIT": approved[:7], "COMPONENT_DIR": "/c", "DATA_PATH": "/d.csv", "MODEL_DIR": "/m",
            "prune_study": types.SimpleNamespace(run=fake_run)}


def test_step5_refuses_without_confirmation(repos):  # noqa: F811
    calls = []
    with pytest.raises(SystemExit, match="CONFIRM_PRUNING = True"):
        exec(code_cells()[4], step5_globals(repos["clone"], repos["c2"], calls))
    assert calls == []


def test_step5_refuses_wrong_revision_even_when_confirmed(repos):  # noqa: F811
    calls = []
    src = code_cells()[4].replace("CONFIRM_PRUNING = False", "CONFIRM_PRUNING = True")
    with pytest.raises(SystemExit, match="not the approved"):
        exec(src, step5_globals(repos["clone"], repos["c1"], calls))
    assert calls == []


def test_step5_runs_once_confirmed_on_the_approved_revision(repos):  # noqa: F811
    calls = []
    src = code_cells()[4].replace("CONFIRM_PRUNING = False", "CONFIRM_PRUNING = True")
    exec(src, step5_globals(repos["clone"], repos["c2"], calls))
    assert len(calls) == 1 and calls[0]["code_commit"] == repos["c2"][:7]

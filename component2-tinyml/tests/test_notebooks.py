"""Safety guards in notebooks/colab_quantization.ipynb.

The cells are executed with local temporary git repositories and a stub
quantisation function, so the official quantisation is never run and no
network access is needed."""
import json
import re
import subprocess
import types
from pathlib import Path

import pytest

from tinyml import notebook_archive as na

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "colab_quantization.ipynb"


def code_cells():
    return na.code_cells(na.load(NB))           # empty cells (added by Colab) are skipped


def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(cwd), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repos(tmp_path):
    """origin with component2-tinyml (c1 -> c2) and an unrelated branch; a clone of it."""
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q", "-b", "component2-tinyml")
    (origin / "f.txt").write_text("1")
    git(origin, "add", "f.txt"); git(origin, "commit", "-q", "-m", "c1")
    c1 = git(origin, "rev-parse", "HEAD")
    (origin / "f.txt").write_text("2")
    git(origin, "commit", "-q", "-am", "c2")
    c2 = git(origin, "rev-parse", "HEAD")
    git(origin, "checkout", "-q", "-b", "other", c1)
    (origin / "g.txt").write_text("x")
    git(origin, "add", "g.txt"); git(origin, "commit", "-q", "-m", "other")
    other = git(origin, "rev-parse", "HEAD")
    git(origin, "checkout", "-q", "component2-tinyml")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "--branch", "component2-tinyml", str(origin), str(clone)], check=True)
    git(clone, "fetch", "-q", "origin", "other")
    return {"clone": clone, "c1": c1, "c2": c2, "other": other}


def step1_source(repo_dir, approved):
    src = code_cells()[0]
    src = src.split("subprocess.run([sys.executable")[0]            # stop before pip install
    src = na.set_setting(src, "APPROVED_COMMIT", approved)      # never the saved value
    return src.replace('"/content/J26-IT-348"', repr(str(repo_dir)))


# ---------------------------------------------------------------- structure

def test_seven_steps_and_guards_present():
    cells = code_cells()
    assert [c.splitlines()[0][:9] for c in cells] == [f"# STEP {i}:" for i in range(1, 8)]
    s1, s5 = cells[0], cells[4]
    assert re.search(r"^APPROVED_COMMIT = ", s1, re.M) and "--is-ancestor" in s1 and "--detach" in s1
    assert "reset" not in s1                                        # never resets to a newer revision
    assert re.search(r"^CONFIRM_QUANTIZATION = ", s5, re.M)
    assert s5.index("if not CONFIRM_QUANTIZATION") < s5.index("quant.run(")
    assert s5.index("head_now != APPROVED_COMMIT") < s5.index("quant.run(")
    s2, s4 = cells[1], cells[3]
    assert "verify_frozen_model(MODEL_DIR)" in s2 and "check_dataset_sha256" in s4


# ---------------------------------------------------------------- Step 1 revision pinning

def test_step1_refuses_missing_or_malformed_hash(monkeypatch):
    def no_git(*a, **k):
        raise AssertionError("git must not run before the hash is validated")
    monkeypatch.setattr(subprocess, "run", no_git)
    for bad in ("", "34fc60e", "Z" * 40):
        with pytest.raises(SystemExit, match="40-character"):
            exec(step1_source("/nonexistent", bad), {})


def test_step1_checks_out_exactly_the_approved_older_revision(repos, capsys):
    g = {}
    exec(step1_source(repos["clone"], repos["c1"]), g)
    assert git(repos["clone"], "rev-parse", "HEAD") == repos["c1"]   # not the newer c2
    assert g["COMMIT"] == repos["c1"][:7]
    assert "Newer commits on origin/component2-tinyml (not used): 1" in capsys.readouterr().out


def test_step1_refuses_unknown_commit(repos):
    with pytest.raises(SystemExit, match="not found"):
        exec(step1_source(repos["clone"], "0" * 40), {})


def test_step1_refuses_commit_from_another_branch(repos):
    with pytest.raises(SystemExit, match="not part of origin/component2-tinyml"):
        exec(step1_source(repos["clone"], repos["other"]), {})
    assert git(repos["clone"], "rev-parse", "HEAD") != repos["other"]


# ---------------------------------------------------------------- Step 5 confirmation guard

def step5_globals(repo_dir, approved, calls):
    def fake_run(*args, **kw):
        calls.append(kw)
        return {"acceptance": {"checks": {"x": True}, "int8_accepted": True, "deployment_candidate": "-"},
                "host_c_tests": {"output": "ALL PASSED"}}, None, None
    return {"subprocess": subprocess, "REPO_DIR": str(repo_dir), "APPROVED_COMMIT": approved,
            "COMMIT": approved[:7],
            "COMPONENT_DIR": "/c", "DATA_PATH": "/d.csv", "MODEL_DIR": "/m",
            "quant": types.SimpleNamespace(run=fake_run)}


def test_step5_refuses_without_confirmation(repos):
    calls = []
    with pytest.raises(SystemExit, match="CONFIRM_QUANTIZATION = True"):
        exec(na.set_setting(code_cells()[4], "CONFIRM_QUANTIZATION", False), step5_globals(repos["clone"], repos["c2"], calls))
    assert calls == []


def test_step5_refuses_wrong_revision_even_when_confirmed(repos):
    calls = []
    src = na.set_setting(code_cells()[4], "CONFIRM_QUANTIZATION", True)
    with pytest.raises(SystemExit, match="not the approved"):
        exec(src, step5_globals(repos["clone"], repos["c1"], calls))     # clone HEAD is c2
    assert calls == []


def test_step5_runs_once_confirmed_on_the_approved_revision(repos):
    calls = []
    src = na.set_setting(code_cells()[4], "CONFIRM_QUANTIZATION", True)
    exec(src, step5_globals(repos["clone"], repos["c2"], calls))
    assert len(calls) == 1 and calls[0]["code_commit"] == repos["c2"][:7]


def test_notebook_is_a_clean_template_or_a_registered_archive():
    """Either the safe template (guard False, no hash, no outputs) or the
    byte-exact archived run recorded in notebooks/ARCHIVED_RUNS.json."""
    state = na.classify(NB)
    assert state in (na.TEMPLATE, na.ARCHIVED), state
    s = na.settings(na.load(NB))
    if state == na.TEMPLATE:
        assert s["CONFIRM_QUANTIZATION"] is False and s["APPROVED_COMMIT"] == ""
    else:
        entry = next(e for e in na.registry()["notebooks"] if e["path"].endswith("/" + NB.name))
        assert s == entry["settings"] and s["APPROVED_COMMIT"] == entry["official_run_commit"]


def test_guard_still_refuses_with_the_template_value():
    """The guard logic in the file refuses when the value is False, whatever was saved."""
    src = na.set_setting(code_cells()[4], "CONFIRM_QUANTIZATION", False)
    assert src.index("if not CONFIRM_QUANTIZATION") < src.index("quant.run(")

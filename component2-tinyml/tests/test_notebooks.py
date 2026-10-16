"""Safety guards in notebooks/colab_quantization.ipynb.

The cells are executed with local temporary git repositories and a stub
quantisation function, so the official quantisation is never run and no
network access is needed."""
import json
import subprocess
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "colab_quantization.ipynb"


def code_cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


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
    src = src.replace('APPROVED_COMMIT = ""', f'APPROVED_COMMIT = "{approved}"')
    return src.replace('"/content/J26-IT-348"', repr(str(repo_dir)))


# ---------------------------------------------------------------- structure

def test_seven_steps_and_guards_present():
    cells = code_cells()
    assert [c.splitlines()[0][:9] for c in cells] == [f"# STEP {i}:" for i in range(1, 8)]
    s1, s5 = cells[0], cells[4]
    assert 'APPROVED_COMMIT = ""' in s1 and "--is-ancestor" in s1 and "--detach" in s1
    assert "reset" not in s1                                        # never resets to a newer revision
    assert "CONFIRM_QUANTIZATION = False" in s5
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
        exec(code_cells()[4], step5_globals(repos["clone"], repos["c2"], calls))
    assert calls == []


def test_step5_refuses_wrong_revision_even_when_confirmed(repos):
    calls = []
    src = code_cells()[4].replace("CONFIRM_QUANTIZATION = False", "CONFIRM_QUANTIZATION = True")
    with pytest.raises(SystemExit, match="not the approved"):
        exec(src, step5_globals(repos["clone"], repos["c1"], calls))     # clone HEAD is c2
    assert calls == []


def test_step5_runs_once_confirmed_on_the_approved_revision(repos):
    calls = []
    src = code_cells()[4].replace("CONFIRM_QUANTIZATION = False", "CONFIRM_QUANTIZATION = True")
    exec(src, step5_globals(repos["clone"], repos["c2"], calls))
    assert len(calls) == 1 and calls[0]["code_commit"] == repos["c2"][:7]

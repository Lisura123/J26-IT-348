"""Clean notebook templates vs archived, already-executed Colab notebooks.

A repository notebook is in exactly one of two states:

  TEMPLATE            safe to run: every CONFIRM_* guard is False, APPROVED_COMMIT
                      (if the notebook has one) is "", and no cell has outputs.
  ARCHIVED_EXECUTED   a notebook saved from Colab after an official run
                      ("Save a copy in GitHub"), kept byte-for-byte with its
                      outputs and settings (guards may be True). It must be
                      listed in notebooks/ARCHIVED_RUNS.json with its exact
                      SHA-256; any later edit makes it unregistered.

Anything else (an unregistered executed notebook, or a registered one whose
bytes changed) is refused by the tests. Archived notebooks keep their code:
the normalised code (empty cells dropped, the values of APPROVED_COMMIT and
CONFIRM_* replaced by a placeholder) must match the reviewed template code
recorded in the registry, so the safety logic itself cannot change silently.
"""
import ast
import hashlib
import json
import re
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[2]
REPO = COMPONENT.parent
REGISTRY = COMPONENT / "notebooks" / "ARCHIVED_RUNS.json"
REGISTRY_VERSION = "archived-notebooks-v1"
SETTING = re.compile(r"^(APPROVED_COMMIT|CONFIRM_\w+) = (.*?)(\s+#.*)?$", re.M)
TEMPLATE, ARCHIVED, UNREGISTERED, MODIFIED = "TEMPLATE", "ARCHIVED_EXECUTED", "UNREGISTERED_EXECUTED", \
    "ARCHIVED_BUT_MODIFIED"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_cells(nb):
    """Source of every non-empty code cell (Colab appends an empty one)."""
    return [s for s in ("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code") if s.strip()]


def settings(nb):
    """Saved values of APPROVED_COMMIT and CONFIRM_* (Python literals)."""
    out = {}
    for src in code_cells(nb):
        for m in SETTING.finditer(src):
            out[m.group(1)] = ast.literal_eval(m.group(2).strip())
    return out


def set_setting(src, name, value):
    """Source with the assignment of `name` set to `value` (comment kept).
    Raises if the assignment is missing, so a test can never silently run
    the saved value instead of its own."""
    pattern = re.compile(rf"^{re.escape(name)} = .*?(\s+#.*)?$", re.M)
    new, n = pattern.subn(lambda m: f"{name} = {value!r}{m.group(1) or ''}", src)
    if n != 1:
        raise ValueError(f"expected exactly one assignment of {name}, found {n}")
    return new


def normalised_code_sha256(nb):
    cells = [SETTING.sub(lambda m: f"{m.group(1)} = <VALUE>", s) for s in code_cells(nb)]
    return hashlib.sha256(json.dumps(cells).encode("utf-8")).hexdigest()


def has_outputs(nb):
    return any(c.get("outputs") or c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code")


def is_template(nb):
    s = settings(nb)
    guards_off = all(v is False for k, v in s.items() if k.startswith("CONFIRM_"))
    no_commit = s.get("APPROVED_COMMIT", "") == ""
    return guards_off and no_commit and not has_outputs(nb)


def registry(path=REGISTRY):
    reg = load(path)
    if reg.get("registry_version") != REGISTRY_VERSION:
        raise ValueError("unknown registry version")
    return reg


def classify(path, reg=None):
    """TEMPLATE, ARCHIVED_EXECUTED, UNREGISTERED_EXECUTED or ARCHIVED_BUT_MODIFIED."""
    path = Path(path).resolve()
    reg = reg or registry()
    try:
        rel = path.relative_to(REPO).as_posix()
    except ValueError:                                  # outside the repository: never registered
        rel = None
    entry = next((e for e in reg["notebooks"] if e["path"] == rel), None)
    if entry is not None:
        return ARCHIVED if file_sha256(path) == entry["sha256"] else MODIFIED
    return TEMPLATE if is_template(load(path)) else UNREGISTERED


def repository_notebooks():
    """Every notebook tracked in the component and at the repository root."""
    return sorted(set(COMPONENT.glob("notebooks/*.ipynb")) | set(REPO.glob("*.ipynb")))

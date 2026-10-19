"""Model update package format pv-ota-1 (software side; NO flashing).

A package is a ZIP file with
  ota_manifest.json         metadata below
  ota_manifest.sha256       SHA-256 of the manifest bytes (package integrity)
  model_contract.json       Float32 contract of the candidate
  quant_params.json         INT8 parameters (only for INT8 packages)

Manifest fields: package_format, package_id, created_utc, model_version,
base_model_version, contract_version, feature_version, feature_config_sha256,
quant_version, backend, model_hash (weights SHA-256), quant_params_sha256,
files {name: SHA-256}, compatibility {labels, features, activation,
output_activation, param_layout, n_params, api_version}, rollback
{rollback_to_model_version, rollback_weights_sha256, policy}, signature
(null: signing is NOT implemented) and deployment_status (NOT_DEPLOYED).

validate_package() checks everything that can be checked without a key.
MockDevice simulates A/B model slots with confirm-or-rollback to test the
status flow. The current firmware compiles the model into the image
(generated headers), so applying a package on a real ESP32 needs runtime
model loading or a firmware image update, plus signing; neither exists yet.
No OTA deployment has taken place.
"""
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .export import contract_weights, weights_sha256
from .fl_adapter import MAX_ABS_WEIGHT, check_weights, feature_config_sha256
from .quantize import QuantParams, params_sha256

PACKAGE_FORMAT = "pv-ota-1"
API_VERSION = "pv-api-1"
MANIFEST = "ota_manifest.json"
MANIFEST_SHA = "ota_manifest.sha256"
ZIP_TIME = (2020, 1, 1, 0, 0, 0)               # fixed: deterministic archives
ROLLBACK_POLICY = ("Keep the previous model in the other slot. Activate the new model as PENDING, run the "
                   "pv_api_init self-check and the golden self-test; confirm only if both pass, otherwise roll "
                   "back to the previous slot.")

# device-side status values reported during an update
STATUS = ("RECEIVED", "REJECTED", "VERIFIED", "STAGED", "ACTIVE_PENDING_CONFIRMATION", "CONFIRMED", "ROLLED_BACK")


def _dump(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def compatibility_block(contract):
    return {"labels": list(contract["labels"]), "features": list(contract["features"]),
            "activation": contract["activation"], "output_activation": contract["output_activation"],
            "param_layout": contract["param_layout"], "n_params": contract["n_params"], "api_version": API_VERSION}


def device_state_from_contract(contract):
    """What a device running this contract reports about itself."""
    return {"active_model_version": contract["model_version"], "active_weights_sha256": contract["weights_sha256"],
            "feature_version": contract["feature_version"], "feature_config_sha256": feature_config_sha256(contract),
            "contract_version": contract["contract_version"], "compatibility": compatibility_block(contract)}


def build_manifest(contract, base_contract, q=None, created_utc=None):
    cand = contract.get("candidate", {})
    if cand.get("base_model_version") != base_contract["model_version"]:
        raise ValueError("candidate was not built on this base model")
    files = {"model_contract.json": _sha(_dump(contract).encode("utf-8"))}
    if q is not None:
        files["quant_params.json"] = _sha(q.to_json().encode("utf-8"))
    m = {"package_format": PACKAGE_FORMAT,
         "created_utc": created_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "model_version": contract["model_version"], "base_model_version": base_contract["model_version"],
         "contract_version": contract["contract_version"], "feature_version": contract["feature_version"],
         "feature_config_sha256": feature_config_sha256(contract),
         "quant_version": q.version if q is not None else None, "backend": "INT8" if q is not None else "FLOAT32",
         "model_hash": contract["weights_sha256"], "quant_params_sha256": params_sha256(q) if q is not None else None,
         "files": files, "compatibility": compatibility_block(contract),
         "rollback": {"rollback_to_model_version": base_contract["model_version"],
                      "rollback_weights_sha256": base_contract["weights_sha256"], "policy": ROLLBACK_POLICY},
         "signature": None, "signature_note": "NOT IMPLEMENTED: packages are unsigned; a device must not "
                                              "install unsigned packages in the field",
         "deployment_status": "NOT_DEPLOYED"}
    m["package_id"] = _sha(json.dumps({k: m[k] for k in ("model_version", "base_model_version", "files")},
                                      sort_keys=True).encode("utf-8"))[:16]
    return m


def build_package(out_path, contract, base_contract, q=None, created_utc=None):
    """Write the package ZIP; refuses to overwrite. Returns the manifest."""
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; packages are never overwritten")
    if q is not None and q.contract_sha256 != _sha(_dump(contract).encode("utf-8")):
        raise ValueError("INT8 parameters do not belong to this contract")
    m = build_manifest(contract, base_contract, q, created_utc)
    mbytes = _dump(m).encode("utf-8")
    entries = [(MANIFEST, mbytes), (MANIFEST_SHA, (_sha(mbytes) + "\n").encode("ascii")),
               ("model_contract.json", _dump(contract).encode("utf-8"))]
    if q is not None:
        entries.append(("quant_params.json", q.to_json().encode("utf-8")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "x", zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            info = zipfile.ZipInfo(name, ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    return m


def read_package(path_or_bytes):
    data = Path(path_or_bytes).read_bytes() if not isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return {n: z.read(n) for n in z.namelist()}


def validate_package(path_or_bytes, device_state, max_abs=MAX_ABS_WEIGHT):
    """Returns {"accepted": bool, "status": VERIFIED|REJECTED, "errors": [...],
    "warnings": [...], "manifest": {...}}. Never raises for a bad package."""
    errors, warnings = [], []
    try:
        files = read_package(path_or_bytes)
    except (zipfile.BadZipFile, OSError) as e:
        return {"accepted": False, "status": "REJECTED", "errors": [f"NOT_A_PACKAGE: {e}"], "warnings": [],
                "manifest": None}
    if MANIFEST not in files or MANIFEST_SHA not in files:
        return {"accepted": False, "status": "REJECTED", "errors": ["MISSING_MANIFEST"], "warnings": [],
                "manifest": None}
    mbytes = files[MANIFEST]
    if files[MANIFEST_SHA].decode("ascii", "replace").strip() != _sha(mbytes):
        errors.append("MANIFEST_INTEGRITY")
    try:
        m = json.loads(mbytes)
    except ValueError:
        return {"accepted": False, "status": "REJECTED", "errors": errors + ["BAD_MANIFEST_JSON"], "warnings": [],
                "manifest": None}
    if m.get("package_format") != PACKAGE_FORMAT:
        errors.append(f"UNKNOWN_FORMAT {m.get('package_format')!r}")
    expected_names = {MANIFEST, MANIFEST_SHA, *m.get("files", {})}
    if set(files) != expected_names:
        errors.append(f"UNEXPECTED_FILES {sorted(set(files) ^ expected_names)}")
    for name, sha in m.get("files", {}).items():
        if name not in files or _sha(files[name]) != sha:
            errors.append(f"FILE_INTEGRITY {name}")
    # compatibility with the running device
    if m.get("feature_version") != device_state["feature_version"]:
        errors.append("INCOMPATIBLE_FEATURE_VERSION")
    if m.get("feature_config_sha256") != device_state["feature_config_sha256"]:
        errors.append("INCOMPATIBLE_PREPROCESSING")
    if m.get("contract_version") != device_state["contract_version"]:
        errors.append("INCOMPATIBLE_CONTRACT_VERSION")
    if m.get("compatibility") != device_state["compatibility"]:
        errors.append("INCOMPATIBLE_MODEL_INTERFACE")
    if m.get("base_model_version") != device_state["active_model_version"]:
        errors.append("BASE_VERSION_MISMATCH")
    if m.get("model_version") == device_state["active_model_version"] or \
            m.get("model_hash") == device_state["active_weights_sha256"]:
        errors.append("ALREADY_ACTIVE")
    rb = m.get("rollback") or {}
    if rb.get("rollback_to_model_version") != device_state["active_model_version"] or \
            rb.get("rollback_weights_sha256") != device_state["active_weights_sha256"]:
        errors.append("ROLLBACK_METADATA")
    # content checks
    if "model_contract.json" in files and not any(e.startswith("FILE_INTEGRITY model") for e in errors):
        try:
            c = json.loads(files["model_contract.json"])
            w = contract_weights(c)
            check_weights(w, max_abs)
            if weights_sha256(w) != m.get("model_hash") or c.get("weights_sha256") != m.get("model_hash"):
                errors.append("MODEL_HASH")
            if feature_config_sha256(c) != m.get("feature_config_sha256"):
                errors.append("CONTRACT_PREPROCESSING_DIFFERS")
            if c.get("model_version") != m.get("model_version"):
                errors.append("CONTRACT_VERSION_DIFFERS")
        except (ValueError, KeyError, TypeError) as e:
            errors.append(f"BAD_CONTRACT {e}")
    elif "model_contract.json" not in files:
        errors.append("MISSING_CONTRACT")
    if m.get("backend") == "INT8":
        try:
            q = QuantParams.from_json(files["quant_params.json"].decode("utf-8"))
            if q.contract_sha256 != _sha(files["model_contract.json"]) or params_sha256(q) != m.get(
                    "quant_params_sha256") or q.version != m.get("quant_version"):
                errors.append("QUANT_PARAMS_MISMATCH")
        except (KeyError, ValueError, TypeError) as e:
            errors.append(f"BAD_QUANT_PARAMS {e}")
    elif m.get("backend") != "FLOAT32":
        errors.append("UNKNOWN_BACKEND")
    if m.get("signature") is None:
        warnings.append("UNSIGNED_PACKAGE: integrity is checked, authenticity is not")
    return {"accepted": not errors, "status": "VERIFIED" if not errors else "REJECTED", "errors": errors,
            "warnings": warnings, "manifest": m}


class MockDevice:
    """Simulated A/B model slots for testing the update status flow. It
    stores package contents in memory; it does not flash anything."""

    def __init__(self, contract):
        self.slots = {"A": {"contract": contract, "state": device_state_from_contract(contract)}, "B": None}
        self.active, self.pending = "A", None
        self.history = []

    @property
    def state(self):
        return self.slots[self.active]["state"]

    def _report(self, status, **kw):
        if status not in STATUS:
            raise ValueError(status)
        self.history.append({"status": status, "active_model_version": self.state["active_model_version"], **kw})
        return status

    def apply(self, package):
        self._report("RECEIVED")
        if self.pending is not None:
            return self._report("REJECTED", errors=["UPDATE_IN_PROGRESS"])
        res = validate_package(package, self.state)
        if not res["accepted"]:
            return self._report("REJECTED", errors=res["errors"])
        self._report("VERIFIED", warnings=res["warnings"])
        contract = json.loads(read_package(package)["model_contract.json"])
        other = "B" if self.active == "A" else "A"
        self.slots[other] = {"contract": contract, "state": device_state_from_contract(contract)}
        self._report("STAGED", slot=other)
        self.previous, self.active, self.pending = self.active, other, other
        return self._report("ACTIVE_PENDING_CONFIRMATION", slot=other)

    def confirm(self, self_check_passed):
        """After activation: the device's self-check result decides."""
        if self.pending is None:
            raise RuntimeError("no update pending")
        self.pending = None
        if self_check_passed:
            return self._report("CONFIRMED")
        self.active = self.previous
        return self._report("ROLLED_BACK")


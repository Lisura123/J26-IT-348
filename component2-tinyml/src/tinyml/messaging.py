"""Prediction message contract pv-msg-1 (Component 2 -> group backend).

The message goes to the backend stream `prediction` (MQTT topic
solar/{site_id}/prediction, measurement `fault_predictions`). The rules
below are taken from Member 3's backend on feature/mongodb-migration
(project-backend/schema.yaml and backend/validate.py, read-only); nothing
here changes that backend:

  - one flat JSON object; lists and nested objects are rejected
  - field names match ^[a-z][a-z0-9_]*$; at most 40 fields
  - required: record_id, site_id, timestamp_utc (UTC, ending in Z)
  - predicted_label, when present, is one of the five labels
  - a missing value is omitted (the backend drops null; it is never 0)

The firmware formatter (firmware/pv_fault/pv_message.c) writes the same
fields; tests compare both. Field meanings: docs/MESSAGING.md.
"""
import json
import math
import re

from .contract import LABELS

SCHEMA_VERSION = "pv-msg-1"
FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
UTC_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
MAX_FIELDS = 40
REQUIRED = ("record_id", "site_id", "timestamp_utc")
PROB_FIELDS = tuple(f"prob_{l.lower()}" for l in LABELS)

PREDICTION_STATUS = ("CLASSIFIED", "SKIPPED_LOW_LIGHT", "SKIPPED_LUX_SATURATED", "SKIPPED_INVALID_INPUT")
INPUT_STATUS = {"CLASSIFIED": "OK", "SKIPPED_LOW_LIGHT": "LOW_LIGHT", "SKIPPED_LUX_SATURATED": "LUX_SATURATED",
                "SKIPPED_INVALID_INPUT": "INVALID_INPUT"}
BACKENDS = ("FLOAT32", "INT8")
CONFIRMATION_STATUS = ("NO_FAULT", "ML_PREDICTED", "CONFIRMATION_PENDING", "CONFIRMED_BY_CONFIGURED_RULES",
                       "INSUFFICIENT_EVIDENCE")
EVIDENCE = ("USABLE", "SKIPPED_BY_QUALITY_GATE", "LOW_CONFIDENCE", "ELECTRICAL_CONTRADICTION", "STALE")

# fields Component 2 defines; anything else is refused by validate_message(strict=True)
FIELDS = (REQUIRED + ("device_id", "message_schema", "model_version", "feature_version", "inference_backend",
                      "quant_version", "prediction_status", "input_status", "decision_source", "predicted_label",
                      "confidence") + PROB_FIELDS +
          ("input_clips", "lut_clips", "sensor_invalid_mask", "confirmation_status", "confirmation_evidence",
           "confirmation_count", "candidate_label", "confirmed_label"))


def topic(site_id):
    if not ID_PATTERN.match(str(site_id)):
        raise ValueError(f"bad site_id {site_id!r}")
    return f"solar/{site_id}/prediction"


def build_message(*, record_id, site_id, timestamp_utc, model_version, feature_version, backend, prediction_status,
                  device_id=None, quant_version=None, label=None, probabilities=None, input_clips=None,
                  lut_clips=None, sensor_invalid_mask=None, confirmation=None):
    """Build a pv-msg-1 message. probabilities: five values in LABELS order
    (only for CLASSIFIED). confirmation: dict with status, evidence, count and
    optional candidate_label / confirmed_label. Values that do not exist are
    omitted, never filled."""
    if prediction_status not in PREDICTION_STATUS:
        raise ValueError(f"unknown prediction_status {prediction_status!r}")
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    classified = prediction_status == "CLASSIFIED"
    msg = {"record_id": record_id, "timestamp_utc": timestamp_utc, "site_id": site_id}
    if device_id is not None:
        msg["device_id"] = device_id
    msg.update({"message_schema": SCHEMA_VERSION, "model_version": model_version, "feature_version": feature_version,
                "inference_backend": backend})
    if backend == "INT8":
        if quant_version is None:
            raise ValueError("INT8 messages need quant_version")
        msg["quant_version"] = quant_version
    msg.update({"prediction_status": prediction_status, "input_status": INPUT_STATUS[prediction_status],
                "decision_source": "ML" if classified else "NONE"})
    if classified:
        if label not in LABELS or probabilities is None or len(probabilities) != len(LABELS):
            raise ValueError("a CLASSIFIED message needs a label and five probabilities")
        probs = [float(p) for p in probabilities]
        msg["predicted_label"] = label
        msg["confidence"] = max(probs)
        msg.update(dict(zip(PROB_FIELDS, probs)))
        if backend == "INT8":
            msg["input_clips"] = int(input_clips or 0)
            msg["lut_clips"] = int(lut_clips or 0)
    elif label is not None or probabilities is not None:
        raise ValueError("a skipped reading has no label or probabilities")
    if sensor_invalid_mask is not None:
        msg["sensor_invalid_mask"] = int(sensor_invalid_mask)
    if confirmation is not None:
        msg["confirmation_status"] = confirmation["status"]
        msg["confirmation_evidence"] = confirmation["evidence"]
        msg["confirmation_count"] = int(confirmation.get("count", 0))
        for key in ("candidate_label", "confirmed_label"):
            if confirmation.get(key) is not None:
                msg[key] = confirmation[key]
    validate_message(msg)
    return msg


def validate_message(msg, strict=True):
    """Raise ValueError if msg breaks the backend rules or pv-msg-1.
    strict=False checks only the backend rules (open stream)."""
    def need(cond, text):
        if not cond:
            raise ValueError(f"prediction message: {text}")
    need(isinstance(msg, dict), "must be a JSON object")
    for k in REQUIRED:
        need(msg.get(k) not in (None, ""), f"missing required field {k}")
    need(len(msg) <= MAX_FIELDS, f"{len(msg)} fields > {MAX_FIELDS}")
    for k, v in msg.items():
        need(FIELD_NAME.match(k) is not None, f"bad field name {k!r}")
        need(not isinstance(v, (dict, list, tuple)), f"field {k} is a complex type")
        need(v is not None, f"field {k} is null (omit it instead)")
        if isinstance(v, float):
            need(math.isfinite(v), f"field {k} is not finite")
    need(UTC_Z.match(str(msg["timestamp_utc"])) is not None, "timestamp_utc must be YYYY-MM-DDTHH:MM:SSZ")
    if "predicted_label" in msg:
        need(msg["predicted_label"] in LABELS, f"predicted_label {msg['predicted_label']!r}")
    if not strict:
        return True
    need(set(msg) <= set(FIELDS), f"unknown fields {sorted(set(msg) - set(FIELDS))}")
    for k in ("record_id", "site_id", "device_id"):
        if k in msg:
            need(ID_PATTERN.match(str(msg[k])) is not None, f"bad {k} {msg[k]!r}")
    need(msg.get("message_schema") == SCHEMA_VERSION, "message_schema")
    need(msg.get("inference_backend") in BACKENDS, "inference_backend")
    need(msg.get("prediction_status") in PREDICTION_STATUS, "prediction_status")
    need(msg.get("input_status") == INPUT_STATUS[msg["prediction_status"]], "input_status")
    classified = msg["prediction_status"] == "CLASSIFIED"
    need(msg.get("decision_source") == ("ML" if classified else "NONE"), "decision_source")
    need(("quant_version" in msg) == (msg["inference_backend"] == "INT8"), "quant_version only for INT8")
    if classified:
        need("predicted_label" in msg and all(f in msg for f in PROB_FIELDS) and "confidence" in msg,
             "CLASSIFIED needs predicted_label, confidence and five probabilities")
        probs = [float(msg[f]) for f in PROB_FIELDS]
        need(all(0.0 <= p <= 1.0 for p in probs) and abs(sum(probs) - 1.0) < 1e-4, "probabilities")
        need(abs(float(msg["confidence"]) - max(probs)) < 1e-6, "confidence must be the max probability")
        need(LABELS[probs.index(max(probs))] == msg["predicted_label"] or msg["inference_backend"] == "INT8",
             "Float32 label must be the argmax of the probabilities")
    else:
        need(not any(f in msg for f in ("predicted_label", "confidence", "input_clips", "lut_clips") + PROB_FIELDS),
             "a skipped reading carries no label, confidence or probabilities")
    if "confirmation_status" in msg:
        need(msg["confirmation_status"] in CONFIRMATION_STATUS, "confirmation_status")
        need(msg.get("confirmation_evidence") in EVIDENCE, "confirmation_evidence")
        for k in ("candidate_label", "confirmed_label"):
            if k in msg:
                need(msg[k] in LABELS, k)
        if msg["confirmation_status"] == "CONFIRMED_BY_CONFIGURED_RULES":
            need("confirmed_label" in msg, "a confirmed status needs confirmed_label")
    return True


def parse_device_lines(text, prefix="PVMSG "):
    """JSON messages from a device or host log (lines starting with prefix)."""
    return [json.loads(line[len(prefix):]) for line in text.splitlines() if line.startswith(prefix)]

"""Prediction message contract pv-msg-1 against the backend rules
(project-backend/schema.yaml and backend/validate.py on Member 3's branch,
mirrored in messaging.validate_message) and the committed sample messages."""
import json
from pathlib import Path

import pytest

from tinyml import messaging
from tinyml.hostc import compile_and_run_runtime, compiler

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "examples"
needs_cc = pytest.mark.skipif(compiler() is None, reason="no C compiler")

BASE = dict(record_id="REC-1", site_id="SITE-A", timestamp_utc="2026-09-10T00:26:40Z", model_version="c2-local-v1",
            feature_version="pv8-v1")
PROBS = [0.01, 0.02, 0.03, 0.04, 0.90]


def classified(**kw):
    args = dict(BASE, backend="INT8", quant_version="int8-v1", prediction_status="CLASSIFIED",
                label="SHORT_CIRCUIT", probabilities=PROBS, input_clips=0, lut_clips=0)
    args.update(kw)
    return messaging.build_message(**args)


def test_classified_message_fields():
    m = classified(device_id="ESP32-A01", confirmation={"status": "CONFIRMATION_PENDING", "evidence": "USABLE",
                                                       "count": 3, "candidate_label": "SHORT_CIRCUIT"})
    assert m["predicted_label"] == "SHORT_CIRCUIT" and m["confidence"] == 0.90 and m["decision_source"] == "ML"
    assert m["prob_short_circuit"] == 0.90 and m["input_status"] == "OK" and len(m) <= messaging.MAX_FIELDS
    assert messaging.topic("SITE-A") == "solar/SITE-A/prediction"


@pytest.mark.parametrize("status", ["SKIPPED_LOW_LIGHT", "SKIPPED_LUX_SATURATED", "SKIPPED_INVALID_INPUT"])
def test_skipped_reading_has_no_label_and_is_never_normal(status):
    m = messaging.build_message(**BASE, backend="FLOAT32", prediction_status=status)
    assert "predicted_label" not in m and "confidence" not in m and m["decision_source"] == "NONE"
    assert not any(f in m for f in messaging.PROB_FIELDS)
    with pytest.raises(ValueError):
        messaging.build_message(**BASE, backend="FLOAT32", prediction_status=status, label="NORMAL")


@pytest.mark.parametrize("mutate, error", [
    (lambda m: m.update(extra={"a": 1}), "complex type"),            # backend: complex types rejected
    (lambda m: m.update(Bad_Name=1), "bad field name"),               # backend: ^[a-z][a-z0-9_]*$
    (lambda m: m.update(timestamp_utc="2026-09-10T00:26:40+05:30"), "timestamp_utc"),   # non-UTC rejected
    (lambda m: m.update(timestamp_utc="2026-09-10T00:26:40"), "timestamp_utc"),         # naive rejected
    (lambda m: m.pop("record_id"), "record_id"),                      # required
    (lambda m: m.update(predicted_label="SOILING"), "predicted_label"),   # allowed_values
    (lambda m: m.update(confidence=None), "null"),
    (lambda m: m.update(confidence=float("nan")), "not finite"),
    (lambda m: m.update({f"f{i}": 1.0 for i in range(40)}), "fields"),     # max 40 fields
])
def test_backend_rules_are_enforced(mutate, error):
    m = classified()
    mutate(m)
    with pytest.raises(ValueError, match=error):
        messaging.validate_message(m, strict=False)


@pytest.mark.parametrize("mutate", [
    lambda m: m.update(prob_dust=0.5),                     # probabilities must sum to 1
    lambda m: m.update(confidence=0.5),                    # confidence = max probability
    lambda m: m.update(input_status="LOW_LIGHT"),          # consistent with prediction_status
    lambda m: m.pop("quant_version"),                      # INT8 messages name the quantisation
    lambda m: m.update(record_id="bad id"),                # ID characters
    lambda m: m.update(confirmation_status="CONFIRMED_BY_CONFIGURED_RULES",
                       confirmation_evidence="USABLE"),    # confirmed needs confirmed_label
    lambda m: m.update(unknown_field=1),                   # pv-msg-1 field list
])
def test_pv_msg_rules_are_enforced(mutate):
    m = classified()
    mutate(m)
    with pytest.raises(ValueError):
        messaging.validate_message(m)


def test_topic_refuses_injection():
    for bad in ("SITE/A", "SITE-A/#", "", "x" * 65):
        with pytest.raises(ValueError):
            messaging.topic(bad)


def test_committed_sample_messages_are_valid():
    files = sorted(EXAMPLES.glob("prediction_*.json"))
    assert len(files) >= 4
    statuses = set()
    for f in files:
        m = json.loads(f.read_text(encoding="utf-8"))
        messaging.validate_message(m)
        statuses.add(m["prediction_status"])
        assert m["site_id"].startswith("SITE-") and "SYNTHETIC" in m["record_id"]
    assert {"CLASSIFIED", "SKIPPED_INVALID_INPUT"} <= statuses


# ---------------------------------------------------------------- device formatter (firmware pv_message.c)

@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    if compiler() is None:
        pytest.skip("no C compiler")
    return compile_and_run_runtime(tmp_path_factory.mktemp("rt"))


@needs_cc
def test_c_messages_follow_the_backend_rules(runtime):
    msgs = messaging.parse_device_lines(runtime["output"])
    assert len(msgs) == 4
    for m in msgs:
        messaging.validate_message(m)
    by_id = {m["record_id"]: m for m in msgs}
    assert by_id["REC-T1"]["inference_backend"] == "INT8" and by_id["REC-T1"]["quant_version"] == "int8-v1"
    assert "device_id" not in by_id["REC-T2"] and "quant_version" not in by_id["REC-T2"]
    skipped = by_id["REC-T3"]
    assert skipped["prediction_status"] == "SKIPPED_LOW_LIGHT" and "predicted_label" not in skipped
    p2 = by_id["REC-P2"]
    assert p2["prediction_status"] == "SKIPPED_INVALID_INPUT" and p2["sensor_invalid_mask"] == 16
    assert p2["confirmation_status"] == "INSUFFICIENT_EVIDENCE"


@needs_cc
def test_c_message_matches_python_builder(runtime):
    m = messaging.parse_device_lines(runtime["output"])[0]
    py = messaging.build_message(
        record_id=m["record_id"], site_id=m["site_id"], device_id=m["device_id"], timestamp_utc=m["timestamp_utc"],
        model_version="c2-local-v1", feature_version="pv8-v1", backend="INT8", quant_version="int8-v1",
        prediction_status="CLASSIFIED", label=m["predicted_label"],
        probabilities=[m[f] for f in messaging.PROB_FIELDS], input_clips=0, lut_clips=0,
        confirmation={"status": "NO_FAULT", "evidence": "USABLE", "count": 0})
    assert list(py) == list(m)                                  # same fields, same order
    for k in py:
        assert py[k] == pytest.approx(m[k]) if isinstance(py[k], float) else py[k] == m[k]

# Prediction Message Contract `pv-msg-1`

**Stream:** backend stream `prediction` → measurement `fault_predictions`, MQTT topic
`solar/{site_id}/prediction`. **Source of the rules:** Member 3's backend on
`origin/feature/mongodb-migration` (`project-backend/schema.yaml`, `backend/validate.py`,
`backend/ingest/mqtt_service.py`), read-only. The backend marks this stream `provisional`,
`mode: open` and asks Member 2 to confirm the field names; this document is that answer.
No backend file is changed.

Implementations: `firmware/pv_fault/pv_message.c` (device) and `src/tinyml/messaging.py`
(reference builder and validator). Tests compare both, and run the committed samples
through Member 3's actual `validate_record()` (`tests/test_member3_compat.py`).

## 1. Backend rules followed

- One flat JSON object; nested objects and lists are rejected.
- Field names `^[a-z][a-z0-9_]*$`, at most 40 fields.
- Required: `record_id`, `site_id`, `timestamp_utc` (`YYYY-MM-DDTHH:MM:SSZ`, UTC only).
- `site_id` must exist in the backend's `sites.csv`; `device_id`, if sent, must match it.
- `predicted_label` ∈ {NORMAL, PARTIAL_SHADING, DUST, OPEN_CIRCUIT, SHORT_CIRCUIT}.
- Missing values are **omitted** (the backend drops null; a value is never sent as 0).

## 2. Fields

| Field | Type | When | Meaning |
|---|---|---|---|
| `record_id` | string | always | join key with the sensor reading (`[A-Za-z0-9_.:-]`, ≤ 64) |
| `timestamp_utc` | string | always | reading time, UTC |
| `site_id` | string | always | |
| `device_id` | string | optional | |
| `message_schema` | `"pv-msg-1"` | always | |
| `model_version` | string | always | e.g. `c2-local-v1` |
| `feature_version` | string | always | `pv8-v1` |
| `inference_backend` | `FLOAT32` / `INT8` | always | |
| `quant_version` | string | INT8 only | `int8-v1` |
| `prediction_status` | `CLASSIFIED` / `SKIPPED_LOW_LIGHT` / `SKIPPED_LUX_SATURATED` / `SKIPPED_INVALID_INPUT` | always | whether a classification exists |
| `input_status` | `OK` / `LOW_LIGHT` / `LUX_SATURATED` / `INVALID_INPUT` | always | quality-gate reason |
| `decision_source` | `ML` / `NONE` | always | `NONE` when skipped |
| `predicted_label` | label | CLASSIFIED only | never sent for a skipped reading (not NORMAL by default) |
| `confidence` | float 0–1 | CLASSIFIED only | max probability |
| `prob_normal`, `prob_partial_shading`, `prob_dust`, `prob_open_circuit`, `prob_short_circuit` | float | CLASSIFIED only | softmax, LABELS order |
| `input_clips`, `lut_clips` | int | INT8, CLASSIFIED | INT8 clipping diagnostics |
| `sensor_invalid_mask` | int | when the sensor adapter is used | bit *i* set: raw input *i* unavailable (0 V, 1 mA, 2 mW, 3 T1, 4 T2, 5 Tamb, 6 lux, 7 dust) |
| `confirmation_status` | see `FAULT_CONFIRMATION_AND_RELAY.md` | with confirmation | diagnostic, not electrical confirmation |
| `confirmation_evidence` | `USABLE` / `SKIPPED_BY_QUALITY_GATE` / `LOW_CONFIDENCE` / `ELECTRICAL_CONTRADICTION` / `STALE` | with confirmation | |
| `confirmation_count` | int | with confirmation | consecutive matching readings |
| `candidate_label`, `confirmed_label` | label | when present | |

Not sent: `inference_latency_ms` (proposed in `ARCHITECTURE.md` §7.3; it needs a measured
board value), `alert_level` and `relay_recommendation` (no validated rule exists; physical
actuation is disabled).

## 3. Samples (`docs/examples/`, record IDs marked SYNTHETIC)

- `prediction_classified_int8.json`: INT8 result; probabilities from the host run of a
  golden vector, not from a device reading.
- `prediction_skipped_missing_panel_temp2.json`: the stated hardware without a second
  panel probe (`sensor_invalid_mask` 16 = `panel_temp2_c`).
- `prediction_skipped_low_light.json`.
- `prediction_confirmed_short_circuit.json`: illustrative values for the confirmed state.

## 4. Requests to Member 3 (backend owner)

1. Accept `pv-msg-1` as the `prediction` field list: add the string fields above to
   `string_fields` and `allowed_values` (`prediction_status`, `input_status`,
   `decision_source`, `inference_backend`, `confirmation_status`, `confirmation_evidence`,
   `candidate_label`, `confirmed_label`), and the float fields to `float_fields`. In open
   mode they are already accepted; the change makes them typed and checked.
2. Keep `predicted_label` optional: a skipped reading has none.
3. `sensor_invalid_mask`, `input_clips`, `lut_clips` and `confirmation_count` are integers;
   the open stream currently stores them as floats.
4. Confirm the MQTT client identity and QoS 1 for devices, and who publishes
   (device directly or a gateway, relevant for LoRa sites).

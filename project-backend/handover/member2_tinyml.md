# Handover: Member 2 (TinyML & Edge AI)

## 1. Overview & MQTT Topic
TinyML model predictions emitted by edge devices are ingested into the `fault_predictions` stream.
- MQTT Topic: `solar/{site_id}/prediction`
- Ingest HTTP Endpoint: `POST /api/v1/ingest/prediction` (Header `X-API-Key: <ASK_MEMBER_3_PRIVATELY>`)
- Schema Mode: `open` (status: provisional)

---

## 2. Ground-Truth Dataset Notice
> **IMPORTANT ARCHITECTURAL RULE:**
> The `label` column in `pv_fault_dataset.csv` is **offline ground truth**, NOT a model inference prediction.
> The backend validator and replay tools explicitly **refuse** `pv_fault_dataset.csv` for the prediction stream.
> Predictions must originate from live model inferences or dedicated validation runs, so nothing is invented or conflated.

---

## 3. Proposed Schema & Allowed Classes

### Allowed Values for `predicted_label`:
The backend schema strictly requires `predicted_label` to be one of the following 5 canonical strings:
1. `NORMAL`
2. `PARTIAL_SHADING`
3. `DUST`
4. `OPEN_CIRCUIT`
5. `SHORT_CIRCUIT`

Any prediction using alternate capitalization or undeclared labels will be rejected (`BAD_VALUE`).

### Proposed Fields Table
| Field Name | Status | Type | Range / Format | Description |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp_utc` | Verified | string | ISO 8601 UTC with `Z` | Timestamp when inference was executed. |
| `site_id` | Verified | string | `SITE-A` .. `SITE-F` | Target site ID. |
| `device_id` | Verified | string | `ESP32-A01` .. `ESP32-F01` | Edge device running the model. |
| `record_id` | Verified | string | Identifier | Unique prediction record ID. |
| `predicted_label` | **PROPOSED** | string | 5 allowed values above | Classification output. |
| `confidence` | **PROPOSED** | float | `0.0` .. `1.0` | Model confidence probability. |
| `prob_normal` | **PROPOSED** | float | `0.0` .. `1.0` | Softmax probability for NORMAL. |
| `prob_partial_shading`| **PROPOSED** | float | `0.0` .. `1.0` | Softmax probability for PARTIAL_SHADING. |
| `prob_dust` | **PROPOSED** | float | `0.0` .. `1.0` | Softmax probability for DUST. |
| `prob_open_circuit` | **PROPOSED** | float | `0.0` .. `1.0` | Softmax probability for OPEN_CIRCUIT. |
| `prob_short_circuit` | **PROPOSED** | float | `0.0` .. `1.0` | Softmax probability for SHORT_CIRCUIT. |
| `inference_latency_ms`| **PROPOSED** | float | `> 0.0` | Optional on-device execution time in ms. |

---

## 4. Proposed Payload Example
```json
{
  "record_id": "PRED-ESP32-0042",
  "timestamp_utc": "2026-09-10T08:15:00Z",
  "site_id": "SITE-A",
  "device_id": "ESP32-A01",
  "predicted_label": "PARTIAL_SHADING",
  "confidence": 0.945,
  "prob_normal": 0.021,
  "prob_partial_shading": 0.945,
  "prob_dust": 0.018,
  "prob_open_circuit": 0.008,
  "prob_short_circuit": 0.008,
  "inference_latency_ms": 34.2
}
```

---

## 5. Action Items for Member 2
Before moving this stream from provisional to verified, please confirm:
1. **Probability Distribution**: Will edge models emit full probability distributions (all 5 `prob_*` fields) or only `confidence` and `predicted_label` to conserve bandwidth?
2. **Inference Latency**: Should edge execution latency (`inference_latency_ms`) or memory footprint be recorded?
3. **Field Naming Finalization**: Once field names are approved, they will be locked into `schema.yaml` strict definitions.

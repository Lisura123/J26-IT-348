# Group J26 IT 348 – shared dummy data contract

All data is SYNTHETIC. Use it for development and integration, not for final results.

## Rules every member follows
- **Time:** stored and exchanged in **UTC only**, ISO 8601 with `Z`, e.g. `2026-09-10T03:15:00Z`.
  Every time column ends in `_utc`. Convert to `Asia/Colombo` (UTC+05:30) only when displaying.
- **IDs:** only the IDs in `shared/sites.csv` (SITE-A…F, ESP32-A01…F01, PV-A01…F01).
- **Join keys:** `record_id` (one sensor reading), `site_id`, `device_id`, `timestamp_utc`,
  `round_id` / `update_id` (federated learning), `global_model_version`.
- **InfluxDB:** `site_id`, `device_id`, `protocol` as tags; readings as fields; InfluxDB time = `timestamp_utc`.

## Timeline (UTC)
| Window | Dates | Used for |
|---|---|---|
| TRAINING | 1–9 Sept 2026 | labelled data each site trains its local model on |
| OPERATION | 10–15 Sept 2026 | live readings: Member 1 → 2 → 4, while FL rounds run |
| FL rounds | every 6 h from 10 Sept 00:00, deadline 5 h after opening | Member 3 |

Readings occur only in daylight (about 06:30–17:30 Sri Lanka time).

## Files by member
| Member | Input files | Output files (consumed by others) |
|---|---|---|
| 1 Health grading | `member1_raw_sensor_stream.csv` | `member1_sensor_health.csv` (reference clean values, health score, quality flags) |
| 2 TinyML | `pv_fault_dataset.csv`, `pv_<label>.csv` | `member2_fault_predictions.csv` |
| 3 Federated learning | `network_link_metrics.csv`, `network_outages.csv`, `fl_client_updates.csv`, `fl_update_transmissions.csv`, `fl_weight_updates.jsonl` | `fl_global_models.csv`, `member3_backend_status.csv`, `fl_global_model_*.json` |
| 4 Digital twin | `digital_twin_input.csv` (join of 1 + 2 + 3) | `digital_twin_output.csv` |

Where a file stands in for another member's output (health score, predictions, FL updates),
replace it with that member's real output as soon as it exists.

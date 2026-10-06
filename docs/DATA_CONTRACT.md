# J26-IT-348 Data Contract

This is the dummy data we use in our group. We use it to build and test
our components until the IoT device is ready. We will not use it for
final results.

## Rules for all members

### Time
- Save all time values in UTC.
- Use this format: `2026-09-10T03:15:00Z`
- All time column names end with `_utc`.
- Change to Sri Lanka time (UTC+05:30) only when showing it on a screen.

### IDs
- Use only the IDs in `sites.csv`.
- Sites: SITE-A to SITE-F
- Devices: ESP32-A01 to ESP32-F01
- Panels: PV-A01 to PV-F01

### Join keys
We use these columns to join data between members:
- `record_id` (one sensor reading)
- `site_id`
- `device_id`
- `timestamp_utc`
- `round_id` / `update_id` (for federated learning)
- `global_model_version`

### InfluxDB
- Tags: `site_id`, `device_id`, `protocol`
- Fields: sensor readings
- Time: `timestamp_utc`

## Timeline (UTC)

| Period | Dates | What we use it for |
|---|---|---|
| Long-term | 16 Mar – 15 Sept 2026 (every 5 min, daytime) | Drift checks (EWMA/CUSUM), thermal model, calibration |
| Training | 1 – 9 Sept 2026 | Labelled data for local model training |
| Operation | 10 – 15 Sept 2026 | Live readings. Data goes Member 1 → 2 → 4 |
| FL rounds | Every 6 hours from 10 Sept 00:00 | Member 3. Each round closes after 5 hours |

Readings are only in daytime (about 6:30 AM – 5:30 PM Sri Lanka time).

## How the data was made
- Panel model: De Soto single-diode model
- Panel datasheet: assumed values (`panel_datasheet.csv`).
  We will change this when we get the real datasheet.
- Temperature model: Faiman thermal model
- Irradiance: lux / 120. When the lux sensor is full (saturated),
  we use GHI from the weather API.
- Wind: from the weather API. We do not have a wind sensor.

## Files for each member

| Member | Input | Output |
|---|---|---|
| 1 – Health grading | `member1_raw_sensor_stream.csv` | `member1_sensor_health.csv` (clean values, health score, quality flags) |
| 2 – TinyML | `pv_fault_dataset.csv`, `pv_<label>.csv` | `member2_fault_predictions.csv` |
| 3 – Federated learning | `network_link_metrics.csv`, `network_outages.csv`, `fl_client_updates.csv`, `fl_update_transmissions.csv`, `fl_weight_updates.jsonl` | `fl_global_models.csv`, `member3_backend_status.csv`, `fl_global_model_*.json` |
| 4 – Digital twin | `digital_twin_input.csv` (Members 1 + 2 + 3 joined), `member4_longterm_5min.csv`, `weather_hourly.csv`, `panel_datasheet.csv`, `member2_shap_values.csv`, `maintenance_log.csv` | `digital_twin_output.csv`, `twin_calibration.json` |

## Note
Some files are only stand-ins for another member's output
(health score, predictions, FL updates). When that member's real
output is ready, we will replace the stand-in file with it.
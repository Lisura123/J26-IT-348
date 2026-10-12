# Shared InfluxDB (group J26-IT-348)

One database for all four components. Owner of the setup: Member 3.

## Connection

| Setting | Value |
|---|---|
| URL | `http://localhost:8086` (or `http://<host-ip>:8086` if one computer hosts it) |
| Organisation | `j26` |
| Bucket | `solar_pv` |
| Token | ask Member 3 privately. Never commit it. Keep it in your own `.env` |

## Start it on your own computer

```powershell
cd component3-federated-backend
copy .env.example .env          
docker compose up -d
```
Open http://localhost:8087 and log in. The bucket `solar_pv` is created automatically.
**Never run `docker compose down -v`**: the `-v` deletes all stored data. Use `docker compose down` or `stop`.

To fill it with the shared files and check the counts (Member 3's loader):
```powershell
python -m src.load_to_influx
python -m src.verify_influx
```

## Rules (from the data contract)

- Times are UTC. The InfluxDB time of a point is the row's UTC time (`timestamp_utc`).
- Tags: `site_id`, `device_id`, `protocol`. Keep tags to a few labels with few values. Everything else is a field.
- IDs come only from `sites.csv` (SITE-A to SITE-F, ESP32-A01 to F01).
- A missing reading is left out. Never write it as zero: a zero current looks like an open-circuit fault.
- Do not store model weights in InfluxDB. Store the file path only.
- Keep a field's type fixed. Writing a float into a field that already holds integers is rejected.

## Measurements

| Measurement | Owner | Time | Tags | Main fields | Status |
|---|---|---|---|---|---|
| `network_link_metrics` | 3 | `timestamp_utc` | site_id, device_id, protocol, gateway_id, link_quality | online, rssi_dbm, snr_db, lora_sf, latency_ms, packet_loss_pct, throughput_kbps | loaded |
| `network_outages` | 3 | `start_utc` | site_id, device_id, protocol, cause | duration_min, end_utc | loaded |
| `fl_client_updates` | 3 | send time | site_id, device_id, protocol, status (+ compression_method, run_id for runs) | round_id, payload_bytes, retries, airtime_s, delta_l2_norm, staleness, reliability_score, aggregation_weight, ... | loaded |
| `fl_global_models` | 3 | `published_at_utc` | aggregation (+ compression_method, run_id for runs) | round_id, global_model_version, clients_received, global_val_accuracy, global_val_loss, model_path, ... | loaded |
| `fl_site_state` | 3 | round opening | site_id, device_id, protocol, run_id | reliability, expected_upload_s, age, participation, selected | filled by runs |
| `ota_deployments` | 3 | `deployed_at_utc` | site_id, device_id, protocol, ota_status, ota_method | global_model_version, ota_bytes, ota_retries, ota_duration_s | loaded |
| `sensor_readings` | 1 (proposed) | `timestamp_utc` | site_id, device_id, protocol | voltage_v, current_ma, power_mw, panel_temp1_c, panel_temp2_c, ambient_temp_c, humidity_pct, pressure_hpa, irradiance_lux, dust_vo_v | to agree |
| `sensor_health` | 1 (proposed) | `timestamp_utc` | site_id, device_id, protocol | health_score, quality flags (columns of `member1_sensor_health.csv`) | to agree |
| `fault_predictions` | 2 (proposed) | `timestamp_utc` | site_id, device_id, protocol | predicted label, confidence (columns of `member2_fault_predictions.csv`) | to agree |
| `twin_output` | 4 (proposed) | `timestamp_utc` | site_id, device_id, protocol | columns of `digital_twin_output.csv` | to agree |

Member 3's FL runs carry an extra `run_id` tag (for example `member3-full-tanh-s0-trace`) so runs stay separate from
the shared files. Filter on it, or ignore it.

## Quick checks

List every measurement in the bucket (Data Explorer, Script Editor):
```
import "influxdata/influxdb/schema"
schema.measurements(bucket: "solar_pv")
```
Count one measurement:
```
from(bucket: "solar_pv")
  |> range(start: 2026-09-01T00:00:00Z, stop: 2026-09-20T00:00:00Z)
  |> filter(fn: (r) => r._measurement == "sensor_readings")
  |> group() |> count()
```

## Writing from Python
```python
import os
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

client = InfluxDBClient(url=os.environ["INFLUX_URL"], token=os.environ["INFLUX_TOKEN"], org="j26")
write = client.write_api(write_options=SYNCHRONOUS)
p = (Point("sensor_readings").tag("site_id", "SITE-A").tag("device_id", "ESP32-A01").tag("protocol", "WIFI")
     .field("voltage_v", 2.27).field("current_ma", 70.4)
     .time("2026-09-10T07:26:00Z", WritePrecision.S))
write.write(bucket="solar_pv", record=p)
```

## Reading from the dashboard (JavaScript)
Use the `@influxdata/influxdb-client` package with the same URL, org, bucket and a **read-only token**.
Member 3 creates one per reader: InfluxDB UI, Load Data, API Tokens, Generate Custom Token, read access to `solar_pv`.

## Sharing across computers
Simplest: everyone runs the database locally and loads the shared files with the loader scripts.
For an integration demo, one computer can host it: others use `http://<host-ip>:8086`, and the host must allow
port 8086 in the Windows firewall.
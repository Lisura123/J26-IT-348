# Shared Database (Group J26-IT-348)

One shared database for all four components, hosted on **MongoDB Atlas** (cloud database).
Database name: `solar_pv`.
Owner of the database setup and Atlas administration: Member 3.

InfluxDB and Docker are no longer used (the git tag pp1 has the old version).

---

## 1. Connection Settings

Keep credentials in your local `.env` file (never commit `.env`). Use placeholders when sharing configs:

| Setting | Variable | Value / Description |
| :--- | :--- | :--- |
| **Connection URI** | `MONGO_URI` | `mongodb+srv://<USER>:<PASSWORD>@<CLUSTER>.mongodb.net/solar_pv?retryWrites=true&w=majority` |
| **Database Name** | `MONGO_DB` | `solar_pv` |
| **Writer Account** | `MONGO_APP_USER` | Application ingest user (assigned `readWrite` on `solar_pv`) |
| **Reader Account** | `MONGO_READER_USER` | Dashboard read-only user (assigned `read` on `solar_pv`) |

Credentials and connection strings are managed by Member 3. Contact Member 3 privately for credentials.

---

## 2. MongoDB Atlas Setup & Access Control

### Cluster Tier
- Hosted on MongoDB Atlas **M0 Sandbox** (Free Tier).
- Database name: `solar_pv`.
- Cloud region: Singapore (`ap-southeast-1`) or Mumbai (`ap-south-1`).

### Role-Based Access Control (Least Privilege)
To prevent accidental data loss or unauthorized modifications, two separate database accounts are provisioned:
1. **Writer Account (`MONGO_APP_USER`)**:
   - Role: `readWrite` scoped strictly to the `solar_pv` database.
   - Used by `project-backend` ingestion services (MQTT and HTTP API) and Member 3's pipeline scripts.
2. **Reader Account (`MONGO_READER_USER`)**:
   - Role: `read` scoped strictly to the `solar_pv` database.
   - Used by `project-backend` dashboard read endpoints (`/api/v1/query/*`). It has zero write, update, or drop permissions.

### IP Access List Per Member
Network access in MongoDB Atlas is protected by an explicit IP allow-list (**Security** -> **Network Access**):
- **Why NOT `0.0.0.0/0`**: Allowing access from anywhere exposes the cluster port to automated Internet scanners, credential-stuffing attacks, and botnets.
- **Member Access Entries**: Each group member adds their current public IP address (CIDR `/32`) via **Add Current IP Address** in the Atlas Console.
- **Temporary Entries**: For mobile or dynamic ISP connections, entries can be configured with an automatic expiration time (e.g. 6 hours). If your IP changes, check your current IP with `Invoke-RestMethod https://api.ipify.org` and update Atlas.

---

## 3. Atlas Free Tier Limits

Keep these M0 limits in mind during development and evaluation:
- **Operation Rate**: Throttled at approximately **100 operations per second**. Loaders and replay scripts must use batch writes (e.g. 100 documents per `bulk_write` chunk) with pacing to avoid rate-limit timeouts.
- **Cluster Quota**: Exactly **one free M0 cluster per Atlas project**.
- **Storage Limit**: **512 MB** total storage across data and indexes.
- **Inactivity Pausing**: Atlas pauses free-tier clusters if no connections occur for 60 consecutive days. Clusters can be resumed from the Atlas console.
- **Internet Requirement**: Atlas is a cloud-hosted database; an active internet connection and access to port `27017` are required.

---

## 4. Collections Catalog

### A. Real-Time Telemetry & Processing Streams (`project-backend`)

During development and testing, streams use the prefix specified by `MEASUREMENT_PREFIX` (e.g. `test_sensor_readings` or `demo_sensor_readings`). In production, prefixes are cleared to write to the base collections:

| Collection | Owner | Document Time | Indexed Fields | Key Payload Fields | Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `sensor_readings` | Member 1 | `timestamp_utc` | `site_id`, `device_id`, `protocol` | `voltage_v`, `current_ma`, `power_mw`, `panel_temp1_c`, `panel_temp2_c`, `ambient_temp_c`, `humidity_pct`, `pressure_hpa`, `irradiance_lux`, `dust_vo_v`, `record_id` | Raw sensor telemetry |
| `sensor_health` | Member 1 | `timestamp_utc` | `site_id`, `device_id`, `protocol` | `health_score`, `quality_flags`, `v_clean_v`, `i_clean_ma`, `p_clean_mw` | Sensor quality & health |
| `fault_predictions` | Member 2 | `timestamp_utc` | `site_id`, `device_id`, `protocol` | `predicted_label`, `confidence`, `prob_normal`, `prob_dust`, `prob_partial_shading`, ... | TinyML on-device fault classification |
| `twin_output` | Member 4 | `timestamp_utc` | `site_id`, `device_id`, `protocol` | Model outputs, degradation metrics, expected yield | Digital twin simulations |

### B. Federated Learning & Network Ops Collections (Member 3)

Member 3 collections are isolated by `run_id` (e.g. `shared` for canonical baseline datasets, or `member3-full-tanh-s11-trace` for evaluation runs). Unique compound indexes guarantee idempotency:

| Collection | Owner | Unique Compound Key | Document Time | Key Fields |
| :--- | :---: | :--- | :--- | :--- |
| `network_link_metrics` | Member 3 | `(run_id, site_id, timestamp_utc)` | `timestamp_utc` | `rssi_dbm`, `snr_db`, `latency_ms`, `packet_loss_pct`, `throughput_kbps` |
| `network_outages` | Member 3 | `(run_id, site_id, start_utc)` | `start_utc` | `end_utc`, `duration_min`, `cause` |
| `fl_client_updates` | Member 3 | `(run_id, update_id)` | `sent_at_utc` | `round_id`, `site_id`, `device_id`, `status`, `payload_bytes`, `airtime_s`, `delta_l2_norm`, `aggregation_weight` |
| `fl_global_models` | Member 3 | `(run_id, round_id)` | `published_at_utc` | `global_model_version`, `global_val_accuracy`, `global_val_loss`, `clients_received`, `model_path` |
| `fl_site_state` | Member 3 | `(run_id, round_id, site_id)` | `round_opened_utc` | `reliability`, `expected_upload_s`, `age`, `participation`, `selected` |
| `ota_deployments` | Member 3 | `(run_id, site_id, round_id)` | `deployed_at_utc` | `global_model_version`, `ota_status`, `ota_method`, `ota_bytes`, `ota_duration_s` |

---

## 5. Rules from the Shared Data Contract

1. **UTC Time Representation**: Devices send ISO 8601 text ending in `Z` (e.g. `2026-09-10T07:26:00Z`). In MongoDB Atlas, it is stored as a BSON UTC date (`datetime`). Display interfaces convert to `Asia/Colombo` (UTC+05:30) only at display time.
2. **Indexed Fields**: Standard indexed fields are `site_id`, `device_id`, and `protocol` on every document.
3. **Canonical Identifiers**: Use only valid IDs defined in `reference/sites.csv` (`SITE-A` .. `SITE-F`, `ESP32-A01` .. `ESP32-F01`, `PV-A01` .. `PV-F01`).
4. **Missing Values**: Missing or unavailable sensor readings must be omitted entirely from the document. Never write `0.0` or `null` for a missing sensor reading (a zero voltage looks like an open-circuit fault).
5. **Model Weights Separation**: Large binary or array model weights are never stored in database documents. Weights remain as local `.json` files in `models/`; only the relative file path (e.g. `models/runs/.../global_v1.json`) is stored in the `model_path` field.
6. **Consistent Field Types**: Keep field types strictly consistent (floats stay float, integers stay integer, strings stay string).

---

## 6. How to Inspect Data

### Using MongoDB Compass (GUI)
1. Launch MongoDB Compass.
2. Paste your connection URI:
   ```
   mongodb+srv://<USER>:<PASSWORD>@<CLUSTER>.mongodb.net/solar_pv
   ```
3. Click **Connect**.
4. Select the `solar_pv` database from the left sidebar.
5. Example queries in the Compass filter bar:
   - Inspect FL global models for a specific run:
     ```json
     {"run_id": "member3-full-tanh-s11-trace"}
     ```
     Sort by: `{"round_id": 1}`
   - Inspect telemetry for SITE-A on a specific day:
     ```json
     {
       "site_id": "SITE-A",
       "timestamp_utc": {
         "$gte": {"$date": "2026-09-10T00:00:00Z"},
         "$lt": {"$date": "2026-09-11T00:00:00Z"}
       }
     }
     ```

### Using `mongosh` (Command-Line Shell)
Connect via the shell:
```bash
mongosh "mongodb+srv://<CLUSTER>.mongodb.net/solar_pv" --username <USER>
```
Run basic count and inspection commands:
```javascript
// Switch to the database
use solar_pv

// Count records in a collection
db.fl_global_models.countDocuments({"run_id": "member3-full-tanh-s11-trace"})

// Find newest telemetry point
db.test_sensor_readings.find({"site_id": "SITE-A"}).sort({"timestamp_utc": -1}).limit(1)
```

---

## 7. Reading and Writing Data from Code

### Writing from Python (`pymongo`)
Always use chunked `bulk_write` with `ReplaceOne(upsert=True)` to prevent duplicates:
```python
from datetime import datetime, timezone
import os
from pymongo import MongoClient, ReplaceOne

uri = os.environ["MONGO_URI"]
client = MongoClient(
    uri,
    serverSelectionTimeoutMS=10000,
    connectTimeoutMS=10000,
    socketTimeoutMS=60000,
    retryWrites=True,
)
db = client["solar_pv"]
col = db["sensor_readings"]

# Ensure unique compound index
col.create_index([("site_id", 1), ("timestamp_utc", 1)], unique=True)

# Document: text ending in Z is parsed into a timezone-aware UTC datetime for BSON
doc = {
    "record_id": "REC-01234",
    "site_id": "SITE-A",
    "device_id": "ESP32-A01",
    "protocol": "WIFI",
    "timestamp_utc": datetime(2026, 9, 10, 7, 26, 0, tzinfo=timezone.utc),
    "voltage_v": 2.27,
    "current_ma": 70.4,
    "power_mw": 159.8,
}

filter_key = {"site_id": doc["site_id"], "timestamp_utc": doc["timestamp_utc"]}
col.bulk_write([ReplaceOne(filter_key, doc, upsert=True)], ordered=False)
```

### Reading via the Dashboard REST API
Members reading telemetry (e.g. Member 4 dashboard) connect through the authenticated read API rather than querying MongoDB directly:
- **Latest Reading**:
  ```http
  GET /api/v1/query/latest?stream=readings&site_id=SITE-A
  X-API-Key: <ASK_MEMBER_3_PRIVATELY>
  ```
- **Time Series Aggregation**:
  ```http
  GET /api/v1/query/series?stream=readings&site_id=SITE-A&field=voltage_v&start=2026-09-10T00:00:00Z&stop=2026-09-11T00:00:00Z&every=1h&fn=mean
  X-API-Key: <ASK_MEMBER_3_PRIVATELY>
  ```
  *Note: In `/api/v1/query/series` with `every=...`, each point's time is the START of its UTC bucket (for example daily means are labeled 00:00:00Z of that day). The last bucket can be partial when `stop` falls inside it.*

---

## 8. Troubleshooting Matrix

| Symptom / Error | Root Cause | Resolution |
| :--- | :--- | :--- |
| **Authentication failed**<br>`OperationFailure: Authentication failed` | Invalid username or password, or special characters (`@`, `:`, `#`) not URL-encoded. | Verify user in Atlas **Database Access**. URL-encode special password characters. |
| **Timeout / SSL Alert**<br>`ServerSelectionTimeoutError`<br>`[SSL: TLSV1_ALERT_INTERNAL_ERROR]` | Client public IP is not in the Atlas IP Access List, or ISP dynamically assigned a new IP. | Find your public IP (`Invoke-RestMethod https://api.ipify.org`), open Atlas **Security** -> **Network Access**, and add your current IP address. |
| **Missing dnspython**<br>`ConfigurationError: The 'dnspython' module must be installed to use mongodb+srv:// URIs` | PyMongo requires `dnspython` to resolve SRV cluster DNS records. | Run `pip install dnspython` (included in `requirements.txt`). |
| **Rate Throttling / Slow Writes**<br>Operations take excessive time or timeout | Sending unbatched single inserts exceeding Atlas 100 ops/s free tier limit. | Batch documents into chunks of 100 via `bulk_write` with optional pause (`MONGO_WRITE_PAUSE_S`). |
| **HTTP 503 from Read API**<br>`{"detail": "database unreachable"}` | Backend API server cannot connect to Atlas cluster. | Verify Atlas connectivity with `python -m backend.tools.verify_backend` or check `GET /health`. |
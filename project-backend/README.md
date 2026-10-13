# Project Backend - Stage 1: Data Contract & Validation Engine

Stage 1 provides pure Python validation for the shared solar PV monitoring platform (J26-IT-348), enforcing the project data contract without external framework dependencies.

## Features
- **Data Contract Enforcement**: Validates UTC timestamps (ending with `Z` or `+00:00`), canonical site IDs (`SITE-A`..`SITE-F`), device IDs, and protocol bindings against `reference/sites.csv`.
- **InfluxDB Schema Alignment**: Strictly separates tags (`site_id`, `device_id`, `protocol`) from fields, never storing tags or timestamps as fields. Omit missing values (`None`, `""`, `NaN`) completely (never writing zeroes).
- **Reserved Measurements Protection**: Prohibits writes to Member 3's reserved measurements (`network_link_metrics`, `network_outages`, `fl_client_updates`, `fl_global_models`, `fl_site_state`, `ota_deployments`), even when prefixed.
- **Prefix Isolation**: Supports `MEASUREMENT_PREFIX` (e.g. `test_`) to isolate development and replay runs.
- **Strict & Open Modes**: Enforces strict typing for `readings` (`sensor_readings`) and open typing for provisional streams (`health`, `prediction`, `twin`).

---

## Setup & Virtual Environment

From `project-backend/`:

```powershell
# Create virtual environment with Python 3.12
py -3.12 -m venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1

# Install Stage 1 dependencies
pip install -r requirements.txt
```

---

## Running Unit Tests

Run test suite via `pytest`:

```powershell
pytest -q
```

All tests run locally in pure Python without requiring Docker or network services.

---

## Running File Validation (`check_file`)

Inspect and validate datasets (e.g., `pv_fault_dataset.csv`):

```powershell
# Validate operational window (3000 rows)
python -m backend.check_file --stream readings --file ..\pv_fault_dataset.csv --window OPERATION

# Validate with row limit
python -m backend.check_file --stream readings --file ..\pv_fault_dataset.csv --limit 500
```

The tool reports:
1. Total rows read, accepted, and rejected.
2. Rejection breakdowns by contract error code.
3. Dropped dataset metadata columns (`window`, `run_id`, `label`, `health_score`).
4. Range warnings (e.g. humidity).
5. Duplicate `(site_id, timestamp_utc)` series collision checks.

---

## For the Dashboard (Member 4)

### Overview & Base Configuration
- **Base URL**: `http://localhost:8000`
- **Required Header**: `X-API-Key: <READ_API_KEY>`
- **Security & Read-Only Guarantee**: `READ_API_KEY` provides strictly read-only access. It can never write, delete, or alter any data in InfluxDB or project streams. The ingestion key (`API_KEY`) cannot access read endpoints, and `READ_API_KEY` cannot access ingest endpoints.
- **Prefix Isolation**: Read routes dynamically query measurements using the current `MEASUREMENT_PREFIX` environment variable (e.g. `test_sensor_readings` during development and test runs, `sensor_readings` in production).

---

### Read Route Examples

#### 1. List Canonical Sites (`GET /api/v1/sites`)
Returns metadata for all 6 canonical sites (`site_id`, `device_id`, `panel_id`, `display_name`, `protocol`, `gateway_id`, `link_profile`).

**PowerShell (`Invoke-RestMethod`):**
```powershell
$headers = @{ "X-API-Key" = "$env:READ_API_KEY" }
$sites = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/sites" -Headers $headers -Method GET
$sites | Format-Table site_id, device_id, display_name, protocol
```

**JavaScript (`fetch`):**
```javascript
const response = await fetch("http://localhost:8000/api/v1/sites", {
  headers: { "X-API-Key": READ_API_KEY }
});
const sites = await response.json();
console.log("Sites:", sites);
```

#### 2. Query Latest Row (`GET /api/v1/query/latest`)
Retrieves the most recent record of a stream for a site, pivoted with all fields into a single object.

**PowerShell (`Invoke-RestMethod`):**
```powershell
$headers = @{ "X-API-Key" = "$env:READ_API_KEY" }
$latest = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/query/latest?stream=readings&site_id=SITE-A" -Headers $headers -Method GET
Write-Host "Latest time: $($latest.time)"
Write-Host "Voltage: $($latest.fields.voltage_v) V"
```

**JavaScript (`fetch`):**
```javascript
const response = await fetch(
  "http://localhost:8000/api/v1/query/latest?stream=readings&site_id=SITE-A",
  { headers: { "X-API-Key": READ_API_KEY } }
);
const latest = await response.json();
console.log(`Site: ${latest.site_id}, Time: ${latest.time}, Voltage: ${latest.fields.voltage_v}`);
```

#### 3. Query Time Series with Optional Aggregation (`GET /api/v1/query/series`)
Retrieves raw points or time-window aggregations (`every` in `1m, 5m, 15m, 1h, 1d`; `fn` in `mean, min, max, last, count`). Maximum range is 31 days.

**PowerShell (`Invoke-RestMethod`):**
```powershell
$headers = @{ "X-API-Key" = "$env:READ_API_KEY" }
$url = "http://localhost:8000/api/v1/query/series?stream=readings&site_id=SITE-A&field=voltage_v&start=2026-09-10T00:00:00Z&stop=2026-09-11T00:00:00Z&every=1h&fn=mean"
$series = Invoke-RestMethod -Uri $url -Headers $headers -Method GET
Write-Host "Points retrieved: $($series.points.Count)"
```

**JavaScript (`fetch`):**
```javascript
const url = "http://localhost:8000/api/v1/query/series?" + new URLSearchParams({
  stream: "readings",
  site_id: "SITE-A",
  field: "voltage_v",
  start: "2026-09-10T00:00:00Z",
  stop: "2026-09-11T00:00:00Z",
  every: "1h",
  fn: "mean"
});

const response = await fetch(url, { headers: { "X-API-Key": READ_API_KEY } });
const series = await response.json();
console.log(`Received ${series.points.length} points for ${series.field}:`, series.points);
```

---

### End-to-End Operational Flow

To run the complete system locally:

1. **Start Mosquitto MQTT Broker**:
   ```powershell
   docker compose up -d
   ```

2. **Start MQTT Ingestion Service**:
   ```powershell
   python -m backend.ingest.mqtt_service
   ```

3. **Start FastAPI Ingest & Query Server**:
   ```powershell
   uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
   ```

4. **Replay Historical Dataset**:
   ```powershell
   # HTTP replay of 3000 operational records:
   python -m backend.tools.replay --stream readings --file ..\pv_fault_dataset.csv --window OPERATION --transport http

   # Or MQTT replay for a single site:
   python -m backend.tools.replay --stream readings --file ..\pv_fault_dataset.csv --window OPERATION --site SITE-C --transport mqtt --speed 0
   ```

5. **Verify Backend Storage**:
   ```powershell
   python -m backend.tools.verify_backend --stream readings --file ..\pv_fault_dataset.csv --window OPERATION
   ```


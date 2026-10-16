# 8-Minute Demo Script: Shared Project Backend & Atlas Ingestion

Demonstrates the native Windows dev environment, isolated demo ingestion (`demo_` prefix), data contract validation and error rejection, read API endpoints, MongoDB Compass verification, and graceful database failure handling.

---

## Before-You-Start Checklist

Run this quick checklist in PowerShell before beginning the presentation:

1. **Atlas IP Allow-List**: Check your public IPv4 address and ensure it is active in the Atlas Network Access list:
   ```powershell
   Invoke-RestMethod https://api.ipify.org
   ```
2. **Mosquitto Broker Running**: Verify the native Mosquitto broker is listening on `127.0.0.1:1883`:
   ```powershell
   Test-NetConnection 127.0.0.1 -Port 1883
   ```
   *Expected: `TcpTestSucceeded : True`*
3. **Environment Variables**: Confirm required variable names are defined in `.env` (names only; never expose secrets):
   ```powershell
   Get-Content .env | Select-String "MONGO_URI", "API_KEY", "READ_API_KEY"
   ```
4. **MongoDB Compass Connected**:
   - Open MongoDB Compass and connect to `<MONGO_URI>`.
   - Expand the `solar_pv` database in the left navigation panel.
5. **Dataset Location Defined**:
   The shared synthetic dataset is maintained under `component3-federated-backend/data/shared/pv_fault_dataset.csv`. Define the PowerShell variable once:
   ```powershell
   $csv = "..\component3-federated-backend\data\shared\pv_fault_dataset.csv"
   Test-Path $csv
   ```
   *Expected: `True`*

---

## Architecture Note: Measurement Prefix Loading

The backend loads `MEASUREMENT_PREFIX` dynamically:
- In `backend/config.py`, `get_measurement_prefix()` checks `os.environ.get("MEASUREMENT_PREFIX")` first.
- If set in the PowerShell environment (`$env:MEASUREMENT_PREFIX = "demo_"`), it overrides `.env` because `load_dotenv()` defaults to non-overwriting mode (`override=False`).
- Running child processes launched via `start_dev.ps1` (FastAPI and the MQTT service) inherit process environment variables from the PowerShell session at startup.
- **Rule for the Demo**: Run `$env:MEASUREMENT_PREFIX = "demo_"` in the **SAME** PowerShell window before calling `.\scripts\start_dev.ps1`. Keep that same window for `replay.py` and `verify_backend.py`. Do NOT edit `.env` for the demo.

---

## 8-Minute Demo Walkthrough

### 1. Launch Services with Demo Prefix (1 minute)

In your PowerShell window, set the isolated demo prefix and start the background services:

```powershell
cd D:\Research-project\J26-IT-348\project-backend
.\.venv\Scripts\Activate.ps1

$csv = "..\component3-federated-backend\data\shared\pv_fault_dataset.csv"
$env:MEASUREMENT_PREFIX = "demo_"

.\scripts\start_dev.ps1
```

*Observe console output: Mosquitto broker verified, background MQTT service started, FastAPI server launched, and `/health` successfully returning status `ok`.*

---

### 2. Live Replay Ingestion (2 minutes)

Replay 300 valid records from the synthetic dataset using the HTTP transport:

```powershell
python -m backend.tools.replay --stream readings --file $csv --window OPERATION --limit 300 --transport http --speed 0.0
```

#### Command Options Verified via `replay.py --help`:
- `--stream readings`: Selects the strict telemetry sensor readings stream.
- `--file $csv`: Points to the canonical shared dataset.
- `--window OPERATION`: Filters records for the operational monitoring window (10–15 Sept 2026).
- `--limit 300`: Ingests a controlled chunk of 300 records for the demo.
- `--transport http`: Uses authenticated batch HTTP POST (`/api/v1/ingest/readings`).
- `--speed 0.0`: Ingests immediately without artificial throttle delay.

*Observe terminal output:*
The tool announces:
```
=== Replay Tool ===
Stream:             readings
Target Measurement: demo_sensor_readings
Rows selected:      300
```
It prints progress, duration, and ingestion throughput (~90–95 rows/s).

*(Optional: Demonstrate MQTT ingestion using `--transport mqtt` which publishes QoS 1 messages to topic `solar/{site_id}/readings`).*

---

### 3. Verify Database Integrity (1 minute)

Verify that all 300 records were ingested idempotently into MongoDB Atlas and that Member 3's reserved collections remain untouched:

```powershell
python -m backend.tools.verify_backend --stream readings --file $csv --window OPERATION --limit 300
```

#### Command Options Verified via `verify_backend.py --help`:
- `--stream readings`, `--file $csv`, `--window OPERATION`, `--limit 300`: Matches replay parameters to verify expected row counts.

*Observe terminal output:*
```
=== Backend Verification ===
Stream:             readings
Target Measurement: demo_sensor_readings
Expected rows:      300
Found rows:         300
Verification:       OK

Reserved Member 3 Measurements (field-value counts):
  [Reserved] fl_client_updates      : ...
  [Reserved] fl_global_models       : ...
  [Reserved] fl_site_state          : ...
  [Reserved] network_link_metrics   : ...
  [Reserved] network_outages        : ...
  [Reserved] ota_deployments        : ...
```

---

### 4. Live Read API Demonstration (1.5 minutes)

Query the read endpoints from PowerShell using the read-only key:

```powershell
$readKey = "<ASK_MEMBER_3_PRIVATELY>"
$headers = @{ "X-API-Key" = $readKey }

# A. Unauthenticated health check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# B. Canonical sites metadata
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/sites" -Headers $headers

# C. Query latest reading for SITE-A
$latest = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/query/latest?stream=readings&site_id=SITE-A" -Headers $headers
$latest | ConvertTo-Json -Depth 3

# D. Query hourly mean power series
$seriesUrl = "http://localhost:8000/api/v1/query/series?stream=readings&site_id=SITE-A&field=voltage_v&start=2026-09-10T00:00:00Z&stop=2026-09-11T00:00:00Z&every=1h&fn=mean"
$series = Invoke-RestMethod -Uri $seriesUrl -Headers $headers
Write-Host "Points retrieved: $($series.points.Count)"
$series.points[0..2]
```

> **Note on Series Aggregation:** In `/api/v1/query/series` with `every=...`, each point's time is the START of its UTC bucket (for example daily means are labeled `00:00:00Z` of that day). The last bucket can be partial when `stop` falls inside it.

---

### 5. Data Contract Enforcement & Bad-Message Rejection (1 minute)

Demonstrate that malformed or non-contract payloads are rejected immediately and logged to the rotating rejects file without writing to the database:

```powershell
$ingestKey = "<ASK_MEMBER_3_PRIVATELY>"
$badHeaders = @{
    "X-API-Key" = $ingestKey
    "Content-Type" = "application/json"
}

# Sending payload with invalid site_id ("SITE-Z") and missing UTC time
$badPayload = @(
    @{
        "record_id" = "BAD-001"
        "timestamp_utc" = "2026-09-10 12:00:00"   # Missing 'Z' offset
        "site_id" = "SITE-Z"                      # Unknown site
        "voltage_v" = 2.5
    }
) | ConvertTo-Json

try {
    Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ingest/readings" -Method POST -Headers $badHeaders -Body $badPayload
} catch {
    $response = $_.Exception.Response
    Write-Host "Rejection Status Code: $($response.StatusCode.value__)" -ForegroundColor Red
}

# Show that the rejection was safely recorded in rejects.jsonl
Get-Content rejects.jsonl -Tail 1
```

*Expected output: HTTP 422 Unprocessable Entity. The reject log records `TIME_NOT_UTC` and `UNKNOWN_SITE` with all secrets scrubbed.*

---

### 6. MongoDB Compass Inspection (30 seconds)

Switch to MongoDB Compass:
1. Refresh the database list under `solar_pv`.
2. Click on the collection **`demo_sensor_readings`**.
3. View the documents:
   - Exactly 300 documents are present.
   - `timestamp_utc` is stored natively as a BSON UTC date.
   - Missing fields are omitted rather than stored as `0.0` or `null`.

---

### 7. Graceful Database Unavailability Handling (1 minute)

Demonstrate the fail-safe behavior when MongoDB Atlas is unreachable:

```powershell
# Point to a non-existent cluster to simulate network/cluster failure
$env:MONGO_URI = "mongodb://badhost.invalid:27017"

python -m backend.tools.verify_backend --stream readings --file $csv --window OPERATION --limit 300

Write-Host "Exit Code: $LASTEXITCODE"
Remove-Item Env:MONGO_URI
```

*Observe behavior:*
- The tool immediately catches the connection failure without crashing with an unhandled stack trace.
- Prints: `CANNOT READ MONGODB: ...` (with all URI credentials scrubbed).
- Exits cleanly with exit code `2`.

---

## Restore Afterwards

Always restore the clean environment after concluding the demonstration:

```powershell
# 1. Clear the demo measurement prefix
Remove-Item Env:MEASUREMENT_PREFIX

# 2. Stop native dev background services
.\scripts\stop_dev.ps1
```

*Verifies that background PIDs from `.dev-pids` are cleanly terminated and the environment is restored.*

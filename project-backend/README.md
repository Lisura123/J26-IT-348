# Project Backend: Solar PV IoT Ingestion & Time-Series Engine

Robust, contract-governed ingestion pipeline and time-series query service for the Solar PV Monitoring Platform (J26-IT-348). Supports real-time MQTT telemetry, high-throughput batch HTTP ingestion, dynamic schema validation, and dashboard querying over InfluxDB 2.7.

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph Edge["Field Edge Devices (SITE-A..SITE-F)"]
        ESP[ESP32 Sensing Node]
        GW[LoRa / LTE Gateway]
    end

    subgraph Broker["MQTT Broker"]
        Mosquitto[Mosquitto 2.0\nPort 1883]
    end

    subgraph Backend["Project Backend (FastAPI & Services)"]
        MQTT_SVC["MQTT Ingest Service\n(backend.ingest.mqtt_service)"]
        API["FastAPI App\n(backend.api.main)"]
        VAL["Validation Engine\n(backend.validate)"]
        WRITER["InfluxWriter\n(backend.influx_writer)"]
        READER["InfluxReader\n(backend.influx_reader)"]
        REJECTS[("Rejects Log\n(rejects.jsonl)")]
    end

    subgraph Storage["InfluxDB 2.7 (solar_pv)"]
        TEST_DATA[("Prefixed Test Streams\ntest_sensor_readings\ntest_sensor_health\ntest_fault_predictions\ntest_twin_output")]
        RESERVED[("Member 3 Reserved Streams\nnetwork_link_metrics\nfl_client_updates\nfl_global_models\n...")]
    end

    subgraph Consumers["Consumers & Dashboard"]
        DASH["Member 4 Dashboard\n(GET /api/v1/query/*)"]
        M3["Member 3 Federated Engine"]
    end

    ESP -->|Publish solar/{site}/readings| Mosquitto
    GW -->|Publish solar/{site}/readings| Mosquitto
    Mosquitto -->|QoS 1 Subscribe| MQTT_SVC
    ESP -.->|HTTP POST /ingest/{stream}| API

    MQTT_SVC --> VAL
    API --> VAL
    VAL -->|Valid PointSpec| WRITER
    VAL -->|Rejected Payload| REJECTS

    WRITER -->|Batch Write| TEST_DATA
    WRITER -.x|REFUSED by Guard| RESERVED

    READER -->|Read-Only Flux| TEST_DATA
    API --> READER
    DASH -->|X-API-Key: READ_KEY| API
    M3 --> RESERVED
```

---

## 2. What Is Verified & How

| Component | Verification Method | Status |
| :--- | :--- | :--- |
| **Data Contract Engine** | 86+ pytest tests verifying UTC timestamps, canonical site lookups, numeric ranges, and strict/open schemas. | **Verified (86 passed)** |
| **Reserved Isolation** | Startup guard and write filters verify that Member 3's 6 reserved measurements remain completely untouched across all tests and replayed batches. | **Verified (Unchanged)** |
| **Replay Tool** | `python -m backend.tools.replay --stream readings --file ..\pv_fault_dataset.csv --window OPERATION --dry-run` selects exactly 3000 rows across 6 sites. | **Verified (3000 rows)** |
| **MQTT Replay & Verify** | Replaying SITE-C OPERATION rows via MQTT (`--speed 0`) and verifying storage via `verify_backend`: exact match (660 rows). | **Verified (OK: 660 rows)** |
| **HTTP Replay & Verify** | Replaying all 3000 OPERATION rows via HTTP and verifying storage via `verify_backend`: exact match (3000 rows). | **Verified (OK: 3000 rows)** |
| **Health Demo Replay** | Replaying 3000 health demo fixture rows and verifying storage via `verify_backend --stream health`: exact match (3000 rows). | **Verified (OK: 3000 rows)** |
| **Repeatable Smoke Test** | Random run marker offsets `voltage_v`; readback verifies that this specific run wrote the exact values. | **Verified (HTTP & MQTT)** |
| **Dashboard Query API** | Verified `GET /api/v1/sites`, `/query/latest`, `/query/series` (raw & aggregated); confirmed 401 on wrong key and ingest key. | **Verified** |

---

## 3. Repository Folder Map

```text
project-backend/
├── .env.example               # Environment template (no secrets)
├── .gitignore                 # Git ignore rules (includes .env, .dev-pids, rejects.jsonl)
├── docker-compose.yml         # Mosquitto MQTT broker configuration
├── pytest.ini                 # Pytest test discovery settings
├── requirements.txt           # Python dependencies
├── schema.yaml                # Stream data contracts & reserved measurements definition
├── backend/
│   ├── check_file.py          # Offline dataset contract validator
│   ├── config.py              # Schema loader, prefix resolver & safety guard
│   ├── influx_reader.py       # Read-only InfluxDB query client & FakeReader
│   ├── influx_writer.py       # Batched InfluxDB synchronous writer client
│   ├── points.py              # InfluxDB PointSpec data representation
│   ├── sites.py               # Canonical sites lookup registry
│   ├── validate.py            # Pure Python data contract validation engine
│   ├── api/
│   │   ├── main.py            # FastAPI ingestion application & CORSMiddleware
│   │   └── routes_query.py    # Dashboard read API routes (sites, latest, series)
│   ├── ingest/
│   │   ├── mqtt_service.py    # Background MQTT subscriber daemon (paho-mqtt)
│   │   ├── pipeline.py        # Central ingestion pipeline & counter tracking
│   │   └── rejects.py         # Rotating JSON lines rejects logger
│   └── tools/
│       ├── replay.py          # Dataset replay tool (HTTP / MQTT with speed control)
│       ├── smoke_test.py      # Repeatable end-to-end smoke test tool
│       └── verify_backend.py  # InfluxDB distinct row storage verification tool
├── examples/                  # Canonical stream payloads (readings, health, prediction, twin)
├── handover/                  # Audience-specific handover documents (Members 1, 2, 3, 4)
├── reference/
│   └── sites.csv              # Canonical pilot sites reference registry
├── scripts/
│   ├── start_dev.ps1          # Starts Mosquitto, MQTT service, and API in separate windows
│   └── stop_dev.ps1           # Stops dev processes cleanly by recorded PIDs
└── tests/
    ├── conftest.py            # Global test fixtures and temporary rejects isolation
    ├── test_stage2.py         # Stage 2 ingestion, MQTT, HTTP, and pipeline unit tests
    ├── test_stage3.py         # Stage 3 read API, auth separation, CORS, replay & verify tests
    └── test_validator.py      # Stage 1 data contract validator unit tests
```

---

## 4. Start in 5 Steps (PowerShell)

### Step 1: Create Virtual Environment
```powershell
cd project-backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Configure Environment
Copy `.env.example` to `.env` and fill in credentials provided privately by Member 3:
```powershell
Copy-Item .env.example .env
# Open .env and populate credentials (never commit this file)
```

### Step 3: Run the Test Suite
```powershell
pytest -q
```
*Expected: 86 passed in ~4 seconds.*

### Step 4: Launch Dev Services
```powershell
.\scripts\start_dev.ps1
```
*This verifies `.env` keys, starts Mosquitto in Docker, launches MQTT service and FastAPI in separate windows, and waits for `/health`.*

### Step 5: Verify Services & Replay Data
```powershell
# 1. Run repeatable smoke test over HTTP:
python -m backend.tools.smoke_test --transport http --file ..\pv_fault_dataset.csv

# 2. Query latest telemetry for SITE-A:
$headers = @{ "X-API-Key" = "<ASK_MEMBER_3_PRIVATELY>" }
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/query/latest?stream=readings&site_id=SITE-A" -Headers $headers

# 3. Stop dev services when finished:
.\scripts\stop_dev.ps1
```

---

## 5. The Measurement Prefix & Go-Live Rule

### Development vs Production Prefix
- **Development & Testing**: `MEASUREMENT_PREFIX=test_` routes telemetry to `test_sensor_readings`, `test_sensor_health`, etc. Dashboard read routes automatically read this same prefixed data.
- **Production (Go-Live)**: `MEASUREMENT_PREFIX=` (empty string).

### Hardcoded Startup Guard
To prevent accidental pollution of Member 3's real measurements or existing data, the backend enforces a startup guard:
```python
# Refuses to start if prefix is empty UNLESS ALLOW_REAL_MEASUREMENTS=1 is explicitly set
if not prefix and not allow_real:
    raise RuntimeError("SAFETY GUARD: Refusing to start to protect production data.")
```

**Go-Live Rule**:
Only set `MEASUREMENT_PREFIX=` and `ALLOW_REAL_MEASUREMENTS=1` after:
1. Members 1, 2, and 4 have confirmed their stream schemas.
2. `verify_backend` passes on test data.
3. Production Mosquitto credentials are configured.

---

## 6. Security Notes

- **Separate API Keys**:
  - `API_KEY` (Ingest Key): Allows `POST /api/v1/ingest/*`. **Cannot** read from query routes.
  - `READ_API_KEY` (Dashboard Key): Allows `GET /api/v1/*`. **Cannot** ingest or write points.
- **Never Commit `.env`**: `.env` and `.dev-pids` are gitignored. Placeholders (`<ASK_MEMBER_3_PRIVATELY>`) are used in all documentation.
- **Anonymous MQTT is for Development Only**:
  - The local `mosquitto/mosquitto.conf` allows anonymous access for easy development.
  - For staging/production, configure password authentication:
    ```bash
    # Create password file inside container:
    mosquitto_passwd -c /mosquitto/config/passwd <username>
    # In mosquitto.conf set:
    # allow_anonymous false
    # password_file /mosquitto/config/passwd
    ```
- **Rotating Leaked InfluxDB Tokens**:
  - If `INFLUX_TOKEN` is leaked, log into the InfluxDB UI (`http://localhost:8087`).
  - Navigate to **Load Data** -> **API Tokens**, generate a new token, update `.env`, and delete the old token in the UI.

---

## 7. Troubleshooting Table

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `HTTP 401 Unauthorized` | Missing or incorrect `X-API-Key` header; or ingest key passed to read route. | Verify header is `X-API-Key`. Ingest key cannot open read routes; dashboard key cannot open ingest routes. |
| `HTTP 503 Service Unavailable` | `API_KEY` or `READ_API_KEY` empty in `.env`. | Add valid keys to `.env` and restart service. |
| `HTTP 422 Unprocessable Entity` | Field injection attempt, invalid regex, undeclared field in strict stream, naive timestamp, or query range > 31 days. | Ensure field matches `^[a-z][a-z0-9_]*$`, timestamps have `Z`, and query range is within 31 days. |
| `RuntimeError: SAFETY GUARD` | `MEASUREMENT_PREFIX` is empty while `ALLOW_REAL_MEASUREMENTS` is not `1`. | In development, set `MEASUREMENT_PREFIX=test_` in `.env`. |
| `TIME_NOT_UTC` Rejection | Timestamp missing `Z` (e.g. `2026-09-10T12:00:00`). | Add explicit UTC `Z` suffix: `2026-09-10T12:00:00Z`. |
| `PROTOCOL_MISMATCH` Rejection | Device sent protocol not matching `reference/sites.csv`. | Verify protocol matches site configuration (e.g. SITE-C is `LORA`, not `WIFI`). |
| InfluxDB connection fails (`ping: False`) | InfluxDB container stopped or port conflict. | Verify InfluxDB container status (`docker ps`) and verify `INFLUX_URL` in `.env`. |

---

## 8. Honest Limitations

1. **Synthetic & Historical Dataset**: Validation was performed against historical data (`pv_fault_dataset.csv`). Real field conditions may introduce intermittent connectivity and jitter.
2. **Pending Member Formats**:
   - `health` stream from the dataset is a **demo fixture**; Member 1 must confirm quality flags.
   - `prediction` stream class probabilities are **proposed**; Member 2 must confirm output schema.
   - `twin_output` stream schema is **pending**; Member 4 must specify output simulation fields.
3. **No Hardware Tested**: ESP32 and LoRa code examples are reference implementations and have not yet been flashed to physical microcontrollers.
4. **MQTT Without TLS**: The current Mosquitto broker runs unencrypted TCP port 1883 for local workstation development; TLS (port 8883) must be added for field deployment over public networks.

---

## 9. Changelog

- **Stage 1**: Implemented pure Python data contract validation engine, canonical sites registry, strict/open stream modes, and offline dataset validation tool.
- **Stage 2**: Built batched InfluxWriter client with MS precision, MQTT ingestion daemon (QoS 1), FastAPI HTTP ingest routes, and rotating rejects logger.
- **Stage 3**: Added dashboard read API (`/sites`, `/query/latest`, `/query/series`) with CORS and injection protection, dataset replay tool, backend verification tool, and repeatable smoke test.
- **Stage 4**: Created comprehensive handover documentation for Members 1 through 4, canonical example payloads, automated PowerShell dev management scripts (`start_dev.ps1`, `stop_dev.ps1`), and root architecture entry point.

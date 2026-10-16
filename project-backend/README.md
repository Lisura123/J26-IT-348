# Project Backend: Solar PV IoT Ingestion & Time-Series Engine

Robust, contract-governed ingestion pipeline and time-series query service for the Solar PV Monitoring Platform (J26-IT-348). Supports real-time MQTT telemetry, high-throughput batch HTTP ingestion, dynamic schema validation, and dashboard querying over **MongoDB Atlas**. Runs entirely natively on Windows. Docker is not used by this project.

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph Edge["Field Edge Devices (SITE-A..SITE-F)"]
        ESP[ESP32 Sensing Node]
        GW[LoRa / LTE Gateway]
    end

    subgraph Broker["MQTT Broker (Native Windows)"]
        Mosquitto[Mosquitto 2.x\nPort 1883]
    end

    subgraph Backend["Project Backend (FastAPI & Services)"]
        MQTT_SVC["MQTT Ingest Service\n(backend.ingest.mqtt_service)"]
        API["FastAPI App\n(backend.api.main)"]
        VAL["Validation Engine\n(backend.validate)"]
        WRITER["MongoWriter\n(backend.mongo_writer)"]
        READER["MongoReader\n(backend.mongo_reader)"]
        REJECTS[("Rejects Log\n(rejects.jsonl)")]
    end

    subgraph Storage["MongoDB Atlas Cloud Database (solar_pv)"]
        TEST_DATA[("Prefixed Test Collections\ntest_sensor_readings\ntest_sensor_health\ntest_fault_predictions\ntest_twin_output")]
        RESERVED[("Member 3 Reserved Collections\nnetwork_link_metrics\nfl_client_updates\nfl_global_models\n...")]
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

    WRITER -->|ReplaceOne Upsert| TEST_DATA
    WRITER -.x|REFUSED by Guard| RESERVED

    READER -->|Read-Only Aggregation| TEST_DATA
    API --> READER
    DASH -->|X-API-Key: READ_KEY| API
    M3 --> RESERVED
```

---

## 2. What Is Verified & How

| Component | Verification Method | Status |
| :--- | :--- | :--- |
| **Data Contract Engine** | 21 pytest tests verifying UTC timestamps, canonical site lookups, numeric ranges, and strict/open schemas. | **Verified (21 passed)** |
| **Stage 2 Ingestion** | 21 tests verifying batched ingestion, MQTT message parser, type preservation, and HTTP limits. | **Verified (21 passed)** |
| **Stage 3 Read API** | 45 tests verifying read authentication, injection protection, aggregation windows, replay, and verify tool. | **Verified (45 passed)** |
| **Stage M1 MongoDB Engine** | 10 tests verifying storage factory, role accounts, BSON UTC datetimes, unique index upserts, HTTP 503 error mapping, and sanitized logging. | **Verified (10 passed)** |
| **Full Unit Test Suite** | 108 passed unit tests running completely in-process without external network dependencies. | **Verified (108 passed, 1 skipped)** |

---

## 3. Repository Folder Map

```text
project-backend/
├── .env.example               # Environment template with MongoDB Atlas variables (no secrets)
├── .gitignore                 # Git ignore rules (includes .env, .dev-pids, rejects.jsonl)
├── pytest.ini                 # Pytest test discovery settings
├── requirements.txt           # Python dependencies (includes pymongo; influxdb-client removed)
├── schema.yaml                # Stream data contracts & reserved measurements definition
├── backend/
│   ├── check_file.py          # Offline dataset contract validator
│   ├── config.py              # Schema loader, prefix resolver & safety guard
│   ├── mongo_reader.py        # Read-only MongoDB Atlas query client & FakeReader
│   ├── mongo_writer.py        # Batched MongoDB Atlas ReplaceOne idempotent upsert client
│   ├── points.py              # PointSpec data representation
│   ├── sites.py               # Canonical sites lookup registry
│   ├── storage.py             # MongoDB client factory, timeouts, role credentials & sanitization
│   ├── validate.py            # Pure Python data contract validation engine
│   ├── api/
│   │   ├── main.py            # FastAPI ingestion application & CORSMiddleware
│   │   └── routes_query.py    # Dashboard read API routes (sites, latest, series)
│   ├── ingest/
│   │   ├── mqtt_service.py    # Background MQTT subscriber daemon (paho-mqtt)
│   │   ├── pipeline.py        # Central ingestion pipeline & counter tracking
│   │   └── rejects.py         # Rotating JSON lines rejects logger with secret scrubbing
│   └── tools/
│       ├── replay.py          # Dataset replay tool (HTTP / MQTT with speed control)
│       ├── smoke_test.py      # Repeatable end-to-end smoke test tool
│       └── verify_backend.py  # MongoDB row count and data contract verification tool
├── examples/                  # Canonical stream payloads (readings, health, prediction, twin)
├── handover/                  # Audience-specific handover documents (Members 1, 2, 3, 4)
├── mosquitto/
│   └── mosquitto.conf.example # Example configuration for native Mosquitto broker
├── reference/
│   └── sites.csv              # Canonical pilot sites reference registry
├── scripts/
│   ├── start_dev.ps1          # Starts MQTT service (if broker active) and FastAPI
│   └── stop_dev.ps1           # Stops dev processes cleanly by recorded PIDs (Windows services untouched)
└── tests/
    ├── conftest.py            # Global test fixtures and temporary rejects isolation
    ├── test_stage2.py         # Stage 2 ingestion, MQTT, HTTP, and pipeline unit tests
    ├── test_stage3.py         # Stage 3 read API, auth separation, CORS, replay & verify tests
    ├── test_stage_m1.py       # Stage M1 MongoDB Atlas migration unit & security tests
    └── test_validator.py      # Stage 1 data contract validator unit tests
```

---

## 4. Native Windows Setup & Start (Mosquitto Windows Service)

### Step 1: Python Virtual Environment
Run the following in PowerShell from `project-backend/`:
```powershell
# Create virtual environment if not already present
py -3.12 -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements (includes pymongo>=4.6.0, dnspython)
pip install -r requirements.txt
```

### Step 2: Configure MongoDB Atlas & Environment
1. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Populate the MongoDB Atlas variables in `.env` (never commit this file):
   - `MONGO_URI`: Atlas connection URI for the application user (e.g. `mongodb+srv://user:pass@cluster.mongodb.net/solar_pv?retryWrites=true&w=majority`).
   - `MONGO_DB`: Target database name (default: `solar_pv`).
   - `MONGO_APP_USER` / `MONGO_APP_PASSWORD`: Application writer account credentials.
   - `MONGO_READER_USER` / `MONGO_READER_PASSWORD`: Read-only account credentials for dashboard queries.
   - `API_KEY` & `READ_API_KEY`: Distinct secrets for ingest and read APIs.
   - `MEASUREMENT_PREFIX`: Set to `test_` for development and testing.

### Step 3: Run the Test Suite
Tests run in-process with mocks and data fixtures without requiring external services:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
*Expected: 108 passed, 1 skipped (gated live integration test).*

### Step 4: Mosquitto MQTT Broker (Windows Service)
For MQTT ingestion, install Eclipse Mosquitto for Windows (e.g. from https://mosquitto.org/download/ or via Windows installer). Run Mosquitto as a Windows service listening on `127.0.0.1:1883` with anonymous access allowed for local development (see [mosquitto.conf.example](file:///d:/Research-project/J26-IT-348/project-backend/mosquitto/mosquitto.conf.example)).

`scripts/start_dev.ps1` and `replay.py` verify the MQTT broker with `Test-NetConnection 127.0.0.1 -Port 1883` (the IPv4 address, not `localhost`, because the broker listens on `127.0.0.1` only):
- **If broker is listening on 127.0.0.1:1883**: `start_dev.ps1` starts the background MQTT ingest service and prints `Mosquitto Broker: 127.0.0.1:1883 (Windows service)`.
- **If broker is not listening**: `start_dev.ps1` prints `MQTT broker not found: install Mosquitto for Windows (see README). Continuing with HTTP only.` and skips MQTT ingest (exiting 0; HTTP ingestion remains fully operational).
- `start_dev.ps1` and `stop_dev.ps1` never start, stop, or modify Windows services or Mosquitto.

### Step 5: Launch Dev Services
```powershell
.\scripts\start_dev.ps1
```
*Launches MQTT ingest service (if broker active) and FastAPI server. Records PIDs in `.dev-pids` and polls `/health`.*

To stop all services:
```powershell
.\scripts\stop_dev.ps1
```
*Stops ONLY the PIDs listed in `.dev-pids` and their descendant processes (found through `ParentProcessId`). Never selects processes by name and never touches Windows services or other system processes.*

---

## 5. MongoDB Atlas Setup & Configuration

### A. Free M0 Cluster Setup
1. Create a free shared **M0 cluster** on MongoDB Atlas (AWS, GCP, or Azure).
2. Choose your closest region for minimal round-trip latency.

### B. Free Tier Limits & Write Throttle
- **100 Operations/Second Rate Limit**: Atlas M0 free tier throttles bulk operations at ~100 writes per second.
  - `MongoWriter` sends `bulk_write` in chunks of `MONGO_WRITE_BATCH` (default 100) with `MONGO_WRITE_PAUSE_S` (default 0.0) and retries failed chunks up to 3 times with exponential backoff on transient network/timeout errors.
  - Replaying 3,000 rows can take 30 seconds or more (~90–95 rows/s). `replay.py` displays an estimated duration at startup and reports live throughput in rows per second.
- **512 MB Storage Limit**: M0 clusters provide up to 512 MB of BSON storage.
- **One Free Cluster Per Project**: Atlas permits only one M0 cluster per Atlas project.
- **Idle Cluster Auto-Termination / Pause**: Free clusters with no activity over consecutive days may be paused by Atlas.
- **Internet Required**: Atlas is cloud-hosted; all team members require active internet connectivity.

### C. Database Access & User Roles
Configure two distinct least-privilege database users in Atlas under **Security** -> **Database Access**:
1. **Writer Account (`MONGO_APP_USER`)**:
   - Privileges: Specific built-in role **`readWrite`** on database **`solar_pv`**.
   - Used by the ingestion pipeline (`MongoWriter`, MQTT service, and HTTP ingest routes).
2. **Reader Account (`MONGO_READER_USER`)**:
   - Privileges: Specific built-in role **`read`** on database **`solar_pv`**.
   - Used exclusively by query endpoints (`MongoReader`, `/api/v1/query/*`) to enforce read-only access.

### D. Network Access & Access List Per Member
Configure access under **Security** -> **Network Access**:
- **Why NOT `0.0.0.0/0`**: Setting `0.0.0.0/0` (allow access from anywhere) exposes the database port to automated internet port scans, credential stuffing bots, and brute-force intrusion attempts.
- **Access List Per Member**: Add each member's specific public IP address (`/32 CIDR`).
- **Dynamic IP Changes**: ISP DHCP reconnects or router restarts change client public IPs. If connections fail with `[SSL: TLSV1_ALERT_INTERNAL_ERROR]`, navigate to **Network Access** in Atlas and click **Add Current IP Address**.

### E. Reader Account Derivation from `MONGO_URI`
In `backend/storage.py`, `build_role_uri(base_uri, role="reader")` derives the read-only connection string dynamically from `MONGO_URI`:
1. Parses the `mongodb+srv://` base URI using `urllib.parse.urlsplit`.
2. Extracts the cluster hostname from `p.netloc` (`host_part = p.netloc.split("@")[-1]`).
3. Reads `MONGO_READER_USER` and `MONGO_READER_PASSWORD` from the environment.
4. URL-encodes user and password via `urllib.parse.quote_plus` to safely support special characters.
5. Replaces **only** the `user:password` segment: `f"{user_enc}:{pass_enc}@{host_part}"`.
6. Preserves the scheme (`mongodb+srv`), cluster address, database path (`/solar_pv`), and query parameters (`retryWrites=true&w=majority`).
7. **Never logs, exposes, or prints raw passwords or connection URIs.**

---

## 6. Service Health API (`/health`)

The health endpoint provides real-time verification of backend status and Atlas connectivity without requiring authentication:

```http
GET /health HTTP/1.1
Host: localhost:8000
```

**Response Format:**
```json
{
  "status": "ok",
  "storage": "mongodb",
  "connected": true,
  "counters": {
    "received": 3000,
    "accepted": 3000,
    "rejected": 0,
    "written": 3000,
    "write_failed": 0
  }
}
```
- `status`: `"ok"` if MongoDB Atlas ping succeeds; `"degraded"` if unreachable.
- `storage`: Always `"mongodb"`.
- `connected`: `true` when MongoDB Atlas ping succeeds, `false` otherwise.
- `counters`: Real-time operational counters from the ingestion pipeline.
*(Note: `influx_connected`, `mongo_connected`, and `database_connected` have been removed in favor of `connected`.)*

---

## 7. MongoDB Atlas Troubleshooting

| Issue / Error | Root Cause | Solution |
| :--- | :--- | :--- |
| **Authentication failed**<br>`OperationFailure: Authentication failed` | Incorrect username or password, unescaped special characters in password, or user lacks role on `solar_pv`. | 1. Verify credentials in Atlas **Database Access**.<br>2. If password contains characters like `@`, `:`, `#`, ensure it is URL-encoded or let `build_role_uri` encode it.<br>3. Verify user privileges on database `solar_pv`. |
| **Server Selection Timeout / SSL Alert**<br>`ServerSelectionTimeoutError`<br>`[SSL: TLSV1_ALERT_INTERNAL_ERROR]` | Current client public IP is missing from Atlas Network Access IP Access List, cluster is paused, or outbound port 27017 is blocked. | 1. Go to Atlas **Security** -> **Network Access**.<br>2. Click **Add IP Address** -> **Add Current IP Address**.<br>3. Note: Dynamic ISP IP changes require adding the new IP.<br>4. Check cluster status in Atlas UI to ensure it is not paused. |
| **dnspython Missing**<br>`ConfigurationError: The 'dnspython' module must be installed to use mongodb+srv:// URIs` | PyMongo requires `dnspython` to resolve SRV records for `mongodb+srv://` connection strings. | Run `pip install dnspython` (included in `requirements.txt`). |
| **Database Unreachable**<br>`HTTP 503: database unreachable` | Query route attempted to read from Atlas while connection timed out or credentials failed. | Run `verify_backend` or check `/health` to inspect connectivity diagnostics. |

---

## 8. Verifying Connectivity Without Printing Secrets

To verify that your MongoDB Atlas cluster is reachable from Windows without logging or exposing connection strings or passwords, run:

```powershell
.\.venv\Scripts\python.exe -c "from backend.mongo_writer import MongoWriter; w = MongoWriter.from_env(); print('MongoDB Atlas Ping:', 'SUCCESS' if w.ping() else 'FAILED')"
```

---

## 9. Changelog

- **Stage 1**: Pure Python contract validation, canonical sites registry, strict/open stream validation.
- **Stage 2**: Ingestion pipeline, MQTT service (QoS 1), FastAPI ingest endpoints, rotating rejects logger.
- **Stage 3**: Read API (`/sites`, `/query/latest`, `/query/series`) with CORS and injection guards, replay and verify tools.
- **Stage 4**: Handover documentation and dev scripts.
- **Stage M1 (MongoDB Atlas Migration & Hardening)**:
  - Replaced InfluxDB with MongoDB Atlas cloud database (`pymongo>=4.6.0`, `dnspython`).
  - Implemented `backend/storage.py` with Atlas timeouts (`serverSelectionTimeoutMS=10000`, `connectTimeoutMS=10000`, `socketTimeoutMS=60000`, `retryWrites=true`).
  - Added role-based reader account URI derivation in `build_role_uri` without exposing credentials.
  - Implemented batch chunking (`MONGO_WRITE_BATCH`, default 100), pause (`MONGO_WRITE_PAUSE_S`), and exponential backoff retry on transient errors in `MongoWriter`.
  - Added throughput reporting (rows/s) and estimated duration calculation in `replay.py`.
  - Updated `/health` endpoint to return `{"status": "ok"|"degraded", "storage": "mongodb", "connected": bool, "counters": {...}}` and removed legacy `*_connected` flags.
  - Updated dev scripts: `start_dev.ps1` checks `127.0.0.1:1883` and starts MQTT service conditionally with `.dev-pids` tracking; `stop_dev.ps1` stops strictly by PID and `ParentProcessId` process tree.
  - Added dataset resolution in tests across `PV_DATASET` env var, `..\pv_fault_dataset.csv`, and `..\component3-federated-backend\data\shared\pv_fault_dataset.csv`.
  - Full test suite passing: 108 passed, 1 skipped.


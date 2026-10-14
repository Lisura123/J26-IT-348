# Handover: Member 4 (Digital Twin & Dashboard)

## 1. Overview & Base Connection
The dashboard connects to the read-only REST API exposed by the backend:
- **Base URL**: `http://localhost:8000`
- **Required Header**: `X-API-Key: <ASK_MEMBER_3_PRIVATELY>`
- **Security Guarantee**: The `READ_API_KEY` is strictly read-only. It has zero write/delete permissions.
- **Prefix Isolation**: Read queries dynamically resolve measurements based on the backend's active `MEASUREMENT_PREFIX` environment variable (e.g. `test_sensor_readings` during development/testing; `sensor_readings` in production).

---

## 2. API Endpoints Reference

### A. List Sites (`GET /api/v1/sites`)
Returns canonical site metadata from `reference/sites.csv`.
- **Query Parameters**: None.
- **Response Format**:
```json
[
  {
    "site_id": "SITE-A",
    "device_id": "ESP32-A01",
    "panel_id": "PV-A01",
    "display_name": "Pilot site A",
    "protocol": "WIFI",
    "gateway_id": "GW-01",
    "link_profile": "strong"
  }
]
```

### B. Query Latest Record (`GET /api/v1/query/latest`)
Pivots all fields for the newest timestamp into a single JSON object.
- **Query Parameters**:
  - `stream` (required): `readings`, `health`, `prediction`, or `twin`
  - `site_id` (required): one of `SITE-A` .. `SITE-F`
- **Response Format (200 OK)**:
```json
{
  "stream": "readings",
  "site_id": "SITE-A",
  "time": "2026-09-14T08:54:55Z",
  "tags": {
    "site_id": "SITE-A",
    "device_id": "ESP32-A01",
    "protocol": "WIFI"
  },
  "fields": {
    "voltage_v": 2.3,
    "current_ma": 38.31,
    "power_mw": 88.11,
    "ambient_temp_c": 29.52,
    "humidity_pct": 79.13,
    "pressure_hpa": 1008.42,
    "irradiance_lux": 49412.0,
    "dust_vo_v": 0.084,
    "reading": 12,
    "record_id": "REC-04032"
  }
}
```
- **Error Codes**:
  - `404 Not Found`: No data stored for this site and stream.
  - `422 Unprocessable Entity`: Invalid stream or unknown site.

### C. Query Time Series & Aggregation (`GET /api/v1/query/series`)
Retrieves raw time-series points or time-window bucket aggregations.
- **Query Parameters**:
  - `stream` (required): `readings`, `health`, `prediction`, or `twin`
  - `site_id` (required): `SITE-A` .. `SITE-F`
  - `field` (required): field name (e.g. `voltage_v`, `power_mw`, `health_score`)
  - `start` (required): ISO 8601 UTC timestamp with `Z` (e.g. `2026-09-10T00:00:00Z`)
  - `stop` (required): ISO 8601 UTC timestamp with `Z` (e.g. `2026-09-11T00:00:00Z`)
  - `every` (optional): aggregation bucket size; allow-list: `1m`, `5m`, `15m`, `1h`, `1d`
  - `fn` (optional): aggregation function; allow-list: `mean`, `min`, `max`, `last`, `count`
  - `limit` (optional): integer (default `1000`, maximum `10000`)
- **Query Guardrails**:
  - Maximum query range between `start` and `stop` is **31 days** (else `422`).
  - Without `every`, raw points up to `limit` are returned.
  - If `every` is provided, `fn` must also be provided.
- **Response Format**:
```json
{
  "stream": "readings",
  "site_id": "SITE-A",
  "field": "voltage_v",
  "points": [
    { "time": "2026-09-10T01:00:00Z", "value": 2.302 },
    { "time": "2026-09-10T02:00:00Z", "value": 2.305 }
  ]
}
```

---

## 3. Client Implementation Examples

### PowerShell Example
```powershell
$apiKey = "<ASK_MEMBER_3_PRIVATELY>"
$headers = @{ "X-API-Key" = $apiKey }

# 1. Get sites
$sites = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/sites" -Headers $headers -Method GET

# 2. Get latest readings for SITE-A
$latest = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/query/latest?stream=readings&site_id=SITE-A" -Headers $headers -Method GET

# 3. Query hourly mean power over 2 days
$url = "http://localhost:8000/api/v1/query/series?stream=readings&site_id=SITE-A&field=power_mw&start=2026-09-10T00:00:00Z&stop=2026-09-12T00:00:00Z&every=1h&fn=mean"
$series = Invoke-RestMethod -Uri $url -Headers $headers -Method GET
Write-Host "Points retrieved: $($series.points.Count)"
```

### JavaScript / TypeScript Example (`fetch`)
```javascript
const READ_API_KEY = "<ASK_MEMBER_3_PRIVATELY>";
const BASE_URL = "http://localhost:8000";

// Fetch aggregated time series
async function fetchAggregatedSeries(stream, siteId, field, startUtc, stopUtc, every, fn) {
  const params = new URLSearchParams({
    stream,
    site_id: siteId,
    field,
    start: startUtc,
    stop: stopUtc,
    every,
    fn,
    limit: "5000"
  });

  const response = await fetch(`${BASE_URL}/api/v1/query/series?${params.toString()}`, {
    method: "GET",
    headers: {
      "X-API-Key": READ_API_KEY
    }
  });

  if (!response.ok) {
    throw new Error(`Query failed: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  return data.points; // Array of { time, value }
}
```

---

## 4. Alternative: Direct InfluxDB Read-Only Token
If Member 4 desires to run custom Flux queries directly against InfluxDB (port 8087):
1. Ask Member 3 to generate a **Read-Only Scoped InfluxDB Token** for bucket `solar_pv`.
2. Do **NOT** use Member 3's all-access administrative token.
3. Use InfluxDB client libraries (`@influxdata/influxdb-client` for JS).

---

## 5. Fields Available Per Stream & Twin Questions

### Available Telemetry Fields (`readings`):
- `voltage_v` (Volts)
- `current_ma` (mA)
- `power_mw` (mW)
- `panel_temp1_c`, `panel_temp2_c`, `ambient_temp_c` (°C)
- `humidity_pct` (%)
- `pressure_hpa` (hPa)
- `irradiance_lux` (Lux)
- `dust_vo_v` (V)
- `reading` (counter)
- `record_id` (ID)

### Health Fields (`health`):
- `health_score` (0.0 to 1.0)
- `record_id` (ID)

### Prediction Fields (`prediction`):
- `predicted_label` (`NORMAL`, `PARTIAL_SHADING`, `DUST`, `OPEN_CIRCUIT`, `SHORT_CIRCUIT`)
- `confidence` (0.0 to 1.0)
- `prob_normal`, `prob_partial_shading`, `prob_dust`, `prob_open_circuit`, `prob_short_circuit`

### Action Items for Member 4 (`twin_output` stream):
Please provide the backend team with the exact field names and types required for digital twin output ingestion:
- Simulated power / expected power?
- Degradation estimate / thermal loss calculation?
- Simulation timestamp and parameters?
Once received, they will be registered into `schema.yaml`.

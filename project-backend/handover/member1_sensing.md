# Handover: Member 1 (Sensing & Data Acquisition)

## 1. Overview & MQTT Topics
Sensor telemetry is ingested via MQTT over QoS 1 or authenticated HTTP POST.
- Telemetry readings topic: `solar/{site_id}/readings`
- Sensor health topic: `solar/{site_id}/health`
- Allowed `site_id` values: `SITE-A`, `SITE-B`, `SITE-C`, `SITE-D`, `SITE-E`, `SITE-F`.
- Matching `device_id` values: `ESP32-A01` .. `ESP32-F01` (see `reference/sites.csv`).

---

## 2. Telemetry (`readings`) Payload & Field Specification

### Real Readings Payload (from dataset, metadata columns dropped)
```json
{
  "record_id": "REC-03001",
  "timestamp_utc": "2026-09-10T01:49:00Z",
  "site_id": "SITE-B",
  "device_id": "ESP32-B01",
  "panel_id": "PV-B01",
  "reading": 1,
  "voltage_v": 2.721,
  "current_ma": 12.87,
  "power_mw": 35.03,
  "panel_temp1_c": 27.62,
  "panel_temp2_c": 27.21,
  "ambient_temp_c": 23.5,
  "humidity_pct": 90.71,
  "pressure_hpa": 1009.89,
  "irradiance_lux": 14350.0,
  "dust_vo_v": 1.226
}
```

### Field Table
| Field Name | Type | Document Role | Unit / Format | Rules & Constraints |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp_utc` | string | Timestamp (`_time`) | ISO 8601 UTC with `Z` | **Required.** Must end in `Z` or `+00:00`. Local or naive times rejected. |
| `site_id` | string | Tag | Identifier | **Required.** Must match `reference/sites.csv` and the MQTT topic. |
| `device_id` | string | Tag | Identifier | If provided, must match site mapping. Auto-filled if omitted. |
| `protocol` | string | Tag | `WIFI`, `LORA`, `LTE` | If provided, must match canonical protocol. Auto-filled if omitted. |
| `record_id` | string | String Field | Identifier | **Required.** Unique record identifier (`REC-xxxxx`). |
| `panel_id` | string | String Field | Identifier | Optional panel identifier (e.g. `PV-B01`). |
| `reading` | integer | Integer Field | Count | Reading counter / sequence number. |
| `voltage_v` | float | Float Field | Volts (V) | PV panel terminal voltage. |
| `current_ma` | float | Float Field | Milliamperes (mA) | PV current. |
| `power_mw` | float | Float Field | Milliwatts (mW) | Calculated power. |
| `panel_temp1_c` | float | Float Field | Degrees Celsius (°C) | Panel surface sensor 1 temperature. |
| `panel_temp2_c` | float | Float Field | Degrees Celsius (°C) | Panel surface sensor 2 temperature. |
| `ambient_temp_c` | float | Float Field | Degrees Celsius (°C) | Ambient air temperature. |
| `humidity_pct` | float | Float Field | Percentage (%) | Relative humidity (0.0 to 100.0). |
| `pressure_hpa` | float | Float Field | Hectopascals (hPa) | Barometric atmospheric pressure. |
| `irradiance_lux` | float | Float Field | Lux | Solar illuminance. |
| `dust_vo_v` | float | Float Field | Volts (V) | Dust sensor analog output voltage. |

---

## 3. Strict Ingestion Rules
1. **UTC Timestamps Only**: Timestamps must be timezone-aware UTC ending in `Z` (e.g. `2026-09-10T07:26:00Z`). Timestamps without `Z` or with non-zero offsets are rejected (`TIME_NOT_UTC`).
2. **Mandatory Record ID**: Every row must carry a distinct `record_id`.
3. **Missing Values Must Be Omitted**: If a sensor fails or disconnects, do **NOT** send `0` or `"null"`. Simply omit the key entirely from the JSON payload. Missing values are omitted in MongoDB; sending `0` pollutes calculations and aggregation.
4. **Fixed Data Types**: Never alternate between numbers and strings. Float values must always be JSON numbers.
5. **No Dataset Metadata**: Metadata columns (`window`, `run_id`, `label`, `health_score`) are stripped in strict mode.

---

## 4. Client Code Examples

### A. ESP32 Arduino Example (PubSubClient + ArduinoJson)
> **Notice**: Not tested on hardware. Adapt WiFi credentials and NTP sync to your hardware profile.

```cpp
// NOT TESTED ON HARDWARE - REFERENCE IMPLEMENTATION
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <time.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* mqtt_server = "192.168.1.100"; // Broker IP
const int mqtt_port = 1883;
const char* mqtt_topic = "solar/SITE-A/readings";

WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(500); }
  // Sync UTC time via NTP
  configTime(0, 0, "pool.ntp.org");
}

String get_iso_time() {
  time_t now;
  time(&now);
  struct tm timeinfo;
  gmtime_r(&now, &timeinfo);
  char buf[30];
  strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &timeinfo);
  return String(buf);
}

void publish_reading() {
  StaticJsonDocument<512> doc;
  doc["record_id"] = "REC-ESP32-001";
  doc["timestamp_utc"] = get_iso_time();
  doc["site_id"] = "SITE-A";
  doc["device_id"] = "ESP32-A01";
  doc["reading"] = 1;
  doc["voltage_v"] = 18.42;
  doc["current_ma"] = 430.5;
  doc["power_mw"] = 7929.8;
  doc["ambient_temp_c"] = 28.5;
  doc["humidity_pct"] = 72.4;

  char jsonBuffer[512];
  serializeJson(doc, jsonBuffer);
  client.publish(mqtt_topic, jsonBuffer, true); // QoS 1 equivalent
}

void setup() {
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
}

void loop() {
  if (!client.connected()) {
    while (!client.connect("ESP32-SITE-A")) { delay(1000); }
  }
  client.loop();
  publish_reading();
  delay(5000);
}
```

### B. Python Paho MQTT Example
```python
import json
from datetime import datetime, timezone
import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "solar/SITE-A/readings"

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="sensor-feeder")
client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
client.loop_start()

payload = {
    "record_id": "REC-99001",
    "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "site_id": "SITE-A",
    "device_id": "ESP32-A01",
    "reading": 100,
    "voltage_v": 18.25,
    "current_ma": 420.0,
    "power_mw": 7665.0,
    "ambient_temp_c": 29.1,
    "humidity_pct": 68.4,
}

info = client.publish(TOPIC, json.dumps(payload), qos=1)
info.wait_for_publish()
print("Published reading successfully.")

client.loop_stop()
client.disconnect()
```

### C. HTTP Ingestion Example
**PowerShell:**
```powershell
$headers = @{
    "X-API-Key" = "<ASK_MEMBER_3_PRIVATELY>"
    "Content-Type" = "application/json"
}

$body = @{
    record_id = "REC-99002"
    timestamp_utc = "2026-09-10T08:00:00Z"
    site_id = "SITE-A"
    voltage_v = 18.15
    current_ma = 415.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ingest/readings" -Headers $headers -Method Post -Body $body
```

**cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/ingest/readings \
  -H "X-API-Key: <ASK_MEMBER_3_PRIVATELY>" \
  -H "Content-Type: application/json" \
  -d '{
    "record_id": "REC-99002",
    "timestamp_utc": "2026-09-10T08:00:00Z",
    "site_id": "SITE-A",
    "voltage_v": 18.15,
    "current_ma": 415.0
  }'
```

---

## 5. Sensor Health (`health`) Stream

- **Topic**: `solar/{site_id}/health`
- **Currently Verified in Schema**:
  - `record_id`: string (mandatory)
  - `timestamp_utc`: string (ISO 8601 UTC with `Z`)
  - `site_id`: string (must match topic)
  - `health_score`: float (0.0 to 1.0)
- **What Member 1 Must Confirm**:
  1. Quality flags representation: should error states be booleans (`sensor_ok: true`) or a status string enum (`status: "NORMAL"`)?
  2. Sub-sensor diagnostics: are individual sensor statuses emitted (e.g. `temp_sensor_healthy`, `irradiance_sensor_healthy`)?
- **Crucial Rule**: In MongoDB, fields maintain exact types (float vs int vs bool). Confirm field types and schema before sending production health records.

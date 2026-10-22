 # Component 1 - Sensing and Grading# Component 1: Adaptive Multi-Sensor PV Panel Health Grading

**Member 1:** Lisura Sigera (IT23389038)
**Group:** J26-IT-348 (IT4010 Research Project, SLIIT)
**Branch:** `component1-sensing-grading`

My component checks the health of a solar panel.
It measures the panel and its environment, cleans the data,
and gives a **health score (0 - 100)** and a **health status**.

| Status | Health score |
|---|---|
| HEALTHY | 70 - 100 |
| DEGRADED | 40 - 70 |
| FAULTY | 0 - 40 |
| NOT_GRADED | too dark to judge (below 500 lux), or the dust sensor is warming up |

Finding the fault **type** is Member 2's part (TinyML). My output is only the score and the status.

---

## 1. Why

A panel can lose power because of dust, shade, an open circuit or a short circuit.
But on a cloudy day the power is also low.
So power alone cannot tell us if the panel is broken.
I compare each reading with a **healthy panel in the same light and temperature**.

---

## 2. Hardware

| Part | Job |
|---|---|
| ESP32 | Runs the full pipeline, sends results over WiFi |
| INA219 (0x40) | Measures the panel current |
| 120 ohm, 5 W resistor | Load for the panel |
| TEMT6000 + ADS1115 (0x48) | Light (ADS1115 because the ESP32 ADC is noisy with WiFi on) |
| 2 x DS18B20 | Panel temperature and air temperature |
| DSM501B | Dust (%) |
| Relay (low trigger) | Cuts the panel to make an open circuit (fault test) |
| SD card module | Saves every raw reading as CSV |
| Panel | 5 V, 200 mA mini panel |

### Wiring

| ESP32 pin | Connected to |
|---|---|
| GPIO 21 / 22 | I2C SDA / SCL (INA219, ADS1115) |
| GPIO 4 | DS18B20 data (both probes) |
| GPIO 35 | DSM501B Vout2 (LPO) |
| GPIO 26 | Relay IN |
| GPIO 5 / 17 / 16 / 23 | SD card CS / SCK / MISO / MOSI |

Panel loop (not connected to the ESP32 ground):

```
Panel(+) -> Relay -> INA219 Vin+ -> [0.1 ohm shunt] -> INA219 Vin- -> 120 ohm load -> Panel(-)
```

**Voltage is calculated, not measured:** the panel negative is kept apart from the ESP32 ground,
so the INA219 bus voltage reads 0 V. I use **V = I x (120 + 0.1)**.
I checked this with a multimeter: the error was 2.5 %.

---

## 3. The pipeline (4 stages)

| Stage | What it does | Algorithm |
|---|---|---|
| 1. Data quality | Removes duplicates, -127 / -999 temperatures, fixes a reversed wire, fixes spikes, finds a shorted load (V > 6.5 V) | Session median, rolling median, physics rule |
| 2. Noise | Smooths the sensors and fills missing temperatures | 1D Kalman filter (R measured from the data) |
| 3. Indicators + weights | Compares each reading with a healthy panel, makes 5 indicators, gives each a weight | pvlib single-diode model + entropy weighting (50 % entropy + 50 % equal) |
| 4. Grading | Gives the health score and status | Mamdani fuzzy logic, 7 rules, centroid |

### The 5 indicators

| Indicator | Shows |
|---|---|
| power_ratio | any power loss |
| voltage_ratio | short circuit |
| current_ratio | open circuit |
| temp_rise | shading / hot spot |
| dust_level | dust |

`1` = as good as a healthy panel.

### Why fuzzy logic

A weighted score lets good indicators hide a bad one.
Fuzzy rules do not. Example rule: **if the electrical part is bad, the panel is faulty.**

---

## 4. Folder structure

```
component1-sensing-grading/
├── data/
│   ├── member1_raw_sensor_stream_SITE-*.csv   (6 sites)
│   ├── member1_sensor_health_SITE-*.csv       (labels, only to CHECK the grade)
│   └── device/                                (real logs from my SD card)
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_data_quality.ipynb       Stage 1
│   ├── 03_kalman_filter.ipynb      Stage 2
│   ├── 04_entropy_weighting.ipynb  Stage 3
│   ├── 05_send_to_backend.ipynb
│   ├── 06_fuzzy_logic.ipynb        Stage 4
│   └── 07_device_data.ipynb        real device data + ESP32 vs Python
├── src/
│   ├── load_data.py
│   ├── data_quality.py
│   ├── kalman_filter.py
│   ├── panel_model.py
│   ├── entropy_weighting.py
│   ├── fuzzy_logic.py
│   └── backend_writer.py
├── firmware/
│   └── pv_sensor_node/
│       ├── pv_sensor_node.ino      full pipeline on the ESP32
│       └── secrets.h               NOT in git (WiFi, backend URL, API key)
└── requirements.txt
```

Every notebook checks that its code gives the **same result** as the file in `src/` (`True`).

---

## 5. Results

### Site datasets (6 sites, 196,055 readings after cleaning)

| | Correct |
|---|---|
| Weighted score only | 98.3 % |
| **Fuzzy logic** | **99.3 %** |

| Fault | Weighted score | Fuzzy logic |
|---|---|---|
| Short circuit found as FAULTY | 0 | **335** |
| Open circuit found as FAULTY | 100 | **706** |
| Dust called HEALTHY (wrong) | 556 | **19** |

Other numbers:
- Stage 1: 782 duplicates removed, 337 shorted-load readings found
- Stage 2: Kalman removes 51 - 77 % of the noise, fills 1,754 missing temperatures

### Real device data (notebook 07)

- `log_003.csv`: 1000 readings (33 min). Relay ON -> current 0.0 mA.
  Kalman removes 65 - 72 % of the real sensor noise.
- `log_030.csv`: the same raw readings through Python and on the ESP32 give
  **the same S, the same health score and the same status** (relay ON, 534 lux -> 15.3 FAULTY).

---

## 6. Firmware (ESP32)

`firmware/pv_sensor_node/pv_sensor_node.ino` runs all 4 stages every 2 seconds.

- pvlib cannot run on the ESP32, so the physics model is a **lookup table**
  (19 light x 6 temperature points, max error 1.9 %).
- The weights are fixed from the site datasets:
  power 0.237, voltage 0.175, current 0.166, temp 0.309, dust 0.113.
- **SD card** keeps the RAW readings (`/log_001.csv`, `/log_002.csv` ...).
- **Backend** gets the filtered values + `quality_flag` + `health_score` + `health_status`,
  in batches of 15 (every 30 s), with a UTC time from NTP.

Serial Monitor commands:

| Key | Action |
|---|---|
| `1` | Relay ON (cut the panel) |
| `0` | Relay OFF |
| `t` | Test mode: grade below 500 lux too (indoor demo), flagged `TEST_MODE` |

### secrets.h (create it yourself, never commit it)

```cpp
#pragma once
#define WIFI_SSID     "your-wifi"
#define WIFI_PASSWORD "your-password"
#define BACKEND_URL   "http://<backend-ip>:8000"
#define API_KEY       "your-api-key"
#define SITE_ID       "SITE-A"
#define DEVICE_ID     "ESP32-A01"
```

---

## 7. How to run

```bash
python3.12 -m venv ~/venvs/j26-component1
source ~/venvs/j26-component1/bin/activate
pip install -r requirements.txt
```

Open the notebooks in order (01 -> 07) and use **Restart & Run All**.

To send to the backend (notebook 05), add a `.env` file (not in git):

```
BACKEND_URL=http://localhost:8000
BACKEND_API_KEY=your-api-key
```

---

## 8. Limitations

- **Voltage is calculated (V = I x R).** A short circuit in low light looks like a normal panel.
- **The light sensor (TEMT6000) stops at about 650 lux**, so outdoor grading on the device is not reliable yet.
- **The dust sensor (DSM501B) gives unstable readings** (from 10 % to 90 %).
- The panel's Voc and Isc are **assumed** (6 V, 220 mA); only 5 V / 200 mA is on the label.
- On the device I tested only an **open circuit** (relay) and **high dust**.
  Shading and short circuits are tested on the datasets.

## 9. Next steps

- Add the ground wire, so the device measures the real voltage
- Replace the light sensor (e.g. BH1750) and check the dust sensor
- Test the device outdoors, with real shading and dust
- Measure the panel's real Voc and Isc for the physics model
# IoT-Enabled Intelligent Solar PV Monitoring and Optimisation System

**Group J26-IT-348 · IT4010 Research Project · SLIIT (2026)**

**Repository:** [https://github.com/Lisura123/J26-IT-348](https://github.com/Lisura123/J26-IT-348)

An end-to-end cyber-physical framework that monitors small solar PV installations from the sensor up to the operator's screen. Low-cost ESP32 nodes grade panel health and classify faults on the device, a protocol-aware federated learning engine keeps the on-device model up to date across sites without sharing raw data, and a physics-informed digital twin explains every loss on a role-based dashboard.

> Industry partner: **Neth Digital (Pvt) Ltd.**, which provides solar monitoring data for the project.

---

## Table of contents

1. [System overview](#1-system-overview)
2. [Research components](#2-research-components)
3. [Branches](#3-branches)
4. [Architecture and data flow](#4-architecture-and-data-flow)
5. [Shared database](#5-shared-database)
6. [Repository layout](#6-repository-layout)
7. [Getting started](#7-getting-started)
8. [Pilot sites](#8-pilot-sites)
9. [Technology stack](#9-technology-stack)
10. [Team](#10-team)

---

## 1. System overview

Low power on a PV panel can mean a fault (dust, shading, open circuit, short circuit) or simply a cloudy afternoon. Power alone cannot tell the two apart. This project answers that question at four levels:

| Question | Answered by |
|---|---|
| *Is this panel healthy, and can the sensor data be trusted?* | Component 1, on the ESP32 |
| *If something is wrong, what kind of fault is it?* | Component 2, TinyML on the ESP32 |
| *How do all sites learn from each other over weak WiFi, LTE and LoRa links?* | Component 3, federated learning and backend |
| *What should the panel be producing right now, why is it not, and what should we do?* | Component 4, digital twin and dashboard |

---

## 2. Research components

| # | Component | Core techniques | Branch |
|---|---|---|---|
| 1 | **Adaptive Multi-Sensor PV Panel Health Grading** | Data quality rules, 1D Kalman filtering, pvlib single-diode reference, entropy weighting, Mamdani fuzzy logic (health score 0 to 100) | [`component1-sensing-grading`](https://github.com/Lisura123/J26-IT-348/tree/component1-sensing-grading) |
| 2 | **PV-Specific TinyML Fault Detection** | 8-16-5 MLP on physics features, INT8 quantisation, fault confirmation, relay isolation, ESP-IDF component | [`component2-tinyml`](https://github.com/Lisura123/J26-IT-348/tree/component2-tinyml) |
| 3 | **Protocol-Aware Federated Learning and Backend** | Flower FedAvg baseline, per-protocol compression (float32 / int8 / top-k), reliability-aware scheduling, asynchronous staleness and fairness weighting | [`component3-federated-backend`](https://github.com/Lisura123/J26-IT-348/tree/component3-federated-backend) |
| 4 | **Physics-Informed Digital Twin and Explainable Dashboard** | Faiman thermal model, De Soto single-diode model, split conformal bands, SHAP explanations, rule-based maintenance, FastAPI + Angular role-based dashboard | [`component4-digital-twin-dashboard`](https://github.com/Lisura123/J26-IT-348/tree/component4-digital-twin-dashboard) |

Each component folder has its own detailed README with methods, results, limitations and run instructions.

### Fault classes (shared by all components)

`NORMAL` · `PARTIAL_SHADING` · `DUST` · `OPEN_CIRCUIT` · `SHORT_CIRCUIT`

### Health status (Component 1)

| Status | Health score |
|---|---|
| `HEALTHY` | 70 to 100 |
| `DEGRADED` | 40 to 70 |
| `FAULTY` | 0 to 40 |
| `NOT_GRADED` | too dark to judge (below 500 lux) or sensor warming up |

---

## 3. Branches

Each member develops on a dedicated branch. `main` holds the integrated, reviewed state of the project.

| Branch | Purpose |
|---|---|
| `main` | Stable, integrated version of the project |
| `develop` | Integration branch where components are merged and tested together |
| `component1-sensing-grading` | Member 1: sensing, data quality and health grading (Python + ESP32 firmware) |
| `component2-tinyml` | Member 2: TinyML training, quantisation and ESP32 runtime |
| `component3-federated-backend` | Member 3: federated learning engine and MongoDB results |
| `component4-digital-twin-dashboard` | Member 4: digital twin, explainability, API and dashboard |
| `feature/project-backend` | Shared ingestion backend (MQTT + HTTP → MongoDB Atlas) |
| `feature/mongodb-migration` | Migration from InfluxDB/Docker to MongoDB Atlas (merged) |
| `integration/member2-member3-fl` | Integration of the TinyML model with the federated learning engine |

Working rules:

- Work only inside your own component folder on your own branch.
- Open a pull request into `develop`; `develop` is merged into `main` after integration tests pass.
- Never commit `.env`, `secrets.h`, credentials or large raw datasets.

---

## 4. Architecture and data flow

```text
  EDGE  (one ESP32 node per site)
 +----------------------------------------------+
 | Sensors                                      |
 |   INA219, TEMT6000 + ADS1115,                |
 |   2 x DS18B20, DSM501B                       |
 +----------------------------------------------+
                        |
                        v
 +----------------------------------------------+
 | C1  Health grading                           |
 |   data quality -> Kalman -> entropy weights  |
 |   -> fuzzy grade (0 to 100)                  |
 +----------------------------------------------+
                        |
                        v
 +----------------------------------------------+
 | C2  TinyML INT8 MLP                          |<-------------+
 |   fault class + confirmation                 |              |
 +----------+-----------------------+-----------+              |
            |                       |                          |
            v                       |                          |
 +-----------------------+          |                          |
 | Relay  (isolation)    |          |                          |
 +-----------------------+          |                          |
                                    |                          |
  LINKS               WiFi / LTE / LoRa                        | global model
                                    |                          | (OTA update)
  BACKEND                           v                          |
 +----------------------------------------------+              |
 | Mosquitto MQTT + FastAPI ingest              |              |
 |   (data contract validation)                 |              |
 +----------------------------------------------+              |
                        |                                      |
                        v                                      |
 +----------------------------------------------+      +-------+-------------+
 | MongoDB Atlas  (database: solar_pv)          |<---->| C3  Federated       |
 |   sensor_readings, sensor_health,            |      |   learning engine   |
 |   fault_predictions, fl_*, twin_output       |      |   compression,      |
 +----------------------------------------------+      |   scheduler, async  |
                        |           ^                  |   aggregation       |
                        |           |  twin_output     +---------------------+
  OPERATIONS            v           |
 +----------------------------------------------+
 | C4  Digital twin                             |
 |   Faiman + single diode + conformal band     |
 +----------------------------------------------+
                        |
                        v
 +----------------------------------------------+
 | SHAP + physics cause + maintenance rules     |
 +----------------------------------------------+
                        |
                        v
 +----------------------------------------------+
 | Angular dashboard                            |
 |   Admin  /  Engineer  /  Manager             |
 +----------------------------------------------+
```

1. **Component 1** reads the sensors every 2 seconds, cleans and grades each reading, and sends batches to the backend (`sensor_readings`, `sensor_health`).
2. **Component 2** classifies each graded reading on the device and can isolate the panel through the relay (`fault_predictions`).
3. The **project backend** validates every message against a shared data contract and upserts it into MongoDB Atlas.
4. **Component 3** trains the shared fault classifier across sites by federated learning and publishes versioned global models (`fl_global_models`, `fl_client_updates`, `fl_site_state`).
5. **Component 4** computes the expected output for every reading, explains any gap with physics and SHAP, raises prioritised alerts, and serves everything to the dashboard (`twin_output`).

---

## 5. Shared database

All components share one **MongoDB Atlas** database, `solar_pv`. InfluxDB and Docker are no longer used (the git tag `pp1` keeps the old version).

| Collection | Owner |
|---|---|
| `sensor_readings`, `sensor_health` | Member 1 |
| `fault_predictions` | Member 2 |
| `network_link_metrics`, `fl_client_updates`, `fl_global_models`, `fl_site_state` | Member 3 |
| `twin_output`, `dashboard_users` | Member 4 |

Two least-privilege accounts are used: a **writer** (`readWrite`) for ingestion and a **reader** (`read`) for the dashboard. Full details, indexes and Atlas limits are in [`docs/DATABASE.md`](docs/DATABASE.md). Credentials are shared privately by Member 3 and must never be committed.

---

## 6. Repository layout

```
J26-IT-348/
├── component1-sensing-grading/        Member 1 (see branch)
├── component2-tinyml/                 Member 2 (see branch)
├── component3-federated-backend/      Member 3 (see branch)
├── component4-digital-twin-dashboard/ Member 4 (see branch)
├── project-backend/                   Shared ingestion and query service
└── docs/
    ├── README.md
    └── DATABASE.md                    Shared MongoDB Atlas design
```

---

## 7. Getting started

```bash
git clone https://github.com/Lisura123/J26-IT-348.git
cd J26-IT-348

# switch to the component you want to work on
git checkout component4-digital-twin-dashboard
```

Each component is an independent Python project with its own `requirements.txt` (Python 3.12 recommended):

```bash
cd <component-folder>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # where provided; fill in your own values
```

Then follow the **How to run** section in that component's README.

| Service | Default port |
|---|---|
| Project backend (FastAPI ingest + query) | `8000` |
| Component 4 digital twin API | `8001` |
| Component 4 Angular dashboard | `4200` |
| Mosquitto MQTT broker | `1883` |

---

## 8. Pilot sites

Six pilot sites around Colombo (Asia/Colombo time zone), each with one ESP32 node and one mini PV panel, deliberately mixing link types so that the federated learning component faces realistic conditions:

| Site | Device | Protocol | Link quality |
|---|---|---|---|
| SITE-A | ESP32-A01 | WiFi | strong |
| SITE-B | ESP32-B01 | WiFi | strong |
| SITE-C | ESP32-C01 | LoRa | medium |
| SITE-D | ESP32-D01 | LTE | variable |
| SITE-E | ESP32-E01 | LoRa | weak |
| SITE-F | ESP32-F01 | WiFi | weak |

> Coordinates in `sites.csv` are placeholders. Most site datasets are synthetic; real device logs are included where noted in each component.

---

## 9. Technology stack

| Layer | Technologies |
|---|---|
| Edge hardware | ESP32, INA219, ADS1115, TEMT6000, DS18B20, DSM501B, relay module, SD card |
| Firmware | Arduino (Component 1), ESP-IDF / C (Component 2) |
| Data and ML | Python, pandas, NumPy, SciPy, scikit-learn, pvlib, SHAP, Flower |
| Messaging | MQTT (Mosquitto), HTTP batch ingest |
| Storage | MongoDB Atlas |
| APIs | FastAPI, JWT, bcrypt |
| Dashboard | Angular 21, Chart.js |
| Experiments | Jupyter, Google Colab |

---

## 10. Team

| Member | Component |
|---|---|
| Member 1 | Adaptive Multi-Sensor PV Panel Health Grading |
| Member 2 | PV-Specific TinyML Fault Detection |
| Member 3 | Protocol-Aware Federated Learning and Backend |
| Member 4 | Physics-Informed Digital Twin and Explainable Dashboard |

**Module:** IT4010 Research Project · Faculty of Computing, Sri Lanka Institute of Information Technology (SLIIT)

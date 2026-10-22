# Component 4 – Physics-Informed Digital Twin & Explainable Dashboard

**IT4010 Research Project · Group J26 IT 348 · Member 4 – Lisura Sigera**

This is my part of our group's solar PV monitoring framework. I built a digital twin that predicts what a *clean, healthy* mini solar panel should produce under the measured conditions, compares that with what the panel actually produces, and turns the difference into a fault cause, a plain-language explanation and a maintenance action. The results are shown on a role-based dashboard and written into our group backend (MongoDB Atlas).

> **Status (PP1, 22 Oct 2026):** the twin, the evaluation, the explanations, the dashboard and the backend integration all work end to end. Field data from Neth Digital is not available yet, so the evaluation below uses our group's development dataset (simulated with known ground truth, 16 Mar – 15 Sept 2026).

---

## How it works

```
Light sensor (TEMT6000) ─► lux ÷ 120 = W/m²  (weather-API GHI if the sensor saturates)
Air probe (DS18B20)  ─┐
Wind (Open-Meteo)    ─┼─► Faiman thermal model ─► expected panel temperature
                      │                                   │
                      └──────────────► De Soto single-diode model (pvlib)
                                                          │
                                   expected power / voltage / current (MPP)
                                                          │
 Actual V, I, P (INA219) ─► residuals ─► 90% conformal band ─► physics fault rules
                                                          │
                     plain explanation · SHAP (TinyML) · episodes ─► alerts R01–R05
                                                          │
                     FastAPI (roles, JWT) ─► Angular dashboard · MongoDB (group backend)
```

| Step | Method | What it does |
| --- | --- | --- |
| Sunlight | lux ÷ 120 | Converts the light sensor to W/m². When the sensor saturates in direct sun, it falls back to Open-Meteo irradiance. |
| Panel temperature | Faiman model, fitted to our panel | T_panel = T_air + G / (U0 + U1 · wind). My fit: U0 = 19.1, U1 = 4.04. |
| Expected power | De Soto single-diode model (pvlib) | Uses the panel datasheet to predict Pmp, Vmp, Imp, Voc and Isc. |
| Normal range | Split conformal prediction (α = 0.1) | A band around the expected power that should contain 9 out of 10 healthy readings (88% – 111% of expected). |
| Fault cause | Physics rules on residuals | Short circuit → open circuit → dust → partial shading → unexplained loss. |
| ML explanation | SHAP KernelExplainer on Member 2's TinyML model | Shows which sensor readings pushed the AI towards its prediction. |
| Alerts | Episodes + maintenance rules R01–R05 | Groups consecutive readings with the same cause into one alert. |
| Device health | Quality-flag decoder | Turns device error codes into "what happened / what to check / effect on the twin". |

---

## Results so far

| Check | Result |
| --- | --- |
| Power prediction, healthy readings | MAPE **5.8%**, RMSE 15.4 mW, R² 0.985 |
| Same model with the true inputs (ablation) | RMSE **0.02 mW** → the remaining error comes from the low-cost sensor chain |
| Panel temperature | RMSE **1.47 °C** with wind · 2.38 °C without wind · 4.70 °C with pvlib defaults (wind = 38% better) |
| Conformal band coverage (target 90%) | **89.8%** random split · **92.0%** temporal split (Mar–Jun → Jul–Sep) |
| Fault-cause accuracy (operation window) | **99.2%** (in-sample, thresholds set on this window) |
| One panel probe only (Member 1's hardware) | Still **99.2%**, using a voltage-drop fallback for shading |
| Alerts | 3,000 readings → **63 alerts** |
| SHAP vs Member 2's reference | Median row correlation **0.89**, sign agreement 85%, additivity error 0 |
| Backend integration | **3,000 / 3,000** records accepted by the group ingest API |

---

## Dashboard

Angular 21 (standalone components, signals) talking to my FastAPI service.

| Page | What it shows | Who sees it |
| --- | --- | --- |
| Sites | Status of all six panels (shape + text, not colour only) | Everyone |
| Site detail | Actual vs expected power with the 90% band; points outside the band | Everyone |
| Alerts | Problem, power lost, "what to do" in plain language | Everyone |
| Alert detail | What happened, what to do, technical details on demand, SHAP chart | SHAP: engineer and admin only |
| Device health | Live device data from MongoDB, decoded into explanations | Engineer and admin only |
| Reports | Results as plain questions and answers, technical metrics behind a toggle | All roles with `view_reports` |

Permissions come from `roles_permissions.csv` and are checked **on the server**. For example, a manager calling the SHAP or device-health endpoint directly gets `403 Forbidden`.

---

## Folder structure

```
component4-digital-twin-dashboard/
├── data/                 group input files (readings, weather, datasheet, roles, users, ...)
├── src/twin/             the twin as a Python package
│   ├── config.py         paths, constants and rule thresholds
│   ├── inputs.py         lux → W/m², weather interpolation, probe cleaning
│   ├── thermal.py        Faiman model + fitting
│   ├── electrical.py     De Soto single-diode operating point
│   ├── core.py           run_twin()
│   ├── conformal.py      split conformal band
│   ├── causes.py         residuals, fault rules, technical + plain explanations
│   ├── tinyml_model.py   Member 2's TinyML model (for SHAP)
│   ├── explain.py        SHAP explainer
│   ├── recommend.py      episodes and rules R01–R05
│   ├── publish.py        sends twin output to the group backend
│   └── device_health.py  quality-flag decoder
├── scripts/              numbered steps (00–16), one commit each
├── outputs/              results written by the scripts
├── api/                  FastAPI service (port 8001)
│   ├── main.py           endpoints
│   ├── auth.py           JWT login, bcrypt check, permission guard
│   └── data_source.py    the only place that reads files / MongoDB
├── dashboard/            Angular app (port 4200)
├── run_pipeline.ps1      rebuilds all outputs in order
└── requirements.txt
```

### Scripts

| Script | Purpose |
| --- | --- |
| `00_explore_data.py` | First look at the datasets |
| `01_check_inputs.py` | Checks the irradiance, wind and probe inputs |
| `02_calibrate_thermal.py` | Fits the Faiman model → `twin_calibration.json` |
| `03_check_twin.py` | Checks the electrical model against the datasheet (555 mW at STC) |
| `04_validate_twin.py` | Scores the twin against the ground truth → `validation_metrics.json` |
| `05_conformal_band.py` | Builds and tests the 90% band → `conformal_band.json` |
| `06_run_operation.py` | Runs the twin on the operation window → `digital_twin_output.csv` |
| `07_check_degradation.py` | Explains why some NORMAL readings fall outside the band (SITE-C/D degradation) |
| `08_shap_tinyml.py` | SHAP for the TinyML model → `shap_values.csv` |
| `09_recommendations.py` | Episodes and alerts → `alerts.csv` |
| `10_check_api.py`, `11_check_endpoints.py` | Login and permission tests for the API |
| `12_publish_to_backend.py` | Publishes the twin output to the group backend |
| `13_check_single_probe.py` | Tests the shading fallback with one panel probe |
| `14_seed_users.py` | Adds dashboard users to MongoDB with bcrypt password hashes |
| `15_inspect_health.py`, `16_check_device_health.py` | Inspect and decode device health data |

---

## How to run

### 1. Python side

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Rebuild all outputs (optional – the results are already in `outputs/`):

```powershell
.\run_pipeline.ps1
```

### 2. Environment file

Create `.env` in this folder. **Never commit it** (it is in `.gitignore`).

```
BACKEND_URL=http://localhost:8000
BACKEND_API_KEY=<ingest key from Member 3>
MONGO_READER_URI=mongodb+srv://<read-only user>:<password>@<cluster>/?readPreference=secondaryPreferred
MONGO_DB=solar_pv
HEALTH_COLLECTION=device_sensor_health
```

Without `MONGO_READER_URI`, the API falls back to the CSV users (demo password) and the device-health page is unavailable.

### 3. Start the services

| Terminal | Folder | Command |
| --- | --- | --- |
| Twin API | `component4-digital-twin-dashboard` | `uvicorn api.main:app --port 8001` |
| Dashboard | `component4-digital-twin-dashboard/dashboard` | `npm install` (first time) → `npm start` |
| Group backend (for publishing) | `project-backend` | `python -m uvicorn backend.api.main:app --port 8000` |

Open http://localhost:4200.

---

## Hardware notes

- **TEMT6000** saturates in direct sunlight. The twin then switches to Open-Meteo irradiance and records which source it used.
- Member 1's device has **two DS18B20 probes: one in the air, one on the panel**. With only one panel probe, the shading rule cannot use a temperature difference, so it falls back to the voltage drop. `13_check_single_probe.py` shows this keeps the fault accuracy at 99.2%.
- The live device data uses `dust_lpo_pct` and health statuses `HEALTHY / DEGRADED / FAULTY / NOT_GRADED`. The device-health page reads these as they are.

## Security

- JWT login; every endpoint declares the permission it needs.
- Dashboard passwords are stored as **bcrypt** hashes in MongoDB, never as plain text.
- The dashboard API uses a **read-only** database user. Only the seed script writes users.
- Secrets live in `.env` files that are not committed.

## Limitations

- The evaluation uses simulated development data. The real test will be field data from Neth Digital.
- The 99.2% fault accuracy is in-sample; the thresholds still need an out-of-sample check.
- Band coverage is lower at SITE-E (81%), so one band for all sites is not enough.
- The dashboard reads the twin results from files; only users and device health come live from MongoDB.


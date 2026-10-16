# Component 3 - Federated Learning and backend

## Member 3: protocol-aware federated learning (PP1)

Python research code. It trains a shared 8-16-5 MLP fault classifier across six simulated solar sites
without sharing raw readings, over WiFi, LTE and LoRa links replayed from the shared network traces.

### Architecture

```mermaid
flowchart LR
  subgraph S["Six solar sites (ESP32 clients)"]
    A["Local readings (TRAINING window)"] --> B["Local training: 8-16-5 MLP"]
  end
  K["Reliability-aware scheduler: history, link quality, fairness"] --> S
  B --> C["Protocol-aware compression: WiFi float32, LTE int8, LoRa top-k + int8, error feedback"]
  C --> D["Network replay: outages, duty cycle, fragments, retries"]
  D --> E["Backend aggregation: weight = samples x staleness x fairness, late updates kept"]
  E --> G["Global model JSON in models/"]
  E --> H[("MongoDB Atlas solar_pv: fl_global_models, fl_client_updates, fl_site_state")]
  H --> I["Members 2 and 4, dashboard"]
```

### Implemented (guide steps 1 to 10)

| Step | Files |
|---|---|
| 1 Data loader | `src/data_loader.py` |
| 2 MongoDB Atlas schema and loader | `src/mongo_client.py`, `src/load_to_mongo.py`, `src/verify_mongo.py` |
| 3 FedAvg rebuilt from shared files | `src/fedavg_baseline.py` |
| 4 Network replay, validated against the shared trace | `src/network_replay.py`, `src/validate_replay.py` |
| 5 FedAvg baseline (Flower `NumPyClient` sites) | `src/fl_baseline.py`, `src/mlp.py`, `src/pv_data.py` |
| 6 Protocol-aware compression | `src/compression.py` |
| 7 Reliability-aware scheduler | `src/scheduler.py` |
| 8 Asynchronous aggregation, staleness and fairness weights | `src/engine.py` |
| 9 Scenarios, evaluation, ablation | `src/scenarios.py`, `src/evaluate.py`, `src/paired.py` |
| 10 Results in the shared format and MongoDB Atlas | `src/export_results.py` |

### Results (5 seeds, mean ± standard deviation)
| | Normal trace: baseline | Normal trace: full | Harsh: baseline | Harsh: full |
|---|---|---|---|---|
| Final accuracy | 0.9423 ± 0.0193 | 0.9369 ± 0.0239 | 0.9257 ± 0.0392 | 0.9177 ± 0.0316 |
| LoRa transfer time (s) | 11,119 | 7,842 | 17,353 ± 549 | 5,229 ± 617 |
| Total bytes | 35,640 | 74,084 | 35,640 | 49,520 ± 3,077 |
| Updates aggregated | 114 | 112 | 96.4 ± 1.1 | 74.4 ± 5.9 |

The method cuts LoRa transfer time by 29% (normal) to 70% (harsh). Accuracy differences are within
seed-to-seed noise. Total bytes rise because WiFi sites send float32, and the scheduler skips sites that
are offline when a round opens, so fewer updates are aggregated under harsh conditions.

### Reproduce

InfluxDB and Docker are no longer used (the git tag pp1 has the old version).

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # fill in the values
python -m src.load_to_mongo
python -m src.verify_mongo
python -m src.fedavg_baseline
python -m src.validate_replay
python -m src.evaluate --scenario trace --seeds 5
python -m src.evaluate --scenario harsh --seeds 5 --ablation
python -m src.paired --scenario harsh
python -m src.make_charts
python -m src.export_results --config full --scenario trace --mongo
```

### Honest notes

- All data is synthetic (see `data/shared/DATA_CONTRACT.md`).
- The shared FL files (`fl_client_updates.csv`, weights, scaler) were generated from a different dataset than
  `pv_fault_dataset.csv` (different sample counts and feature scales). They are used for the network traces and
  file formats. The models here are trained from scratch on `pv_fault_dataset.csv` with a fresh scaler.
- The harsh scenario is a design choice (4 extra outages per site per day, packet loss x3, 80% of LoRa uploads at
  SF10 or worse, 2.5 h deadline), not a measurement of real sites.
- Flower is used through its `NumPyClient` interface in a deterministic round loop, not its simulation engine.
- WiFi and LTE radio energy is an estimate. OTA delivery to real devices and hardware tests are not done yet.

# 5-Minute Demo Guide: Member 3 (Federated Learning & Protocol Adaptation)

Demonstrates Member 3's protocol-aware federated learning engine, live round-by-round export to **MongoDB Atlas**, and visualization in **MongoDB Compass**.

---

## Before-You-Start Checklist

Complete this quick checklist in PowerShell before presenting:

1. **Atlas IP Allow-List**: Check your public IPv4 address and ensure it is active in the Atlas Network Access list:
   ```powershell
   Invoke-RestMethod https://api.ipify.org
   ```
2. **Environment Variable Setup**: Confirm that `.env` contains `MONGO_URI` and `MONGO_DB` (variable names only; never expose credentials):
   ```powershell
   Get-Content .env | Select-String "MONGO_URI", "MONGO_DB"
   ```
3. **MongoDB Compass Connected**:
   - Open MongoDB Compass and connect to your Atlas cluster (`<MONGO_URI>`).
   - Navigate to the `solar_pv` database in the left pane.
4. **Pre-Generated Charts & Models**: Ensure evaluation and chart artifacts are ready to show:
   ```powershell
   python -m src.make_charts
   ```
   Verify that PNG figures exist in `results/figures/` (e.g. `results/figures/fig1_learning_curves.png`).

---

## 5-Minute Demo Walkthrough

### 1. Show the Research Artifacts & Charts (1 minute)
Open and present the comparative performance figures generated in `results/figures/`:
- **Learning Curves (`fig1_learning_curves.png`)**: Shows convergence across baseline and protocol-aware schemes.
- **LoRa Transfer Time & Duty Cycle (`fig2_lora_transfer.png`)**: Highlights that protocol-aware top-k compression and scheduling reduce LoRa on-air transfer time by 29% (normal trace) up to 70% (harsh trace) while maintaining target classification accuracy.

---

### 2. Live Round-by-Round Export to Atlas (2.5 minutes)
Run a live federated learning trace export that trains deterministic rounds and writes global models, client updates, and site states directly to MongoDB Atlas with a 2-second delay between rounds to observe live ingestion:

```powershell
cd D:\Research-project\J26-IT-348\component3-federated-backend
.\.venv\Scripts\Activate.ps1

python -m src.export_results --config full --scenario trace --seed 11 --mongo --delay 2
```

#### Command Options Verified via `src/export_results.py`:
- `--config full`: Selects protocol-aware compression and fairness-weighted aggregation (`src/evaluate.py`).
- `--scenario trace`: Uses the realistic replay scenario derived from shared network link traces.
- `--seed 11`: Deterministic pseudo-random seed setting `run_id` to `member3-full-tanh-s11-trace`.
- `--mongo`: Flag enabling direct chunked upsert into MongoDB Atlas.
- `--delay 2`: Pauses 2.0 seconds between rounds to simulate live round completions.

During execution, note the terminal output showing progressive rounds (0 through 20) being published and written to Atlas.

---

### 3. Live Inspection in MongoDB Compass (1 minute)
Switch to MongoDB Compass to demonstrate the stored documents:
1. Select database: **`solar_pv`**.
2. Select collection: **`fl_global_models`**.
3. Apply filter:
   ```json
   {"run_id": "member3-full-tanh-s11-trace"}
   ```
4. Set Sort:
   ```json
   {"round_id": 1}
   ```
5. Click **Find**. Observe the 21 rounds (round 0 to 20):
   - `published_at_utc`: Stored as a BSON UTC date.
   - `global_val_accuracy`: Incremental model accuracy progression.
   - `model_path`: Points to the local weight artifact (e.g. `models/runs/member3-full-tanh-s11-trace/global_v20.json`), keeping raw weight tensors out of the database.
6. Briefly show collection **`fl_client_updates`** with filter `{"run_id": "member3-full-tanh-s11-trace"}`:
   - Each client update shows `status`, `payload_bytes`, `airtime_s`, `delta_l2_norm`, and `aggregation_weight`.

---

### 4. Verify Shared Baseline Consistency (30 seconds)
Run the verification tool to confirm the canonical shared collections remain intact:
```powershell
python -m src.verify_mongo
```
*Expected: All shared collections (`network_link_metrics`, `network_outages`, `fl_client_updates`, `fl_global_models`, `ota_deployments`) report `OK`.*

---

## Fallback Plan (If Atlas is Unreachable)

> **Fallback Statement for Evaluators:**
> *"If our cloud Atlas cluster is temporarily unreachable due to network connectivity or IP allow-list propagation delays, the entire federated learning simulation, model training, and export run offline deterministically without `--mongo`:
> `python -m src.export_results --config full --scenario trace --seed 11`
> All models and contract-compliant CSV traces (`fl_global_models.csv`, `fl_client_updates.csv`, `fl_site_state.csv`) are written locally to `results/export/member3-full-tanh-s11-trace/`."*

---

## Honest Research Limitations

1. **Synthetic Trace Data**: All sensor data and network link metrics are synthetic, designed to model realistic LoRa/LTE packet drops and daylight solar generation patterns.
2. **Model Weight Separation**: Model weights are intentionally kept out of the database and stored locally as JSON artifacts; only the file path is recorded in `fl_global_models`.
3. **Hardware Simulation**: Radio energy consumption and on-air duty cycle are computed from empirical analytical models; on-device physical OTA firmware flashing is simulated.

---

## Restore Afterwards

When the demo is concluded:
1. No background services were launched by this demo script.
2. Local artifacts generated in `results/export/member3-full-tanh-s11-trace` and `models/runs/member3-full-tanh-s11-trace` may be retained or cleaned up as needed.

# Handover: Member 3 (Operations & Federated Backend)

## 1. Key Management & Rotation Guide

### A. Backend API Keys Rotation
The backend uses two distinct API keys defined in `.env`:
- `API_KEY`: Ingestion key used by publishers and replay tools (`POST /api/v1/ingest/*`).
- `READ_API_KEY`: Read-only key used by dashboard consumers (`GET /api/v1/*`).

**Rotation Procedure:**
1. Generate new high-entropy random keys (e.g. using `python -c "import secrets; print(secrets.token_urlsafe(32))"`).
2. Update `.env` with the new keys.
3. Restart the API service:
   ```powershell
   # Stop existing service
   .\scripts\stop_dev.ps1
   # Start with new configuration
   .\scripts\start_dev.ps1
   ```
4. Privately distribute the updated `API_KEY` to Member 1 and `READ_API_KEY` to Member 4.

### B. MongoDB Atlas Database User Rotation
If a database user's password or connection string is compromised:
1. Log into MongoDB Atlas Console: `https://cloud.mongodb.com`.
2. Navigate to **Security** -> **Database Access**.
3. Locate the compromised user (e.g. `MONGO_APP_USER` or `MONGO_READER_USER`).
4. Click **Edit**, enter a new secure random password, and click **Update User**.
5. Update `MONGO_APP_PASSWORD` or `MONGO_READER_PASSWORD` (and `MONGO_URI` if credentials are embedded) in `.env`.
6. Restart backend ingestion services (`.\scripts\start_dev.ps1`).

---

## 2. MongoDB Atlas Configuration & Role Separation

### Least-Privilege Role Setup
- **Writer Account (`MONGO_APP_USER`)**: Assigned `readWrite` on database `solar_pv`.
- **Reader Account (`MONGO_READER_USER`)**: Assigned `read` on database `solar_pv`.
- **Administrative Account (`MONGO_ROOT_USER`)**: Assigned `dbAdmin` / `readWriteAnyDatabase`.

### Atlas Network Access (IP Access List)
1. In the Atlas Console, navigate to **Security** -> **Network Access**.
2. Click **Add IP Address**.
3. Choose **Add Current IP Address** for secure developer workstation access.
4. If testing across changing networks, configure temporary access with high-entropy passwords.

---

## 3. Switching from Test (`test_`) to Real Production Measurements

### Safety Guard Mechanism
The backend includes a hardcoded startup guard: if `MEASUREMENT_PREFIX` is empty, startup will fail with a `RuntimeError` unless `ALLOW_REAL_MEASUREMENTS=1` is explicitly set in the environment.

### Go-Live Production Switch Procedure
Only execute when:
1. Members 1, 2, and 4 have explicitly signed off on their schemas.
2. `python -m backend.tools.verify_backend --stream readings --file ..\component3-federated-backend\data\shared\pv_fault_dataset.csv --window OPERATION` returns `OK`.
3. All unit tests pass: `pytest -q`.

**Steps to activate production:**
1. Edit `.env`:
   ```bash
   MEASUREMENT_PREFIX=
   ALLOW_REAL_MEASUREMENTS=1
   ```
2. Restart backend ingestion and API services.
3. Live data will now be routed to `sensor_readings`, `sensor_health`, `fault_predictions`, and `twin_output`. Member 3's reserved measurements (`network_link_metrics`, `fl_client_updates`, etc.) remain completely protected by validator guardrails.

---

## 4. Cleaning Test Data (`OPTIONAL - VERIFY TARGET FIRST`)

> [!CAUTION]
> **VERIFY DATABASE & COLLECTION NAME BEFORE RUNNING CLEANUP!**
> Never run collection drops or deletes against production collections or Member 3's reserved collections (`network_*`, `fl_*`, `ota_*`).
> Always verify that the database is `solar_pv` and target collection begins with `test_` before executing.

If Member 3 needs to optionally purge development test collections (`test_sensor_*`) prior to go-live:

```javascript
// Using mongosh or MongoDB Compass:
use solar_pv;

// Verify existing collections first:
show collections;

// OPTIONAL: Drop only development test collections
db.test_sensor_readings.drop();
db.test_sensor_health.drop();
db.test_fault_predictions.drop();
db.test_twin_output.drop();
```

---

## 5. MongoDB Atlas Backup & Maintenance
- **Automated Cloud Backups**: MongoDB Atlas provides continuous cloud backup snapshots under the **Backup** tab.
- **On-Demand Dump (`mongodump`)**:
  ```powershell
  mongodump --uri="<YOUR_ATLAS_URI>" --db=solar_pv --out="./backup_$(Get-Date -Format 'yyyyMMdd')"
  ```

---

## 6. Go-Live Production Checklist

- [ ] All 97+ pytest suite tests passing (`pytest -q`).
- [ ] Schema confirmation signed off by Member 1 (Sensing) for health flags.
- [ ] Schema confirmation signed off by Member 2 (TinyML) for prediction classes.
- [ ] Schema confirmation signed off by Member 4 (Digital Twin) for twin output fields.
- [ ] `smoke_test` passes cleanly on both HTTP and MQTT transports.
- [ ] Native Mosquitto MQTT broker configured with password file / authentication for production.
- [ ] `MEASUREMENT_PREFIX` set to empty string and `ALLOW_REAL_MEASUREMENTS=1` enabled in production environment.
- [ ] Final verification run confirms Member 3's 6 reserved measurements remain intact (document count = 0 / untouched).

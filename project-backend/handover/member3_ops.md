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

### B. Leaked InfluxDB Token Emergency Rotation
If an `INFLUX_TOKEN` is accidentally compromised:
1. Log into InfluxDB UI: `http://localhost:8087`.
2. Navigate to **Load Data** -> **API Tokens**.
3. Create a **New All-Access Token** (or scoped bucket token).
4. Update `INFLUX_TOKEN` in `.env`.
5. In the UI, click on the old compromised token and select **Delete Token**.
6. Restart backend ingestion services.

---

## 2. Switching from Test (`test_`) to Real Production Measurements

### Safety Guard Mechanism
The backend includes a hardcoded startup guard: if `MEASUREMENT_PREFIX` is empty, startup will fail with a `RuntimeError` unless `ALLOW_REAL_MEASUREMENTS=1` is explicitly set in the environment.

### Go-Live Production Switch Procedure
Only execute when:
1. Members 1, 2, and 4 have explicitly signed off on their schemas.
2. `python -m backend.tools.verify_backend --stream readings --file ..\pv_fault_dataset.csv --window OPERATION` returns `OK`.
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

## 3. Cleaning Test Data (`DOCUMENTED ONLY - DO NOT EXECUTE LIGHTLY`)

> [!CAUTION]
> **DOUBLE-CHECK WARNING BEFORE RUNNING DELETE COMMANDS!**
> Never run `influx delete` without double-checking the `--predicate` filter.
> Never target Member 3's reserved measurements (`network_*`, `fl_*`, `ota_*`).
> Always verify bucket name and date range before pressing enter.

If Member 3 needs to purge development test measurements (`test_sensor_*`) prior to go-live:

```bash
# Example syntax using the InfluxDB CLI tool inside the container:
# PURGE ONLY PREFIXED TEST MEASUREMENTS:
influx delete \
  --bucket solar_pv \
  --start '1970-01-01T00:00:00Z' \
  --stop '2030-01-01T00:00:00Z' \
  --predicate '_measurement == "test_sensor_readings" or _measurement == "test_sensor_health"' \
  --token <ASK_MEMBER_3_PRIVATELY> \
  --org j26
```

---

## 4. InfluxDB Volume Backup & Maintenance Advice
- **Docker Volume Storage**: InfluxDB persistence is stored in the Docker named volume or bind mount associated with the database container.
- **Hot Backup via Influx CLI**:
  ```bash
  influx backup /backup/influx_$(date +%Y%m%d) \
    --token <ASK_MEMBER_3_PRIVATELY> \
    --org j26
  ```
- **File System Snapshot**: If using Docker volume mounts, ensure the host directory is included in regular automated filesystem snapshots.

---

## 5. Go-Live Production Checklist

- [ ] All 86+ pytest suite tests passing (`pytest -q`).
- [ ] Schema confirmation signed off by Member 1 (Sensing) for health flags.
- [ ] Schema confirmation signed off by Member 2 (TinyML) for prediction classes.
- [ ] Schema confirmation signed off by Member 4 (Digital Twin) for twin output fields.
- [ ] `smoke_test` passes cleanly on both HTTP and MQTT transports.
- [ ] Mosquitto MQTT broker configured with password file / authentication for production.
- [ ] `MEASUREMENT_PREFIX` set to empty string and `ALLOW_REAL_MEASUREMENTS=1` enabled in production environment.
- [ ] Final verification run confirms Member 3's 6 reserved measurements remain intact.

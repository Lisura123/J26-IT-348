# Stream Payload Examples

This directory provides canonical JSON payload examples for each stream in the platform:

| File | Stream | Mode | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `readings.json` | `readings` (`sensor_readings`) | Strict | **Verified (Production)** | Real row from `pv_fault_dataset.csv` with dataset metadata columns stripped. |
| `health.json` | `health` (`sensor_health`) | Open | **Demo Fixture** | Real row from dataset containing `health_score`. Member 1 (Sensing) must confirm final fields and quality flags. |
| `prediction.json` | `prediction` (`fault_predictions`) | Open | **Placeholder (Proposed)** | Model inference structure. Protected with `_comment` key to prevent accidental ingestion until confirmed by Member 2. |
| `twin.json` | `twin` (`twin_output`) | Open | **Placeholder (Proposed)** | Simulation output structure. Protected with `_comment` key to prevent accidental ingestion until confirmed by Member 4. |

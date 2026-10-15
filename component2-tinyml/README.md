# Component 2 - TinyML Fault Detection

## Member 2: PV-specific on-device TinyML fault detection

Classifies each PV reading on the ESP32 into one of five classes:
`NORMAL`, `PARTIAL_SHADING`, `DUST`, `OPEN_CIRCUIT`, `SHORT_CIRCUIT`.
Predictions go to the group backend (`fault_predictions` stream) and the model
must stay compatible with Member 3's federated learning and Member 4's digital twin.

### Status

| Phase | Status |
|---|---|
| 1 Component initialization, column contract | done |
| 2 Dataset validation | planned |
| 3 Dataset provenance documentation | planned |
| 4 Sensor feature and label verification | planned |
| 5 Run-wise data splitting | planned |
| Model training, quantization, ESP32 deployment | not started |

### Structure

```
component2-tinyml/
  src/tinyml/      Python package (contract.py: column roles and labels)
  tests/           pytest tests
  docs/            provenance and design notes (planned)
  scripts/         numbered pipeline scripts (planned)
```

### Setup

```bash
cd component2-tinyml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

### Data rules

- `health_score` is a label proxy and is never a model input.
- Splits are run-wise (`run_id`), never random row splits.
- `pv_fault_dataset.csv` is labelled as synthetic in the group data contract;
  its provenance is still being verified with Member 1.

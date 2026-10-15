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
| Architecture design (`docs/ARCHITECTURE.md`) | done |
| 2 Dataset validation (`src/tinyml/validate.py`, `results/dataset_validation.md`) | done |
| 3 Dataset provenance documentation (`docs/DATA_PROVENANCE.md`) | done (provenance open) |
| 4 Sensor feature and label verification | planned |
| 5 Run-wise data splitting (`src/tinyml/splits.py`, `results/split_summary.md`) | done |
| Model training, quantization, ESP32 deployment | not started |

### Structure

```
component2-tinyml/
  src/tinyml/      Python package (contract.py: column roles; validate.py: dataset checks)
  tests/           pytest tests
  docs/            architecture, provenance and design notes
  scripts/         numbered pipeline scripts
  results/         generated reports (small, committed)
  data/raw/        datasets, local only (git-ignored)
```

### Setup

```bash
cd component2-tinyml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest

# put pv_fault_dataset.csv (and optional pv_<label>.csv, sites.csv,
# member2_fault_predictions.csv) in data/raw/, then:
python scripts/01_validate_dataset.py --data-dir data/raw
python scripts/02_make_splits.py --data-dir data/raw   # run-wise split assignments
```

Training code (local or Colab) loads the saved splits instead of re-splitting:

```python
from tinyml.splits import load_assignments, iter_folds
assignments, meta = load_assignments("results/splits_runwise_v1.json", df, dataset_sha256=sha)
for name, train_idx, test_idx in iter_folds(df, assignments, "cv"):
    ...
```

### Data rules

- `health_score` is a label proxy and is never a model input.
- Splits are run-wise (`run_id`), never random row splits.
- `pv_fault_dataset.csv` is the PV sensor dataset provided by Member 1;
  its provenance is still being verified with Member 1.

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
| 4 Sensor feature and label verification | done (part of dataset validation) |
| 5 Run-wise data splitting (`src/tinyml/splits.py`, `results/split_summary.md`) | done |
| PV feature layer (`src/tinyml/features.py`, `docs/FEATURES.md`) | done; not yet evaluated |
| Google Colab environment check (`notebooks/colab_train.ipynb`, `results/colab_env_check.json`) | done; run in Colab, 91 tests passed |
| Phase 7 baseline comparison code (`experiment.py`, `docs/BASELINES.md`, `notebooks/colab_phase7_baselines.ipynb`) | code ready; experiments to be run in Colab |
| Baseline comparison results (`docs/PHASE7_RESULTS.md`) | done; physics8 + 8-16-5 tanh MLP selected |
| Final training and export code (`train.py`, `export.py`, `docs/TRAINING_EXPORT.md`, `notebooks/colab_final_training.ipynb`) | done |
| Final model trained in Colab (`models/`, run record in `docs/TRAINING_EXPORT.md` §10) | done; artifacts verified (commit `9a641d5`, seed 0, 2,924 TRAINING rows, weights SHA-256 `94e7ab93…`). Not yet tested on OPERATION |
| Independent OPERATION evaluation code and protocol (`holdout.py`, `docs/OPERATION_EVAL.md`, `notebooks/colab_operation_eval.ipynb`) | code ready; protocol fixed; OPERATION not yet evaluated |
| Quantization, ESP32 deployment, FL integration | not started |

Colab (scikit-learn 1.6.1) and local (1.9.1) gave identical splits and features. The
scikit-learn-based `health_score` leakage score differed (0.6945 in Colab vs 0.7457 locally).
The library version difference is the likely cause but has not been confirmed, so model
training results must record the library versions used.

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
python scripts/03_validate_features.py --data-dir data/raw   # feature check, TRAINING only
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

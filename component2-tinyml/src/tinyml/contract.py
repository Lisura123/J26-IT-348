"""Column contract for the PV fault dataset used by Component 2.

Single place that says which columns are sensor inputs, which are labels or
identifiers, and which must never reach a model. Sources:
- project-backend/schema.yaml (feature/project-backend): readings + prediction streams
- component3-federated-backend/src/pv_data.py: FL feature and label order
"""

# Canonical class names. Order matches Member 3's FL model output layer,
# so class index i means the same thing on the ESP32, in FL and in the twin.
LABELS = ("NORMAL", "PARTIAL_SHADING", "DUST", "OPEN_CIRCUIT", "SHORT_CIRCUIT")

# Numeric sensor readings sent by the device (backend `readings` stream).
SENSOR_COLUMNS = (
    "voltage_v", "current_ma", "power_mw",
    "panel_temp1_c", "panel_temp2_c", "ambient_temp_c",
    "humidity_pct", "pressure_hpa", "irradiance_lux", "dust_vo_v",
)

# The 8 raw inputs of the current FL model (Member 3, 8-16-5 MLP).
FL_RAW_FEATURES = (
    "voltage_v", "current_ma", "power_mw", "panel_temp1_c", "panel_temp2_c",
    "ambient_temp_c", "irradiance_lux", "dust_vo_v",
)

ID_COLUMNS = ("record_id", "timestamp_utc", "site_id", "device_id", "panel_id", "reading")
LABEL_COLUMN = "label"
GROUP_COLUMN = "run_id"      # one experiment run; splits must keep a run together
WINDOW_COLUMN = "window"     # TRAINING / OPERATION period from the data contract

# Columns that encode the answer. health_score alone gives ~75% run-wise
# accuracy and existed before Member 1's grading code, so it is a label proxy.
LEAKAGE_COLUMNS = ("health_score",)

NEVER_MODEL_INPUTS = (*ID_COLUMNS, LABEL_COLUMN, GROUP_COLUMN, WINDOW_COLUMN, *LEAKAGE_COLUMNS)


def classify_columns(columns):
    """Sort a CSV header into contract roles.

    Returns a dict with the sensor, id, meta and leakage columns that are
    present, plus `missing` (contract columns not in the header) and
    `unknown` (header columns the contract does not know about).
    """
    cols = list(columns)
    meta = (LABEL_COLUMN, GROUP_COLUMN, WINDOW_COLUMN)
    known = set(SENSOR_COLUMNS) | set(ID_COLUMNS) | set(meta) | set(LEAKAGE_COLUMNS)
    expected = (*SENSOR_COLUMNS, *ID_COLUMNS, *meta)
    return {
        "sensor": [c for c in cols if c in SENSOR_COLUMNS],
        "id": [c for c in cols if c in ID_COLUMNS],
        "meta": [c for c in cols if c in meta],
        "leakage": [c for c in cols if c in LEAKAGE_COLUMNS],
        "missing": [c for c in expected if c not in cols],
        "unknown": [c for c in cols if c not in known],
    }

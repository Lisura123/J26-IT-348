"""Read-only validation of the PV fault dataset (Phase 2).

Every check takes DataFrames and returns plain Python data. Nothing here
writes to or modifies input files. `validate_dataset` runs all checks and
splits the outcome into errors (dataset must not be used as is) and
warnings (usable, but must be reported).
"""
import hashlib
from pathlib import Path

import pandas as pd

from .contract import (
    GROUP_COLUMN, ID_COLUMNS, LABEL_COLUMN, LABELS, LEAKAGE_COLUMNS, SENSOR_COLUMNS,
    WINDOW_COLUMN, classify_columns,
)

EXPECTED_INTERVAL_S = 5.0
INTERVAL_TOLERANCE_S = 0.5
POWER_TOLERANCE_MW = 0.5            # rounding of V (3 dp) and I (2 dp) stays well below this
BH1750_MAX_LUX = 65535              # common ESP32 lux sensor; actual sensor model undocumented
LUX_SATURATION = 100000             # LUX_SATURATION used by Member 4 (config.py)
TEMP_SENTINELS = (-127.0, -999.0)   # DS18B20 error codes
UTC_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"

# (low, high, source). Values outside are reported, never changed.
PHYSICAL_LIMITS = {
    "voltage_v": (0.0, 4.0, "6-cell panel, Voc 3.6 V (panel_datasheet.csv, ASSUMED)"),
    "current_ma": (0.0, 250.0, "Isc 210 mA (panel_datasheet.csv, ASSUMED)"),
    "power_mw": (0.0, 700.0, "Pmax 555 mW (panel_datasheet.csv, ASSUMED)"),
    "panel_temp1_c": (-10.0, 85.0, "DS18B20 on an outdoor panel"),
    "panel_temp2_c": (-10.0, 85.0, "DS18B20 on an outdoor panel"),
    "ambient_temp_c": (10.0, 45.0, "Sri Lanka ambient"),
    "humidity_pct": (0.0, 100.0, "relative humidity"),
    "pressure_hpa": (950.0, 1050.0, "near sea level"),
    "irradiance_lux": (0.0, LUX_SATURATION, "LUX_SATURATION (component 4 config)"),
    "dust_vo_v": (0.0, 3.3, "ESP32 ADC full scale"),
}

# Prediction outputs (backend schema + Member 4 stand-in names).
PREDICTION_COLUMNS = ("predicted_label", "predicted_fault", "confidence", "prob_normal")

# Class code inside run_id, e.g. SITE-A-NOR-07 -> NOR.
RUN_CODE = {"NOR": "NORMAL", "PAR": "PARTIAL_SHADING", "DUS": "DUST",
            "OPE": "OPEN_CIRCUIT", "SHO": "SHORT_CIRCUIT"}

PART_FILES = {"pv_normal.csv": "NORMAL", "pv_partial_shading.csv": "PARTIAL_SHADING",
              "pv_dust.csv": "DUST", "pv_open_circuit.csv": "OPEN_CIRCUIT",
              "pv_short_circuit.csv": "SHORT_CIRCUIT"}

PROVENANCE_NOTE = (
    "PV sensor dataset provided by Member 1. Provenance is UNRESOLVED until firmware, "
    "raw logs and run records are provided."
)


# ---------------------------------------------------------------- helpers

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_for_part_file(path):
    """Map 'pv_dust.csv' (any prefix, e.g. upload ids) to its label, else None."""
    name = Path(path).name
    for suffix, label in PART_FILES.items():
        if name == suffix or name.endswith("-" + suffix) or name.endswith("_" + suffix):
            return label
    return None


def _times(df):
    return pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")


def _has(df, *cols):
    return all(c in df.columns for c in cols)


# ---------------------------------------------------------------- checks

# [GUIDE] A. Dataset validation: refuse model predictions as training labels
# WHAT: stops if the file has prediction columns (predicted_label, confidence, ...).
# WHY:  a model's own outputs are not ground truth; training on them would be circular.
# INPUT: a DataFrame.  OUTPUT: nothing, or a ValueError.
# EVIDENCE: results/dataset_validation.md (member2_fault_predictions.csv is reference only).
def assert_not_predictions(df, name="dataset"):
    """Refuse model-output files as ground truth for training."""
    found = [c for c in PREDICTION_COLUMNS if c in df.columns]
    if found:
        raise ValueError(f"{name} contains prediction columns {found}; "
                         "model predictions are never ground-truth labels")


def check_schema(df):
    roles = classify_columns(df.columns)
    non_numeric = [c for c in roles["sensor"] if not pd.api.types.is_numeric_dtype(df[c])]
    bad_ts = None
    if "timestamp_utc" in df:
        ts = df["timestamp_utc"].astype(str)
        bad_ts = int((~ts.str.match(UTC_PATTERN) | _times(df).isna()).sum())
    return {**roles, "rows": len(df), "non_numeric_sensor": non_numeric,
            "timestamps_not_utc_z": bad_ts}


# [GUIDE] A. Fault labels
# WHAT: counts rows per label and lists unknown labels and labels with no rows.
# WHY:  all 5 classes (NORMAL, PARTIAL_SHADING, DUST, OPEN_CIRCUIT, SHORT_CIRCUIT)
#       must exist with the exact names used on the ESP32 and in Member 3's FL.
# OUTPUT: {"counts", "unknown", "absent"}; validate_dataset() turns problems into errors.
# EVIDENCE: results/dataset_validation.md (1,200 rows per class).
def check_labels(df):
    counts = df[LABEL_COLUMN].value_counts().to_dict()
    return {"counts": {k: int(v) for k, v in counts.items()},
            "unknown": sorted(map(str, set(counts) - set(LABELS))),
            "absent": [l for l in LABELS if l not in counts]}


# [GUIDE] A. Missing values and sensor error codes
# WHAT: NaN count per column, plus DS18B20 error codes (-127 / -999) in the panel probes.
# WHY:  a missing or error reading must never be treated as a real measurement.
# OUTPUT: {"nan_by_column", "temp_sentinels"}; any NaN is an error in validate_dataset().
# EVIDENCE: results/dataset_validation.md (no missing values, no error codes).
def check_missing(df):
    na = df.isna().sum()
    sentinels = {c: int(df[c].isin(TEMP_SENTINELS).sum())
                 for c in ("panel_temp1_c", "panel_temp2_c") if c in df}
    return {"nan_by_column": {c: int(n) for c, n in na.items() if n},
            "temp_sentinels": {c: n for c, n in sentinels.items() if n}}


# [GUIDE] A. Duplicate records
# WHAT: duplicate record_id, duplicate (device_id, timestamp) pairs, and rows that are
#       identical except for record_id.
# WHY:  duplicates could appear in both train and test and inflate the scores.
# OUTPUT: counts per duplicate type (record_id / device-timestamp duplicates are errors).
# EVIDENCE: results/dataset_validation.md.
def check_duplicates(df):
    out = {}
    if "record_id" in df:
        out["record_id"] = int(df["record_id"].duplicated().sum())
    if _has(df, "device_id", "timestamp_utc"):
        out["device_timestamp"] = int(df.duplicated(["device_id", "timestamp_utc"]).sum())
    values = [c for c in df.columns if c != "record_id"]
    out["identical_rows_except_id"] = int(df.duplicated(values).sum())
    return out


def check_runs(df):
    g = df.groupby(GROUP_COLUMN)
    size = g.size()
    out = {
        "runs": int(len(size)),
        "runs_per_label": {k: int(v) for k, v in
                           df.groupby(LABEL_COLUMN)[GROUP_COLUMN].nunique().items()},
        "readings_per_run": {"min": int(size.min()), "max": int(size.max())},
        "mixed_label_runs": sorted(g[LABEL_COLUMN].nunique().loc[lambda s: s > 1].index),
    }
    for col in ("site_id", "device_id", WINDOW_COLUMN):
        if col in df:
            out[f"mixed_{col}_runs"] = sorted(g[col].nunique().loc[lambda s: s > 1].index)
    if "reading" in df:
        bad = g["reading"].apply(lambda s: sorted(s) != list(range(1, len(s) + 1)))
        out["bad_reading_sequence_runs"] = sorted(bad[bad].index)
    # run_id carries the class code: it must never be a model input
    code = df[GROUP_COLUMN].astype(str).str.extract(r"-([A-Z]{3})-\d+$")[0].map(RUN_CODE)
    out["run_id_encodes_label_share"] = round(float((code == df[LABEL_COLUMN]).mean()), 4)
    return out


def check_timestamps(df):
    """Order and spacing of readings inside each run."""
    d = df.assign(_t=_times(df))
    order_col = "reading" if "reading" in d else "_t"
    d = d.sort_values([GROUP_COLUMN, order_col])
    dt = d.groupby(GROUP_COLUMN)["_t"].diff().dt.total_seconds()
    dt = dt[d.groupby(GROUP_COLUMN).cumcount() > 0]
    off = (dt - EXPECTED_INTERVAL_S).abs() > INTERVAL_TOLERANCE_S
    non_increasing = d.loc[dt.index[dt <= 0], GROUP_COLUMN].unique()
    file_order_ok = bool(df.groupby(GROUP_COLUMN, sort=False)
                         .apply(lambda x: _times(x).is_monotonic_increasing).all())
    starts = d.groupby(GROUP_COLUMN)["_t"].min()
    return {
        "intervals": int(len(dt)),
        "intervals_off_expected": int(off.sum()),
        "interval_min_s": float(dt.min()) if len(dt) else None,
        "interval_max_s": float(dt.max()) if len(dt) else None,
        "non_increasing_runs": sorted(non_increasing),
        "file_order_monotonic_per_run": file_order_ok,
        "runs_starting_at_second_00": int((starts.dt.second == 0).sum()),
        "first_utc": str(d["_t"].min()), "last_utc": str(d["_t"].max()),
    }


def check_run_overlaps(df):
    """Run pairs whose time spans overlap. A single physical panel cannot run
    two experiments at once, so cross-site overlaps matter for provenance."""
    r = (df.assign(_t=_times(df)).groupby(GROUP_COLUMN)
         .agg(start=("_t", "min"), end=("_t", "max"), site=("site_id", "first"))
         .sort_values("start"))
    same, cross = [], []
    rows = list(r.itertuples())
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if b.start > a.end:
                break
            (same if a.site == b.site else cross).append((a.Index, b.Index))
    return {"same_device": same, "cross_site": cross}


# [GUIDE] A. Invalid sensor readings / physical ranges
# WHAT: min, max and out-of-range counts for every sensor, using PHYSICAL_LIMITS above.
# WHY:  values outside what the panel and sensors can produce point to sensor or logging faults.
# HOW:  limits come from an ASSUMED panel datasheet (see the PHYSICAL_LIMITS sources), so
#       out-of-range values are reported as warnings, never changed or deleted.
# EVIDENCE: results/dataset_validation.md (ranges table, lux saturation counts).
def check_ranges(df):
    out = {}
    for c, (lo, hi, src) in PHYSICAL_LIMITS.items():
        if c in df:
            s = df[c].replace(list(TEMP_SENTINELS), pd.NA).dropna() if "temp" in c else df[c]
            out[c] = {"min": float(s.min()), "max": float(s.max()),
                      "below": int((s < lo).sum()), "above": int((s > hi).sum()),
                      "limits": [lo, hi], "source": src}
    if "irradiance_lux" in df:
        lux = df["irradiance_lux"]
        out["irradiance_lux"]["above_bh1750_max"] = int((lux > BH1750_MAX_LUX).sum())
        out["irradiance_lux"]["at_saturation"] = int((lux >= LUX_SATURATION).sum())
    return out


# [GUIDE] A. Physical consistency: P = V x I
# WHAT: checks power_mw == voltage_v * current_ma (V x mA = mW) within a small tolerance.
# WHY:  if the three readings disagree, at least one of them is wrong.
# EVIDENCE: results/dataset_validation.md (largest residual well below the 0.5 mW tolerance).
def check_power(df, tol_mw=POWER_TOLERANCE_MW):
    """power_mw must equal voltage_v * current_ma (V x mA = mW)."""
    resid = df["power_mw"] - df["voltage_v"] * df["current_ma"]
    return {"max_abs_resid_mw": round(float(resid.abs().max()), 4),
            "rows_over_tol": int((resid.abs() > tol_mw).sum()), "tol_mw": tol_mw}


# [GUIDE] A. Data leakage check
# WHAT: (1) for each label-proxy column (health_score) it measures how well that column
#       ALONE predicts the label, run-wise; (2) it flags that a random ROW split is unsafe.
# WHY:  a column that already "knows" the answer must never be a model input, and the
#       60 near-identical readings of one run must never be split across train and test.
# HOW:  depth-3 decision tree, StratifiedGroupKFold grouped by run_id (chance is about 20 %).
# OUTPUT: {"single_column_runwise_accuracy", "random_row_split_unsafe"}.
# EVIDENCE: results/dataset_validation.md: health_score alone gives 74.6 % run-wise
# accuracy, so it is excluded from model inputs (contract.LEAKAGE_COLUMNS).
def check_leakage(df, seed=0):
    """Columns that reveal the label. For each leakage column, run-wise
    accuracy of a depth-3 tree on that column alone (chance ~ 1/5)."""
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.tree import DecisionTreeClassifier

    present = [c for c in LEAKAGE_COLUMNS if c in df]
    out = {"leakage_columns_present": present, "single_column_runwise_accuracy": {}}
    n_splits = min(5, int(df.groupby(LABEL_COLUMN)[GROUP_COLUMN].nunique().min()))
    if n_splits >= 2:
        cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        for c in present:
            pred = cross_val_predict(DecisionTreeClassifier(max_depth=3, random_state=seed),
                                     df[[c]], df[LABEL_COLUMN], groups=df[GROUP_COLUMN], cv=cv)
            out["single_column_runwise_accuracy"][c] = round(
                float((pred == df[LABEL_COLUMN]).mean()), 4)
    out["random_row_split_unsafe"] = bool(df.groupby(GROUP_COLUMN).size().max() > 1)
    return out


def check_split_requirements(df):
    runs = df.drop_duplicates(GROUP_COLUMN)
    out = {"runs_per_label_window": None, "sites_missing_labels": {}}
    if WINDOW_COLUMN in df:
        tab = pd.crosstab(runs[LABEL_COLUMN], runs[WINDOW_COLUMN])
        out["runs_per_label_window"] = {l: {w: int(n) for w, n in r.items()}
                                        for l, r in tab.iterrows()}
    if "site_id" in df:
        for site, row in pd.crosstab(df["site_id"], df[LABEL_COLUMN]).iterrows():
            missing = [l for l in LABELS if row.get(l, 0) == 0]
            if missing:
                out["sites_missing_labels"][site] = missing
    return out


def check_ids_against_sites(df, sites):
    return {c: sorted(set(df[c]) - set(sites[c]))
            for c in ("site_id", "device_id", "panel_id") if c in df and c in sites}


def compare_part(combined, part, label):
    """Is `part` an exact copy of the combined rows for `label`?

    Matches on record_id and compares every shared column. Files are never
    concatenated, so overlapping copies cannot duplicate records.
    """
    base = combined.set_index("record_id")
    ids = part["record_id"]
    inside = ids.isin(base.index)
    cols = [c for c in part.columns if c in combined.columns and c != "record_id"]
    mine = part[inside].drop_duplicates("record_id").set_index("record_id")[cols]
    ref = base.loc[mine.index, cols]
    diff = ~((mine == ref) | (mine.isna() & ref.isna()))
    expected = combined.loc[combined[LABEL_COLUMN] == label, "record_id"]
    same_rows_same_order = (len(part) == len(expected) and list(ids) == list(expected)
                            and not diff.any(axis=None))
    return {
        "rows": int(len(part)),
        "duplicate_ids_in_file": int(ids.duplicated().sum()),
        "not_in_combined": int((~inside).sum()),
        "rows_with_value_mismatch": int(diff.any(axis=1).sum()),
        "columns_with_mismatch": [c for c in cols if diff[c].any()],
        "wrong_label_rows": int((part[LABEL_COLUMN] != label).sum()) if LABEL_COLUMN in part else None,
        "combined_rows_for_label": int(len(expected)),
        "combined_rows_missing_from_file": int(len(set(expected) - set(ids))),
        "exact_subset": bool(inside.all() and not diff.any(axis=None) and not ids.duplicated().any()),
        "identical_to_combined_slice": bool(same_rows_same_order),
    }


def describe_prediction_file(pred, combined):
    """Facts about a prediction file; it is reported, never used as labels."""
    label_col = next((c for c in ("predicted_label", "predicted_fault") if c in pred), None)
    j = pred.merge(combined[["record_id", LABEL_COLUMN] +
                            ([WINDOW_COLUMN] if WINDOW_COLUMN in combined else [])],
                   on="record_id", how="left")
    out = {"rows": int(len(pred)), "prediction_column": label_col,
           "ids_in_combined": int(j[LABEL_COLUMN].notna().sum())}
    if WINDOW_COLUMN in j:
        out["windows"] = {k: int(v) for k, v in j[WINDOW_COLUMN].value_counts().items()}
    if label_col:
        known = j[LABEL_COLUMN].notna()
        out["agreement_with_labels"] = round(
            float((j.loc[known, label_col] == j.loc[known, LABEL_COLUMN]).mean()), 4)
    return out


def provenance_warnings(df, timestamps, overlaps, ranges, sites=None):
    """Observations that bear on whether this is real measured data.
    They are flags for the provenance review, not proof either way."""
    w = [PROVENANCE_NOTE]
    if timestamps["intervals"] and timestamps["intervals_off_expected"] == 0:
        w.append(f"All {timestamps['intervals']} intra-run intervals are exactly "
                 f"{EXPECTED_INTERVAL_S:g} s (no clock jitter or dropped readings).")
    runs = df[GROUP_COLUMN].nunique()
    if timestamps["runs_starting_at_second_00"] == runs:
        w.append(f"All {runs} runs start at second :00.")
    sizes = df.groupby(GROUP_COLUMN).size()
    if sizes.nunique() == 1:
        w.append(f"Every run has exactly {int(sizes.iloc[0])} readings.")
    if overlaps["cross_site"]:
        w.append(f"{len(overlaps['cross_site'])} run pairs on different sites overlap in time "
                 f"(e.g. {overlaps['cross_site'][0][0]} and {overlaps['cross_site'][0][1]}); "
                 "a single laboratory panel cannot run both.")
    lux = ranges.get("irradiance_lux", {})
    if lux.get("above_bh1750_max"):
        w.append(f"{lux['above_bh1750_max']} readings exceed {BH1750_MAX_LUX} lx "
                 "(BH1750 maximum); the lux sensor model is undocumented.")
    if df["site_id"].nunique() > 1:
        w.append(f"Data comes from {df['site_id'].nunique()} sites/devices.")
    if sites is not None:
        mix = [c for c in sites.columns if c.startswith("mix_")]
        if mix:
            w.append(f"sites.csv contains per-site class-mix columns {mix}, "
                     "which look like data-generation parameters.")
    return w


# ---------------------------------------------------------------- driver

# [GUIDE] A. Entry point of the dataset validation (called by scripts/01_validate_dataset.py)
# WHAT: runs every check above and sorts problems into errors (data must not be used)
#       and warnings (usable, but must be reported).
# INPUT: the combined pv_fault_dataset.csv (+ optional per-class files, sites.csv).
# OUTPUT: a dict with every check result plus "errors" and "warnings".
# EVIDENCE: results/dataset_validation.md -> "PASS (with warnings)". Nothing is modified.
def validate_dataset(combined, parts=None, sites=None, predictions=None):
    """Run every check. `parts` maps label -> per-class DataFrame."""
    assert_not_predictions(combined, "combined dataset")
    r = {"schema": check_schema(combined)}
    errors, warnings = [], []
    if r["schema"]["missing"]:
        errors.append(f"missing required columns: {r['schema']['missing']}")
        r["errors"], r["warnings"] = errors, warnings
        return r

    r["labels"] = check_labels(combined)
    r["missing"] = check_missing(combined)
    r["duplicates"] = check_duplicates(combined)
    r["runs"] = check_runs(combined)
    r["timestamps"] = check_timestamps(combined)
    r["overlaps"] = check_run_overlaps(combined)
    r["ranges"] = check_ranges(combined)
    r["power"] = check_power(combined)
    r["leakage"] = check_leakage(combined)
    r["split"] = check_split_requirements(combined)
    if sites is not None:
        r["unknown_ids"] = check_ids_against_sites(combined, sites)
    r["parts"] = {lab: compare_part(combined, p, lab) for lab, p in (parts or {}).items()}
    if predictions is not None:
        r["predictions"] = describe_prediction_file(predictions, combined)
    r["provenance"] = provenance_warnings(combined, r["timestamps"], r["overlaps"],
                                          r["ranges"], sites)

    s = r["schema"]
    if s["unknown"]:
        warnings.append(f"columns not in contract: {s['unknown']}")
    if s["non_numeric_sensor"]:
        errors.append(f"non-numeric sensor columns: {s['non_numeric_sensor']}")
    if s["timestamps_not_utc_z"]:
        errors.append(f"{s['timestamps_not_utc_z']} timestamps are not UTC 'YYYY-MM-DDTHH:MM:SSZ'")
    if r["labels"]["unknown"]:
        errors.append(f"unknown labels: {r['labels']['unknown']}")
    if r["labels"]["absent"]:
        errors.append(f"labels with no rows: {r['labels']['absent']}")
    if r["missing"]["nan_by_column"]:
        errors.append(f"missing values: {r['missing']['nan_by_column']}")
    if r["missing"]["temp_sentinels"]:
        warnings.append(f"DS18B20 error codes present: {r['missing']['temp_sentinels']}")
    if r["duplicates"].get("record_id"):
        errors.append(f"{r['duplicates']['record_id']} duplicate record_id values")
    if r["duplicates"].get("device_timestamp"):
        errors.append(f"{r['duplicates']['device_timestamp']} duplicate device/timestamp pairs")
    if r["duplicates"]["identical_rows_except_id"]:
        warnings.append(f"{r['duplicates']['identical_rows_except_id']} rows identical except record_id")
    for key in ("mixed_label_runs", "mixed_site_id_runs", "mixed_device_id_runs", "mixed_window_runs"):
        if r["runs"].get(key):
            errors.append(f"{key}: {r['runs'][key]}")
    if r["runs"].get("bad_reading_sequence_runs"):
        warnings.append(f"runs with gaps in 'reading': {r['runs']['bad_reading_sequence_runs']}")
    t = r["timestamps"]
    if t["non_increasing_runs"]:
        errors.append(f"timestamps not increasing in runs: {t['non_increasing_runs']}")
    if t["intervals_off_expected"]:
        warnings.append(f"{t['intervals_off_expected']} intervals differ from "
                        f"{EXPECTED_INTERVAL_S:g} s by more than {INTERVAL_TOLERANCE_S:g} s")
    if r["power"]["rows_over_tol"]:
        errors.append(f"{r['power']['rows_over_tol']} rows where power_mw != voltage_v * current_ma "
                      f"(> {r['power']['tol_mw']} mW)")
    for c, v in r["ranges"].items():
        if v["below"] or v["above"]:
            warnings.append(f"{c}: {v['below']} below / {v['above']} above {v['limits']}")
    lux = r["ranges"].get("irradiance_lux", {})
    if lux.get("at_saturation"):
        warnings.append(f"irradiance_lux at saturation ({LUX_SATURATION}) in {lux['at_saturation']} rows")
    for c, acc in r["leakage"]["single_column_runwise_accuracy"].items():
        warnings.append(f"leakage: '{c}' alone predicts the label with {acc:.1%} run-wise "
                        "accuracy; it must never be a model input")
    if r["runs"]["run_id_encodes_label_share"] > 0.5:
        warnings.append("leakage: run_id contains the class code; use it only for grouping")
    if r.get("unknown_ids"):
        bad = {k: v for k, v in r["unknown_ids"].items() if v}
        if bad:
            errors.append(f"IDs not in sites.csv: {bad}")
    for lab, p in r["parts"].items():
        if not p["exact_subset"]:
            errors.append(f"per-class file for {lab} is not an exact subset of the combined dataset")
        elif p["combined_rows_missing_from_file"]:
            warnings.append(f"per-class file for {lab} misses "
                            f"{p['combined_rows_missing_from_file']} combined rows")
    if r["split"]["sites_missing_labels"]:
        warnings.append(f"sites missing classes (affects leave-one-site-out): "
                        f"{r['split']['sites_missing_labels']}")
    r["errors"], r["warnings"] = errors, warnings
    return r


def render_markdown(r, files=None):
    """Human-readable report for results/dataset_validation.md."""
    L = ["# Dataset validation report", ""]
    L.append("Generated by `scripts/01_validate_dataset.py`. Input files are read only; "
             "nothing was modified. This report does **not** establish that the data is "
             "verified laboratory data (see Provenance).")
    L.append("")
    if files:
        L += ["## Input files", "", "| File | Rows | SHA-256 |", "|---|---|---|"]
        L += [f"| `{f['name']}` | {f['rows']} | `{f['sha256']}` |" for f in files]
        L.append("")
    status = "FAIL" if r["errors"] else "PASS (with warnings)" if r["warnings"] else "PASS"
    L += [f"## Result: **{status}**", "", f"Errors: {len(r['errors'])} · Warnings: {len(r['warnings'])}", ""]
    if r["errors"]:
        L += ["### Errors", ""] + [f"- {e}" for e in r["errors"]] + [""]
    if r["warnings"]:
        L += ["### Warnings", ""] + [f"- {w}" for w in r["warnings"]] + [""]
    if "labels" not in r:
        return "\n".join(L)

    s, runs, t = r["schema"], r["runs"], r["timestamps"]
    L += ["## Schema", "",
          f"- Rows: {s['rows']}",
          f"- Sensor columns: {', '.join(s['sensor'])}",
          f"- ID columns: {', '.join(s['id'])}; meta: {', '.join(s['meta'])}",
          f"- Leakage columns present: {', '.join(s['leakage']) or 'none'}",
          f"- Missing / unknown columns: {s['missing'] or 'none'} / {s['unknown'] or 'none'}", ""]
    L += ["## Classes and runs", "", "| Label | Rows | Runs |", "|---|---|---|"]
    L += [f"| {l} | {r['labels']['counts'].get(l, 0)} | {runs['runs_per_label'].get(l, 0)} |" for l in LABELS]
    L += ["", f"- Runs: {runs['runs']}; readings per run: {runs['readings_per_run']['min']}"
              f"–{runs['readings_per_run']['max']}",
          f"- Runs with mixed label/site/device/window: "
          f"{len(runs['mixed_label_runs'])}/{len(runs.get('mixed_site_id_runs', []))}/"
          f"{len(runs.get('mixed_device_id_runs', []))}/{len(runs.get('mixed_window_runs', []))}", ""]
    if r["split"]["runs_per_label_window"]:
        L += ["Runs per label and window:", "", "| Label | " +
              " | ".join(sorted(next(iter(r['split']['runs_per_label_window'].values())))) + " |"]
        L.append("|---|" + "---|" * len(next(iter(r["split"]["runs_per_label_window"].values()))))
        for l, row in r["split"]["runs_per_label_window"].items():
            L.append(f"| {l} | " + " | ".join(str(row[w]) for w in sorted(row)) + " |")
        L.append("")
    L += ["## Timestamps", "",
          f"- Range: {t['first_utc']} → {t['last_utc']}",
          f"- Intra-run intervals: {t['intervals']}; off {EXPECTED_INTERVAL_S:g} s: "
          f"{t['intervals_off_expected']} (min {t['interval_min_s']} s, max {t['interval_max_s']} s)",
          f"- Runs with non-increasing timestamps: {len(t['non_increasing_runs'])}",
          f"- Duplicates: {r['duplicates']}", ""]
    L += ["## Physical ranges", "", "| Column | Min | Max | Below | Above | Limits | Source |",
          "|---|---|---|---|---|---|---|"]
    for c, v in r["ranges"].items():
        L.append(f"| {c} | {v['min']:g} | {v['max']:g} | {v['below']} | {v['above']} | "
                 f"{v['limits']} | {v['source']} |")
    lux = r["ranges"].get("irradiance_lux", {})
    L += ["", f"- Lux above BH1750 max ({BH1750_MAX_LUX}): {lux.get('above_bh1750_max')}; "
              f"at saturation ({LUX_SATURATION}): {lux.get('at_saturation')}",
          f"- Power consistency: max |P − V·I| = {r['power']['max_abs_resid_mw']} mW; "
          f"rows over {r['power']['tol_mw']} mW: {r['power']['rows_over_tol']}", ""]
    lk = r["leakage"]
    L += ["## Leakage and split requirements", "",
          f"- Leakage columns: {lk['leakage_columns_present']}; single-column run-wise accuracy: "
          f"{lk['single_column_runwise_accuracy']} (chance ≈ 0.2)",
          f"- run_id encodes the label for {runs['run_id_encodes_label_share']:.0%} of rows "
          "→ group key only, never a feature",
          "- Readings within a run are near-identical → **random row splits are not allowed**; "
          "use run-wise splits (Phase 4)",
          f"- Sites missing classes: {r['split']['sites_missing_labels'] or 'none'}", ""]
    if r["parts"]:
        L += ["## Per-class files vs combined dataset", "",
              "| Label | Rows | Exact subset | Identical slice | Mismatched rows | Not in combined |",
              "|---|---|---|---|---|---|"]
        for lab, p in r["parts"].items():
            L.append(f"| {lab} | {p['rows']} | {p['exact_subset']} | {p['identical_to_combined_slice']} | "
                     f"{p['rows_with_value_mismatch']} | {p['not_in_combined']} |")
        L += ["", "Per-class files are used for verification only; training uses the combined "
                  "file so records are never duplicated.", ""]
    if "predictions" in r:
        p = r["predictions"]
        L += ["## Prediction file (not ground truth)", "",
              f"- Rows: {p['rows']}; ids found in combined: {p['ids_in_combined']}; "
              f"windows: {p.get('windows')}",
              f"- Agreement of `{p['prediction_column']}` with labels: "
              f"{p.get('agreement_with_labels')}",
              "- Rejected as a training label source (`assert_not_predictions`).", ""]
    L += ["## Provenance warnings", ""] + [f"- {w}" for w in r["provenance"]] + [""]
    return "\n".join(L)

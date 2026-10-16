"""Data contract validator for project-backend.
Pure Python validation without external Pydantic dependencies.
Enforces UTC Z timestamps, canonical site lookups, tag/field separation,
reserved measurement protection, and strict vs open mode schema rules.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.config import get_final_measurement, get_stream_config, is_measurement_reserved
from backend.points import PointSpec
from backend.sites import get_device_id, get_protocol, valid_site_ids

# Reject codes according to data contract specification
REJECT_RESERVED_MEASUREMENT = "RESERVED_MEASUREMENT"
REJECT_MISSING_REQUIRED = "MISSING_REQUIRED"
REJECT_UNKNOWN_SITE = "UNKNOWN_SITE"
REJECT_DEVICE_MISMATCH = "DEVICE_MISMATCH"
REJECT_PROTOCOL_MISMATCH = "PROTOCOL_MISMATCH"
REJECT_TIME_FORMAT = "TIME_FORMAT"
REJECT_TIME_NOT_UTC = "TIME_NOT_UTC"
REJECT_BAD_TYPE = "BAD_TYPE"
REJECT_BAD_VALUE = "BAD_VALUE"
REJECT_BAD_FIELD_NAME = "BAD_FIELD_NAME"
REJECT_EMPTY_POINT = "EMPTY_POINT"
REJECT_TOPIC_MISMATCH = "TOPIC_MISMATCH"

# Regex for field name format in open mode
FIELD_NAME_REGEX = re.compile(r"^[a-z][a-z0-9_]*$")

# Regex to check ISO 8601 UTC timestamp structure
ISO_TIME_REGEX = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|z|[+-]\d{2}:?\d{2})?$"
)


@dataclass
class ValidationResult:
    """Outcome of validating a single input record."""
    success: bool
    point: Optional[PointSpec] = None
    reject_code: Optional[str] = None
    reject_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    dropped_columns: List[str] = field(default_factory=list)


def is_missing(val: Any) -> bool:
    """Returns True if the value represents a missing/null/NaN reading.
    Real numeric 0 and 0.0 return False (they are kept).
    """
    if val is None:
        return True
    if isinstance(val, str) and val.strip() in ("", "nan", "NaN", "null", "None"):
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    # Handle pandas NA/NaN if present
    try:
        import pandas as pd
        if pd.isna(val):
            return True
    except Exception:
        pass
    return False


def parse_utc_timestamp(val: Any) -> Tuple[Optional[datetime], Optional[str], Optional[str]]:
    """Validates and parses a timestamp into a timezone-aware UTC datetime.
    Returns (datetime, None, None) on success, or (None, reject_code, reject_reason) on failure.
    Enforces:
    - Non-empty ISO 8601 format
    - Must be UTC (explicit Z or +00:00 / +0000 offset)
    - Naive timestamps and non-UTC offsets are rejected, never converted.
    """
    if is_missing(val):
        return None, REJECT_MISSING_REQUIRED, "timestamp_utc is missing or empty"

    if isinstance(val, datetime):
        if val.tzinfo is None:
            return None, REJECT_TIME_NOT_UTC, "Naive datetime without timezone is rejected"
        offset = val.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            return None, REJECT_TIME_NOT_UTC, f"Timestamp has non-UTC offset: {val.tzinfo}"
        return val.astimezone(timezone.utc), None, None

    if not isinstance(val, str):
        return None, REJECT_TIME_FORMAT, f"Timestamp must be string or datetime, got {type(val).__name__}"

    s = val.strip()
    if not ISO_TIME_REGEX.match(s):
        return None, REJECT_TIME_FORMAT, f"Invalid ISO 8601 timestamp format: '{s}'"

    # Check timezone suffix explicitly
    has_z = s.endswith("Z") or s.endswith("z")
    has_utc_offset = s.endswith("+00:00") or s.endswith("-00:00") or s.endswith("+0000") or s.endswith("-0000") or s.endswith("+00") or s.endswith("-00")

    # If it ends with another offset like +05:30 or -04:00
    has_other_offset = bool(re.search(r"[+-]\d{2}:?\d{2}$", s)) and not has_utc_offset

    if has_other_offset:
        return None, REJECT_TIME_NOT_UTC, f"Non-UTC timestamp offset rejected: '{s}'"

    if not (has_z or has_utc_offset):
        return None, REJECT_TIME_NOT_UTC, f"Naive timestamp without UTC indicator rejected: '{s}'"

    # Parse ISO string
    # Standardize 'Z' to '+00:00' for datetime.fromisoformat if needed
    iso_clean = s.replace("Z", "+00:00").replace("z", "+00:00").replace(" ", "T")
    try:
        dt = datetime.fromisoformat(iso_clean)
        offset = dt.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            return None, REJECT_TIME_NOT_UTC, f"Timestamp does not resolve to UTC: '{s}'"
        return dt.astimezone(timezone.utc), None, None
    except ValueError as e:
        return None, REJECT_TIME_FORMAT, f"Failed to parse timestamp '{s}': {e}"


def _cast_float(val: Any) -> Tuple[Optional[float], bool]:
    """Attempts to cast val to float. Returns (float_value, success)."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val), True
    if isinstance(val, str):
        try:
            return float(val.strip()), True
        except ValueError:
            return None, False
    return None, False


def _cast_int(val: Any) -> Tuple[Optional[int], bool]:
    """Attempts to cast val to int. Returns (int_value, success)."""
    if isinstance(val, bool):
        return None, False
    if isinstance(val, int):
        return val, True
    if isinstance(val, float):
        if val.is_integer():
            return int(val), True
        return None, False
    if isinstance(val, str):
        s = val.strip()
        try:
            return int(s), True
        except ValueError:
            try:
                f = float(s)
                if f.is_integer():
                    return int(f), True
            except ValueError:
                pass
            return None, False
    return None, False


def validate_record(stream_name: str, record: Dict[str, Any]) -> ValidationResult:
    """Validates a single record dictionary against the stream definition.
    Returns ValidationResult with PointSpec or reject code and reason.
    """
    stream_cfg = get_stream_config(stream_name)
    if not stream_cfg:
        return ValidationResult(
            success=False,
            reject_code=REJECT_BAD_VALUE,
            reject_reason=f"Unknown stream name: '{stream_name}'",
        )

    # 1. Resolve final measurement name and check against reserved list
    try:
        final_measurement = get_final_measurement(stream_name)
    except Exception as e:
        return ValidationResult(
            success=False,
            reject_code=REJECT_BAD_VALUE,
            reject_reason=str(e),
        )

    if is_measurement_reserved(final_measurement):
        return ValidationResult(
            success=False,
            reject_code=REJECT_RESERVED_MEASUREMENT,
            reject_reason=f"Final measurement '{final_measurement}' is reserved for Member 3",
        )

    # 2. Check required columns
    required_cols = stream_cfg.get("required", [])
    for col in required_cols:
        val = record.get(col)
        if is_missing(val):
            return ValidationResult(
                success=False,
                reject_code=REJECT_MISSING_REQUIRED,
                reject_reason=f"Missing required field: '{col}'",
            )

    # 3. Validate site_id
    site_id = str(record.get("site_id", "")).strip()
    if site_id not in valid_site_ids():
        return ValidationResult(
            success=False,
            reject_code=REJECT_UNKNOWN_SITE,
            reject_reason=f"Unknown site_id: '{site_id}'",
        )

    # 4. Validate or auto-fill device_id
    expected_device = get_device_id(site_id)
    provided_device = record.get("device_id")
    if not is_missing(provided_device):
        if str(provided_device).strip() != expected_device:
            return ValidationResult(
                success=False,
                reject_code=REJECT_DEVICE_MISMATCH,
                reject_reason=(
                    f"Device '{provided_device}' does not match expected "
                    f"'{expected_device}' for site '{site_id}'"
                ),
            )
    device_id = expected_device

    # 5. Validate or auto-fill protocol
    expected_protocol = get_protocol(site_id)
    provided_protocol = record.get("protocol")
    if not is_missing(provided_protocol):
        if str(provided_protocol).strip() != expected_protocol:
            return ValidationResult(
                success=False,
                reject_code=REJECT_PROTOCOL_MISMATCH,
                reject_reason=(
                    f"Protocol '{provided_protocol}' does not match expected "
                    f"'{expected_protocol}' for site '{site_id}'"
                ),
            )
    protocol = expected_protocol

    # 6. Parse and validate timestamp_utc
    time_val = record.get("timestamp_utc")
    point_time, time_err_code, time_err_reason = parse_utc_timestamp(time_val)
    if point_time is None:
        return ValidationResult(
            success=False,
            reject_code=time_err_code,
            reject_reason=time_err_reason,
        )

    # 7. Prepare tags (site_id, device_id, protocol)
    tags = {
        "site_id": site_id,
        "device_id": device_id,
        "protocol": protocol,
    }

    # Columns that must NEVER be stored as fields (they are tags or point time)
    non_field_cols = {"site_id", "device_id", "protocol", "timestamp_utc"}

    # Stream configuration options
    mode = stream_cfg.get("mode", "open")
    float_fields: Set[str] = set(stream_cfg.get("float_fields", []))
    int_fields: Set[str] = set(stream_cfg.get("int_fields", []))
    string_fields: Set[str] = set(stream_cfg.get("string_fields", []))
    drop_cols: Set[str] = set(stream_cfg.get("drop_columns", []))
    warnings_cfg: Dict[str, Dict[str, float]] = stream_cfg.get("warnings", {}) or {}
    allowed_values_cfg: Dict[str, List[str]] = stream_cfg.get("allowed_values", {}) or {}

    # Proposed fields in open mode
    float_fields_proposed: Set[str] = set(stream_cfg.get("float_fields_proposed", []))

    fields: Dict[str, Any] = {}
    warnings: List[str] = []
    dropped_columns: List[str] = []

    # 8. Process every column in the input record
    for col, raw_val in record.items():
        if col in non_field_cols:
            continue

        # Configured drop columns (e.g. dataset metadata)
        if col in drop_cols:
            dropped_columns.append(col)
            continue

        # If value is missing (None, "", NaN) -> Omit completely (never zero)
        if is_missing(raw_val):
            continue

        # Handle float fields
        if col in float_fields or (mode == "open" and col in float_fields_proposed):
            f_val, ok = _cast_float(raw_val)
            if not ok:
                return ValidationResult(
                    success=False,
                    reject_code=REJECT_BAD_TYPE,
                    reject_reason=f"Field '{col}' expected float, got '{raw_val}'",
                )
            fields[col] = f_val

            # Check warnings (e.g., humidity_pct outside 0..100)
            if col in warnings_cfg:
                w_rule = warnings_cfg[col]
                min_v = w_rule.get("min")
                max_v = w_rule.get("max")
                if min_v is not None and f_val < min_v:
                    warnings.append(f"{col} ({f_val}) below minimum {min_v}")
                if max_v is not None and f_val > max_v:
                    warnings.append(f"{col} ({f_val}) above maximum {max_v}")
            continue

        # Handle int fields
        if col in int_fields:
            i_val, ok = _cast_int(raw_val)
            if not ok:
                return ValidationResult(
                    success=False,
                    reject_code=REJECT_BAD_TYPE,
                    reject_reason=f"Field '{col}' expected int, got '{raw_val}'",
                )
            fields[col] = i_val
            continue

        # Handle string fields
        if col in string_fields:
            if isinstance(raw_val, (dict, list)):
                return ValidationResult(
                    success=False,
                    reject_code=REJECT_BAD_TYPE,
                    reject_reason=f"Field '{col}' expected string, got {type(raw_val).__name__}",
                )
            s_val = str(raw_val).strip()

            # Check allowed values if defined
            if col in allowed_values_cfg:
                allowed = allowed_values_cfg[col]
                if s_val not in allowed:
                    return ValidationResult(
                        success=False,
                        reject_code=REJECT_BAD_VALUE,
                        reject_reason=f"Field '{col}' value '{s_val}' not in allowed {allowed}",
                    )
            fields[col] = s_val
            continue

        # Handling fields not in predefined schema
        if mode == "strict":
            # Strict mode drops unknown columns with a counter
            dropped_columns.append(col)
            continue

        # Open mode rules
        if mode == "open":
            # Check field name regex
            if not FIELD_NAME_REGEX.match(col):
                return ValidationResult(
                    success=False,
                    reject_code=REJECT_BAD_FIELD_NAME,
                    reject_reason=f"Invalid field name '{col}' in open mode; must match ^[a-z][a-z0-9_]*$",
                )

            # Reject lists and dicts
            if isinstance(raw_val, (dict, list)):
                return ValidationResult(
                    success=False,
                    reject_code=REJECT_BAD_TYPE,
                    reject_reason=f"Field '{col}' contains {type(raw_val).__name__}, complex types rejected",
                )

            # Booleans stay boolean
            if isinstance(raw_val, bool):
                fields[col] = raw_val
            elif isinstance(raw_val, (int, float)):
                # Numbers become float (ints converted)
                fields[col] = float(raw_val)
            elif isinstance(raw_val, str):
                s_val = raw_val.strip()
                # If numeric string, convert to float in open mode
                f_val, ok = _cast_float(s_val)
                if ok:
                    fields[col] = f_val
                else:
                    fields[col] = s_val
            else:
                return ValidationResult(
                    success=False,
                    reject_code=REJECT_BAD_TYPE,
                    reject_reason=f"Unsupported type {type(raw_val).__name__} for field '{col}'",
                )

            # Enforce max 40 fields per point
            if len(fields) > 40:
                return ValidationResult(
                    success=False,
                    reject_code=REJECT_BAD_VALUE,
                    reject_reason="Point exceeds maximum of 40 fields",
                )

    # 9. Verify point is not empty
    if not fields:
        return ValidationResult(
            success=False,
            reject_code=REJECT_EMPTY_POINT,
            reject_reason="Point contains no valid fields after validation",
        )

    point_spec = PointSpec(
        measurement=final_measurement,
        time=point_time,
        tags=tags,
        fields=fields,
    )

    return ValidationResult(
        success=True,
        point=point_spec,
        warnings=warnings,
        dropped_columns=dropped_columns,
    )

"""Configuration and schema loader for project-backend.
Reads schema.yaml, resolves measurement prefixes, and enforces reserved measurements.
No secrets are handled in this module.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dotenv import load_dotenv
import yaml

# Automatically load environment variables from .env if present
load_dotenv()

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.yaml"

_SCHEMA_DATA: Optional[Dict[str, Any]] = None


def load_schema(schema_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads and caches schema.yaml."""
    path = schema_path or DEFAULT_SCHEMA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    with open(path, mode="r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    global _SCHEMA_DATA
    _SCHEMA_DATA = data
    return data


def get_schema() -> Dict[str, Any]:
    """Returns the cached schema dict, loading from DEFAULT_SCHEMA_PATH if needed."""
    if _SCHEMA_DATA is None:
        load_schema()
    return _SCHEMA_DATA


def get_measurement_prefix() -> str:
    """Returns the effective measurement prefix.
    Environment variable MEASUREMENT_PREFIX overrides schema.yaml settings.
    """
    env_prefix = os.environ.get("MEASUREMENT_PREFIX")
    if env_prefix is not None:
        return env_prefix
    schema = get_schema()
    settings = schema.get("settings", {}) or {}
    return settings.get("measurement_prefix", "") or ""


def get_reserved_measurements() -> Set[str]:
    """Returns the set of reserved measurement names owned by Member 3."""
    schema = get_schema()
    reserved = schema.get("reserved_measurements", []) or []
    return set(reserved)


def get_stream_config(stream_name: str) -> Optional[Dict[str, Any]]:
    """Returns the raw configuration dict for the given stream name."""
    schema = get_schema()
    streams = schema.get("streams", {}) or {}
    return streams.get(stream_name)


def get_final_measurement(stream_name: str) -> str:
    """Computes the final measurement name for a stream after applying the prefix."""
    cfg = get_stream_config(stream_name)
    if not cfg:
        raise ValueError(f"Unknown stream: '{stream_name}'")
    base_name = cfg.get("measurement", stream_name)
    prefix = get_measurement_prefix()
    return f"{prefix}{base_name}"


def is_measurement_reserved(measurement_name: str) -> bool:
    """Checks whether the final measurement name collides with Member 3's reserved measurements."""
    return measurement_name in get_reserved_measurements()


def verify_startup_safety() -> None:
    """SAFETY GUARD: Refuses startup if the measurement prefix is empty unless
    ALLOW_REAL_MEASUREMENTS equals '1'.
    """
    prefix = get_measurement_prefix()
    allow_real = os.environ.get("ALLOW_REAL_MEASUREMENTS", "").strip() == "1"
    if not prefix and not allow_real:
        raise RuntimeError(
            "SAFETY GUARD: Measurement prefix is empty and ALLOW_REAL_MEASUREMENTS is not '1'. "
            "Refusing to start to protect production data."
        )

"""Rotating JSON lines logger for rejected payloads.
Outputs to rejects.jsonl (configurable via REJECTS_PATH).
Secrets are never included; record snippets are truncated to 500 characters.
"""
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import threading
from typing import Any, Dict, Optional

DEFAULT_REJECTS_FILE = Path(__file__).resolve().parent.parent.parent / "rejects.jsonl"
VALID_STREAMS = {"readings", "health", "prediction", "twin"}

_LOCK = threading.Lock()
_LOGGER: Optional[logging.Logger] = None
_LOGGER_FILE_PATH: Optional[Path] = None


def get_rejects_file() -> Path:
    """Returns the effective rejects file path from REJECTS_PATH env var or default."""
    env_path = os.environ.get("REJECTS_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_REJECTS_FILE


class _RejectsPathProxy:
    """Path-like proxy for backwards compatibility with REJECTS_FILE imports."""

    @property
    def _path(self) -> Path:
        return get_rejects_file()

    def __fspath__(self) -> str:
        return str(self._path)

    def __str__(self) -> str:
        return str(self._path)

    def __repr__(self) -> str:
        return f"RejectsPathProxy({self._path})"

    def exists(self) -> bool:
        return self._path.exists()

    def read_text(self, *args, **kwargs) -> str:
        return self._path.read_text(*args, **kwargs)

    def stat(self, *args, **kwargs):
        return self._path.stat(*args, **kwargs)


REJECTS_FILE = _RejectsPathProxy()


def _get_rejects_logger() -> logging.Logger:
    global _LOGGER, _LOGGER_FILE_PATH
    current_target = get_rejects_file()

    if _LOGGER is not None and _LOGGER_FILE_PATH == current_target:
        return _LOGGER

    logger = logging.getLogger("solar_backend_rejects")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Close and remove existing handlers if path changed
    for h in list(logger.handlers):
        try:
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)

    # Ensure parent directory exists
    current_target.parent.mkdir(parents=True, exist_ok=True)

    # 5 MB max per file, 3 backups (total ~15 MB max)
    handler = RotatingFileHandler(
        filename=str(current_target),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    _LOGGER = logger
    _LOGGER_FILE_PATH = current_target
    return _LOGGER


def log_reject(
    transport: str,
    stream: Optional[str],
    code: str,
    detail: str,
    record: Any,
    topic: Optional[str] = None,
) -> None:
    """Logs a rejected message to rejects.jsonl.
    Truncates record to 500 characters and scrubs sensitive tokens.
    `stream` must hold the stream name (readings, health, prediction, twin) or None.
    `topic` holds the MQTT topic (or None for HTTP).
    """
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Serialize and truncate record
    if isinstance(record, (dict, list)):
        try:
            rec_str = json.dumps(record, default=str)
        except Exception:
            rec_str = str(record)
    else:
        rec_str = str(record)

    if len(rec_str) > 500:
        rec_str = rec_str[:500] + "...[truncated]"

    # Scrub potential secrets from environment
    for key in ("INFLUX_TOKEN", "API_KEY", "READ_API_KEY", "INFLUX_PASSWORD"):
        val = os.environ.get(key)
        if val and len(val) >= 4:
            detail = detail.replace(val, "***")
            rec_str = rec_str.replace(val, "***")

    eff_stream = stream if stream in VALID_STREAMS else None

    entry = {
        "time": now_utc,
        "transport": transport,
        "stream": eff_stream,
        "topic": topic,
        "code": code,
        "detail": detail,
        "record": rec_str,
    }

    with _LOCK:
        logger = _get_rejects_logger()
        logger.info(json.dumps(entry))

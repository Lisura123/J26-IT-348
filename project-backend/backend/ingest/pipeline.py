"""Shared ingestion pipeline for MQTT and HTTP transports.
Validates records against the data contract, logs rejects, and batches writes to MongoDB.
Maintains thread-safe counters.
"""
from collections import Counter
import threading
from typing import Any, Dict, List, Optional

from backend.mongo_writer import MongoWriter
from backend.ingest.rejects import log_reject
from backend.points import PointSpec
from backend.validate import REJECT_TOPIC_MISMATCH, validate_record

_COUNTER_LOCK = threading.Lock()
_COUNTERS = {
    "received": 0,
    "accepted": 0,
    "rejected": 0,
    "written": 0,
    "write_failed": 0,
}

_DEFAULT_WRITER: Optional[MongoWriter] = None


def get_default_writer() -> MongoWriter:
    """Returns a shared MongoWriter instance or creates one."""
    global _DEFAULT_WRITER
    if _DEFAULT_WRITER is None:
        _DEFAULT_WRITER = MongoWriter.from_env()
    return _DEFAULT_WRITER


def get_counters() -> Dict[str, int]:
    """Returns a copy of the thread-safe ingestion counters."""
    with _COUNTER_LOCK:
        return dict(_COUNTERS)


def reset_counters() -> None:
    """Resets the thread-safe counters (useful for tests)."""
    with _COUNTER_LOCK:
        for k in _COUNTERS:
            _COUNTERS[k] = 0


def process_records(
    stream: str,
    records: List[Dict[str, Any]],
    transport: str,
    topic_site: Optional[str] = None,
    writer: Optional[Any] = None,
    topic: Optional[str] = None,
) -> Dict[str, Any]:
    """Processes a batch of records from MQTT or HTTP.
    1. Validates records with data contract.
    2. Rejects mismatched topic site_id with TOPIC_MISMATCH.
    3. Batches accepted points to MongoDB writer.
    4. Logs rejections to rejects.jsonl.
    5. Updates thread-safe counters.

    Returns:
        dict: {accepted, rejected, written, write_failed, reasons: {code: count}}
    """
    if not isinstance(records, list):
        records = [records]

    active_writer = writer if writer is not None else get_default_writer()
    effective_topic = topic or (f"solar/{topic_site}/{stream}" if transport == "mqtt" and topic_site else None)

    batch_accepted = 0
    batch_rejected = 0
    batch_written = 0
    batch_write_failed = 0
    reasons: Counter[str] = Counter()
    valid_points: List[PointSpec] = []

    for raw_rec in records:
        if not isinstance(raw_rec, dict):
            batch_rejected += 1
            reasons["BAD_TYPE"] += 1
            log_reject(
                transport=transport,
                stream=stream,
                code="BAD_TYPE",
                detail=f"Record must be a JSON object, got {type(raw_rec).__name__}",
                record=raw_rec,
                topic=effective_topic,
            )
            continue

        rec = dict(raw_rec)

        # For MQTT, site_id is anchored by the topic
        if topic_site:
            payload_site = rec.get("site_id")
            if payload_site and str(payload_site).strip() != topic_site:
                batch_rejected += 1
                reasons[REJECT_TOPIC_MISMATCH] += 1
                log_reject(
                    transport=transport,
                    stream=stream,
                    code=REJECT_TOPIC_MISMATCH,
                    detail=f"Payload site_id '{payload_site}' does not match topic site '{topic_site}'",
                    record=rec,
                    topic=effective_topic,
                )
                continue
            rec["site_id"] = topic_site

        # Validate with stage 1 validator
        res = validate_record(stream, rec)
        if not res.success or res.point is None:
            code = res.reject_code or "UNKNOWN"
            batch_rejected += 1
            reasons[code] += 1
            log_reject(
                transport=transport,
                stream=stream,
                code=code,
                detail=res.reject_reason or "",
                record=rec,
                topic=effective_topic,
            )
        else:
            batch_accepted += 1
            valid_points.append(res.point)

    # Batch write valid points to MongoDB
    if valid_points:
        w_cnt, f_cnt, err_txt = active_writer.write(valid_points)
        batch_written = w_cnt
        batch_write_failed = f_cnt
        if f_cnt > 0:
            reasons["WRITE_FAILED"] += f_cnt
            for pt in valid_points:
                log_reject(
                    transport=transport,
                    stream=stream,
                    code="WRITE_FAILED",
                    detail=err_txt or "MongoDB write failed",
                    record={"measurement": pt.measurement, "tags": pt.tags, "time": pt.time.isoformat()},
                    topic=effective_topic,
                )

    # Update global thread-safe counters
    with _COUNTER_LOCK:
        _COUNTERS["received"] += len(records)
        _COUNTERS["accepted"] += batch_accepted
        _COUNTERS["rejected"] += batch_rejected
        _COUNTERS["written"] += batch_written
        _COUNTERS["write_failed"] += batch_write_failed

    return {
        "accepted": batch_accepted,
        "rejected": batch_rejected,
        "written": batch_written,
        "write_failed": batch_write_failed,
        "reasons": dict(reasons),
    }

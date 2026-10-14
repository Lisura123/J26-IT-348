"""Backend verification tool for project-backend.
Compares expected valid rows from input CSV against distinct (site_id, _time)
rows found in InfluxDB, protecting reserved measurements.

Usage:
    python -m backend.tools.verify_backend --stream readings --file ../pv_fault_dataset.csv --window OPERATION --site SITE-C
    python -m backend.tools.verify_backend --stream readings --file ../pv_fault_dataset.csv --window OPERATION
    python -m backend.tools.verify_backend --stream health --file ../pv_fault_dataset.csv --window OPERATION
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv

from backend.config import get_final_measurement, get_reserved_measurements, load_schema
from backend.influx_reader import InfluxReader, ReadError
from backend.tools.replay import load_and_filter_records
from backend.validate import parse_utc_timestamp, validate_record

load_dotenv()


def verify_dataset(
    stream: str,
    file_path: Path,
    window: Optional[str] = None,
    sites: Optional[List[str]] = None,
    limit: Optional[int] = None,
    reader: Optional[Any] = None,
) -> int:
    """Performs verification of expected vs found rows.
    Returns 0 on OK, 1 on MISMATCH, 2 on InfluxDB ReadError.
    """
    load_schema()
    target_measurement = get_final_measurement(stream)
    reserved_measurements = sorted(get_reserved_measurements())

    # 1. Load candidate records using replay filtering rules
    raw_records = load_and_filter_records(
        file_path=file_path,
        stream=stream,
        window=window,
        sites=sites,
        limit=limit,
    )

    # 2. Expected = rows accepted by stage 1 validator
    accepted_records: List[Dict[str, Any]] = []
    for r in raw_records:
        res = validate_record(stream, r)
        if res.success:
            accepted_records.append(r)

    expected = len(accepted_records)

    # Determine site and time boundaries
    if accepted_records:
        parsed_times = []
        for r in accepted_records:
            t_dt, _, _ = parse_utc_timestamp(r.get("timestamp_utc"))
            if t_dt:
                parsed_times.append(t_dt)

        dt_start = min(parsed_times) if parsed_times else None
        dt_stop = max(parsed_times) if parsed_times else None
        filter_sites = sorted(set(r.get("site_id", "") for r in accepted_records if r.get("site_id")))
    else:
        dt_start = None
        dt_stop = None
        filter_sites = sites or []

    # 3. Found = distinct (site_id, _time) rows in InfluxDB
    active_reader = reader or InfluxReader.from_env()

    try:
        found = active_reader.count_distinct_rows(
            stream=stream,
            sites=filter_sites,
            start=dt_start,
            stop=dt_stop,
        )

        stored_ids: Set[str] = set()
        if expected != found:
            stored_ids = active_reader.get_stored_record_ids(
                stream=stream,
                sites=filter_sites,
                start=dt_start,
                stop=dt_stop,
            )

        reserved_counts: Dict[str, int] = {}
        for m in reserved_measurements:
            reserved_counts[m] = active_reader.count_field_values(m)

    except ReadError as e:
        print(f"CANNOT READ INFLUXDB: {e}")
        return 2

    print("=== Backend Verification ===")
    print(f"Stream:             {stream}")
    print(f"Target Measurement: {target_measurement}")
    print(f"Filter sites:       {filter_sites}")
    if dt_start and dt_stop:
        print(f"Time range:         {dt_start.strftime('%Y-%m-%dT%H:%M:%SZ')} .. {dt_stop.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"\nExpected rows:      {expected}")
    print(f"Found rows:         {found}")

    is_ok = (expected == found)
    status_str = "OK" if is_ok else "MISMATCH"
    print(f"Verification:       {status_str}")

    # On MISMATCH, list up to 10 missing record_ids
    if not is_ok:
        print("\nMISMATCH Details:")
        missing_ids = []
        for r in accepted_records:
            rid = r.get("record_id")
            if rid and rid not in stored_ids and rid not in missing_ids:
                missing_ids.append(rid)
                if len(missing_ids) >= 10:
                    break

        if missing_ids:
            print(f"  Missing record_ids (first {len(missing_ids)}): {missing_ids}")
        else:
            print("  (No specific missing record_ids found or stream lacks record_id)")

    # 4. Print field-value counts of the six reserved Member 3 measurements
    print("\nReserved Member 3 Measurements (field-value counts):")
    for m in reserved_measurements:
        cnt = reserved_counts.get(m, 0)
        print(f"  [Reserved] {m:22s} : {cnt:6d} field values")

    print("\nNote: InfluxDB points sharing the same series (measurement + tags) and timestamp overwrite each other.")

    return 0 if is_ok else 1


def main():
    parser = argparse.ArgumentParser(description="Backend verification tool for project-backend")
    parser.add_argument("--stream", required=True, choices=["readings", "health", "prediction", "twin"])
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--window", choices=["TRAINING", "OPERATION"], default=None)
    parser.add_argument("--site", action="append", default=[], help="Target site (repeatable)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ret = verify_dataset(
        stream=args.stream,
        file_path=args.file,
        window=args.window,
        sites=args.site,
        limit=args.limit,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()

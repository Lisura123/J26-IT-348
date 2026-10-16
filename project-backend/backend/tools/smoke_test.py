"""Repeatable smoke test tool for project-backend ingestion.
Sends 6 valid records (3 from SITE-A, 3 from SITE-C) with a unique run marker
on voltage_v, and 3 deliberately invalid records.

Verifies:
  - All 6 rows exist in MongoDB with the exact marked voltage_v value (proves transport wrote them)
  - For HTTP: 6 accepted, 3 rejected, reject codes: TIME_NOT_UTC, UNKNOWN_SITE, PROTOCOL_MISMATCH
  - For MQTT: rejects file grew by 3 lines with transport "mqtt" and the 3 codes
  - All 6 reserved Member 3 measurements remain completely untouched

Usage:
    python -m backend.tools.smoke_test --transport http --file ..\\pv_fault_dataset.csv
    python -m backend.tools.smoke_test --transport mqtt --file ..\\pv_fault_dataset.csv
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import httpx
import paho.mqtt.client as mqtt

from backend.config import get_final_measurement, get_reserved_measurements, load_schema
from backend.mongo_reader import MongoReader, ReadError
from backend.ingest.rejects import get_rejects_file

load_dotenv()

EXPECTED_REJECT_CODES = {"TIME_NOT_UTC", "UNKNOWN_SITE", "PROTOCOL_MISMATCH"}


def extract_smoke_records(
    csv_path: Path,
    marker: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[Tuple[str, str], float]]:
    """Extracts 3 OPERATION rows from SITE-A and 3 from SITE-C,
    stripping dataset metadata columns (window, run_id, label, health_score).
    Adds (marker * 0.001) to voltage_v so values are uniquely marked per run.
    Builds 3 bad records.
    Returns: (valid_records, invalid_records, expected_voltages)
    """
    if not csv_path.exists():
        print(f"Error: Dataset file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    site_a_rows: List[Dict[str, Any]] = []
    site_c_rows: List[Dict[str, Any]] = []

    drop_cols = {"window", "run_id", "label", "health_score"}
    delta = marker * 0.001

    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("window") != "OPERATION":
                continue

            sid = row.get("site_id")
            clean_rec = {k: v for k, v in row.items() if k not in drop_cols and v != ""}

            # Convert numeric types
            for fld in (
                "voltage_v", "current_ma", "power_mw", "panel_temp1_c",
                "panel_temp2_c", "ambient_temp_c", "humidity_pct",
                "pressure_hpa", "irradiance_lux", "dust_vo_v"
            ):
                if fld in clean_rec:
                    try:
                        clean_rec[fld] = float(clean_rec[fld])
                    except ValueError:
                        pass
            if "reading" in clean_rec:
                try:
                    clean_rec["reading"] = int(clean_rec["reading"])
                except ValueError:
                    pass

            if sid == "SITE-A" and len(site_a_rows) < 3:
                site_a_rows.append(clean_rec)
            elif sid == "SITE-C" and len(site_c_rows) < 3:
                site_c_rows.append(clean_rec)

            if len(site_a_rows) == 3 and len(site_c_rows) == 3:
                break

    valid_records = site_a_rows + site_c_rows
    if len(valid_records) != 6:
        print(f"Error: Could not extract 6 valid rows (got {len(valid_records)})", file=sys.stderr)
        sys.exit(1)

    # Apply voltage_v marker to each valid record
    expected_voltages: Dict[Tuple[str, str], float] = {}
    for r in valid_records:
        orig_v = float(r["voltage_v"])
        marked_v = round(orig_v + delta, 4)
        r["voltage_v"] = marked_v
        expected_voltages[(r["site_id"], r["timestamp_utc"])] = marked_v

    # 3 Deliberately bad records
    # 1. Timestamp without Z (naive) -> TIME_NOT_UTC
    bad_time = dict(site_a_rows[0])
    bad_time["record_id"] = "BAD-REC-TIME"
    bad_time["timestamp_utc"] = "2026-09-10T07:26:00"

    # 2. Unknown site -> UNKNOWN_SITE
    bad_site = dict(site_a_rows[1])
    bad_site["record_id"] = "BAD-REC-SITE"
    bad_site["site_id"] = "SITE-Z"

    # 3. Protocol mismatch: SITE-C is LoRa, send WIFI -> PROTOCOL_MISMATCH
    bad_proto = dict(site_c_rows[0])
    bad_proto["record_id"] = "BAD-REC-PROTO"
    bad_proto["protocol"] = "WIFI"

    invalid_records = [bad_time, bad_site, bad_proto]
    return valid_records, invalid_records, expected_voltages


def run_http_smoke_test(
    valid_recs: List[Dict[str, Any]],
    invalid_recs: List[Dict[str, Any]],
    api_url: str,
    api_key: str,
) -> Dict[str, Any]:
    """Sends records via HTTP API."""
    url = f"{api_url}/api/v1/ingest/readings"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    all_recs = valid_recs + invalid_recs
    print(f"\n[HTTP] Sending {len(all_recs)} records to {url}...")
    with httpx.Client(timeout=10.0) as client:
        resp = client.post(url, headers=headers, json=all_recs)

    if resp.status_code != 200:
        print(f"HTTP Ingestion failed: status={resp.status_code}, body={resp.text}", file=sys.stderr)
        sys.exit(1)

    return resp.json()


def run_mqtt_smoke_test(
    valid_recs: List[Dict[str, Any]],
    invalid_recs: List[Dict[str, Any]],
    host: str,
    port: int,
) -> None:
    """Publishes records via MQTT."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"smoke-tester-{random.randint(100, 999)}")
    print(f"\n[MQTT] Connecting to broker at {host}:{port}...")
    client.connect(host, port, keepalive=60)
    client.loop_start()

    # Publish valid records
    for r in valid_recs:
        sid = r["site_id"]
        topic = f"solar/{sid}/readings"
        info = client.publish(topic, json.dumps(r), qos=1)
        info.wait_for_publish()

    # Publish invalid records
    for r in invalid_recs:
        sid = r["site_id"]
        topic = f"solar/{sid}/readings"
        info = client.publish(topic, json.dumps(r), qos=1)
        info.wait_for_publish()

    print("[MQTT] Published 6 valid and 3 invalid records.")
    print("[MQTT] Waiting 3 seconds for service queue and buffer flush...")
    time.sleep(3.0)

    client.loop_stop()
    client.disconnect()


def verify_mongo_voltages(
    reader: Any,
    measurement: str,
    expected_voltages: Dict[Tuple[str, str], float],
) -> List[str]:
    """Reads back records by (site_id, timestamp_utc) and requires voltage_v to match marked values."""
    found_map: Dict[Tuple[str, str], float] = {}

    if hasattr(reader, "db") and reader.db is not None:
        col = reader.db[measurement]
        sites = list(set(sid for sid, _ in expected_voltages.keys()))
        docs = col.find({"site_id": {"$in": sites}})
        for d in docs:
            sid = str(d.get("site_id", ""))
            dt = d.get("timestamp_utc")
            if isinstance(dt, datetime):
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                time_str = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                time_str = str(dt)
            val = float(d.get("voltage_v", 0.0))
            found_map[(sid, time_str)] = round(val, 4)
    elif hasattr(reader, "rows"):
        for r in getattr(reader, "rows", []):
            if r.get("stream") == measurement or r.get("stream") == "readings":
                sid = r.get("site_id")
                dt = r.get("time_utc")
                if isinstance(dt, datetime):
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    time_str = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    time_str = str(dt)
                val = float(r.get("fields", {}).get("voltage_v", 0.0))
                found_map[(sid, time_str)] = round(val, 4)

    errors = []
    for (sid, t_str), exp_val in expected_voltages.items():
        if (sid, t_str) not in found_map:
            errors.append(f"Missing row in MongoDB: site={sid}, time={t_str}")
        else:
            found_val = found_map[(sid, t_str)]
            if abs(found_val - exp_val) > 1e-3:
                errors.append(
                    f"Voltage mismatch for site={sid}, time={t_str}: expected {exp_val}, found {found_val}"
                )

    return errors


def read_new_reject_entries(
    rejects_path: Path,
    initial_line_count: int,
) -> List[Dict[str, Any]]:
    """Reads newly appended entries from rejects.jsonl."""
    if not rejects_path.exists():
        return []
    with open(rejects_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    new_lines = all_lines[initial_line_count:]
    entries = []
    for line in new_lines:
        line_str = line.strip()
        if line_str:
            try:
                entries.append(json.loads(line_str))
            except Exception:
                pass
    return entries


def main():
    parser = argparse.ArgumentParser(description="Repeatable ingestion smoke test for project-backend")
    parser.add_argument("--transport", required=True, choices=["http", "mqtt"])
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument("--mqtt-host", default=os.environ.get("MQTT_HOST", "localhost"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.environ.get("MQTT_PORT", 1883)))
    args = parser.parse_args()

    load_schema()
    readings_measurement = get_final_measurement("readings")
    reserved_measurements = sorted(get_reserved_measurements())

    run_marker = random.randint(100, 9999)
    print(f"=== Repeatable Smoke Test: Transport={args.transport.upper()} (Run Marker: {run_marker}) ===")
    print(f"Target readings measurement: '{readings_measurement}'")

    # 1. Connect to MongoDB and record counts BEFORE
    reader = MongoReader.from_env()
    before_counts: Dict[str, int] = {}
    try:
        if not reader.ping():
            print("CANNOT READ MONGODB: MongoDB ping failed")
            sys.exit(2)

        for m in reserved_measurements:
            cnt = reader.count_field_values(m)
            before_counts[m] = cnt
    except ReadError as e:
        print(f"CANNOT READ MONGODB: {e}")
        sys.exit(2)

    print("\nReading baseline MongoDB document counts BEFORE test...")
    for m in reserved_measurements:
        print(f"  [Reserved] {m:22s} : {before_counts[m]:6d} documents")

    # 2. Extract records with marked voltage_v
    valid_recs, invalid_recs, expected_voltages = extract_smoke_records(args.file, run_marker)
    print(f"\nExtracted 6 valid records with marked voltage_v values:")
    for (sid, t_str), v in expected_voltages.items():
        print(f"  [{sid}] {t_str} -> marked voltage_v = {v}")

    # Track rejects file size before run
    rejects_path = get_rejects_file()
    initial_reject_lines = 0
    if rejects_path.exists():
        with open(rejects_path, "r", encoding="utf-8") as f:
            initial_reject_lines = len(f.readlines())

    # 3. Send records
    http_result = None
    if args.transport == "http":
        if not args.api_key:
            print("Error: API_KEY must be provided or set in environment for HTTP test", file=sys.stderr)
            sys.exit(1)
        http_result = run_http_smoke_test(valid_recs, invalid_recs, args.api_url, args.api_key)
    elif args.transport == "mqtt":
        run_mqtt_smoke_test(valid_recs, invalid_recs, args.mqtt_host, args.mqtt_port)

    # 4. Check MongoDB reserved measurement counts AFTER
    print("\nReading MongoDB document counts AFTER test...")
    reserved_changed = False
    try:
        for m in reserved_measurements:
            cnt = reader.count_field_values(m)
            diff = cnt - before_counts[m]
            status = "OK (unchanged)" if diff == 0 else f"VIOLATION (+{diff})"
            print(f"  [Reserved] {m:22s} : {cnt:6d} documents -> {status}")
            if diff != 0:
                reserved_changed = True

        # 5. Read back by (site_id, timestamp_utc) and verify exact marked voltage_v
        print("\nVerifying readback of 6 marked rows by (site_id, timestamp_utc)...")
        voltage_errors = verify_mongo_voltages(reader, readings_measurement, expected_voltages)
    except ReadError as e:
        print(f"CANNOT READ MONGODB: {e}")
        sys.exit(2)
    finally:
        reader.close()

    # 6. Evaluate all pass conditions
    failed_checks: List[str] = []

    if reserved_changed:
        failed_checks.append("Reserved Member 3 measurements were modified")

    if voltage_errors:
        failed_checks.extend(voltage_errors)
    else:
        print("  [Pass] All 6 valid rows read back with exact marked voltage_v values.")

    if args.transport == "http" and http_result:
        print("\nHTTP Response Verification:")
        accepted = http_result.get("accepted")
        rejected = http_result.get("rejected")
        reasons = http_result.get("reasons", {})
        print(f"  Accepted: {accepted} (expected 6)")
        print(f"  Rejected: {rejected} (expected 3)")
        print(f"  Reasons:  {reasons}")

        if accepted != 6:
            failed_checks.append(f"HTTP accepted count is {accepted}, expected 6")
        if rejected != 3:
            failed_checks.append(f"HTTP rejected count is {rejected}, expected 3")
        if not EXPECTED_REJECT_CODES.issubset(reasons.keys()):
            failed_checks.append(f"HTTP missing expected reject codes {EXPECTED_REJECT_CODES - set(reasons.keys())}")

    elif args.transport == "mqtt":
        print("\nMQTT Rejects Log Verification:")
        new_entries = read_new_reject_entries(rejects_path, initial_reject_lines)
        mqtt_entries = [e for e in new_entries if e.get("transport") == "mqtt"]
        mqtt_codes = {e.get("code") for e in mqtt_entries}
        print(f"  New MQTT reject entries: {len(mqtt_entries)} (expected 3)")
        print(f"  Observed reject codes:   {mqtt_codes}")

        if len(mqtt_entries) != 3:
            failed_checks.append(f"MQTT rejects log grew by {len(mqtt_entries)} entries, expected exactly 3")
        if not EXPECTED_REJECT_CODES.issubset(mqtt_codes):
            failed_checks.append(f"MQTT rejects log missing expected codes {EXPECTED_REJECT_CODES - mqtt_codes}")

    # 7. Final status
    if failed_checks:
        print("\n=== SMOKE TEST FAILED ===", file=sys.stderr)
        print("Failed checks:", file=sys.stderr)
        for err in failed_checks:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n=== SMOKE TEST PASSED: All checks verified and repeatable ===")
        sys.exit(0)


if __name__ == "__main__":
    main()

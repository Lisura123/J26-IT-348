"""Replay tool for project-backend.
Replays CSV data over HTTP or MQTT with speed control, dry-run mode,
batching, and startup safety guards.

Usage:
    python -m backend.tools.replay --stream readings --file ../pv_fault_dataset.csv --window OPERATION --dry-run
    python -m backend.tools.replay --stream readings --file ../pv_fault_dataset.csv --window OPERATION --site SITE-C --transport mqtt --speed 0
    python -m backend.tools.replay --stream readings --file ../pv_fault_dataset.csv --window OPERATION --transport http
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import httpx
import paho.mqtt.client as mqtt

from backend.config import get_final_measurement, load_schema, verify_startup_safety
from backend.validate import parse_utc_timestamp

load_dotenv()

DATASET_METADATA_COLS = {"window", "run_id", "label"}
READINGS_NUMERIC_FLOATS = {
    "voltage_v", "current_ma", "power_mw", "panel_temp1_c",
    "panel_temp2_c", "ambient_temp_c", "humidity_pct",
    "pressure_hpa", "irradiance_lux", "dust_vo_v"
}


def is_pv_fault_dataset(header: List[str]) -> bool:
    """Checks whether the CSV header is that of pv_fault_dataset.csv."""
    return DATASET_METADATA_COLS.issubset(set(header))


def load_and_filter_records(
    file_path: Path,
    stream: str,
    window: Optional[str] = None,
    sites: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Loads CSV, applies stream rules, window/site/limit filters, and type normalization."""
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    with open(file_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        # Check refusal rule for prediction and twin streams
        if is_pv_fault_dataset(header):
            if stream in ("prediction", "twin"):
                print(
                    f"Refusal Error: '{file_path.name}' contains ground-truth labels, which are NOT model predictions. "
                    f"Streams '{stream}' accept only CSVs whose columns match their schema.",
                    file=sys.stderr,
                )
                sys.exit(1)

        records: List[Dict[str, Any]] = []
        is_dataset = is_pv_fault_dataset(header)
        site_filter_set = set(sites) if sites else None

        for row in reader:
            # Window filter
            if window and row.get("window"):
                if row.get("window") != window:
                    continue

            # Site filter
            sid = row.get("site_id")
            if site_filter_set and sid not in site_filter_set:
                continue

            # Stream specific transformations
            if is_dataset and stream == "readings":
                # Strict-mode drop of dataset metadata columns
                drop_cols = {"window", "run_id", "label", "health_score"}
                clean_rec = {k: v for k, v in row.items() if k not in drop_cols and v != ""}
                # Cast known numeric types
                for fld in READINGS_NUMERIC_FLOATS:
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
                rec = clean_rec

            elif is_dataset and stream == "health":
                # Demo fixture: record_id, timestamp_utc, site_id, device_id, health_score
                health_score = None
                if row.get("health_score") not in (None, ""):
                    try:
                        health_score = float(row["health_score"])
                    except ValueError:
                        pass

                rec = {
                    "record_id": row.get("record_id"),
                    "timestamp_utc": row.get("timestamp_utc"),
                    "site_id": row.get("site_id"),
                    "device_id": row.get("device_id"),
                }
                if health_score is not None:
                    rec["health_score"] = health_score
                if "protocol" in row and row["protocol"]:
                    rec["protocol"] = row["protocol"]

            else:
                # Normal CSV matching schema
                rec = {k: v for k, v in row.items() if v != ""}

            records.append(rec)
            if limit and len(records) >= limit:
                break

    return records


def _send_http_request_with_retries(
    client: httpx.Client,
    url: str,
    headers: Dict[str, str],
    payload: Any,
    max_retries: int = 3,
    initial_backoff: float = 0.5,
) -> httpx.Response:
    """Sends HTTP POST with up to 3 retries and exponential backoff on connection errors."""
    for attempt in range(max_retries + 1):
        try:
            return client.post(url, headers=headers, json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as e:
            if attempt == max_retries:
                print(f"HTTP connection failed after {max_retries} retries ({type(e).__name__}): {e}", file=sys.stderr)
                raise
            delay = initial_backoff * (2 ** attempt)
            time.sleep(delay)


def run_http_replay(
    records: List[Dict[str, Any]],
    stream: str,
    api_url: str,
    api_key: str,
    batch_size: int,
    speed: float,
) -> None:
    """Sends records via HTTP ingest API with batching, retries, and speed control.
    Adds up API responses and prints totals: accepted, rejected, written, write_failed
    and reasons per code. Exits non-zero if write_failed > 0 or any request returned non-200.
    """
    url = f"{api_url}/api/v1/ingest/{stream}"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    total = len(records)
    est_sec = total / 100.0
    print(f"\n[HTTP] Starting replay of {total} records to {url} (batch_size={batch_size}, speed={speed})...")
    print(f"  Estimated duration: ~{est_sec:.1f}s (Atlas free tier throttles writes at ~100 ops/s; e.g. 3000 rows can take ~30s or more)")

    total_accepted = 0
    total_rejected = 0
    total_written = 0
    total_write_failed = 0
    total_reasons: Counter[str] = Counter()
    has_http_error = False

    # Explicit timeouts: connect 10s, read 30s
    timeout = httpx.Timeout(30.0, connect=10.0, read=30.0)
    start_time = time.time()

    with httpx.Client(timeout=timeout) as client:
        prev_dt: Optional[datetime] = None
        sent = 0

        if speed == 0:
            # Send in batches of batch_size
            for i in range(0, total, batch_size):
                batch = records[i : i + batch_size]
                try:
                    resp = _send_http_request_with_retries(client, url, headers, batch)
                except Exception as e:
                    print(f"HTTP Request exception: {e}", file=sys.stderr)
                    has_http_error = True
                    break

                if resp.status_code != 200:
                    print(f"HTTP Error {resp.status_code}: {resp.text}", file=sys.stderr)
                    has_http_error = True
                    break

                try:
                    data = resp.json()
                except Exception:
                    data = {}

                total_accepted += data.get("accepted", 0)
                total_rejected += data.get("rejected", 0)
                total_written += data.get("written", 0)
                total_write_failed += data.get("write_failed", 0)
                for code, cnt in data.get("reasons", {}).items():
                    total_reasons[code] += cnt

                sent += len(batch)
                elapsed = max(0.001, time.time() - start_time)
                rate = sent / elapsed
                if sent % 500 < batch_size or sent == total:
                    print(f"  Progress: {sent}/{total} rows sent ({rate:.1f} rows/s)...")
        else:
            # Send row by row with speed delay
            for idx, rec in enumerate(records):
                curr_dt, _, _ = parse_utc_timestamp(rec.get("timestamp_utc"))
                if prev_dt and curr_dt and speed > 0:
                    gap_seconds = (curr_dt - prev_dt).total_seconds()
                    if gap_seconds > 0:
                        delay = min(gap_seconds / speed, 2.0)
                        time.sleep(delay)
                prev_dt = curr_dt

                try:
                    resp = _send_http_request_with_retries(client, url, headers, rec)
                except Exception as e:
                    print(f"HTTP Request exception: {e}", file=sys.stderr)
                    has_http_error = True
                    break

                if resp.status_code != 200:
                    print(f"HTTP Error {resp.status_code}: {resp.text}", file=sys.stderr)
                    has_http_error = True
                    break

                try:
                    data = resp.json()
                except Exception:
                    data = {}

                total_accepted += data.get("accepted", 0)
                total_rejected += data.get("rejected", 0)
                total_written += data.get("written", 0)
                total_write_failed += data.get("write_failed", 0)
                for code, cnt in data.get("reasons", {}).items():
                    total_reasons[code] += cnt

                sent += 1
                elapsed = max(0.001, time.time() - start_time)
                rate = sent / elapsed
                if sent % 500 == 0 or sent == total:
                    print(f"  Progress: {sent}/{total} rows sent ({rate:.1f} rows/s)...")

    # Print totals
    total_elapsed = max(0.001, time.time() - start_time)
    final_rate = sent / total_elapsed
    print("\n=== HTTP Replay Totals ===")
    print(f"  Duration:     {total_elapsed:.2f}s")
    print(f"  Throughput:   {final_rate:.1f} rows/s")
    print(f"  Accepted:     {total_accepted}")
    print(f"  Rejected:     {total_rejected}")
    print(f"  Written:      {total_written}")
    print(f"  Write Failed: {total_write_failed}")
    if total_reasons:
        print(f"  Reasons:      {dict(sorted(total_reasons.items()))}")
    else:
        print("  Reasons:      {}")

    if has_http_error:
        print("Error: Replay encountered a non-200 HTTP response or connection failure.", file=sys.stderr)
        sys.exit(1)

    if total_write_failed > 0:
        print(f"Error: {total_write_failed} records failed to write to database.", file=sys.stderr)
        sys.exit(1)


def check_mqtt_broker(host: str = "127.0.0.1", port: int = 1883, timeout: float = 5.0) -> bool:
    """Checks whether the MQTT broker is reachable on host:port.
    Uses 'Test-NetConnection 127.0.0.1 -Port 1883' (IPv4 address, not 'localhost') on Windows.
    Falls back to a standard socket connection check.
    """
    check_host = "127.0.0.1" if host in ("localhost", "::1") else host
    if sys.platform == "win32":
        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Test-NetConnection {check_host} -Port {port} -WarningAction SilentlyContinue).TcpTestSucceeded",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if res.returncode == 0 and "True" in res.stdout:
                return True
            return False
        except Exception:
            pass

    # Socket fallback (cross-platform or if powershell invocation fails)
    try:
        with socket.create_connection((check_host, port), timeout=timeout):
            return True
    except OSError:
        return False


def run_mqtt_replay(
    records: List[Dict[str, Any]],
    stream: str,
    mqtt_host: str,
    mqtt_port: int,
    batch_size: int,
    speed: float,
    file_path: Optional[Path] = None,
    check_broker: bool = True,
) -> None:
    """Publishes records via MQTT with QoS 1 and speed control."""
    if mqtt_host in ("localhost", "::1"):
        mqtt_host = "127.0.0.1"

    if check_broker:
        print(f"\n[MQTT] Checking MQTT broker at {mqtt_host}:{mqtt_port} with Test-NetConnection {mqtt_host} -Port {mqtt_port}...")
        if not check_mqtt_broker(mqtt_host, mqtt_port):
            print(
                f"Error: MQTT broker unreachable at {mqtt_host}:{mqtt_port} "
                f"(Test-NetConnection {mqtt_host} -Port {mqtt_port} failed). "
                f"Ensure Mosquitto is running on 127.0.0.1:1883.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"  [OK] MQTT broker is active on {mqtt_host}:{mqtt_port}.")

    client_id = f"replay-{stream}-{os.getpid()}"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    print(f"\n[MQTT] Connecting to broker at {mqtt_host}:{mqtt_port}...")
    client.connect(mqtt_host, mqtt_port, keepalive=60)
    client.loop_start()

    total = len(records)
    est_sec = total / 100.0
    print(f"[MQTT] Starting replay of {total} records (speed={speed})...")
    print(f"  Estimated duration: ~{est_sec:.1f}s (Atlas free tier throttles writes at ~100 ops/s; e.g. 3000 rows can take ~30s or more)")
    start_time = time.time()
    prev_dt: Optional[datetime] = None
    sent = 0

    if speed == 0:
        # Group into batches of batch_size by site_id, or send row-by-row
        i = 0
        while i < total:
            current_site = records[i].get("site_id")
            batch = []
            while i < total and len(batch) < batch_size and records[i].get("site_id") == current_site:
                batch.append(records[i])
                i += 1

            topic = f"solar/{current_site}/{stream}"
            payload = json.dumps(batch)
            info = client.publish(topic, payload, qos=1)
            info.wait_for_publish()
            sent += len(batch)
            elapsed = max(0.001, time.time() - start_time)
            rate = sent / elapsed
            if sent % 500 < batch_size or sent == total:
                print(f"  Progress: {sent}/{total} rows sent ({rate:.1f} rows/s)...")
    else:
        # Send row by row with delay
        for rec in records:
            curr_dt, _, _ = parse_utc_timestamp(rec.get("timestamp_utc"))
            if prev_dt and curr_dt and speed > 0:
                gap_seconds = (curr_dt - prev_dt).total_seconds()
                if gap_seconds > 0:
                    delay = min(gap_seconds / speed, 2.0)
                    time.sleep(delay)
            prev_dt = curr_dt

            sid = rec.get("site_id")
            topic = f"solar/{sid}/{stream}"
            payload = json.dumps(rec)
            info = client.publish(topic, payload, qos=1)
            info.wait_for_publish()
            sent += 1
            elapsed = max(0.001, time.time() - start_time)
            rate = sent / elapsed
            if sent % 500 == 0 or sent == total:
                print(f"  Progress: {sent}/{total} rows sent ({rate:.1f} rows/s)...")

    print("[MQTT] All messages published. Waiting 3 seconds for service buffer flush...")
    time.sleep(3.0)

    client.loop_stop()
    client.disconnect()

    total_elapsed = max(0.001, time.time() - start_time)
    final_rate = sent / total_elapsed
    print(f"\n[MQTT] Replay complete: {sent} rows published in {total_elapsed:.2f}s ({final_rate:.1f} rows/s).")
    print("Reminder: Run verify_backend to confirm all published points were written to MongoDB:")
    if file_path:
        print(f"  python -m backend.tools.verify_backend --stream {stream} --file {file_path}")
    else:
        print(f"  python -m backend.tools.verify_backend --stream {stream} --file <dataset_csv>")


def main():
    parser = argparse.ArgumentParser(description="Replay tool for project-backend")
    parser.add_argument("--stream", required=True, choices=["readings", "health", "prediction", "twin"])
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--window", choices=["TRAINING", "OPERATION"], default=None)
    parser.add_argument("--site", action="append", default=[], help="Target site (repeatable)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--transport", choices=["mqtt", "http"], default="http")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--speed", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")

    # Network args
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    raw_mqtt_host = os.environ.get("MQTT_HOST", "127.0.0.1")
    default_mqtt_host = "127.0.0.1" if raw_mqtt_host in ("localhost", "::1") else raw_mqtt_host
    parser.add_argument("--mqtt-host", default=default_mqtt_host)
    parser.add_argument("--mqtt-port", type=int, default=int(os.environ.get("MQTT_PORT", 1883)))

    args = parser.parse_args()

    # Safety guard applies to every tool that writes (unless dry-run)
    if not args.dry_run:
        verify_startup_safety()

    load_schema()
    target_measurement = get_final_measurement(args.stream)

    records = load_and_filter_records(
        file_path=args.file,
        stream=args.stream,
        window=args.window,
        sites=args.site,
        limit=args.limit,
    )

    sites_found = sorted(set(r.get("site_id", "") for r in records if r.get("site_id")))
    times_found = [r.get("timestamp_utc", "") for r in records if r.get("timestamp_utc")]
    time_min = min(times_found) if times_found else "N/A"
    time_max = max(times_found) if times_found else "N/A"

    print("=== Replay Tool ===")
    print(f"Stream:             {args.stream}")
    print(f"Target Measurement: {target_measurement}")
    print(f"Rows selected:      {len(records)}")
    print(f"Sites selected:     {sites_found}")
    print(f"Time range:         {time_min} .. {time_max}")

    if args.dry_run:
        print("\n--- DRY-RUN MODE: No data sent ---")
        print("Sample payloads (up to 3):")
        for sample in records[:3]:
            print(json.dumps(sample, indent=2))
        print(f"\nDry-run completed: {len(records)} rows verified.")
        sys.exit(0)

    start_time = time.time()
    if args.transport == "http":
        if not args.api_key:
            print("Error: Ingest API_KEY must be provided or configured in environment", file=sys.stderr)
            sys.exit(1)
        run_http_replay(
            records=records,
            stream=args.stream,
            api_url=args.api_url,
            api_key=args.api_key,
            batch_size=args.batch_size,
            speed=args.speed,
        )
    elif args.transport == "mqtt":
        target_mqtt_host = "127.0.0.1" if args.mqtt_host in ("localhost", "::1") else args.mqtt_host
        run_mqtt_replay(
            records=records,
            stream=args.stream,
            mqtt_host=target_mqtt_host,
            mqtt_port=args.mqtt_port,
            batch_size=args.batch_size,
            speed=args.speed,
            file_path=args.file,
        )

    elapsed = time.time() - start_time
    print(f"\nReplay summary: {len(records)} rows replayed via {args.transport.upper()} in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()

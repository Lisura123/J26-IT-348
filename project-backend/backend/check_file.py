"""CLI tool to check and validate CSV files against the data contract.
Read-only inspection tool.
Usage:
    python -m backend.check_file --stream readings --file <csv_path> [--window OPERATION] [--limit N]
"""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

from backend.validate import validate_record


def check_file(
    stream: str,
    file_path: Path,
    window: str = "",
    limit: int = 0,
) -> None:
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Checking {file_path.name} against stream '{stream}' ===")
    if window:
        print(f"Filter: window == '{window}'")
    if limit > 0:
        print(f"Limit: max {limit} rows")

    rows_read = 0
    rows_accepted = 0
    reject_counts: Counter[str] = Counter()
    first_rejected: List[Tuple[int, str, str, Dict[str, Any]]] = []
    dropped_col_counts: Counter[str] = Counter()
    total_warnings = 0
    warning_examples: List[str] = []

    # Duplicate tracking: (site_id, timestamp_utc)
    seen_series_time: Dict[Tuple[str, str], int] = defaultdict(int)

    with open(file_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            # Optional window filter
            row_window = row.get("window", "")
            if window and row_window != window:
                continue

            rows_read += 1

            # Validate record
            res = validate_record(stream, row)

            if res.success and res.point:
                rows_accepted += 1
                # Track series time for duplicate detection
                site_id = res.point.tags.get("site_id", "")
                ts_str = row.get("timestamp_utc", "")
                seen_series_time[(site_id, ts_str)] += 1

                for col in res.dropped_columns:
                    dropped_col_counts[col] += 1

                if res.warnings:
                    total_warnings += len(res.warnings)
                    if len(warning_examples) < 5:
                        warning_examples.extend(res.warnings[: 5 - len(warning_examples)])
            else:
                reject_code = res.reject_code or "UNKNOWN"
                reject_counts[reject_code] += 1
                if len(first_rejected) < 5:
                    first_rejected.append((row_idx, reject_code, res.reject_reason or "", row))

            if limit > 0 and rows_read >= limit:
                break

    # Analyze duplicates
    duplicates = [(k, count) for k, count in seen_series_time.items() if count > 1]
    duplicate_count = sum(count - 1 for _, count in duplicates)

    # Print summary
    print(f"\nRows read:     {rows_read}")
    print(f"Rows accepted: {rows_accepted}")
    print(f"Rows rejected: {sum(reject_counts.values())}")

    if reject_counts:
        print("\nRejections by reason code:")
        for code, cnt in sorted(reject_counts.items(), key=lambda x: -x[1]):
            print(f"  {code:22s} : {cnt:6d}")

    if dropped_col_counts:
        print("\nDropped columns (dataset metadata / unknown in strict mode):")
        for col, cnt in sorted(dropped_col_counts.items()):
            print(f"  {col:20s} : {cnt:6d}")

    print(f"\nWarnings: {total_warnings}")
    if warning_examples:
        for w in warning_examples:
            print(f"  - {w}")

    print(f"\nDuplicate (site_id, timestamp_utc) pairs: {duplicate_count}")
    print("  Note: Documents sharing the same unique key (site_id, timestamp_utc) overwrite each other.")
    if duplicates:
        print("  Examples of duplicates:")
        for (sid, ts), count in duplicates[:5]:
            print(f"    Site: {sid}, Timestamp: {ts} (appeared {count} times)")
    else:
        print("  None found: All accepted timestamps are distinct per site.")

    if first_rejected:
        print("\nFirst rejected rows (up to 5):")
        for line_no, code, reason, sample_row in first_rejected:
            rec_id = sample_row.get("record_id", "N/A")
            sid = sample_row.get("site_id", "N/A")
            ts = sample_row.get("timestamp_utc", "N/A")
            print(f"  [Row {line_no}] Code: {code} | Reason: {reason}")
            print(f"    Preview: record_id={rec_id}, site_id={sid}, timestamp={ts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CSV file against data contract")
    parser.add_argument("--stream", required=True, help="Stream name (e.g. readings, health, prediction, twin)")
    parser.add_argument("--file", required=True, type=Path, help="Path to input CSV file")
    parser.add_argument("--window", default="", help="Filter by window column (e.g. OPERATION, TRAINING)")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of rows to process")
    args = parser.parse_args()

    check_file(
        stream=args.stream,
        file_path=args.file,
        window=args.window,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

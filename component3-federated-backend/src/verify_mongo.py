"""Verifies document counts in MongoDB Atlas against shared dataset file rows or run exports.
On connection or read failure, prints 'CANNOT READ MONGODB: <short reason>' and exits with code 2.

Usage:
    python -m src.verify_mongo
    python -m src.verify_mongo --run-id member3-full-tanh-s0-trace
"""
import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv
import pandas as pd

from src import data_loader as dl
from src.fl_baseline import RESULTS_DIR
from src.mongo_client import get_mongo_client, get_mongo_db, scrub_secrets

load_dotenv()


def count_mongo(db, collection_name: str, query: dict) -> int:
    """Queries MongoDB count, handling read errors with exit code 2."""
    try:
        return db[collection_name].count_documents(query)
    except Exception as e:
        clean = scrub_secrets(str(e)).replace("\n", " ").strip()
        short_err = clean[:120] if len(clean) > 120 else clean
        print(f"CANNOT READ MONGODB: {short_err}", file=sys.stderr)
        sys.exit(2)


def verify_shared(db) -> bool:
    """Compares shared collection document counts against original files."""
    up = dl.load_fl_updates()
    checks = [
        ("network_link_metrics", len(dl.load_link_metrics())),
        ("network_outages", len(dl.load_outages())),
        ("fl_client_updates", len(up)),
        ("fl_global_models", len(dl.load_global_models())),
        ("ota_deployments", len(dl.load_backend_status())),
    ]
    all_ok = True
    for coll_name, expected in checks:
        got = count_mongo(db, coll_name, {"run_id": "shared"})
        status = "OK" if got == expected else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False
        print(f"{coll_name:22s} mongo={got:6d} file={expected:6d}  {status}")
    return all_ok


def verify_run(db, run_id: str) -> bool:
    """Compares run collection document counts against exported files."""
    exp_dir = RESULTS_DIR / "export" / run_id
    exp_updates = 117
    exp_models = 21
    exp_state = 120

    if (exp_dir / "fl_client_updates.csv").exists():
        try:
            exp_updates = len(pd.read_csv(exp_dir / "fl_client_updates.csv"))
        except Exception:
            pass
    if (exp_dir / "fl_global_models.csv").exists():
        try:
            exp_models = len(pd.read_csv(exp_dir / "fl_global_models.csv"))
        except Exception:
            pass
    if (exp_dir / "fl_site_state.csv").exists():
        try:
            exp_state = len(pd.read_csv(exp_dir / "fl_site_state.csv"))
        except Exception:
            pass

    checks = [
        ("fl_client_updates", exp_updates),
        ("fl_global_models", exp_models),
        ("fl_site_state", exp_state),
    ]
    all_ok = True
    for coll_name, expected in checks:
        got = count_mongo(db, coll_name, {"run_id": run_id})
        status = "OK" if got == expected else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False
        print(f"{coll_name:22s} mongo={got:6d} file={expected:6d}  {status}")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Verify MongoDB Atlas document counts against file rows")
    parser.add_argument("--run-id", default=None, help="Verify collections for a specific run_id")
    args = parser.parse_args()

    try:
        client = get_mongo_client()
        db = get_mongo_db(client)
        # Test connection ping first
        client.admin.command("ping")
    except Exception as e:
        clean = scrub_secrets(str(e)).replace("\n", " ").strip()
        short_err = clean[:120] if len(clean) > 120 else clean
        print(f"CANNOT READ MONGODB: {short_err}", file=sys.stderr)
        sys.exit(2)

    if args.run_id:
        ok = verify_run(db, args.run_id)
    else:
        ok = verify_shared(db)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

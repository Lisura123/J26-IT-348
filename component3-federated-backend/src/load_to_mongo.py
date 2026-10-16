"""Loads shared dataset files into MongoDB Atlas with run_id 'shared'.
Mirrors the joins and logic from the former ingestion loader.
Safe and idempotent to run multiple times.

Usage:
    python -m src.load_to_mongo
    python -m src.load_to_mongo --only network
    python -m src.load_to_mongo --only ota
"""
import argparse
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv
import pandas as pd

from src import data_loader as dl
from src.mongo_client import get_mongo_client, get_mongo_db, row_to_document, write_documents

load_dotenv()
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_network_link_metrics(db) -> int:
    """Loads network_link_metrics from shared link_metrics."""
    df = dl.load_link_metrics()
    cols = [
        "timestamp_utc",
        "site_id",
        "device_id",
        "protocol",
        "gateway_id",
        "link_quality",
        "online",
        "rssi_dbm",
        "snr_db",
        "lora_sf",
        "latency_ms",
        "packet_loss_pct",
        "throughput_kbps",
    ]
    docs = []
    for r in df.to_dict("records"):
        clean_row = {c: r[c] for c in cols if c in r}
        clean_row["run_id"] = "shared"
        docs.append(row_to_document(clean_row))

    count = write_documents(db["network_link_metrics"], "network_link_metrics", docs)
    print(f"network_link_metrics: wrote {count} points")
    return count


def load_network_outages(db) -> int:
    """Loads network_outages from shared outages."""
    out = dl.load_outages()
    out["end_utc"] = out["end_utc"].astype(str)
    cols = [
        "start_utc",
        "end_utc",
        "site_id",
        "device_id",
        "protocol",
        "cause",
        "duration_min",
    ]
    docs = []
    for r in out.to_dict("records"):
        clean_row = {c: r[c] for c in cols if c in r}
        clean_row["run_id"] = "shared"
        docs.append(row_to_document(clean_row))

    count = write_documents(db["network_outages"], "network_outages", docs)
    print(f"network_outages: wrote {count} points")
    return count


def load_fl_client_updates(db) -> int:
    """Loads fl_client_updates joined with transmissions."""
    up = dl.load_fl_updates()
    tx = dl.load_transmissions()[[
        "update_id",
        "scheduled_send_utc",
        "fragments",
        "airtime_s",
        "duty_cycle_wait_s",
        "radio_energy_mj",
    ]]
    df = up.merge(tx, on="update_id", how="left")
    cols = [
        "sent_at_utc",
        "scheduled_send_utc",
        "site_id",
        "device_id",
        "protocol",
        "status",
        "update_id",
        "round_id",
        "base_model_version",
        "num_samples",
        "local_loss_before",
        "local_loss_after",
        "local_accuracy_before",
        "local_accuracy_after",
        "delta_l2_norm",
        "payload_bytes",
        "fragments",
        "retries",
        "airtime_s",
        "duty_cycle_wait_s",
        "transfer_duration_s",
        "radio_energy_mj",
    ]
    docs = []
    for r in df.to_dict("records"):
        clean_row = {c: r[c] for c in cols if c in r}
        clean_row["run_id"] = "shared"
        docs.append(row_to_document(clean_row))

    count = write_documents(db["fl_client_updates"], "fl_client_updates", docs)
    print(f"fl_client_updates: wrote {count} points")
    return count


def load_fl_global_models(db) -> int:
    """Loads fl_global_models with model_path."""
    gm = dl.load_global_models()
    gm["model_path"] = gm["global_model_version"].apply(
        lambda v: f"models/global_{v}.json" if (MODELS_DIR / f"global_{v}.json").exists() else None
    )
    cols = [
        "published_at_utc",
        "round_id",
        "global_model_version",
        "aggregation",
        "clients_selected",
        "clients_received",
        "clients_late",
        "clients_failed",
        "total_samples",
        "global_val_loss",
        "global_val_accuracy",
        "mean_update_l2_norm",
        "model_size_int8_bytes",
        "model_path",
    ]
    docs = []
    for r in gm.to_dict("records"):
        clean_row = {c: r[c] for c in cols if c in r}
        clean_row["run_id"] = "shared"
        docs.append(row_to_document(clean_row))

    count = write_documents(db["fl_global_models"], "fl_global_models", docs)
    print(f"fl_global_models: wrote {count} points")
    return count


def load_ota_deployments(db) -> int:
    """Loads ota_deployments keeping all 126 rows."""
    st = dl.load_backend_status()
    cols = [
        "deployed_at_utc",
        "site_id",
        "device_id",
        "protocol",
        "ota_status",
        "ota_method",
        "round_id",
        "global_model_version",
        "ota_bytes",
        "ota_retries",
        "ota_duration_s",
        "ota_waited_offline_s",
        "ota_rssi_dbm",
        "ota_packet_loss_pct",
        "ota_fragments",
    ]
    docs = []
    for r in st.to_dict("records"):
        clean_row = {c: r[c] for c in cols if c in r}
        clean_row["run_id"] = "shared"
        docs.append(row_to_document(clean_row))

    count = write_documents(db["ota_deployments"], "ota_deployments", docs)
    print(f"ota_deployments: wrote {count} points")
    return count


def main():
    parser = argparse.ArgumentParser(description="Load shared dataset files to MongoDB Atlas")
    parser.add_argument(
        "--only",
        choices=[
            "network",
            "network_link_metrics",
            "network_outages",
            "updates",
            "fl_client_updates",
            "models",
            "fl_global_models",
            "ota",
            "ota_deployments",
        ],
        default=None,
        help="Load only a specific collection or group",
    )
    args = parser.parse_args()

    client = get_mongo_client()
    db = get_mongo_db(client)

    target = args.only
    if target in (None, "network", "network_link_metrics"):
        load_network_link_metrics(db)
    if target in (None, "network", "network_outages"):
        load_network_outages(db)
    if target in (None, "updates", "fl_client_updates"):
        load_fl_client_updates(db)
    if target in (None, "models", "fl_global_models"):
        load_fl_global_models(db)
    if target in (None, "ota", "ota_deployments"):
        load_ota_deployments(db)


if __name__ == "__main__":
    main()

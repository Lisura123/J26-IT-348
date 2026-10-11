"""Step 10: export one run in the shared file formats and (optionally) write it to InfluxDB."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import data_loader as dl
from src.engine import run_fl
from src.evaluate import BASE, FULL
from src.fl_baseline import RESULTS_DIR

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {"baseline": BASE, "full": FULL}

GLOBAL_COLS = ["round_id", "global_model_version", "aggregation", "clients_selected", "clients_received",
               "total_samples", "global_val_loss", "global_val_accuracy", "mean_update_l2_norm",
               "published_at_utc", "model_size_int8_bytes", "clients_late", "clients_failed",
               "compression_method", "model_path", "run_id"]          # first 13 = fl_global_models.csv
UPDATE_COLS = ["update_id", "round_id", "site_id", "device_id", "base_model_version", "status", "num_samples",
               "local_epochs", "batch_size", "learning_rate", "local_loss_before", "local_loss_after",
               "local_accuracy_before", "local_accuracy_after", "delta_l2_norm", "float_delta_l2_norm",
               "quantization_error_l2", "delta_int8_scale", "payload_bytes", "train_time_ms", "class_counts",
               "protocol", "retries", "transfer_duration_s", "sent_at_utc", "received_at_utc",
               # from fl_update_transmissions.csv
               "scheduled_send_utc", "deadline_utc", "tx_start_utc", "tx_end_utc", "waited_offline_s",
               "lora_sf", "fragments", "airtime_s", "duty_cycle_wait_s", "radio_energy_mj", "delivered",
               # protocol-aware additions
               "compression_method", "staleness", "reliability_score", "aggregation_weight", "used_in_round",
               "run_id"]
STATE_COLS = ["round_id", "site_id", "device_id", "protocol", "round_opened_utc", "reliability",
              "expected_upload_s", "age", "participation", "selected", "run_id"]

def z(ts):
    """UTC timestamp as 2026-09-10T03:15:00Z (the data contract format); empty when missing."""
    if ts is None or pd.isna(ts):
        return ""
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

def as_contract(df):
    df = df.copy()
    for c in df.columns:
        if c.endswith("_utc"):
            df[c] = df[c].map(z)
    return df

def influx_counts(run_id, expected):
    from src.load_to_influx import client, ORG, BUCKET
    q = client.query_api()
    for meas, field, n in expected:
        flux = f'''from(bucket: "{BUCKET}")
          |> range(start: 2026-09-01T00:00:00Z, stop: 2026-09-20T00:00:00Z)
          |> filter(fn: (r) => r._measurement == "{meas}" and r._field == "{field}" and r.run_id == "{run_id}")
          |> group() |> count()'''
        t = q.query(flux)
        got = t[0].records[0].get_value() if t and t[0].records else 0
        print(f"  {meas:18s} influx={got:5d} file={n:5d}  {'OK' if got == n else 'MISMATCH'}")

def write_influx(glob, upd, state, run_id, delay):
    from src.load_to_influx import write_df
    glob = glob.assign(published_at_utc=pd.to_datetime(glob.published_at_utc, utc=True, format="ISO8601"))
    for r in sorted(glob.round_id.unique()):                 # one round at a time, like a live run
        write_df(glob[glob.round_id == r], "fl_global_models", "published_at_utc",
                 ["aggregation", "compression_method", "run_id"],
                 ["round_id", "global_model_version", "clients_selected", "clients_received", "clients_late",
                  "clients_failed", "total_samples", "global_val_loss", "global_val_accuracy",
                  "mean_update_l2_norm", "model_size_int8_bytes", "model_path"])
        u = upd[upd.round_id == r]
        if len(u):
            write_df(u, "fl_client_updates", "sent_at_utc",
                     ["site_id", "device_id", "protocol", "status", "compression_method", "run_id"],
                     ["update_id", "round_id", "base_model_version", "num_samples", "local_loss_before",
                      "local_loss_after", "local_accuracy_before", "local_accuracy_after", "delta_l2_norm",
                      "payload_bytes", "fragments", "retries", "airtime_s", "duty_cycle_wait_s",
                      "transfer_duration_s", "radio_energy_mj", "staleness", "reliability_score",
                      "aggregation_weight"])
        s = state[state.round_id == r] if len(state) else state
        if len(s):
            write_df(s, "fl_site_state", "round_opened_utc", ["site_id", "device_id", "protocol", "run_id"],
                     ["reliability", "expected_upload_s", "age", "participation", "selected"])
        if delay:
            time.sleep(delay)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full", choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--activation", default="tanh", choices=["relu", "tanh", "sigmoid"])
    ap.add_argument("--scenario", default="trace", choices=["trace", "stochastic", "harsh"])
    ap.add_argument("--influx", action="store_true", help="also write to InfluxDB")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between rounds when writing to InfluxDB")
    a = ap.parse_args()

    run_id = f"member3-{a.config}-{a.activation}-s{a.seed}-{a.scenario}"
    df = run_fl(a.seed, a.activation, scenario=a.scenario, verbose=False, save=False, **CONFIGS[a.config])
    out = RESULTS_DIR / "export" / run_id
    out.mkdir(parents=True, exist_ok=True)

    # model weights stay as JSON files; only the path goes into the tables / InfluxDB
    model_dir = ROOT / "models" / "runs" / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    for ver, weights in df.attrs["models"].items():
        with open(model_dir / f"global_{ver}.json", "w", encoding="utf-8") as f:
            json.dump({"version": ver, "weights": weights}, f)
    glob = df.copy()
    glob["model_path"] = glob.global_model_version.map(lambda v: f"models/runs/{run_id}/global_{v}.json")
    glob["run_id"] = run_id
    glob.loc[glob.round_id == 0, "aggregation"] = "init"       # as in the shared file
    upd = df.attrs["updates"].assign(run_id=run_id, train_time_ms=np.nan)
    for c in ("staleness", "used_in_round"):
        upd[c] = upd[c].astype("Int64")
    state = df.attrs["state"].rename(columns={"time_utc": "round_opened_utc"}).assign(run_id=run_id) \
        if len(df.attrs["state"]) else pd.DataFrame(columns=STATE_COLS)

    as_contract(glob[GLOBAL_COLS]).to_csv(out / "fl_global_models.csv", index=False)
    as_contract(upd[UPDATE_COLS]).to_csv(out / "fl_client_updates.csv", index=False)
    if len(state):
        as_contract(state[STATE_COLS]).to_csv(out / "fl_site_state.csv", index=False)

    shared = pd.read_csv(dl.DATA_DIR / "fl_global_models.csv").columns.tolist()
    print(f"exported {run_id} to {out}")
    print(f"  fl_global_models.csv : {len(glob)} rows, shared columns present: "
          f"{all(c in GLOBAL_COLS for c in shared)}")
    print(f"  fl_client_updates.csv: {len(upd)} rows")
    print(f"  fl_site_state.csv    : {len(state)} rows")
    print(f"  model files          : {len(df.attrs['models'])} in {model_dir}")

    if a.influx:
        print("writing to InfluxDB ...")
        write_influx(glob, upd, state, run_id, a.delay)
        influx_counts(run_id, [("fl_global_models", "round_id", len(glob)),
                               ("fl_client_updates", "round_id", len(upd)),
                               ("fl_site_state", "reliability", len(state))])

if __name__ == "__main__":
    main()
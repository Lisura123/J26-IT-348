import os
import pandas as pd
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from src import data_loader as dl

load_dotenv()
ORG, BUCKET = os.environ["INFLUX_ORG"], os.environ["INFLUX_BUCKET"]
client = InfluxDBClient(url=os.environ["INFLUX_URL"],
                        token=os.environ["INFLUX_TOKEN"], org=ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

def write_df(df, measurement, time_col, tags, fields):
    df = df.dropna(subset=[time_col])
    points = []
    for r in df.to_dict("records"):
        p = Point(measurement).time(r[time_col].to_pydatetime(), WritePrecision.S)
        for t in tags:
            if pd.notna(r.get(t)):
                p.tag(t, str(r[t]))
        for f in fields:
            v = r.get(f)
            if v is None or pd.isna(v):      # missing = leave out, NEVER zero
                continue
            p.field(f, v.item() if hasattr(v, "item") else v)
        points.append(p)
    write_api.write(bucket=BUCKET, org=ORG, record=points)
    print(f"{measurement}: wrote {len(points)} points")

def load_network():
    lm = dl.load_link_metrics()
    write_df(lm, "network_link_metrics", "timestamp_utc",
             ["site_id", "device_id", "protocol", "gateway_id", "link_quality"],
             ["online", "rssi_dbm", "snr_db", "lora_sf", "latency_ms",
              "packet_loss_pct", "throughput_kbps"])
    out = dl.load_outages()
    out["end_utc"] = out["end_utc"].astype(str)
    write_df(out, "network_outages", "start_utc",
             ["site_id", "device_id", "protocol", "cause"],
             ["duration_min", "end_utc"])

from pathlib import Path
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def load_fl_updates_measurement():
    up = dl.load_fl_updates()
    tx = dl.load_transmissions()[["update_id", "scheduled_send_utc", "fragments",
                                  "airtime_s", "duty_cycle_wait_s", "radio_energy_mj"]]
    df = up.merge(tx, on="update_id", how="left")
    df["time_utc"] = df["sent_at_utc"].fillna(df["scheduled_send_utc"])
    write_df(df, "fl_client_updates", "time_utc",
             ["site_id", "device_id", "protocol", "status"],
             ["update_id", "round_id", "base_model_version", "num_samples",
              "local_loss_before", "local_loss_after",
              "local_accuracy_before", "local_accuracy_after",
              "delta_l2_norm", "payload_bytes", "fragments", "retries",
              "airtime_s", "duty_cycle_wait_s", "transfer_duration_s",
              "radio_energy_mj"])

def load_global_models_measurement():
    gm = dl.load_global_models()
    # model weights stay in models/*.json; only the path goes to InfluxDB
    gm["model_path"] = gm["global_model_version"].apply(
        lambda v: f"models/global_{v}.json" if (MODELS_DIR / f"global_{v}.json").exists() else None)
    write_df(gm, "fl_global_models", "published_at_utc",
             ["aggregation"],
             ["round_id", "global_model_version", "clients_selected",
              "clients_received", "clients_late", "clients_failed",
              "total_samples", "global_val_loss", "global_val_accuracy",
              "mean_update_l2_norm", "model_size_int8_bytes", "model_path"])

def load_ota_measurement():
    st = dl.load_backend_status()
    write_df(st, "ota_deployments", "deployed_at_utc",
             ["site_id", "device_id", "protocol", "ota_status", "ota_method"],
             ["round_id", "global_model_version", "ota_bytes", "ota_retries",
              "ota_duration_s", "ota_waited_offline_s", "ota_rssi_dbm",
              "ota_packet_loss_pct", "ota_fragments"])

if __name__ == "__main__":      
    load_network()
    load_fl_updates_measurement()
    load_global_models_measurement()
    load_ota_measurement()
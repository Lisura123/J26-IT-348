import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient
from src import data_loader as dl

load_dotenv()
ORG, BUCKET = os.environ["INFLUX_ORG"], os.environ["INFLUX_BUCKET"]
client = InfluxDBClient(url=os.environ["INFLUX_URL"],
                        token=os.environ["INFLUX_TOKEN"], org=ORG)
q = client.query_api()

def influx_count(measurement, field):
    flux = f'''
    from(bucket: "{BUCKET}")
      |> range(start: 2026-09-01T00:00:00Z, stop: 2026-09-20T00:00:00Z)
      |> filter(fn: (r) => r._measurement == "{measurement}" and r._field == "{field}")
      |> group()
      |> count()
    '''
    tables = q.query(flux)
    return tables[0].records[0].get_value() if tables and tables[0].records else 0

up = dl.load_fl_updates()
checks = [
    ("network_link_metrics", "online", len(dl.load_link_metrics())),
    ("network_outages", "duration_min", len(dl.load_outages())),
    ("fl_client_updates", "round_id", len(up)),
    ("fl_global_models", "round_id", len(dl.load_global_models())),
    ("ota_deployments", "round_id",
     int(dl.load_backend_status()["deployed_at_utc"].notna().sum())),
]
for m, f, expected in checks:
    got = influx_count(m, f)
    print(f"{m:22s} influx={got:6d} file={expected:6d}  {'OK' if got == expected else 'MISMATCH'}")
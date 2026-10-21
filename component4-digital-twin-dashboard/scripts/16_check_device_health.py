import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dotenv import load_dotenv
from pymongo import MongoClient
from twin.device_health import explain_flag, explain_reading, split_flags

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
col_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("HEALTH_COLLECTION", "device_sensor_health")
site = sys.argv[2] if len(sys.argv) > 2 else "SITE-A"

db = MongoClient(os.getenv("MONGO_READER_URI"), serverSelectionTimeoutMS=10000)[os.getenv("MONGO_DB", "solar_pv")]
docs = list(db[col_name].find({"site_id": site}, {"_id": 0}).sort("timestamp_utc", -1).limit(500))
print(f"{col_name} · {site} · last {len(docs)} readings\n")

latest = docs[0]
r = explain_reading(latest)
print(f"LATEST {latest['timestamp_utc']} · {r['health_status']} · {r['status_text']}")
if r["context"]:
    print("  ", r["context"])
for i in r["issues"]:
    print(f"   [{i['severity']}] {i['sensor']}: {i['what']}")

counts = Counter(c for d in docs for c in split_flags(d.get("quality_flag")) if c != "OK")
print(f"\nIssues in the last {len(docs)} readings:")
if not counts:
    print("   none")
for code, n in counts.most_common():
    e = explain_flag(code)
    print(f"\n   {code} × {n}  [{e['severity']}] {e['sensor']}")
    print(f"     What happened : {e['what']}")
    print(f"     Check         : {e['check']}")
    print(f"     Twin impact   : {e['twin']}")
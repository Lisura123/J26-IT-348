import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
uri, db_name = os.getenv("MONGO_READER_URI"), os.getenv("MONGO_DB", "solar_pv")
if not uri:
    sys.exit("MONGO_READER_URI missing in .env")

db = MongoClient(uri, serverSelectionTimeoutMS=10000)[db_name]
print("Collections:", sorted(db.list_collection_names()))

name = sys.argv[1] if len(sys.argv) > 1 else "test_sensor_health"
col = db[name]
n = col.count_documents({})
print(f"\n{name}: {n} documents")
if n == 0:
    sys.exit("Empty - ask Member 1 / Member 3 whether health data has been ingested yet.")

print("\nLatest document:")
for k, v in col.find_one(sort=[("timestamp_utc", -1)], projection={"_id": 0}).items():
    print(f"  {k}: {v}")

fields = Counter()
for doc in col.find({}, {"_id": 0}).limit(1000):
    fields.update(doc.keys())
print("\nFields (in first 1000 docs):", dict(fields))

for f in ("health_status", "quality_flag"):
    if f in fields:
        top = col.aggregate([{"$group": {"_id": f"${f}", "n": {"$sum": 1}}}, {"$sort": {"n": -1}}, {"$limit": 15}])
        print(f"\n{f}:", {d["_id"]: d["n"] for d in top})

print("\nPer site:", {d["_id"]: d["n"] for d in col.aggregate([{"$group": {"_id": "$site_id", "n": {"$sum": 1}}}])})

import json
import math
import os

import requests
from dotenv import load_dotenv

load_dotenv()   # reads BACKEND_URL and BACKEND_API_KEY from the .env file

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
API_KEY = os.environ.get("BACKEND_API_KEY", "")

BATCH_SIZE = 500   # backend accepts max 5000 records and 5 MB per request


NUMBER_COLUMNS = [
    "voltage_v", "current_ma", "power_mw",
    "panel_temp1_c", "panel_temp2_c", "ambient_temp_c",
    "irradiance_lux",
    "dust_vo_v",              
    "dust_lpo_pct",           
    "health_score",           
]
TEXT_COLUMNS = ["record_id", "site_id", "device_id", "panel_id",
                "quality_flag", "health_status"]


def make_records(df):
    # turn the dataframe into a list of JSON records for the backend
    records = []
    for _, row in df.iterrows():
        rec = {}

        # time must be UTC text ending with "Z", or the backend rejects it
        rec["timestamp_utc"] = row["timestamp_utc"].strftime("%Y-%m-%dT%H:%M:%SZ")

        for col in TEXT_COLUMNS:
            if col in df.columns:
                rec[col] = str(row[col])

        for col in NUMBER_COLUMNS:
            if col in df.columns:
                value = row[col]
                # data contract rule: a missing value is LEFT OUT, never sent as 0
                if value is None or math.isnan(value):
                    continue
                rec[col] = round(float(value), 4)

        records.append(rec)
    return records


def send_to_backend(df, dry_run=False):
    # send all records in small batches
    # dry_run=True -> do not send, only save the first batch to a file to check it
    records = make_records(df)

    if dry_run:
        with open("dry_run_health.json", "w") as f:
            json.dump(records[:BATCH_SIZE], f, indent=2)
        print("Dry run: saved", min(len(records), BATCH_SIZE), "records to dry_run_health.json")
        return {"accepted": 0, "rejected": 0}

    if not API_KEY:
        raise RuntimeError("BACKEND_API_KEY is empty. Add it to your .env file.")

    url = BACKEND_URL + "/api/v1/ingest/health"
    headers = {"X-API-Key": API_KEY}
    total = {"accepted": 0, "rejected": 0}

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        response = requests.post(url, json=batch, headers=headers, timeout=30)
        response.raise_for_status()   # stop if the key is wrong or backend is down

        result = response.json()
        total["accepted"] += result.get("accepted", 0)
        total["rejected"] += result.get("rejected", 0)
        if result.get("rejected", 0) > 0:
            print("Rejected in this batch:", result.get("reasons"))

    print("Sent:", len(records), "| accepted:", total["accepted"], "| rejected:", total["rejected"])
    return total
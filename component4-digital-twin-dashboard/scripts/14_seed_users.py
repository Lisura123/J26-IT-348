import getpass
import sys
from datetime import datetime, timezone
from pathlib import Path

import bcrypt
import pandas as pd
from dotenv import dotenv_values
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ENV = ROOT.parent / "project-backend" / ".env"

env = dotenv_values(BACKEND_ENV)
uri, db_name = env.get("MONGO_URI"), env.get("MONGO_DB", "solar_pv")
if not uri:
    sys.exit(f"MONGO_URI not found in {BACKEND_ENV}")

users = pd.read_csv(ROOT / "data" / "dashboard_users.csv")
col = MongoClient(uri, serverSelectionTimeoutMS=10000)[db_name]["dashboard_users"]
col.create_index("email", unique=True)

for u in users.itertuples():
    while True:
        pw = getpass.getpass(f"Password for {u.email} ({u.role}): ")
        if len(pw) < 8:
            print("  at least 8 characters, try again")
        elif pw != getpass.getpass("  repeat: "):
            print("  passwords do not match, try again")
        else:
            break
    doc = {
        "user_id": u.user_id,
        "display_name": u.display_name,
        "email": u.email.lower(),
        "role": u.role,
        "site_access": u.site_access,
        "display_timezone": u.display_timezone,
        "active": str(u.active) == "True",
        "password_hash": bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode(),
        "updated_at_utc": datetime.now(timezone.utc),
    }
    col.replace_one({"email": doc["email"]}, doc, upsert=True)
    print(f"  saved {u.email}")

print(f"\n{col.count_documents({})} users in {db_name}.dashboard_users")
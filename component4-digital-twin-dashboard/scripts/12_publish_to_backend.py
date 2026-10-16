import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from dotenv import load_dotenv
from twin.config import OUT, ROOT
from twin.publish import publish

load_dotenv(ROOT / ".env")
url = os.getenv("BACKEND_URL", "http://localhost:8000")
key = os.getenv("BACKEND_API_KEY")
if not key:
    sys.exit("BACKEND_API_KEY missing - add it to .env")

out = pd.read_csv(OUT / "digital_twin_output.csv")
result = publish(out, url, key)
print(json.dumps(result, indent=1))
if result["written"] != result["sent"]:
    sys.exit("Not every record was written - check 'reasons' above")
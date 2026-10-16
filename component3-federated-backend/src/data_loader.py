from pathlib import Path
import json
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "shared"

def _read_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name)
    # Data contract: every *_utc column is UTC
    for col in df.columns:
        if col.endswith("_utc"):
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df

def load_sites():              return _read_csv("sites.csv")
def load_link_metrics():       return _read_csv("network_link_metrics.csv")
def load_outages():            return _read_csv("network_outages.csv")
def load_fl_updates():         return _read_csv("fl_client_updates.csv")
def load_transmissions():      return _read_csv("fl_update_transmissions.csv")
def load_global_models():      return _read_csv("fl_global_models.csv")
def load_backend_status():     return _read_csv("member3_backend_status.csv")

def load_weight_updates():
    rows = []
    with open(DATA_DIR / "fl_weight_updates.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

if __name__ == "__main__":
    print("sites          ", load_sites().shape)
    print("link metrics   ", load_link_metrics().shape)
    print("outages        ", load_outages().shape)
    print("fl updates     ", load_fl_updates().shape)
    print("transmissions  ", load_transmissions().shape)
    print("global models  ", load_global_models().shape)
    print("backend status ", load_backend_status().shape)
    print("weight updates ", len(load_weight_updates()))
    print(load_link_metrics()["timestamp_utc"].dtype)  # datetime64[ns, UTC]

def load_global_history():
    with open(DATA_DIR / "fl_global_model_history.json", encoding="utf-8") as f:
        return json.load(f)["versions"]
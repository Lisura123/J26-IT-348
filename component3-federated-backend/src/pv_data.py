import numpy as np
import pandas as pd
from src import data_loader as dl

FEATURES = ["voltage_v", "current_ma", "power_mw", "panel_temp1_c", "panel_temp2_c",
            "ambient_temp_c", "irradiance_lux", "dust_vo_v"]
LABELS = ["NORMAL", "PARTIAL_SHADING", "DUST", "OPEN_CIRCUIT", "SHORT_CIRCUIT"]
SITES = ["SITE-A", "SITE-B", "SITE-C", "SITE-D", "SITE-E", "SITE-F"]
LABEL_ID = {l: i for i, l in enumerate(LABELS)}

def load_pv():
    d = pd.read_csv(dl.DATA_DIR / "pv_fault_dataset.csv")
    d["timestamp_utc"] = pd.to_datetime(d["timestamp_utc"], utc=True)
    return d

class Scaler:
    """Standardiser fitted once on the pooled TRAINING window, then fixed for everyone."""
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self
    def transform(self, X):
        return (X - self.mean) / self.std

def get_data():
    """Returns (site_data, val, scaler). site_data[site] = (X, y); val = (X, y) from OPERATION."""
    d = load_pv()
    train = d[d["window"] == "TRAINING"]
    val = d[d["window"] == "OPERATION"]
    scaler = Scaler().fit(train[FEATURES].to_numpy(float))
    def xy(g):
        return (scaler.transform(g[FEATURES].to_numpy(float)),
                g["label"].map(LABEL_ID).to_numpy())
    site_data = {s: xy(train[train.site_id == s]) for s in SITES}
    return site_data, xy(val), scaler
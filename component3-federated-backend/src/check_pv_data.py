import numpy as np
import pandas as pd
from src import data_loader as dl

FEATURES = ["voltage_v", "current_ma", "power_mw", "panel_temp1_c", "panel_temp2_c",
            "ambient_temp_c", "irradiance_lux", "dust_vo_v"]
LABELS = ["NORMAL", "PARTIAL_SHADING", "DUST", "OPEN_CIRCUIT", "SHORT_CIRCUIT"]

d = pd.read_csv(dl.DATA_DIR / "pv_fault_dataset.csv")
d["timestamp_utc"] = pd.to_datetime(d["timestamp_utc"], utc=True)

print("window values:", d["window"].value_counts().to_dict())
print(pd.crosstab(d.site_id, d.window))
print("missing feature values:", int(d[FEATURES].isna().sum().sum()))
tr = d[d.window == "TRAINING"]
print("\nTRAINING window, rows by site and label")
print(pd.crosstab(tr.site_id, tr.label))
print("training time range:", tr.timestamp_utc.min(), "to", tr.timestamp_utc.max())

# --- does v1.0 reproduce the local_loss_before of round 1? ---
hist = dl.load_global_history() if hasattr(dl, "load_global_history") else None
import json
with open(dl.DATA_DIR / "fl_global_model_history.json", encoding="utf-8") as f:
    H = json.load(f)
mean, std = np.array(H["scaler_mean"]), np.array(H["scaler_std"])
w = {k: np.array(v, float) for k, v in H["versions"]["v1.0"].items()}
print("\nweight shapes:", {k: v.shape for k, v in w.items()})

def mat(M, n_in):
    return M if M.shape[0] == n_in else M.T

def forward(X, act):
    z = X @ mat(w["W1"], 8) + w["b1"]
    a = {"relu": lambda z: np.maximum(z, 0), "tanh": np.tanh,
         "sigmoid": lambda z: 1 / (1 + np.exp(-z))}[act](z)
    logits = a @ mat(w["W2"], 16) + w["b2"]
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
    return p

ups = dl.load_fl_updates() if hasattr(dl, "load_fl_updates") else pd.read_csv(dl.DATA_DIR / "fl_client_updates.csv")
r1 = ups[ups.round_id == 1].set_index("site_id")

print("\nsite  n_train  csv_loss  relu    tanh    sigmoid | csv_acc  relu_acc")
for site, g in tr.groupby("site_id"):
    X = (g[FEATURES].to_numpy(float) - mean) / std
    y = g["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    res = {}
    for act in ("relu", "tanh", "sigmoid"):
        p = forward(X, act)
        res[act] = (-np.log(p[np.arange(len(y)), y] + 1e-12).mean(), (p.argmax(1) == y).mean())
    c = r1.loc[site]
    print(f"{site}  {len(g):6d}  {c.local_loss_before:8.4f}  {res['relu'][0]:.4f}  "
          f"{res['tanh'][0]:.4f}  {res['sigmoid'][0]:.4f} | {c.local_accuracy_before:.4f}  {res['relu'][1]:.4f}")
import json
import numpy as np
import pandas as pd
from src import data_loader as dl

FEATURES = ["voltage_v", "current_ma", "power_mw", "panel_temp1_c", "panel_temp2_c",
            "ambient_temp_c", "irradiance_lux", "dust_vo_v"]
LABELS = ["NORMAL", "PARTIAL_SHADING", "DUST", "OPEN_CIRCUIT", "SHORT_CIRCUIT"]
ACTS = {"relu": lambda z: np.maximum(z, 0), "tanh": np.tanh,
        "sigmoid": lambda z: 1 / (1 + np.exp(-z))}

d = pd.read_csv(dl.DATA_DIR / "pv_fault_dataset.csv")
with open(dl.DATA_DIR / "fl_global_model_history.json", encoding="utf-8") as f:
    H = json.load(f)
mean, std = np.array(H["scaler_mean"]), np.array(H["scaler_std"])
csv = dl.load_global_models().set_index("global_model_version")

def evaluate(w, X, y, act):
    a = ACTS[act](X @ w["W1"] + w["b1"])
    z = a @ w["W2"] + w["b2"]
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    return -np.log(p[np.arange(len(y)), y] + 1e-12).mean(), (p.argmax(1) == y).mean()

print("OPERATION class balance:", d[d.window == "OPERATION"].label.value_counts().to_dict())
for window in ["OPERATION", "TRAINING"]:
    g = d[d.window == window]
    X = (g[FEATURES].to_numpy(float) - mean) / std
    y = g.label.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    print(f"\n{window} rows: {len(g)}")
    print("version  shared_val_acc | relu_acc  tanh_acc  sigmoid_acc")
    for ver in ["v1.0", "v1.5", "v1.10", "v1.20"]:
        w = {k: np.array(v, float) for k, v in H["versions"][ver].items()}
        accs = {a: evaluate(w, X, y, a)[1] for a in ACTS}
        print(f"{ver:7s}  {csv.loc[ver, 'global_val_accuracy']:14.4f} | "
              f"{accs['relu']:8.4f}  {accs['tanh']:8.4f}  {accs['sigmoid']:11.4f}")
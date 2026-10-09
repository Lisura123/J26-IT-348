import json
from pathlib import Path
import numpy as np
from src import data_loader as dl

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

updates_meta = dl.load_fl_updates().set_index("update_id")
rows = dl.load_weight_updates()
history = dl.load_global_history()
global_csv = dl.load_global_models().set_index("round_id")

LAYOUT = [(n, tuple(s)) for n, s in rows[0]["param_layout"]]

def flatten(w):
    return np.concatenate([np.asarray(w[n], float).ravel() for n, _ in LAYOUT])

def unflatten(vec):
    out, i = {}, 0
    for n, shape in LAYOUT:
        size = int(np.prod(shape))
        out[n] = vec[i:i + size].reshape(shape).tolist()
        i += size
    return out

def delta_int8(row):
    return np.asarray(row["delta_int8"], float) * row["int8_scale"]

def delta_float(row):
    return flatten(row["delta_float_device_only"])

def fedavg_step(w_old, deltas, samples):
    share = samples / samples.sum()            # n_i / N
    return w_old + (deltas * share[:, None]).sum(axis=0)

def rebuild(delta_fn, rounds=20):
    w = flatten(history["v1.0"])
    chain = {0: w}
    for r in range(1, rounds + 1):
        rs = [x for x in rows if x["round_id"] == r]
        samples = np.array([updates_meta.loc[x["update_id"], "num_samples"] for x in rs], float)
        deltas = np.array([delta_fn(x) for x in rs])
        w = fedavg_step(w, deltas, samples)
        chain[r] = w
    return chain

if __name__ == "__main__":
    f_chain = rebuild(delta_float)
    i_chain = rebuild(delta_int8)

    print("round  n  samples(csv)  |int8 - shared|  |float - shared|  |float - int8|")
    for r in range(1, 21):
        ref = flatten(history[f"v1.{r}"])
        n = sum(1 for x in rows if x["round_id"] == r)
        e_i = np.abs(i_chain[r] - ref).max()
        e_f = np.abs(f_chain[r] - ref).max()
        e_fi = np.abs(f_chain[r] - i_chain[r]).max()
        csv_n = int(global_csv.loc[r, "clients_received"])
        flag = "" if n == csv_n else "  <-- clients_received mismatch"
        print(f"{r:5d} {n:2d} {int(global_csv.loc[r,'total_samples']):12d}  "
              f"{e_i:14.1e}  {e_f:15.1e}  {e_fi:13.1e}{flag}")

    # Save each shared version as JSON (weights stay out of InfluxDB)
    for ver, w in history.items():
        with open(MODELS_DIR / f"global_{ver}.json", "w", encoding="utf-8") as f:
            json.dump({"version": ver, "weights": w}, f)
    print(f"\nSaved {len(history)} model files to {MODELS_DIR}")
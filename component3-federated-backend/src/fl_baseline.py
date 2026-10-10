"""Step 5: plain FedAvg baseline. Flower NumPyClient sites + round loop with network replay."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from flwr.client import NumPyClient

from src import data_loader as dl, mlp, pv_data
from src.network_replay import NetworkReplay

START = pd.Timestamp("2026-09-10T00:00:00Z")
ROUND_H, DEADLINE_H, N_ROUNDS = 6, 5, 20
PAYLOAD_BYTES = 297                       # int8 delta + header, as in the shared trace
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def to_list(p):
    return [p[n] for n, _ in mlp.LAYOUT]

def from_list(arrays):
    return {n: np.asarray(a, float) for (n, _), a in zip(mlp.LAYOUT, arrays)}

def quantize_int8(delta):
    """What the ESP32 really sends: int8 values plus one scale."""
    scale = np.abs(delta).max() / 127.0 or 1.0
    return np.clip(np.round(delta / scale), -127, 127) * scale

class SiteClient(NumPyClient):
    def __init__(self, site, X, y, activation, seed):
        self.site, self.X, self.y = site, X, y
        self.activation, self.seed = activation, seed
        self.site_idx = pv_data.SITES.index(site)

    def fit(self, parameters, config):
        rng = np.random.default_rng([self.seed, int(config["round"]), self.site_idx])
        new_p, _ = mlp.train_local(from_list(parameters), self.X, self.y, rng, self.activation)
        return to_list(new_p), len(self.y), {}

def check_trace(nr, trace):
    """Do our replay rules reproduce RECEIVED vs LATE for the shared (non failed) uploads?"""
    bad = total = 0
    for (r, site), t in trace.iterrows():
        if t.status == "FAILED":
            continue
        deadline = START + pd.Timedelta(hours=ROUND_H * r + DEADLINE_H)
        o = nr.replay(site, t.protocol, t.scheduled_send_utc, deadline, PAYLOAD_BYTES,
                      int(t.lora_sf) if t.protocol == "LORA" else None, int(t.retries))
        transfer = o["transfer_duration_s"] if o["transfer_duration_s"] is not None else float(t.transfer_duration_s)
        late = (o["tx_start_utc"] + pd.Timedelta(seconds=transfer)) > deadline
        total += 1
        bad += int(late != (t.status == "LATE"))
    print(f"replay status check: {total - bad}/{total} RECEIVED/LATE decisions match the shared trace")

def run_baseline(seed=0, activation="relu"):
    site_data, (Xv, yv), _ = pv_data.get_data()
    clients = {s: SiteClient(s, *site_data[s], activation, seed) for s in pv_data.SITES}
    nr = NetworkReplay()
    trace = dl.load_transmissions().set_index(["round_id", "site_id"])
    check_trace(nr, trace)

    w = mlp.init_params(np.random.default_rng(seed), activation)
    rows = []

    def log(r, published, sel, rec, late, failed, samples, l2):
        loss, acc, per = mlp.evaluate(w, Xv, yv, activation)
        row = dict(round_id=r, global_model_version=f"v1.{r}", aggregation="FedAvg",
                   clients_selected=sel, clients_received=rec, total_samples=samples,
                   global_val_loss=round(loss, 4), global_val_accuracy=round(acc, 4),
                   mean_update_l2_norm=round(l2, 5), published_at_utc=published,
                   model_size_int8_bytes=229, clients_late=late, clients_failed=failed,
                   compression_method="int8", bytes_attempted=PAYLOAD_BYTES * sel,
                   bytes_received=PAYLOAD_BYTES * rec)
        row.update({f"acc_{pv_data.LABELS[c]}": round(per[c], 4) for c in range(5)})
        rows.append(row)

    log(0, START.isoformat(), 0, 0, 0, 0, 0, 0.0)
    for r in range(1, N_ROUNDS + 1):
        opened = START + pd.Timedelta(hours=ROUND_H * r)
        deadline = opened + pd.Timedelta(hours=DEADLINE_H)
        w_vec = mlp.flatten(w)
        got, late, failed = [], 0, 0
        for site in pv_data.SITES:
            weights, n, _ = clients[site].fit(to_list(w), {"round": r})
            delta = quantize_int8(mlp.flatten(from_list(weights)) - w_vec)
            t = trace.loc[(r, site)]
            o = nr.replay(site, t.protocol, t.scheduled_send_utc, deadline, PAYLOAD_BYTES,
                          int(t.lora_sf) if t.protocol == "LORA" else None, int(t.retries))
            transfer = o["transfer_duration_s"] if o["transfer_duration_s"] is not None else float(t.transfer_duration_s)
            arrival = o["tx_start_utc"] + pd.Timedelta(seconds=transfer)
            if t.status == "FAILED":
                failed += 1
            elif arrival > deadline:
                late += 1                       # plain FedAvg discards late updates
            else:
                got.append((n, delta))
        l2 = 0.0
        if got:
            N = float(sum(n for n, _ in got))
            agg = sum((n / N) * d for n, d in got)       # FedAvg: w + sum (n_i / N) * delta_i
            w = mlp.unflatten(w_vec + agg)
            l2 = float(np.mean([np.linalg.norm(d) for _, d in got]))
        log(r, deadline.isoformat(), len(pv_data.SITES), len(got), late, failed,
            int(sum(n for n, _ in got)), l2)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"baseline_{activation}_seed{seed}.csv"
    df.to_csv(out, index=False)
    hit = df[df.global_val_accuracy >= 0.95]
    t95 = (pd.Timestamp(hit.iloc[0].published_at_utc) - START).total_seconds() / 3600 if len(hit) else None
    print(df[["round_id", "clients_received", "clients_late", "clients_failed",
              "global_val_loss", "global_val_accuracy"]].to_string(index=False))
    print(f"\nfinal accuracy {df.global_val_accuracy.iloc[-1]:.4f} | "
          f"time to 95%: {'not reached' if t95 is None else f'{t95:.0f} h'} | saved {out}")
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--activation", default="relu", choices=["relu", "tanh", "sigmoid"])
    a = ap.parse_args()
    run_baseline(a.seed, a.activation)
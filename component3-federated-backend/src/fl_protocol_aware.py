"""Step 6: FedAvg + protocol-aware compression with error feedback (same schedule and trace as the baseline)."""
import argparse
import numpy as np
import pandas as pd

from src import data_loader as dl, mlp, pv_data
from src.compression import compress
from src.fl_baseline import (START, ROUND_H, DEADLINE_H, N_ROUNDS, PAYLOAD_BYTES,
                             RESULTS_DIR, SiteClient, to_list, from_list)
from src.network_replay import NetworkReplay

def run(seed=0, activation="relu", error_feedback=True):
    site_data, (Xv, yv), _ = pv_data.get_data()
    clients = {s: SiteClient(s, *site_data[s], activation, seed) for s in pv_data.SITES}
    nr = NetworkReplay()
    trace = dl.load_transmissions().set_index(["round_id", "site_id"])
    w = mlp.init_params(np.random.default_rng(seed), activation)
    residual = {s: np.zeros(229) for s in pv_data.SITES}
    rows = []

    def log(r, published, rec, late, failed, samples, bytes_att, bytes_rec, lora_s, methods):
        loss, acc, per = mlp.evaluate(w, Xv, yv, activation)
        row = dict(round_id=r, global_model_version=f"v1.{r}", aggregation="FedAvg",
                   clients_selected=6 if r else 0, clients_received=rec, total_samples=samples,
                   global_val_loss=round(loss, 4), global_val_accuracy=round(acc, 4),
                   published_at_utc=published, clients_late=late, clients_failed=failed,
                   compression_method="protocol-aware", bytes_attempted=bytes_att,
                   bytes_received=bytes_rec, lora_transfer_s=round(lora_s, 1),
                   methods=",".join(methods))
        row.update({f"acc_{pv_data.LABELS[c]}": round(per[c], 4) for c in range(5)})
        rows.append(row)

    log(0, START.isoformat(), 0, 0, 0, 0, 0, 0, 0.0, [])
    for r in range(1, N_ROUNDS + 1):
        deadline = START + pd.Timedelta(hours=ROUND_H * r + DEADLINE_H)
        w_vec = mlp.flatten(w)
        got, late, failed = [], 0, 0
        b_att = b_rec = 0
        lora_s = 0.0
        methods = []
        for site in pv_data.SITES:
            weights, n, _ = clients[site].fit(to_list(w), {"round": r})
            x = mlp.flatten(from_list(weights)) - w_vec
            if error_feedback:
                x = x + residual[site]
            t = trace.loc[(r, site)]
            sf = int(t.lora_sf) if t.protocol == "LORA" else None
            sent, payload, method = compress(x, t.protocol, sf)
            if error_feedback:
                residual[site] = x - sent
            methods.append(f"{site[-1]}:{method}")
            o = nr.replay(site, t.protocol, t.scheduled_send_utc, deadline, payload, sf, int(t.retries))
            transfer = o["transfer_duration_s"] if o["transfer_duration_s"] is not None else float(t.transfer_duration_s)
            if t.protocol == "LORA":
                lora_s += transfer
            arrival = o["tx_start_utc"] + pd.Timedelta(seconds=transfer)
            b_att += payload
            if t.status == "FAILED":
                failed += 1
            elif arrival > deadline:
                late += 1
            else:
                got.append((n, sent)); b_rec += payload
        if got:
            N = float(sum(n for n, _ in got))
            w = mlp.unflatten(w_vec + sum((n / N) * d for n, d in got))
        log(r, deadline.isoformat(), len(got), late, failed, int(sum(n for n, _ in got)),
            b_att, b_rec, lora_s, methods)

    df = pd.DataFrame(rows)
    tag = "pa" if error_feedback else "pa_noef"
    out = RESULTS_DIR / f"{tag}_{activation}_seed{seed}.csv"
    df.to_csv(out, index=False)

    # baseline LoRa transfer time for the same trace, sending the dense 297 byte payload
    base_lora = 0.0
    for (r, site), t in trace.iterrows():
        if t.protocol == "LORA":
            deadline = START + pd.Timedelta(hours=ROUND_H * r + DEADLINE_H)
            o = nr.replay(site, t.protocol, t.scheduled_send_utc, deadline, PAYLOAD_BYTES,
                          int(t.lora_sf), int(t.retries))
            base_lora += o["transfer_duration_s"]
    hit = df[df.global_val_accuracy >= 0.95]
    t95 = (pd.Timestamp(hit.iloc[0].published_at_utc) - START).total_seconds() / 3600 if len(hit) else None
    print(df[["round_id", "clients_received", "global_val_loss", "global_val_accuracy",
              "bytes_attempted", "lora_transfer_s"]].to_string(index=False))
    print("\nmethods used in round 1:", df.methods.iloc[1])
    print(f"final accuracy {df.global_val_accuracy.iloc[-1]:.4f} | time to 95%: "
          f"{'not reached' if t95 is None else f'{t95:.0f} h'}")
    print(f"total bytes attempted: {int(df.bytes_attempted.sum())} (baseline {PAYLOAD_BYTES * 6 * N_ROUNDS})")
    print(f"total LoRa transfer time: {df.lora_transfer_s.sum():.0f} s (baseline {base_lora:.0f} s)")
    print("saved", out)
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--activation", default="relu", choices=["relu", "tanh", "sigmoid"])
    ap.add_argument("--no-error-feedback", action="store_true")
    a = ap.parse_args()
    run(a.seed, a.activation, not a.no_error_feedback)
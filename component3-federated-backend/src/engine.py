"""One FL engine for Steps 5 to 8: compression, scheduling and async aggregation are switches."""
import argparse
import numpy as np
import pandas as pd

from src import data_loader as dl, mlp, pv_data
from src.compression import compress, int8_dense, DENSE_INT8_BYTES
from src.fl_baseline import (START, ROUND_H, DEADLINE_H, N_ROUNDS, RESULTS_DIR,
                             SiteClient, to_list, from_list)
from src.network_replay import NetworkReplay, lora_airtime, DUTY_CYCLE
from src.scheduler import Scheduler

STALE_FLOOR = 0.25      # staleness factor never drops below this
FAIR_GAMMA = 1.0        # fairness boost: factor = 1 + GAMMA * (1 - delivery rate)

def link_stats(lm):
    out = {}
    for p, g in lm.groupby("protocol"):
        snr = g["snr_db"].dropna()
        out[p] = (g["rssi_dbm"].mean(), g["rssi_dbm"].std(),
                  snr.mean() if len(snr) else 0.0, snr.std() if len(snr) > 1 else 1.0)
    return out

def staleness_factor(tau):
    return max(STALE_FLOOR, 1.0 / (1.0 + tau))

def run_fl(seed=0, activation="tanh", compression="int8", error_feedback=False,
           scheduler=False, async_late=False, staleness=False, fairness=False,
           tag=None, verbose=True):
    site_data, (Xv, yv), _ = pv_data.get_data()
    clients = {s: SiteClient(s, *site_data[s], activation, seed) for s in pv_data.SITES}
    nr = NetworkReplay()
    trace = dl.load_transmissions().set_index(["round_id", "site_id"])
    sched = Scheduler(pv_data.SITES, link_stats(nr.lm)) if scheduler else None
    w = mlp.init_params(np.random.default_rng(seed), activation)
    residual = {s: np.zeros(229) for s in pv_data.SITES}
    ok = {s: 0 for s in pv_data.SITES}          # on-time deliveries so far
    tries = {s: 0 for s in pv_data.SITES}       # uploads attempted so far
    pending = []                                # late updates still on their way
    rows, state_rows = [], []
    agg_name = "FedAvg" if not (async_late or staleness or fairness) else "RelAsync"

    def encode(x, protocol, sf):
        if compression == "int8":
            return int8_dense(x), DENSE_INT8_BYTES, "int8"
        return compress(x, protocol, sf)

    def log(r, published, selected, rec, late, failed, late_used, samples, b_att, b_rec, lora_s):
        loss, acc, per = mlp.evaluate(w, Xv, yv, activation)
        row = dict(round_id=r, global_model_version=f"v1.{r}", aggregation=agg_name,
                   clients_selected=len(selected), clients_received=rec, total_samples=samples,
                   global_val_loss=round(loss, 4), global_val_accuracy=round(acc, 4),
                   published_at_utc=published, clients_late=late, clients_failed=failed,
                   late_used=late_used, compression_method=compression,
                   bytes_attempted=b_att, bytes_received=b_rec, lora_transfer_s=round(lora_s, 1),
                   selected_sites="".join(s[-1] for s in selected))
        row.update({f"acc_{pv_data.LABELS[c]}": round(per[c], 4) for c in range(5)})
        rows.append(row)

    log(0, START.isoformat(), [], 0, 0, 0, 0, 0, 0, 0, 0.0)
    for r in range(1, N_ROUNDS + 1):
        opened = START + pd.Timedelta(hours=ROUND_H * r)
        deadline = opened + pd.Timedelta(hours=DEADLINE_H)
        w_vec = mlp.flatten(w)

        feats, rel = {}, {}
        if sched:
            for s in pv_data.SITES:
                t = trace.loc[(r, s)]
                feats[s] = sched.features(t.protocol, nr.link_at(s, opened))
            selected, rel = sched.select(feats)
        else:
            selected = list(pv_data.SITES)

        got, late, failed, b_att, b_rec, lora_s = [], 0, 0, 0, 0, 0.0
        arrivals, outcome = [], {}
        for site in selected:
            weights, n, _ = clients[site].fit(to_list(w), {"round": r})
            x = mlp.flatten(from_list(weights)) - w_vec
            if error_feedback:
                x = x + residual[site]
            t = trace.loc[(r, site)]
            sf = int(t.lora_sf) if t.protocol == "LORA" else None
            sent, payload, _m = encode(x, t.protocol, sf)
            if error_feedback:
                residual[site] = x - sent
            o = nr.replay(site, t.protocol, t.scheduled_send_utc, deadline, payload, sf, int(t.retries))
            transfer = o["transfer_duration_s"] if o["transfer_duration_s"] is not None else float(t.transfer_duration_s)
            if t.protocol == "LORA":
                lora_s += transfer
            arrival = o["tx_start_utc"] + pd.Timedelta(seconds=transfer)
            b_att += payload
            rate = (ok[site] + 1) / (tries[site] + 2)       # delivery rate known BEFORE this upload
            tries[site] += 1
            if t.status == "FAILED":
                failed += 1; outcome[site] = False
            elif arrival > deadline:
                late += 1; outcome[site] = False
                if async_late:                               # keep it, it arrives after the deadline
                    pending.append(dict(site=site, n=n, delta=sent, base=r, arrival=arrival, rate=rate))
            else:
                got.append(dict(site=site, n=n, delta=sent, tau=0, rate=rate)); b_rec += payload
                outcome[site] = True; arrivals.append(arrival)
                ok[site] += 1

        close = deadline
        if sched and selected and all(outcome.values()):
            close = max(arrivals)

        late_used = 0
        if async_late:
            for p in [p for p in pending if p["arrival"] <= close]:
                got.append(dict(site=p["site"], n=p["n"], delta=p["delta"], tau=r - p["base"], rate=p["rate"]))
                late_used += 1
                pending.remove(p)

        if got:
            f = [1.0 + FAIR_GAMMA * (1.0 - g["rate"]) if fairness else 1.0 for g in got]
            s_ = [staleness_factor(g["tau"]) if staleness else 1.0 for g in got]
            norm = float(sum(g["n"] * fi for g, fi in zip(got, f)))
            step = sum(((g["n"] * fi * si) / norm) * g["delta"] for g, fi, si in zip(got, f, s_))
            w = mlp.unflatten(w_vec + step)

        if sched:
            for s in selected:
                sched.observe(s, feats[s], outcome[s])
            for s in pv_data.SITES:
                t = trace.loc[(r, s)]
                sf = int(t.lora_sf) if t.protocol == "LORA" else None
                payload = encode(np.zeros(229), t.protocol, sf)[1]
                exp = lora_airtime(payload, sf) / DUTY_CYCLE if t.protocol == "LORA" else 2.0
                end = nr.offline_until(s, opened)
                if end is not None:
                    exp += (end - opened).total_seconds()
                state_rows.append(dict(round_id=r, site_id=s, reliability=round(rel[s], 4),
                                       expected_upload_s=round(exp, 1), age=sched.age[s],
                                       participation=sched.part[s], selected=int(s in selected)))
        log(r, close.isoformat(), selected, len(got) - late_used, late, failed, late_used,
            int(sum(g["n"] for g in got)), b_att, b_rec, lora_s)

    df = pd.DataFrame(rows)
    tag = tag or (f"{compression}{'_ef' if error_feedback else ''}{'_sched' if scheduler else ''}"
                  f"{'_async' if async_late else ''}{'_stale' if staleness else ''}{'_fair' if fairness else ''}")
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"engine_{tag}_{activation}_seed{seed}.csv"
    df.to_csv(out, index=False)
    if state_rows:
        pd.DataFrame(state_rows).to_csv(RESULTS_DIR / f"site_state_{tag}_{activation}_seed{seed}.csv", index=False)
    if verbose:
        print(df[["round_id", "clients_selected", "clients_received", "late_used", "global_val_loss",
                  "global_val_accuracy", "selected_sites"]].to_string(index=False))
        print(f"\nfinal accuracy {df.global_val_accuracy.iloc[-1]:.4f}")
        print(f"total bytes attempted {int(df.bytes_attempted.sum())} | LoRa transfer {df.lora_transfer_s.sum():.0f} s"
              f" | late updates used {int(df.late_used.sum())}"
              + (f" | Jain fairness {sched.jain():.3f}" if sched else ""))
        print("saved", out)
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--activation", default="tanh", choices=["relu", "tanh", "sigmoid"])
    ap.add_argument("--compression", default="int8", choices=["int8", "protocol_aware"])
    ap.add_argument("--error-feedback", action="store_true")
    ap.add_argument("--scheduler", action="store_true")
    ap.add_argument("--async-late", action="store_true")
    ap.add_argument("--staleness", action="store_true")
    ap.add_argument("--fairness", action="store_true")
    a = ap.parse_args()
    run_fl(a.seed, a.activation, a.compression, a.error_feedback, a.scheduler,
           a.async_late, a.staleness, a.fairness)
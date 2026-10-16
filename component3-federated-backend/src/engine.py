"""One FL engine for Steps 5 to 10: compression, scheduling, async aggregation, network scenarios, records."""
import argparse
import numpy as np
import pandas as pd

from src import data_loader as dl, mlp, pv_data
from src.compression import compress, int8_dense, DENSE_INT8_BYTES
from src.fl_baseline import (START, ROUND_H, DEADLINE_H, N_ROUNDS, RESULTS_DIR,
                             SiteClient, to_list, from_list)
from src.network_replay import NetworkReplay, lora_airtime, lora_fragments, DUTY_CYCLE
from src import scenarios as scen
from src.scheduler import Scheduler

LORA_MW = 120.0          # LoRa radio power: radio_energy_mj / airtime_s = 120 in the shared trace
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
           scenario="trace", tag=None, verbose=True, save=True):
    site_data, (Xv, yv), _ = pv_data.get_data()
    clients = {s: SiteClient(s, *site_data[s], activation, seed) for s in pv_data.SITES}
    sp = scen.SCENARIOS[scenario]
    deadline_h = sp["deadline_h"] if sp else DEADLINE_H
    nr = NetworkReplay()
    if sp:
        nr.out = pd.concat([nr.out, scen.extra_outages(sp, seed, pv_data.SITES, START, ROUND_H * (N_ROUNDS + 1))],
                           ignore_index=True)
    base_transfer = dl.load_transmissions().groupby("protocol").transfer_duration_s.median()
    trace = dl.load_transmissions().set_index(["round_id", "site_id"])
    sched = Scheduler(pv_data.SITES, link_stats(nr.lm)) if scheduler else None
    w = mlp.init_params(np.random.default_rng(seed), activation)
    residual = {s: np.zeros(229) for s in pv_data.SITES}
    ok = {s: 0 for s in pv_data.SITES}          # on-time deliveries so far
    tries = {s: 0 for s in pv_data.SITES}       # uploads attempted so far
    pending = []                                # late updates still on their way
    rows, state_rows, updates = [], [], []
    models = {"v1.0": {n: np.asarray(v).tolist() for n, v in w.items()}}
    shared_tx = dl.load_transmissions()
    energy_ref = shared_tx.groupby("protocol").radio_energy_mj.median()     # WiFi / LTE energy per 297 byte upload
    class_counts = {s_: {pv_data.LABELS[c]: int((site_data[s_][1] == c).sum()) for c in range(5)}
                    for s_ in pv_data.SITES}
    agg_name = "FedAvg" if not (async_late or staleness or fairness) else "RelAsync"

    def encode(x, protocol, sf):
        if compression == "int8":
            return int8_dense(x), DENSE_INT8_BYTES, "int8"
        return compress(x, protocol, sf)

    def log(r, published, selected, rec, late, failed, late_used, samples, b_att, b_rec, lora_s, l2=0.0):
        loss, acc, per = mlp.evaluate(w, Xv, yv, activation)
        row = dict(round_id=r, global_model_version=f"v1.{r}", aggregation=agg_name,
                   clients_selected=len(selected), clients_received=rec, total_samples=samples,
                   global_val_loss=round(loss, 4), global_val_accuracy=round(acc, 4),
                   published_at_utc=published, clients_late=late, clients_failed=failed,
                   mean_update_l2_norm=round(l2, 5), model_size_int8_bytes=229,
                   late_used=late_used, compression_method=compression,
                   bytes_attempted=b_att, bytes_received=b_rec, lora_transfer_s=round(lora_s, 1),
                   selected_sites="".join(s[-1] for s in selected))
        row.update({f"acc_{pv_data.LABELS[c]}": round(per[c], 4) for c in range(5)})
        rows.append(row)

    log(0, START.isoformat(), [], 0, 0, 0, 0, 0, 0, 0, 0.0)
    for r in range(1, N_ROUNDS + 1):
        opened = START + pd.Timedelta(hours=ROUND_H * r)
        deadline = opened + pd.Timedelta(hours=deadline_h)
        w_vec = mlp.flatten(w)

        feats, rel = {}, {}
        if sched:
            for s in pv_data.SITES:
                t = trace.loc[(r, s)]
                link = nr.link_at(s, opened)
                if sp:
                    link = scen.harsh_link(link, sp, nr.offline_until(s, opened) is not None)
                feats[s] = sched.features(t.protocol, link)
            selected, rel = sched.select(feats)
        else:
            selected = list(pv_data.SITES)

        got, late, failed, b_att, b_rec, lora_s = [], 0, 0, 0, 0, 0.0
        arrivals, outcome = [], {}
        for site in selected:
            Xs, ys = site_data[site]
            lb, ab, _ = mlp.evaluate(w, Xs, ys, activation)
            weights, n, _ = clients[site].fit(to_list(w), {"round": r})
            la, aa, _ = mlp.evaluate(from_list(weights), Xs, ys, activation)
            x_raw = mlp.flatten(from_list(weights)) - w_vec
            x = x_raw + residual[site] if error_feedback else x_raw
            t = trace.loc[(r, site)]
            if sp is None:                                   # shared trace
                scheduled = t.scheduled_send_utc
                sf = int(t.lora_sf) if t.protocol == "LORA" else None
            else:                                            # generated events
                ctx = scen.draw_context(seed, r, pv_data.SITES.index(site), t.protocol,
                                        opened, deadline_h, nr, site, sp)
                scheduled, sf = ctx["scheduled"], ctx["sf"]
            sent, payload, _m = encode(x, t.protocol, sf)
            if error_feedback:
                residual[site] = x - sent
            if sp is None:
                retries, will_fail = int(t.retries), t.status == "FAILED"
            else:
                frags = lora_fragments(payload, sf) if t.protocol == "LORA" else 1
                retries, will_fail = scen.draw_retries(ctx["rng"], ctx["p"], frags, t.protocol)
            o = nr.replay(site, t.protocol, scheduled, deadline, payload, sf, retries)
            fallback = float(t.transfer_duration_s) if sp is None else float(base_transfer[t.protocol])
            transfer = o["transfer_duration_s"] if o["transfer_duration_s"] is not None else fallback
            if t.protocol == "LORA":
                lora_s += transfer
            arrival = o["tx_start_utc"] + pd.Timedelta(seconds=transfer)
            b_att += payload
            rate = (ok[site] + 1) / (tries[site] + 2)       # delivery rate known BEFORE this upload
            tries[site] += 1
            status = "FAILED" if will_fail else ("LATE" if arrival > deadline else "RECEIVED")
            airtime = o["airtime_s"]
            energy = LORA_MW * airtime if t.protocol == "LORA" else float(energy_ref[t.protocol]) * payload / 297.0
            rec = dict(update_id=f"R{r:02d}-{site}", round_id=r, site_id=site,
                       device_id=f"ESP32-{site[-1]}01", base_model_version=f"v1.{r - 1}", status=status,
                       num_samples=int(n), local_epochs=3, batch_size=32, learning_rate=0.05,
                       local_loss_before=round(lb, 4), local_loss_after=round(la, 4),
                       local_accuracy_before=round(ab, 4), local_accuracy_after=round(aa, 4),
                       delta_l2_norm=round(float(np.linalg.norm(sent)), 5),
                       float_delta_l2_norm=round(float(np.linalg.norm(x_raw)), 5),
                       quantization_error_l2=round(float(np.linalg.norm(x - sent)), 6),
                       delta_int8_scale=(round(float(np.abs(sent).max() / 127.0), 8) if _m != "float32" else None),
                       payload_bytes=int(payload), class_counts=str(class_counts[site]).replace("'", '"'),
                       protocol=t.protocol, retries=int(retries),
                       transfer_duration_s=round(float(transfer), 2),
                       scheduled_send_utc=scheduled, deadline_utc=deadline,
                       sent_at_utc=o["tx_start_utc"], received_at_utc=(arrival if status != "FAILED" else None),
                       tx_start_utc=o["tx_start_utc"], tx_end_utc=arrival,
                       waited_offline_s=round(float(o["waited_offline_s"]), 1),
                       lora_sf=sf, fragments=int(o["fragments"]),
                       airtime_s=(round(float(airtime), 3) if airtime is not None else None),
                       duty_cycle_wait_s=round(float(o["duty_cycle_wait_s"]), 1),
                       radio_energy_mj=round(float(energy), 1), delivered=status != "FAILED",
                       compression_method=_m, staleness=None,
                       reliability_score=(round(rel[site], 4) if sched else None),
                       aggregation_weight=0.0, used_in_round=None)
            updates.append(rec)
            if will_fail:
                failed += 1; outcome[site] = False
            elif arrival > deadline:
                late += 1; outcome[site] = False
                if async_late:                               # keep it, it arrives after the deadline
                    pending.append(dict(site=site, n=n, delta=sent, base=r, arrival=arrival, rate=rate, rec=rec))
            else:
                got.append(dict(site=site, n=n, delta=sent, tau=0, rate=rate, rec=rec)); b_rec += payload
                outcome[site] = True; arrivals.append(arrival)
                ok[site] += 1

        close = deadline
        if sched and selected and all(outcome.values()):
            close = max(arrivals)

        late_used = 0
        l2 = 0.0
        if async_late:
            for p in [p for p in pending if p["arrival"] <= close]:
                got.append(dict(site=p["site"], n=p["n"], delta=p["delta"], tau=r - p["base"], rate=p["rate"], rec=p["rec"]))
                b_rec += 0; late_used += 1
                pending.remove(p)

        if got:
            f = [1.0 + FAIR_GAMMA * (1.0 - g["rate"]) if fairness else 1.0 for g in got]
            s_ = [staleness_factor(g["tau"]) if staleness else 1.0 for g in got]
            norm = float(sum(g["n"] * fi for g, fi in zip(got, f)))
            step = sum(((g["n"] * fi * si) / norm) * g["delta"] for g, fi, si in zip(got, f, s_))
            w = mlp.unflatten(w_vec + step)
            for g, fi, si in zip(got, f, s_):
                g["rec"]["aggregation_weight"] = round((g["n"] * fi * si) / norm, 6)
                g["rec"]["staleness"] = g["tau"]
                g["rec"]["used_in_round"] = r
            l2 = float(np.mean([np.linalg.norm(g["delta"]) for g in got]))

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
                state_rows.append(dict(round_id=r, site_id=s, device_id=f"ESP32-{s[-1]}01", protocol=t.protocol, time_utc=opened,
                                       reliability=round(rel[s], 4),
                                       expected_upload_s=round(exp, 1), age=sched.age[s],
                                       participation=sched.part[s], selected=int(s in selected)))
        log(r, close.isoformat(), selected, len(got) - late_used, late, failed, late_used,
            int(sum(g["n"] for g in got)), b_att, b_rec, lora_s, l2)
        models[f"v1.{r}"] = {n: np.asarray(v).tolist() for n, v in w.items()}

    df = pd.DataFrame(rows)
    df.attrs["updates"] = pd.DataFrame(updates)
    df.attrs["state"] = pd.DataFrame(state_rows)
    df.attrs["models"] = models
    tag = tag or (f"{compression}{'_ef' if error_feedback else ''}{'_sched' if scheduler else ''}"
                  f"{'_async' if async_late else ''}{'_stale' if staleness else ''}{'_fair' if fairness else ''}{'' if scenario == 'trace' else '_' + scenario}")
    if not save:
        return df
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
    ap.add_argument("--scenario", default="trace", choices=list(scen.SCENARIOS))
    a = ap.parse_args()
    run_fl(a.seed, a.activation, a.compression, a.error_feedback, a.scheduler,
           a.async_late, a.staleness, a.fairness, a.scenario)
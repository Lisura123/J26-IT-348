"""Step 9: baseline vs full method over several seeds, plus the ablation study."""
import argparse
import numpy as np
import pandas as pd

from src import pv_data
from src.engine import run_fl
from src.fl_baseline import RESULTS_DIR
from src.metrics import time_to

BASE = dict(compression="int8", error_feedback=False, scheduler=False,
            async_late=False, staleness=False, fairness=False)
FULL = dict(compression="protocol_aware", error_feedback=True, scheduler=True,
            async_late=True, staleness=True, fairness=True)

def without(**off):
    d = dict(FULL)
    d.update(off)
    return d

MAIN = {"baseline": BASE, "full method": FULL}
ABLATION = {
    "full - compression": without(compression="int8"),
    "full - error feedback": without(error_feedback=False),
    "full - scheduler": without(scheduler=False),
    "full - late updates": without(async_late=False),
    "full - staleness": without(staleness=False),
    "full - fairness": without(fairness=False),
}

def run_metrics(df):
    last = df.iloc[-1]
    per = {f"acc_{l}": float(last[f"acc_{l}"]) for l in pv_data.LABELS}
    return dict(final_acc=float(last.global_val_accuracy), worst_class=min(per.values()),
                t90=time_to(df, 0.90), t95=time_to(df, 0.95),
                bytes=int(df.bytes_attempted.sum()), lora_s=float(df.lora_transfer_s.sum()),
                attempted=int(df.clients_selected.sum()),
                used=int((df.clients_received + df.late_used).sum()),
                lost=int((df.clients_failed + df.clients_late - df.late_used).sum()), **per)

def fmt(x, nd=4):
    sd = np.std(x, ddof=1) if len(x) > 1 else 0.0
    return f"{np.mean(x):.{nd}f} ± {sd:.{nd}f}"

def summarize(runs):
    out = []
    for name, g in runs.groupby("config", sort=False):
        n = len(g)
        row = {"config": name,
               "final accuracy": fmt(g.final_acc),
               "worst class acc": fmt(g.worst_class),
               "bytes": fmt(g.bytes, 0),
               "LoRa transfer s": fmt(g.lora_s, 0),
               "lost updates": fmt(g.lost, 1),
               "updates used": fmt(g.used, 1),}
        for col, label in (("t90", "time to 90% (h)"), ("t95", "time to 95% (h)")):
            ok = g[col].dropna()
            row[label] = "not reached" if len(ok) == 0 else f"{fmt(ok, 1)} [{len(ok)}/{n}]"
        out.append(row)
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="trace", choices=["trace", "stochastic", "harsh"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--activation", default="tanh", choices=["relu", "tanh", "sigmoid"])
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--only", default="", help='comma separated config names, e.g. "baseline,full method"')
    ap.add_argument("--tag", default="", help="suffix for the output files, keeps earlier results safe")
    a = ap.parse_args()

    configs = dict(MAIN)
    if a.ablation:
        configs.update(ABLATION)
    if a.only:
        keep = [x.strip() for x in a.only.split(",")]
        configs = {k: v for k, v in {**MAIN, **ABLATION}.items() if k in keep}
    suffix = f"_{a.tag}" if a.tag else ""

    rows, total, k = [], len(configs) * a.seeds, 0
    for name, kw in configs.items():
        for seed in range(a.seeds):
            k += 1
            print(f"[{k}/{total}] {a.scenario} | {name} | seed {seed}", flush=True)
            df = run_fl(seed, a.activation, scenario=a.scenario, verbose=False, save=False, **kw)
            rows.append(dict(config=name, seed=seed, **run_metrics(df)))
    runs = pd.DataFrame(rows)
    summ = summarize(runs)
    RESULTS_DIR.mkdir(exist_ok=True)
    runs.to_csv(RESULTS_DIR / f"eval_{a.scenario}{suffix}_runs.csv", index=False)
    summ.to_csv(RESULTS_DIR / f"eval_{a.scenario}{suffix}_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(f"\n=== {a.scenario} scenario, {a.seeds} seeds, mean ± std ===")
    print(summ.to_string(index=False))
    print("\nper-class final accuracy (mean over seeds):")
    print(runs.groupby("config", sort=False)[[f"acc_{l}" for l in pv_data.LABELS]].mean().round(3).to_string())
if __name__ == "__main__":
    main()
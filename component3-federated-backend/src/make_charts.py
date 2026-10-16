"""Step 11: charts for the PP1 slides. Runs baseline and full method on both scenarios, then plots."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import pv_data
from src.engine import run_fl
from src.evaluate import BASE, FULL
from src.fl_baseline import RESULTS_DIR

CHART_DIR = RESULTS_DIR / "charts"
COL = {"baseline": "#4c72b0", "full method": "#dd8452"}
SCEN_TITLE = {"trace": "Normal trace (shared files)", "harsh": "Harsh scenario"}

def collect(seeds):
    runs = {}
    for scen in ("trace", "harsh"):
        for name, kw in (("baseline", BASE), ("full method", FULL)):
            runs[(scen, name)] = []
            for seed in range(seeds):
                print(f"running {scen} | {name} | seed {seed}", flush=True)
                runs[(scen, name)].append(run_fl(seed, "tanh", scenario=scen, verbose=False, save=False, **kw))
    return runs

def bars(ax, values, title, ylabel, fmt="{:.0f}"):
    """values[(scenario, config)] = list over seeds."""
    x = np.arange(2)
    w = 0.36
    for i, name in enumerate(("baseline", "full method")):
        m = [np.mean(values[(s, name)]) for s in ("trace", "harsh")]
        sd = [np.std(values[(s, name)], ddof=1) if len(values[(s, name)]) > 1 else 0 for s in ("trace", "harsh")]
        b = ax.bar(x + (i - 0.5) * w, m, w, yerr=sd, capsize=4, color=COL[name], label=name)
        for xi, mi, si in zip(x + (i - 0.5) * w, m, sd):
            ax.text(xi, mi + si, fmt.format(mi), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Normal trace", "Harsh"])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    runs = collect(a.seeds)

    # 1. accuracy per round, mean +- std over seeds
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, scen in zip(axes, ("trace", "harsh")):
        for name in ("baseline", "full method"):
            acc = np.array([d.global_val_accuracy.to_numpy() for d in runs[(scen, name)]])
            m, sd = acc.mean(0), acc.std(0, ddof=1) if len(acc) > 1 else np.zeros(acc.shape[1])
            r = np.arange(len(m))
            ax.plot(r, m, color=COL[name], label=name, lw=2)
            ax.fill_between(r, m - sd, m + sd, color=COL[name], alpha=0.2)
        ax.set_title(SCEN_TITLE[scen], fontsize=11)
        ax.set_xlabel("FL round (one round every 6 h)")
        ax.set_xticks(range(0, 21, 2))
        ax.set_ylim(0.15, 1.0)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Global validation accuracy")
    axes[0].legend(loc="lower right")
    fig.suptitle(f"Accuracy per round, mean and standard deviation over {a.seeds} seeds", fontsize=12)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "accuracy_curves.png", dpi=160)
    plt.close(fig)

    # 2. communication and participation
    metric = {
        "lora": lambda d: d.lora_transfer_s.sum(),
        "bytes": lambda d: d.bytes_attempted.sum(),
        "used": lambda d: (d.clients_received + d.late_used).sum(),
    }
    vals = {k: {key: [f(d) for d in v] for key, v in runs.items()} for k, f in metric.items()}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    bars(axes[0], vals["lora"], "LoRa transfer time (lower is better)", "seconds, all rounds")
    bars(axes[1], vals["bytes"], "Total bytes attempted", "bytes")
    bars(axes[2], vals["used"], "Updates aggregated (of 120 possible)", "updates")
    axes[0].legend(frameon=False)
    fig.suptitle("Communication cost and participation", fontsize=12)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "communication.png", dpi=160)
    plt.close(fig)

    # 3. per-class final accuracy under the harsh scenario
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(5)
    for i, name in enumerate(("baseline", "full method")):
        per = np.array([[d.iloc[-1][f"acc_{l}"] for l in pv_data.LABELS] for d in runs[("harsh", name)]])
        sd = per.std(0, ddof=1) if len(per) > 1 else np.zeros(5)
        ax.bar(x + (i - 0.5) * 0.36, per.mean(0), 0.36, yerr=sd, capsize=3, color=COL[name], label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", "\n") for l in pv_data.LABELS], fontsize=8)
    ax.set_ylabel("Final accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class accuracy, harsh scenario (errors concentrate in two classes)", fontsize=10)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "per_class_harsh.png", dpi=160)
    plt.close(fig)

    # 4. ablation, from the saved 5 seed evaluation
    path = RESULTS_DIR / "eval_harsh_runs.csv"
    if path.exists():
        ev = pd.read_csv(path)
        order = list(ev.config.unique())[::-1]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
        for ax, col, title, xl in zip(axes, ("final_acc", "lora_s", "used"),
                                      ("Final accuracy", "LoRa transfer time", "Updates aggregated"),
                                      ("accuracy", "seconds", "updates")):
            g = ev.groupby("config")[col]
            m, sd = g.mean().reindex(order), g.std(ddof=1).reindex(order).fillna(0)
            ax.barh(order, m, xerr=sd, color="#8da0cb", capsize=3)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(xl)
            if col == "final_acc":
                ax.set_xlim(max(0.8, float((m - sd).min()) - 0.02), 1.0)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Ablation under the harsh scenario (mean and standard deviation over seeds)", fontsize=12)
        fig.tight_layout()
        fig.savefig(CHART_DIR / "ablation_harsh.png", dpi=160)
        plt.close(fig)
    print("charts saved in", CHART_DIR)

if __name__ == "__main__":
    main()
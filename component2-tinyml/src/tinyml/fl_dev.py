"""Development comparison of docs/FL_EVALUATION_PROTOCOL.md (Integration Task 7).

Per outer fold k of the committed run-wise split (results/splits_runwise_v1.json):

  EVAL_k   the TRAINING runs with cv_fold == k: scored once, after all training
  POOL_k   the other TRAINING runs; per site, ordered by first timestamp (ties by run_id):
    INIT_k the earliest floor(n/2) runs of each site
    FL_k   the remaining runs of each site
  pre-flight (protocol 4.3) stops with BLOCKED_EMPTY_SITE or INVALID_FOLD; runs are never moved
  cfg_k = features.fit_pipeline(INIT_k), frozen for every model of the fold

  M0       official recipe (models.make_model("mlp")) on INIT_k
  LOCAL-s  M0 + Member 3's mlp.train_local on site s's FL_k rows (60 epochs)
  FL       M0 + Member 3's engine.run_fl over FL_k (20 rounds; engine change PENDING SAMUDI REVIEW)
  CENT-FT  M0 + mlp.train_local on all FL_k rows pooled (60 epochs)
  CENT     official recipe on INIT_k + FL_k with cfg_k (sensitivity: cfg refitted on POOL_k)

Engine monitoring set = INIT_k rows (Option A). They were used to train M0, so the
engine's global_val_* numbers are TRAINING-DERIVED DIAGNOSTICS ONLY: never validation
accuracy, never used to choose a round, tune, choose a model or claim improvement.
EVAL_k is scored only at the fixed final round.

These are development models (DEVELOPMENT_MODEL_NOT_OFFICIAL); Model v1 and its
artifacts are never read for training and never written. OPERATION rows are dropped
on entry (fl_data.training_rows). Data: the PV Research Dataset (provenance as in
docs/DATA_PROVENANCE.md) or, in tests, software-generated Test Data. FL sites are
simulated partitions; communication numbers come from replayed traces.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from . import fl_data as fd
from . import fl_run
from .contract import GROUP_COLUMN, LABEL_COLUMN, LABELS
from .evaluate import classification_metrics, run_level
from .export import extract_weights, flatten, forward, weights_sha256
from .features import FeatureConfig, STATUS_OK, fit_pipeline, input_status, transform
from .fl_deploy import CONFIDENCE_THRESHOLD, _create, confidence_crossings
from .models import make_model
from .quantize import compare_predictions
from .splits import load_assignments, run_table

PROTOCOL = "docs/FL_EVALUATION_PROTOCOL.md"
ROUNDS = 20                        # protocol section 6 (Member 3's N_ROUNDS)
LOCAL_EPOCHS = 60                  # 20 rounds x 3 local epochs (protocol section 6)
SGD = {"batch": 32, "lr": 0.05}    # Member 3's train_local defaults
SEEDS = (0, 1, 2)
FOLDS = (0, 1, 2, 3, 4)
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 2000, 0
MODEL_LABEL = "DEVELOPMENT_MODEL_NOT_OFFICIAL"
DEV_RUNS_DIR = fd.COMPONENT / "fl_dev_runs"         # git-ignored, create-only
MONITOR_NOTE = ("engine global_val_* use INIT_k rows, which trained M0: TRAINING-DERIVED DIAGNOSTICS ONLY; "
                "not validation accuracy; never used to select a round, tune, select a model or claim improvement")
COMM_NOTE = "SIMULATED: replayed network traces in Member 3's engine, not radio measurements"
BUDGET_NOTE = ("Equal passes over the data per participant, not equal compute or equal learning: FL sites restart "
               "from the global model each round and late/failed updates are discarded; LOCAL-s does the work of "
               "one FL site on its own; CENT-FT does the combined FL work on pooled data in one sequence; M0 and "
               "CENT use Adam until convergence (official recipe), the others SGD. Step counts are recorded.")


class PreflightError(RuntimeError):
    """Protocol 4.3 failure: the fold is not run and nothing is repaired."""

    def __init__(self, code, detail):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


# ---------------------------------------------------------------- groups and pre-flight

def fold_groups(tr, assignments, k):
    """EVAL / INIT / FL run ids of fold k (protocol 4.1). tr: TRAINING rows only."""
    if set(tr["window"]) != {"TRAINING"}:
        raise ValueError("fold_groups needs TRAINING rows only")
    runs = run_table(tr)
    runs["start"] = tr.groupby(GROUP_COLUMN)["timestamp_utc"].min()
    missing = set(runs.index) - set(assignments.index)
    if missing:
        raise ValueError(f"{len(missing)} runs are not in the split file")
    runs["fold"] = assignments.loc[runs.index, "cv_fold"].astype(int)
    eval_runs = set(runs.index[runs.fold == k])
    init, fl = set(), {}
    for site, g in runs[runs.fold != k].groupby("site_id"):
        order = list(g.reset_index().sort_values(["start", GROUP_COLUMN])[GROUP_COLUMN])   # ties by run_id
        m = len(order) // 2
        init |= set(order[:m])
        fl[site] = order[m:]
    return {"fold": k, "EVAL": sorted(eval_runs), "INIT": sorted(init), "FL": {s: sorted(v) for s, v in fl.items()}}


def preflight(tr, groups, sites):
    """Protocol 4.3 checks. Returns the pre-flight table or raises PreflightError."""
    ev, ini = set(groups["EVAL"]), set(groups["INIT"])
    fl_all = {r for v in groups["FL"].values() for r in v}
    all_runs = set(tr[GROUP_COLUMN])
    if ev & ini or ev & fl_all or ini & fl_all or (ev | ini | fl_all) != all_runs:
        raise PreflightError("INVALID_FOLD", "EVAL, INIT and FL must be disjoint and cover every TRAINING run")
    ok = (input_status(tr, FeatureConfig()) == STATUS_OK).to_numpy()   # gates are fixed constants
    by_run = tr.assign(_ok=ok).groupby(GROUP_COLUMN)
    ok_rows = by_run["_ok"].sum()
    for s in sites:
        if not groups["FL"].get(s) or ok_rows.loc[groups["FL"][s]].sum() == 0:
            raise PreflightError("BLOCKED_EMPTY_SITE", f"{s} has no status-OK FL readings in fold {groups['fold']}")
    init_df = tr[tr[GROUP_COLUMN].isin(ini) & ok]
    missing = [l for l in LABELS if not (init_df[LABEL_COLUMN] == l).any()]
    if missing:
        raise PreflightError("INVALID_FOLD", f"INIT lacks classes {missing}")
    normal = int((init_df[LABEL_COLUMN] == "NORMAL").sum())
    if normal < 3:
        raise PreflightError("INVALID_FOLD", f"INIT has {normal} status-OK NORMAL rows; V_ref needs 3")
    return {"fold": groups["fold"], "status": "OK", "eval_runs": len(ev), "init_runs": len(ini), "fl_runs": len(fl_all),
            "fl_runs_per_site": {s: len(groups["FL"].get(s, [])) for s in sites},
            "init_class_runs": {l: int(tr[tr[GROUP_COLUMN].isin(ini)].groupby(GROUP_COLUMN)[LABEL_COLUMN].first()
                                       .eq(l).sum()) for l in LABELS},
            "init_normal_ok_rows": normal}


# ---------------------------------------------------------------- models

def _rows(tr, run_ids):
    return tr[tr[GROUP_COLUMN].isin(set(run_ids))]


def fit_mlp(part, cfg, seed):
    """Official recipe (models.make_model("mlp")) on status-OK rows, given cfg; weights in LABELS order."""
    X, status = transform(part, cfg)
    y = part.loc[(status == STATUS_OK).to_numpy(), LABEL_COLUMN].to_numpy()
    return extract_weights(make_model("mlp", seed=seed).fit(X, y))


def member3_mlp():
    """Member 3's src/mlp.py (train_local is the same function the FL sites run)."""
    return fd.load_member3_module("mlp")


def sgd_from(weights, part, cfg, rng_key, epochs=LOCAL_EPOCHS):
    """Member 3's mlp.train_local from a copy of weights; returns (weights, SGD steps)."""
    X, y, _ = fd._xy(part, cfg)
    new, _ = member3_mlp().train_local({k: v.copy() for k, v in weights.items()}, X, y,
                                       np.random.default_rng(rng_key), "tanh", epochs=epochs, **SGD)
    return new, int(epochs * -(-len(y) // SGD["batch"]))


def engine_data(tr, groups, cfg, sites):
    """(site_data FL_k, monitoring set INIT_k, cfg) for engine.run_fl."""
    site_data = {s: fd._xy(_rows(tr, groups["FL"][s]), cfg)[:2] for s in sites}
    monitor = fd._xy(_rows(tr, groups["INIT"]), cfg)[:2]
    return site_data, monitor, cfg


def communication(df):
    u = df.attrs["updates"]
    return {"note": COMM_NOTE, "rounds": int(df.round_id.max()),
            "bytes_attempted": int(df.bytes_attempted.sum()), "bytes_received": int(df.bytes_received.sum()),
            "updates": {k: int(v) for k, v in u.status.value_counts().items()},
            "lora_transfer_s": float(df.lora_transfer_s.sum()), "radio_energy_mj": float(u.radio_energy_mj.sum())}


def per_round(df, m0, X_pool):
    """Monitoring records (never used to choose anything)."""
    p0 = forward(m0, X_pool)[1].argmax(axis=1)
    out = []
    for _, r in df.iterrows():
        w = {k: np.asarray(v) for k, v in df.attrs["models"][r.global_model_version].items()}
        out.append({"round": int(r.round_id), "drift_l2_from_m0": float(np.linalg.norm(flatten(w) - flatten(m0))),
                    "label_change_rate_vs_m0_on_pool": float((forward(w, X_pool)[1].argmax(axis=1) != p0).mean()),
                    "monitor_loss_training_derived": float(r.global_val_loss),
                    "clients_received": int(r.clients_received), "clients_late": int(r.clients_late),
                    "clients_failed": int(r.clients_failed), "mean_update_l2_norm": float(r.mean_update_l2_norm)})
    return out


# ---------------------------------------------------------------- evaluation (EVAL_k only, after training)

def evaluate_models(models, eval_df):
    """models: {name: (weights, cfg)}; all scored on the same EVAL readings."""
    rows, preds, ok_ref = {}, {}, None
    for name, (w, cfg) in models.items():
        X, status = transform(eval_df, cfg)
        ok = (status == STATUS_OK).to_numpy()
        if ok_ref is None:
            ok_ref = ok
        elif not np.array_equal(ok, ok_ref):
            raise ValueError(f"{name} would be scored on different readings")
        preds[name] = forward(w, X)[1]
    part = eval_df[ok_ref]
    y, run_ids, sites = part[LABEL_COLUMN].to_numpy(), part[GROUP_COLUMN].to_numpy(), part["site_id"].to_numpy()
    p0 = preds["M0"]
    for name, p in preds.items():
        idx = p.argmax(axis=1)
        lab = np.array([LABELS[i] for i in idx])
        m = classification_metrics(y, lab)
        rows[name] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "per_class": m["per_class"],
                      "open_circuit_recall": m["safety_recall"]["OPEN_CIRCUIT"],
                      "short_circuit_recall": m["safety_recall"]["SHORT_CIRCUIT"],
                      "confusion_matrix": m["confusion_matrix"], "run_level": run_level(y, lab, run_ids),
                      "confidence_at_or_above_0_90": float((p.max(axis=1) >= CONFIDENCE_THRESHOLD).mean()),
                      "changes_vs_m0": compare_predictions(p0.argmax(axis=1), idx),
                      "confidence_crossings_vs_m0": confidence_crossings(p0, p),
                      "per_site": {s: {"readings": int((sites == s).sum()),
                                       "runs": int(len(set(run_ids[sites == s]))),
                                       "correct": int((lab[sites == s] == y[sites == s]).sum())}
                                   for s in sorted(set(sites))}}
    table = {"run_id": run_ids.tolist(), "site_id": sites.tolist(), "y": y.tolist(),
             **{n: [LABELS[i] for i in p.argmax(axis=1)] for n, p in preds.items()}}
    return rows, table


def paired_run_bootstrap(tables, a, b, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Macro-F1(a) - macro-F1(b) over pooled out-of-fold tables, resampling whole
    runs (each table is one fold x seed; a run keeps all its readings)."""
    y = np.concatenate([t["y"] for t in tables])
    pa = np.concatenate([t[a] for t in tables])
    pb = np.concatenate([t[b] for t in tables])
    runs = np.concatenate([t["run_id"] for t in tables])
    uniq = np.unique(runs)
    idx = {r: np.flatnonzero(runs == r) for r in uniq}

    def diff(sel):
        labels = [l for l in LABELS if (y[sel] == l).any()]
        return (f1_score(y[sel], pa[sel], labels=labels, average="macro", zero_division=0)
                - f1_score(y[sel], pb[sel], labels=labels, average="macro", zero_division=0))
    rng = np.random.default_rng(seed)
    boots = np.array([diff(np.concatenate([idx[r] for r in rng.choice(uniq, len(uniq))])) for _ in range(resamples)])
    return {"a": a, "b": b, "metric": "macro_f1", "observed_difference": float(diff(np.arange(len(y)))),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "runs": int(len(uniq)), "resamples": resamples, "seed": seed}


# ---------------------------------------------------------------- one fold x seed

def run_fold(df, assignments, k, seed, engine=None, rounds=ROUNDS, sites=None):
    """All models of fold k for one seed. EVAL_k is used only in the last step."""
    tr = fd.training_rows(df)                                   # OPERATION dropped here
    engine = engine or fl_run.member3_engine()
    if engine.N_ROUNDS != rounds:
        raise ValueError(f"engine N_ROUNDS {engine.N_ROUNDS} != fixed final round {rounds}")
    sites = list(sites or engine.pv_data.SITES)
    groups = fold_groups(tr, assignments, k)
    table = preflight(tr, groups, sites)
    init_df = _rows(tr, groups["INIT"])
    fl_df = {s: _rows(tr, groups["FL"][s]) for s in sites}
    fl_all = _rows(tr, [r for s in sites for r in groups["FL"][s]])
    pool = _rows(tr, set(groups["INIT"]) | {r for s in sites for r in groups["FL"][s]})
    use = {"fit_pipeline": groups["INIT"], "M0": groups["INIT"], "FL": sorted(set(fl_all[GROUP_COLUMN])),
           "CENT": sorted(set(pool[GROUP_COLUMN])), "CENT_FT": sorted(set(fl_all[GROUP_COLUMN])),
           "LOCAL": {s: groups["FL"][s] for s in sites}, "engine_monitor": groups["INIT"]}

    cfg = fit_pipeline(init_df, fold=f"dev_fold{k}_INIT")       # frozen for the fold
    m0 = fit_mlp(init_df, cfg, seed)
    m0_sha = weights_sha256(m0)
    models = {"M0": (m0, cfg)}
    steps = {}
    for i, s in enumerate(sites):
        w, steps[f"LOCAL-{s}"] = sgd_from(m0, fl_df[s], cfg, [seed, k, 1000 + i])
        models[f"LOCAL-{s}"] = (w, cfg)
    hist = fl_run.run_engine(m0, engine_data(tr, groups, cfg, sites), seed=seed, engine=engine)
    fl_w = {n: np.asarray(v, dtype=np.float64) for n, v in hist.attrs["models"][f"v1.{rounds}"].items()}
    if not all(np.array_equal(np.asarray(hist.attrs["models"]["v1.0"][n]), m0[n]) for n in m0):
        raise AssertionError("the engine did not start from M0")
    models["FL"] = (fl_w, cfg)
    steps["FL_per_site_per_round"] = {s: int(3 * -(-len(fd._xy(fl_df[s], cfg)[1]) // SGD["batch"])) for s in sites}
    w, steps["CENT-FT"] = sgd_from(m0, fl_all, cfg, [seed, k, 2000])
    models["CENT-FT"] = (w, cfg)
    models["CENT"] = (fit_mlp(pool, cfg, seed), cfg)
    cfg_pool = fit_pipeline(pool, fold=f"dev_fold{k}_POOL")      # sensitivity analysis only
    models["CENT-POOLCFG"] = (fit_mlp(pool, cfg_pool, seed), cfg_pool)
    if weights_sha256(m0) != m0_sha:
        raise AssertionError("M0 changed while training the other models")

    X_pool = transform(pool, cfg)[0]
    rounds_log = per_round(hist, m0, X_pool)
    eval_df = _rows(tr, groups["EVAL"])                          # first and only use of EVAL_k
    metrics, table_eval = evaluate_models(models, eval_df)
    use["EVAL_scored"] = groups["EVAL"]
    return {"protocol": PROTOCOL, "label": MODEL_LABEL, "fold": k, "seed": seed, "final_round": rounds,
            "preflight": table, "groups": groups, "data_use": use, "monitor_note": MONITOR_NOTE,
            "budget_note": BUDGET_NOTE, "sgd_steps": steps, "communication": communication(hist),
            "rounds": rounds_log, "metrics": metrics, "eval_table": table_eval,
            "weights": {n: {k_: v.tolist() for k_, v in w.items()} for n, (w, _) in models.items()},
            "engine_change": "engine.run_fl(data, init_weights) PENDING SAMUDI REVIEW"}


def write_result(result, run_id, allowed_root=DEV_RUNS_DIR):
    """<allowed_root>/<run_id>/fold<k>_seed<s>.json, create-only, never in a protected folder."""
    from .global_update import candidate_version
    candidate_version(run_id)                                    # path-safe characters only
    d = fl_run.safe_target(Path(allowed_root) / run_id, allowed_root)
    d.mkdir(parents=True, exist_ok=True)
    p = fl_run.safe_target(d / f"fold{result['fold']}_seed{result['seed']}.json", allowed_root)
    _create(p, json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    return p


def run_protocol(dataset_path, run_id, confirm_protocol_run=False, engine=None, allowed_root=DEV_RUNS_DIR):
    """Full 5-fold x 3-seed run on the PV Research Dataset. Refuses unless
    confirm_protocol_run=True (a separate, approved step)."""
    if confirm_protocol_run is not True:
        raise PermissionError("the protocol run needs explicit approval: confirm_protocol_run=True")
    df = fd.load_dataset(dataset_path)
    assignments, _ = load_assignments(fd.SPLITS_PATH, dataset_sha256=fd.file_sha256(dataset_path))
    tables, paths = [], []
    for k in FOLDS:
        for s in SEEDS:
            r = run_fold(df, assignments, k, s, engine=engine)
            paths.append(write_result(r, run_id, allowed_root))
            tables.append(r["eval_table"])
    summary = {"protocol": PROTOCOL, "primary": paired_run_bootstrap(tables, "FL", "M0"),
               "secondary": [paired_run_bootstrap(tables, "FL", b) for b in ("CENT", "CENT-FT")]}
    return summary, paths

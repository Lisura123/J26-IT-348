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
from importlib import metadata
from pathlib import Path

import numpy as np

from . import fl_data as fd
from . import fl_run
from .contract import GROUP_COLUMN, LABEL_COLUMN, LABELS
from .evaluate import SAFETY_CLASSES, classification_metrics, run_level
from .export import (ACTIVATION, N_PARAMS, OUTPUT_ACTIVATION, PARAM_LAYOUT, extract_weights, flatten, forward,
                     verify_artifacts, weights_sha256)
from .features import FEATURE_VERSION, FEATURES, FeatureConfig, STATUS_OK, fit_pipeline, input_status, transform
from .fl_deploy import CONFIDENCE_THRESHOLD, _create, confidence_crossings
from .models import make_model
from .quantize import compare_predictions
from .splits import load_assignments, run_table
from .train import DATASET_SHA256, environment, git_commit

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
COMM_NOTE = ("SIMULATED TRACE ESTIMATES: replayed network traces in Member 3's engine; not physical radio "
             "measurements")
SITE_RECORD_COLUMNS = ("round_id", "site_id", "status", "num_samples", "local_loss_before", "local_loss_after",
                       "float_delta_l2_norm", "delta_l2_norm", "aggregation_weight", "used_in_round", "payload_bytes")
CRITERIA_NOTE = ("protocol section 8 criteria evaluated mechanically from the pooled numbers; they need human "
                 "review and are not a claim on their own")
BUDGET_NOTE = ("Equal passes over the data per participant, not equal compute or equal learning: FL sites restart "
               "from the global model each round and late/failed updates are discarded; LOCAL-s does the work of "
               "one FL site on its own; CENT-FT does the combined FL work on pooled data in one sequence; M0 and "
               "CENT use Adam until convergence (official recipe), the others SGD. Step counts are recorded.")


class PreflightError(RuntimeError):
    """Protocol 4.3 failure: the fold is not run and nothing is repaired."""

    def __init__(self, code, detail):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


class IntegrityError(RuntimeError):
    """A shared check failed (dataset, split file, engine settings, protected files): stop everything."""


def reproducibility(dataset_sha256=None):
    """Software versions, model architecture, data and protocol identity, fixed settings (G6)."""
    try:
        flwr_version = metadata.version("flwr")
    except metadata.PackageNotFoundError:
        flwr_version = None
    proto = fd.COMPONENT / PROTOCOL
    return {"environment": dict(environment(), flwr=flwr_version), "code_commit": git_commit(fd.COMPONENT),
            "architecture": {"inputs": len(FEATURES), "hidden": 16, "outputs": len(LABELS),
                             "activation": ACTIVATION, "output_activation": OUTPUT_ACTIVATION,
                             "param_layout": [[n, list(s)] for n, s in PARAM_LAYOUT], "n_params": N_PARAMS,
                             "feature_version": FEATURE_VERSION, "features": list(FEATURES), "labels": list(LABELS)},
            "dataset_sha256": dataset_sha256,
            "protocol": {"file": PROTOCOL, "sha256": fd.file_sha256(proto) if proto.exists() else None},
            "split_file": {"file": fd.SPLITS_PATH.name, "sha256": fd.file_sha256(fd.SPLITS_PATH)},
            "fixed": {"folds": list(FOLDS), "seeds": list(SEEDS), "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS,
                      "sgd": SGD, "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
                      "confidence_threshold": CONFIDENCE_THRESHOLD}}


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
    """Network figures from the engine's own records (G2). SIMULATED TRACE ESTIMATES."""
    u = df.attrs["updates"]
    p = u.payload_bytes.astype(float)
    return {"note": COMM_NOTE, "rounds": int(df.round_id.max()),
            "updates": {k: int(v) for k, v in u.status.value_counts().items()},
            "payload_bytes_per_update": {"min": int(p.min()), "max": int(p.max()), "mean": float(p.mean())},
            "payload_bytes_total": int(p.sum()),
            "bytes_attempted": int(df.bytes_attempted.sum()), "bytes_received": int(df.bytes_received.sum()),
            "lora_airtime_s": float(u.airtime_s.dropna().astype(float).sum()),
            "lora_transfer_s": float(df.lora_transfer_s.sum()), "radio_energy_mj": float(u.radio_energy_mj.sum())}


def site_records(df):
    """Per site and round, from Member 3's update records (G3); nothing recomputed.
    aggregation_weight is the site's share of the FedAvg step (0 when not used)."""
    u = df.attrs["updates"]
    out = []
    for r in u[list(SITE_RECORD_COLUMNS)].to_dict("records"):
        out.append({k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else
                        (v.item() if hasattr(v, "item") else v)) for k, v in r.items()})
    return out


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

def fault_counts(confusion):
    """Missed faults and false alarms for OPEN_CIRCUIT and SHORT_CIRCUIT from a
    confusion matrix (rows = true, columns = predicted, LABELS order) (G1)."""
    cm = np.asarray(confusion)
    out = {}
    for f in SAFETY_CLASSES:
        i = LABELS.index(f)
        out[f] = {"true": int(cm[i].sum()), "detected": int(cm[i, i]), "missed": int(cm[i].sum() - cm[i, i]),
                  "false_alarms": int(cm[:, i].sum() - cm[i, i])}
    return out

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
                      "confusion_matrix": m["confusion_matrix"], "open_short_counts": fault_counts(m["confusion_matrix"]),
                      "run_level": run_level(y, lab, run_ids),
                      "confidence_at_or_above_0_90": float((p.max(axis=1) >= CONFIDENCE_THRESHOLD).mean()),
                      "changes_vs_m0": compare_predictions(p0.argmax(axis=1), idx),
                      "confidence_crossings_vs_m0": confidence_crossings(p0, p),
                      "per_site": {s: {"readings": int((sites == s).sum()),
                                       "runs": int(len(set(run_ids[sites == s]))),
                                       "correct": int((lab[sites == s] == y[sites == s]).sum())}
                                   for s in sorted(set(sites))}}
    table = {"run_id": run_ids.tolist(), "site_id": sites.tolist(), "y": y.tolist(),
             **{n: [LABELS[i] for i in p.argmax(axis=1)] for n, p in preds.items()},
             **{f"conf:{n}": p.max(axis=1).tolist() for n, p in preds.items()}}
    return rows, table


def macro_f1_codes(y, p, n=len(LABELS)):
    """Macro-F1 over the classes present in y, from integer class codes; per class
    2tp/(2tp+fp+fn), 0 when undefined (same as sklearn f1_score(labels=present,
    average="macro", zero_division=0)), fast enough for the bootstrap."""
    cm = np.bincount(y * n + p, minlength=n * n).reshape(n, n)
    tp = np.diag(cm)
    d = 2 * tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
    present = cm.sum(axis=1) > 0
    f = np.divide(2 * tp, d, out=np.zeros(n), where=d > 0)
    return float(f[present].mean())


def paired_run_bootstrap(tables, a, b, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Macro-F1(a) - macro-F1(b) over pooled out-of-fold tables, resampling whole
    runs (each table is one fold x seed; a run keeps all its readings). b may be a
    list of models: then the mean of their macro-F1 values is used."""
    bs = [b] if isinstance(b, str) else list(b)
    code = {l: i for i, l in enumerate(LABELS)}
    enc = lambda key: np.array([code[v] for t in tables for v in t[key]], dtype=np.int64)   # noqa: E731
    y, pa, pbs = enc("y"), enc(a), [enc(x) for x in bs]
    runs = np.concatenate([t["run_id"] for t in tables])
    uniq = np.unique(runs)
    idx = {r: np.flatnonzero(runs == r) for r in uniq}

    def diff(sel):
        return macro_f1_codes(y[sel], pa[sel]) - float(np.mean([macro_f1_codes(y[sel], pb[sel]) for pb in pbs]))
    rng = np.random.default_rng(seed)
    boots = np.array([diff(np.concatenate([idx[r] for r in rng.choice(uniq, len(uniq))])) for _ in range(resamples)])
    return {"a": a, "b": b, "metric": "macro_f1", "observed_difference": float(diff(np.arange(len(y)))),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "runs": int(len(uniq)), "resamples": resamples, "seed": seed}


# ---------------------------------------------------------------- one fold x seed

def run_fold(df, assignments, k, seed, engine=None, rounds=ROUNDS, sites=None, dataset_sha256=None):
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
    return {"status": "OK", "protocol": PROTOCOL, "label": MODEL_LABEL, "fold": k, "seed": seed,
            "final_round": rounds, "reproducibility": reproducibility(dataset_sha256),
            "preflight": table, "groups": groups, "data_use": use, "monitor_note": MONITOR_NOTE,
            "budget_note": BUDGET_NOTE, "sgd_steps": steps, "communication": communication(hist),
            "site_records": site_records(hist),
            "site_classes_fl": {s: sorted(set(fl_df[s][LABEL_COLUMN])) for s in sites},
            "rounds": rounds_log, "metrics": metrics, "eval_table": table_eval,
            "weights": {n: {k_: v.tolist() for k_, v in w.items()} for n, (w, _) in models.items()},
            "engine_change": "engine.run_fl(data, init_weights) PENDING SAMUDI REVIEW"}


def write_result(result, run_id, allowed_root=DEV_RUNS_DIR, name=None):
    """<allowed_root>/<run_id>/<name> (default fold<k>_seed<s>.json), create-only,
    never in a protected folder."""
    from .global_update import candidate_version
    candidate_version(run_id)                                    # path-safe characters only
    d = fl_run.safe_target(Path(allowed_root) / run_id, allowed_root)
    d.mkdir(parents=True, exist_ok=True)
    p = fl_run.safe_target(d / (name or f"fold{result['fold']}_seed{result['seed']}.json"), allowed_root)
    _create(p, json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    return p


# ---------------------------------------------------------------- summary (G4)

def _cat(results, key):
    return np.concatenate([np.asarray(r["eval_table"][key]) for r in results])


def _pooled(results, name):
    y, p, c = _cat(results, "y"), _cat(results, name), _cat(results, f"conf:{name}")
    m = classification_metrics(y, p)
    return {"readings": int(len(y)), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "open_circuit_recall": m["safety_recall"]["OPEN_CIRCUIT"],
            "short_circuit_recall": m["safety_recall"]["SHORT_CIRCUIT"],
            "open_short_counts": fault_counts(m["confusion_matrix"]),
            "per_class_recall": {l: m["per_class"][l]["recall"] for l in LABELS},
            "below_0_90_share": float((c < CONFIDENCE_THRESHOLD).mean()),
            "crossings_vs_m0": {k: int(sum(r["metrics"][name]["confidence_crossings_vs_m0"][k] for r in results))
                                for k in ("fell_below", "rose_above", "crossings")}}


SPREAD_KEYS = ("macro_f1", "accuracy", "open_circuit_recall", "short_circuit_recall", "confidence_at_or_above_0_90")


def _spread(results, name, key):
    """Fold-to-fold and seed-to-seed spread of one per-result metric; results
    where the metric is undefined (None) are listed, not filled in."""
    v = {f"fold{r['fold']}_seed{r['seed']}": r["metrics"][name][key] for r in results}
    ok = [r for r in results if r["metrics"][name][key] is not None]
    if not ok:
        return {"by_fold_seed": v, "mean": None, "undefined": sorted(v)}
    arr = np.array([r["metrics"][name][key] for r in ok], dtype=float)

    def mean_by(field):
        out = {}
        for r in ok:
            out.setdefault(r[field], []).append(r["metrics"][name][key])
        return {str(k): float(np.mean(x)) for k, x in sorted(out.items())}
    return {"by_fold_seed": v, "mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "min": float(arr.min()),
            "max": float(arr.max()), "by_fold": mean_by("fold"), "by_seed": mean_by("seed"),
            "undefined": sorted(k for k, x in v.items() if x is None)}


def _missing_class_rows(results, sites):
    """Per site: EVAL readings of classes absent from that site's FL data in that
    fold, read correctly by FL and by LOCAL-s (protocol section 8, criterion 2)."""
    out = {}
    for s in sites:
        n = fl_ok = loc_ok = 0
        for r in results:
            t, seen = r["eval_table"], set(r["site_classes_fl"][s])
            for y, a, b in zip(t["y"], t["FL"], t[f"LOCAL-{s}"]):
                if y not in seen:
                    n, fl_ok, loc_ok = n + 1, fl_ok + (a == y), loc_ok + (b == y)
        out[s] = {"readings": n, "fl_correct": fl_ok, "local_correct": loc_ok}
    return out


def _criteria(models, comps, missing):
    fl, m0 = models["FL"]["pooled"], models["M0"]["pooled"]

    def le(a, b):
        return None if a is None or b is None else a <= b
    os_ok = [le(m0[f"{k}_recall"], fl[f"{k}_recall"]) for k in ("open_circuit", "short_circuit")]
    fa_ok = [le(fl["open_short_counts"][f]["false_alarms"], m0["open_short_counts"][f]["false_alarms"])
             for f in SAFETY_CLASSES]
    n = sum(v["readings"] for v in missing.values())
    return {"note": CRITERIA_NOTE,
            "1_fl_minus_m0_ci95_above_zero": comps["FL_vs_M0"]["ci95"][0] > 0,
            "2a_fl_not_worse_than_local_mean": comps["FL_vs_LOCAL_mean"]["observed_difference"] >= 0,
            "2b_fl_better_on_classes_a_site_lacks": (None if n == 0 else
                                                     sum(v["fl_correct"] for v in missing.values())
                                                     > sum(v["local_correct"] for v in missing.values())),
            "3_open_short_recall_not_lower_and_false_alarms_not_higher":
                None if None in os_ok + fa_ok else all(os_ok + fa_ok),
            "4_below_0_90_share_not_higher": fl["below_0_90_share"] <= m0["below_0_90_share"]}


def summarize(results, failures, meta, planned):
    """Pooled out-of-fold summary of the successful fold x seed results. Failed
    ones are listed, never filled in."""
    out = {"protocol": PROTOCOL, "label": MODEL_LABEL, "reproducibility": meta,
           "counts": {"planned": planned, "succeeded": len(results), "failed": len(failures)},
           "used": sorted(f"fold{r['fold']}_seed{r['seed']}" for r in results),
           "failed": [{k: f[k] for k in ("fold", "seed", "error_code", "error")} for f in failures],
           "folds_used": sorted({r["fold"] for r in results}),
           "notes": {"monitoring": MONITOR_NOTE, "communication": COMM_NOTE, "budget": BUDGET_NOTE,
                     "evaluation": "EVAL_k readings at the fixed final round; prediction metrics on unseen runs "
                                   "of the PV Research Dataset (or Test Data); not field performance",
                     "interpretation": "each fold has 2 runs per class; only the pooled five-fold result is "
                                       "interpreted, next to its fold and seed spread"}}
    if not results:
        out["status"] = "NO_SUCCESSFUL_RESULTS"
        return out
    names = list(results[0]["metrics"])
    sites = sorted(results[0]["site_classes_fl"])
    locals_ = [n for n in names if n.startswith("LOCAL-")]
    models = {n: {"pooled": _pooled(results, n),
                  "spread": {k: _spread(results, n, k) for k in SPREAD_KEYS}} for n in names}
    tables = [r["eval_table"] for r in results]
    comps = {"FL_vs_M0": paired_run_bootstrap(tables, "FL", "M0"),
             "FL_vs_LOCAL_mean": paired_run_bootstrap(tables, "FL", locals_),
             "FL_vs_CENT": paired_run_bootstrap(tables, "FL", "CENT"),
             "FL_vs_CENT-FT": paired_run_bootstrap(tables, "FL", "CENT-FT")}
    fl, m0 = models["FL"]["pooled"], models["M0"]["pooled"]
    safety = {f: {"M0": m0["open_short_counts"][f], "FL": fl["open_short_counts"][f],
                  "FL_minus_M0_missed": fl["open_short_counts"][f]["missed"] - m0["open_short_counts"][f]["missed"],
                  "FL_minus_M0_false_alarms": (fl["open_short_counts"][f]["false_alarms"]
                                               - m0["open_short_counts"][f]["false_alarms"])} for f in SAFETY_CLASSES}
    comm = {"note": COMM_NOTE,
            "payload_bytes_total": int(sum(r["communication"]["payload_bytes_total"] for r in results)),
            "lora_airtime_s": float(sum(r["communication"]["lora_airtime_s"] for r in results)),
            "radio_energy_mj": float(sum(r["communication"]["radio_energy_mj"] for r in results)),
            "updates": {}}
    for r in results:
        for k, v in r["communication"]["updates"].items():
            comm["updates"][k] = comm["updates"].get(k, 0) + v
    missing = _missing_class_rows(results, sites)
    out.update({"status": "OK", "models": models, "local_mean_macro_f1": float(np.mean(
                    [models[n]["pooled"]["macro_f1"] for n in locals_])),
                "comparisons": comps, "open_short_safety": safety,
                "confidence_below_0_90": {n: models[n]["pooled"]["below_0_90_share"] for n in names},
                "classes_missing_at_site": missing, "communication": comm,
                "section8_criteria": _criteria(models, comps, missing)})
    return out


# ---------------------------------------------------------------- experiment (G5)

def run_experiment(df, assignments, run_id, dataset_sha256=None, engine=None, allowed_root=DEV_RUNS_DIR,
                   folds=FOLDS, seeds=SEEDS, rounds=ROUNDS, confirm_protocol_run=False, dataset_path=None):
    """Every fold x seed, one result or FAILED record each, then summary.json.
    A failing fold x seed is recorded and the others continue; a shared integrity
    failure (Model v1 artifacts, dataset, split file, engine settings) stops
    everything. No retries, no changed settings. Refuses the PV Research Dataset
    without confirm_protocol_run=True."""
    from .global_update import candidate_version
    if dataset_sha256 == DATASET_SHA256 and confirm_protocol_run is not True:
        raise PermissionError("the PV Research Dataset run needs explicit approval: confirm_protocol_run=True")
    candidate_version(run_id)
    run_dir = fl_run.safe_target(Path(allowed_root) / run_id, allowed_root)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; runs are never mixed or overwritten")
    engine = engine or fl_run.member3_engine()
    split_sha = fd.file_sha256(fd.SPLITS_PATH)

    def integrity():
        try:
            verify_artifacts(fd.MODELS_DIR)
        except Exception as e:                                   # noqa: BLE001 - reported as integrity failure
            raise IntegrityError(f"Model v1 artifacts: {e}") from e
        if engine.N_ROUNDS != rounds:
            raise IntegrityError(f"engine N_ROUNDS {engine.N_ROUNDS} != {rounds}")
        if fd.file_sha256(fd.SPLITS_PATH) != split_sha:
            raise IntegrityError("split file changed during the run")
        if dataset_path is not None and fd.file_sha256(dataset_path) != dataset_sha256:
            raise IntegrityError("dataset changed during the run")

    integrity()
    meta = reproducibility(dataset_sha256)
    run_dir.mkdir(parents=True)
    results, failures, paths = [], [], []
    for k in folds:
        for s in seeds:
            failure = None
            try:
                r = run_fold(df, assignments, k, s, engine=engine, rounds=rounds, dataset_sha256=dataset_sha256)
            except (IntegrityError, AssertionError, PermissionError):
                raise
            except PreflightError as e:
                failure = (e.code, e.detail)
            except Exception as e:                               # noqa: BLE001 - recorded, never hidden
                failure = (type(e).__name__, str(e))
            if failure:
                rec = {"status": "FAILED", "fold": k, "seed": s, "error_code": failure[0], "error": failure[1],
                       "metrics": None, "note": "no metrics: this fold x seed failed and is not filled in"}
                paths.append(write_result(rec, run_id, allowed_root, name=f"fold{k}_seed{s}.FAILED.json"))
                failures.append(rec)
            else:
                paths.append(write_result(r, run_id, allowed_root))
                results.append(r)
            integrity()
    summary = summarize(results, failures, meta, planned=len(folds) * len(seeds))
    paths.append(write_result(summary, run_id, allowed_root, name="summary.json"))
    return summary, paths


def run_protocol(dataset_path, run_id, confirm_protocol_run=False, engine=None, allowed_root=DEV_RUNS_DIR):
    """Full 5-fold x 3-seed x 20-round run on the PV Research Dataset. Refuses
    unless confirm_protocol_run=True (a separate, approved step). A dataset or
    split-file mismatch stops before anything is written."""
    if confirm_protocol_run is not True:
        raise PermissionError("the protocol run needs explicit approval: confirm_protocol_run=True")
    df = fd.load_dataset(dataset_path)                           # SHA-256 checked
    sha = fd.file_sha256(dataset_path)
    assignments, _ = load_assignments(fd.SPLITS_PATH, dataset_sha256=sha)
    return run_experiment(df, assignments, run_id, sha, engine, allowed_root, FOLDS, SEEDS, ROUNDS,
                          confirm_protocol_run=True, dataset_path=dataset_path)

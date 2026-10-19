"""Phase 7 baseline comparison: models x feature sets x protocols.

Protocols allowed here (model selection must not touch the OPERATION window):
  cv             Phase 4 run-wise 5-fold CV over TRAINING runs
  loso_training  leave-one-site-out restricted to TRAINING runs

The TRAINING -> OPERATION test is the final test and is refused unless
allow_final_test=True is passed explicitly (it is not used in Phase 7).

Feature sets (all fitted on the training part of each fold only):
  physics8               the 8 PV features (pv8-v1)
  physics7_no_v_deficit  ablation without v_deficit (FEATURES.md, Finding B)
  raw8                   Member 3's 8 raw sensor columns, standardised
Every feature set uses the same quality gate (status OK rows), so all
models are compared on exactly the same readings.
"""
import platform
import time

import numpy as np
import pandas as pd
import sklearn

from .contract import FL_RAW_FEATURES, GROUP_COLUMN, LABELS, WINDOW_COLUMN
from .evaluate import classification_metrics, coverage, run_level
from .features import FEATURES, fold_matrices
from .models import MODEL_NAMES, STOCHASTIC, is_applicable, make_model, model_size
from .splits import iter_folds

FEATURE_SETS = ("physics8", "physics7_no_v_deficit", "raw8")
PROTOCOLS = ("cv", "loso_training")
DEFAULT_SEEDS = (0, 1, 2)


def feature_set_matrices(df, fold, feature_set):
    """Return (X_train, X_test, names) for one feature set, from the output of
    features.fold_matrices (same rows, same gate, train-only fitting)."""
    if feature_set == "physics8":
        return fold["train"]["X"], fold["test"]["X"], list(FEATURES)
    if feature_set == "physics7_no_v_deficit":
        keep = [i for i, f in enumerate(FEATURES) if f != "v_deficit"]
        return fold["train"]["X"][:, keep], fold["test"]["X"][:, keep], [FEATURES[i] for i in keep]
    if feature_set == "raw8":
        cols = list(FL_RAW_FEATURES)
        tr = df.loc[fold["train"]["index"], cols].to_numpy(float)
        te = df.loc[fold["test"]["index"], cols].to_numpy(float)
        mean, std = tr.mean(axis=0), tr.std(axis=0)
        std[std < 1e-9] = 1.0
        return ((tr - mean) / std).astype(np.float32), ((te - mean) / std).astype(np.float32), cols
    raise ValueError(f"unknown feature set {feature_set!r}; choose from {FEATURE_SETS}")


def protocol_folds(df, assignments, protocol, allow_final_test=False):
    if protocol == "temporal":
        if not allow_final_test:
            raise PermissionError("the TRAINING -> OPERATION test is the final test; "
                                  "it is not used for model selection")
        return list(iter_folds(df, assignments, "temporal"))
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    training = df[df[WINDOW_COLUMN] == "TRAINING"]
    base = "cv" if protocol == "cv" else "loso"
    return [f for f in iter_folds(training, assignments, base) if len(f[1]) and len(f[2])]


def run_comparison(df, assignments, protocols=PROTOCOLS, models=MODEL_NAMES,
                   feature_sets=FEATURE_SETS, seeds=DEFAULT_SEEDS, allow_final_test=False,
                   progress=print):
    """Train and evaluate every (protocol, fold, feature set, model, seed).

    Returns environment info, one record per fit (fold-level metrics), the
    pooled metrics per (protocol, feature set, model, seed) computed from the
    combined held-out predictions of all folds, and the classes present in
    each held-out partition."""
    records, pooled_preds, test_classes = [], {}, []
    for protocol in protocols:
        for name, tr, te in protocol_folds(df, assignments, protocol, allow_final_test):
            if not allow_final_test:
                used = df.loc[list(tr) + list(te), WINDOW_COLUMN]
                assert (used == "TRAINING").all(), f"{name}: non-TRAINING rows in model selection"
            fold = fold_matrices(df, tr, te, fold=name)
            cov = coverage(df.loc[te], fold["test"]["status"])
            y_tr, y_te = fold["train"]["y"], fold["test"]["y"]
            runs_te = df.loc[fold["test"]["index"], GROUP_COLUMN].to_numpy()
            test_classes.append({"protocol": protocol, "fold": name,
                                 "classes_in_partition": [l for l in LABELS if cov["per_class"][l]["total"]],
                                 "classes_classified": [l for l in LABELS if (y_te == l).any()],
                                 "runs": int(df.loc[te, GROUP_COLUMN].nunique())})
            for fs in feature_sets:
                X_tr, X_te, names = feature_set_matrices(df, fold, fs)
                assert len(X_tr) == len(y_tr) and len(X_te) == len(y_te), "feature sets must share rows"
                for model_name in models:
                    if not is_applicable(model_name, names):
                        continue                    # e.g. rules need temp_spread (not in raw8)
                    for seed in (seeds if model_name in STOCHASTIC else seeds[:1]):
                        t0 = time.perf_counter()
                        model = make_model(model_name, seed=seed, feature_names=names)
                        model.fit(X_tr, y_tr)
                        pred = model.predict(X_te)
                        rec = {"protocol": protocol, "fold": name, "feature_set": fs,
                               "model": model_name, "seed": seed,
                               "train_rows": int(len(y_tr)), "test_rows": int(len(y_te)),
                               "metrics": classification_metrics(y_te, pred),
                               "run_level": run_level(y_te, pred, runs_te),
                               "coverage": cov, "size": model_size(model),
                               "fit_seconds": round(time.perf_counter() - t0, 3)}
                        if model_name == "rules":
                            rec["thresholds_scaled"] = model.thresholds_
                        if model_name == "mlp":
                            rec["mlp_iterations"] = int(model.n_iter_)
                        records.append(rec)
                        key = (protocol, fs, model_name, seed)
                        acc = pooled_preds.setdefault(key, {"y": [], "p": [], "run": []})
                        acc["y"].append(y_te); acc["p"].append(pred); acc["run"].append(runs_te)
            progress(f"{protocol}/{name}: done")
    pooled = []
    for (protocol, fs, model_name, seed), acc in pooled_preds.items():
        y, p, runs = (np.concatenate(acc[k]) for k in ("y", "p", "run"))
        pooled.append({"protocol": protocol, "feature_set": fs, "model": model_name, "seed": seed,
                       "folds": len(acc["y"]), "metrics": classification_metrics(y, p),
                       "run_level": run_level(y, p, runs)})
    return {"environment": environment(), "seeds": list(seeds), "protocols": list(protocols),
            "models": list(models), "feature_sets": list(feature_sets), "records": records,
            "pooled": pooled, "test_classes": test_classes}


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__}


def pooled_table(result):
    """Reading-level and run-level metrics from the pooled held-out
    predictions, averaged over seeds (deterministic models have one seed)."""
    rows = []
    for r in result["pooled"]:
        m, rl = r["metrics"], r["run_level"]
        rows.append({"protocol": r["protocol"], "feature_set": r["feature_set"], "model": r["model"],
                     "pooled_macro_f1": m["macro_f1"], "pooled_balanced_acc": m["balanced_accuracy"],
                     "pooled_open_recall": m["safety_recall"]["OPEN_CIRCUIT"],
                     "pooled_short_recall": m["safety_recall"]["SHORT_CIRCUIT"],
                     "runs": rl["runs"], "run_vote_acc": rl["majority_vote_accuracy"],
                     "worst_run_acc": rl["worst_run_accuracy"], "worst_run": rl["worst_run"]})
    d = pd.DataFrame(rows)
    return (d.groupby(["protocol", "feature_set", "model"])
             .agg(pooled_macro_f1=("pooled_macro_f1", "mean"),
                  pooled_macro_f1_seed_std=("pooled_macro_f1", "std"),
                  pooled_balanced_acc=("pooled_balanced_acc", "mean"),
                  pooled_open_recall=("pooled_open_recall", "mean"),
                  pooled_short_recall=("pooled_short_recall", "mean"),
                  runs=("runs", "first"), run_vote_acc=("run_vote_acc", "mean"),
                  worst_run_acc=("worst_run_acc", "min"), seeds=("pooled_macro_f1", "size"))
             .reset_index())


def summary_table(result):
    """Fold-level macro-F1 (mean, std, min over folds and seeds), merged with
    the pooled metrics from pooled_table()."""
    rows = []
    for r in result["records"]:
        m = r["metrics"]
        rows.append({"protocol": r["protocol"], "feature_set": r["feature_set"], "model": r["model"],
                     "fold": r["fold"], "seed": r["seed"], "macro_f1": m["macro_f1"],
                     "balanced_accuracy": m["balanced_accuracy"],
                     "open_recall": m["safety_recall"]["OPEN_CIRCUIT"],
                     "short_recall": m["safety_recall"]["SHORT_CIRCUIT"],
                     "run_vote_acc": r["run_level"]["majority_vote_accuracy"],
                     "worst_run_acc": r["run_level"]["worst_run_accuracy"],
                     "size": r["size"]["count"]})
    d = pd.DataFrame(rows)
    agg = d.groupby(["protocol", "feature_set", "model"]).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        macro_f1_min=("macro_f1", "min"), balanced_acc_mean=("balanced_accuracy", "mean"),
        open_recall_mean=("open_recall", "mean"), short_recall_mean=("short_recall", "mean"),
        run_vote_acc_mean=("run_vote_acc", "mean"), worst_run_acc_min=("worst_run_acc", "min"),
        size=("size", "median"), fits=("macro_f1", "size"))
    keys = ["protocol", "feature_set", "model"]
    return agg.reset_index().merge(pooled_table(result), on=keys, how="left")


def ablation_table(table, protocol, metric="pooled_macro_f1"):
    """physics8 minus physics7_no_v_deficit per model (Finding B), for a
    pooled or fold-mean macro-F1 column."""
    t = table[table.protocol == protocol].set_index(["model", "feature_set"])[metric].unstack()
    out = pd.DataFrame({"with_v_deficit": t.get("physics8"),
                        "without_v_deficit": t.get("physics7_no_v_deficit")})
    out["difference"] = out.with_v_deficit - out.without_v_deficit
    return out

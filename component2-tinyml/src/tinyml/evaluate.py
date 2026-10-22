"""Evaluation metrics and coverage reporting (architecture section 8.2).

Classification metrics are computed on classified (status-OK) readings and
are always reported together with coverage, so skipped readings are never
silently dropped (FEATURES.md, Finding A).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .contract import GROUP_COLUMN, LABEL_COLUMN, LABELS
from .features import STATUS_OK

SAFETY_CLASSES = ("OPEN_CIRCUIT", "SHORT_CIRCUIT")


# [GUIDE] F. ALL classification metrics are calculated in this one function.
# Three kinds of results use it, and they must not be confused:
#   1. 5-fold CV (experiment.run_comparison): model selection, TRAINING window only
#      -> results/baselines_phase7.md (MLP macro-F1 0.981)
#   2. training-fit accuracy (train.train_final): the model predicting its own training
#      rows -> 1.0; a fit check, NOT a performance claim
#   3. final OPERATION test (holdout.run_evaluation): frozen model, unseen 50 %, run once
#      -> results/operation_eval.md (accuracy 0.979, macro-F1 0.978)
# INPUT: true labels and predicted labels.  OUTPUT: dict of metrics.
def classification_metrics(y_true, y_pred):
    """Macro-F1 (primary), balanced accuracy, accuracy, per-class precision /
    recall / F1 / support and the confusion matrix (rows = true, LABELS order).
    Classes absent from y_true get recall None and are excluded from macro-F1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = [l for l in LABELS if (y_true == l).any()]
    # Precision = of the readings predicted as class C, how many really are C (false alarms).
    # Recall    = of the readings that really are C, how many were found (missed faults).
    # F1        = one number combining precision and recall for each class.
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(LABELS), zero_division=0)
    per_class = {}
    for i, l in enumerate(LABELS):
        has = l in present
        per_class[l] = {"precision": float(p[i]), "recall": float(r[i]) if has else None,
                        "f1": float(f[i]) if has else None, "support": int(s[i])}
    return {
        "n": int(len(y_true)),
        # [GUIDE] F. Macro-F1 (the primary metric): the F1 of every class present, averaged
        # equally, so rare fault classes count as much as NORMAL.
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        # mean recall over the classes present in y_true (same as balanced accuracy,
        # without sklearn's warning when a fold predicts a class it does not contain)
        "balanced_accuracy": float(np.mean([per_class[l]["recall"] for l in present])) if present else None,
        # [GUIDE] F. Accuracy: share of all classified readings with the correct label.
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classes_present": present,
        "per_class": per_class,
        "safety_recall": {l: per_class[l]["recall"] for l in SAFETY_CLASSES},
        # [GUIDE] F. Confusion matrix: rows = true class, columns = predicted class, LABELS order.
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
    }


def run_level(y_true, y_pred, run_ids):
    """Majority vote per run: share of runs whose majority prediction is the
    true label, plus the worst single-run reading accuracy."""
    d = pd.DataFrame({"run": np.asarray(run_ids), "y": np.asarray(y_true), "p": np.asarray(y_pred)})
    if d.empty:
        return {"runs": 0, "majority_vote_accuracy": None, "worst_run_accuracy": None, "worst_run": None}
    g = d.groupby("run")
    vote_ok = g.apply(lambda x: x.p.mode().iloc[0] == x.y.iloc[0], include_groups=False)
    acc = g.apply(lambda x: (x.p == x.y).mean(), include_groups=False)
    return {"runs": int(len(acc)), "majority_vote_accuracy": float(vote_ok.mean()),
            "worst_run_accuracy": float(acc.min()), "worst_run": str(acc.idxmin())}


def coverage(df_part, status):
    """Total / classified / skipped readings, skip reasons, and coverage per
    class and per run, for one evaluation partition."""
    status = status.reindex(df_part.index)
    ok = status == STATUS_OK
    d = pd.DataFrame({"label": df_part[LABEL_COLUMN], "run": df_part[GROUP_COLUMN], "ok": ok})
    per_class = {l: {"total": int((d.label == l).sum()), "classified": int((ok & (d.label == l)).sum())}
                 for l in LABELS}
    for v in per_class.values():
        v["coverage"] = v["classified"] / v["total"] if v["total"] else None
    runs = d.groupby("run").agg(total=("ok", "size"), classified=("ok", "sum"))
    partial = runs[runs.classified < runs.total]
    return {
        "total": int(len(d)), "classified": int(ok.sum()), "skipped": int((~ok).sum()),
        "coverage": float(ok.mean()) if len(d) else None,
        "skip_reasons": {k: int(v) for k, v in status[~ok].value_counts().items()},
        "per_class": per_class,
        "runs_not_fully_classified": {r: {"classified": int(x.classified), "total": int(x.total)}
                                      for r, x in partial.iterrows()},
    }

"""Baseline classifiers for the Phase 7 comparison.

Every model has fit(X, y) / predict(X) and is created by make_model(name, seed).
Inputs are the scaled feature matrices from tinyml.features; models never see
labels, run_id, site_id or health_score as inputs.

  rules   interpretable one-threshold-per-fault rules, thresholds learned on the
          training fold (the "is ML needed at all?" baseline)
  logreg  multinomial logistic (softmax) regression
  tree    decision tree, depth <= 4
  forest  random forest, 20 trees, depth <= 6 (upper-bound baseline)
  mlp     8-16-5 tanh MLP: the architecture planned for the ESP32 and FL
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from .contract import LABELS

MODEL_NAMES = ("rules", "logreg", "tree", "forest", "mlp")
STOCHASTIC = ("forest", "mlp")          # results depend on the seed


class ThresholdRules:
    """One threshold per fault, checked in a fixed order:

        OPEN_CIRCUIT     if current_ma  < t
        SHORT_CIRCUIT    if voltage_v   < t
        DUST             if dust_vo_v   > t
        PARTIAL_SHADING  if temp_spread > t
        otherwise NORMAL

    Each threshold is chosen on the training data only: the candidate (from
    the 1st-99th percentiles of that feature) that best separates the fault
    from all other classes, by balanced accuracy.
    """
    RULES = (("OPEN_CIRCUIT", "current_ma", "<"), ("SHORT_CIRCUIT", "voltage_v", "<"),
             ("DUST", "dust_vo_v", ">"), ("PARTIAL_SHADING", "temp_spread", ">"))

    def __init__(self, feature_names):
        self.feature_names = list(feature_names)
        missing = [f for _, f, _ in self.RULES if f not in self.feature_names]
        if missing:
            raise ValueError(f"rule baseline needs features {missing}")

    def fit(self, X, y):
        y = np.asarray(y)
        self.thresholds_ = {}
        for label, feat, op in self.RULES:
            x = X[:, self.feature_names.index(feat)]
            pos = y == label
            best, best_t = -1.0, None
            for t in np.unique(np.percentile(x, np.linspace(1, 99, 99))):
                hit = x < t if op == "<" else x > t
                score = (hit[pos].mean() + (~hit[~pos]).mean()) / 2 if pos.any() and (~pos).any() else 0
                if score > best:
                    best, best_t = score, float(t)
            self.thresholds_[label] = best_t
        self.classes_ = np.array(LABELS)
        return self

    def predict(self, X):
        out = np.full(len(X), "NORMAL", dtype=object)
        decided = np.zeros(len(X), bool)
        for label, feat, op in self.RULES:
            x = X[:, self.feature_names.index(feat)]
            t = self.thresholds_[label]
            hit = (x < t if op == "<" else x > t) & ~decided
            out[hit] = label
            decided |= hit
        return out


def is_applicable(name, feature_names):
    """The rule baseline needs current, voltage, dust and temp_spread inputs."""
    if name == "rules":
        return all(f in list(feature_names) for _, f, _ in ThresholdRules.RULES)
    return True


def make_model(name, seed=0, feature_names=None):
    if name == "rules":
        return ThresholdRules(feature_names)
    if name == "logreg":
        return LogisticRegression(max_iter=5000)
    if name == "tree":
        return DecisionTreeClassifier(max_depth=4, random_state=seed)
    if name == "forest":
        return RandomForestClassifier(n_estimators=20, max_depth=6, random_state=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(16,), activation="tanh", solver="adam",
                             max_iter=2000, random_state=seed)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def model_size(model):
    """Learned parameters (weights + biases) or tree nodes. A size indicator
    for comparison only; it is not a measured flash or RAM figure."""
    if isinstance(model, ThresholdRules):
        return {"kind": "thresholds", "count": len(model.thresholds_)}
    if isinstance(model, LogisticRegression):
        return {"kind": "parameters", "count": int(model.coef_.size + model.intercept_.size)}
    if isinstance(model, MLPClassifier):
        n = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
        return {"kind": "parameters", "count": int(n)}
    if isinstance(model, DecisionTreeClassifier):
        return {"kind": "tree_nodes", "count": int(model.tree_.node_count)}
    if isinstance(model, RandomForestClassifier):
        return {"kind": "tree_nodes", "count": int(sum(e.tree_.node_count for e in model.estimators_))}
    return {"kind": "unknown", "count": None}

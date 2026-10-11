import json
import numpy as np


class TinyMLModel:
    def __init__(self, history_path="fl_global_model_history.json"):
        h = json.load(open(history_path))
        self.features = h["features"]
        self.labels = h["labels"]
        self.mu = np.array(h["scaler_mean"])
        self.sd = np.array(h["scaler_std"])
        self.versions = {v: {k: np.array(a) for k, a in w.items()} for v, w in h["versions"].items()}

    def predict_proba(self, X, version="v1.20"):
        X = np.asarray(X[self.features] if hasattr(X, "columns") else X, float)
        w = self.versions[version]
        z = (X - self.mu) / self.sd
        h = np.maximum(0, z @ w["W1"] + w["b1"])
        o = h @ w["W2"] + w["b2"]
        o -= o.max(1, keepdims=True)
        e = np.exp(o)
        return e / e.sum(1, keepdims=True)

    def predict(self, X, version="v1.20"):
        return np.array(self.labels)[self.predict_proba(X, version).argmax(1)]
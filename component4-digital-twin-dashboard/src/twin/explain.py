import numpy as np
import pandas as pd
import shap

from .tinyml_model import TinyMLModel


class TinyMLExplainer:
    def __init__(self, model: TinyMLModel, background: pd.DataFrame):
        self.model = model
        self.background = background[model.features].to_numpy(float)
        self._explainers = {}

    def _explainer(self, version: str, label: str):
        key = (version, label)
        if key not in self._explainers:
            k = self.model.labels.index(label)
            f = lambda X: self.model.predict_proba(X, version)[:, k]   # noqa: E731
            self._explainers[key] = shap.KernelExplainer(f, self.background)
        return self._explainers[key]

    def explain_row(self, x: pd.Series, version: str, label: str) -> dict:
        ex = self._explainer(version, label)
        xv = x[self.model.features].to_numpy(float)[None, :]
        sv = ex.shap_values(xv, silent=True)[0]
        prob = float(self.model.predict_proba(xv, version)[0, self.model.labels.index(label)])
        base = float(ex.expected_value)
        return {"base_value": base, "prob": prob,
                "values": dict(zip(self.model.features, map(float, sv))),
                "additivity_error": abs(base + float(np.sum(sv)) - prob)}


def top_features(values: dict, n=3) -> list[str]:
    order = sorted(values.items(), key=lambda kv: -abs(kv[1]))[:n]
    return [f"{k} ({v:+.3f})" for k, v in order]
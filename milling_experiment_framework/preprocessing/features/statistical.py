from __future__ import annotations

import numpy as np

from milling_experiment_framework.registry.preprocessor_registry import PREPROCESSOR_REGISTRY


@PREPROCESSOR_REGISTRY.register("statistical_features")
class StatisticalFeaturesStep:
    def __init__(self, config: dict):
        self.config = config
        self.params: dict = {}

    def fit_transform(self, data, split_col: str = "split"):
        X = data.primary_X()
        if X.ndim != 3:
            self.params = {"status": "no_op", "reason": "input is already feature-like"}
            return data
        feats = []
        names = []
        funcs = [
            ("mean", np.mean),
            ("std", np.std),
            ("min", np.min),
            ("max", np.max),
            ("rms", lambda a, axis: np.sqrt(np.mean(np.square(a), axis=axis))),
        ]
        for name, func in funcs:
            vals = func(X, axis=2)
            feats.append(vals)
            names.extend([f"ch{ch}_{name}" for ch in range(X.shape[1])])
        data.X = np.concatenate(feats, axis=1).astype("float32")
        data.X_timeseries = None
        self.params = {"feature_names": names, "feature_dim": int(data.X.shape[1])}
        return data

from __future__ import annotations

import numpy as np

from milling_experiment_framework.registry.preprocessor_registry import PREPROCESSOR_REGISTRY


@PREPROCESSOR_REGISTRY.register("normalization")
class NormalizationStep:
    def __init__(self, config: dict):
        self.config = config
        self.params: dict = {}

    def fit_transform(self, data, split_col: str = "split"):
        X = data.primary_X()
        train_mask = data.metadata[split_col].to_numpy() == "train"
        method = self.config.get("method", "zscore")
        if method == "none":
            self.params = {"method": "none"}
            return data
        if X.ndim == 3:
            axes = (0, 2)
            keep_shape = (1, X.shape[1], 1)
        else:
            axes = 0
            keep_shape = (1, X.shape[1])
        train_X = X[train_mask]
        if method == "zscore":
            mean = train_X.mean(axis=axes).reshape(keep_shape)
            std = train_X.std(axis=axes).reshape(keep_shape)
            std = np.where(std < 1e-8, 1.0, std)
            data.replace_primary_X(((X - mean) / std).astype("float32"))
            self.params = {"method": method, "fit_on": "train", "mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()}
        elif method == "minmax":
            min_v = train_X.min(axis=axes).reshape(keep_shape)
            max_v = train_X.max(axis=axes).reshape(keep_shape)
            denom = np.where((max_v - min_v) < 1e-8, 1.0, max_v - min_v)
            data.replace_primary_X(((X - min_v) / denom).astype("float32"))
            self.params = {
                "method": method,
                "fit_on": "train",
                "min": min_v.reshape(-1).tolist(),
                "max": max_v.reshape(-1).tolist(),
            }
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        return data

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from milling_experiment_framework.registry.model_registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("random_forest")
class RandomForestModel:
    model_type = "ML"
    input_type = "feature-based"

    def __init__(self, config: dict, task_config: dict):
        self.config = config
        self.task_config = task_config
        params = dict(config.get("params", {}))
        params.setdefault("n_estimators", 100)
        params.setdefault("random_state", 42)
        if task_config.get("type") == "regression":
            self.estimator = RandomForestRegressor(**params)
        else:
            self.estimator = RandomForestClassifier(**params)

    def fit(self, X, y, **kwargs):
        self.estimator.fit(_feature_matrix(X), y)
        return self

    def predict(self, X):
        return self.estimator.predict(_feature_matrix(X))

    def predict_proba(self, X):
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(_feature_matrix(X))
        return None

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path)

    def feature_importance(self):
        return getattr(self.estimator, "feature_importances_", None)


def _feature_matrix(X):
    X = np.asarray(X)
    if X.ndim > 2:
        return X.reshape(X.shape[0], -1)
    return X

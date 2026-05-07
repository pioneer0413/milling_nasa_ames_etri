from __future__ import annotations

import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Ridge


def make_ridge_adaboost(n_estimators: int = 26, learning_rate: float = 0.006, random_state: int = 0) -> AdaBoostRegressor:
    """Create a Ridge-based AdaBoost regressor with sklearn-version compatibility."""

    ridge = Ridge(alpha=1.0, random_state=random_state)
    try:
        return AdaBoostRegressor(estimator=ridge, n_estimators=int(n_estimators), learning_rate=float(learning_rate), random_state=random_state)
    except TypeError:
        return AdaBoostRegressor(base_estimator=ridge, n_estimators=int(n_estimators), learning_rate=float(learning_rate), random_state=random_state)


class RidgeAdaBoostStackingRegressor:
    def __init__(self, n_estimators: int = 26, learning_rate: float = 0.006, random_state: int = 0):
        self.model = make_ridge_adaboost(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    def fit(self, x_meta: np.ndarray, y: np.ndarray) -> "RidgeAdaBoostStackingRegressor":
        if x_meta.ndim != 2:
            raise ValueError(f"x_meta must be 2D, got {x_meta.shape}")
        self.model.fit(x_meta, y.reshape(-1))
        return self

    def predict(self, x_meta: np.ndarray) -> np.ndarray:
        if x_meta.ndim != 2:
            raise ValueError(f"x_meta must be 2D, got {x_meta.shape}")
        return self.model.predict(x_meta).reshape(-1)


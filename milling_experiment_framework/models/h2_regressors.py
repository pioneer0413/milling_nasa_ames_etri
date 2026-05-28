from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from milling_experiment_framework.models.dl.feature_gru_regressor import FeatureGRURegressor


H2_MODEL_FAMILIES = {
    "random_forest": "Feature-based ML",
    "xgboost": "Feature-based ML",
    "svr": "Feature-based ML",
    "linear_regression": "Feature-based ML",
    "mlp": "Neural Network",
    "feature_gru": "Neural Network",
}


H2_MODEL_ALIASES = {
    "rf": "random_forest",
    "randomforest": "random_forest",
    "xgb": "xgboost",
    "xgb_regressor": "xgboost",
    "support_vector_regression": "svr",
    "linear": "linear_regression",
    "gru": "feature_gru",
    "feature-based_gru": "feature_gru",
    "feature_based_gru": "feature_gru",
}


H2_DEFAULT_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 100,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
    },
    "svr": {
        "kernel": "rbf",
        "C": 10.0,
        "epsilon": 0.03,
        "gamma": "scale",
    },
    "linear_regression": {},
    "mlp": {
        "hidden_layer_sizes": (64, 32, 16),
        "max_iter": 200,
        "learning_rate_init": 0.001,
    },
    "feature_gru": {
        "gru_hidden_size": 32,
        "gru_num_layers": 1,
        "regression_head_hidden_dim": 32,
        "dropout": 0.0,
        "epochs": 200,
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "patience": 0,
        "validation_fraction": 0.0,
        "device": "cpu",
    },
}


class H2FeatureGRUEstimator(BaseEstimator, RegressorMixin):
    """Small sklearn-compatible wrapper around the feature GRU regressor.

    H2 atomic tables are tabular feature vectors. The wrapper presents each row
    as a one-step feature sequence, letting the existing GRU module participate
    in the same fit/predict pipeline as the feature-based baselines.
    """

    def __init__(
        self,
        gru_hidden_size: int = 32,
        gru_num_layers: int = 1,
        regression_head_hidden_dim: int = 32,
        dropout: float = 0.0,
        epochs: int = 200,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        patience: int = 0,
        validation_fraction: float = 0.0,
        device: str = "cpu",
        random_state: int | None = None,
    ) -> None:
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.regression_head_hidden_dim = regression_head_hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.device = device
        self.random_state = random_state

    def fit(self, X, y):
        x = self._as_float_array(X)
        target = np.asarray(y, dtype="float32").reshape(-1)
        if x.ndim != 2:
            raise ValueError(f"feature_gru expects a 2D feature matrix, got shape={x.shape}")
        if len(x) != len(target):
            raise ValueError("feature_gru X/y length mismatch")

        seed = int(self.random_state or 0)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device(self.device if self.device == "cuda" and torch.cuda.is_available() else "cpu")

        self.y_mean_ = float(target.mean())
        y_std = float(target.std())
        self.y_std_ = y_std if y_std > 1e-8 else 1.0
        y_norm = ((target - self.y_mean_) / self.y_std_).astype("float32")

        self.model_ = FeatureGRURegressor(
            input_dim=x.shape[1],
            gru_hidden_size=int(self.gru_hidden_size),
            gru_num_layers=int(self.gru_num_layers),
            regression_head_hidden_dim=int(self.regression_head_hidden_dim),
            dropout=float(self.dropout),
        ).to(device)
        self.input_dim_ = int(x.shape[1])

        indices = np.arange(len(x))
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        val_size = int(round(len(indices) * float(self.validation_fraction))) if len(indices) >= 5 else 0
        val_idx = indices[:val_size]
        train_idx = indices[val_size:] if val_size else indices
        if len(train_idx) == 0:
            train_idx = indices
            val_idx = np.array([], dtype=int)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = torch.nn.MSELoss()
        best_state = deepcopy(self.model_.state_dict())
        best_val = float("inf")
        bad_epochs = 0
        batch_size = max(1, int(self.batch_size))

        for _ in range(max(1, int(self.epochs))):
            self.model_.train()
            rng.shuffle(train_idx)
            for start in range(0, len(train_idx), batch_size):
                batch_idx = train_idx[start : start + batch_size]
                xb = torch.as_tensor(x[batch_idx], dtype=torch.float32, device=device).unsqueeze(1)
                yb = torch.as_tensor(y_norm[batch_idx], dtype=torch.float32, device=device)
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()

            if len(val_idx):
                self.model_.eval()
                with torch.no_grad():
                    xv = torch.as_tensor(x[val_idx], dtype=torch.float32, device=device).unsqueeze(1)
                    yv = torch.as_tensor(y_norm[val_idx], dtype=torch.float32, device=device)
                    val_loss = float(loss_fn(self.model_(xv), yv).detach().cpu())
                if val_loss + 1e-9 < best_val:
                    best_val = val_loss
                    best_state = deepcopy(self.model_.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if bad_epochs >= int(self.patience):
                    break

        if len(val_idx):
            self.model_.load_state_dict(best_state)
        self.device_ = device
        return self

    def predict(self, X):
        x = self._as_float_array(X)
        if x.ndim != 2 or x.shape[1] != self.input_dim_:
            raise ValueError(f"feature_gru expected feature matrix with {self.input_dim_} columns, got shape={x.shape}")
        self.model_.eval()
        preds: list[np.ndarray] = []
        batch_size = max(1, int(self.batch_size))
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=self.device_).unsqueeze(1)
                pred = self.model_(xb).detach().cpu().numpy()
                preds.append(pred)
        out = np.concatenate(preds) if preds else np.array([], dtype="float32")
        return out * self.y_std_ + self.y_mean_

    def _as_float_array(self, X) -> np.ndarray:
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        return np.asarray(X, dtype="float32")


def canonical_model_name(model_name: str) -> str:
    key = model_name.strip().lower()
    return H2_MODEL_ALIASES.get(key, key)


def supported_h2_models() -> list[str]:
    return list(H2_MODEL_FAMILIES)


def resolve_h2_model_params(model_name: str, model_defaults: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    name = canonical_model_name(model_name)
    if name not in H2_MODEL_FAMILIES:
        raise ValueError(f"Unsupported H2 model: {model_name}. Supported models: {supported_h2_models()}")
    params = deepcopy(H2_DEFAULT_MODEL_PARAMS[name])
    if model_defaults and name in model_defaults:
        params.update(deepcopy(model_defaults[name]))
    return params


def resolve_h2_model_defaults(model_defaults: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    return {name: resolve_h2_model_params(name, model_defaults or {}) for name in H2_MODEL_FAMILIES}


def create_h2_regressor(model_name: str, seed: int, model_defaults: dict[str, dict[str, Any]] | None = None):
    name = canonical_model_name(model_name)
    params = resolve_h2_model_params(name, model_defaults)
    if name in {"random_forest", "xgboost", "mlp", "feature_gru"}:
        params.setdefault("random_state", seed)
    if name == "random_forest":
        return RandomForestRegressor(**params)
    if name == "xgboost":
        return XGBRegressor(**params)
    if name == "svr":
        return SVR(**params)
    if name == "linear_regression":
        return LinearRegression(**params)
    if name == "mlp":
        return MLPRegressor(**params)
    if name == "feature_gru":
        return H2FeatureGRUEstimator(**params)
    raise ValueError(f"Unsupported H2 model: {model_name}")


def create_h2_feature_pipeline(model_name: str, seed: int, model_defaults: dict[str, dict[str, Any]] | None = None) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", create_h2_regressor(model_name, seed, model_defaults)),
        ]
    )


def h2_model_catalog(model_defaults: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    resolved = resolve_h2_model_defaults(model_defaults)
    return [
        {
            "model_name": name,
            "family": H2_MODEL_FAMILIES[name],
            "estimator": {
                "random_forest": "sklearn.ensemble.RandomForestRegressor",
                "xgboost": "xgboost.XGBRegressor",
                "svr": "sklearn.svm.SVR",
                "linear_regression": "sklearn.linear_model.LinearRegression",
                "mlp": "sklearn.neural_network.MLPRegressor",
                "feature_gru": "milling_experiment_framework.models.dl.feature_gru_regressor.FeatureGRURegressor",
            }[name],
            "input_type": "tabular feature vector",
            "pipeline": "StandardScaler -> estimator",
            "params": resolved[name],
        }
        for name in H2_MODEL_FAMILIES
    ]

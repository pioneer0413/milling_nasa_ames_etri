from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from milling_experiment_framework.models.h2_regressors import (
    create_h2_feature_pipeline,
    create_h2_regressor,
    supported_h2_models,
)


def test_h2_feature_pipelines_fit_and_predict():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(18, 5))
    y = 0.3 * x[:, 0] - 0.1 * x[:, 1] + rng.normal(scale=0.01, size=18)
    defaults = {
        "random_forest": {"n_estimators": 5, "n_jobs": 1},
        "xgboost": {"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
        "mlp": {"hidden_layer_sizes": (8,), "max_iter": 20},
        "feature_gru": {"gru_hidden_size": 8, "regression_head_hidden_dim": 8, "epochs": 3, "patience": 2},
        "feature_film_gru": {
            "gru_hidden_size": 8,
            "regression_head_hidden_dim": 8,
            "metadata_feature_dim": 2,
            "film_hidden_dim": 8,
            "epochs": 3,
            "patience": 2,
        },
    }
    for model_name in supported_h2_models():
        model = create_h2_feature_pipeline(model_name, seed=3, model_defaults=defaults)
        model.fit(x, y)
        pred = model.predict(x[:4])
        assert pred.shape == (4,)
        assert np.isfinite(pred).all()


def test_h2_process_style_pipeline_fit_and_predict():
    frame = pd.DataFrame(
        {
            "f0": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "f1": [1.0, 1.5, 1.8, 2.1, 2.4, 2.8],
            "material_name": ["cast_iron", "cast_iron", "steel", "steel", "cast_iron", "steel"],
        }
    )
    y = np.array([0.1, 0.15, 0.22, 0.28, 0.34, 0.41])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), ["f0", "f1"]),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), ["material_name"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    defaults = {
        "random_forest": {"n_estimators": 5, "n_jobs": 1},
        "xgboost": {"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
        "mlp": {"hidden_layer_sizes": (8,), "max_iter": 30},
        "feature_gru": {"gru_hidden_size": 8, "regression_head_hidden_dim": 8, "epochs": 3, "patience": 2},
        "feature_film_gru": {
            "gru_hidden_size": 8,
            "regression_head_hidden_dim": 8,
            "metadata_feature_dim": 2,
            "film_hidden_dim": 8,
            "epochs": 3,
            "patience": 2,
        },
    }
    for model_name in supported_h2_models():
        model = Pipeline([("preprocess", preprocessor), ("model", create_h2_regressor(model_name, seed=5, model_defaults=defaults))])
        model.fit(frame, y)
        pred = model.predict(frame.iloc[:3])
        assert pred.shape == (3,)
        assert np.isfinite(pred).all()

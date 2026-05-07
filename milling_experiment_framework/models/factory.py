from __future__ import annotations

import milling_experiment_framework.models.dl.cnn1d  # noqa: F401
import milling_experiment_framework.models.dl.cumulative_s_dnn  # noqa: F401
import milling_experiment_framework.models.dl.hybrid_lstm_process  # noqa: F401
import milling_experiment_framework.models.ml.random_forest  # noqa: F401
from milling_experiment_framework.registry.model_registry import MODEL_REGISTRY


def create_model(model_config: dict, task_config: dict):
    model_cls = MODEL_REGISTRY.get(model_config.get("name"))
    return model_cls(model_config, task_config)

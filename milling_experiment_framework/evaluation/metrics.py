from __future__ import annotations

from milling_experiment_framework.evaluation.classification import classification_metrics
from milling_experiment_framework.evaluation.regression import regression_metrics


def compute_metrics(task_type: str, y_true, y_pred, metrics: list[str]) -> dict:
    if task_type == "regression":
        return regression_metrics(y_true, y_pred, [m.lower() for m in metrics])
    return classification_metrics(y_true, y_pred, [m.lower() for m in metrics])

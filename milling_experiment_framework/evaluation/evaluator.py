from __future__ import annotations

import pandas as pd

from milling_experiment_framework.evaluation.domain_eval import grouped_metrics
from milling_experiment_framework.evaluation.metrics import compute_metrics


class Evaluator:
    def __init__(self, task_config: dict, evaluation_config: dict):
        self.task_config = task_config
        self.evaluation_config = evaluation_config

    def evaluate(self, predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        metrics = self.evaluation_config.get("metrics", [])
        task_type = self.task_config.get("type", "classification")
        result = {"overall": compute_metrics(task_type, predictions["y_true"], predictions["y_pred"], metrics), "splits": {}}
        for split, group in predictions.groupby("split"):
            result["splits"][split] = compute_metrics(task_type, group["y_true"], group["y_pred"], metrics)
        group_frame = grouped_metrics(predictions, task_type, metrics, self.evaluation_config.get("group_metrics", []))
        return result, group_frame

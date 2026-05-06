from __future__ import annotations

import pandas as pd

from milling_experiment_framework.evaluation.metrics import compute_metrics


def grouped_metrics(predictions: pd.DataFrame, task_type: str, metrics: list[str], group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in group_cols:
        if col not in predictions.columns:
            continue
        for value, group in predictions.groupby(col, dropna=False):
            metric_values = compute_metrics(task_type, group["y_true"], group["y_pred"], metrics)
            rows.append({"group_column": col, "group_value": value, "num_samples": len(group), **metric_values})
    return pd.DataFrame(rows)

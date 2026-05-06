from __future__ import annotations

import pandas as pd


def prediction_errors(predictions: pd.DataFrame, task_type: str) -> pd.DataFrame:
    frame = predictions.copy()
    if task_type == "regression":
        frame["residual"] = frame["y_true"] - frame["y_pred"]
        frame["absolute_error"] = frame["residual"].abs()
        return frame.sort_values("absolute_error", ascending=False)
    return frame.loc[frame["y_true"] != frame["y_pred"]].copy()

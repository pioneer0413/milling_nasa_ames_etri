from __future__ import annotations

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, metrics: list[str]) -> dict:
    out = {}
    if "mae" in metrics:
        out["mae"] = float(mean_absolute_error(y_true, y_pred))
    if "rmse" in metrics:
        out["rmse"] = float(mean_squared_error(y_true, y_pred, squared=False))
    if "r2" in metrics or "r2_score" in metrics:
        out["r2"] = float(r2_score(y_true, y_pred))
    return out

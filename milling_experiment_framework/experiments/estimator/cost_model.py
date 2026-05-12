from __future__ import annotations

from typing import Any


DEFAULT_RUNTIME_SEC = {
    "random_forest": 5.0,
    "rf": 5.0,
    "svr": 20.0,
    "xgboost": 10.0,
    "lightgbm": 8.0,
    "mlp": 30.0,
    "cnn1d": 60.0,
    "lstm": 120.0,
    "hybrid_lstm_process": 180.0,
    "cumulative_s_dnn": 600.0,
    "default_model": 30.0,
}


def runtime_for_model(model: str, config: dict[str, Any], runtime_overrides: dict | None = None) -> tuple[float, str, str]:
    runtime_cfg = config.get("estimation", {}).get("runtime_per_atomic_sec", {})
    runtime_overrides = runtime_overrides or {}
    if model in runtime_overrides:
        return float(runtime_overrides[model]), "user_override", f"runtime_overrides[{model}]"
    if "__default__" in runtime_overrides:
        return float(runtime_overrides["__default__"]), "user_override", "runtime_overrides default"
    if model in runtime_cfg:
        return float(runtime_cfg[model]), "config_runtime_per_atomic", f"estimation.runtime_per_atomic_sec.{model}"
    if "__default__" in runtime_cfg:
        return float(runtime_cfg["__default__"]), "config_runtime_per_atomic", "estimation.runtime_per_atomic_sec.__default__"
    return float(DEFAULT_RUNTIME_SEC.get(model, DEFAULT_RUNTIME_SEC["default_model"])), "heuristic_default", "built-in heuristic"


def estimate_disk_mb(total_sub_runs: int, checkpoint_size_mb: float, prediction_rows: int, figure_count: int) -> float:
    metrics_mb = max(0.1, total_sub_runs * 0.002)
    predictions_mb = max(0.1, prediction_rows * 0.00025)
    figures_mb = figure_count * 0.25
    checkpoints_mb = total_sub_runs * checkpoint_size_mb
    return float(metrics_mb + predictions_mb + figures_mb + checkpoints_mb)


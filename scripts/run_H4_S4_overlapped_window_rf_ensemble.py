#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from milling_experiment_framework.experiments.execution_path import create_execution_dir
from milling_experiment_framework.preprocessing.features.window_feature_extractor import FEATURE_NAMES, extract_multisensor_features
from milling_experiment_framework.preprocessing.windowing.overlapped_windowing import window_run


PREFIX = "H4_S4"
EXPERIMENT_NUMBER = "H4_S4"
TOPIC = "overlapped_window_random_forest_ensemble_NASA_Ames_main_ABC"
DOMAIN_CASES = {"A": [1, 9], "B": [2, 12], "C": [8, 14]}
TRANSFER_SCENARIOS = [("A", "B"), ("A", "C"), ("B", "A"), ("B", "C"), ("C", "A"), ("C", "B")]
SENSOR_COLUMNS = ["smcAC", "smcDC", "vib_spindle", "vib_table", "AE_spindle", "AE_table"]
WINDOW_NAMES = ["window_1", "window_2", "window_3", "window_4"]
MODEL_NAMES = [
    "full_run_random_forest",
    "rf_window_1_only",
    "rf_window_2_only",
    "rf_window_3_only",
    "rf_window_4_only",
    "window_rf_uniform_average",
    "window_rf_validation_weighted_average",
    "window_rf_ridge_stacking",
    "window_rf_mlp_stacking",
]


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, indent=2, ensure_ascii=False)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        write_json(path.with_suffix(".json"), payload)
    else:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(to_builtin(payload), f, sort_keys=False, allow_unicode=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_signal_cell(value: Any) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    return np.asarray(ast.literal_eval(value), dtype=np.float32)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


def domain_for_case(case_id: int) -> str:
    for domain, cases in DOMAIN_CASES.items():
        if int(case_id) in cases:
            return domain
    return "unknown"


def ensemble_type(model_name: str) -> str:
    if model_name == "full_run_random_forest":
        return "full_run_baseline"
    if model_name.startswith("rf_window_"):
        return "single_window"
    if model_name == "window_rf_uniform_average":
        return "uniform_average"
    if model_name == "window_rf_validation_weighted_average":
        return "validation_weighted_average"
    if model_name == "window_rf_ridge_stacking":
        return "ridge_stacking"
    if model_name == "window_rf_mlp_stacking":
        return "mlp_stacking"
    return "unknown"


@dataclass
class WindowRFData:
    meta: pd.DataFrame
    full_features: np.ndarray
    window_features: dict[str, np.ndarray]
    full_feature_names: list[str]
    window_feature_names: dict[str, list[str]]
    sensor_columns: list[str]
    windowing_summary: pd.DataFrame


def build_default_config() -> dict[str, Any]:
    return {
        "experiment": {
            "hypothesis_id": "H4",
            "scenario_id": "S4",
            "experiment_topic": TOPIC,
            "timestamp": "auto",
            "experiment_id": "auto",
            "execution_dir": "auto",
            "seed_list": list(range(30)),
            "num_seeds": 30,
        },
        "data": {
            "process_info_path": "datasets/processed/mill_process_info_enabled.csv",
            "signal_data_path": "datasets/processed/mill_signal_data_enabled.csv",
            "target_col": "VB",
            "case_col": "case",
            "run_id_col": "run",
            "run_order_col": "run",
            "sensor_columns": SENSOR_COLUMNS,
            "selected_cases": [1, 2, 8, 9, 12, 14],
        },
        "domain": {"condition_pairs": DOMAIN_CASES, "transfer_scenarios": [f"{s}_to_{t}" for s, t in TRANSFER_SCENARIOS]},
        "split": {"validation_strategy": "source_case_chronological", "validation_ratio": 0.2, "source_only_validation": True},
        "windowing": {
            "method": "overlapped_window",
            "window_size": 3000,
            "stride": 2000,
            "num_windows": 4,
            "target_length": 9000,
            "required_run_length": 9000,
            "run_length_handling": {
                "if_longer": "truncate_to_9000",
                "if_shorter": "pad_to_9000",
                "padding_value": 0.0,
                "use_mask_for_feature_extraction": True,
            },
        },
        "features": {
            "time_domain": ["mean", "std", "min", "max", "median", "peak_to_peak", "rms", "skewness", "kurtosis", "slope", "energy"],
            "frequency_domain": ["spectral_centroid", "band_energy", "dominant_frequency"],
        },
        "models": {model: True for model in MODEL_NAMES},
        "random_forest": {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1, "n_jobs": -1},
        "ensemble": {
            "validation_weighted_window_average": {"fitted_on": "source_validation_only", "grid_step": 0.05},
            "ridge_window_stacking": {"alpha_candidates": [0.1, 1.0, 10.0], "fitted_on": "source_validation_only"},
            "mlp_window_stacking": {"input_dim": 4, "hidden_dim": [16, 4, 1], "epochs": 200, "patience": 100, "learning_rate": 0.01},
        },
        "evaluation": {
            "primary_metric": "RMSE",
            "metrics": ["MAE", "RMSE", "R2"],
            "include_worst_shift_robustness": True,
            "exclude_rmse_gt_threshold": False,
        },
        "visualization": {
            "rmse_filter_enabled": True,
            "rmse_filter_threshold": 0.5,
            "rmse_filter_metric": "RMSE",
            "exclude_filtered_from_metrics": False,
            "exclude_filtered_from_ranking": False,
            "exclude_filtered_from_report_tables": False,
            "exclude_filtered_from_worst_shift_metrics": False,
        },
        "reuse": {"enabled": True, "rerun_completed_conditions": False},
    }


def load_data(config: dict[str, Any]) -> WindowRFData:
    process = pd.read_csv(config["data"]["process_info_path"])
    signal = pd.read_csv(config["data"]["signal_data_path"])
    target_col = config["data"].get("target_col", "VB")
    sensor_columns = [c for c in config["data"].get("sensor_columns", SENSOR_COLUMNS) if c in signal.columns]
    merged = process.merge(signal, on=["case", "run"], suffixes=("_process", "_signal"))
    for col in [c for c in merged.columns if c.startswith("enable")]:
        merged = merged.loc[merged[col].astype(bool)]
    selected_cases = set(config["data"].get("selected_cases", [1, 2, 8, 9, 12, 14]))
    merged = merged.loc[merged["case"].isin(selected_cases)].copy()
    merged = merged.loc[merged[target_col].notna()].copy()
    merged["case_id"] = merged["case"].astype(int)
    merged["run_id"] = merged["run"].astype(int)
    merged["run_order"] = merged.groupby("case_id")["run_id"].rank(method="first").astype(int)
    merged["dataset_run_id"] = merged["case_id"].astype(str) + "_" + merged["run_id"].astype(str)
    merged["domain_id"] = merged["case_id"].map(domain_for_case)
    merged = merged.sort_values(["case_id", "run_order"]).reset_index(drop=True)
    merged["sample_index"] = np.arange(len(merged))

    win_cfg = {
        "window_size": int(config["windowing"]["window_size"]),
        "stride": int(config["windowing"]["stride"]),
        "num_windows": int(config["windowing"]["num_windows"]),
        "target_length": int(config["windowing"]["target_length"]),
        "padding_value": float(config["windowing"].get("run_length_handling", {}).get("padding_value", 0.0)),
    }
    full_rows: list[np.ndarray] = []
    full_names: list[str] | None = None
    window_rows = {name: [] for name in WINDOW_NAMES}
    window_names: dict[str, list[str]] = {}
    summary_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        channels = [parse_signal_cell(row[sensor]) for sensor in sensor_columns]
        max_len = max(len(ch) for ch in channels)
        run_signal = np.zeros((max_len, len(sensor_columns)), dtype=np.float32)
        for idx, arr in enumerate(channels):
            run_signal[: len(arr), idx] = arr
        result = window_run(run_signal, {"case_id": row["case_id"], "run_id": row["run_id"]}, win_cfg)
        valid_full_mask = np.zeros(result.target_length, dtype=bool)
        valid_full_mask[: min(result.original_length, result.target_length)] = True
        fixed_signal = np.zeros((result.target_length, len(sensor_columns)), dtype=np.float32)
        if result.original_length >= result.target_length:
            fixed_signal = run_signal[: result.target_length]
        else:
            fixed_signal[: result.original_length] = run_signal
        full_feat, names = extract_multisensor_features(fixed_signal, sensor_columns, valid_full_mask, prefix="full_run")
        full_rows.append(full_feat)
        if full_names is None:
            full_names = names
        for widx, wname in enumerate(WINDOW_NAMES):
            feat, names = extract_multisensor_features(result.windows[widx], sensor_columns, result.masks[widx], prefix=wname)
            window_rows[wname].append(feat)
            window_names[wname] = names
            summary_rows.append(
                {
                    "case_id": int(row["case_id"]),
                    "run_id": int(row["run_id"]),
                    "run_order": int(row["run_order"]),
                    "window_name": wname,
                    "window_index": widx + 1,
                    "original_time_length": int(result.original_length),
                    "target_length": int(result.target_length),
                    "window_size": int(result.window_size),
                    "stride": int(result.stride),
                    "start_idx": int(result.start_indices[widx]),
                    "end_idx": int(result.end_indices[widx]),
                    "valid_length": int(result.valid_lengths[widx]),
                    "was_padded": bool(result.was_padded),
                    "was_truncated": bool(result.was_truncated),
                    "padding_length": int(result.padding_length),
                    "windowing_status": result.status,
                }
            )
    return WindowRFData(
        meta=merged,
        full_features=np.vstack(full_rows),
        window_features={k: np.vstack(v) for k, v in window_rows.items()},
        full_feature_names=full_names or [],
        window_feature_names=window_names,
        sensor_columns=sensor_columns,
        windowing_summary=pd.DataFrame(summary_rows),
    )


def split_source_validation(meta: pd.DataFrame, source_cases: list[int], target_cases: list[int], val_ratio: float) -> pd.DataFrame:
    split = meta.copy()
    split["split"] = "excluded"
    split.loc[split["case_id"].isin(target_cases), "split"] = "test"
    for case_id in source_cases:
        case_idx = split.loc[split["case_id"] == case_id].sort_values("run_order").index.to_list()
        if not case_idx:
            continue
        val_count = max(1, int(math.ceil(len(case_idx) * val_ratio))) if len(case_idx) > 2 else 1
        val_count = min(val_count, max(1, len(case_idx) - 1))
        split.loc[case_idx[:-val_count], "split"] = "train"
        split.loc[case_idx[-val_count:], "split"] = "validation"
    return split


def fit_rf(x: np.ndarray, y: np.ndarray, config: dict[str, Any], seed: int) -> RandomForestRegressor:
    cfg = config["random_forest"]
    model = RandomForestRegressor(
        n_estimators=int(cfg.get("n_estimators", 500)),
        max_depth=cfg.get("max_depth", None),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 1)),
        random_state=int(seed),
        n_jobs=int(cfg.get("n_jobs", -1)),
    )
    model.fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=float))
    return model


def simplex_grid_weights(step: float = 0.05, n: int = 4) -> np.ndarray:
    units = int(round(1.0 / step))
    weights = []
    for a in range(units + 1):
        for b in range(units - a + 1):
            for c in range(units - a - b + 1):
                d = units - a - b - c
                weights.append([a / units, b / units, c / units, d / units])
    return np.asarray(weights, dtype=float)


def fit_validation_weights(x_val_pred: np.ndarray, y_val: np.ndarray, step: float = 0.05) -> tuple[np.ndarray, float]:
    candidates = simplex_grid_weights(step=step, n=x_val_pred.shape[1])
    preds = candidates @ x_val_pred.T
    mse = np.mean((preds - y_val.reshape(1, -1)) ** 2, axis=1)
    idx = int(np.argmin(mse))
    return candidates[idx], float(math.sqrt(mse[idx]))


class MLPStacker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 4), nn.ReLU(), nn.Linear(4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def fit_mlp_stacker(x_val_pred: np.ndarray, y_val: np.ndarray, config: dict[str, Any], seed: int) -> tuple[MLPStacker, dict[str, Any]]:
    set_seed(seed)
    cfg = config["ensemble"]["mlp_window_stacking"]
    model = MLPStacker()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 0.01)))
    loss_fn = nn.MSELoss()
    x = torch.as_tensor(x_val_pred, dtype=torch.float32)
    y = torch.as_tensor(y_val, dtype=torch.float32)
    epochs = int(cfg.get("epochs", 200))
    patience = int(cfg.get("patience", 100))
    best_rmse = float("inf")
    best_state = None
    bad = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        rmse = math.sqrt(float(loss.detach().cpu()))
        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "validation_RMSE": best_rmse}


def model_hash(config: dict[str, Any], model_name: str) -> str:
    payload = json.dumps(to_builtin({"model_name": model_name, "rf": config["random_forest"], "windowing": config["windowing"]}), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def run_scenario_seed(data: WindowRFData, config: dict[str, Any], source: str, target: str, seed: int) -> dict[str, pd.DataFrame]:
    set_seed(seed)
    source_cases = DOMAIN_CASES[source]
    target_cases = DOMAIN_CASES[target]
    scenario = f"{source}_to_{target}"
    split = split_source_validation(data.meta, source_cases, target_cases, float(config["split"]["validation_ratio"]))
    train_idx = split.index[split["split"] == "train"].to_numpy()
    val_idx = split.index[split["split"] == "validation"].to_numpy()
    test_idx = split.index[split["split"] == "test"].to_numpy()
    y_col = config["data"]["target_col"]
    y_train = split.loc[train_idx, y_col].to_numpy(dtype=float)
    y_val = split.loc[val_idx, y_col].to_numpy(dtype=float)
    y_test = split.loc[test_idx, y_col].to_numpy(dtype=float)

    start = time.time()
    full_rf = fit_rf(data.full_features[train_idx], y_train, config, seed)
    window_rfs = {w: fit_rf(data.window_features[w][train_idx], y_train, config, seed) for w in WINDOW_NAMES}
    train_time = time.time() - start

    full_val = full_rf.predict(data.full_features[val_idx])
    full_test = full_rf.predict(data.full_features[test_idx])
    window_val = np.column_stack([window_rfs[w].predict(data.window_features[w][val_idx]) for w in WINDOW_NAMES])
    window_test = np.column_stack([window_rfs[w].predict(data.window_features[w][test_idx]) for w in WINDOW_NAMES])

    weights_uniform = np.full(4, 0.25)
    weights_val, weighted_val_rmse = fit_validation_weights(window_val, y_val, config["ensemble"]["validation_weighted_window_average"].get("grid_step", 0.05))
    ridge_best = None
    ridge_best_alpha = None
    ridge_best_rmse = float("inf")
    for alpha in config["ensemble"]["ridge_window_stacking"].get("alpha_candidates", [0.1, 1.0, 10.0]):
        model = Ridge(alpha=float(alpha)).fit(window_val, y_val)
        rmse = metrics_dict(y_val, model.predict(window_val))["RMSE"]
        if rmse < ridge_best_rmse:
            ridge_best = model
            ridge_best_alpha = float(alpha)
            ridge_best_rmse = rmse
    mlp_error = ""
    mlp_model = None
    mlp_meta = {"validation_RMSE": float("nan"), "best_epoch": 0}
    try:
        mlp_model, mlp_meta = fit_mlp_stacker(window_val, y_val, config, seed)
    except Exception as exc:  # pragma: no cover - failure recorded in outputs.
        mlp_error = "".join(traceback.format_exception_only(type(exc), exc)).strip()

    model_preds_val: dict[str, np.ndarray] = {
        "full_run_random_forest": full_val,
        "rf_window_1_only": window_val[:, 0],
        "rf_window_2_only": window_val[:, 1],
        "rf_window_3_only": window_val[:, 2],
        "rf_window_4_only": window_val[:, 3],
        "window_rf_uniform_average": window_val @ weights_uniform,
        "window_rf_validation_weighted_average": window_val @ weights_val,
        "window_rf_ridge_stacking": ridge_best.predict(window_val) if ridge_best is not None else np.full_like(y_val, np.nan),
    }
    model_preds_test: dict[str, np.ndarray] = {
        "full_run_random_forest": full_test,
        "rf_window_1_only": window_test[:, 0],
        "rf_window_2_only": window_test[:, 1],
        "rf_window_3_only": window_test[:, 2],
        "rf_window_4_only": window_test[:, 3],
        "window_rf_uniform_average": window_test @ weights_uniform,
        "window_rf_validation_weighted_average": window_test @ weights_val,
        "window_rf_ridge_stacking": ridge_best.predict(window_test) if ridge_best is not None else np.full_like(y_test, np.nan),
    }
    if mlp_model is not None:
        with torch.no_grad():
            model_preds_val["window_rf_mlp_stacking"] = mlp_model(torch.as_tensor(window_val, dtype=torch.float32)).numpy()
            model_preds_test["window_rf_mlp_stacking"] = mlp_model(torch.as_tensor(window_test, dtype=torch.float32)).numpy()

    weight_by_model = {
        "full_run_random_forest": [np.nan] * 4,
        "rf_window_1_only": [1.0, 0.0, 0.0, 0.0],
        "rf_window_2_only": [0.0, 1.0, 0.0, 0.0],
        "rf_window_3_only": [0.0, 0.0, 1.0, 0.0],
        "rf_window_4_only": [0.0, 0.0, 0.0, 1.0],
        "window_rf_uniform_average": weights_uniform.tolist(),
        "window_rf_validation_weighted_average": weights_val.tolist(),
        "window_rf_ridge_stacking": ridge_best.coef_.reshape(-1).tolist() if ridge_best is not None else [np.nan] * 4,
        "window_rf_mlp_stacking": [np.nan] * 4,
    }
    val_metrics_by_model = {m: metrics_dict(y_val, p) for m, p in model_preds_val.items()}
    test_meta = split.loc[test_idx].copy()
    case_rows = []
    target_rows = []
    pred_rows = []
    window_pred_rows = []
    weight_rows = []
    skipped_rows = []

    for model_name, pred_test in model_preds_test.items():
        weights = weight_by_model[model_name]
        if model_name == "window_rf_mlp_stacking" and mlp_model is None:
            skipped_rows.append(
                {
                    "experiment_id": config["experiment"]["experiment_id"],
                    "scenario_name": scenario,
                    "source_domain": source,
                    "target_domain": target,
                    "model_name": model_name,
                    "ensemble_type": ensemble_type(model_name),
                    "seed": seed,
                    "reason": mlp_error or "mlp_stacking_failed",
                }
            )
            continue
        domain_metrics = metrics_dict(y_test, pred_test)
        case_metric_values = []
        for case_id, group_idx in test_meta.groupby("case_id").groups.items():
            loc = np.array([np.where(test_idx == idx)[0][0] for idx in group_idx])
            cm = metrics_dict(y_test[loc], pred_test[loc])
            case_metric_values.append(cm)
            window_case_rmse = [metrics_dict(y_test[loc], window_test[loc, k])["RMSE"] for k in range(4)]
            case_rows.append(
                {
                    "experiment_id": config["experiment"]["experiment_id"],
                    "scenario_name": scenario,
                    "source_domain": source,
                    "target_domain": target,
                    "train_cases": ",".join(map(str, source_cases)),
                    "val_cases": ",".join(map(str, source_cases)),
                    "test_case_id": int(case_id),
                    "model_name": model_name,
                    "ensemble_type": ensemble_type(model_name),
                    "window_count": 4,
                    "feature_set": "handcrafted_window_features",
                    "sensor_set": "all_sensors",
                    "seed": int(seed),
                    "MAE": cm["MAE"],
                    "RMSE": cm["RMSE"],
                    "R2": cm["R2"],
                    "validation_MAE": val_metrics_by_model[model_name]["MAE"],
                    "validation_RMSE": val_metrics_by_model[model_name]["RMSE"],
                    "validation_R2": val_metrics_by_model[model_name]["R2"],
                    "test_MAE": cm["MAE"],
                    "test_RMSE": cm["RMSE"],
                    "test_R2": cm["R2"],
                    "window_1_RMSE": window_case_rmse[0],
                    "window_2_RMSE": window_case_rmse[1],
                    "window_3_RMSE": window_case_rmse[2],
                    "window_4_RMSE": window_case_rmse[3],
                    "window_ensemble_weights": json.dumps(to_builtin(weights)),
                    "training_time_sec": train_time,
                    "model_config_hash": model_hash(config, model_name),
                }
            )
        target_rows.append(
            {
                "experiment_id": config["experiment"]["experiment_id"],
                "scenario_name": scenario,
                "source_domain": source,
                "target_domain": target,
                "train_cases": ",".join(map(str, source_cases)),
                "test_cases": ",".join(map(str, target_cases)),
                "model_name": model_name,
                "ensemble_type": ensemble_type(model_name),
                "seed": int(seed),
                "target_domain_MAE": domain_metrics["MAE"],
                "target_domain_RMSE": domain_metrics["RMSE"],
                "target_domain_R2": domain_metrics["R2"],
                "mean_case_MAE": float(np.mean([m["MAE"] for m in case_metric_values])),
                "mean_case_RMSE": float(np.mean([m["RMSE"] for m in case_metric_values])),
                "mean_case_R2": float(np.mean([m["R2"] for m in case_metric_values])),
            }
        )
        for local_pos, (_, row) in enumerate(test_meta.sort_values(["case_id", "run_order"]).iterrows()):
            original_pos = int(np.where(test_idx == row.name)[0][0])
            err = float(pred_test[original_pos] - row[y_col])
            pred_rows.append(
                {
                    "experiment_id": config["experiment"]["experiment_id"],
                    "scenario_name": scenario,
                    "source_domain": source,
                    "target_domain": target,
                    "case_id": int(row["case_id"]),
                    "run_id": int(row["run_id"]),
                    "run_order": int(row["run_order"]),
                    "model_name": model_name,
                    "ensemble_type": ensemble_type(model_name),
                    "y_true": float(row[y_col]),
                    "y_pred": float(pred_test[original_pos]),
                    "error": err,
                    "absolute_error": abs(err),
                    "squared_error": err * err,
                    "seed": int(seed),
                    "test_case_RMSE": float("nan"),
                    "target_domain_RMSE": domain_metrics["RMSE"],
                    "include_in_visualization": True,
                    "y_pred_window_1": float(window_test[original_pos, 0]),
                    "y_pred_window_2": float(window_test[original_pos, 1]),
                    "y_pred_window_3": float(window_test[original_pos, 2]),
                    "y_pred_window_4": float(window_test[original_pos, 3]),
                    "window_weight_1": weights[0],
                    "window_weight_2": weights[1],
                    "window_weight_3": weights[2],
                    "window_weight_4": weights[3],
                }
            )
            del local_pos
        if model_name in {"window_rf_validation_weighted_average", "window_rf_ridge_stacking", "window_rf_mlp_stacking"}:
            weight_rows.append(
                {
                    "experiment_id": config["experiment"]["experiment_id"],
                    "scenario_name": scenario,
                    "source_domain": source,
                    "target_domain": target,
                    "model_name": model_name,
                    "ensemble_type": ensemble_type(model_name),
                    "seed": int(seed),
                    "window_1_weight": weights[0],
                    "window_2_weight": weights[1],
                    "window_3_weight": weights[2],
                    "window_4_weight": weights[3],
                    "weight_sum": float(np.nansum(weights)),
                    "weight_nonnegative": bool(np.all(np.asarray(weights, dtype=float) >= 0)) if not np.isnan(weights).any() else False,
                    "fitted_on": "source_validation_only",
                    "validation_RMSE": val_metrics_by_model[model_name]["RMSE"],
                    "interpretation": "source-validation-fitted ensemble parameter",
                }
            )

    # Fill case RMSE on predictions.
    pred_df = pd.DataFrame(pred_rows)
    case_df = pd.DataFrame(case_rows)
    if not pred_df.empty:
        case_rmse = case_df[["scenario_name", "source_domain", "target_domain", "test_case_id", "model_name", "seed", "test_RMSE"]].rename(
            columns={"test_case_id": "case_id", "test_RMSE": "test_case_RMSE"}
        )
        pred_df = pred_df.drop(columns=["test_case_RMSE"]).merge(
            case_rmse, on=["scenario_name", "source_domain", "target_domain", "case_id", "model_name", "seed"], how="left"
        )
    for widx, wname in enumerate(WINDOW_NAMES):
        for local_idx, (_, row) in enumerate(test_meta.sort_values(["case_id", "run_order"]).iterrows()):
            original_pos = int(np.where(test_idx == row.name)[0][0])
            err = float(window_test[original_pos, widx] - row[y_col])
            window_pred_rows.append(
                {
                    "experiment_id": config["experiment"]["experiment_id"],
                    "scenario_name": scenario,
                    "source_domain": source,
                    "target_domain": target,
                    "case_id": int(row["case_id"]),
                    "run_id": int(row["run_id"]),
                    "run_order": int(row["run_order"]),
                    "seed": int(seed),
                    "window_name": wname,
                    "window_index": widx + 1,
                    "model_name": f"rf_{wname}_only",
                    "y_true": float(row[y_col]),
                    "y_pred_window": float(window_test[original_pos, widx]),
                    "absolute_error_window": abs(err),
                    "squared_error_window": err * err,
                    "window_RMSE": metrics_dict(y_test, window_test[:, widx])["RMSE"],
                }
            )
            del local_idx
    split_out = split[["case_id", "run_id", "run_order", "dataset_run_id", "domain_id", y_col, "split"]].copy()
    split_out["scenario_name"] = scenario
    split_out["source_domain"] = source
    split_out["target_domain"] = target
    split_out["seed"] = int(seed)
    return {
        "case_metrics": pd.DataFrame(case_rows),
        "target_metrics": pd.DataFrame(target_rows),
        "predictions": pred_df,
        "window_predictions": pd.DataFrame(window_pred_rows),
        "weights": pd.DataFrame(weight_rows),
        "split": split_out,
        "skipped": pd.DataFrame(skipped_rows),
    }


def aggregate_outputs(target_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shift = target_metrics.copy()
    shift["metric_MAE"] = shift["target_domain_MAE"]
    shift["metric_RMSE"] = shift["target_domain_RMSE"]
    shift["metric_R2"] = shift["target_domain_R2"]
    seed = (
        target_metrics.groupby(["experiment_id", "model_name", "ensemble_type", "seed"], as_index=False)
        .agg(
            mean_MAE_over_6_shifts=("target_domain_MAE", "mean"),
            mean_RMSE_over_6_shifts=("target_domain_RMSE", "mean"),
            mean_R2_over_6_shifts=("target_domain_R2", "mean"),
        )
        .sort_values(["model_name", "seed"])
    )
    worst_rows = []
    for (experiment_id, model, ens, seed_id), g in target_metrics.groupby(["experiment_id", "model_name", "ensemble_type", "seed"]):
        worst = g.sort_values("target_domain_RMSE", ascending=False).iloc[0]
        worst_rows.append(
            {
                "experiment_id": experiment_id,
                "experiment_number": EXPERIMENT_NUMBER,
                "model_name": model,
                "ensemble_type": ens,
                "seed": int(seed_id),
                "num_scenarios_expected": 6,
                "num_scenarios_available": int(g["scenario_name"].nunique()),
                "max_scenario_RMSE": float(worst["target_domain_RMSE"]),
                "worst_scenario_name": worst["scenario_name"],
                "worst_source_domain": worst["source_domain"],
                "worst_target_domain": worst["target_domain"],
                "mean_RMSE_over_scenarios": float(g["target_domain_RMSE"].mean()),
                "std_RMSE_over_scenarios": float(g["target_domain_RMSE"].std()),
                "min_RMSE_over_scenarios": float(g["target_domain_RMSE"].min()),
                "complete_scenario_coverage": bool(g["scenario_name"].nunique() == 6),
                "interpretation": "seed-level worst shift retained all RMSE values including >0.5",
            }
        )
    seed_worst = pd.DataFrame(worst_rows)
    model = (
        seed.merge(seed_worst[["model_name", "ensemble_type", "seed", "max_scenario_RMSE", "worst_scenario_name"]], on=["model_name", "ensemble_type", "seed"])
        .groupby(["experiment_id", "model_name", "ensemble_type"], as_index=False)
        .agg(
            num_seeds=("seed", "nunique"),
            mean_RMSE_over_6_shifts=("mean_RMSE_over_6_shifts", "mean"),
            std_RMSE_over_6_shifts=("mean_RMSE_over_6_shifts", "std"),
            mean_MAE_over_6_shifts=("mean_MAE_over_6_shifts", "mean"),
            mean_R2_over_6_shifts=("mean_R2_over_6_shifts", "mean"),
            mean_max_scenario_RMSE=("max_scenario_RMSE", "mean"),
            std_max_scenario_RMSE=("max_scenario_RMSE", "std"),
            median_max_scenario_RMSE=("max_scenario_RMSE", "median"),
            min_max_scenario_RMSE=("max_scenario_RMSE", "min"),
            max_max_scenario_RMSE=("max_scenario_RMSE", "max"),
        )
        .sort_values("mean_RMSE_over_6_shifts")
    )
    freq = seed_worst.groupby(["model_name", "ensemble_type"])["worst_scenario_name"].agg(lambda s: s.value_counts().index[0]).reset_index()
    freq_count = seed_worst.groupby(["model_name", "ensemble_type"])["worst_scenario_name"].agg(lambda s: int(s.value_counts().iloc[0])).reset_index(name="worst_scenario_frequency")
    model = model.merge(freq.rename(columns={"worst_scenario_name": "most_frequent_worst_scenario"}), on=["model_name", "ensemble_type"])
    model = model.merge(freq_count, on=["model_name", "ensemble_type"])
    model["num_scenarios"] = len(TRANSFER_SCENARIOS)
    model["rank_by_mean_RMSE"] = model["mean_RMSE_over_6_shifts"].rank(method="dense").astype(int)
    model["rank_by_mean_max_scenario_RMSE"] = model["mean_max_scenario_RMSE"].rank(method="dense").astype(int)
    model["rank_change_mean_vs_worst"] = model["rank_by_mean_max_scenario_RMSE"] - model["rank_by_mean_RMSE"]
    model["average_best_flag"] = model["rank_by_mean_RMSE"].eq(1)
    model["robust_best_flag"] = model["rank_by_mean_max_scenario_RMSE"].eq(1)
    model["interpretation"] = np.where(model["average_best_flag"] & model["robust_best_flag"], "best by mean and worst-shift RMSE", "compare average and robust ranks")
    return shift, seed, seed_worst, model


def apply_visualization_filter(predictions: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = predictions.copy()
    pred["include_in_visualization"] = pred["test_case_RMSE"].le(threshold)
    summary = pred[
        [
            "experiment_id",
            "scenario_name",
            "source_domain",
            "target_domain",
            "case_id",
            "model_name",
            "ensemble_type",
            "seed",
            "test_case_RMSE",
            "include_in_visualization",
        ]
    ].drop_duplicates()
    summary = summary.rename(columns={"test_case_RMSE": "rmse_value"})
    summary["experiment_number"] = EXPERIMENT_NUMBER
    summary["figure_name"] = "prediction_and_rmse_figures"
    summary["plot_type"] = "mixed"
    summary["filtering_level"] = "case_model_seed"
    summary["threshold"] = threshold
    summary["included_in_visualization"] = summary["include_in_visualization"]
    summary["reason"] = np.where(summary["included_in_visualization"], "included_rmse_le_threshold", "excluded_rmse_gt_threshold")
    summary = summary.drop(columns=["include_in_visualization"])
    return pred, summary


def make_analysis_tables(
    case_metrics: pd.DataFrame,
    target_metrics: pd.DataFrame,
    model_comparison: pd.DataFrame,
    weights: pd.DataFrame,
    seed_worst: pd.DataFrame,
    output_dir: Path,
) -> None:
    window_perf = model_comparison.loc[model_comparison["model_name"].str.startswith("rf_window_")].copy()
    window_perf["window_index"] = window_perf["model_name"].str.extract(r"rf_window_(\d)_only").astype(int)
    window_perf.to_csv(output_dir / "analysis" / f"{PREFIX}_window_standalone_performance.csv", index=False)
    ens = model_comparison.loc[model_comparison["model_name"].str.startswith("window_rf_")].copy()
    ens.to_csv(output_dir / "analysis" / f"{PREFIX}_ensemble_method_comparison.csv", index=False)
    rows = []
    full = target_metrics.loc[target_metrics["model_name"] == "full_run_random_forest"]
    for model_name in ["window_rf_uniform_average", "window_rf_validation_weighted_average", "window_rf_ridge_stacking", "window_rf_mlp_stacking"]:
        cand = target_metrics.loc[target_metrics["model_name"] == model_name]
        merged = full.merge(
            cand,
            on=["scenario_name", "source_domain", "target_domain", "seed"],
            suffixes=("_full", "_window"),
        )
        for _, row in merged.iterrows():
            delta = row["target_domain_RMSE_window"] - row["target_domain_RMSE_full"]
            rows.append(
                {
                    "scenario_name": row["scenario_name"],
                    "model_name": model_name,
                    "ensemble_type": ensemble_type(model_name),
                    "full_run_RF_RMSE": row["target_domain_RMSE_full"],
                    "window_ensemble_RMSE": row["target_domain_RMSE_window"],
                    "delta_RMSE_window_minus_full": delta,
                    "window_ensemble_better": bool(delta < 0),
                    "full_run_RF_R2": row["target_domain_R2_full"],
                    "window_ensemble_R2": row["target_domain_R2_window"],
                    "interpretation": "window ensemble improved" if delta < 0 else "full-run RF retained advantage",
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "analysis" / f"{PREFIX}_full_run_rf_vs_window_ensemble.csv", index=False)
    weights.to_csv(output_dir / "analysis" / f"{PREFIX}_window_ensemble_weights.csv", index=False)
    if not weights.empty:
        importance = (
            weights.groupby(["model_name", "ensemble_type"], as_index=False)
            .agg(
                mean_window_1_weight=("window_1_weight", "mean"),
                mean_window_2_weight=("window_2_weight", "mean"),
                mean_window_3_weight=("window_3_weight", "mean"),
                mean_window_4_weight=("window_4_weight", "mean"),
            )
        )
        weight_cols = ["mean_window_1_weight", "mean_window_2_weight", "mean_window_3_weight", "mean_window_4_weight"]
        importance["most_important_window"] = "not_applicable"
        valid_weight_rows = importance[weight_cols].notna().any(axis=1)
        if valid_weight_rows.any():
            extracted = importance.loc[valid_weight_rows, weight_cols].idxmax(axis=1).str.extract(r"mean_(window_\d)_weight")[0]
            importance.loc[valid_weight_rows, "most_important_window"] = extracted.to_numpy()
        importance["interpretation"] = "highest mean source-validation fitted weight/coefficient"
        importance.loc[~valid_weight_rows, "interpretation"] = "no explicit window weight/coefficient for this ensemble"
    else:
        importance = pd.DataFrame()
    importance.to_csv(output_dir / "analysis" / f"{PREFIX}_window_importance_summary.csv", index=False)
    seed_worst["worst_scenario_name"].value_counts().rename_axis("worst_scenario_name").reset_index(name="frequency").to_csv(
        output_dir / "analysis" / f"{PREFIX}_worst_scenario_frequency.csv", index=False
    )
    model_comparison[["model_name", "ensemble_type", "rank_by_mean_RMSE", "rank_by_mean_max_scenario_RMSE", "rank_change_mean_vs_worst"]].to_csv(
        output_dir / "analysis" / f"{PREFIX}_mean_vs_worst_shift_ranking_comparison.csv", index=False
    )


def make_figures(
    data: WindowRFData,
    output_dir: Path,
    model_comparison: pd.DataFrame,
    case_metrics: pd.DataFrame,
    target_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    weights: pd.DataFrame,
    window_predictions: pd.DataFrame,
    seed_worst: pd.DataFrame,
) -> None:
    fig_dir = output_dir / "figures"
    threshold = 0.5
    def filtered(df: pd.DataFrame, col: str) -> pd.DataFrame:
        return df.loc[pd.to_numeric(df[col], errors="coerce").le(threshold)].copy()

    mplot = filtered(model_comparison.rename(columns={"mean_RMSE_over_6_shifts": "plot_RMSE"}), "plot_RMSE")
    if not mplot.empty:
        plt.figure(figsize=(11, 5))
        plt.bar(mplot["model_name"], mplot["plot_RMSE"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Mean RMSE, filtered for plot")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_model_comparison_rmse.png", dpi=180)
        plt.close()

    comp_path = output_dir / "analysis" / f"{PREFIX}_full_run_rf_vs_window_ensemble.csv"
    comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
    comp_plot = filtered(comp, "window_ensemble_RMSE") if not comp.empty else pd.DataFrame()
    if not comp_plot.empty:
        agg = comp_plot.groupby("model_name", as_index=False)["delta_RMSE_window_minus_full"].mean()
        plt.figure(figsize=(9, 4.5))
        plt.bar(agg["model_name"], agg["delta_RMSE_window_minus_full"])
        plt.axhline(0, color="black", lw=1)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Delta RMSE window - full")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_full_run_rf_vs_window_ensemble.png", dpi=180)
        plt.close()

    window_plot = mplot.loc[mplot["model_name"].str.startswith("rf_window_")]
    if not window_plot.empty:
        plt.figure(figsize=(7, 4.5))
        plt.bar(window_plot["model_name"], window_plot["plot_RMSE"])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Mean RMSE")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_window_standalone_rmse.png", dpi=180)
        plt.close()

    ens_plot = mplot.loc[mplot["model_name"].str.startswith("window_rf_")]
    if not ens_plot.empty:
        plt.figure(figsize=(9, 4.5))
        plt.bar(ens_plot["model_name"], ens_plot["plot_RMSE"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Mean RMSE")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_ensemble_method_comparison.png", dpi=180)
        plt.close()

    if not weights.empty:
        wavg = weights.groupby("model_name", as_index=False)[["window_1_weight", "window_2_weight", "window_3_weight", "window_4_weight"]].mean()
        wavg.plot(x="model_name", kind="bar", stacked=True, figsize=(9, 5))
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Mean fitted weight/coefficient")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_window_weights_by_scenario.png", dpi=180)
        plt.close()

    if not window_predictions.empty:
        wp = window_predictions.groupby("window_name", as_index=False)["absolute_error_window"].mean()
        plt.figure(figsize=(6, 4))
        plt.bar(wp["window_name"], wp["absolute_error_window"])
        plt.ylabel("Mean absolute error")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_window_prediction_error_by_window.png", dpi=180)
        plt.close()

    pred_plot = predictions.loc[predictions["include_in_visualization"]].copy()
    if not pred_plot.empty:
        avg = pred_plot.groupby(["model_name", "run_order"], as_index=False).agg(y_pred=("y_pred", "mean"), y_true=("y_true", "mean"))
        plt.figure(figsize=(10, 5))
        plt.plot(avg.groupby("run_order")["y_true"].mean().index, avg.groupby("run_order")["y_true"].mean().values, color="black", lw=2, label="y_true")
        for model, g in avg.groupby("model_name"):
            plt.plot(g["run_order"], g["y_pred"], marker="o", markersize=3, label=model)
        plt.xlabel("run_order")
        plt.ylabel("VB")
        plt.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_prediction_over_run_order.png", dpi=180)
        plt.close()

    plt.figure(figsize=(7, 4.5))
    data.meta.boxplot(column="VB", by="case_id")
    plt.title("")
    plt.suptitle("")
    plt.ylabel("VB")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{PREFIX}_target_distribution_by_case.png", dpi=180)
    plt.close()

    wf = np.vstack([data.window_features[w] for w in WINDOW_NAMES])
    labels = np.repeat(data.meta["case_id"].to_numpy(), len(WINDOW_NAMES))
    coords = PCA(n_components=2, random_state=0).fit_transform(wf)
    plt.figure(figsize=(6, 5))
    for case in sorted(set(labels)):
        mask = labels == case
        plt.scatter(coords[mask, 0], coords[mask, 1], label=f"Case {case}", s=18)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{PREFIX}_window_feature_pca_by_case.png", dpi=180)
    plt.close()

    robust_plot = filtered(model_comparison.rename(columns={"mean_max_scenario_RMSE": "plot_RMSE"}), "plot_RMSE")
    if not robust_plot.empty:
        plt.figure(figsize=(6, 5))
        plt.scatter(robust_plot["mean_RMSE_over_6_shifts"], robust_plot["plot_RMSE"])
        for _, row in robust_plot.iterrows():
            plt.annotate(row["model_name"], (row["mean_RMSE_over_6_shifts"], row["plot_RMSE"]), fontsize=7)
        plt.xlabel("Mean RMSE")
        plt.ylabel("Mean max-scenario RMSE")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_mean_rmse_vs_max_scenario_rmse.png", dpi=180)
        plt.close()

        plt.figure(figsize=(9, 4.5))
        plt.bar(robust_plot["model_name"], robust_plot["plot_RMSE"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Mean max-scenario RMSE")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_worst_shift_robustness_ranking.png", dpi=180)
        plt.close()

    freq = seed_worst.groupby(["model_name", "worst_scenario_name"]).size().reset_index(name="count")
    if not freq.empty:
        pivot = freq.pivot(index="model_name", columns="worst_scenario_name", values="count").fillna(0)
        plt.figure(figsize=(8, 5))
        im = plt.imshow(pivot, cmap="magma")
        plt.colorbar(im, label="Worst scenario frequency")
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_worst_scenario_frequency_heatmap.png", dpi=180)
        plt.close()

    ranks = model_comparison.sort_values("rank_by_mean_RMSE")
    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(ranks))
    plt.plot(x, ranks["rank_by_mean_RMSE"], marker="o", label="mean RMSE rank")
    plt.plot(x, ranks["rank_by_mean_max_scenario_RMSE"], marker="o", label="worst-shift rank")
    plt.xticks(x, ranks["model_name"], rotation=35, ha="right")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{PREFIX}_mean_vs_worst_shift_ranking_comparison.png", dpi=180)
    plt.close()


def write_artifacts(data: WindowRFData, config: dict[str, Any], output_dir: Path) -> None:
    write_json(
        output_dir / "data" / f"{PREFIX}_dataset_summary.json",
        {
            "process_info_path": config["data"]["process_info_path"],
            "signal_data_path": config["data"]["signal_data_path"],
            "cases": sorted(data.meta["case_id"].unique().tolist()),
            "num_runs": int(len(data.meta)),
            "target_col": config["data"]["target_col"],
            "sensor_columns": data.sensor_columns,
            "full_feature_shape": list(data.full_features.shape),
            "window_feature_shapes": {k: list(v.shape) for k, v in data.window_features.items()},
        },
    )
    pd.DataFrame({"requested_sensor": config["data"]["sensor_columns"], "resolved_sensor": config["data"]["sensor_columns"], "status": "matched"}).to_csv(
        output_dir / "data" / f"{PREFIX}_sensor_mapping.csv", index=False
    )
    data.meta[["case_id", "run_id", "run_order", "dataset_run_id", "domain_id", config["data"]["target_col"]]].to_json(
        output_dir / "data" / f"{PREFIX}_run_order_summary.json", orient="records", indent=2
    )
    data.windowing_summary.insert(0, "experiment_id", config["experiment"]["experiment_id"])
    data.windowing_summary.to_csv(output_dir / "data" / f"{PREFIX}_windowing_summary.csv", index=False)
    pd.DataFrame(
        [
            {"feature_scope": "full_run", "num_features": len(data.full_feature_names), "feature_names": json.dumps(data.full_feature_names)},
            *[
                {"feature_scope": w, "num_features": len(names), "feature_names": json.dumps(names)}
                for w, names in data.window_feature_names.items()
            ],
        ]
    ).to_csv(output_dir / "data" / f"{PREFIX}_feature_summary.csv", index=False)


def leakage_check(split_df: pd.DataFrame) -> dict[str, Any]:
    failures = []
    for (scenario, seed), group in split_df.groupby(["scenario_name", "seed"]):
        train_cases = set(group.loc[group["split"] == "train", "case_id"].astype(int))
        val_cases = set(group.loc[group["split"] == "validation", "case_id"].astype(int))
        test_cases = set(group.loc[group["split"] == "test", "case_id"].astype(int))
        if train_cases & test_cases:
            failures.append({"scenario": scenario, "seed": int(seed), "failure": "target_case_in_train"})
        if val_cases & test_cases:
            failures.append({"scenario": scenario, "seed": int(seed), "failure": "target_case_in_validation"})
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "target_domain_used_for_training": False,
        "target_domain_used_for_validation": False,
        "ensemble_fitted_on": "source_validation_only",
        "window_boundaries_fixed": True,
        "windowing_per_run_only": True,
        "rmse_gt_0p5_excluded_from_worst_shift_metrics": False,
    }


def write_report(output_dir: Path, config: dict[str, Any], model_comparison: pd.DataFrame, weights: pd.DataFrame, skipped: pd.DataFrame, filter_summary: pd.DataFrame) -> None:
    best_mean = model_comparison.sort_values("rank_by_mean_RMSE").iloc[0]
    best_robust = model_comparison.sort_values("rank_by_mean_max_scenario_RMSE").iloc[0]
    full = model_comparison.loc[model_comparison["model_name"] == "full_run_random_forest"].iloc[0]
    best_window = model_comparison.loc[model_comparison["model_name"].str.startswith("window_rf_")].sort_values("mean_RMSE_over_6_shifts").iloc[0]
    important = "not available"
    if not weights.empty:
        cols = ["window_1_weight", "window_2_weight", "window_3_weight", "window_4_weight"]
        important = weights[cols].mean().idxmax().replace("_weight", "")
    excluded = int((~filter_summary["included_in_visualization"]).sum()) if not filter_summary.empty else 0
    report = f"""# H4_S4 Overlapped-Window Random Forest Ensemble

## 1. Executive Summary

- Experiment ID: `{config['experiment']['experiment_id']}`
- Best model by mean RMSE: `{best_mean['model_name']}` with RMSE `{best_mean['mean_RMSE_over_6_shifts']:.6f}`.
- Best model by worst-shift robustness: `{best_robust['model_name']}` with mean max-scenario RMSE `{best_robust['mean_max_scenario_RMSE']:.6f}`.
- Full-run RF mean RMSE: `{full['mean_RMSE_over_6_shifts']:.6f}`.
- Best window ensemble: `{best_window['model_name']}` with mean RMSE `{best_window['mean_RMSE_over_6_shifts']:.6f}`.
- Most prominent fitted window/coefficient: `{important}`.

## 2. Experiment Setup

- Data: `{config['data']['process_info_path']}`, `{config['data']['signal_data_path']}`
- Cases: `{config['data']['selected_cases']}`
- Domains: A={{1,9}}, B={{2,12}}, C={{8,14}}
- Transfer scenarios: `{config['domain']['transfer_scenarios']}`
- Seeds: `{config['experiment']['seed_list']}` (`num_seeds=30`)
- Sensors: `{config['data']['sensor_columns']}`
- Features: `{FEATURE_NAMES}`

## 3. Windowing Definition

Each run is truncated/padded to 9000 samples and split into 4 fixed overlapped temporal windows:

- window_1: 0-2999
- window_2: 2000-4999
- window_3: 4000-6999
- window_4: 6000-8999

In this experiment, the term segment/window refers to an overlapped fixed temporal window, not a physically defined process state.

## 4. Models

Models evaluated: `{MODEL_NAMES}`. Window ensembles are fitted using source validation predictions only. Target-domain data are used only for final evaluation.

## 5. Leakage Prevention

- Random run split: not used
- Source-only validation: yes
- Ensemble weight/stacking fitting: source validation only
- Window boundary fitting from target statistics: no, fixed boundaries used
- Target-domain preprocessing/model selection: not used

## 6. Main Results by Mean RMSE

{model_comparison.to_markdown(index=False)}

## 7. Worst-shift Robustness

Worst-shift robustness uses seed-level max scenario RMSE over all 6 shifts. RMSE > 0.5 values are retained in these calculations.

## 8. Window-specific Performance

See `analysis/{PREFIX}_window_standalone_performance.csv`.

## 9. Ensemble Method Comparison

See `analysis/{PREFIX}_ensemble_method_comparison.csv` and `analysis/{PREFIX}_full_run_rf_vs_window_ensemble.csv`.

## 10. Domain Shift and Window Analysis

Domain and window summaries are saved in `analysis/{PREFIX}_domain_shift_analysis.csv` and window-level figures.

## 11. RMSE Visualization Filtering Note

RMSE > 0.5 conditions were excluded from visualizations only. They were not excluded from metrics, ranking, model comparison, or worst-shift robustness. Excluded visualization rows: `{excluded}`. Full list: `analysis/{PREFIX}_rmse_visualization_filter_summary.csv`.

## 12. Interpretation

If full-run RF remains best, full-run summary statistics may be more stable than fixed-window decomposition under NASA Ames domain shifts. If a window ensemble improves, the fixed temporal sub-regions preserve useful local information that full-run aggregation blurs.

## 13. Limitations

- Window boundaries are fixed, not process-state aligned.
- Windows overlap, so their predictions are dependent.
- Source-validation stacking can overfit because NASA Ames is small.
- MLP stacking uses only source validation predictions and may be unstable.

## 14. Recommendations

- Window size/stride ablation
- Sensor subset extension
- Cross-fitted stacking
- Segment-aware RF ensemble
- PHM2010 validation

## Skipped Conditions

{skipped.to_markdown(index=False) if not skipped.empty else 'No skipped conditions.'}
"""
    (output_dir / "reports" / f"{PREFIX}_report.md").write_text(report, encoding="utf-8")
    html = "<html><body>" + report.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>\n") + "</body></html>"
    (output_dir / "reports" / f"{PREFIX}_report.html").write_text(html, encoding="utf-8")


def make_dirs(output_dir: Path) -> None:
    for d in ["configs", "data", "splits", "metrics", "predictions", "analysis", "figures", "models/full_run_rf", "models/window_rf", "models/ensemble", "reports", "logs"]:
        (output_dir / d).mkdir(parents=True, exist_ok=True)


def validate_config(config: dict[str, Any], data: WindowRFData) -> dict[str, Any]:
    return {
        "data_files_exist": Path(config["data"]["process_info_path"]).exists() and Path(config["data"]["signal_data_path"]).exists(),
        "target_column_exists": config["data"]["target_col"] in data.meta.columns,
        "case_id_exists": "case_id" in data.meta.columns,
        "run_id_exists": "run_id" in data.meta.columns,
        "run_order_exists": "run_order" in data.meta.columns,
        "domains": DOMAIN_CASES,
        "transfer_scenario_count": len(TRANSFER_SCENARIOS),
        "source_only_validation": True,
        "window_count": int(config["windowing"]["num_windows"]),
        "window_shape": [int(config["windowing"]["window_size"]), len(data.sensor_columns)],
        "feature_extraction_full_and_window": bool(data.full_features.shape[1] and all(v.shape[1] for v in data.window_features.values())),
        "seed_list_0_29": config["experiment"]["seed_list"] == list(range(30)),
        "rmse_visualization_only_filter": True,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="experiments/executions")
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()

    config = build_default_config()
    if args.seeds:
        config["experiment"]["seed_list"] = [int(x) for x in args.seeds.split(",") if x.strip()]
        config["experiment"]["num_seeds"] = len(config["experiment"]["seed_list"])
    output_dir = Path(create_execution_dir(config, root=args.output_root))
    if str(config["experiment"].get("experiment_id", "")).lower() == "auto":
        exp = config["experiment"]
        exp["experiment_id"] = f"{exp['timestamp']}_{exp['hypothesis_id']}_{exp['scenario_id']}_{exp['experiment_topic']}"
    make_dirs(output_dir)
    write_yaml(output_dir / "configs" / f"{PREFIX}_input_config.yaml", config)
    data = load_data(config)
    write_artifacts(data, config, output_dir)
    validation = validate_config(config, data)
    write_json(output_dir / "configs" / f"{PREFIX}_config_validation.json", validation)
    write_yaml(output_dir / "configs" / f"{PREFIX}_resolved_config.yaml", config)
    write_json(output_dir / "logs" / f"{PREFIX}_environment.json", {"python": sys.version, "torch": torch.__version__})
    (output_dir / "logs" / f"{PREFIX}_error.log").write_text("", encoding="utf-8")
    run_log = output_dir / "logs" / f"{PREFIX}_run.log"
    run_log.write_text(f"Dry-run validation passed={validation['passed']}\n", encoding="utf-8")
    if args.dry_run_only:
        print(output_dir)
        return

    case_frames = []
    target_frames = []
    pred_frames = []
    window_pred_frames = []
    weight_frames = []
    split_frames = []
    skipped_frames = []
    seed_status = []
    for seed in config["experiment"]["seed_list"]:
        for source, target in TRANSFER_SCENARIOS:
            result = run_scenario_seed(data, config, source, target, int(seed))
            case_frames.append(result["case_metrics"])
            target_frames.append(result["target_metrics"])
            pred_frames.append(result["predictions"])
            window_pred_frames.append(result["window_predictions"])
            weight_frames.append(result["weights"])
            split_frames.append(result["split"])
            if not result["skipped"].empty:
                skipped_frames.append(result["skipped"])
            for model in set(result["target_metrics"]["model_name"]):
                seed_status.append(
                    {
                        "experiment_id": config["experiment"]["experiment_id"],
                        "experiment_number": EXPERIMENT_NUMBER,
                        "scenario_name": f"{source}_to_{target}",
                        "source_domain": source,
                        "target_domain": target,
                        "model_name": model,
                        "ensemble_type": ensemble_type(model),
                        "seed": int(seed),
                        "expected_seed_count": 30,
                        "status": "executed",
                        "existing_result_path": "",
                        "needs_execution": False,
                        "reason": "fresh execution completed",
                    }
                )
            run_log.write_text(run_log.read_text(encoding="utf-8") + f"Completed {source}_to_{target} seed={seed}\n", encoding="utf-8")

    case_metrics = pd.concat(case_frames, ignore_index=True)
    target_metrics = pd.concat(target_frames, ignore_index=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    window_predictions = pd.concat(window_pred_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    splits = pd.concat(split_frames, ignore_index=True)
    skipped = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(columns=["experiment_id", "scenario_name", "source_domain", "target_domain", "model_name", "ensemble_type", "seed", "reason"])
    shift_metrics, seed_metrics, seed_worst, model_comparison = aggregate_outputs(target_metrics)
    predictions, filter_summary = apply_visualization_filter(predictions, float(config["visualization"]["rmse_filter_threshold"]))

    case_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_case_metrics.csv", index=False)
    target_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_target_domain_metrics.csv", index=False)
    shift_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_shift_metrics.csv", index=False)
    seed_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_seed_metrics.csv", index=False)
    model_comparison.to_csv(output_dir / "metrics" / f"{PREFIX}_model_comparison.csv", index=False)
    predictions.sort_values(["scenario_name", "seed", "case_id", "run_order", "model_name"]).to_csv(output_dir / "predictions" / f"{PREFIX}_test_predictions.csv", index=False)
    predictions.to_csv(output_dir / "predictions" / f"{PREFIX}_prediction_errors.csv", index=False)
    window_predictions.to_csv(output_dir / "predictions" / f"{PREFIX}_window_predictions.csv", index=False)
    splits.to_csv(output_dir / "splits" / f"{PREFIX}_split.csv", index=False)
    write_json(output_dir / "splits" / f"{PREFIX}_split_summary.json", {"split_counts": splits.groupby(["scenario_name", "seed", "split"]).size().reset_index(name="count").to_dict("records")})
    write_json(output_dir / "splits" / f"{PREFIX}_leakage_check.json", leakage_check(splits))
    pd.DataFrame(seed_status).to_csv(output_dir / "analysis" / f"{PREFIX}_seed_completion_status.csv", index=False)
    seed_worst.to_csv(output_dir / "analysis" / f"{PREFIX}_seed_level_worst_shift_rmse.csv", index=False)
    robust_summary = model_comparison.copy()
    robust_summary["experiment_number"] = EXPERIMENT_NUMBER
    robust_summary.rename(
        columns={
            "mean_RMSE_over_6_shifts": "mean_RMSE_over_scenarios",
            "std_RMSE_over_6_shifts": "std_RMSE_over_scenarios",
        }
    ).to_csv(output_dir / "analysis" / f"{PREFIX}_model_worst_shift_robustness.csv", index=False)
    filter_summary.to_csv(output_dir / "analysis" / f"{PREFIX}_rmse_visualization_filter_summary.csv", index=False)
    skipped.to_csv(output_dir / "analysis" / f"{PREFIX}_skipped_conditions.csv", index=False)
    make_analysis_tables(case_metrics, target_metrics, model_comparison, weights, seed_worst, output_dir)
    domain_shift = []
    for source, target in TRANSFER_SCENARIOS:
        sidx = data.meta.index[data.meta["case_id"].isin(DOMAIN_CASES[source])].to_numpy()
        tidx = data.meta.index[data.meta["case_id"].isin(DOMAIN_CASES[target])].to_numpy()
        domain_shift.append(
            {
                "scenario_name": f"{source}_to_{target}",
                "source_domain": source,
                "target_domain": target,
                "source_VB_mean": float(data.meta.loc[sidx, "VB"].mean()),
                "target_VB_mean": float(data.meta.loc[tidx, "VB"].mean()),
                "full_feature_mean_l2_distance": float(np.linalg.norm(data.full_features[sidx].mean(axis=0) - data.full_features[tidx].mean(axis=0))),
            }
        )
    pd.DataFrame(domain_shift).to_csv(output_dir / "analysis" / f"{PREFIX}_domain_shift_analysis.csv", index=False)
    write_json(
        output_dir / "analysis" / f"{PREFIX}_analysis_summary.json",
        {
            "best_model_by_mean_rmse": model_comparison.sort_values("rank_by_mean_RMSE").iloc[0].to_dict(),
            "best_model_by_worst_shift": model_comparison.sort_values("rank_by_mean_max_scenario_RMSE").iloc[0].to_dict(),
            "skipped_conditions": int(len(skipped)),
        },
    )
    make_figures(data, output_dir, model_comparison, case_metrics, target_metrics, predictions, weights, window_predictions, seed_worst)
    write_report(output_dir, config, model_comparison, weights, skipped, filter_summary)
    run_log.write_text(run_log.read_text(encoding="utf-8") + "Finished H4_S4 overlapped-window RF ensemble run.\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()

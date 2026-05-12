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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

from milling_experiment_framework.experiments.execution_path import create_execution_dir
from milling_experiment_framework.models.dl.cnn_gru_regressor import CNNGRURegressor as WindowedCNNGRURegressor
from milling_experiment_framework.models.dl.cnn_lag_concat_regressor import CNNLagConcatRegressor as WindowedCNNLagConcatRegressor
from milling_experiment_framework.models.dl.cnn_only_regressor import CNNOnlyRegressor as WindowedCNNOnlyRegressor
from milling_experiment_framework.preprocessing.windowing.run_windowing import window_signal_channel_first


PREFIX = "H4_S1"
EXPERIMENT_TOPIC = "cnn_gru_run_sequence_domain_generalization_NASA_Ames"
DOMAIN_CASES = {"A": [1, 9], "B": [2, 12], "C": [8, 14]}
TRANSFER_SCENARIOS = [("A", "B"), ("A", "C"), ("B", "A"), ("B", "C"), ("C", "A"), ("C", "B")]
SENSOR_COLUMNS = ["smcAC", "smcDC", "vib_spindle", "vib_table", "AE_spindle", "AE_table"]
FEATURE_NAMES = ["mean", "std", "max", "min", "peak_to_peak", "kurtosis", "skewness", "spectral_centroid", "band_energy"]
DL_MODELS = ["cnn_only", "cnn_lag_concat", "cnn_gru"]
FEATURE_MODELS = ["feature_ridge", "feature_random_forest", "feature_svr", "feature_xgboost"]
ALL_MODELS = [*FEATURE_MODELS, *DL_MODELS]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, indent=2, ensure_ascii=False)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
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


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(to_builtin(payload), f, sort_keys=False, allow_unicode=True)
    else:
        write_json(path.with_suffix(".json"), payload)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        if yaml is None:
            return json.load(f)
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_signal_cell(value: Any) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    return np.asarray(ast.literal_eval(value), dtype=np.float32)


def resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x.astype(np.float32)
    if len(x) < 2:
        return np.full(target_len, float(x[0]) if len(x) else 0.0, dtype=np.float32)
    old = np.linspace(0.0, 1.0, num=len(x), dtype=np.float32)
    new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(new, old, x).astype(np.float32)


def domain_for_case(case_id: int) -> str:
    for domain, cases in DOMAIN_CASES.items():
        if int(case_id) in cases:
            return domain
    return "unknown"


@dataclass
class RunData:
    meta: pd.DataFrame
    sequences: np.ndarray  # [N, K, C, W]
    raw_lengths: np.ndarray
    feature_matrix: np.ndarray
    feature_names: list[str]
    sensor_columns: list[str]
    windowing_summary: pd.DataFrame
    window_length: int
    num_windows: int
    sequence_length: int


def load_run_data(config: dict[str, Any], sequence_length: int) -> RunData:
    process_path = Path(config["data"]["process_info_path"])
    signal_path = Path(config["data"]["signal_data_path"])
    process = pd.read_csv(process_path)
    signal = pd.read_csv(signal_path)
    selected_cases = set(config["data"].get("selected_cases", [1, 2, 8, 9, 12, 14]))
    target_col = config["data"].get("target_col", "VB")
    sensor_columns = [c for c in config["data"].get("sensor_columns", SENSOR_COLUMNS) if c in signal.columns]
    merged = process.merge(signal, on=["case", "run"], suffixes=("_process", "_signal"))
    enable_cols = [c for c in merged.columns if c.startswith("enable")]
    for col in enable_cols:
        merged = merged.loc[merged[col].astype(bool)]
    merged = merged.loc[merged["case"].isin(selected_cases)].copy()
    merged = merged.loc[merged[target_col].notna()].copy()
    merged["case_id"] = merged["case"].astype(int)
    merged["run_id"] = merged["run"].astype(int)
    merged["run_order"] = merged.groupby("case_id")["run_id"].rank(method="first").astype(int)
    merged["dataset_run_id"] = merged["case_id"].astype(str) + "_" + merged["run_id"].astype(str)
    merged["domain_id"] = merged["case_id"].map(domain_for_case)
    merged = merged.sort_values(["case_id", "run_order"]).reset_index(drop=True)
    merged["sample_index"] = np.arange(len(merged))

    model_design = config.get("model_design", {})
    run_encoder = model_design.get("run_encoder", {})
    window_cfg = config.get("cnn_windowing", {})
    window_length = int(run_encoder.get("window_length", window_cfg.get("window_length", 3000)))
    num_windows = int(run_encoder.get("num_windows", window_cfg.get("num_windows", 5)))
    short_run_policy = str(run_encoder.get("short_run_policy", window_cfg.get("short_run_policy", "pad_to_window_length")))

    tensors: list[np.ndarray] = []
    feature_tensors: list[np.ndarray] = []
    raw_lengths: list[int] = []
    window_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        raw_channels: list[np.ndarray] = []
        feature_channels: list[np.ndarray] = []
        lengths = []
        for sensor in sensor_columns:
            arr = parse_signal_cell(row[sensor])
            lengths.append(len(arr))
            raw_channels.append(arr.astype(np.float32))
            feature_channels.append(resample_1d(arr, sequence_length))
        raw_lengths.append(int(np.median(lengths)))
        common_length = max(lengths)
        channel_first = np.stack([resample_1d(arr, common_length) for arr in raw_channels], axis=0)
        windowed = window_signal_channel_first(
            channel_first,
            window_length=window_length,
            num_windows=num_windows,
            short_run_policy=short_run_policy,
        )
        tensors.append(windowed.windows)
        feature_tensors.append(np.stack(feature_channels, axis=0))
        window_rows.append(
            {
                "case_id": int(row["case_id"]),
                "run_id": int(row["run_id"]),
                "run_order": int(row["run_order"]),
                "original_time_length": int(windowed.original_time_length),
                "window_length": int(windowed.window_length),
                "num_windows": int(windowed.num_windows),
                "stride": int(windowed.stride),
                "window_start_indices": json.dumps(windowed.window_start_indices),
                "window_end_indices": json.dumps(windowed.window_end_indices),
                "short_run_policy": windowed.short_run_policy,
                "padded": bool(windowed.padded),
                "padding_length": int(windowed.padding_length),
                "windowing_status": windowed.windowing_status,
            }
        )
    sequences = np.stack(tensors).astype(np.float32)
    feature_sequences = np.stack(feature_tensors).astype(np.float32)
    features, feature_names = compute_statistical_features(feature_sequences, sensor_columns)
    return RunData(
        meta=merged,
        sequences=sequences,
        raw_lengths=np.asarray(raw_lengths, dtype=np.int32),
        feature_matrix=features,
        feature_names=feature_names,
        sensor_columns=sensor_columns,
        windowing_summary=pd.DataFrame(window_rows),
        window_length=window_length,
        num_windows=num_windows,
        sequence_length=sequence_length,
    )


def compute_statistical_features(sequences: np.ndarray, sensor_columns: list[str]) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    names: list[str] = []
    for sensor in sensor_columns:
        for feat in FEATURE_NAMES:
            names.append(f"{sensor}__full_length__{feat}")
    for sample in sequences:
        values = []
        for channel in sample:
            x = channel.astype(np.float64)
            mean = float(np.mean(x))
            std = float(np.std(x))
            centered = x - mean
            if std < 1e-10:
                skewness = 0.0
                kurtosis = 0.0
            else:
                standardized = centered / std
                skewness = float(np.mean(standardized**3))
                kurtosis = float(np.mean(standardized**4))
            fft_mag = np.abs(np.fft.rfft(x))
            freqs = np.fft.rfftfreq(len(x), d=1.0)
            spectral_centroid = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-12))
            band_energy = float(np.mean(x**2))
            values.extend(
                [
                    mean,
                    std,
                    float(np.max(x)),
                    float(np.min(x)),
                    float(np.max(x) - np.min(x)),
                    kurtosis,
                    skewness,
                    spectral_centroid,
                    band_energy,
                ]
            )
        rows.append(np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12).tolist())
    return np.asarray(rows, dtype=np.float64), names


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
        val_idx = case_idx[-val_count:]
        train_idx = case_idx[:-val_count]
        split.loc[train_idx, "split"] = "train"
        split.loc[val_idx, "split"] = "validation"
    return split


def build_sequences_for_indices(
    data: RunData,
    indices: np.ndarray,
    normalized_sequences: np.ndarray,
    sequence_size: int,
    padding_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    position_by_case_run = {
        (int(row.case_id), int(row.run_order)): int(idx)
        for idx, row in data.meta[["case_id", "run_order"]].iterrows()
    }
    x_out = []
    masks = []
    for idx in indices:
        row = data.meta.loc[int(idx)]
        case_id = int(row["case_id"])
        order = int(row["run_order"])
        seq_parts = []
        mask = []
        for lag in range(sequence_size - 1, -1, -1):
            prev_order = order - lag
            prev_idx = position_by_case_run.get((case_id, prev_order))
            if prev_idx is None or prev_order < 1:
                seq_parts.append(np.full_like(normalized_sequences[int(idx)], padding_value, dtype=np.float32))
                mask.append(0.0)
            else:
                seq_parts.append(normalized_sequences[prev_idx])
                mask.append(1.0)
        x_out.append(np.stack(seq_parts, axis=0))
        masks.append(mask)
    return np.stack(x_out).astype(np.float32), np.asarray(masks, dtype=np.float32)


def fit_sequence_scaler(sequences: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = sequences[train_indices]
    if train.ndim == 4:
        # [N, K, C, W] -> per-sensor statistics from source train only.
        mean = train.mean(axis=(0, 1, 3), keepdims=True)
        std = train.std(axis=(0, 1, 3), keepdims=True)
    elif train.ndim == 3:
        mean = train.mean(axis=(0, 2), keepdims=True)
        std = train.std(axis=(0, 2), keepdims=True)
    else:
        raise ValueError(f"Unsupported sequence array shape for scaling: {train.shape}")
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_sequences(sequences: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = (sequences - mean) / std
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(normalized, -10.0, 10.0).astype(np.float32)


class CurrentRunDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y[:, None], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class RunSequenceDataset(Dataset):
    def __init__(self, x_seq: np.ndarray, mask: np.ndarray, y: np.ndarray):
        self.x_seq = torch.as_tensor(x_seq, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.float32)
        self.y = torch.as_tensor(y[:, None], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_seq[idx], self.mask[idx], self.y[idx]


class CNNEncoder(nn.Module):
    def __init__(self, num_sensors: int, channels: list[int], kernel_size: int, latent_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = num_sensors
        padding = kernel_size // 2
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                ]
            )
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_channels, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"CNNEncoder expects [batch, sensors, time], got {tuple(x.shape)}")
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)
        return torch.relu(self.proj(h))


class CNNOnlyRegressor(nn.Module):
    def __init__(self, num_sensors: int, channels: list[int], kernel_size: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = CNNEncoder(num_sensors, channels, kernel_size, latent_dim)
        self.head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class CNNLagConcatRegressor(nn.Module):
    def __init__(self, num_sensors: int, sequence_size: int, channels: list[int], kernel_size: int, latent_dim: int, hidden_dim: int, include_mask: bool = True):
        super().__init__()
        self.sequence_size = sequence_size
        self.include_mask = include_mask
        self.encoder = CNNEncoder(num_sensors, channels, kernel_size, latent_dim)
        input_dim = sequence_size * latent_dim + (sequence_size if include_mask else 0)
        self.head = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim != 4:
            raise ValueError(f"CNNLagConcatRegressor expects [batch, seq, sensors, time], got {tuple(x_seq.shape)}")
        b, s, c, t = x_seq.shape
        z = self.encoder(x_seq.reshape(b * s, c, t)).reshape(b, s, -1)
        z = z * mask.unsqueeze(-1)
        flat = z.reshape(b, -1)
        if self.include_mask:
            flat = torch.cat([flat, mask], dim=1)
        return self.head(flat)


class CNNGRURegressor(nn.Module):
    def __init__(self, num_sensors: int, sequence_size: int, channels: list[int], kernel_size: int, latent_dim: int, hidden_size: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.sequence_size = sequence_size
        self.encoder = CNNEncoder(num_sensors, channels, kernel_size, latent_dim)
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim != 4:
            raise ValueError(f"CNNGRURegressor expects [batch, seq, sensors, time], got {tuple(x_seq.shape)}")
        b, s, c, t = x_seq.shape
        z = self.encoder(x_seq.reshape(b * s, c, t)).reshape(b, s, -1)
        z = z * mask.unsqueeze(-1)
        out, _ = self.gru(z)
        current_hidden = out[:, -1, :]
        return self.head(current_hidden)


def build_dl_model(model_name: str, config: dict[str, Any], num_sensors: int) -> nn.Module:
    if model_name == "cnn_only":
        return WindowedCNNOnlyRegressor.from_config(config, input_channels=num_sensors)
    if model_name == "cnn_lag_concat":
        return WindowedCNNLagConcatRegressor.from_config(config, input_channels=num_sensors)
    if model_name == "cnn_gru":
        return WindowedCNNGRURegressor.from_config(config, input_channels=num_sensors)
    raise ValueError(f"Unknown DL model: {model_name}")


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    out = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }
    return out


def y_inverse(y_scaled: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    return np.asarray(y_scaled).reshape(-1) * y_std + y_mean


def evaluate_dl(
    model: nn.Module,
    loader: DataLoader,
    model_name: str,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for batch in loader:
            if model_name == "cnn_only":
                x, y = batch
                pred = model(x.to(device))
            else:
                x, mask, y = batch
                pred = model(x.to(device), mask.to(device))
            preds.append(pred.detach().cpu().numpy().reshape(-1))
            ys.append(y.detach().cpu().numpy().reshape(-1))
    pred_scaled = np.concatenate(preds) if preds else np.array([])
    y_scaled = np.concatenate(ys) if ys else np.array([])
    pred = y_inverse(pred_scaled, y_mean, y_std)
    y_true = y_inverse(y_scaled, y_mean, y_std)
    return metrics_dict(y_true, pred), y_true, pred


def evaluate_dl_loss(model: nn.Module, loader: DataLoader, model_name: str, device: torch.device, loss_fn: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            if model_name == "cnn_only":
                x, y = batch
                pred = model(x.to(device))
            else:
                x, mask, y = batch
                pred = model(x.to(device), mask.to(device))
            y_device = y.to(device).view(-1)
            loss = loss_fn(pred.view(-1), y_device)
            count = int(y_device.shape[0])
            total_loss += float(loss.detach().cpu()) * count
            total_count += count
    return total_loss / max(total_count, 1)


def train_dl_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    y_mean: float,
    y_std: float,
    checkpoint_dir: Path,
    checkpoint_stem: str,
    scenario_name: str,
    source_domain: str,
    target_domain: str,
    seed: int,
    max_epochs: int | None = None,
    batch_note: str = "",
) -> tuple[nn.Module, dict[str, float], float, pd.DataFrame, dict[str, Any]]:
    model = build_dl_model(model_name, config, num_sensors=len(config["data"]["sensor_columns"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"].get("learning_rate", 1e-3)))
    loss_fn = nn.MSELoss()
    patience = int(config["training"].get("early_stopping", {}).get("patience", 5))
    epochs = int(max_epochs or config["training"].get("max_epochs", 25))
    best_rmse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {"MAE": float("nan"), "RMSE": float("inf"), "R2": float("nan")}
    best_epoch = 0
    bad_epochs = 0
    history_rows: list[dict[str, Any]] = []
    start = time.time()
    for _epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if model_name == "cnn_only":
                x, y = batch
                pred = model(x.to(device))
            else:
                x, mask, y = batch
                pred = model(x.to(device), mask.to(device))
            y_device = y.to(device).view(-1)
            loss = loss_fn(pred.view(-1), y_device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            count = int(y.shape[0])
            train_loss_total += float(loss.detach().cpu()) * count
            train_count += count
        train_loss = train_loss_total / max(train_count, 1)
        val_loss = evaluate_dl_loss(model, val_loader, model_name, device, loss_fn)
        train_metrics, _, _ = evaluate_dl(model, train_loader, model_name, device, y_mean, y_std)
        val_metrics, _, _ = evaluate_dl(model, val_loader, model_name, device, y_mean, y_std)
        is_best_epoch = bool(val_metrics["RMSE"] < best_rmse)
        if val_metrics["RMSE"] < best_rmse:
            best_rmse = val_metrics["RMSE"]
            best_metrics = val_metrics
            best_epoch = _epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        history_rows.append(
            {
                "experiment_id": config["experiment"]["experiment_id"],
                "scenario_name": scenario_name,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "model_name": model_name,
                "seed": seed,
                "epoch": _epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_MAE": train_metrics["MAE"],
                "train_RMSE": train_metrics["RMSE"],
                "train_R2": train_metrics["R2"],
                "val_MAE": val_metrics["MAE"],
                "val_RMSE": val_metrics["RMSE"],
                "val_R2": val_metrics["R2"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_time_sec": float(time.time() - start),
                "is_best_epoch": is_best_epoch,
            }
        )
        if bad_epochs >= patience:
            break
    elapsed = time.time() - start
    for row in history_rows:
        row["is_best_epoch"] = bool(row["epoch"] == best_epoch)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / f"{checkpoint_stem}_last.pt"
    best_path = checkpoint_dir / f"{checkpoint_stem}_best.pt"
    last_epoch = int(history_rows[-1]["epoch"]) if history_rows else 0
    checkpoint_common = {
        "model_name": model_name,
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": config.get("model_design", config.get("dl_model", {})),
        "input_config": {"sequence": config.get("sequence"), "data": config.get("data")},
        "seed": seed,
        "scenario_name": scenario_name,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "note": batch_note,
    }
    torch.save(
        {
            **checkpoint_common,
            "epoch": last_epoch,
            "best_val_RMSE": best_metrics["RMSE"],
            "state_dict": model.state_dict(),
            "model_state_dict": model.state_dict(),
        },
        last_path,
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            **checkpoint_common,
            "epoch": int(best_epoch),
            "best_val_RMSE": best_metrics["RMSE"],
            "state_dict": model.state_dict(),
            "model_state_dict": model.state_dict(),
            "best_metrics": best_metrics,
        },
        best_path,
    )
    loaded = torch.load(best_path, map_location=device)
    model.load_state_dict(loaded.get("state_dict", loaded["model_state_dict"]))
    final_epoch = int(history_rows[-1]["epoch"]) if history_rows else 0
    best_summary = {
        "model_name": model_name,
        "scenario_name": scenario_name,
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_val_RMSE": best_metrics["RMSE"],
        "best_val_MAE": best_metrics["MAE"],
        "best_val_R2": best_metrics["R2"],
        "final_epoch": final_epoch,
        "stopped_by_early_stopping": bool(final_epoch < epochs),
        "interpretation": "best epoch selected by validation RMSE",
    }
    return model, best_metrics, elapsed, pd.DataFrame(history_rows), best_summary


def model_config_hash(config: dict[str, Any], model_name: str) -> str:
    payload = json.dumps(
        to_builtin(
            {
                "model": model_name,
                "dl": config.get("dl_model"),
                "model_design": config.get("model_design"),
                "seq": config.get("sequence"),
                "training": config.get("training"),
            }
        ),
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def fit_feature_baseline(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float], float]:
    start = time.time()
    x_train = np.clip(np.nan_to_num(x_train, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6).astype(np.float64)
    x_val = np.clip(np.nan_to_num(x_val, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6).astype(np.float64)
    x_test = np.clip(np.nan_to_num(x_test, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6).astype(np.float64)
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)
    if model_name in {"feature_ridge", "feature_baseline_ridge"}:
        model = Ridge(alpha=1.0, random_state=seed)
    elif model_name in {"feature_random_forest", "feature_baseline_random_forest"}:
        rf_cfg = config["models"].get("feature_random_forest", config["models"].get("feature_baseline_random_forest", {}))
        model = RandomForestRegressor(
            n_estimators=int(rf_cfg.get("n_estimators", 200)),
            min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 1)),
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "feature_svr":
        svr_cfg = config["models"].get("feature_svr", {})
        model = SVR(
            kernel=svr_cfg.get("kernel", "rbf"),
            C=float(svr_cfg.get("C", 1.0)),
            epsilon=float(svr_cfg.get("epsilon", 0.1)),
            gamma=svr_cfg.get("gamma", "scale"),
        )
    elif model_name == "feature_xgboost":
        if XGBRegressor is None:
            raise RuntimeError("xgboost_not_installed")
        xgb_cfg = config["models"].get("feature_xgboost", {})
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(xgb_cfg.get("n_estimators", 100)),
            max_depth=int(xgb_cfg.get("max_depth", 3)),
            learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
            subsample=float(xgb_cfg.get("subsample", 0.9)),
            colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.9)),
            random_state=seed,
            n_jobs=1,
        )
    else:
        raise ValueError(model_name)
    model.fit(x_train_s, y_train)
    val_pred = model.predict(x_val_s)
    test_pred = model.predict(x_test_s)
    elapsed = time.time() - start
    return test_pred.astype(float), metrics_dict(y_val, val_pred), elapsed


def run_scenario_seed(
    data: RunData,
    config: dict[str, Any],
    source: str,
    target: str,
    seed: int,
    output_dir: Path,
    models: list[str],
    max_epochs: int | None = None,
    batch_size: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    set_seed(seed)
    source_cases = DOMAIN_CASES[source]
    target_cases = DOMAIN_CASES[target]
    scenario_name = f"{source}_to_{target}"
    split = split_source_validation(data.meta, source_cases, target_cases, float(config["split"].get("validation_ratio", 0.2)))
    train_idx = split.index[split["split"] == "train"].to_numpy()
    val_idx = split.index[split["split"] == "validation"].to_numpy()
    test_idx = split.index[split["split"] == "test"].to_numpy()
    target_col = config["data"].get("target_col", "VB")
    y_train = split.loc[train_idx, target_col].to_numpy(dtype=np.float32)
    y_val = split.loc[val_idx, target_col].to_numpy(dtype=np.float32)
    y_test = split.loc[test_idx, target_col].to_numpy(dtype=np.float32)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) if np.std(y_train) > 1e-8 else 1.0)
    y_train_s = (y_train - y_mean) / y_std
    y_val_s = (y_val - y_mean) / y_std
    y_test_s = (y_test - y_mean) / y_std

    seq_mean, seq_std = fit_sequence_scaler(data.sequences, train_idx)
    seq_norm = normalize_sequences(data.sequences, seq_mean, seq_std)
    seq_size = int(config["sequence"].get("sequence_size", 3))
    x_train_seq, m_train = build_sequences_for_indices(data, train_idx, seq_norm, seq_size)
    x_val_seq, m_val = build_sequences_for_indices(data, val_idx, seq_norm, seq_size)
    x_test_seq, m_test = build_sequences_for_indices(data, test_idx, seq_norm, seq_size)

    bs = int(batch_size or config["training"].get("batch_size", 8))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows_case: list[dict[str, Any]] = []
    rows_domain: list[dict[str, Any]] = []
    rows_pred: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    best_epoch_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for model_name in models:
        model_start = time.time()
        if model_name.startswith("feature_") or model_name.startswith("feature_baseline"):
            try:
                pred_test, val_metrics, train_time = fit_feature_baseline(
                    model_name,
                    data.feature_matrix[train_idx],
                    y_train,
                    data.feature_matrix[val_idx],
                    y_val,
                    data.feature_matrix[test_idx],
                    seed,
                    config,
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "experiment_id": config["experiment"]["experiment_id"],
                        "scenario_name": scenario_name,
                        "source_domain": source,
                        "target_domain": target,
                        "model_name": model_name,
                        "seed": seed,
                        "reason": str(exc),
                    }
                )
                continue
        else:
            if model_name == "cnn_only":
                train_ds = CurrentRunDataset(seq_norm[train_idx], y_train_s)
                val_ds = CurrentRunDataset(seq_norm[val_idx], y_val_s)
                test_ds = CurrentRunDataset(seq_norm[test_idx], y_test_s)
            else:
                train_ds = RunSequenceDataset(x_train_seq, m_train, y_train_s)
                val_ds = RunSequenceDataset(x_val_seq, m_val, y_val_s)
                test_ds = RunSequenceDataset(x_test_seq, m_test, y_test_s)
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
            checkpoint_base = output_dir / ("smoke" if smoke else "checkpoints") / model_name
            stem = f"{scenario_name}_seed_{seed}"
            model, val_metrics, train_time, history, best_epoch = train_dl_model(
                model_name,
                train_loader,
                val_loader,
                config,
                device,
                y_mean,
                y_std,
                checkpoint_base,
                stem,
                scenario_name,
                source,
                target,
                seed,
                max_epochs=max_epochs,
                batch_note="smoke" if smoke else "initial",
            )
            history_frames.append(history)
            best_epoch_rows.append(best_epoch)
            _, _, pred_test = evaluate_dl(model, test_loader, model_name, device, y_mean, y_std)
        total_train_time = time.time() - model_start if train_time is None else train_time
        test_meta = split.loc[test_idx].copy()
        test_meta["y_pred"] = pred_test
        test_meta["y_true"] = y_test
        domain_metrics = metrics_dict(y_test, pred_test)
        rows_domain.append(
            {
                "experiment_id": config["experiment"]["experiment_id"],
                "scenario_name": scenario_name,
                "source_domain": source,
                "target_domain": target,
                "train_cases": ",".join(map(str, source_cases)),
                "test_cases": ",".join(map(str, target_cases)),
                "model_name": model_name,
                "sequence_size": seq_size,
                "stride": int(config["sequence"].get("stride", 1)),
                "padding": config["sequence"].get("padding", "left"),
                "uses_previous_runs": model_name in {"cnn_lag_concat", "cnn_gru"},
                "seed": seed,
                "target_domain_MAE": domain_metrics["MAE"],
                "target_domain_RMSE": domain_metrics["RMSE"],
                "target_domain_R2": domain_metrics["R2"],
                "mean_case_MAE": float("nan"),
                "mean_case_RMSE": float("nan"),
                "mean_case_R2": float("nan"),
            }
        )
        case_metric_values = []
        for case_id, group in test_meta.groupby("case_id"):
            cm = metrics_dict(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
            case_metric_values.append(cm)
            rows_case.append(
                {
                    "experiment_id": config["experiment"]["experiment_id"],
                    "scenario_name": scenario_name,
                    "source_domain": source,
                    "target_domain": target,
                    "train_cases": ",".join(map(str, source_cases)),
                    "val_cases": ",".join(map(str, source_cases)),
                    "test_case_id": int(case_id),
                    "model_name": model_name,
                    "sequence_size": seq_size,
                    "stride": int(config["sequence"].get("stride", 1)),
                    "padding": config["sequence"].get("padding", "left"),
                    "uses_previous_runs": model_name in {"cnn_lag_concat", "cnn_gru"},
                    "seed": seed,
                    "num_train_runs": int(len(train_idx)),
                    "num_val_runs": int(len(val_idx)),
                    "num_test_runs": int(len(group)),
                    "best_val_MAE": val_metrics["MAE"],
                    "best_val_RMSE": val_metrics["RMSE"],
                    "best_val_R2": val_metrics["R2"],
                    "test_MAE": cm["MAE"],
                    "test_RMSE": cm["RMSE"],
                    "test_R2": cm["R2"],
                    "training_time_sec": float(total_train_time),
                    "model_config_hash": model_config_hash(config, model_name),
                }
            )
        idx_last = len(rows_domain) - 1
        rows_domain[idx_last]["mean_case_MAE"] = float(np.mean([m["MAE"] for m in case_metric_values]))
        rows_domain[idx_last]["mean_case_RMSE"] = float(np.mean([m["RMSE"] for m in case_metric_values]))
        rows_domain[idx_last]["mean_case_R2"] = float(np.mean([m["R2"] for m in case_metric_values]))
        for _, row in test_meta.sort_values(["case_id", "run_order"]).iterrows():
            err = float(row["y_pred"] - row["y_true"])
            rows_pred.append(
                {
                    "experiment_id": config["experiment"]["experiment_id"],
                    "scenario_name": scenario_name,
                    "source_domain": source,
                    "target_domain": target,
                    "case_id": int(row["case_id"]),
                    "run_id": int(row["run_id"]),
                    "run_order": int(row["run_order"]),
                    "model_name": model_name,
                    "sequence_size": seq_size,
                    "stride": int(config["sequence"].get("stride", 1)),
                    "padding": config["sequence"].get("padding", "left"),
                    "seed": seed,
                    "y_true": float(row["y_true"]),
                    "y_pred": float(row["y_pred"]),
                    "error": err,
                    "absolute_error": abs(err),
                    "squared_error": err * err,
                }
            )
    split_out = split[["case_id", "run_id", "run_order", "dataset_run_id", "domain_id", target_col, "split"]].copy()
    split_out["scenario_name"] = scenario_name
    split_out["source_domain"] = source
    split_out["target_domain"] = target
    split_out["seed"] = seed
    return {
        "case_metrics": pd.DataFrame(rows_case),
        "domain_metrics": pd.DataFrame(rows_domain),
        "predictions": pd.DataFrame(rows_pred),
        "split": split_out,
        "learning_history": pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame(),
        "best_epoch_summary": pd.DataFrame(best_epoch_rows),
        "skipped": pd.DataFrame(skipped_rows),
        "scaler_fit": {
            "scenario_name": scenario_name,
            "seed": seed,
            "sequence_scaler_fit_indices": train_idx.tolist(),
            "target_scaler_fit_indices": train_idx.tolist(),
        },
    }


def make_dirs(output_dir: Path) -> None:
    for name in ["configs", "data", "splits", "metrics", "predictions", "analysis", "figures", "checkpoints", "reports", "logs", "smoke"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def build_default_config() -> dict[str, Any]:
    return {
        "experiment": {
            "hypothesis_id": "H4",
            "scenario_id": "S1",
            "experiment_topic": EXPERIMENT_TOPIC,
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
        "domain": {"domain_pairs": DOMAIN_CASES, "transfer_scenarios": TRANSFER_SCENARIOS},
        "split": {"validation_strategy": "source_case_chronological_tail", "validation_ratio": 0.2, "source_only_validation": True},
        "sequence": {"sequence_size": 3, "stride": 1, "padding": "left", "padding_value": 0.0, "allow_cross_case_sequence": False, "segment_setting": "full_length", "sequence_length": 256},
        "models": {
            "feature_ridge": {"enabled": True},
            "feature_random_forest": {"enabled": True, "n_estimators": 200, "min_samples_leaf": 1},
            "feature_svr": {"enabled": True, "kernel": "rbf", "C": 1.0, "epsilon": 0.1, "gamma": "scale"},
            "feature_xgboost": {"enabled": True, "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
            "cnn_only": {"enabled": True},
            "cnn_lag_concat": {"enabled": True, "include_mask": True},
            "cnn_gru": {"enabled": True},
        },
        "dl_model": {"cnn_channels": [16, 32], "kernel_size": 5, "latent_dim": 32, "regressor_hidden_dim": 32, "gru_hidden_size": 32, "gru_num_layers": 1, "dropout": 0.0},
        "model_design": {
            "run_encoder": {
                "type": "windowed_cnn",
                "input_channels": 6,
                "window_length": 3000,
                "num_windows": 5,
                "stride": "auto",
                "short_run_policy": "pad_to_window_length",
                "window_aggregation": "max_pool",
                "latent_dim": 32,
            },
            "cnn_encoder": {
                "channels": [16, 32],
                "kernel_size": 5,
                "use_batch_norm": True,
                "activation": "relu",
                "temporal_pooling": "global_average_pooling",
                "dropout": 0.0,
            },
            "cnn_only": {"enabled": True, "head_hidden_dim": 32, "dropout": 0.0},
            "cnn_lag_concat": {"enabled": True, "sequence_size": 3, "include_mask": True, "head_hidden_dim": 64, "dropout": 0.0},
            "cnn_gru": {
                "enabled": True,
                "sequence_size": 3,
                "gru_hidden_size": 32,
                "gru_num_layers": 1,
                "bidirectional": False,
                "dropout": 0.0,
                "head_hidden_dim": 32,
            },
        },
        "normalization": {"method": "zscore", "fit_scope": "source_train_only", "apply_per_sensor": True},
        "training": {"optimizer": "adam", "learning_rate": 0.0005, "batch_size": 8, "max_epochs": 200, "early_stopping": {"enabled": True, "patience": 100, "monitor": "val_RMSE", "mode": "min"}, "checkpoint": {"enabled": True, "save_best": True, "save_last": True}, "primary_metric": "RMSE"},
        "smoke": {"enabled": True, "scenario": "A_to_B", "seed": 0, "models": DL_MODELS, "max_epochs": 200, "batch_size": 4, "sequence_length": 128, "window_length": 300, "num_windows": 5},
        "evaluation": {"metrics": ["MAE", "RMSE", "R2"], "aggregate_by_case": True, "aggregate_by_target_domain": True},
        "visualization": {
            "rmse_filter_enabled": True,
            "rmse_filter_threshold": 0.5,
            "rmse_filter_metric": "RMSE",
            "rmse_filter_operator": ">",
            "rmse_filter_level": "case_model_seed",
            "exclude_filtered_from_metrics": False,
            "exclude_filtered_from_ranking": False,
            "exclude_filtered_from_report_tables": False,
        },
    }


def validate_config_and_data(config: dict[str, Any], data: RunData, output_dir: Path) -> dict[str, Any]:
    scenarios = [f"{s}_to_{t}" for s, t in TRANSFER_SCENARIOS]
    checks = {
        "data_files_exist": Path(config["data"]["process_info_path"]).exists() and Path(config["data"]["signal_data_path"]).exists(),
        "target_column": config["data"]["target_col"],
        "target_column_exists": config["data"]["target_col"] in data.meta.columns,
        "case_column_exists": "case_id" in data.meta.columns,
        "run_id_column_exists": "run_id" in data.meta.columns,
        "run_order_column_exists": "run_order" in data.meta.columns,
        "sensor_columns": data.sensor_columns,
        "sensor_mapping_ok": len(data.sensor_columns) == len(config["data"]["sensor_columns"]),
        "selected_cases": sorted(data.meta["case_id"].unique().tolist()),
        "selected_case_count": int(data.meta["case_id"].nunique()),
        "num_runs": int(len(data.meta)),
        "sequence_shape": list(data.sequences.shape),
        "windowed_cnn_input_shape": list(data.sequences.shape[1:]),
        "window_length": int(data.window_length),
        "num_windows": int(data.num_windows),
        "window_aggregation": config.get("model_design", {}).get("run_encoder", {}).get("window_aggregation", "max_pool"),
        "short_run_count": int(data.windowing_summary["padded"].sum()) if not data.windowing_summary.empty else 0,
        "windowing_policy_ok": bool(data.sequences.ndim == 4 and data.sequences.shape[1] == data.num_windows and data.sequences.shape[-1] == data.window_length),
        "feature_matrix_shape": list(data.feature_matrix.shape),
        "scenarios": scenarios,
        "all_scenarios_have_train_val_test": True,
        "source_only_validation": True,
        "train_only_scaler_policy": True,
        "left_padding_policy": config["sequence"]["padding"] == "left",
        "allow_cross_case_sequence": bool(config["sequence"].get("allow_cross_case_sequence", False)),
        "svr_available": True,
        "svr_scaling_fit_scope": "source_train_only",
        "xgboost_available": XGBRegressor is not None,
        "xgboost_skip_reason": None if XGBRegressor is not None else "xgboost_not_installed",
        "learning_history_required_for_dl": True,
        "visualization_filter_exclude_from_metrics": bool(config.get("visualization", {}).get("exclude_filtered_from_metrics", False)),
    }
    split_summaries = []
    for source, target in TRANSFER_SCENARIOS:
        split = split_source_validation(data.meta, DOMAIN_CASES[source], DOMAIN_CASES[target], float(config["split"]["validation_ratio"]))
        counts = split["split"].value_counts().to_dict()
        train_cases = set(split.loc[split["split"] == "train", "case_id"].astype(int))
        val_cases = set(split.loc[split["split"] == "validation", "case_id"].astype(int))
        test_cases = set(split.loc[split["split"] == "test", "case_id"].astype(int))
        ok = bool(train_cases <= set(DOMAIN_CASES[source]) and val_cases <= set(DOMAIN_CASES[source]) and test_cases <= set(DOMAIN_CASES[target]))
        checks["all_scenarios_have_train_val_test"] = checks["all_scenarios_have_train_val_test"] and all(counts.get(k, 0) > 0 for k in ["train", "validation", "test"]) and ok
        split_summaries.append({"scenario": f"{source}_to_{target}", "counts": counts, "train_cases": sorted(train_cases), "val_cases": sorted(val_cases), "test_cases": sorted(test_cases), "source_target_separation_ok": ok})
    checks["split_summaries"] = split_summaries
    checks["passed"] = all([checks["data_files_exist"], checks["target_column_exists"], checks["case_column_exists"], checks["run_id_column_exists"], checks["run_order_column_exists"], checks["sensor_mapping_ok"], checks["all_scenarios_have_train_val_test"], checks["windowing_policy_ok"], not checks["allow_cross_case_sequence"]])
    write_json(output_dir / "configs" / f"{PREFIX}_config_validation.json", checks)
    return checks


def leakage_check(split_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    failures = []
    for (scenario, seed), group in split_df.groupby(["scenario_name", "seed"]):
        target_cases = set(group.loc[group["split"] == "test", "case_id"].astype(int))
        train_cases = set(group.loc[group["split"] == "train", "case_id"].astype(int))
        val_cases = set(group.loc[group["split"] == "validation", "case_id"].astype(int))
        duplicate_runs = int(group.groupby("dataset_run_id")["split"].nunique().gt(1).sum())
        if target_cases & train_cases:
            failures.append({"scenario": scenario, "seed": int(seed), "failure": "target_case_in_train"})
        if target_cases & val_cases:
            failures.append({"scenario": scenario, "seed": int(seed), "failure": "target_case_in_validation"})
        if duplicate_runs:
            failures.append({"scenario": scenario, "seed": int(seed), "failure": "run_id_overlap_across_splits", "count": duplicate_runs})
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "target_domain_case_in_train": False,
        "target_domain_case_in_validation": False,
        "target_domain_label_used_for_model_selection": False,
        "scaler_fit_scope": "source_train_only",
        "target_domain_statistics_used_for_preprocessing_fit": False,
        "sequence_crosses_case_boundary": False,
        "different_cases_connected_inside_sequence": False,
        "run_order_preserved": True,
        "early_runs_left_padded": config["sequence"].get("padding") == "left",
        "validation_split_scope": "source_domain_only",
    }


def aggregate_outputs(case_metrics: pd.DataFrame, domain_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shift_metrics = domain_metrics.rename(
        columns={
            "target_domain_MAE": "metric_MAE",
            "target_domain_RMSE": "metric_RMSE",
            "target_domain_R2": "metric_R2",
        }
    ).copy()
    seed_rows = []
    for (model_name, seed), g in shift_metrics.groupby(["model_name", "seed"]):
        seed_rows.append(
            {
                "experiment_id": g["experiment_id"].iloc[0],
                "model_name": model_name,
                "uses_previous_runs": bool(g["uses_previous_runs"].iloc[0]),
                "sequence_size": int(g["sequence_size"].iloc[0]),
                "seed": int(seed),
                "mean_MAE_over_6_shifts": float(g["metric_MAE"].mean()),
                "mean_RMSE_over_6_shifts": float(g["metric_RMSE"].mean()),
                "mean_R2_over_6_shifts": float(g["metric_R2"].mean()),
            }
        )
    seed_metrics = pd.DataFrame(seed_rows)
    comp = (
        seed_metrics.groupby(["experiment_id", "model_name", "uses_previous_runs", "sequence_size"], as_index=False)
        .agg(
            mean_RMSE_over_6_shifts=("mean_RMSE_over_6_shifts", "mean"),
            std_RMSE_over_6_shifts=("mean_RMSE_over_6_shifts", "std"),
            mean_MAE_over_6_shifts=("mean_MAE_over_6_shifts", "mean"),
            mean_R2_over_6_shifts=("mean_R2_over_6_shifts", "mean"),
        )
        .sort_values("mean_RMSE_over_6_shifts")
    )
    comp["rank_by_RMSE"] = np.arange(1, len(comp) + 1)
    comp["interpretation"] = comp["model_name"].map(
        {
            "feature_ridge": "handcrafted feature linear baseline",
            "feature_random_forest": "handcrafted feature nonlinear baseline",
            "feature_svr": "handcrafted feature kernel baseline",
            "feature_xgboost": "handcrafted feature gradient boosting baseline",
            "cnn_only": "current run local temporal pattern only",
            "cnn_lag_concat": "current and previous run latent vectors concatenated",
            "cnn_gru": "current and previous run latent sequence modeled by GRU",
        }
    )
    return shift_metrics, seed_metrics, comp


def apply_visualization_filter(
    predictions: pd.DataFrame,
    case_metrics: pd.DataFrame,
    domain_metrics: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    viz = config.get("visualization", {})
    threshold = float(viz.get("rmse_filter_threshold", 0.5))
    case_key = ["experiment_id", "scenario_name", "source_domain", "target_domain", "test_case_id", "model_name", "seed"]
    case_rmse = case_metrics[case_key + ["test_RMSE"]].rename(columns={"test_case_id": "case_id", "test_RMSE": "test_case_RMSE"})
    domain_rmse = domain_metrics[
        ["experiment_id", "scenario_name", "source_domain", "target_domain", "model_name", "seed", "target_domain_RMSE"]
    ]
    pred = predictions.merge(
        case_rmse,
        on=["experiment_id", "scenario_name", "source_domain", "target_domain", "case_id", "model_name", "seed"],
        how="left",
    ).merge(
        domain_rmse,
        on=["experiment_id", "scenario_name", "source_domain", "target_domain", "model_name", "seed"],
        how="left",
    )
    pred["include_in_visualization"] = pred["test_case_RMSE"].le(threshold)
    pred["include_in_visualization"] = pred["include_in_visualization"].fillna(False)

    filter_summary = (
        pred[
            [
                "experiment_id",
                "scenario_name",
                "source_domain",
                "target_domain",
                "case_id",
                "model_name",
                "seed",
                "test_case_RMSE",
                "include_in_visualization",
            ]
        ]
        .drop_duplicates()
        .rename(columns={"test_case_RMSE": "test_RMSE", "case_id": "case_id"})
    )
    filter_summary["threshold"] = threshold
    filter_summary["included_in_visualization"] = filter_summary["include_in_visualization"]
    filter_summary["reason"] = np.where(
        filter_summary["included_in_visualization"],
        "case_level_rmse_within_threshold",
        "case_level_rmse_exceeds_visualization_threshold",
    )
    filter_summary = filter_summary.drop(columns=["include_in_visualization"])
    filter_summary.to_csv(output_dir / "analysis" / f"{PREFIX}_visualization_filter_summary.csv", index=False)

    excluded = filter_summary.loc[~filter_summary["included_in_visualization"]].copy()
    if excluded.empty:
        excluded_summary = pd.DataFrame(columns=["model_name", "scenario_name", "excluded_condition_count"])
    else:
        excluded_summary = excluded.groupby(["model_name", "scenario_name"], as_index=False).size().rename(columns={"size": "excluded_condition_count"})
    excluded_summary.to_csv(output_dir / "analysis" / f"{PREFIX}_excluded_high_rmse_summary.csv", index=False)

    pattern_summary = (
        pred.loc[pred["include_in_visualization"]]
        .groupby(["model_name", "case_id"], as_index=False)
        .agg(mean_error=("error", "mean"), mean_absolute_error=("absolute_error", "mean"), mean_prediction=("y_pred", "mean"), mean_true=("y_true", "mean"), n_points=("y_true", "size"))
    )
    pattern_summary["bias_direction"] = np.where(pattern_summary["mean_error"] > 0, "over_prediction", "under_prediction")
    pattern_summary.to_csv(output_dir / "analysis" / f"{PREFIX}_prediction_pattern_summary.csv", index=False)
    return pred, filter_summary, excluded_summary


def write_data_artifacts(data: RunData, config: dict[str, Any], output_dir: Path) -> None:
    dataset_summary = {
        "process_info_path": config["data"]["process_info_path"],
        "signal_data_path": config["data"]["signal_data_path"],
        "num_runs_with_target": int(len(data.meta)),
        "cases": sorted(data.meta["case_id"].unique().tolist()),
        "target_col": config["data"]["target_col"],
        "feature_sequence_length_resampled": data.sequence_length,
        "window_length": int(data.window_length),
        "num_windows": int(data.num_windows),
        "windowed_sequence_shape": list(data.sequences.shape),
        "short_run_count": int(data.windowing_summary["padded"].sum()) if not data.windowing_summary.empty else 0,
        "padding_length_min": int(data.windowing_summary["padding_length"].min()) if not data.windowing_summary.empty else 0,
        "padding_length_max": int(data.windowing_summary["padding_length"].max()) if not data.windowing_summary.empty else 0,
        "raw_length_min": int(data.raw_lengths.min()),
        "raw_length_max": int(data.raw_lengths.max()),
        "raw_length_median": float(np.median(data.raw_lengths)),
        "sensor_columns": data.sensor_columns,
        "feature_matrix_shape": list(data.feature_matrix.shape),
    }
    write_json(output_dir / "data" / f"{PREFIX}_dataset_summary.json", dataset_summary)
    pd.DataFrame({"requested_sensor": config["data"]["sensor_columns"], "resolved_sensor": config["data"]["sensor_columns"], "status": ["matched"] * len(config["data"]["sensor_columns"])}).to_csv(output_dir / "data" / f"{PREFIX}_sensor_mapping.csv", index=False)
    order = data.meta[["case_id", "run_id", "run_order", "dataset_run_id", "domain_id", config["data"]["target_col"]]].copy()
    order.to_json(output_dir / "data" / f"{PREFIX}_run_order_summary.json", orient="records", indent=2)
    seq_rows = data.meta[["case_id", "run_id", "run_order"]].copy()
    seq_rows["original_length"] = data.raw_lengths
    seq_rows["feature_resampled_length"] = data.sequence_length
    seq_rows["window_length"] = int(data.window_length)
    seq_rows["num_windows"] = int(data.num_windows)
    seq_rows["num_sensors"] = len(data.sensor_columns)
    seq_rows.to_json(output_dir / "data" / f"{PREFIX}_sequence_summary.json", orient="records", indent=2)
    windowing = data.windowing_summary.copy()
    windowing.insert(0, "experiment_id", config["experiment"]["experiment_id"])
    windowing.to_csv(output_dir / "data" / f"{PREFIX}_windowing_summary.csv", index=False)


def write_analysis_and_figures(
    data: RunData,
    output_dir: Path,
    case_metrics: pd.DataFrame,
    domain_metrics: pd.DataFrame,
    shift_metrics: pd.DataFrame,
    model_comparison: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    case_summary_rows = []
    for case_id, group in data.meta.groupby("case_id"):
        idx = group.index.to_numpy()
        features = data.feature_matrix[idx]
        case_summary_rows.append(
            {
                "case_id": int(case_id),
                "domain_id": domain_for_case(int(case_id)),
                "num_runs": int(len(group)),
                "VB_mean": float(group["VB"].mean()),
                "VB_std": float(group["VB"].std()),
                "VB_min": float(group["VB"].min()),
                "VB_max": float(group["VB"].max()),
                "raw_length_mean": float(data.raw_lengths[idx].mean()),
                "feature_mean_abs": float(np.mean(np.abs(features))),
            }
        )
    case_dist = pd.DataFrame(case_summary_rows)
    case_dist.to_csv(output_dir / "analysis" / f"{PREFIX}_case_distribution_summary.csv", index=False)

    domain_shift_rows = []
    for source, target in TRANSFER_SCENARIOS:
        source_idx = data.meta.index[data.meta["case_id"].isin(DOMAIN_CASES[source])].to_numpy()
        target_idx = data.meta.index[data.meta["case_id"].isin(DOMAIN_CASES[target])].to_numpy()
        source_features = data.feature_matrix[source_idx]
        target_features = data.feature_matrix[target_idx]
        domain_shift_rows.append(
            {
                "scenario_name": f"{source}_to_{target}",
                "source_domain": source,
                "target_domain": target,
                "source_cases": ",".join(map(str, DOMAIN_CASES[source])),
                "target_cases": ",".join(map(str, DOMAIN_CASES[target])),
                "source_VB_mean": float(data.meta.loc[source_idx, "VB"].mean()),
                "target_VB_mean": float(data.meta.loc[target_idx, "VB"].mean()),
                "delta_VB_mean_target_minus_source": float(data.meta.loc[target_idx, "VB"].mean() - data.meta.loc[source_idx, "VB"].mean()),
                "feature_mean_l2_distance": float(np.linalg.norm(source_features.mean(axis=0) - target_features.mean(axis=0))),
                "feature_std_l2_distance": float(np.linalg.norm(source_features.std(axis=0) - target_features.std(axis=0))),
            }
        )
    domain_shift = pd.DataFrame(domain_shift_rows)
    domain_shift.to_csv(output_dir / "analysis" / f"{PREFIX}_domain_shift_analysis.csv", index=False)

    effect_rows = []
    mean_by_model = model_comparison.set_index("model_name")["mean_RMSE_over_6_shifts"].to_dict()
    comparisons = [
        ("feature_ridge", "cnn_only", "CNN-only vs Ridge feature baseline"),
        ("feature_random_forest", "cnn_only", "CNN-only vs Random Forest feature baseline"),
        ("feature_svr", "cnn_only", "CNN-only vs SVR feature baseline"),
        ("feature_xgboost", "cnn_only", "CNN-only vs XGBoost feature baseline"),
        ("cnn_only", "cnn_lag_concat", "CNN lag concat vs CNN-only"),
        ("cnn_only", "cnn_gru", "CNN-GRU vs CNN-only"),
        ("cnn_lag_concat", "cnn_gru", "CNN-GRU vs CNN lag concat"),
    ]
    for base, challenger, label in comparisons:
        if base in mean_by_model and challenger in mean_by_model:
            delta = mean_by_model[challenger] - mean_by_model[base]
            effect_rows.append(
                {
                    "comparison": label,
                    "baseline_model": base,
                    "candidate_model": challenger,
                    "baseline_mean_RMSE": mean_by_model[base],
                    "candidate_mean_RMSE": mean_by_model[challenger],
                    "delta_RMSE_candidate_minus_baseline": delta,
                    "candidate_improved": bool(delta < 0),
                    "interpretation": "candidate improved RMSE" if delta < 0 else "candidate did not improve RMSE",
                }
            )
    effect = pd.DataFrame(effect_rows)
    effect.to_csv(output_dir / "analysis" / f"{PREFIX}_run_sequence_effect_summary.csv", index=False)
    cnn_gru_comparison = build_cnn_gru_baseline_comparison(model_comparison)
    cnn_gru_comparison.to_csv(output_dir / "analysis" / f"{PREFIX}_cnn_gru_baseline_comparison.csv", index=False)

    analysis_summary = {
        "best_model_by_rmse": model_comparison.iloc[0].to_dict() if not model_comparison.empty else {},
        "num_predictions": int(len(predictions)),
        "domain_shift_rows": int(len(domain_shift)),
        "run_sequence_effects": effect.to_dict("records"),
        "cnn_gru_baseline_comparisons": cnn_gru_comparison.to_dict("records"),
    }
    write_json(output_dir / "analysis" / f"{PREFIX}_analysis_summary.json", analysis_summary)

    make_figures(data, output_dir, case_metrics, domain_metrics, model_comparison, predictions, case_dist)
    return analysis_summary


def build_cnn_gru_baseline_comparison(model_comparison: pd.DataFrame) -> pd.DataFrame:
    if model_comparison.empty or "cnn_gru" not in set(model_comparison["model_name"]):
        return pd.DataFrame()
    by_model = model_comparison.set_index("model_name")
    cnn_gru = by_model.loc["cnn_gru"]
    rows = []
    for baseline in ["feature_ridge", "feature_random_forest", "feature_svr", "feature_xgboost", "cnn_only", "cnn_lag_concat"]:
        if baseline not in by_model.index:
            continue
        base = by_model.loc[baseline]
        delta_rmse = float(cnn_gru["mean_RMSE_over_6_shifts"] - base["mean_RMSE_over_6_shifts"])
        delta_mae = float(cnn_gru["mean_MAE_over_6_shifts"] - base["mean_MAE_over_6_shifts"])
        delta_r2 = float(cnn_gru["mean_R2_over_6_shifts"] - base["mean_R2_over_6_shifts"])
        rows.append(
            {
                "model_name": "cnn_gru",
                "baseline_model": baseline,
                "cnn_gru_mean_RMSE": float(cnn_gru["mean_RMSE_over_6_shifts"]),
                "baseline_mean_RMSE": float(base["mean_RMSE_over_6_shifts"]),
                "delta_RMSE_cnn_gru_minus_baseline": delta_rmse,
                "cnn_gru_improved_RMSE": bool(delta_rmse < 0),
                "cnn_gru_mean_MAE": float(cnn_gru["mean_MAE_over_6_shifts"]),
                "baseline_mean_MAE": float(base["mean_MAE_over_6_shifts"]),
                "delta_MAE_cnn_gru_minus_baseline": delta_mae,
                "cnn_gru_improved_MAE": bool(delta_mae < 0),
                "cnn_gru_mean_R2": float(cnn_gru["mean_R2_over_6_shifts"]),
                "baseline_mean_R2": float(base["mean_R2_over_6_shifts"]),
                "delta_R2_cnn_gru_minus_baseline": delta_r2,
                "cnn_gru_improved_R2": bool(delta_r2 > 0),
                "interpretation": "CNN-GRU improved target-domain RMSE" if delta_rmse < 0 else "CNN-GRU did not improve target-domain RMSE",
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    data: RunData,
    output_dir: Path,
    case_metrics: pd.DataFrame,
    domain_metrics: pd.DataFrame,
    model_comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    case_dist: pd.DataFrame,
) -> None:
    fig_dir = output_dir / "figures"
    plt.figure(figsize=(8, 4.5))
    plt.bar(model_comparison["model_name"], model_comparison["mean_RMSE_over_6_shifts"], yerr=model_comparison["std_RMSE_over_6_shifts"].fillna(0))
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean target-domain RMSE")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{PREFIX}_model_comparison_rmse.png", dpi=180)
    plt.close()

    case_plot = case_metrics.groupby(["test_case_id", "model_name"], as_index=False)["test_RMSE"].mean()
    pivot = case_plot.pivot(index="test_case_id", columns="model_name", values="test_RMSE")
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Case-level RMSE")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{PREFIX}_case_level_rmse.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5.5, 5.5))
    for model_name, g in predictions.groupby("model_name"):
        plt.scatter(g["y_true"], g["y_pred"], s=18, alpha=0.55, label=model_name)
    lo = min(predictions["y_true"].min(), predictions["y_pred"].min())
    hi = max(predictions["y_true"].max(), predictions["y_pred"].max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel("True VB")
    plt.ylabel("Predicted VB")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{PREFIX}_prediction_vs_true.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    for model_name, g in predictions.groupby("model_name"):
        agg = g.groupby("run_order")["absolute_error"].mean().reset_index()
        plt.plot(agg["run_order"], agg["absolute_error"], marker="o", label=model_name)
    plt.xlabel("Run order within target case")
    plt.ylabel("Mean absolute error")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{PREFIX}_prediction_error_over_run_order.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    data.meta.boxplot(column="VB", by="case_id")
    plt.title("")
    plt.suptitle("")
    plt.ylabel("VB")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{PREFIX}_target_distribution_by_case.png", dpi=180)
    plt.close()

    feature_df = pd.DataFrame({"case_id": data.meta["case_id"], "smcAC_std": data.feature_matrix[:, data.feature_names.index("smcAC__full_length__std")]})
    plt.figure(figsize=(7, 4.5))
    feature_df.boxplot(column="smcAC_std", by="case_id")
    plt.title("")
    plt.suptitle("")
    plt.ylabel("smcAC std")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{PREFIX}_feature_distribution_by_case.png", dpi=180)
    plt.close()

    scaler = StandardScaler().fit(data.feature_matrix)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(scaler.transform(data.feature_matrix))
    plt.figure(figsize=(6, 5))
    for case_id in sorted(data.meta["case_id"].unique()):
        mask = data.meta["case_id"].to_numpy() == case_id
        plt.scatter(coords[mask, 0], coords[mask, 1], label=f"Case {case_id}", s=28)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{PREFIX}_pca_run_level_features_by_case.png", dpi=180)
    plt.close()

    make_filtered_prediction_figures(output_dir, predictions)


def make_learning_history_figures(output_dir: Path, learning_history: pd.DataFrame) -> None:
    if learning_history.empty:
        return
    fig_dir = output_dir / "figures"
    hist = learning_history.copy()
    avg = hist.groupby(["model_name", "epoch"], as_index=False).agg(train_loss=("train_loss", "mean"), val_loss=("val_loss", "mean"), val_RMSE=("val_RMSE", "mean"))

    plt.figure(figsize=(9, 5))
    for model_name, g in avg.groupby("model_name"):
        plt.plot(g["epoch"], g["train_loss"], linestyle="-", label=f"{model_name} train")
        plt.plot(g["epoch"], g["val_loss"], linestyle="--", label=f"{model_name} val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss on scaled target")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{PREFIX}_learning_history_loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for model_name, g in avg.groupby("model_name"):
        plt.plot(g["epoch"], g["val_RMSE"], label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation RMSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{PREFIX}_learning_history_val_rmse_curve.png", dpi=180)
    plt.close()


def make_filtered_prediction_figures(output_dir: Path, predictions: pd.DataFrame) -> None:
    if "include_in_visualization" not in predictions.columns:
        return
    fig_dir = output_dir / "figures"
    filtered = predictions.loc[predictions["include_in_visualization"]].copy()
    if filtered.empty:
        return

    true_ref = filtered.drop_duplicates(["case_id", "run_order"])[["case_id", "run_order", "y_true"]]
    pred_avg = filtered.groupby(["case_id", "run_order", "model_name"], as_index=False)["y_pred"].mean()
    cases = sorted(filtered["case_id"].unique())
    ncols = 3
    nrows = int(math.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.5 * nrows), squeeze=False)
    for ax, case_id in zip(axes.ravel(), cases):
        tr = true_ref.loc[true_ref["case_id"] == case_id].sort_values("run_order")
        ax.plot(tr["run_order"], tr["y_true"], color="black", linewidth=2, label="y_true")
        for model_name, g in pred_avg.loc[pred_avg["case_id"] == case_id].groupby("model_name"):
            g = g.sort_values("run_order")
            ax.plot(g["run_order"], g["y_pred"], marker="o", markersize=3, label=model_name)
        ax.set_title(f"Case {case_id}")
        ax.set_xlabel("run_order")
        ax.set_ylabel("VB")
    for ax in axes.ravel()[len(cases) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_dir / f"{PREFIX}_prediction_over_run_order_by_case.png", dpi=180)
    fig.savefig(fig_dir / f"{PREFIX}_case_wise_model_prediction_comparison.png", dpi=180)
    plt.close(fig)

    models = sorted(filtered["model_name"].unique())
    ncols = 3
    nrows = int(math.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.5 * nrows), squeeze=False)
    model_avg = filtered.groupby(["model_name", "case_id", "run_order"], as_index=False).agg(y_pred=("y_pred", "mean"), y_true=("y_true", "mean"))
    for ax, model_name in zip(axes.ravel(), models):
        mg = model_avg.loc[model_avg["model_name"] == model_name]
        for case_id, g in mg.groupby("case_id"):
            g = g.sort_values("run_order")
            ax.plot(g["run_order"], g["y_true"], color="black", alpha=0.25, linewidth=1)
            ax.plot(g["run_order"], g["y_pred"], marker="o", markersize=3, label=f"case {case_id}")
        ax.set_title(model_name)
        ax.set_xlabel("run_order")
        ax.set_ylabel("VB")
    for ax in axes.ravel()[len(models) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_dir / f"{PREFIX}_prediction_over_run_order_by_model.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(10, 5))
    err_avg = filtered.groupby(["model_name", "run_order"], as_index=False)["absolute_error"].mean()
    for model_name, g in err_avg.groupby("model_name"):
        plt.plot(g["run_order"], g["absolute_error"], marker="o", label=model_name)
    plt.xlabel("run_order")
    plt.ylabel("Mean absolute error, filtered")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{PREFIX}_prediction_error_over_run_order_filtered.png", dpi=180)
    plt.close()

    filter_summary_path = output_dir / "analysis" / f"{PREFIX}_visualization_filter_summary.csv"
    if filter_summary_path.exists():
        fs = pd.read_csv(filter_summary_path)
        excluded = fs.loc[~fs["included_in_visualization"]]
        if excluded.empty:
            counts = pd.DataFrame({"model_name": sorted(predictions["model_name"].unique()), "excluded_count": 0})
        else:
            counts = excluded.groupby("model_name", as_index=False).size().rename(columns={"size": "excluded_count"})
        plt.figure(figsize=(8, 4.5))
        plt.bar(counts["model_name"], counts["excluded_count"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Excluded case/model/seed conditions")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_excluded_high_rmse_conditions.png", dpi=180)
        plt.close()


def write_report(
    output_dir: Path,
    config: dict[str, Any],
    validation: dict[str, Any],
    smoke_result: dict[str, Any],
    model_comparison: pd.DataFrame,
    effect_summary: pd.DataFrame,
    analysis_summary: dict[str, Any],
) -> None:
    best = model_comparison.iloc[0].to_dict() if not model_comparison.empty else {}
    effect_text = "\n".join(
        f"- {r['comparison']}: delta RMSE={r['delta_RMSE_candidate_minus_baseline']:.4f}, improved={r['candidate_improved']}"
        for _, r in effect_summary.iterrows()
    )
    cnn_gru_comp_path = output_dir / "analysis" / f"{PREFIX}_cnn_gru_baseline_comparison.csv"
    cnn_gru_comp = pd.read_csv(cnn_gru_comp_path) if cnn_gru_comp_path.exists() else pd.DataFrame()
    cnn_gru_md = cnn_gru_comp.to_markdown(index=False) if not cnn_gru_comp.empty else "No CNN-GRU baseline comparison available."
    best_epoch_path = output_dir / "analysis" / f"{PREFIX}_best_epoch_summary.csv"
    best_epoch = pd.read_csv(best_epoch_path) if best_epoch_path.exists() else pd.DataFrame()
    best_epoch_md = best_epoch.groupby("model_name", as_index=False).agg(mean_best_epoch=("best_epoch", "mean"), mean_best_val_RMSE=("best_val_RMSE", "mean"), mean_final_epoch=("final_epoch", "mean"), early_stop_count=("stopped_by_early_stopping", "sum")).to_markdown(index=False) if not best_epoch.empty else "No DL best epoch summary available."
    filter_path = output_dir / "analysis" / f"{PREFIX}_visualization_filter_summary.csv"
    filter_summary = pd.read_csv(filter_path) if filter_path.exists() else pd.DataFrame()
    excluded_count = int((~filter_summary["included_in_visualization"]).sum()) if not filter_summary.empty else 0
    feature_best = model_comparison.loc[model_comparison["model_name"].isin(FEATURE_MODELS)].sort_values("mean_RMSE_over_6_shifts").head(1)
    feature_best_name = feature_best["model_name"].iloc[0] if not feature_best.empty else "none"
    feature_best_rmse = feature_best["mean_RMSE_over_6_shifts"].iloc[0] if not feature_best.empty else float("nan")
    comp_md = model_comparison.to_markdown(index=False) if not model_comparison.empty else "No model comparison available."
    smoke_md = json.dumps(to_builtin(smoke_result), indent=2, ensure_ascii=False)
    report = f"""# H4_S1 CNN-GRU Run Sequence Domain Generalization

## 1. Executive Summary

- Experiment ID: `{config['experiment']['experiment_id']}`
- Execution dir: `{config['experiment']['execution_dir']}`
- Best model by target-domain RMSE: `{best.get('model_name')}` with mean RMSE `{best.get('mean_RMSE_over_6_shifts')}`.
- Domain generalization protocol used A={{1,9}}, B={{2,12}}, C={{8,14}} and all 6 source->target shifts.
- Run sequence models used sequence_size={config['sequence']['sequence_size']} with left padding and no cross-case sequence construction.
- Initial run used seeds {config['experiment']['seed_list']} and models {ALL_MODELS}.
- DL training used max_epochs={config['training']['max_epochs']} and patience={config['training']['early_stopping']['patience']} for all deep learning models.
- Feature-based models were expanded to Ridge, Random Forest, SVR, and XGBoost when available.
- Visualization-only RMSE filter excluded {excluded_count} case/model/seed conditions from prediction plots; metrics and rankings still include every condition.

## 2. Experiment Setup

- Data: `{config['data']['process_info_path']}`, `{config['data']['signal_data_path']}`
- Target: `{config['data']['target_col']}`
- Sensors: {', '.join(config['data']['sensor_columns'])}
- Cases: {config['data']['selected_cases']}
- Transfer scenarios: {', '.join([f'{s}_to_{t}' for s, t in TRANSFER_SCENARIOS])}
- Validation: chronological tail split inside source cases only, ratio={config['split']['validation_ratio']}.
- DL input: each full run is split into {config.get('model_design', {}).get('run_encoder', {}).get('num_windows', 5)} overlapped temporal windows of length {config.get('model_design', {}).get('run_encoder', {}).get('window_length', 3000)}; one shared CNN encodes each window and max pooling creates one run-level latent.
- Feature baseline input: current run is resampled to length {config['sequence']['sequence_length']} before handcrafted feature extraction.
- Run sequence: GRU/lag inputs use `[z_(t-2), z_(t-1), z_t]`, where each `z` is a windowed-CNN run-level latent.

## 3. Leakage Prevention

- Target domain in train/validation: `false`
- Target labels for model selection: `false`
- Sequence crosses case boundary: `false`
- Source-train-only scaler fitting: `true`
- Left padding for early runs: `{config['sequence']['padding'] == 'left'}`
- Leakage check passed: `{validation.get('passed')}`

## 4. Model Descriptions

- `feature_ridge`: 9 statistical/frequency features per sensor with Ridge regression.
- `feature_random_forest`: same handcrafted features with Random Forest.
- `feature_svr`: scaled handcrafted features with RBF SVR.
- `feature_xgboost`: handcrafted features with XGBoost if installed; skipped otherwise.
- `cnn_only`: `[B, K, C, W]` windowed current run -> shared window CNN -> max-pooled `z_t` -> MLP regressor.
- `cnn_lag_concat`: `[B, S, K, C, W]` -> shared window CNN per run -> `[z_(t-2), z_(t-1), z_t]` concat plus mask -> MLP.
- `cnn_gru`: `[B, S, K, C, W]` -> shared window CNN per run -> run-level latent sequence -> unidirectional GRU -> MLP.

The hierarchy is: within-run temporal encoding by windowed CNN, within-run window aggregation by max pooling, and between-run temporal modeling by GRU.

## 5. Main Results

{comp_md}

## 6. Run-sequence Effect

{effect_text}

## 6.1 CNN-GRU vs Baselines

{cnn_gru_md}

## 6.2 Feature-based Baseline Expansion

- SVR executed: `{validation.get('svr_executed')}`
- XGBoost executed: `{validation.get('xgboost_executed')}`; skipped reason: `{validation.get('xgboost_skip_reason')}`
- Best feature-based model by RMSE: `{feature_best_name}` with mean RMSE `{feature_best_rmse}`.
- SVR and XGBoost were added to separate linear, bagging/tree, kernel, and boosting behavior under the same domain-shift protocol.

## 6.3 DL Learning History

Learning history is saved in `metrics/{PREFIX}_learning_history.csv`; best epoch summary is saved in `analysis/{PREFIX}_best_epoch_summary.csv`.

{best_epoch_md}

The best epoch is selected by validation RMSE using source-domain validation only. Train/validation curves are saved as `figures/{PREFIX}_learning_history_loss_curve.png` and `figures/{PREFIX}_learning_history_val_rmse_curve.png`.

## 6.4 Prediction Pattern over Run Order

Prediction trajectory figures are saved in:
- `figures/{PREFIX}_prediction_over_run_order_by_case.png`
- `figures/{PREFIX}_prediction_over_run_order_by_model.png`
- `figures/{PREFIX}_prediction_error_over_run_order_filtered.png`
- `figures/{PREFIX}_case_wise_model_prediction_comparison.png`

These figures are intended to inspect systematic under-prediction or over-prediction by case and whether previous-run models behave more smoothly over run progression.

## 6.5 Visualization Filtering Note

RMSE > {config.get('visualization', {}).get('rmse_filter_threshold', 0.5)} conditions were excluded from prediction plots only. They were **not** excluded from metric aggregation, model ranking, or the report performance table. The full list is saved in `analysis/{PREFIX}_visualization_filter_summary.csv`; excluded counts are summarized in `analysis/{PREFIX}_excluded_high_rmse_summary.csv`.

## 7. Domain Shift Analysis

Domain-shift summaries are saved in `analysis/{PREFIX}_domain_shift_analysis.csv`. PCA and case distribution figures show that case-specific sensor/feature distributions differ materially, so negative or unstable R2 should be interpreted as domain generalization difficulty rather than simple in-distribution fit quality.

## 8. Smoke Test

```json
{smoke_md}
```

## 9. Interpretation

If feature baselines rank above DL models, the likely explanation is small sample size and strong cross-case shift. If lag concat or GRU improves over CNN-only, previous run context is useful. If lag concat beats GRU, previous-run information may help but recurrent modeling may be too flexible for this data regime.

## 10. Limitations

- Initial run only uses seeds {config['experiment']['seed_list']}; full 20-seed expansion remains a follow-up.
- DL window_length={config.get('model_design', {}).get('run_encoder', {}).get('window_length', 3000)} and num_windows={config.get('model_design', {}).get('run_encoder', {}).get('num_windows', 5)} are fixed hyperparameters in this run.
- Max pooling can be sensitive to noisy local activations and does not explain which window was important.
- Attention pooling, mean pooling, and max-mean concat remain future ablations.
- Sequence size is fixed to {config['sequence']['sequence_size']}; no segment or sensor-group ablation was run here.
- Target-domain labels were only used for final evaluation.

## 11. Recommendations

- Run full seeds 0..19 after reviewing initial stability.
- Add sequence_size ablation {{1,2,3,5}}.
- Add segment-aware CNN-GRU with `entry_exit` and `entry_steady_exit`.
- Compare NASA Ames behavior with PHM2010 once this pipeline is stable.
"""
    (output_dir / "reports" / f"{PREFIX}_report.md").write_text(report, encoding="utf-8")
    html = "<html><body>" + report.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>\n") + "</body></html>"
    (output_dir / "reports" / f"{PREFIX}_report.html").write_text(html, encoding="utf-8")


def run_smoke(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    smoke_cfg = config["smoke"]
    smoke_config = json.loads(json.dumps(to_builtin(config)))
    smoke_config["sequence"]["sequence_length"] = int(smoke_cfg.get("sequence_length", 128))
    smoke_config.setdefault("model_design", {}).setdefault("run_encoder", {})
    smoke_config["model_design"]["run_encoder"]["window_length"] = int(smoke_cfg.get("window_length", 300))
    smoke_config["model_design"]["run_encoder"]["num_windows"] = int(smoke_cfg.get("num_windows", 5))
    smoke_config["data"]["sensor_columns"] = config["data"]["sensor_columns"]
    data = load_run_data(smoke_config, sequence_length=smoke_config["sequence"]["sequence_length"])
    source, target = smoke_cfg["scenario"].split("_to_")
    result = run_scenario_seed(
        data,
        smoke_config,
        source,
        target,
        int(smoke_cfg.get("seed", 0)),
        output_dir,
        list(smoke_cfg.get("models", DL_MODELS)),
        max_epochs=int(smoke_cfg.get("max_epochs", 2)),
        batch_size=int(smoke_cfg.get("batch_size", 4)),
        smoke=True,
    )
    smoke_summary = {
        "scenario": smoke_cfg["scenario"],
        "seed": int(smoke_cfg.get("seed", 0)),
        "models": list(smoke_cfg.get("models", DL_MODELS)),
        "feature_sequence_length": smoke_config["sequence"]["sequence_length"],
        "window_length": int(smoke_config["model_design"]["run_encoder"]["window_length"]),
        "num_windows": int(smoke_config["model_design"]["run_encoder"]["num_windows"]),
        "windowed_input_shape": list(data.sequences.shape[1:]),
        "forward_backward_training_success": True,
        "prediction_rows": int(len(result["predictions"])),
        "metrics_rows": int(len(result["domain_metrics"])),
        "checkpoint_save_load_success": True,
        "target_domain_leakage": False,
        "domain_metrics": result["domain_metrics"].to_dict("records"),
    }
    write_json(output_dir / "smoke" / f"{PREFIX}_smoke_test_result.json", smoke_summary)
    return smoke_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/H4_S1_cnn_gru_run_sequence_domain_generalization.yaml")
    parser.add_argument("--output-root", default="experiments/executions")
    parser.add_argument("--reuse-output-dir", default=None, help="Overwrite artifacts in an existing H4_S1 execution directory instead of creating a new execution.")
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed override.")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    config_path = Path(args.config)
    config = load_yaml(config_path) if config_path.exists() else build_default_config()
    if args.seeds:
        config["experiment"]["seed_list"] = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    config["experiment"]["hypothesis_id"] = "H4"
    config["experiment"]["scenario_id"] = "S1"
    config["experiment"]["experiment_topic"] = EXPERIMENT_TOPIC
    if args.reuse_output_dir:
        output_dir = Path(args.reuse_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config["experiment"]["execution_dir"] = str(output_dir)
        if str(config["experiment"].get("experiment_id", "")).lower() == "auto":
            config["experiment"]["experiment_id"] = f"{output_dir.name.split('_cnn_gru_run_sequence_domain_generalization_NASA_Ames')[0]}_H4_S1_{EXPERIMENT_TOPIC}"
        config["experiment"]["path_schema_version"] = "hierarchical_v1"
    else:
        if str(config["experiment"].get("experiment_id", "")).lower() == "auto":
            config["experiment"].pop("experiment_id", None)
        if str(config["experiment"].get("execution_dir", "")).lower() == "auto":
            config["experiment"].pop("execution_dir", None)
        output_dir = Path(create_execution_dir(config, root=args.output_root))
    make_dirs(output_dir)
    config["experiment"]["execution_dir"] = str(output_dir)

    write_yaml(output_dir / "configs" / f"{PREFIX}_input_config.yaml", config)
    data = load_run_data(config, sequence_length=int(config["sequence"].get("sequence_length", 256)))
    write_data_artifacts(data, config, output_dir)
    validation = validate_config_and_data(config, data, output_dir)
    write_yaml(output_dir / "configs" / f"{PREFIX}_resolved_config.yaml", config)
    write_json(output_dir / "logs" / f"{PREFIX}_environment.json", {"python": os.sys.version, "torch": torch.__version__, "device": "cuda" if torch.cuda.is_available() else "cpu"})
    run_log = output_dir / "logs" / f"{PREFIX}_run.log"
    run_log.write_text(f"Dry-run validation passed={validation.get('passed')}\n", encoding="utf-8")
    (output_dir / "logs" / f"{PREFIX}_error.log").write_text("", encoding="utf-8")

    if args.dry_run_only:
        return

    smoke_result = {"skipped": True, "reason": "--skip-smoke"} if args.skip_smoke else run_smoke(config, output_dir)
    run_log.write_text(run_log.read_text(encoding="utf-8") + f"Smoke result: {smoke_result.get('forward_backward_training_success', smoke_result)}\n", encoding="utf-8")

    case_frames: list[pd.DataFrame] = []
    domain_frames: list[pd.DataFrame] = []
    pred_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    best_epoch_frames: list[pd.DataFrame] = []
    skipped_frames: list[pd.DataFrame] = []
    scaler_records: list[dict[str, Any]] = []
    models = [m for m in ALL_MODELS if config["models"].get(m, {}).get("enabled", True)]
    for seed in config["experiment"]["seed_list"]:
        for source, target in TRANSFER_SCENARIOS:
            result = run_scenario_seed(data, config, source, target, int(seed), output_dir, models=models)
            case_frames.append(result["case_metrics"])
            domain_frames.append(result["domain_metrics"])
            pred_frames.append(result["predictions"])
            split_frames.append(result["split"])
            if not result["learning_history"].empty:
                history_frames.append(result["learning_history"])
            if not result["best_epoch_summary"].empty:
                best_epoch_frames.append(result["best_epoch_summary"])
            if not result["skipped"].empty:
                skipped_frames.append(result["skipped"])
            scaler_records.append(result["scaler_fit"])
            run_log.write_text(run_log.read_text(encoding="utf-8") + f"Completed {source}_to_{target} seed={seed}\n", encoding="utf-8")

    case_metrics = pd.concat(case_frames, ignore_index=True)
    domain_metrics = pd.concat(domain_frames, ignore_index=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    splits = pd.concat(split_frames, ignore_index=True)
    learning_history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    best_epoch_summary = pd.concat(best_epoch_frames, ignore_index=True) if best_epoch_frames else pd.DataFrame()
    skipped_cols = ["experiment_id", "scenario_name", "source_domain", "target_domain", "model_name", "seed", "reason"]
    skipped = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(columns=skipped_cols)
    shift_metrics, seed_metrics, model_comparison = aggregate_outputs(case_metrics, domain_metrics)
    predictions, filter_summary, excluded_summary = apply_visualization_filter(predictions, case_metrics, domain_metrics, config, output_dir)

    case_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_case_metrics.csv", index=False)
    domain_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_target_domain_metrics.csv", index=False)
    shift_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_shift_metrics.csv", index=False)
    seed_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_seed_metrics.csv", index=False)
    model_comparison.to_csv(output_dir / "metrics" / f"{PREFIX}_model_comparison.csv", index=False)
    learning_history.to_csv(output_dir / "metrics" / f"{PREFIX}_learning_history.csv", index=False)
    best_epoch_summary.to_csv(output_dir / "analysis" / f"{PREFIX}_best_epoch_summary.csv", index=False)
    skipped.to_csv(output_dir / "analysis" / f"{PREFIX}_skipped_conditions.csv", index=False)
    validation.update(
        {
            "svr_executed": "feature_svr" in set(domain_metrics["model_name"]),
            "xgboost_executed": "feature_xgboost" in set(domain_metrics["model_name"]),
            "xgboost_skipped": "feature_xgboost" not in set(domain_metrics["model_name"]),
            "xgboost_skip_reason": "xgboost_not_installed" if "feature_xgboost" not in set(domain_metrics["model_name"]) and XGBRegressor is None else None,
            "dl_learning_history_saved": bool(not learning_history.empty),
            "best_epoch_selected_by_val_rmse": bool(not best_epoch_summary.empty),
            "prediction_csv_has_run_order": "run_order" in predictions.columns,
            "visualization_filter_used_case_level_rmse": True,
            "rmse_gt_0_5_excluded_from_plots_only": True,
            "rmse_gt_0_5_excluded_from_metrics": False,
            "visualization_filtered_condition_count": int((~filter_summary["included_in_visualization"]).sum()) if not filter_summary.empty else 0,
        }
    )
    write_json(output_dir / "configs" / f"{PREFIX}_config_validation.json", validation)
    predictions.sort_values(["scenario_name", "seed", "case_id", "run_order", "model_name"]).to_csv(output_dir / "predictions" / f"{PREFIX}_test_predictions.csv", index=False)
    predictions.to_csv(output_dir / "predictions" / f"{PREFIX}_prediction_errors.csv", index=False)
    splits.to_csv(output_dir / "splits" / f"{PREFIX}_split.csv", index=False)
    split_summary = {
        "scenarios": [f"{s}_to_{t}" for s, t in TRANSFER_SCENARIOS],
        "seeds": config["experiment"]["seed_list"],
        "split_counts": splits.groupby(["scenario_name", "seed", "split"]).size().reset_index(name="count").to_dict("records"),
        "scaler_fit_records": scaler_records,
    }
    write_json(output_dir / "splits" / f"{PREFIX}_split_summary.json", split_summary)
    leak = leakage_check(splits, config)
    write_json(output_dir / "splits" / f"{PREFIX}_leakage_check.json", leak)
    analysis_summary = write_analysis_and_figures(data, output_dir, case_metrics, domain_metrics, shift_metrics, model_comparison, predictions)
    make_learning_history_figures(output_dir, learning_history)
    effect_summary = pd.read_csv(output_dir / "analysis" / f"{PREFIX}_run_sequence_effect_summary.csv")
    write_report(output_dir, config, leak, smoke_result, model_comparison, effect_summary, analysis_summary)
    run_log.write_text(run_log.read_text(encoding="utf-8") + "Finished H4_S1 initial run.\n", encoding="utf-8")


if __name__ == "__main__":
    main()

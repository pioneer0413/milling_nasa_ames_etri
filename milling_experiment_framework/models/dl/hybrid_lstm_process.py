from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from milling_experiment_framework.registry.model_registry import MODEL_REGISTRY


TemporalOutputMode = Literal["last_hidden", "last_timestep", "mean_pool", "attention_pool"]
InputMode = Literal["sensor_only", "process_only", "sensor_plus_process"]


class AttentionPool(nn.Module):
    """Small additive attention pooling layer over LSTM outputs."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Tanh(), nn.Linear(feature_dim, 1))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(sequence), dim=1)
        return torch.sum(sequence * weights, dim=1)


class LSTMTemporalEncoder(nn.Module):
    """Stacked LSTM temporal encoder for batch-first multi-sensor sequences."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False,
        temporal_output_mode: TemporalOutputMode = "last_hidden",
        output_dropout: float | None = None,
    ):
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.temporal_output_mode = temporal_output_mode
        self.num_directions = 2 if self.bidirectional else 1
        self.output_dim = self.hidden_size * self.num_directions
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(float(output_dropout if output_dropout is not None else dropout))
        self.attention = AttentionPool(self.output_dim) if temporal_output_mode == "attention_pool" else None

    def forward(self, sensor_sequence: torch.Tensor) -> torch.Tensor:
        if sensor_sequence.ndim != 3:
            raise ValueError(
                "sensor_sequence must have shape (batch_size, sequence_length, num_sensors); "
                f"got {tuple(sensor_sequence.shape)}"
            )
        if sensor_sequence.shape[-1] != self.input_size:
            raise ValueError(f"Expected num_sensors/input_size={self.input_size}, got {sensor_sequence.shape[-1]}")
        if sensor_sequence.shape[1] <= 0:
            raise ValueError("sequence_length must be positive.")
        output, (hidden, _) = self.lstm(sensor_sequence)
        if self.temporal_output_mode == "last_hidden":
            if self.bidirectional:
                pooled = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                pooled = hidden[-1]
        elif self.temporal_output_mode == "last_timestep":
            pooled = output[:, -1, :]
        elif self.temporal_output_mode == "mean_pool":
            pooled = output.mean(dim=1)
        elif self.temporal_output_mode == "attention_pool":
            if self.attention is None:
                raise RuntimeError("attention pooling layer was not initialized.")
            pooled = self.attention(output)
        else:
            raise ValueError(f"Unsupported temporal_output_mode: {self.temporal_output_mode}")
        return self.output_dropout(pooled)


class RegressionHead(nn.Module):
    """Configurable nonlinear regression head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (32, 8),
        output_dim: int = 1,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("regression head input_dim must be positive.")
        act: nn.Module
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "gelu":
            act = nn.GELU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        layers: list[nn.Module] = []
        current = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current, int(hidden_dim)), act.__class__(), nn.Dropout(float(dropout))])
            current = int(hidden_dim)
        layers.append(nn.Linear(current, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridLSTMProcessRegressor(nn.Module):
    """Hybrid LSTM + process-information regressor for tool wear (VB) prediction.

    Parameters follow the 2020 hybrid LSTM information model: a stacked LSTM temporal
    encoder extracts sequence features, those features are concatenated with process
    information, and a nonlinear fully connected head predicts VB.
    """

    def __init__(
        self,
        num_sensors: int,
        process_feature_dim: int = 0,
        hidden_size: int = 256,
        num_layers: int = 3,
        lstm_dropout: float = 0.2,
        bidirectional: bool = False,
        temporal_output_mode: TemporalOutputMode = "last_hidden",
        regression_hidden_dims: list[int] | tuple[int, ...] = (32, 8),
        regression_dropout: float = 0.2,
        output_dim: int = 1,
        use_process_info: bool = True,
        input_mode: InputMode = "sensor_plus_process",
    ):
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.process_feature_dim = int(process_feature_dim)
        self.use_process_info = bool(use_process_info)
        self.input_mode = input_mode
        if self.input_mode not in {"sensor_only", "process_only", "sensor_plus_process"}:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")
        if self.input_mode in {"sensor_only", "sensor_plus_process"}:
            self.temporal_encoder = LSTMTemporalEncoder(
                input_size=self.num_sensors,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=lstm_dropout,
                bidirectional=bidirectional,
                temporal_output_mode=temporal_output_mode,
                output_dropout=lstm_dropout,
            )
            temporal_dim = self.temporal_encoder.output_dim
        else:
            self.temporal_encoder = None
            temporal_dim = 0
        process_dim = self.process_feature_dim if self.input_mode in {"process_only", "sensor_plus_process"} and self.use_process_info else 0
        if self.input_mode == "process_only" and process_dim <= 0:
            raise ValueError("process_only mode requires process_feature_dim > 0 and use_process_info=True.")
        self.hybrid_dim = temporal_dim + process_dim
        self.regression_head = RegressionHead(
            input_dim=self.hybrid_dim,
            hidden_dims=regression_hidden_dims,
            output_dim=output_dim,
            dropout=regression_dropout,
            activation="relu",
        )

    def forward(self, sensor_sequence: torch.Tensor | None, process_features: torch.Tensor | None = None) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.input_mode in {"sensor_only", "sensor_plus_process"}:
            if sensor_sequence is None:
                raise ValueError(f"{self.input_mode} mode requires sensor_sequence.")
            parts.append(self.temporal_encoder(sensor_sequence))  # type: ignore[arg-type, union-attr]
        if self.input_mode in {"process_only", "sensor_plus_process"} and self.use_process_info:
            if process_features is None:
                raise ValueError(f"{self.input_mode} mode requires process_features.")
            if process_features.ndim != 2:
                raise ValueError(f"process_features must have shape (batch_size, num_process_features), got {tuple(process_features.shape)}")
            if process_features.shape[1] != self.process_feature_dim:
                raise ValueError(f"Expected process_feature_dim={self.process_feature_dim}, got {process_features.shape[1]}")
            if parts and process_features.shape[0] != parts[0].shape[0]:
                raise ValueError("sensor_sequence and process_features batch sizes do not match.")
            parts.append(process_features)
        hybrid = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        out = self.regression_head(hybrid)
        if out.ndim != 2 or out.shape[1] != 1:
            raise RuntimeError(f"HybridLSTMProcessRegressor output must be (batch_size, 1), got {tuple(out.shape)}")
        return out


@MODEL_REGISTRY.register("hybrid_lstm_process")
class HybridLSTMProcessModel:
    model_type = "DL"
    input_type = "hybrid"

    def __init__(self, config: dict, task_config: dict):
        params = dict(config.get("params", config.get("model", {}).get("params", {})))
        lstm = dict(config.get("lstm", params.get("lstm", {})))
        head = dict(config.get("regression_head", params.get("regression_head", {})))
        num_sensors = params.get("num_sensors", config.get("num_sensors", lstm.get("input_size", 1)))
        process_feature_dim = params.get("process_feature_dim", config.get("process_feature_dim", 0))
        if str(num_sensors).lower() == "auto":
            num_sensors = 1
        if str(process_feature_dim).lower() == "auto":
            process_feature_dim = 0
        self.task_config = task_config
        self.module = HybridLSTMProcessRegressor(
            num_sensors=int(num_sensors),
            process_feature_dim=int(process_feature_dim),
            hidden_size=int(lstm.get("hidden_size", 256)),
            num_layers=int(lstm.get("num_layers", 3)),
            lstm_dropout=float(lstm.get("dropout", 0.2)),
            bidirectional=bool(lstm.get("bidirectional", False)),
            temporal_output_mode=lstm.get("temporal_output_mode", "last_hidden"),
            regression_hidden_dims=head.get("hidden_dims", [32, 8]),
            regression_dropout=float(head.get("dropout", 0.2)),
            output_dim=int(head.get("output_dim", 1)),
            use_process_info=bool(params.get("use_process_info", config.get("use_process_info", True))),
            input_mode=params.get("input_mode", "sensor_plus_process"),
        )

    def fit(self, X, y, **kwargs):
        raise RuntimeError("HybridLSTMProcessModel is trained by a hybrid sequence trainer/entrypoint.")

    def predict(self, X):
        raise RuntimeError("Use scripts/run_hybrid_lstm_process_experiment.py for hybrid sequence prediction.")

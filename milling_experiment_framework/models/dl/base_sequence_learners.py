from __future__ import annotations

from typing import Literal

import torch
from torch import nn


def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class SequenceRegressionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (64,),
        dropout: float = 0.5,
        activation: str = "relu",
        output_dim: int = 1,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        layers: list[nn.Module] = []
        current = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current, int(hidden_dim)))
            layers.append(make_activation(activation))
            layers.append(nn.Dropout(float(dropout)))
            current = int(hidden_dim)
        layers.append(nn.Linear(current, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if out.ndim != 2 or out.shape[1] != 1:
            raise RuntimeError(f"Regressor output must be (batch, 1), got {tuple(out.shape)}")
        return out


class _RecurrentRegressor(nn.Module):
    recurrent_cls: type[nn.RNNBase]

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.5,
        batch_first: bool = True,
        head_hidden_dims: list[int] | tuple[int, ...] = (64,),
        rnn_nonlinearity: Literal["tanh", "relu"] = "tanh",
    ):
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        kwargs = {}
        if self.recurrent_cls is nn.RNN:
            kwargs["nonlinearity"] = rnn_nonlinearity
        self.recurrent = self.recurrent_cls(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=batch_first,
            **kwargs,
        )
        self.head = SequenceRegressionHead(hidden_size, head_hidden_dims, dropout=dropout)

    def forward(self, sensor_sequence: torch.Tensor) -> torch.Tensor:
        if sensor_sequence.ndim != 3:
            raise ValueError(f"sensor_sequence must be (batch, sequence_length, channels), got {tuple(sensor_sequence.shape)}")
        _, hidden = self.recurrent(sensor_sequence)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.head(hidden[-1])


class RNNRegressor(_RecurrentRegressor):
    recurrent_cls = nn.RNN


class GRURegressor(_RecurrentRegressor):
    recurrent_cls = nn.GRU


class LSTMRegressor(_RecurrentRegressor):
    recurrent_cls = nn.LSTM


class CNN1DRegressor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: list[int] | tuple[int, ...] = (128, 64, 32),
        kernel_size: int = 3,
        dropout: float = 0.5,
        pooling: str = "max",
        head_hidden_dims: list[int] | tuple[int, ...] = (64,),
    ):
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        layers: list[nn.Module] = []
        current = int(input_channels)
        pad = int(kernel_size) // 2
        for out_channels in channels:
            layers.extend([nn.Conv1d(current, int(out_channels), kernel_size=kernel_size, padding=pad), nn.ReLU(), nn.Dropout(float(dropout))])
            if pooling == "max":
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            current = int(out_channels)
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        self.features = nn.Sequential(*layers)
        self.head = SequenceRegressionHead(current, head_hidden_dims, dropout=dropout)

    def forward(self, sensor_sequence: torch.Tensor) -> torch.Tensor:
        if sensor_sequence.ndim != 3:
            raise ValueError(f"sensor_sequence must be (batch, sequence_length, channels), got {tuple(sensor_sequence.shape)}")
        x = sensor_sequence.transpose(1, 2)
        return self.head(self.features(x))


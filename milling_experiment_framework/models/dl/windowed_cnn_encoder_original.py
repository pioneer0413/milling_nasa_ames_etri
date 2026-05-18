from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class WindowedCNNConfig:
    input_channels: int = 6
    num_windows: int = 5
    window_length: int = 3000
    channels: tuple[int, ...] = (16, 32)
    kernel_size: int = 5
    latent_dim: int = 32
    use_batch_norm: bool = True
    dropout: float = 0.0
    aggregation: str = "max_pool"


def _make_cnn(
    input_channels: int,
    channels: list[int] | tuple[int, ...],
    kernel_size: int,
    use_batch_norm: bool,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = int(input_channels)
    padding = int(kernel_size) // 2
    for layer_idx, out_channels in enumerate(channels):
        layers.append(nn.Conv1d(in_channels, int(out_channels), kernel_size=int(kernel_size), padding=padding))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(int(out_channels)))
        layers.append(nn.ReLU())
        if layer_idx == 0:
            layers.append(nn.MaxPool1d(kernel_size=2))
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        in_channels = int(out_channels)
    layers.append(nn.AdaptiveAvgPool1d(1))
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)


class WindowedCNNEncoder(nn.Module):
    """Shared-window CNN run encoder.

    Input shape is ``[batch, num_windows, num_sensors, window_length]``.
    The same CNN is applied to every window, producing window-level latents
    ``[batch, num_windows, latent_dim]``. Run-level latent ``z_t`` is max-pooled
    over the window axis.
    """

    def __init__(
        self,
        input_channels: int = 6,
        num_windows: int = 5,
        window_length: int = 3000,
        channels: list[int] | tuple[int, ...] = (16, 32),
        kernel_size: int = 5,
        latent_dim: int = 32,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        aggregation: str = "max_pool",
    ):
        super().__init__()
        if aggregation != "max_pool":
            raise ValueError(f"H4_S1 currently supports only max_pool aggregation, got {aggregation}")
        if input_channels <= 0 or num_windows <= 0 or window_length <= 0 or latent_dim <= 0:
            raise ValueError("input_channels, num_windows, window_length, and latent_dim must be positive.")
        if not channels:
            raise ValueError("channels must not be empty.")
        self.config = WindowedCNNConfig(
            input_channels=int(input_channels),
            num_windows=int(num_windows),
            window_length=int(window_length),
            channels=tuple(int(c) for c in channels),
            kernel_size=int(kernel_size),
            latent_dim=int(latent_dim),
            use_batch_norm=bool(use_batch_norm),
            dropout=float(dropout),
            aggregation=str(aggregation),
        )
        self.input_channels = int(input_channels)
        self.num_windows = int(num_windows)
        self.window_length = int(window_length)
        self.latent_dim = int(latent_dim)
        self.aggregation = aggregation
        self.cnn = _make_cnn(input_channels, channels, kernel_size, use_batch_norm, dropout)
        self.proj = nn.Sequential(nn.Linear(int(channels[-1]), self.latent_dim), nn.ReLU())

    def forward(self, x: torch.Tensor, return_window_latents: bool = False):
        if x.ndim != 4:
            raise ValueError(f"WindowedCNNEncoder expects [B,K,C,W], got {tuple(x.shape)}")
        batch_size, num_windows, num_channels, window_length = x.shape
        if num_windows != self.num_windows:
            raise ValueError(f"Expected num_windows={self.num_windows}, got {num_windows}")
        if num_channels != self.input_channels:
            raise ValueError(f"Expected input_channels={self.input_channels}, got {num_channels}")
        if window_length != self.window_length:
            raise ValueError(f"Expected window_length={self.window_length}, got {window_length}")
        x_flat = x.reshape(batch_size * num_windows, num_channels, window_length)
        h_flat = self.proj(self.cnn(x_flat))
        window_latents = h_flat.reshape(batch_size, num_windows, self.latent_dim)
        z, _ = torch.max(window_latents, dim=1)
        if return_window_latents:
            return z, window_latents
        return z


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(int(input_dim), int(hidden_dim)), nn.ReLU()]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(int(hidden_dim), 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if out.ndim != 2 or out.shape[1] != 1:
            raise RuntimeError(f"RegressionHead must produce [B,1], got {tuple(out.shape)}")
        return out.squeeze(1)


def make_windowed_encoder_from_config(config: dict, input_channels: int) -> WindowedCNNEncoder:
    design = config.get("model_design", {})
    run_encoder = design.get("run_encoder", {})
    cnn = design.get("cnn_encoder", {})
    legacy = config.get("dl_model", {})
    return WindowedCNNEncoder(
        input_channels=int(run_encoder.get("input_channels", input_channels)),
        num_windows=int(run_encoder.get("num_windows", 5)),
        window_length=int(run_encoder.get("window_length", 3000)),
        channels=tuple(int(x) for x in cnn.get("channels", legacy.get("cnn_channels", [16, 32]))),
        kernel_size=int(cnn.get("kernel_size", legacy.get("kernel_size", 5))),
        latent_dim=int(run_encoder.get("latent_dim", legacy.get("latent_dim", 32))),
        use_batch_norm=bool(cnn.get("use_batch_norm", True)),
        dropout=float(cnn.get("dropout", legacy.get("dropout", 0.0))),
        aggregation=str(run_encoder.get("window_aggregation", "max_pool")),
    )

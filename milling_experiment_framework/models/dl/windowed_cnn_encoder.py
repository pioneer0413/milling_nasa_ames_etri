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
    # Kept for backward compatibility with existing configs.
    # In the revised encoder, the primary representation is a (B, W') vector,
    # and this value is typically used by downstream models (e.g. an embedding
    # dimension before a GRU) rather than being produced directly by the encoder.
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
    """Build a sensor-wise temporal CNN.

    This CNN preserves the input channel count (sensor identity) by using
    depthwise convolutions (groups = input_channels). It performs *temporal*
    convolution within each window independently and preserves the time axis
    length (via padding), so callers can pool across the window axis later.

    Note: ``channels`` is used only to determine the number of stacked blocks
    (its values are ignored) to remain compatible with existing configs.
    """

    layers: list[nn.Module] = []
    in_channels = int(input_channels)
    if in_channels <= 0:
        raise ValueError("input_channels must be positive")
    if not channels:
        raise ValueError("channels must not be empty")

    padding = int(kernel_size) // 2
    num_blocks = len(channels)
    for _layer_idx in range(num_blocks):
        layers.append(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=int(kernel_size),
                padding=padding,
                groups=in_channels,
            )
        )
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(in_channels))
        layers.append(nn.ReLU())
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
    return nn.Sequential(*layers)


class WindowedCNNEncoder(nn.Module):
    """Windowed sensor-wise CNN encoder with window-axis max pooling.

    Revised behavior (compared to the original implementation kept in
    ``windowed_cnn_encoder_original.py``):

    - Preserves sensor channels during the first CNN stage via depthwise conv.
    - Pools over the window axis (K) to pick the strongest activation per sensor.
    - Produces a per-run 1D representation over time: ``[B, W']``.

    Pipeline summary:

    1) ``(B,C,K,W) -> (B,C,K,W')`` depthwise temporal conv per window
    2) ``(B,C,K,W') -> (B,C,1,W')`` max over K
    3) ``(B,C,1,W') -> (B,1,1,W')`` pointwise (1x1) conv across channels
    4) ``(B,1,1,W') -> (B,W')`` squeeze

    Accepted input layouts:

    - ``[B, C, K, W]`` (preferred)
    - ``[B, K, C, W]`` (legacy; auto-detected)
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
        self.sensor_cnn = _make_cnn(input_channels, channels, kernel_size, use_batch_norm, dropout)
        # Mix sensors into a single-channel temporal representation.
        self.mix_cnn = nn.Sequential(
            nn.Conv1d(self.input_channels, 1, kernel_size=1, padding=0),
            nn.ReLU(),
        )
        # Depthwise conv uses padding that preserves time length.
        self.output_length = int(self.window_length)

    @property
    def output_dim(self) -> int:
        """Dimensionality of the produced per-run vector representation."""

        return int(self.output_length)

    def _normalize_input_layout(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x to [B, C, K, W] and validate expected sizes."""

        if x.ndim != 4:
            raise ValueError(f"WindowedCNNEncoder expects a 4D tensor, got {tuple(x.shape)}")
        b, d1, d2, w = x.shape
        if w != self.window_length:
            raise ValueError(f"Expected window_length={self.window_length}, got {w}")

        # Preferred: [B, C, K, W]
        if d1 == self.input_channels and d2 == self.num_windows:
            return x
        # Legacy: [B, K, C, W]
        if d1 == self.num_windows and d2 == self.input_channels:
            return x.permute(0, 2, 1, 3)

        raise ValueError(
            "WindowedCNNEncoder expects [B,C,K,W] or [B,K,C,W] with "
            f"C={self.input_channels}, K={self.num_windows}, W={self.window_length}; got {tuple(x.shape)}"
        )

    def forward(self, x: torch.Tensor, return_window_latents: bool = False):
        x = self._normalize_input_layout(x)  # [B, C, K, W]
        batch_size, num_channels, num_windows, window_length = x.shape

        # Apply the same sensor-wise CNN to each window independently.
        x_flat = x.permute(0, 2, 1, 3).reshape(batch_size * num_windows, num_channels, window_length)
        h_flat = self.sensor_cnn(x_flat)  # [B*K, C, W']
        _, _, w_out = h_flat.shape
        if w_out != self.output_length:
            # Keep a strict check so downstream dims remain predictable.
            raise RuntimeError(f"Unexpected output_length={w_out}, expected {self.output_length}")

        h = h_flat.reshape(batch_size, num_windows, num_channels, w_out).permute(0, 2, 1, 3)  # [B, C, K, W']
        # Window-axis max pooling: select strongest activation among windows per sensor.
        h_max = h.max(dim=2, keepdim=True).values  # [B, C, 1, W']

        # Mix sensors into a single temporal representation and squeeze.
        y = self.mix_cnn(h_max.squeeze(2)).squeeze(1)  # [B, W']

        if return_window_latents:
            # Provide a per-window temporal representation for diagnostics/tests:
            # mean over sensors => [B, K, W'].
            window_latents = h.mean(dim=1).permute(0, 1, 2)  # [B, K, W']
            return y, window_latents
        return y


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

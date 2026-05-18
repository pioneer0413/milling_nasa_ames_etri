from __future__ import annotations

from typing import Any

import torch
from torch import nn

from milling_experiment_framework.models.dl.cnn1d import CNN1DNet


class CNN1DOnlyRegressor(nn.Module):
    """Single-run regressor using full-length temporal CNN (no windowing)."""

    def __init__(
        self,
        input_channels: int = 6,
        hidden_dim: int = 64,
        head_hidden_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.cnn1d = CNN1DNet(
            num_channels=int(input_channels),
            num_classes=1,
            hidden_dim=int(hidden_dim),
            task_type="regression",
        )
        # Note: CNN1DNet already includes the final Linear(hidden_dim -> 1)
        # and outputs a single scalar, so we don't need an additional head.

    @classmethod
    def from_config(cls, config: dict[str, Any], input_channels: int) -> "CNN1DOnlyRegressor":
        model_cfg = config.get("model_design", {}).get("cnn1d_only", {})
        legacy = config.get("dl_model", {})
        return cls(
            input_channels=int(input_channels),
            hidden_dim=int(model_cfg.get("hidden_dim", legacy.get("cnn_hidden_dim", 64))),
            head_hidden_dim=int(model_cfg.get("head_hidden_dim", legacy.get("regressor_hidden_dim", 32))),
            dropout=float(model_cfg.get("dropout", legacy.get("dropout", 0.0))),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, W] (batch, channels, time)
        # or [B, K, C, W] if called with windowed input
        # For windowed input, aggregate windows via max-pooling over K dimension
        if x.ndim == 4:
            # Windowed input: [B, K, C, W] -> max pool over K -> [B, C, W]
            b, k, c, w = x.shape
            # Reshape to [B, K, C, W] -> [B, C, K, W] for maxpool
            x = x.permute(0, 2, 1, 3)  # [B, C, K, W]
            # Max pool over K (dimension 2)
            x = torch.nn.functional.max_pool2d(x, kernel_size=(k, 1)).squeeze(2)  # [B, C, W]
        elif x.ndim != 3:
            raise ValueError(f"CNN1DOnlyRegressor expects [B,C,W] or [B,K,C,W], got {tuple(x.shape)}")

        return self.cnn1d(x)

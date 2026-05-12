from __future__ import annotations

from typing import Any

import torch
from torch import nn

from milling_experiment_framework.models.dl.windowed_cnn_encoder import (
    RegressionHead,
    WindowedCNNEncoder,
    make_windowed_encoder_from_config,
)


class CNNOnlyRegressor(nn.Module):
    """Current-run regressor using a shared windowed CNN run encoder."""

    def __init__(
        self,
        encoder: WindowedCNNEncoder,
        latent_dim: int,
        head_hidden_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = RegressionHead(latent_dim, head_hidden_dim, dropout=dropout)

    @classmethod
    def from_config(cls, config: dict[str, Any], input_channels: int) -> "CNNOnlyRegressor":
        encoder = make_windowed_encoder_from_config(config, input_channels=input_channels)
        model_cfg = config.get("model_design", {}).get("cnn_only", {})
        legacy = config.get("dl_model", {})
        return cls(
            encoder=encoder,
            latent_dim=int(encoder.config.latent_dim),
            head_hidden_dim=int(model_cfg.get("head_hidden_dim", legacy.get("regressor_hidden_dim", 32))),
            dropout=float(model_cfg.get("dropout", legacy.get("dropout", 0.0))),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, C, W]
        z = self.encoder(x)
        return self.head(z)

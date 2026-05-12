from __future__ import annotations

from typing import Any

import torch
from torch import nn

from milling_experiment_framework.models.dl.windowed_cnn_encoder import (
    RegressionHead,
    WindowedCNNEncoder,
    make_windowed_encoder_from_config,
)


class CNNLagConcatRegressor(nn.Module):
    """Run-sequence baseline that concatenates windowed CNN latents."""

    def __init__(
        self,
        encoder: WindowedCNNEncoder,
        latent_dim: int,
        sequence_size: int = 3,
        include_mask: bool = True,
        head_hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.sequence_size = int(sequence_size)
        self.include_mask = bool(include_mask)
        input_dim = self.sequence_size * int(latent_dim) + (self.sequence_size if self.include_mask else 0)
        self.head = RegressionHead(input_dim, head_hidden_dim, dropout=dropout)

    @classmethod
    def from_config(cls, config: dict[str, Any], input_channels: int) -> "CNNLagConcatRegressor":
        encoder = make_windowed_encoder_from_config(config, input_channels=input_channels)
        seq_size = int(config.get("sequence", {}).get("sequence_size", 3))
        model_cfg = config.get("model_design", {}).get("cnn_lag_concat", {})
        legacy_models = config.get("models", {}).get("cnn_lag_concat", {})
        legacy = config.get("dl_model", {})
        return cls(
            encoder=encoder,
            latent_dim=int(encoder.config.latent_dim),
            sequence_size=int(model_cfg.get("sequence_size", seq_size)),
            include_mask=bool(model_cfg.get("include_mask", legacy_models.get("include_mask", True))),
            head_hidden_dim=int(model_cfg.get("head_hidden_dim", legacy.get("regressor_hidden_dim", 64))),
            dropout=float(model_cfg.get("dropout", legacy.get("dropout", 0.0))),
        )

    def forward(self, x_seq: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x_seq: [B, S, K, C, W], mask: [B, S]
        if x_seq.ndim != 5:
            raise ValueError(f"CNNLagConcatRegressor expects [B, S, K, C, W], got {tuple(x_seq.shape)}")
        b, s, k, c, w = x_seq.shape
        if s != self.sequence_size:
            raise ValueError(f"Expected sequence_size={self.sequence_size}, got {s}")
        if mask is None:
            mask = torch.ones((b, s), dtype=x_seq.dtype, device=x_seq.device)
        if mask.shape != (b, s):
            raise ValueError(f"Expected mask shape {(b, s)}, got {tuple(mask.shape)}")

        z = self.encoder(x_seq.reshape(b * s, k, c, w)).reshape(b, s, -1)
        z = z * mask.unsqueeze(-1)
        flat = z.reshape(b, -1)
        if self.include_mask:
            flat = torch.cat([flat, mask.to(dtype=flat.dtype)], dim=1)
        return self.head(flat)

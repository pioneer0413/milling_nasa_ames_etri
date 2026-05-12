from __future__ import annotations

from typing import Any

import torch
from torch import nn

from milling_experiment_framework.models.dl.windowed_cnn_encoder import (
    RegressionHead,
    WindowedCNNEncoder,
    make_windowed_encoder_from_config,
)


class CNNGRURegressor(nn.Module):
    """Proposed run-sequence model over windowed CNN run-level latents."""

    def __init__(
        self,
        encoder: WindowedCNNEncoder,
        latent_dim: int,
        sequence_size: int = 3,
        gru_hidden_size: int = 32,
        gru_num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        head_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if bidirectional:
            raise ValueError("CNNGRURegressor uses a unidirectional GRU for chronological consistency.")
        self.encoder = encoder
        self.sequence_size = int(sequence_size)
        self.gru = nn.GRU(
            input_size=int(latent_dim),
            hidden_size=int(gru_hidden_size),
            num_layers=int(gru_num_layers),
            dropout=float(dropout) if int(gru_num_layers) > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.head = RegressionHead(int(gru_hidden_size), int(head_hidden_dim), dropout=0.0)

    @classmethod
    def from_config(cls, config: dict[str, Any], input_channels: int) -> "CNNGRURegressor":
        encoder = make_windowed_encoder_from_config(config, input_channels=input_channels)
        seq_size = int(config.get("sequence", {}).get("sequence_size", 3))
        model_cfg = config.get("model_design", {}).get("cnn_gru", {})
        legacy = config.get("dl_model", {})
        return cls(
            encoder=encoder,
            latent_dim=int(encoder.config.latent_dim),
            sequence_size=int(model_cfg.get("sequence_size", seq_size)),
            gru_hidden_size=int(model_cfg.get("gru_hidden_size", legacy.get("gru_hidden_size", 32))),
            gru_num_layers=int(model_cfg.get("gru_num_layers", legacy.get("gru_num_layers", 1))),
            bidirectional=bool(model_cfg.get("bidirectional", False)),
            dropout=float(model_cfg.get("dropout", legacy.get("dropout", 0.0))),
            head_hidden_dim=int(model_cfg.get("head_hidden_dim", legacy.get("regressor_hidden_dim", 32))),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x_seq: [B, S, K, C, W]. With left padding, the last step is always current run.
        if x_seq.ndim != 5:
            raise ValueError(f"CNNGRURegressor expects [B, S, K, C, W], got {tuple(x_seq.shape)}")
        b, s, k, c, w = x_seq.shape
        if s != self.sequence_size:
            raise ValueError(f"Expected sequence_size={self.sequence_size}, got {s}")
        if mask is None:
            mask = torch.ones((b, s), dtype=x_seq.dtype, device=x_seq.device)
        if mask.shape != (b, s):
            raise ValueError(f"Expected mask shape {(b, s)}, got {tuple(mask.shape)}")

        z = self.encoder(x_seq.reshape(b * s, k, c, w)).reshape(b, s, -1)
        z = z * mask.unsqueeze(-1)
        out, _ = self.gru(z)
        current_hidden = out[:, -1, :]
        return self.head(current_hidden)

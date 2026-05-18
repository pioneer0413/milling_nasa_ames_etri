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
    """Proposed run-sequence model over windowed CNN run-level latents.
    
    Supports GRU, LSTM, and RNN as sequence encoders.
    """

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
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()
        if bidirectional:
            raise ValueError("CNNGRURegressor uses a unidirectional RNN for chronological consistency.")
        self.encoder = encoder
        self.sequence_size = int(sequence_size)
        self.rnn_type = str(rnn_type).lower()

        if self.rnn_type not in {"gru", "lstm", "rnn"}:
            raise ValueError(f"rnn_type must be 'gru', 'lstm', or 'rnn', got '{self.rnn_type}'")

        # Encoder now produces a (B, W') vector; embed to a compact latent_dim before RNN.
        self.embed = nn.Sequential(nn.Linear(int(self.encoder.output_dim), int(latent_dim)), nn.ReLU())
        
        rnn_class = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[self.rnn_type]
        self.rnn = rnn_class(
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
            rnn_type=str(model_cfg.get("rnn_type", "gru")),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x_seq: [B, S, C, K, W] (preferred) or [B, S, K, C, W] (legacy).
        # With left padding, the last step is always current run.
        if x_seq.ndim != 5:
            raise ValueError(f"CNNGRURegressor expects [B, S, K, C, W], got {tuple(x_seq.shape)}")
        b, s, d2, d3, w = x_seq.shape
        if s != self.sequence_size:
            raise ValueError(f"Expected sequence_size={self.sequence_size}, got {s}")
        if mask is None:
            mask = torch.ones((b, s), dtype=x_seq.dtype, device=x_seq.device)
        if mask.shape != (b, s):
            raise ValueError(f"Expected mask shape {(b, s)}, got {tuple(mask.shape)}")

        if d2 == self.encoder.input_channels and d3 == self.encoder.num_windows:
            x_flat = x_seq.reshape(b * s, d2, d3, w)  # [B*S, C, K, W]
        elif d2 == self.encoder.num_windows and d3 == self.encoder.input_channels:
            x_flat = x_seq.reshape(b * s, d2, d3, w)  # [B*S, K, C, W]
        else:
            raise ValueError(
                "x_seq must be [B,S,C,K,W] or [B,S,K,C,W] with "
                f"C={self.encoder.input_channels}, K={self.encoder.num_windows}; got {tuple(x_seq.shape)}"
            )

        z_raw = self.encoder(x_flat).reshape(b, s, -1)  # [B, S, W']
        z = self.embed(z_raw)  # [B, S, latent_dim]
        z = z * mask.unsqueeze(-1)
        rnn_output = self.rnn(z)
        # Handle LSTM output (tuple) vs GRU/RNN output (tensor)
        out = rnn_output[0] if isinstance(rnn_output, tuple) else rnn_output
        current_hidden = out[:, -1, :]
        return self.head(current_hidden)

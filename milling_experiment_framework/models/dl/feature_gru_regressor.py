from __future__ import annotations

from typing import Any

import torch
from torch import nn

from milling_experiment_framework.registry.model_registry import MODEL_REGISTRY


class FeatureGRURegressor(nn.Module):
    """GRU regressor over handcrafted run-level feature sequences.
    
    Supports GRU, LSTM, and RNN as sequence encoders.
    """

    def __init__(
        self,
        input_dim: int,
        gru_hidden_size: int = 32,
        gru_num_layers: int = 1,
        regression_head_hidden_dim: int = 32,
        dropout: float = 0.0,
        bidirectional: bool = False,
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError(f"feature_gru requires input_dim > 0, got {input_dim}")
        if bidirectional:
            raise ValueError("feature_gru uses a unidirectional RNN for chronological consistency.")
        self.input_dim = int(input_dim)
        self.gru_hidden_size = int(gru_hidden_size)
        self.gru_num_layers = int(gru_num_layers)
        self.rnn_type = str(rnn_type).lower()

        if self.rnn_type not in {"gru", "lstm", "rnn"}:
            raise ValueError(f"rnn_type must be 'gru', 'lstm', or 'rnn', got '{self.rnn_type}'")

        rnn_class = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[self.rnn_type]
        self.rnn = rnn_class(
            input_size=self.input_dim,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            dropout=float(dropout) if self.gru_num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(self.gru_hidden_size, int(regression_head_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(regression_head_hidden_dim), 1),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any], input_dim: int) -> "FeatureGRURegressor":
        model_cfg = config.get("feature_gru", {}).get("model", {})
        return cls(
            input_dim=int(input_dim),
            gru_hidden_size=int(model_cfg.get("gru_hidden_size", 32)),
            gru_num_layers=int(model_cfg.get("gru_num_layers", 1)),
            regression_head_hidden_dim=int(model_cfg.get("regression_head_hidden_dim", 32)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            bidirectional=bool(model_cfg.get("bidirectional", False)),
            rnn_type=str(model_cfg.get("rnn_type", "gru")),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict scalar target from ``x_seq`` with shape ``[B, S, F]``."""
        if x_seq.ndim != 3:
            raise ValueError(f"feature_gru expects x_seq [B, S, F], got {tuple(x_seq.shape)}")
        batch_size, sequence_size, feature_dim = x_seq.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                "feature_gru feature dimension mismatch: "
                f"expected input_dim={self.input_dim}, got x_seq.shape={tuple(x_seq.shape)}"
            )
        if mask is None:
            mask = torch.ones((batch_size, sequence_size), dtype=x_seq.dtype, device=x_seq.device)
        if mask.shape != (batch_size, sequence_size):
            raise ValueError(f"feature_gru expected mask shape {(batch_size, sequence_size)}, got {tuple(mask.shape)}")

        # Left padding means the last timestep is always the current run.
        x_seq = x_seq * mask.unsqueeze(-1)
        rnn_output = self.rnn(x_seq)
        # Handle LSTM output (tuple) vs GRU/RNN output (tensor)
        output = rnn_output[0] if isinstance(rnn_output, tuple) else rnn_output
        current_hidden = output[:, -1, :]
        return self.head(current_hidden).view(-1)


@MODEL_REGISTRY.register("feature_gru")
def create_feature_gru_from_registry(model_config: dict[str, Any], task_config: dict[str, Any]) -> FeatureGRURegressor:
    input_dim = model_config.get("input_dim") or task_config.get("input_dim") or task_config.get("feature_dim")
    if input_dim in {None, "auto"}:
        raise ValueError("feature_gru registry creation requires resolved input_dim.")
    merged = {"feature_gru": {"model": {**model_config, "input_dim": input_dim}}}
    return FeatureGRURegressor.from_config(merged, input_dim=int(input_dim))

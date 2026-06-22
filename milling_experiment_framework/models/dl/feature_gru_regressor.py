from __future__ import annotations

from typing import Any

import torch
from torch import nn

from milling_experiment_framework.registry.model_registry import MODEL_REGISTRY


class MetadataFiLMLayer(nn.Module):
    """Apply FiLM only to metadata features while preserving signal features."""

    def __init__(
        self,
        input_dim: int,
        metadata_feature_dim: int = 3,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.metadata_feature_dim = int(metadata_feature_dim)
        if self.metadata_feature_dim <= 0:
            raise ValueError(f"metadata_feature_dim must be > 0, got {metadata_feature_dim}")
        if self.metadata_feature_dim >= self.input_dim:
            raise ValueError(
                "metadata_feature_dim must be smaller than input_dim so signal features remain unmodified: "
                f"input_dim={input_dim}, metadata_feature_dim={metadata_feature_dim}"
            )
        self.signal_feature_dim = self.input_dim - self.metadata_feature_dim
        film_hidden_dim = int(hidden_dim or max(8, self.metadata_feature_dim * 4))
        layers: list[nn.Module] = [
            nn.Linear(self.signal_feature_dim, film_hidden_dim),
            nn.ReLU(),
        ]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(film_hidden_dim, self.metadata_feature_dim * 2))
        self.generator = nn.Sequential(*layers)
        last = self.generator[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.shape[-1] != self.input_dim:
            raise ValueError(
                "metadata FiLM feature dimension mismatch: "
                f"expected input_dim={self.input_dim}, got x_seq.shape={tuple(x_seq.shape)}"
            )
        signal_features = x_seq[..., : self.signal_feature_dim]
        metadata_features = x_seq[..., self.signal_feature_dim :]
        gamma_beta = self.generator(signal_features)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        modulated_metadata = metadata_features * (1.0 + gamma) + beta
        return torch.cat([signal_features, modulated_metadata], dim=-1)


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


class FeatureFiLMGRURegressor(FeatureGRURegressor):
    """Feature GRU with FiLM applied only to the trailing metadata features."""

    def __init__(
        self,
        input_dim: int,
        gru_hidden_size: int = 32,
        gru_num_layers: int = 1,
        regression_head_hidden_dim: int = 32,
        dropout: float = 0.0,
        bidirectional: bool = False,
        rnn_type: str = "gru",
        metadata_feature_dim: int = 3,
        film_hidden_dim: int | None = None,
        film_dropout: float = 0.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            regression_head_hidden_dim=regression_head_hidden_dim,
            dropout=dropout,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
        )
        self.metadata_feature_dim = int(metadata_feature_dim)
        self.film = MetadataFiLMLayer(
            input_dim=self.input_dim,
            metadata_feature_dim=self.metadata_feature_dim,
            hidden_dim=film_hidden_dim,
            dropout=film_dropout,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any], input_dim: int) -> "FeatureFiLMGRURegressor":
        model_cfg = config.get("feature_film_gru", {}).get("model", {})
        if not model_cfg:
            model_cfg = config.get("feature_gru", {}).get("model", {})
        return cls(
            input_dim=int(input_dim),
            gru_hidden_size=int(model_cfg.get("gru_hidden_size", 32)),
            gru_num_layers=int(model_cfg.get("gru_num_layers", 1)),
            regression_head_hidden_dim=int(model_cfg.get("regression_head_hidden_dim", 32)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            bidirectional=bool(model_cfg.get("bidirectional", False)),
            rnn_type=str(model_cfg.get("rnn_type", "gru")),
            metadata_feature_dim=int(model_cfg.get("metadata_feature_dim", model_cfg.get("meta_dim", 3))),
            film_hidden_dim=(
                None
                if model_cfg.get("film_hidden_dim") in {None, "auto"}
                else int(model_cfg.get("film_hidden_dim"))
            ),
            film_dropout=float(model_cfg.get("film_dropout", 0.0)),
        )

    def apply_metadata_film(self, x_seq: torch.Tensor) -> torch.Tensor:
        return self.film(x_seq)

    def forward(
        self,
        x_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_seq = self.apply_metadata_film(x_seq)
        return super().forward(x_seq, mask=mask, lengths=lengths)


@MODEL_REGISTRY.register("feature_gru")
def create_feature_gru_from_registry(model_config: dict[str, Any], task_config: dict[str, Any]) -> FeatureGRURegressor:
    input_dim = model_config.get("input_dim") or task_config.get("input_dim") or task_config.get("feature_dim")
    if input_dim in {None, "auto"}:
        raise ValueError("feature_gru registry creation requires resolved input_dim.")
    merged = {"feature_gru": {"model": {**model_config, "input_dim": input_dim}}}
    return FeatureGRURegressor.from_config(merged, input_dim=int(input_dim))


@MODEL_REGISTRY.register("feature_film_gru")
def create_feature_film_gru_from_registry(
    model_config: dict[str, Any],
    task_config: dict[str, Any],
) -> FeatureFiLMGRURegressor:
    input_dim = model_config.get("input_dim") or task_config.get("input_dim") or task_config.get("feature_dim")
    if input_dim in {None, "auto"}:
        raise ValueError("feature_film_gru registry creation requires resolved input_dim.")
    merged = {"feature_film_gru": {"model": {**model_config, "input_dim": input_dim}}}
    return FeatureFiLMGRURegressor.from_config(merged, input_dim=int(input_dim))

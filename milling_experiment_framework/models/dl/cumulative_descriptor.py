from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CumulativeDescriptor:
    """Deterministic cumulative descriptor for sequence arrays.

    For each sample and channel, this computes
    c(n, k) = cumulative_sum(x[:n, k]) / sqrt(abs(cumulative_sum(x[:n, k])) + eps).
    The transform is applied within each sample only, so it does not use other
    runs, future samples, target labels, or domain-level statistics.
    """

    def __init__(self, epsilon: float = 1e-8, input_representation: str = "cumulative"):
        self.epsilon = float(epsilon)
        self.input_representation = input_representation
        if input_representation not in {"raw", "cumulative", "raw_plus_cumulative"}:
            raise ValueError(f"Unsupported input_representation: {input_representation}")

    def transform_sequence(self, x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype="float32")
        if array.ndim not in {2, 3}:
            raise ValueError(f"Expected sequence shape (length, channels) or (batch, length, channels), got {array.shape}")
        cumulative = np.cumsum(array, axis=-2)
        descriptor = cumulative / np.sqrt(np.abs(cumulative) + self.epsilon)
        if self.input_representation == "raw":
            out = array
        elif self.input_representation == "cumulative":
            out = descriptor
        else:
            out = np.concatenate([array, descriptor], axis=-1)
        if not np.isfinite(out).all():
            raise ValueError("Cumulative descriptor produced NaN or inf values.")
        return out.astype("float32")


class CumulativeDescriptorLayer(nn.Module):
    """Torch implementation of the cumulative descriptor."""

    def __init__(self, epsilon: float = 1e-8, input_representation: str = "cumulative"):
        super().__init__()
        self.epsilon = float(epsilon)
        self.input_representation = input_representation
        if input_representation not in {"raw", "cumulative", "raw_plus_cumulative"}:
            raise ValueError(f"Unsupported input_representation: {input_representation}")

    def forward(self, sensor_sequence: torch.Tensor) -> torch.Tensor:
        if sensor_sequence.ndim != 3:
            raise ValueError(f"sensor_sequence must be (batch, sequence_length, channels), got {tuple(sensor_sequence.shape)}")
        cumulative = torch.cumsum(sensor_sequence, dim=1)
        descriptor = cumulative / torch.sqrt(torch.abs(cumulative) + self.epsilon)
        if self.input_representation == "raw":
            out = sensor_sequence
        elif self.input_representation == "cumulative":
            out = descriptor
        else:
            out = torch.cat([sensor_sequence, descriptor], dim=2)
        if not torch.isfinite(out).all():
            raise ValueError("Cumulative descriptor produced NaN or inf values.")
        return out


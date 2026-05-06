from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetOutput:
    X: np.ndarray | None = None
    y: np.ndarray | None = None
    metadata: pd.DataFrame | None = None
    X_timeseries: np.ndarray | None = None
    X_features: np.ndarray | None = None
    sensor_channels: list[str] = field(default_factory=list)
    dataset_info: dict[str, Any] = field(default_factory=dict)

    def primary_X(self) -> np.ndarray:
        if self.X is not None:
            return self.X
        if self.X_timeseries is not None:
            return self.X_timeseries
        raise ValueError("DatasetOutput has no primary X array.")

    def replace_primary_X(self, X: np.ndarray) -> None:
        if self.X is not None:
            self.X = X
        elif self.X_timeseries is not None:
            self.X_timeseries = X
        else:
            raise ValueError("DatasetOutput has no primary X array.")


@dataclass
class SplitOutput:
    assignments: pd.DataFrame
    summary: dict[str, Any]
    leakage_check: dict[str, Any]


@dataclass
class PreprocessOutput:
    data: DatasetOutput
    summary: dict[str, Any]
    artifacts: dict[str, Any]
    shape_trace: list[dict[str, Any]]

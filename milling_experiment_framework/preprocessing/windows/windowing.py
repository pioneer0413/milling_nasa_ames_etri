from __future__ import annotations

import numpy as np

from milling_experiment_framework.registry.preprocessor_registry import PREPROCESSOR_REGISTRY


@PREPROCESSOR_REGISTRY.register("windowing")
class WindowingStep:
    def __init__(self, config: dict):
        self.config = config
        self.params: dict = {}

    def fit_transform(self, data, split_col: str = "split"):
        X = data.primary_X()
        if X.ndim != 3:
            raise ValueError("windowing requires timeseries X with shape [num_samples, num_channels, sequence_length]")
        window_size = int(self.config.get("window_size", 128))
        stride = int(self.config.get("stride", window_size))
        if window_size > X.shape[2]:
            raise ValueError(f"window_size={window_size} exceeds sequence_length={X.shape[2]}")
        windows = []
        labels = []
        rows = []
        for sample_idx in range(X.shape[0]):
            start_positions = range(0, X.shape[2] - window_size + 1, stride)
            for window_idx, start in enumerate(start_positions):
                end = start + window_size
                windows.append(X[sample_idx, :, start:end])
                labels.append(data.y[sample_idx])
                row = data.metadata.iloc[sample_idx].copy()
                row["parent_sample_id"] = row["sample_id"]
                row["sample_id"] = f"{row['sample_id']}_w{window_idx:03d}"
                row["window_start"] = start
                row["window_end"] = end
                rows.append(row)
        import pandas as pd

        data.replace_primary_X(np.asarray(windows, dtype="float32"))
        data.y = np.asarray(labels)
        data.metadata = pd.DataFrame(rows).reset_index(drop=True)
        self.params = {
            "window_size": window_size,
            "stride": stride,
            "overlap": max(0, window_size - stride),
            "num_windows": int(len(windows)),
        }
        return data

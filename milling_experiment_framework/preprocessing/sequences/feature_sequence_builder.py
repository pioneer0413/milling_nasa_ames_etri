from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureSequenceOutput:
    x_seq: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    metadata: pd.DataFrame


class FeatureSequenceBuilder:
    """Build left-padded run-level handcrafted feature sequences per case."""

    def __init__(
        self,
        sequence_size: int = 3,
        padding_value: float = 0.0,
        allow_cross_case_sequence: bool = False,
    ) -> None:
        self.sequence_size = int(sequence_size)
        self.padding_value = float(padding_value)
        self.allow_cross_case_sequence = bool(allow_cross_case_sequence)
        if self.sequence_size < 1:
            raise ValueError(f"sequence_size must be >= 1, got {sequence_size}")
        if self.allow_cross_case_sequence:
            raise ValueError("feature_gru does not allow cross-case feature sequences.")

    def build_sequences(
        self,
        feature_matrix: np.ndarray,
        metadata_df: pd.DataFrame,
        indices: np.ndarray | list[int] | None = None,
    ) -> FeatureSequenceOutput:
        features = np.asarray(feature_matrix, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"feature_matrix must be [N, F], got {features.shape}")
        if features.shape[1] <= 0:
            raise ValueError("feature_matrix must have feature_dim > 0")
        if len(metadata_df) != features.shape[0]:
            raise ValueError(f"metadata rows {len(metadata_df)} do not match feature rows {features.shape[0]}")
        for col in ["case_id", "run_order"]:
            if col not in metadata_df.columns:
                raise ValueError(f"metadata_df missing required column: {col}")

        selected = np.asarray(indices if indices is not None else metadata_df.index.to_numpy(), dtype=int)
        position_by_case_run = {
            (int(row.case_id), int(row.run_order)): int(idx)
            for idx, row in metadata_df[["case_id", "run_order"]].iterrows()
        }
        x_out: list[np.ndarray] = []
        masks: list[list[float]] = []
        lengths: list[int] = []
        for idx in selected:
            row = metadata_df.loc[int(idx)]
            case_id = int(row["case_id"])
            order = int(row["run_order"])
            seq_parts = []
            mask = []
            for lag in range(self.sequence_size - 1, -1, -1):
                prev_order = order - lag
                prev_idx = position_by_case_run.get((case_id, prev_order))
                if prev_idx is None or prev_order < 1:
                    seq_parts.append(np.full(features.shape[1], self.padding_value, dtype=np.float32))
                    mask.append(0.0)
                else:
                    seq_parts.append(features[prev_idx].astype(np.float32, copy=False))
                    mask.append(1.0)
            x_out.append(np.stack(seq_parts, axis=0))
            masks.append(mask)
            lengths.append(int(sum(mask)))
        return FeatureSequenceOutput(
            x_seq=np.stack(x_out).astype(np.float32),
            mask=np.asarray(masks, dtype=np.float32),
            lengths=np.asarray(lengths, dtype=np.int64),
            metadata=metadata_df.loc[selected].copy(),
        )


def fit_transform_feature_preprocessor(
    feature_matrix: np.ndarray,
    train_indices: np.ndarray | list[int],
    imputer_strategy: str = "median",
) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    """Fit imputer/scaler on source train rows only, then transform all rows."""
    features = np.asarray(feature_matrix, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"feature_matrix must be [N, F], got {features.shape}")
    train_idx = np.asarray(train_indices, dtype=int)
    if train_idx.size == 0:
        raise ValueError("Cannot fit feature preprocessor with empty source train indices.")
    imputer = SimpleImputer(strategy=imputer_strategy)
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(features[train_idx])
    scaler.fit(train_imputed)
    transformed = scaler.transform(imputer.transform(features))
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(transformed, -10.0, 10.0).astype(np.float32), imputer, scaler

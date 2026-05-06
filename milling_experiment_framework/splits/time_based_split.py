from __future__ import annotations

from typing import Any

import pandas as pd

from milling_experiment_framework.core.types import SplitOutput
from milling_experiment_framework.registry.split_registry import SPLIT_REGISTRY
from milling_experiment_framework.splits.leakage import leakage_check
from milling_experiment_framework.splits.random_split import _summary


@SPLIT_REGISTRY.register("time_based")
class TimeBasedSplit:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def split(self, metadata: pd.DataFrame, seed: int) -> SplitOutput:
        sort_col = "timestamp" if "timestamp" in metadata.columns else "sequence_index"
        assignments = metadata.sort_values(sort_col).copy()
        n = len(assignments)
        test_ratio = float(self.config.get("test_ratio", 0.2))
        val_ratio = float(self.config.get("validation_ratio", 0.2))
        train_end = int(n * (1 - test_ratio - val_ratio))
        val_end = int(n * (1 - test_ratio))
        assignments["split"] = "test"
        assignments.iloc[:train_end, assignments.columns.get_loc("split")] = "train"
        assignments.iloc[train_end:val_end, assignments.columns.get_loc("split")] = "validation"
        assignments = assignments.sort_index()
        return SplitOutput(assignments, _summary(assignments, "time_based", None, seed), leakage_check(assignments))

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from milling_experiment_framework.core.types import SplitOutput
from milling_experiment_framework.registry.split_registry import SPLIT_REGISTRY
from milling_experiment_framework.splits.leakage import leakage_check
from milling_experiment_framework.splits.random_split import _summary


@SPLIT_REGISTRY.register("dataset_run_wise")
class DatasetRunWiseSplit:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def split(self, metadata: pd.DataFrame, seed: int) -> SplitOutput:
        group_key = self.config.get("group_key", "dataset_run_id")
        if group_key not in metadata.columns:
            raise ValueError(f"metadata missing group_key={group_key}")
        rng = np.random.default_rng(seed)
        groups = np.array(sorted(metadata[group_key].astype(str).unique()))
        rng.shuffle(groups)
        val_ratio = float(self.config.get("validation_ratio", 0.2))
        test_ratio = float(self.config.get("test_ratio", 0.2))
        n_test = max(1, int(round(len(groups) * test_ratio)))
        n_val = max(1, int(round(len(groups) * val_ratio)))
        test_groups = set(groups[:n_test])
        val_groups = set(groups[n_test : n_test + n_val])
        assignments = metadata.copy()
        assignments["split"] = "train"
        assignments.loc[assignments[group_key].astype(str).isin(val_groups), "split"] = "validation"
        assignments.loc[assignments[group_key].astype(str).isin(test_groups), "split"] = "test"
        return SplitOutput(
            assignments=assignments,
            summary=_summary(assignments, "dataset_run_wise", group_key, seed),
            leakage_check=leakage_check(assignments, group_key),
        )

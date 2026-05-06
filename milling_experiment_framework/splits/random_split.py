from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from milling_experiment_framework.core.types import SplitOutput
from milling_experiment_framework.registry.split_registry import SPLIT_REGISTRY
from milling_experiment_framework.splits.leakage import leakage_check


@SPLIT_REGISTRY.register("random")
class RandomSplit:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def split(self, metadata: pd.DataFrame, seed: int) -> SplitOutput:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(metadata))
        rng.shuffle(idx)
        val_ratio = float(self.config.get("validation_ratio", 0.2))
        test_ratio = float(self.config.get("test_ratio", 0.2))
        n_test = int(round(len(idx) * test_ratio))
        n_val = int(round(len(idx) * val_ratio))
        test_idx = idx[:n_test]
        val_idx = idx[n_test : n_test + n_val]
        train_idx = idx[n_test + n_val :]
        assignments = metadata.copy()
        assignments["split"] = "train"
        assignments.loc[val_idx, "split"] = "validation"
        assignments.loc[test_idx, "split"] = "test"
        return SplitOutput(
            assignments=assignments,
            summary=_summary(assignments, "random", None, seed),
            leakage_check=leakage_check(assignments),
        )


def _summary(assignments: pd.DataFrame, strategy: str, group_key: str | None, seed: int) -> dict:
    label_counts = (
        assignments.groupby(["split", "label"], dropna=False).size().reset_index(name="count").to_dict(orient="records")
    )
    return {
        "strategy": strategy,
        "group_key": group_key,
        "seed": seed,
        "split_counts": assignments["split"].value_counts().to_dict(),
        "label_distribution": label_counts,
    }

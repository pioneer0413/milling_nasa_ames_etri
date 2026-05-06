from __future__ import annotations

import milling_experiment_framework.splits.dataset_run_wise_split  # noqa: F401
import milling_experiment_framework.splits.condition_wise_split  # noqa: F401
import milling_experiment_framework.splits.machine_wise_split  # noqa: F401
import milling_experiment_framework.splits.random_split  # noqa: F401
import milling_experiment_framework.splits.time_based_split  # noqa: F401
import milling_experiment_framework.splits.tool_wise_split  # noqa: F401
from milling_experiment_framework.core.types import SplitOutput
from milling_experiment_framework.registry.split_registry import SPLIT_REGISTRY


def create_split(metadata, split_config: dict, seed: int) -> SplitOutput:
    strategy = split_config.get("strategy", "random")
    split_cls = SPLIT_REGISTRY.get(strategy)
    return split_cls(split_config).split(metadata, seed)

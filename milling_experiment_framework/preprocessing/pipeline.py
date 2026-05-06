from __future__ import annotations

from typing import Any

import milling_experiment_framework.preprocessing.features.statistical  # noqa: F401
import milling_experiment_framework.preprocessing.normalization.zscore  # noqa: F401
import milling_experiment_framework.preprocessing.steady_cut.modes  # noqa: F401
import milling_experiment_framework.preprocessing.windows.windowing  # noqa: F401
from milling_experiment_framework.core.types import DatasetOutput, PreprocessOutput
from milling_experiment_framework.registry.preprocessor_registry import PREPROCESSOR_REGISTRY


def _shape(data: DatasetOutput) -> dict[str, Any]:
    shape = {
        "X": list(data.X.shape) if data.X is not None else None,
        "X_timeseries": list(data.X_timeseries.shape) if data.X_timeseries is not None else None,
        "X_features": list(data.X_features.shape) if data.X_features is not None else None,
        "y": list(data.y.shape) if data.y is not None else None,
        "metadata": list(data.metadata.shape) if data.metadata is not None else None,
    }
    return shape


class PreprocessingPipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def fit_transform(self, data: DatasetOutput) -> PreprocessOutput:
        trace = [{"stage": "input", **_shape(data)}]
        summaries = []
        artifacts: dict[str, Any] = {"normalization_params": {}, "feature_summary": {}, "steady_cut_summary": {}}
        for step_cfg in self.config.get("steps", []):
            name = step_cfg["name"]
            step_cls = PREPROCESSOR_REGISTRY.get(name)
            step = step_cls(step_cfg)
            before = _shape(data)
            data = step.fit_transform(data)
            after = _shape(data)
            step_summary = {"name": name, "params": getattr(step, "params", {}), "before": before, "after": after}
            summaries.append(step_summary)
            trace.append({"stage": name, **after})
            if name == "normalization":
                artifacts["normalization_params"] = step.params
            elif name == "statistical_features":
                artifacts["feature_summary"] = step.params
            elif name == "steady_cut":
                artifacts["steady_cut_summary"] = step.params
        return PreprocessOutput(
            data=data,
            summary={"output_type": self.config.get("output_type"), "steps": summaries},
            artifacts=artifacts,
            shape_trace=trace,
        )

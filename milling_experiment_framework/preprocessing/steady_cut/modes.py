from __future__ import annotations

from milling_experiment_framework.preprocessing.windows.windowing import WindowingStep
from milling_experiment_framework.registry.preprocessor_registry import PREPROCESSOR_REGISTRY


@PREPROCESSOR_REGISTRY.register("steady_cut")
class SteadyCutStep:
    def __init__(self, config: dict):
        self.config = config
        self.params: dict = {}

    def fit_transform(self, data, split_col: str = "split"):
        mode = self.config.get("mode", "full_signal")
        if mode == "full_signal":
            self.params = {"mode": mode, "method": self.config.get("method", "none"), "status": "no_op"}
            data.metadata["steady_cut_mode"] = mode
            return data
        if mode == "sliding_window":
            window_cfg = {
                "window_size": self.config.get("window_size", self.config.get("segment_length", 128)),
                "stride": self.config.get("stride", self.config.get("window_size", 128)),
            }
            step = WindowingStep(window_cfg)
            data = step.fit_transform(data, split_col)
            data.metadata["steady_cut_mode"] = mode
            self.params = {"mode": mode, "method": self.config.get("method", "none"), **step.params}
            return data
        self.params = {"mode": mode, "method": self.config.get("method"), "status": "record_only"}
        data.metadata["steady_cut_mode"] = mode
        return data

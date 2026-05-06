from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from milling_experiment_framework.core.types import DatasetOutput
from milling_experiment_framework.registry.dataset_registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("example_milling")
class ExampleMillingLoader:
    """Small deterministic milling-like dataset for smoke tests and templates."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def load(self) -> DatasetOutput:
        raw_path = self.config.get("raw_path")
        metadata_path = self.config.get("metadata_path")
        if raw_path and Path(raw_path).with_suffix(".npz").exists() and metadata_path and Path(metadata_path).exists():
            arr = np.load(Path(raw_path).with_suffix(".npz"))
            metadata = pd.read_csv(metadata_path)
            return DatasetOutput(
                X=arr["X"].astype("float32"),
                y=arr["y"],
                metadata=metadata,
                sensor_channels=self.config.get("sensor_channels", ["force_x", "force_y", "spindle_current"]),
                dataset_info=self._dataset_info(),
            )
        return self._generate()

    def _generate(self) -> DatasetOutput:
        seed = int(self.config.get("seed", 42))
        n = int(self.config.get("num_samples", 96))
        c = int(self.config.get("num_channels", 3))
        length = int(self.config.get("sequence_length", 256))
        classes = int(self.config.get("num_classes", 3))
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 1, length, dtype="float32")
        X = np.zeros((n, c, length), dtype="float32")
        y = np.arange(n) % classes
        for i in range(n):
            label = int(y[i])
            dataset_run = i // 4
            for ch in range(c):
                freq = (label + 1) * (ch + 1) * 3.0
                amplitude = 0.6 + 0.25 * label + 0.05 * ch
                transient = np.ones_like(t)
                transient[:20] = np.linspace(0.2, 1.0, 20)
                signal = amplitude * np.sin(2 * np.pi * freq * t + 0.15 * dataset_run)
                X[i, ch] = transient * signal + rng.normal(0, 0.08, size=length)
        metadata = pd.DataFrame(
            {
                "sample_id": [f"sample_{i:04d}" for i in range(n)],
                "label": y,
                "dataset_run_id": [f"dataset_run_{i // 4:03d}" for i in range(n)],
                "sequence_index": np.arange(n),
                "domain_id": [f"domain_{i % 2}" for i in range(n)],
                "condition_id": [f"condition_{i % 4}" for i in range(n)],
                "machine_id": [f"machine_{i % 2}" for i in range(n)],
                "tool_id": [f"tool_{i % 3}" for i in range(n)],
                "material_id": "aluminum",
                "process_id": "milling",
                "speed": 1000 + 50 * (np.arange(n) % 4),
                "feed": 0.05 + 0.01 * (np.arange(n) % 3),
                "depth_of_cut": 0.5 + 0.1 * (np.arange(n) % 2),
                "cutting_state": "cutting",
                "steady_cut_mode": self.config.get("steady_cut_mode", "full_signal"),
            }
        )
        return DatasetOutput(
            X=X,
            y=y,
            metadata=metadata,
            sensor_channels=self.config.get("sensor_channels", ["force_x", "force_y", "spindle_current"][:c]),
            dataset_info=self._dataset_info(generated=True),
        )

    def _dataset_info(self, generated: bool = False) -> dict[str, Any]:
        return {
            "name": self.config.get("name", "example_milling"),
            "dataset_version": self.config.get("dataset_version", "v1"),
            "raw_path": self.config.get("raw_path"),
            "metadata_path": self.config.get("metadata_path"),
            "generated": generated,
        }

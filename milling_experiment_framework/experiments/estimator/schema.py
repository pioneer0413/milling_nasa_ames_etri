from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


AXIS_NAMES = [
    "models",
    "seeds",
    "shifts",
    "sensor_combinations",
    "segment_settings",
    "feature_combinations",
    "process_combinations",
    "input_representations",
    "modes",
    "base_learners",
    "sequence_modes",
    "hyperparameter_grid",
]


@dataclass
class GridAxes:
    models: list[Any] = field(default_factory=lambda: ["default_model"])
    seeds: list[Any] = field(default_factory=lambda: [0])
    shifts: list[Any] = field(default_factory=lambda: ["default_shift"])
    sensor_combinations: list[Any] = field(default_factory=lambda: ["default_sensor"])
    segment_settings: list[Any] = field(default_factory=lambda: ["default_segment"])
    feature_combinations: list[Any] = field(default_factory=lambda: ["default_feature"])
    process_combinations: list[Any] = field(default_factory=lambda: ["none"])
    input_representations: list[Any] = field(default_factory=lambda: ["default_input"])
    modes: list[Any] = field(default_factory=lambda: ["default_mode"])
    base_learners: list[Any] = field(default_factory=list)
    sequence_modes: list[Any] = field(default_factory=lambda: ["default_sequence"])
    hyperparameter_grid: list[Any] = field(default_factory=lambda: ["default_hparam"])

    def copy_with(self, **updates: list[Any]) -> "GridAxes":
        values = {name: list(getattr(self, name)) for name in AXIS_NAMES}
        values.update({key: list(value) for key, value in updates.items()})
        return GridAxes(**values)

    def count(self, name: str) -> int:
        return max(1, len(getattr(self, name)))

    def count_dict(self) -> dict[str, int]:
        return {name: self.count(name) for name in AXIS_NAMES}


@dataclass
class CountRow:
    phase: str
    condition_group: str
    model: str
    mode: str
    input_representation: str
    counts: dict[str, int]
    atomic_executions: int
    sub_runs_per_atomic: int
    total_sub_runs: int
    notes: str = ""


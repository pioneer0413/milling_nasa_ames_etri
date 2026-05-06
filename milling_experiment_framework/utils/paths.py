from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    experiment_id: str

    @property
    def execution_dir(self) -> Path:
        return self.root / "experiments" / "executions" / self.experiment_id

    def subdir(self, name: str) -> Path:
        path = self.execution_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def prepare_standard_dirs(self) -> None:
        for name in [
            "configs",
            "data",
            "splits",
            "preprocessing",
            "model",
            "checkpoints",
            "artifacts",
            "logs",
            "metrics",
            "predictions",
            "analysis",
            "figures",
            "reports",
        ]:
            self.subdir(name)

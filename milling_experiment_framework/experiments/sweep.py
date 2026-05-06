from __future__ import annotations

from pathlib import Path

from milling_experiment_framework.experiments.experiment import ExperimentExecution


def run_sweep(config_paths: list[str | Path], root: str | Path = ".") -> list[dict]:
    return [ExperimentExecution(path, root=root).run() for path in config_paths]

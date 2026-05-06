from __future__ import annotations

from pathlib import Path

from milling_experiment_framework.experiments.experiment import ExperimentExecution


def run_pipeline(config_path: str | Path, root: str | Path = ".", smoke: bool = False, dry_run: bool = False) -> dict:
    return ExperimentExecution(config_path, root=root, smoke=smoke, dry_run=dry_run).run()

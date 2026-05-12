from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from milling_experiment_framework.experiments.execution_path import (
    LEGACY_PATH_SCHEMA_VERSION,
    PATH_SCHEMA_VERSION,
    build_execution_dir,
    current_timestamp,
    metadata_from_execution_dir,
    parse_experiment_identifier,
)


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    experiment_id: str

    @cached_property
    def execution_dir(self) -> Path:
        executions_root = self.root / "experiments" / "executions"
        parsed = parse_experiment_identifier(self.experiment_id)
        if parsed.get("hypothesis_id") and parsed.get("scenario_id") and parsed.get("experiment_topic"):
            timestamp = parsed.get("timestamp") or current_timestamp()
            return Path(
                build_execution_dir(
                    executions_root,
                    parsed["hypothesis_id"],
                    parsed["scenario_id"],
                    timestamp,
                    parsed["experiment_topic"],
                )
            )
        return executions_root / self.experiment_id

    @cached_property
    def path_metadata(self) -> dict:
        parsed = parse_experiment_identifier(self.experiment_id)
        if parsed.get("hypothesis_id") and parsed.get("scenario_id") and parsed.get("experiment_topic"):
            meta = metadata_from_execution_dir(self.execution_dir)
            return {
                "hypothesis_id": meta["hypothesis_id"],
                "scenario_id": meta["scenario_id"],
                "timestamp": meta["timestamp"],
                "experiment_topic": meta["experiment_topic"],
                "experiment_id": self.experiment_id,
                "execution_dir": str(self.execution_dir),
                "legacy_execution_dir": str(self.root / "experiments" / "executions" / self.experiment_id),
                "path_schema_version": PATH_SCHEMA_VERSION,
            }
        return {
            "hypothesis_id": None,
            "scenario_id": None,
            "timestamp": None,
            "experiment_topic": None,
            "experiment_id": self.experiment_id,
            "execution_dir": str(self.execution_dir),
            "legacy_execution_dir": str(self.execution_dir),
            "path_schema_version": LEGACY_PATH_SCHEMA_VERSION,
        }

    def apply_to_config(self, config: dict) -> None:
        experiment = config.setdefault("experiment", {})
        experiment.update({k: v for k, v in self.path_metadata.items() if v is not None})

    def report_metadata_markdown(self) -> str:
        meta = self.path_metadata
        return (
            "\n## Execution Path\n\n"
            f"- experiment_id: `{meta.get('experiment_id')}`\n"
            f"- hypothesis_id: `{meta.get('hypothesis_id')}`\n"
            f"- scenario_id: `{meta.get('scenario_id')}`\n"
            f"- experiment_topic: `{meta.get('experiment_topic')}`\n"
            f"- execution_dir: `{meta.get('execution_dir')}`\n"
            f"- path_schema_version: `{meta.get('path_schema_version')}`\n"
        )

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

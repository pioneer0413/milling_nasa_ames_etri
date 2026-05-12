from __future__ import annotations

from pathlib import Path

from milling_experiment_framework.experiments.execution_path import find_experiment_dirs


def resolve_resume_checkpoint(experiment_id: str | None, checkpoint: str | None) -> str | None:
    if not checkpoint:
        return None
    if checkpoint in {"best", "last"} and experiment_id:
        matches = find_experiment_dirs("experiments/executions", keyword=experiment_id, include_legacy=True)
        if matches:
            return str(Path(matches[-1]) / "checkpoints" / f"{checkpoint}.pt")
        return f"experiments/executions/{experiment_id}/checkpoints/{checkpoint}.pt"
    return checkpoint

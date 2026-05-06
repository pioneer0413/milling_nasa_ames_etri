from __future__ import annotations


def resolve_resume_checkpoint(experiment_id: str | None, checkpoint: str | None) -> str | None:
    if not checkpoint:
        return None
    if checkpoint in {"best", "last"} and experiment_id:
        return f"experiments/executions/{experiment_id}/checkpoints/{checkpoint}.pt"
    return checkpoint

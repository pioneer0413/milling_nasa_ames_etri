from __future__ import annotations

from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, directory: Path, config: dict, config_hash: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.config_hash = config_hash
        self.interval_paths: list[Path] = []

    def save(self, name: str, model, optimizer, scheduler, epoch: int, best_metric: float | None) -> Path:
        path = self.directory / name
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict() if optimizer else None,
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                "best_metric": best_metric,
                "config_hash": self.config_hash,
            },
            path,
        )
        return path

    def save_interval(self, model, optimizer, scheduler, epoch: int, best_metric: float | None) -> None:
        interval = self.config.get("save_interval")
        if not interval or epoch % int(interval) != 0:
            return
        path = self.save(f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, epoch, best_metric)
        self.interval_paths.append(path)
        max_keep = self.config.get("max_keep")
        if max_keep:
            while len(self.interval_paths) > int(max_keep):
                old = self.interval_paths.pop(0)
                old.unlink(missing_ok=True)

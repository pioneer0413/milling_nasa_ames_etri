from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from milling_experiment_framework.training.checkpoint import CheckpointManager


class DLTrainer:
    def __init__(self, model, training_config: dict, checkpoint_config: dict, config_hash: str, output_dir: Path):
        self.model = model.module
        self.task_type = model.task_config.get("type", "classification")
        self.training_config = training_config
        self.checkpoint_config = checkpoint_config
        self.device = torch.device("cuda" if torch.cuda.is_available() and training_config.get("device") == "cuda" else "cpu")
        self.model.to(self.device)
        self.ckpt = CheckpointManager(output_dir / "checkpoints", checkpoint_config, config_hash)

    def fit(self, X, y, metadata) -> pd.DataFrame:
        train_mask = metadata["split"].to_numpy() == "train"
        val_mask = metadata["split"].to_numpy() == "validation"
        batch_size = int(self.training_config.get("batch_size", 32))
        train_loader = self._loader(X[train_mask], y[train_mask], batch_size, shuffle=True)
        val_loader = self._loader(X[val_mask], y[val_mask], batch_size, shuffle=False) if val_mask.any() else None
        opt = torch.optim.Adam(self.model.parameters(), lr=float(self.training_config.get("learning_rate", 0.001)))
        loss_fn = nn.MSELoss() if self.task_type == "regression" else nn.CrossEntropyLoss()
        epochs = int(self.training_config.get("epochs", 3))
        best = None
        rows = []
        mode = self.checkpoint_config.get("mode", "min")
        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train_loader, opt, loss_fn, train=True)
            val_loss = self._run_epoch(val_loader, opt, loss_fn, train=False) if val_loader else train_loss
            improved = best is None or (val_loss < best if mode == "min" else val_loss > best)
            if improved:
                best = val_loss
                if self.checkpoint_config.get("enabled", True) and self.checkpoint_config.get("save_best", True):
                    self.ckpt.save("best.pt", self.model, opt, None, epoch, best)
            if self.checkpoint_config.get("enabled", True):
                if self.checkpoint_config.get("save_last", True):
                    self.ckpt.save("last.pt", self.model, opt, None, epoch, best)
                self.ckpt.save_interval(self.model, opt, None, epoch, best)
            rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best_metric": best})
        return pd.DataFrame(rows)

    def predict(self, X) -> tuple[np.ndarray, np.ndarray | None]:
        self.model.eval()
        preds = []
        probs = []
        with torch.no_grad():
            for start in range(0, len(X), 512):
                xb = torch.as_tensor(X[start : start + 512], dtype=torch.float32, device=self.device)
                out = self.model(xb)
                if self.task_type == "regression":
                    preds.append(out.squeeze(1).cpu().numpy())
                else:
                    prob = torch.softmax(out, dim=1)
                    probs.append(prob.cpu().numpy())
                    preds.append(prob.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds), np.concatenate(probs) if probs else None

    def _loader(self, X, y, batch_size: int, shuffle: bool):
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        if self.task_type == "regression":
            y_tensor = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
        else:
            y_tensor = torch.as_tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)

    def _run_epoch(self, loader, opt, loss_fn, train: bool) -> float:
        if loader is None:
            return float("nan")
        self.model.train(train)
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            if train:
                opt.zero_grad()
            out = self.model(xb)
            loss = loss_fn(out, yb)
            if train:
                loss.backward()
                opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            count += len(xb)
        return total / max(count, 1)

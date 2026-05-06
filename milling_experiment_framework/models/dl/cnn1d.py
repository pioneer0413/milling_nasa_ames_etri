from __future__ import annotations

import torch
from torch import nn

from milling_experiment_framework.registry.model_registry import MODEL_REGISTRY


class CNN1DNet(nn.Module):
    def __init__(self, num_channels: int, num_classes: int, hidden_dim: int = 64, task_type: str = "classification"):
        super().__init__()
        self.task_type = task_type
        out_dim = 1 if task_type == "regression" else num_classes
        self.net = nn.Sequential(
            nn.Conv1d(num_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


@MODEL_REGISTRY.register("cnn1d")
class CNN1DModel:
    model_type = "DL"
    input_type = "timeseries-based"

    def __init__(self, config: dict, task_config: dict):
        params = dict(config.get("params", {}))
        self.task_config = task_config
        self.module = CNN1DNet(
            num_channels=int(params.get("num_channels", 3)),
            num_classes=int(params.get("num_classes", task_config.get("num_classes", 2))),
            hidden_dim=int(params.get("hidden_dim", 64)),
            task_type=task_config.get("type", "classification"),
        )

    def fit(self, X, y, **kwargs):
        raise RuntimeError("CNN1DModel is trained by training.trainer.DLTrainer")

    def predict(self, X):
        self.module.eval()
        with torch.no_grad():
            logits = self.module(torch.as_tensor(X, dtype=torch.float32))
            if self.task_config.get("type") == "regression":
                return logits.squeeze(1).cpu().numpy()
            return logits.argmax(dim=1).cpu().numpy()

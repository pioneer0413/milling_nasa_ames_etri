from __future__ import annotations

from pathlib import Path

import pandas as pd

from milling_experiment_framework.visualization.figure_export import save_figure_dual


def plot_learning_curve(history: pd.DataFrame, path: Path) -> None:
    if history.empty:
        return
    y_cols = [c for c in ["train_loss", "val_loss"] if c in history and pd.api.types.is_numeric_dtype(history[c])]
    if not y_cols:
        return
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    ax = history.plot(x="epoch", y=y_cols, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.figure.tight_layout()
    save_figure_dual(ax.figure, path)
    plt.close(ax.figure)

from __future__ import annotations

from pathlib import Path

import pandas as pd


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
    ax.figure.savefig(path)
    plt.close(ax.figure)

from __future__ import annotations

from pathlib import Path

from milling_experiment_framework.visualization.figure_export import save_figure_dual


def plot_prediction_vs_target(y_true, y_pred, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, s=12)
    ax.set_xlabel("target")
    ax.set_ylabel("prediction")
    fig.tight_layout()
    save_figure_dual(fig, path)
    plt.close(fig)

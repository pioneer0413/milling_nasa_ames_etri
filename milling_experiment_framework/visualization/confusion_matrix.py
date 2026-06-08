from __future__ import annotations

from pathlib import Path

from sklearn.metrics import ConfusionMatrixDisplay

from milling_experiment_framework.visualization.figure_export import save_figure_dual


def plot_confusion_matrix(y_true, y_pred, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.figure_.tight_layout()
    save_figure_dual(disp.figure_, path)
    plt.close(disp.figure_)

from __future__ import annotations

from pathlib import Path

from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(y_true, y_pred, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.figure_.tight_layout()
    disp.figure_.savefig(path)
    plt.close(disp.figure_)

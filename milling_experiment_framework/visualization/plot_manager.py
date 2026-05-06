from __future__ import annotations

from pathlib import Path

import pandas as pd

from milling_experiment_framework.visualization.confusion_matrix import plot_confusion_matrix
from milling_experiment_framework.visualization.learning_curve import plot_learning_curve
from milling_experiment_framework.visualization.regression_plots import plot_prediction_vs_target


def create_standard_plots(task_type: str, predictions: pd.DataFrame, history: pd.DataFrame, figures_dir: Path) -> None:
    if not history.empty:
        plot_learning_curve(history, figures_dir / "learning_curve.png")
    if task_type == "regression":
        plot_prediction_vs_target(predictions["y_true"], predictions["y_pred"], figures_dir / "prediction_vs_target.png")
    else:
        plot_confusion_matrix(predictions["y_true"], predictions["y_pred"], figures_dir / "confusion_matrix.png")

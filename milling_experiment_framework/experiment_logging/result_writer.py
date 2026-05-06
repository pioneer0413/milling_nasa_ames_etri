from __future__ import annotations

import pandas as pd

from milling_experiment_framework.utils.io import write_csv, write_json


def write_metrics(metrics_dir, metrics: dict, metrics_csv_row: dict) -> None:
    write_json(metrics_dir / "metrics.json", metrics)
    write_csv(metrics_dir / "metrics.csv", pd.DataFrame([metrics_csv_row]))

from __future__ import annotations

from pathlib import Path

import pandas as pd


def collect_comparison(root: Path, experiment_ids: list[str], metric: str) -> pd.DataFrame:
    rows = []
    for experiment_id in experiment_ids:
        metric_path = root / "experiments" / "executions" / experiment_id / "metrics" / "metrics.csv"
        if metric_path.exists():
            row = pd.read_csv(metric_path).iloc[0].to_dict()
            rows.append(row)
    frame = pd.DataFrame(rows)
    if metric in frame.columns:
        frame = frame.sort_values(metric, ascending=False)
    return frame

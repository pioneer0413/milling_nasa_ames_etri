from __future__ import annotations

import pandas as pd


def analysis_summary(predictions: pd.DataFrame, metrics: dict) -> dict:
    return {
        "num_predictions": int(len(predictions)),
        "splits": predictions["split"].value_counts().to_dict() if "split" in predictions else {},
        "overall_metrics": metrics.get("overall", {}),
    }

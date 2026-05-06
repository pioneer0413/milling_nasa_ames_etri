from __future__ import annotations

import pandas as pd


def metrics_summary_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for scope, values in metrics.get("splits", {}).items():
        rows.append({"scope": scope, **values})
    rows.append({"scope": "overall", **metrics.get("overall", {})})
    return pd.DataFrame(rows)

from __future__ import annotations

import pandas as pd


def label_distribution(metadata: pd.DataFrame, split_col: str | None = None) -> pd.DataFrame:
    cols = ["label"]
    if split_col and split_col in metadata.columns:
        cols = [split_col, "label"]
    return metadata.groupby(cols, dropna=False).size().reset_index(name="count")


def metadata_summary(metadata: pd.DataFrame) -> dict:
    return {
        "num_rows": int(len(metadata)),
        "columns": list(metadata.columns),
        "domain_columns": [
            c
            for c in ["domain_id", "condition_id", "machine_id", "tool_id", "material_id", "process_id"]
            if c in metadata.columns
        ],
    }

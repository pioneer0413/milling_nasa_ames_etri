from __future__ import annotations

import pandas as pd


def leakage_check(assignments: pd.DataFrame, group_key: str | None = None) -> dict:
    result = {"ok": True, "sample_overlap": {}, "group_overlap": {}, "group_key": group_key}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        left = set(assignments.loc[assignments["split"] == a, "sample_id"].astype(str))
        right = set(assignments.loc[assignments["split"] == b, "sample_id"].astype(str))
        overlap = sorted(left & right)
        result["sample_overlap"][f"{a}_{b}"] = overlap
        if overlap:
            result["ok"] = False

    if group_key and group_key in assignments.columns:
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            left = set(assignments.loc[assignments["split"] == a, group_key].astype(str))
            right = set(assignments.loc[assignments["split"] == b, group_key].astype(str))
            overlap = sorted(left & right)
            result["group_overlap"][f"{a}_{b}"] = overlap
            if overlap:
                result["ok"] = False
    return result

from __future__ import annotations

from typing import Any

import pandas as pd


COMMON_VB_PREPROCESSING_RULES: dict[str, Any] = {
    "vb_missing_policy": "linear interpolation within each case ordered by run",
    "first_run_policy": "exclude the minimum run id of every case from train/validation/test datasets",
    "order": ["interpolate_missing_vb", "exclude_first_run_per_case"],
}


def apply_common_vb_prediction_preprocessing(
    data: pd.DataFrame,
    *,
    case_col: str = "case",
    run_col: str = "run",
    target_col: str = "VB",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply framework-wide VB prediction preprocessing rules.

    The rules are intentionally independent of model family and experiment:
    missing VB values are filled by case-wise linear interpolation over run
    order, then each case's first run is removed from all downstream splits.
    """

    required = {case_col, run_col, target_col}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Common VB preprocessing missing columns: {sorted(missing)}")

    out = data.copy()
    out = out.sort_values([case_col, run_col]).reset_index(drop=True)
    before_rows = int(len(out))
    missing_before = int(out[target_col].isna().sum())

    out[target_col] = (
        out.groupby(case_col, group_keys=False)[target_col]
        .apply(lambda s: s.astype("float64").interpolate(method="linear", limit_direction="both"))
        .astype("float64")
    )
    missing_after_interpolation = int(out[target_col].isna().sum())
    if missing_after_interpolation:
        unresolved = (
            out.loc[out[target_col].isna(), [case_col, run_col]]
            .drop_duplicates()
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Common VB preprocessing could not interpolate all missing VB values; "
            f"unresolved examples: {unresolved}"
        )

    first_runs = out.groupby(case_col, as_index=False)[run_col].min().rename(columns={run_col: "excluded_first_run"})
    out = out.merge(first_runs, on=case_col, how="left", validate="many_to_one")
    excluded_mask = out[run_col].eq(out["excluded_first_run"])
    excluded_rows = out.loc[excluded_mask, [case_col, run_col]].drop_duplicates()
    out = out.loc[~excluded_mask].drop(columns=["excluded_first_run"]).reset_index(drop=True)

    report = {
        "rules": COMMON_VB_PREPROCESSING_RULES,
        "case_column": case_col,
        "run_column": run_col,
        "target_column": target_col,
        "rows_before": before_rows,
        "rows_after": int(len(out)),
        "rows_excluded_by_first_run_rule": int(before_rows - len(out)),
        "missing_vb_before_interpolation": missing_before,
        "missing_vb_after_interpolation": missing_after_interpolation,
        "missing_vb_filled_by_interpolation": int(missing_before - missing_after_interpolation),
        "case_count": int(out[case_col].nunique()),
        "excluded_first_runs": [
            {str(case_col): int(row[case_col]), str(run_col): int(row[run_col])}
            for _, row in excluded_rows.sort_values([case_col, run_col]).iterrows()
        ],
    }
    return out, report

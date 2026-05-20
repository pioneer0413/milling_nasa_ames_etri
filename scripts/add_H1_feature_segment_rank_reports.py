#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def minmax_by_group(frame: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.Series:
    def _scale(s: pd.Series) -> pd.Series:
        lo = s.min(skipna=True)
        hi = s.max(skipna=True)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(np.ones(len(s)), index=s.index, dtype="float64")
        return (s - lo) / (hi - lo)

    return frame.groupby(group_cols, dropna=False)[value_col].transform(_scale)


def rank_average_report(frame: pd.DataFrame, title: str, score_col: str, rank_col: str) -> str:
    top_rank = frame.sort_values([rank_col, score_col], ascending=[True, False]).head(20)
    top_score = frame.sort_values([score_col, rank_col], ascending=[False, True]).head(20)
    return f"""# {title}

## Rank Definition

Lower rank is better. `mean_rank` is the average rank of each `feature x segment` across the experiment's native comparison units.

## Top By Mean Rank

{markdown_table(top_rank)}

## Top By Mean Score

{markdown_table(top_score)}
"""


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    out = frame.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    header = "| " + " | ".join(map(str, out.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in out.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    html_text = "<!doctype html><html><head><meta charset='utf-8'><title>Feature Segment Ranking</title></head><body>" + html.escape(text).replace("\n", "<br>\n") + "</body></html>"
    path.with_suffix(".html").write_text(html_text, encoding="utf-8")


def process_s1(path: Path) -> dict[str, Any]:
    source = path / "metrics" / "association_feature_level.csv"
    df = pd.read_csv(source)
    df = df.loc[df["status"].eq("ok")].copy()
    group_cols = ["case_id", "sensor_name"]
    df["mutual_information_norm"] = minmax_by_group(df, group_cols, "mutual_information")
    df["feature_segment_score"] = (
        df["pearson_abs"].astype(float)
        + df["spearman_abs"].astype(float)
        + df["mutual_information_norm"].astype(float)
    ) / 3.0
    df["rank_within_case_sensor"] = df.groupby(group_cols, dropna=False)["feature_segment_score"].rank(ascending=False, method="average")
    df["rank_pearson_within_case_sensor"] = df.groupby(group_cols, dropna=False)["pearson_abs"].rank(ascending=False, method="average")
    df["rank_spearman_within_case_sensor"] = df.groupby(group_cols, dropna=False)["spearman_abs"].rank(ascending=False, method="average")
    df["rank_mi_within_case_sensor"] = df.groupby(group_cols, dropna=False)["mutual_information"].rank(ascending=False, method="average")
    out = (
        df.groupby(["feature_name", "segment_setting"], dropna=False)
        .agg(
            mean_rank=("rank_within_case_sensor", "mean"),
            median_rank=("rank_within_case_sensor", "median"),
            std_rank=("rank_within_case_sensor", "std"),
            best_rank=("rank_within_case_sensor", "min"),
            worst_rank=("rank_within_case_sensor", "max"),
            mean_score=("feature_segment_score", "mean"),
            mean_abs_pearson=("pearson_abs", "mean"),
            mean_abs_spearman=("spearman_abs", "mean"),
            mean_mutual_information=("mutual_information", "mean"),
            mean_pearson_rank=("rank_pearson_within_case_sensor", "mean"),
            mean_spearman_rank=("rank_spearman_within_case_sensor", "mean"),
            mean_mi_rank=("rank_mi_within_case_sensor", "mean"),
            rank_count=("rank_within_case_sensor", "size"),
            num_cases=("case_id", "nunique"),
            num_sensors=("sensor_name", "nunique"),
        )
        .reset_index()
    )
    out["std_rank"] = out["std_rank"].fillna(0.0)
    out["rank_average"] = out[["mean_pearson_rank", "mean_spearman_rank", "mean_mi_rank"]].mean(axis=1)
    out["overall_rank_by_mean_rank"] = out["mean_rank"].rank(ascending=True, method="first").astype(int)
    out["overall_rank_by_rank_average"] = out["rank_average"].rank(ascending=True, method="first").astype(int)
    out = out.sort_values(["overall_rank_by_mean_rank", "overall_rank_by_rank_average"]).reset_index(drop=True)
    output = path / "analysis" / "H1_S1_feature_segment_rank_average.csv"
    out.to_csv(output, index=False)
    report = rank_average_report(
        out[
            [
                "feature_name",
                "segment_setting",
                "mean_rank",
                "rank_average",
                "mean_score",
                "mean_abs_spearman",
                "mean_abs_pearson",
                "mean_mutual_information",
                "overall_rank_by_mean_rank",
            ]
        ],
        "H1_S1 Feature x Segment Association Rank Average",
        "mean_score",
        "mean_rank",
    )
    report += "\n## Source\n\n" + f"- `{source}`\n"
    write_report(path / "reports" / "H1_S1_feature_segment_rank_average_report.md", report)
    return {"path": str(output), "top": out.head(10).to_dict(orient="records")}


def process_s2(path: Path) -> dict[str, Any]:
    source = path / "analysis" / "H1_S2_case_level_suitability_results.csv"
    df = pd.read_csv(source)
    df = df.loc[df["calculation_status"].eq("ok")].copy()
    group_cols = ["case_id", "sensor_name"]
    df["feature_segment_score"] = df["suitability_harmonic_mean"].astype(float)
    df["rank_within_case_sensor"] = df.groupby(group_cols, dropna=False)["feature_segment_score"].rank(ascending=False, method="average")
    df["rank_monotonicity_within_case_sensor"] = df.groupby(group_cols, dropna=False)["monotonicity"].rank(ascending=False, method="average")
    df["rank_trendability_within_case_sensor"] = df.groupby(group_cols, dropna=False)["trendability"].rank(ascending=False, method="average")
    df["rank_sum_legacy_within_case_sensor"] = df.groupby(group_cols, dropna=False)["suitability_sum_legacy"].rank(ascending=False, method="average")
    out = (
        df.groupby(["feature_name", "feature_group", "segment_setting"], dropna=False)
        .agg(
            mean_rank=("rank_within_case_sensor", "mean"),
            median_rank=("rank_within_case_sensor", "median"),
            std_rank=("rank_within_case_sensor", "std"),
            best_rank=("rank_within_case_sensor", "min"),
            worst_rank=("rank_within_case_sensor", "max"),
            mean_score=("feature_segment_score", "mean"),
            mean_monotonicity=("monotonicity", "mean"),
            mean_trendability=("trendability", "mean"),
            mean_sum_legacy=("suitability_sum_legacy", "mean"),
            mean_monotonicity_rank=("rank_monotonicity_within_case_sensor", "mean"),
            mean_trendability_rank=("rank_trendability_within_case_sensor", "mean"),
            mean_sum_legacy_rank=("rank_sum_legacy_within_case_sensor", "mean"),
            rank_count=("rank_within_case_sensor", "size"),
            num_cases=("case_id", "nunique"),
            num_sensors=("sensor_name", "nunique"),
        )
        .reset_index()
    )
    out["std_rank"] = out["std_rank"].fillna(0.0)
    out["rank_average"] = out[["mean_monotonicity_rank", "mean_trendability_rank", "mean_sum_legacy_rank"]].mean(axis=1)
    out["overall_rank_by_mean_rank"] = out["mean_rank"].rank(ascending=True, method="first").astype(int)
    out["overall_rank_by_rank_average"] = out["rank_average"].rank(ascending=True, method="first").astype(int)
    out = out.sort_values(["overall_rank_by_mean_rank", "overall_rank_by_rank_average"]).reset_index(drop=True)
    output = path / "analysis" / "H1_S2_feature_segment_rank_average.csv"
    out.to_csv(output, index=False)
    report = rank_average_report(
        out[
            [
                "feature_name",
                "segment_setting",
                "mean_rank",
                "rank_average",
                "mean_score",
                "mean_monotonicity",
                "mean_trendability",
                "mean_sum_legacy",
                "overall_rank_by_mean_rank",
            ]
        ],
        "H1_S2 Feature x Segment Suitability Rank Average",
        "mean_score",
        "mean_rank",
    )
    report += "\n## Source\n\n" + f"- `{source}`\n"
    write_report(path / "reports" / "H1_S2_feature_segment_rank_average_report.md", report)
    return {"path": str(output), "top": out.head(10).to_dict(orient="records")}


def process_s3(path: Path) -> dict[str, Any]:
    source = path / "analysis" / "H1_S3_robustness_case_sensor_feature_segment.csv"
    df = pd.read_csv(source)
    df = df.loc[df["calculation_status"].eq("ok")].copy()
    group_cols = ["case_id", "sensor_name"]
    df["feature_segment_score"] = df["robustness"].astype(float)
    df["rank_within_case_sensor"] = df.groupby(group_cols, dropna=False)["feature_segment_score"].rank(ascending=False, method="average")
    df["rank_deviation_within_case_sensor"] = df.groupby(group_cols, dropna=False)["mean_abs_relative_deviation"].rank(ascending=True, method="average")
    out = (
        df.groupby(["feature_name", "feature_group", "segment_setting"], dropna=False)
        .agg(
            mean_rank=("rank_within_case_sensor", "mean"),
            median_rank=("rank_within_case_sensor", "median"),
            std_rank=("rank_within_case_sensor", "std"),
            best_rank=("rank_within_case_sensor", "min"),
            worst_rank=("rank_within_case_sensor", "max"),
            mean_score=("feature_segment_score", "mean"),
            median_score=("feature_segment_score", "median"),
            mean_abs_relative_deviation=("mean_abs_relative_deviation", "mean"),
            mean_deviation_rank=("rank_deviation_within_case_sensor", "mean"),
            rank_count=("rank_within_case_sensor", "size"),
            num_cases=("case_id", "nunique"),
            num_sensors=("sensor_name", "nunique"),
        )
        .reset_index()
    )
    out["std_rank"] = out["std_rank"].fillna(0.0)
    out["rank_average"] = out[["mean_rank", "mean_deviation_rank"]].mean(axis=1)
    out["overall_rank_by_mean_rank"] = out["mean_rank"].rank(ascending=True, method="first").astype(int)
    out["overall_rank_by_rank_average"] = out["rank_average"].rank(ascending=True, method="first").astype(int)
    out = out.sort_values(["overall_rank_by_mean_rank", "overall_rank_by_rank_average"]).reset_index(drop=True)
    output = path / "analysis" / "H1_S3_feature_segment_rank_average.csv"
    out.to_csv(output, index=False)
    report = rank_average_report(
        out[
            [
                "feature_name",
                "segment_setting",
                "mean_rank",
                "rank_average",
                "mean_score",
                "median_score",
                "mean_abs_relative_deviation",
                "overall_rank_by_mean_rank",
            ]
        ],
        "H1_S3 Feature x Segment Robustness Rank Average",
        "mean_score",
        "mean_rank",
    )
    report += "\n## Source\n\n" + f"- `{source}`\n"
    write_report(path / "reports" / "H1_S3_feature_segment_rank_average_report.md", report)
    return {"path": str(output), "top": out.head(10).to_dict(orient="records")}


def process_sx(path: Path) -> dict[str, Any]:
    source = path / "analysis" / "H1_feature_segment_integrated_scores.csv"
    df = pd.read_csv(source)
    for col in ["association_rank", "suitability_rank", "robustness_rank", "integrated_rank"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column in Sx integrated scores: {col}")
    df["rank_average"] = df[["association_rank", "suitability_rank", "robustness_rank"]].mean(axis=1)
    df["rank_average_std"] = df[["association_rank", "suitability_rank", "robustness_rank"]].std(axis=1).fillna(0.0)
    df["overall_rank_by_rank_average"] = df["rank_average"].rank(ascending=True, method="first").astype(int)
    df["overall_rank_by_integrated_score"] = df["integrated_balanced_score"].rank(ascending=False, method="first").astype(int)
    df = df.sort_values(["overall_rank_by_rank_average", "overall_rank_by_integrated_score"]).reset_index(drop=True)
    output = path / "analysis" / "H1_feature_segment_integrated_rank_average.csv"
    df.to_csv(output, index=False)
    report = f"""# H1_Sx Integrated Feature x Segment Rank Average

## Rank Definition

`rank_average = mean(association_rank, suitability_rank, robustness_rank)`. Lower is better. This table complements `integrated_balanced_score`, which uses normalized metric values rather than rank positions.

## Top By Rank Average

{markdown_table(df[['feature_name','segment_setting','association_rank','suitability_rank','robustness_rank','rank_average','rank_average_std','integrated_balanced_score','overall_rank_by_rank_average']].head(20))}

## Top By Integrated Balanced Score

{markdown_table(df.sort_values('integrated_balanced_score', ascending=False)[['feature_name','segment_setting','association_rank','suitability_rank','robustness_rank','rank_average','integrated_balanced_score','integrated_rank']].head(20))}

## Source

- `{source}`
"""
    write_report(path / "reports" / "H1_feature_segment_integrated_rank_average_report.md", report)
    return {"path": str(output), "top": df.head(10).to_dict(orient="records")}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    paths = {
        "S1": Path(args.s1),
        "S2": Path(args.s2),
        "S3": Path(args.s3),
        "Sx": Path(args.sx),
    }
    paths = {key: (value if value.is_absolute() else root / value) for key, value in paths.items()}
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "S1": process_s1(paths["S1"]),
        "S2": process_s2(paths["S2"]),
        "S3": process_s3(paths["S3"]),
        "Sx": process_sx(paths["Sx"]),
    }
    summary_path = paths["Sx"] / "analysis" / "H1_all_rank_average_report_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Add H1 feature x segment ranking-average outputs and reports.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--s1", default="experiments/executions/H1/S1/2026-05-20_104038_589496_feture_VB_segment_association_eda")
    parser.add_argument("--s2", default="experiments/executions/H1/S2/20260520_104913_feature_VB_segment_suitability_eda")
    parser.add_argument("--s3", default="experiments/executions/H1/S3/2026-05-20_110004_robustness_segment_feature_analysis")
    parser.add_argument("--sx", default="experiments/executions/H1/Sx/2026-05-20_111008_feature_segment_integrated_association_suitability_robustness_analysis")
    result = run(parser.parse_args())
    print(json.dumps({k: v["path"] for k, v in result.items() if isinstance(v, dict) and "path" in v}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SENSOR_ORDER = ["smcAC", "smcDC", "vib_spindle", "vib_table", "AE_spindle", "AE_table"]


DEFAULT_INPUTS = {
    "S1": "experiments/executions/H1/S1/2026-05-20_104038_589496_feture_VB_segment_association_eda/analysis/H1_S1_feature_segment_rank_average.csv",
    "S2": "experiments/executions/H1/S2/20260522_112816_feature_VB_segment_suitability_eda/analysis/H1_S2_feature_segment_rank_average.csv",
    "S3": "experiments/executions/H1/S3/2026-05-20_110004_robustness_segment_feature_analysis/analysis/H1_S3_feature_segment_rank_average.csv",
}


SCORE_NOTES = {
    "S1": "Association primary score: mean of Pearson abs, Spearman abs, and normalized mutual information.",
    "S2": "Suitability primary score: suitability_harmonic = HM(M, T, P). suitability_baseline = M + T + P is reported alongside it.",
    "S3": "Robustness primary score: mean robustness; higher is smoother/more robust under the configured trend model.",
}


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


def ordered_sensors(frame: pd.DataFrame) -> list[str]:
    present = frame["sensor_name"].dropna().astype(str).unique().tolist()
    ordered = [sensor for sensor in SENSOR_ORDER if sensor in present]
    ordered.extend(sorted(sensor for sensor in present if sensor not in ordered))
    return ordered


def top3_by_sensor(frame: pd.DataFrame, experiment: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for sensor in ordered_sensors(frame):
        group = frame.loc[frame["sensor_name"].eq(sensor)].copy()
        group = group.sort_values(["mean_score", "overall_rank_by_mean_rank"], ascending=[False, True]).head(3)
        group.insert(0, "rank_within_sensor_by_score", range(1, len(group) + 1))
        group.insert(0, "experiment", experiment)
        rows.append(group)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def select_report_columns(frame: pd.DataFrame, experiment: str) -> pd.DataFrame:
    base = [
        "experiment",
        "sensor_name",
        "rank_within_sensor_by_score",
        "feature_name",
        "segment_setting",
        "mean_score",
        "mean_rank",
        "overall_rank_by_mean_rank",
    ]
    extras_by_experiment = {
        "S1": ["mean_abs_spearman", "mean_abs_pearson", "mean_mutual_information"],
        "S2": ["mean_monotonicity", "mean_trendability", "mean_prognosability", "suitability_baseline", "suitability_harmonic"],
        "S3": ["median_score", "mean_abs_relative_deviation"],
    }
    cols = [c for c in [*base, *extras_by_experiment[experiment]] if c in frame.columns]
    return frame.loc[:, cols]


def write_report(output_dir: Path, tables: dict[str, pd.DataFrame], inputs: dict[str, Path]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(tables.values(), ignore_index=True)
    csv_path = output_dir / "H1_S1_S2_S3_sensor_top3_by_score.csv"
    combined.to_csv(csv_path, index=False)

    sections = [
        "# H1 Sensor-wise Top 3 Score Report",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- Selection rule: sort each sensor by `mean_score` descending and keep top 3.",
        "- Aggregation unit: `feature_name x sensor_name x segment_setting`.",
        "",
        "## Inputs",
        "",
    ]
    for experiment, path in inputs.items():
        sections.append(f"- {experiment}: `{path}`")
    sections.extend(["", "## Score Definitions", ""])
    for experiment, note in SCORE_NOTES.items():
        sections.append(f"- {experiment}: {note}")
    sections.append("")

    for experiment in ["S1", "S2", "S3"]:
        table = tables[experiment]
        sections.extend([f"## {experiment}", ""])
        report_table = select_report_columns(table, experiment)
        sections.append(markdown_table(report_table))
        sections.append("")

    sections.extend(["## Combined CSV", "", f"- `{csv_path}`", ""])
    report = "\n".join(sections)
    md_path = output_dir / "H1_S1_S2_S3_sensor_top3_by_score_report.md"
    md_path.write_text(report, encoding="utf-8")
    html_text = "<!doctype html><html><head><meta charset='utf-8'><title>H1 Sensor Top 3</title></head><body>" + html.escape(report).replace("\n", "<br>\n") + "</body></html>"
    md_path.with_suffix(".html").write_text(html_text, encoding="utf-8")
    return {
        "report_path": str(md_path),
        "csv_path": str(csv_path),
        "rows": int(len(combined)),
        "rows_by_experiment": {key: int(len(value)) for key, value in tables.items()},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    inputs = {
        "S1": Path(args.s1),
        "S2": Path(args.s2),
        "S3": Path(args.s3),
    }
    inputs = {key: path if path.is_absolute() else root / path for key, path in inputs.items()}
    tables = {}
    for experiment, path in inputs.items():
        frame = pd.read_csv(path)
        required = {"sensor_name", "feature_name", "segment_setting", "mean_score"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{experiment} missing columns in {path}: {missing}")
        tables[experiment] = top3_by_sensor(frame, experiment)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    return write_report(output_dir, tables, inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write H1 S1/S2/S3 sensor-wise top 3 score report.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--s1", default=DEFAULT_INPUTS["S1"])
    parser.add_argument("--s2", default=DEFAULT_INPUTS["S2"])
    parser.add_argument("--s3", default=DEFAULT_INPUTS["S3"])
    parser.add_argument("--output-dir", default="experiments/executions/H1/reports")
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

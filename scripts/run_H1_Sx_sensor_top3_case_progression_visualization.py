#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_H1_S3_association_suitability_segment_cross_analysis import (  # noqa: E402
    find_or_create_entry_steady_exit_features,
    identify_sensors,
    load_dataset,
)
from run_H1_S7_feature_run_progression_visualization import (  # noqa: E402
    FEATURE_GROUPS,
    FEATURE_GROUP_COLORS,
    SEGMENT_STYLES,
    SENSOR_GRID,
    normalize_01,
    slug,
)


PREFIX = "H1_Sx"
TOPIC = "sensor_top3_case_progression_visualization"
DEFAULT_SX_DIR = (
    "experiments/executions/H1/Sx/"
    "2026-05-20_111008_feature_segment_integrated_association_suitability_robustness_analysis"
)
EXPERIMENTS = {
    "S1": {
        "label": "Association",
        "score_col": "s1_mean_score",
        "rank_col": "s1_overall_rank_by_mean_rank",
        "score_label": "association score",
    },
    "S2": {
        "label": "Suitability",
        "score_col": "s2_mean_score",
        "rank_col": "s2_overall_rank_by_mean_rank",
        "score_label": "suitability score",
    },
    "S3": {
        "label": "Robustness",
        "score_col": "s3_mean_score",
        "rank_col": "s3_overall_rank_by_mean_rank",
        "score_label": "robustness",
        "selection_mode": "score_desc",
    },
    "Sx": {
        "label": "Comprehensive",
        "score_col": "score_average_s1_s2_s3",
        "rank_col": "comprehensive_rank",
        "score_label": "comprehensive rank",
        "selection_mode": "rank_asc",
    },
}
SENSOR_ORDER = [sensor for row in SENSOR_GRID for sensor in row]


def select_sensor_top3(rank_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for experiment_id, spec in EXPERIMENTS.items():
        cols = [
            "feature_name",
            "sensor_name",
            "segment_setting",
            "comprehensive_rank",
            "rank_harmonic_s1_s2_s3",
            "rank_arithmetic_average_s1_s2_s3",
            spec["score_col"],
            spec["rank_col"],
        ]
        available = list(dict.fromkeys(col for col in cols if col in rank_table.columns))
        subset = rank_table.loc[rank_table[spec["score_col"]].notna(), available].copy()
        if spec.get("selection_mode") == "rank_asc":
            subset = subset.sort_values(
                ["sensor_name", spec["rank_col"], "rank_harmonic_s1_s2_s3", spec["score_col"]],
                ascending=[True, True, True, False],
            )
        else:
            subset = subset.sort_values(
                ["sensor_name", spec["score_col"], spec["rank_col"]],
                ascending=[True, False, True],
            )
        subset["sensor_top_rank"] = subset.groupby("sensor_name", dropna=False).cumcount() + 1
        subset = subset.loc[subset["sensor_top_rank"].le(3)].copy()
        subset["experiment_id"] = experiment_id
        subset["experiment_label"] = spec["label"]
        subset["score_col"] = spec["score_col"]
        subset["rank_col"] = spec["rank_col"]
        subset["mean_score"] = subset[spec["score_col"]]
        subset["overall_rank"] = subset[spec["rank_col"]]
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)


def vb_series(case_sensor_data: pd.DataFrame) -> pd.DataFrame:
    return (
        case_sensor_data[["run", "VB"]]
        .dropna()
        .drop_duplicates()
        .sort_values("run")
        .reset_index(drop=True)
    )


def add_vb(ax: plt.Axes, case_sensor_data: pd.DataFrame) -> None:
    vb = vb_series(case_sensor_data)
    if vb.empty:
        return
    ax.plot(
        vb["run"],
        normalize_01(vb["VB"]),
        color="black",
        marker=".",
        linestyle="-",
        linewidth=2.1,
        markersize=4.2,
        alpha=0.95,
        label="VB",
        zorder=5,
    )


def line_style(row: pd.Series) -> tuple[str, str, Any]:
    feature = str(row["feature_name"])
    segment = str(row["segment_setting"])
    group = FEATURE_GROUPS.get(feature, "statistical")
    color = FEATURE_GROUP_COLORS.get(group, "#4b5563")
    marker, linestyle = SEGMENT_STYLES.get(segment, ("o", "-"))
    return color, marker, linestyle


def plot_case(
    feature_long: pd.DataFrame,
    selected: pd.DataFrame,
    experiment_id: str,
    case_id: int,
    output_path: Path,
) -> None:
    spec = EXPERIMENTS[experiment_id]
    experiment_selected = selected.loc[selected["experiment_id"].eq(experiment_id)].copy()
    case_data = feature_long.loc[feature_long["case_id"].eq(case_id)].copy()

    fig, axes = plt.subplots(3, 2, figsize=(14.0, 10.5), sharex=False, sharey=True, squeeze=False)
    for row_idx, row in enumerate(SENSOR_GRID):
        for col_idx, sensor in enumerate(row):
            ax = axes[row_idx, col_idx]
            sensor_case = case_data.loc[case_data["sensor_name"].eq(sensor)]
            sensor_top = experiment_selected.loc[experiment_selected["sensor_name"].eq(sensor)].sort_values("sensor_top_rank")
            if sensor_case.empty or sensor_top.empty:
                ax.axis("off")
                continue
            add_vb(ax, sensor_case)
            for _, top_row in sensor_top.iterrows():
                line_data = sensor_case.loc[
                    sensor_case["feature_name"].eq(top_row["feature_name"])
                    & sensor_case["segment_setting"].eq(top_row["segment_setting"])
                ].sort_values("run")
                if line_data.empty:
                    continue
                color, marker, linestyle = line_style(top_row)
                metric = (
                    f"rank {int(top_row['overall_rank'])}"
                    if spec.get("selection_mode") == "rank_asc"
                    else f"{float(top_row['mean_score']):.3f}"
                )
                label = f"#{int(top_row['sensor_top_rank'])} {top_row['feature_name']} / {top_row['segment_setting']} ({metric})"
                ax.plot(
                    line_data["run"],
                    normalize_01(line_data["feature_value"]),
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.55,
                    markersize=3.4,
                    alpha=0.9,
                    label=label,
                )
            ax.set_title(sensor, fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Run", fontsize=8)
            ax.set_ylabel("normalized trend (0-1)", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.legend(loc="best", fontsize=6, frameon=True, framealpha=0.72)

    handles = [
        Line2D([0], [0], color=color, marker="o", linestyle="-", linewidth=1.6, markersize=4, label=group)
        for group, color in FEATURE_GROUP_COLORS.items()
    ]
    handles.append(Line2D([0], [0], color="black", marker=".", linestyle="-", linewidth=2.1, markersize=4, label="VB"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(
        f"{PREFIX} {experiment_id} Sensor Top-3 {spec['label']} Progression / Case {case_id}",
        y=0.985,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_sensor_top3_lines(
    ax: plt.Axes,
    sensor_case: pd.DataFrame,
    sensor_top: pd.DataFrame,
    *,
    linewidth: float,
    markersize: float,
    alpha: float,
) -> None:
    add_vb(ax, sensor_case)
    for _, top_row in sensor_top.iterrows():
        line_data = sensor_case.loc[
            sensor_case["feature_name"].eq(top_row["feature_name"])
            & sensor_case["segment_setting"].eq(top_row["segment_setting"])
        ].sort_values("run")
        if line_data.empty:
            continue
        color, marker, linestyle = line_style(top_row)
        ax.plot(
            line_data["run"],
            normalize_01(line_data["feature_value"]),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=markersize,
            alpha=alpha,
        )


def plot_memo_grid(
    feature_long: pd.DataFrame,
    selected: pd.DataFrame,
    experiment_id: str,
    output_path: Path,
) -> None:
    spec = EXPERIMENTS[experiment_id]
    experiment_selected = selected.loc[selected["experiment_id"].eq(experiment_id)].copy()
    cases = sorted(int(case_id) for case_id in feature_long["case_id"].unique())
    sensors = [sensor for sensor in SENSOR_ORDER if sensor in set(feature_long["sensor_name"])]
    fig, axes = plt.subplots(
        len(cases),
        len(sensors),
        figsize=(24.0, 17.0),
        sharex=False,
        sharey=True,
        squeeze=False,
    )
    for row_idx, case_id in enumerate(cases):
        case_data = feature_long.loc[feature_long["case_id"].eq(case_id)].copy()
        for col_idx, sensor in enumerate(sensors):
            ax = axes[row_idx, col_idx]
            sensor_case = case_data.loc[case_data["sensor_name"].eq(sensor)]
            sensor_top = experiment_selected.loc[experiment_selected["sensor_name"].eq(sensor)].sort_values("sensor_top_rank")
            if sensor_case.empty or sensor_top.empty:
                ax.axis("off")
                continue
            draw_sensor_top3_lines(ax, sensor_case, sensor_top, linewidth=1.0, markersize=2.0, alpha=0.82)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.22, linewidth=0.5)
            ax.tick_params(axis="both", labelsize=6)
            if row_idx == 0:
                ax.set_title(sensor, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"Case {case_id}\n0-1", fontsize=8)
            else:
                ax.set_ylabel("")
            if row_idx == len(cases) - 1:
                ax.set_xlabel("Run", fontsize=7)
            else:
                ax.set_xlabel("")
            if row_idx != len(cases) - 1:
                ax.set_xticklabels([])

    handles = [
        Line2D([0], [0], color=color, marker="o", linestyle="-", linewidth=1.8, markersize=4, label=group)
        for group, color in FEATURE_GROUP_COLORS.items()
    ]
    handles.append(Line2D([0], [0], color="black", marker=".", linestyle="-", linewidth=2.0, markersize=4, label="VB"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(
        f"{PREFIX} {experiment_id} Sensor Top-3 {spec['label']} Progression Across Cases",
        y=0.992,
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.975), h_pad=0.8, w_pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_figures(feature_long: pd.DataFrame, selected: pd.DataFrame, figures_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    cases = sorted(int(case_id) for case_id in feature_long["case_id"].unique())
    for experiment_id in EXPERIMENTS:
        for case_id in cases:
            output_path = (
                figures_dir
                / "sensor_top3_case_progression"
                / experiment_id
                / f"{PREFIX}_{experiment_id}_sensor_top3_case_{case_id:02d}.png"
            )
            plot_case(feature_long, selected, experiment_id, case_id, output_path)
            records.append(
                {
                    "view": "case_grid",
                    "experiment_id": experiment_id,
                    "case_id": case_id,
                    "output_path": str(output_path),
                    "status": "generated",
                }
            )
        memo_path = (
            figures_dir
            / "sensor_top3_case_progression"
            / "memo_grid"
            / f"{PREFIX}_{experiment_id}_sensor_top3_all_cases_grid.png"
        )
        plot_memo_grid(feature_long, selected, experiment_id, memo_path)
        records.append(
            {
                "view": "memo_grid",
                "experiment_id": experiment_id,
                "case_id": "",
                "output_path": str(memo_path),
                "status": "generated",
            }
        )
    return pd.DataFrame(records)


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    out = frame.copy() if max_rows is None else frame.head(max_rows).copy()
    if out.empty:
        return "_No rows._"
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    header = "| " + " | ".join(map(str, out.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in out.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def write_report(output_dir: Path, selected: pd.DataFrame, manifest: pd.DataFrame, summary: dict[str, Any]) -> None:
    report_sections = []
    for experiment_id, spec in EXPERIMENTS.items():
        table = selected.loc[selected["experiment_id"].eq(experiment_id)].copy()
        table = table.sort_values(["sensor_name", "sensor_top_rank"])
        table = table[
            [
                "sensor_name",
                "sensor_top_rank",
                "feature_name",
                "segment_setting",
                "mean_score",
                "overall_rank",
                "comprehensive_rank",
            ]
        ]
        report_sections.append(f"## {experiment_id} Sensor Top-3 By {spec['score_label'].title()}\n\n{markdown_table(table)}")

    figure_links = []
    for experiment_id in EXPERIMENTS:
        experiment_manifest = manifest.loc[
            manifest["experiment_id"].eq(experiment_id) & manifest["view"].eq("case_grid")
        ].sort_values("case_id")
        links = "\n".join(
            f"- Case {int(row.case_id)}: `{Path(row.output_path).relative_to(output_dir)}`"
            for row in experiment_manifest.itertuples(index=False)
        )
        figure_links.append(f"## {experiment_id} Case Figures\n\n{links}")

    memo_links = []
    memo_manifest = manifest.loc[manifest["view"].eq("memo_grid")].sort_values("experiment_id")
    for row in memo_manifest.itertuples(index=False):
        memo_links.append(f"- {row.experiment_id}: `{Path(row.output_path).relative_to(output_dir)}`")

    text = f"""# H1_Sx Sensor Top-3 Case Progression Visualization

## Purpose

This view checks whether the feature-sensor-segment combinations selected by S1, S2, S3, and the comprehensive ranking look visually consistent with the metric that selected them.

## Plot Design

- One figure per `experiment/ranking x case`.
- Each figure uses a 3x2 sensor grid.
- Each sensor axis shows only that experiment's sensor-wise Top-3 feature-segment combinations.
- VB is included as a black line in every sensor axis.
- Every feature line and VB line is independently min-max normalized to 0-1 within the current case/sensor axis.
- Feature group controls color; segment setting controls marker and line style.

## Summary

- Sx directory: `{summary["sx_dir"]}`
- Rank table: `{summary["rank_table"]}`
- Figures generated: {len(manifest)}
- Technical memo grid figures: {int(manifest["view"].eq("memo_grid").sum())}
- Cases: {summary["cases"]}

{chr(10).join(report_sections)}

## Technical Memo Grid Figures

Each figure is a 6 case x 6 sensor small-multiple view for one scenario/ranking.

{chr(10).join(memo_links)}

{chr(10).join(figure_links)}
"""
    report_path = output_dir / "reports" / f"{PREFIX}_sensor_top3_case_progression_report.md"
    report_path.write_text(text, encoding="utf-8")
    html_text = "<!doctype html><html><head><meta charset='utf-8'><title>H1_Sx Sensor Top-3 Progression</title></head><body>" + html.escape(text).replace("\n", "<br>\n") + "</body></html>"
    report_path.with_suffix(".html").write_text(html_text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    sx_dir = root / args.sx_dir
    rank_table_path = sx_dir / "analysis" / "comprehensive_rank_average.csv"
    figures_dir = sx_dir / "figures"
    analysis_dir = sx_dir / "analysis"
    reports_dir = sx_dir / "reports"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rank_table = pd.read_csv(rank_table_path)
    selected = select_sensor_top3(rank_table)

    data = load_dataset(
        root / args.process_info_path,
        root / args.signal_data_path,
        root / args.heuristic_sequence_path,
    )
    sensors_df = identify_sensors(data)
    sensors = sensors_df.loc[sensors_df["used"], "sensor_name"].tolist()
    feature_long, construction = find_or_create_entry_steady_exit_features(data, sensors)

    manifest = build_figures(feature_long, selected, figures_dir)
    selected_path = analysis_dir / f"{PREFIX}_sensor_top3_case_progression_selection.csv"
    manifest_path = analysis_dir / f"{PREFIX}_sensor_top3_case_progression_manifest.csv"
    summary_path = analysis_dir / f"{PREFIX}_sensor_top3_case_progression_summary.json"
    selected.to_csv(selected_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sx_dir": str(sx_dir),
        "rank_table": str(rank_table_path),
        "selection_csv": str(selected_path),
        "manifest_csv": str(manifest_path),
        "report": str(reports_dir / f"{PREFIX}_sensor_top3_case_progression_report.md"),
        "cases": sorted(int(case_id) for case_id in feature_long["case_id"].unique()),
        "sensors": sensors,
        "num_selected_rows": int(len(selected)),
        "num_figures": int(len(manifest)),
        "normalization": "line_minmax_0_1",
        "vb_included": True,
        "feature_construction": construction,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    write_report(sx_dir, selected, manifest, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot H1 S1/S2/S3 sensor-wise Top-3 feature progression by case.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--sx-dir", default=DEFAULT_SX_DIR)
    parser.add_argument("--process-info-path", default="datasets/processed/mill_process_info_enabled.csv")
    parser.add_argument("--signal-data-path", default="datasets/processed/mill_signal_data_enabled.csv")
    parser.add_argument("--heuristic-sequence-path", default="datasets/metadata/heuristic_sequence_peng2026.csv")
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

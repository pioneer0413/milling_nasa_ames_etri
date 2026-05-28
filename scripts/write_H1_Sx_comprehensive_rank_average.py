#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


KEY_COLS = ["feature_name", "sensor_name", "segment_setting"]
DEFAULT_S1 = "experiments/executions/H1/S1/2026-05-20_104038_589496_feture_VB_segment_association_eda/analysis/H1_S1_feature_segment_rank_average.csv"
DEFAULT_S2 = "experiments/executions/H1/S2/20260522_112816_feature_VB_segment_suitability_eda/analysis/H1_S2_feature_segment_rank_average.csv"
DEFAULT_S3 = "experiments/executions/H1/S3/2026-05-20_110004_robustness_segment_feature_analysis/analysis/H1_S3_feature_segment_rank_average.csv"
DEFAULT_OUTPUT = "experiments/executions/H1/Sx/2026-05-20_111008_feature_segment_integrated_association_suitability_robustness_analysis/analysis/comprehensive_rank_average.csv"
DEFAULT_REPORT = "experiments/executions/H1/Sx/2026-05-20_111008_feature_segment_integrated_association_suitability_robustness_analysis/reports/H1_feature_segment_integrated_rank_average_report.md"
CATEGORY_RANK_COLS = {
    "Association": "s1_overall_rank_by_mean_rank",
    "Suitability": "s2_overall_rank_by_mean_rank",
    "Robustness": "s3_overall_rank_by_mean_rank",
}
SENSOR_ORDER = ["smcAC", "smcDC", "vib_spindle", "vib_table", "AE_spindle", "AE_table"]
SENSOR_COLORS = {
    "smcAC": "#2563eb",
    "smcDC": "#16a34a",
    "vib_spindle": "#d97706",
    "vib_table": "#7c3aed",
    "AE_spindle": "#dc2626",
    "AE_table": "#0891b2",
}


def load_with_prefix(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in KEY_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing key columns: {missing}")
    if df.duplicated(KEY_COLS).any():
        dupes = df.loc[df.duplicated(KEY_COLS, keep=False), KEY_COLS].drop_duplicates()
        raise ValueError(f"{path} has duplicate key rows: {dupes.head(10).to_dict(orient='records')}")
    rename = {col: f"{prefix}_{col}" for col in df.columns if col not in KEY_COLS}
    return df.rename(columns=rename)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    out = out.fillna("")
    header = "| " + " | ".join(map(str, out.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in out.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def harmonic_mean(frame: pd.DataFrame, columns: list[str], eps: float = 1e-12) -> pd.Series:
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    valid = values.notna() & values.gt(0)
    denom = (1.0 / (values.where(valid) + eps)).sum(axis=1, skipna=True)
    count = valid.sum(axis=1)
    result = count / denom
    return result.where(count.gt(0), pd.NA)


def rank_trajectory_frame(joined: pd.DataFrame) -> pd.DataFrame:
    cols = [
        *KEY_COLS,
        "comprehensive_rank",
        "rank_average_s1_s2_s3",
        "rank_harmonic_s1_s2_s3",
        "rank_arithmetic_average_s1_s2_s3",
        *CATEGORY_RANK_COLS.values(),
    ]
    data = joined[[col for col in cols if col in joined.columns]].copy()
    for col in CATEGORY_RANK_COLS.values():
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=list(CATEGORY_RANK_COLS.values()))
    data["side_rank_mean"] = data[["s1_overall_rank_by_mean_rank", "s3_overall_rank_by_mean_rank"]].mean(axis=1)
    data["suitability_rank_gap"] = data["s2_overall_rank_by_mean_rank"] - data["side_rank_mean"]
    data["trajectory_label"] = (
        data["feature_name"].astype(str) + " | " + data["sensor_name"].astype(str) + " | " + data["segment_setting"].astype(str)
    )
    return data


def plot_rank_lines(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    color_col: str = "sensor_name",
    background: bool = True,
    linewidth: float = 1.8,
    marker_size: float = 20.0,
) -> None:
    x = list(range(len(CATEGORY_RANK_COLS)))
    rank_cols = list(CATEGORY_RANK_COLS.values())
    if background:
        for _, row in data.iterrows():
            ax.plot(x, row[rank_cols].astype(float), color="#9ca3af", alpha=0.065, linewidth=0.65, zorder=1)
    for _, row in data.iterrows():
        color = SENSOR_COLORS.get(str(row[color_col]), "#111827")
        y = row[rank_cols].astype(float).to_numpy()
        ax.plot(x, y, color=color, alpha=0.9, linewidth=linewidth, zorder=3)
        ax.scatter(x, y, color=color, s=marker_size, edgecolor="white", linewidth=0.5, zorder=4)
    ax.set_xticks(x, list(CATEGORY_RANK_COLS))
    ax.invert_yaxis()
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.grid(axis="x", color="#f3f4f6", linewidth=0.8)
    ax.set_ylabel("Rank (1 = best)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_rank_trajectory_figures(joined: pd.DataFrame, report_path: Path) -> list[Path]:
    data = rank_trajectory_frame(joined)
    if data.empty:
        return []

    figures_dir = report_path.parent.parent / "figures"
    analysis_dir = report_path.parent.parent / "analysis"
    figures_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    max_rank = max(float(data[col].max()) for col in CATEGORY_RANK_COLS.values())
    sensors = [sensor for sensor in SENSOR_ORDER if sensor in set(data["sensor_name"])]
    sensors += [sensor for sensor in sorted(set(data["sensor_name"])) if sensor not in sensors]
    x = list(range(len(CATEGORY_RANK_COLS)))
    rank_cols = list(CATEGORY_RANK_COLS.values())
    outputs: list[Path] = []

    points_path = analysis_dir / "H1_rank_trajectory_points.csv"
    data.to_csv(points_path, index=False)
    outputs.append(points_path)

    overview_path = figures_dir / "H1_rank_trajectory_overview.png"
    highlight = data.sort_values(["comprehensive_rank", "rank_average_s1_s2_s3"]).head(30)
    fig, ax = plt.subplots(figsize=(9.5, 7.0), dpi=180)
    for _, row in data.iterrows():
        ax.plot(x, row[rank_cols].astype(float), color="#9ca3af", alpha=0.055, linewidth=0.65, zorder=1)
    plot_rank_lines(ax, highlight, background=False, linewidth=2.0, marker_size=24)
    ax.set_ylim(max_rank + 10, 0)
    ax.set_title("H1 Rank Trajectory: Top Comprehensive Feature-Sensor-Segment Combinations", pad=14)
    legend = [
        Line2D([0], [0], color=SENSOR_COLORS.get(sensor, "#111827"), lw=2.2, marker="o", markersize=5, label=sensor)
        for sensor in sensors
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title="Sensor")
    fig.tight_layout()
    fig.savefig(overview_path, bbox_inches="tight")
    plt.close(fig)
    outputs.append(overview_path)

    by_sensor_path = figures_dir / "H1_rank_trajectory_by_sensor.png"
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.0), dpi=180, sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, sensor in zip(axes_flat, sensors):
        sensor_data = data.loc[data["sensor_name"].eq(sensor)].copy()
        sensor_top = sensor_data.sort_values(["comprehensive_rank", "rank_average_s1_s2_s3"]).head(5)
        for _, row in sensor_data.iterrows():
            ax.plot(x, row[rank_cols].astype(float), color="#9ca3af", alpha=0.12, linewidth=0.7, zorder=1)
        plot_rank_lines(ax, sensor_top, background=False, linewidth=2.0, marker_size=18)
        for _, row in sensor_top.head(3).iterrows():
            y = float(row["s3_overall_rank_by_mean_rank"])
            label = f"{row['feature_name']} / {row['segment_setting']}"
            ax.text(2.04, y, label, fontsize=7.0, va="center", color=SENSOR_COLORS.get(sensor, "#111827"))
        ax.set_title(sensor)
        ax.set_ylim(max_rank + 10, 0)
    for ax in axes_flat[len(sensors) :]:
        ax.axis("off")
    fig.suptitle("H1 Rank Trajectory By Sensor", y=0.995)
    fig.tight_layout()
    fig.savefig(by_sensor_path, bbox_inches="tight")
    plt.close(fig)
    outputs.append(by_sensor_path)

    vshape_path = figures_dir / "H1_rank_trajectory_suitability_dip_candidates.png"
    candidates = data.loc[
        data["s1_overall_rank_by_mean_rank"].le(160)
        & data["s3_overall_rank_by_mean_rank"].le(160)
        & data["suitability_rank_gap"].gt(40)
    ].copy()
    if candidates.empty:
        candidates = data.sort_values("suitability_rank_gap", ascending=False).head(24)
    else:
        candidates = candidates.sort_values(["suitability_rank_gap", "side_rank_mean"], ascending=[False, True]).head(24)
    fig, ax = plt.subplots(figsize=(9.5, 7.0), dpi=180)
    for _, row in data.iterrows():
        ax.plot(x, row[rank_cols].astype(float), color="#9ca3af", alpha=0.05, linewidth=0.65, zorder=1)
    plot_rank_lines(ax, candidates, background=False, linewidth=1.9, marker_size=22)
    for _, row in candidates.head(8).iterrows():
        y = float(row["s2_overall_rank_by_mean_rank"])
        label = f"{row['feature_name']} | {row['sensor_name']} | {row['segment_setting']}"
        ax.text(1.04, y, label, fontsize=7.0, va="center", color=SENSOR_COLORS.get(str(row["sensor_name"]), "#111827"))
    ax.set_ylim(max_rank + 10, 0)
    ax.set_title("H1 Rank Trajectory: Suitability-Dip Candidates", pad=14)
    legend = [
        Line2D([0], [0], color=SENSOR_COLORS.get(sensor, "#111827"), lw=2.2, marker="o", markersize=5, label=sensor)
        for sensor in sensors
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title="Sensor")
    fig.tight_layout()
    fig.savefig(vshape_path, bbox_inches="tight")
    plt.close(fig)
    outputs.append(vshape_path)

    interactive_path = figures_dir / "H1_rank_trajectory_interactive.html"
    width = 1100
    height = 760
    left = 110
    right = 170
    top = 70
    bottom = 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_pos = [left, left + plot_width / 2, left + plot_width]

    def y_pos(value: float) -> float:
        return top + ((float(value) - 1.0) / max(max_rank - 1.0, 1.0)) * plot_height

    tick_values = [1, 50, 100, 150, 200, 250, 300, 350, 400, int(max_rank)]
    tick_values = sorted({tick for tick in tick_values if 1 <= tick <= max_rank})
    grid = []
    for tick in tick_values:
        y = y_pos(tick)
        grid.append(f"<line x1='{left}' x2='{left + plot_width}' y1='{y:.2f}' y2='{y:.2f}' class='grid' />")
        grid.append(f"<text x='{left - 14}' y='{y + 4:.2f}' text-anchor='end' class='tick'>{tick}</text>")
    axes = []
    for xpos, label in zip(x_pos, CATEGORY_RANK_COLS):
        axes.append(f"<line x1='{xpos:.2f}' x2='{xpos:.2f}' y1='{top}' y2='{top + plot_height}' class='axis' />")
        axes.append(f"<text x='{xpos:.2f}' y='{height - 30}' text-anchor='middle' class='axis-label'>{label}</text>")
    line_groups = []
    for _, row in data.iterrows():
        points = " ".join(f"{xpos:.2f},{y_pos(float(row[col])):.2f}" for xpos, col in zip(x_pos, rank_cols))
        sensor = str(row["sensor_name"])
        color = SENSOR_COLORS.get(sensor, "#111827")
        label = (
            f"<b>{html.escape(str(row['feature_name']))}</b><br>"
            f"Sensor: {html.escape(sensor)}<br>"
            f"Segment: {html.escape(str(row['segment_setting']))}<br>"
            f"Association rank: {int(row['s1_overall_rank_by_mean_rank'])}<br>"
            f"Suitability rank: {int(row['s2_overall_rank_by_mean_rank'])}<br>"
            f"Robustness rank: {int(row['s3_overall_rank_by_mean_rank'])}<br>"
            f"Comprehensive rank: {int(row['comprehensive_rank'])}"
        )
        line_groups.append(
            "<g class='traj' "
            f"data-sensor='{html.escape(sensor, quote=True)}'>"
            f"<polyline class='visible-line' points='{points}' stroke='{color}' />"
            f"<polyline class='hit-line' points='{points}' data-label='{html.escape(label, quote=True)}' />"
            "</g>"
        )
    checkboxes = "\n".join(
        "<label>"
        f"<input type='checkbox' data-sensor='{html.escape(sensor, quote=True)}' checked>"
        f"<span style='background:{SENSOR_COLORS.get(sensor, '#111827')}'></span>{html.escape(sensor)}"
        "</label>"
        for sensor in sensors
    )
    interactive_html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>H1 Rank Trajectory Interactive</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; color: #111827; background: #ffffff; }}
.wrap {{ display: grid; grid-template-columns: 1fr 230px; gap: 16px; padding: 22px; }}
h1 {{ font-size: 22px; margin: 0 0 6px; }}
p {{ margin: 0 0 14px; color: #4b5563; }}
svg {{ width: 100%; height: auto; border: 1px solid #e5e7eb; background: #ffffff; }}
.grid {{ stroke: #e5e7eb; stroke-width: 1; }}
.axis {{ stroke: #d1d5db; stroke-width: 1; }}
.tick {{ fill: #6b7280; font-size: 12px; }}
.axis-label {{ fill: #111827; font-size: 16px; font-weight: 700; }}
.visible-line {{ fill: none; stroke-width: 1.3; opacity: 0.28; }}
.hit-line {{ fill: none; stroke: transparent; stroke-width: 13; pointer-events: stroke; cursor: crosshair; }}
.traj.active .visible-line {{ stroke-width: 4; opacity: 1; }}
.panel {{ border: 1px solid #e5e7eb; padding: 14px; height: fit-content; }}
.panel h2 {{ font-size: 15px; margin: 0 0 10px; }}
.panel label {{ display: flex; align-items: center; gap: 8px; margin: 9px 0; font-size: 14px; }}
.panel span {{ width: 12px; height: 12px; display: inline-block; border-radius: 50%; }}
#tooltip {{ position: fixed; display: none; pointer-events: none; z-index: 10; background: rgba(17, 24, 39, 0.94); color: white; padding: 9px 11px; border-radius: 6px; font-size: 13px; line-height: 1.45; max-width: 320px; }}
</style>
</head>
<body>
<div class="wrap">
<main>
<h1>H1 Rank Trajectory</h1>
<p>Hover a line to identify the feature-sensor-segment. Rank 1 is at the top.</p>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="H1 rank trajectory">
{"".join(grid)}
{"".join(axes)}
<text x="28" y="{top + plot_height / 2:.2f}" transform="rotate(-90 28 {top + plot_height / 2:.2f})" text-anchor="middle" class="axis-label">Rank (1 = best)</text>
{"".join(line_groups)}
</svg>
</main>
<aside class="panel">
<h2>Sensor Filter</h2>
{checkboxes}
</aside>
</div>
<div id="tooltip"></div>
<script>
const tooltip = document.getElementById('tooltip');
document.querySelectorAll('.hit-line').forEach((line) => {{
  line.addEventListener('mouseenter', () => {{
    line.parentElement.classList.add('active');
    tooltip.style.display = 'block';
    tooltip.innerHTML = line.dataset.label;
  }});
  line.addEventListener('mousemove', (event) => {{
    tooltip.style.left = `${{event.clientX + 14}}px`;
    tooltip.style.top = `${{event.clientY + 14}}px`;
  }});
  line.addEventListener('mouseleave', () => {{
    line.parentElement.classList.remove('active');
    tooltip.style.display = 'none';
  }});
}});
document.querySelectorAll('input[data-sensor]').forEach((checkbox) => {{
  checkbox.addEventListener('change', () => {{
    document.querySelectorAll(`.traj[data-sensor="${{checkbox.dataset.sensor}}"]`).forEach((group) => {{
      group.style.display = checkbox.checked ? '' : 'none';
    }});
  }});
}});
</script>
</body>
</html>
"""
    interactive_path.write_text(interactive_html, encoding="utf-8")
    outputs.append(interactive_path)

    return outputs


def write_report(path: Path, joined: pd.DataFrame, inputs: dict[str, str], figure_paths: list[Path]) -> None:
    report_cols = [
        "comprehensive_rank",
        "feature_name",
        "sensor_name",
        "segment_setting",
        "s1_overall_rank_by_mean_rank",
        "s1_mean_score",
        "s2_overall_rank_by_mean_rank",
        "s2_mean_score",
        "s3_overall_rank_by_mean_rank",
        "s3_mean_score",
        "rank_harmonic_s1_s2_s3",
        "rank_arithmetic_average_s1_s2_s3",
        "rank_average_s1_s2_s3",
        "score_average_s1_s2_s3",
        "available_experiment_count",
    ]
    available = [col for col in report_cols if col in joined.columns]
    top_comprehensive = joined[available].head(25)

    def top_by(rank_col: str, score_col: str) -> pd.DataFrame:
        cols = [
            "feature_name",
            "sensor_name",
            "segment_setting",
            rank_col,
            score_col,
            "rank_harmonic_s1_s2_s3",
            "rank_average_s1_s2_s3",
            "comprehensive_rank",
        ]
        cols = [col for col in cols if col in joined.columns]
        return joined.loc[joined[rank_col].notna(), cols].sort_values([rank_col, score_col], ascending=[True, False]).head(15)

    def top3_by_sensor(rank_col: str, score_col: str) -> pd.DataFrame:
        base_cols = [
            "sensor_name",
            "feature_name",
            "segment_setting",
            score_col,
            rank_col,
            "rank_harmonic_s1_s2_s3",
            "rank_average_s1_s2_s3",
            "comprehensive_rank",
        ]
        base_cols = [col for col in base_cols if col in joined.columns]
        ranked = joined.loc[joined[rank_col].notna(), base_cols].sort_values(
            ["sensor_name", score_col, rank_col],
            ascending=[True, False, True],
        )
        ranked["sensor_top_rank"] = ranked.groupby("sensor_name", dropna=False).cumcount() + 1
        ranked = ranked.loc[ranked["sensor_top_rank"].le(3)]
        return ranked[["sensor_name", "sensor_top_rank", *[col for col in base_cols if col != "sensor_name"]]]

    def top3_comprehensive_by_sensor() -> pd.DataFrame:
        cols = [
            "sensor_name",
            "feature_name",
            "segment_setting",
            "comprehensive_rank",
            "rank_harmonic_s1_s2_s3",
            "rank_arithmetic_average_s1_s2_s3",
            "score_average_s1_s2_s3",
            "s1_overall_rank_by_mean_rank",
            "s2_overall_rank_by_mean_rank",
            "s3_overall_rank_by_mean_rank",
        ]
        cols = [col for col in cols if col in joined.columns]
        ranked = joined.loc[joined["comprehensive_rank"].notna(), cols].sort_values(
            ["sensor_name", "comprehensive_rank"],
            ascending=[True, True],
        )
        ranked["sensor_top_rank"] = ranked.groupby("sensor_name", dropna=False).cumcount() + 1
        ranked = ranked.loc[ranked["sensor_top_rank"].le(3)]
        return ranked[["sensor_name", "sensor_top_rank", *[col for col in cols if col != "sensor_name"]]]

    text = f"""# H1 S1/S2/S3 Comprehensive Rank Average

## Rank Definition

This report restores the explicit S1, S2, and S3 columns.

- S1: association rank average table, with `s1_mean_score` updated to the harmonic mean of `pearson_abs`, `spearman_abs`, and case-sensor normalized mutual information.
- S2: suitability rank average table, using harmonic suitability as the primary score.
- S3: robustness rank average table.
- `rank_harmonic_s1_s2_s3 = harmonic_mean(S1 overall rank, S2 overall rank, S3 overall rank)`. Lower is better.
- `rank_average_s1_s2_s3` is kept as the same harmonic rank value for backward-compatible downstream views.
- `rank_arithmetic_average_s1_s2_s3` preserves the previous simple arithmetic mean for comparison.
- `comprehensive_rank` sorts complete S1/S2/S3 evidence first, then lower harmonic rank, then higher score average.
- Sensor Top-3 sections are selected and sorted within each sensor by the corresponding experiment's `mean_score` descending; rank is used only as a tie-breaker.
- Comprehensive Sensor Top-3 is selected within each sensor by `comprehensive_rank` ascending.

## Top Comprehensive Rank

{markdown_table(top_comprehensive)}

## Sensor Top-3 By Comprehensive Rank

{markdown_table(top3_comprehensive_by_sensor())}

## Rank Trajectory Figures

Each line connects the same `feature_name x sensor_name x segment_setting` across Association, Suitability, and Robustness ranks. The Y-axis is inverted so rank 1 is at the top.

- Overview with all combinations in the background and top comprehensive combinations highlighted: `../figures/H1_rank_trajectory_overview.png`
- Sensor-wise 3x2 view with the strongest combinations in each sensor highlighted: `../figures/H1_rank_trajectory_by_sensor.png`
- Suitability-dip candidates, where Association and Robustness ranks are comparatively strong but Suitability rank drops: `../figures/H1_rank_trajectory_suitability_dip_candidates.png`
- Interactive hover view for identifying individual combinations: `../figures/H1_rank_trajectory_interactive.html`
- Plot source table: `../analysis/H1_rank_trajectory_points.csv`

![H1 rank trajectory overview](../figures/H1_rank_trajectory_overview.png)

![H1 rank trajectory by sensor](../figures/H1_rank_trajectory_by_sensor.png)

![H1 rank trajectory suitability dip candidates](../figures/H1_rank_trajectory_suitability_dip_candidates.png)

## Top By S1 Association Rank

{markdown_table(top_by("s1_overall_rank_by_mean_rank", "s1_mean_score"))}

## Sensor Top-3 By S1 Association Mean Score

{markdown_table(top3_by_sensor("s1_overall_rank_by_mean_rank", "s1_mean_score"))}

## Top By S2 Suitability Rank

{markdown_table(top_by("s2_overall_rank_by_mean_rank", "s2_mean_score"))}

## Sensor Top-3 By S2 Suitability Mean Score

{markdown_table(top3_by_sensor("s2_overall_rank_by_mean_rank", "s2_mean_score"))}

## Top By S3 Robustness Rank

{markdown_table(top_by("s3_overall_rank_by_mean_rank", "s3_mean_score"))}

## Sensor Top-3 By S3 Robustness Mean Score

{markdown_table(top3_by_sensor("s3_overall_rank_by_mean_rank", "s3_mean_score"))}

## Sensor Top-3 Case Progression Figures

These figures are intended as a visual check of whether each experiment's sensor-wise Top-3 feature-sensor-segment selections look consistent with the metric that selected them.

- Report: `H1_Sx_sensor_top3_case_progression_report.md`
- Figures: `../figures/sensor_top3_case_progression/<S1|S2|S3|Sx>/H1_Sx_<S1|S2|S3|Sx>_sensor_top3_case_<case>.png`
- Selection table: `../analysis/H1_Sx_sensor_top3_case_progression_selection.csv`
- Manifest: `../analysis/H1_Sx_sensor_top3_case_progression_manifest.csv`

## Sources

- S1: `{inputs["s1"]}`
- S2: `{inputs["s2"]}`
- S3: `{inputs["s3"]}`
- Comprehensive CSV: `{DEFAULT_OUTPUT}`
- Figures: {", ".join(f"`{path}`" for path in figure_paths)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    html_text = "<!doctype html><html><head><meta charset='utf-8'><title>H1 Comprehensive Rank Average</title></head><body>" + html.escape(text).replace("\n", "<br>\n") + "</body></html>"
    path.with_suffix(".html").write_text(html_text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    s1 = Path(args.s1)
    s2 = Path(args.s2)
    s3 = Path(args.s3)
    output = Path(args.output)
    report = Path(args.report)
    s1 = s1 if s1.is_absolute() else root / s1
    s2 = s2 if s2.is_absolute() else root / s2
    s3 = s3 if s3.is_absolute() else root / s3
    output = output if output.is_absolute() else root / output
    report = report if report.is_absolute() else root / report

    frames = {
        "s1": load_with_prefix(s1, "s1"),
        "s2": load_with_prefix(s2, "s2"),
        "s3": load_with_prefix(s3, "s3"),
    }
    joined = frames["s1"].merge(frames["s2"], on=KEY_COLS, how="outer").merge(frames["s3"], on=KEY_COLS, how="outer")

    for col in ["s1_overall_rank_by_mean_rank", "s2_overall_rank_by_mean_rank", "s3_overall_rank_by_mean_rank"]:
        if col not in joined.columns:
            joined[col] = pd.NA
    rank_cols = ["s1_overall_rank_by_mean_rank", "s2_overall_rank_by_mean_rank", "s3_overall_rank_by_mean_rank"]
    joined["rank_arithmetic_average_s1_s2_s3"] = joined[rank_cols].mean(axis=1, skipna=True)
    joined["rank_harmonic_s1_s2_s3"] = harmonic_mean(joined, rank_cols)
    joined["rank_average_s1_s2_s3"] = joined["rank_harmonic_s1_s2_s3"]
    joined["score_average_s1_s2_s3"] = joined[["s1_mean_score", "s2_mean_score", "s3_mean_score"]].mean(axis=1, skipna=True)
    joined["available_experiment_count"] = joined[["s1_mean_score", "s2_mean_score", "s3_mean_score"]].notna().sum(axis=1)
    joined = joined.sort_values(
        ["available_experiment_count", "rank_harmonic_s1_s2_s3", "score_average_s1_s2_s3"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    joined.insert(0, "comprehensive_rank", range(1, len(joined) + 1))

    output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output, index=False)
    inputs = {"s1": str(s1), "s2": str(s2), "s3": str(s3)}
    figure_paths = save_rank_trajectory_figures(joined, report)
    write_report(report, joined, inputs, figure_paths)
    summary = {
        "output": str(output),
        "report": str(report),
        "figures": [str(path) for path in figure_paths],
        "rows": int(len(joined)),
        "columns": int(len(joined.columns)),
        "input_rows": {key: int(len(value)) for key, value in frames.items()},
        "available_experiment_count": joined["available_experiment_count"].value_counts().sort_index().to_dict(),
        "comprehensive_rank_method": "sort by complete evidence, harmonic mean of S1/S2/S3 overall ranks, then score average",
        "inputs": inputs,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Join H1 S1/S2/S3 rank-average tables into Sx comprehensive_rank_average.csv.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--s1", default=DEFAULT_S1)
    parser.add_argument("--s2", default=DEFAULT_S2)
    parser.add_argument("--s3", default=DEFAULT_S3)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

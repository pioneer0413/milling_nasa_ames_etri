#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_H1_S3_association_suitability_segment_cross_analysis import (  # noqa: E402
    CASE_SCOPE,
    SEGMENT_SETTINGS,
    TARGET_FEATURES,
    find_or_create_entry_steady_exit_features,
    identify_sensors,
    load_dataset,
)


PREFIX = "H1_S7"
TOPIC = "feature_run_progression_visualization"
SENSOR_GRID = [
    ["smcAC", "smcDC"],
    ["vib_spindle", "vib_table"],
    ["AE_spindle", "AE_table"],
]
FEATURE_GROUPS = {
    "mean": "statistical",
    "std": "statistical",
    "max": "statistical",
    "min": "statistical",
    "peak_to_peak": "statistical",
    "kurtosis": "shape",
    "skewness": "shape",
    "spectral_centroid": "frequency-domain",
    "band_energy": "frequency-domain",
}
FEATURE_GROUP_COLORS = {
    "statistical": "#0072B2",
    "shape": "#D55E00",
    "frequency-domain": "#009E73",
}
FEATURE_STYLES = {
    "mean": ("o", "-"),
    "std": ("s", "--"),
    "max": ("^", "-."),
    "min": ("v", ":"),
    "peak_to_peak": ("D", (0, (3, 1, 1, 1))),
    "kurtosis": ("P", "-"),
    "skewness": ("X", "--"),
    "spectral_centroid": ("*", "-"),
    "band_energy": ("h", "--"),
}
SEGMENT_COLORS = {
    "full_length": "#6F4E7C",
    "steady": "#4E79A7",
    "entry": "#F28E2B",
    "exit": "#E15759",
    "entry_steady": "#76B7B2",
    "entry_exit": "#59A14F",
    "steady_exit": "#EDC948",
    "entry_steady_exit": "#B07AA1",
}
SEGMENT_STYLES = {
    "full_length": ("o", "-"),
    "steady": ("s", "-"),
    "entry": ("^", "--"),
    "exit": ("v", "--"),
    "entry_steady": ("D", "-."),
    "entry_exit": ("P", ":"),
    "steady_exit": ("X", (0, (3, 1, 1, 1))),
    "entry_steady_exit": ("h", (0, (5, 2))),
}
PLOT_VERSION = "2026-05-22.2"


def slug(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def make_output_dir(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = root / "experiments" / "executions" / "H1" / "S7" / f"{timestamp}_{TOPIC}"
    for dirname in ["configs", "data", "analysis", "figures", "reports", "logs"]:
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)
    return output_dir


def stable_feature_hash(feature_long: pd.DataFrame) -> str:
    cols = ["case_id", "run", "VB", "sensor_name", "segment_setting", "feature_name", "feature_value"]
    compact = feature_long.loc[:, cols].sort_values(cols[:-1]).reset_index(drop=True)
    digest = pd.util.hash_pandas_object(compact, index=True).to_numpy(dtype=np.uint64).tobytes()
    return hashlib.sha256(digest).hexdigest()[:16]


def figure_cache_key(view: str, case_id: int, facet: str, data_hash: str) -> str:
    raw = json.dumps(
        {
            "prefix": PREFIX,
            "plot_version": PLOT_VERSION,
            "view": view,
            "case_id": int(case_id),
            "facet": facet,
            "segments": SEGMENT_SETTINGS,
            "features": TARGET_FEATURES,
            "sensor_grid": SENSOR_GRID,
            "normalization": "line_minmax_0_1",
            "vb_included": True,
            "data_hash": data_hash,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def normalize_01(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    out = np.full(arr.shape, np.nan, dtype="float64")
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    lo = float(np.min(arr[finite]))
    hi = float(np.max(arr[finite]))
    if hi <= lo:
        out[finite] = 0.5
    else:
        out[finite] = (arr[finite] - lo) / (hi - lo)
    return out


def sensor_axes(axes: np.ndarray, available_sensors: list[str]):
    available = set(available_sensors)
    for row_idx, row in enumerate(SENSOR_GRID):
        for col_idx, sensor in enumerate(row):
            ax = axes[row_idx, col_idx]
            if sensor in available:
                yield ax, sensor
            else:
                ax.axis("off")


def vb_series(case_data: pd.DataFrame) -> pd.DataFrame:
    return (
        case_data[["run", "VB"]]
        .dropna()
        .drop_duplicates()
        .sort_values("run")
        .reset_index(drop=True)
    )


def add_vb_line(ax: plt.Axes, case_data: pd.DataFrame) -> None:
    vb = vb_series(case_data)
    if vb.empty:
        return
    ax.plot(
        vb["run"],
        normalize_01(vb["VB"]),
        color="black",
        marker=".",
        markersize=4.2,
        linewidth=2.0,
        linestyle="-",
        alpha=0.95,
        zorder=5,
    )


def apply_axis_style(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Run", fontsize=8)
    ax.set_ylabel("normalized trend (0-1)", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def add_feature_legend(fig: plt.Figure) -> None:
    handles = []
    for feature in TARGET_FEATURES:
        marker, linestyle = FEATURE_STYLES[feature]
        group = FEATURE_GROUPS[feature]
        handles.append(
            Line2D(
                [0],
                [0],
                color=FEATURE_GROUP_COLORS[group],
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=4,
                label=f"{feature} ({group})",
            )
        )
    handles.append(Line2D([0], [0], color="black", marker=".", linewidth=2.0, markersize=4, label="VB"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7, frameon=False)


def add_segment_legend(fig: plt.Figure) -> None:
    handles = []
    for segment in SEGMENT_SETTINGS:
        marker, linestyle = SEGMENT_STYLES[segment]
        handles.append(
            Line2D(
                [0],
                [0],
                color=SEGMENT_COLORS[segment],
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=4,
                label=segment,
            )
        )
    handles.append(Line2D([0], [0], color="black", marker=".", linewidth=2.0, markersize=4, label="VB"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7, frameon=False)


def plot_segment_case(feature_long: pd.DataFrame, sensors: list[str], segment: str, case_id: int, output_path: Path) -> None:
    case_segment = feature_long.loc[
        feature_long["case_id"].eq(case_id) & feature_long["segment_setting"].eq(segment)
    ].copy()
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=False, squeeze=False)
    for ax, sensor in sensor_axes(axes, sensors):
        sensor_data = case_segment.loc[case_segment["sensor_name"].eq(sensor)]
        for feature in TARGET_FEATURES:
            data = sensor_data.loc[sensor_data["feature_name"].eq(feature)].sort_values("run")
            if data.empty:
                continue
            marker, linestyle = FEATURE_STYLES[feature]
            group = FEATURE_GROUPS[feature]
            ax.plot(
                data["run"],
                normalize_01(data["feature_value"]),
                color=FEATURE_GROUP_COLORS[group],
                marker=marker,
                linestyle=linestyle,
                markersize=3.0,
                linewidth=1.25,
                alpha=0.88,
            )
        add_vb_line(ax, sensor_data)
        apply_axis_style(ax, sensor)
    fig.suptitle(f"{PREFIX} Segment X Case: {segment} / case {case_id}", y=0.985, fontsize=13)
    add_feature_legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.965))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_case(feature_long: pd.DataFrame, sensors: list[str], feature: str, case_id: int, output_path: Path) -> None:
    case_feature = feature_long.loc[
        feature_long["case_id"].eq(case_id) & feature_long["feature_name"].eq(feature)
    ].copy()
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=False, squeeze=False)
    for ax, sensor in sensor_axes(axes, sensors):
        sensor_data = case_feature.loc[case_feature["sensor_name"].eq(sensor)]
        for segment in SEGMENT_SETTINGS:
            data = sensor_data.loc[sensor_data["segment_setting"].eq(segment)].sort_values("run")
            if data.empty:
                continue
            marker, linestyle = SEGMENT_STYLES[segment]
            ax.plot(
                data["run"],
                normalize_01(data["feature_value"]),
                color=SEGMENT_COLORS[segment],
                marker=marker,
                linestyle=linestyle,
                markersize=3.0,
                linewidth=1.25,
                alpha=0.88,
            )
        add_vb_line(ax, sensor_data)
        apply_axis_style(ax, sensor)
    fig.suptitle(f"{PREFIX} Feature X Case: {feature} / case {case_id}", y=0.985, fontsize=13)
    add_segment_legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.965))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_or_reuse(
    *,
    cache_dir: Path,
    output_path: Path,
    cache_key: str,
    render_func: Any,
    record_base: dict[str, Any],
) -> dict[str, Any]:
    cache_file = cache_dir / f"{cache_key}.png"
    record = {**record_base, "cache_key": cache_key, "output_path": str(output_path), "cache_path": str(cache_file)}
    if cache_file.exists():
        link_or_copy(cache_file, output_path)
        return {**record, "status": "skipped_cache_hit"}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    render_func(cache_file)
    link_or_copy(cache_file, output_path)
    return {**record, "status": "generated"}


def build_figures(feature_long: pd.DataFrame, sensors: list[str], output_dir: Path, data_hash: str) -> pd.DataFrame:
    cache_dir = output_dir.parents[0] / "_cache" / "figures" / data_hash
    records: list[dict[str, Any]] = []
    cases = sorted(int(x) for x in feature_long["case_id"].unique())

    for segment in SEGMENT_SETTINGS:
        for case_id in cases:
            out = output_dir / "figures" / "segment_x_case" / slug(segment) / f"{PREFIX}_{slug(segment)}_case_{case_id:02d}.png"
            key = figure_cache_key("segment_x_case", case_id, segment, data_hash)
            records.append(
                render_or_reuse(
                    cache_dir=cache_dir,
                    output_path=out,
                    cache_key=key,
                    render_func=lambda path, segment=segment, case_id=case_id: plot_segment_case(
                        feature_long, sensors, segment, case_id, path
                    ),
                    record_base={"view": "segment_x_case", "case_id": case_id, "segment_setting": segment, "feature_name": ""},
                )
            )

    for feature in TARGET_FEATURES:
        for case_id in cases:
            out = output_dir / "figures" / "feature_x_case" / slug(feature) / f"{PREFIX}_{slug(feature)}_case_{case_id:02d}.png"
            key = figure_cache_key("feature_x_case", case_id, feature, data_hash)
            records.append(
                render_or_reuse(
                    cache_dir=cache_dir,
                    output_path=out,
                    cache_key=key,
                    render_func=lambda path, feature=feature, case_id=case_id: plot_feature_case(
                        feature_long, sensors, feature, case_id, path
                    ),
                    record_base={"view": "feature_x_case", "case_id": case_id, "segment_setting": "", "feature_name": feature},
                )
            )
    return pd.DataFrame(records)


def summarize_progression(feature_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["case_id", "sensor_name", "segment_setting", "feature_name"]
    for keys, group in feature_long.groupby(group_cols, dropna=False):
        case_id, sensor, segment, feature = keys
        g = group.sort_values("run")
        x = g["run"].to_numpy(dtype="float64")
        y = g["feature_value"].to_numpy(dtype="float64")
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if len(x) < 2 or np.std(x) <= 0 or np.std(y) <= 0:
            slope = np.nan
            corr = np.nan
            delta = np.nan
            status = "skipped"
        else:
            slope = float(np.polyfit(x, y, 1)[0])
            corr = float(np.corrcoef(x, y)[0, 1])
            delta = float(y[-1] - y[0])
            status = "ok"
        rows.append(
            {
                "case_id": int(case_id),
                "sensor_name": sensor,
                "segment_setting": segment,
                "feature_name": feature,
                "n_runs": int(len(x)),
                "first_value": float(y[0]) if len(y) else np.nan,
                "last_value": float(y[-1]) if len(y) else np.nan,
                "delta_last_minus_first": delta,
                "linear_slope_per_run": slope,
                "pearson_run_corr": corr,
                "abs_pearson_run_corr": abs(corr) if np.isfinite(corr) else np.nan,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, figure_manifest: pd.DataFrame, progression: pd.DataFrame, summary: dict[str, Any]) -> None:
    top = (
        progression.loc[progression["status"].eq("ok")]
        .assign(abs_slope=lambda d: d["linear_slope_per_run"].abs())
        .sort_values("abs_slope", ascending=False)
        .head(20)
    )
    cache_counts = figure_manifest["status"].value_counts().to_dict()
    report = f"""# H1_S7 Feature Run Progression Visualization

## Scope

- Cases: {summary["cases"]}
- Sensors: {", ".join(summary["sensors"])}
- Segments: {", ".join(SEGMENT_SETTINGS)}
- Features: {", ".join(TARGET_FEATURES)}

## Views

- `figures/segment_x_case/<segment>/H1_S7_<segment>_case_<case>.png`: 3x2 sensor grid; each sensor axis overlays the nine normalized feature trends and normalized VB.
- `figures/feature_x_case/<feature>/H1_S7_<feature>_case_<case>.png`: 3x2 sensor grid; each sensor axis overlays the eight normalized segment trends and normalized VB.

All plotted values are min-max normalized to 0-1 within each line, so the figures emphasize trend similarity rather than physical units.
VB is always drawn as a black line. In the segment x case view, statistical, shape, and frequency-domain feature groups share group colors while individual features use distinct marker/line styles.

## Skip Logic

Execution directories still use creation timestamps, but figure reuse is keyed by semantic identity:
`view + case + segment/feature + data hash + plot version`.
The stable cache lives under `experiments/executions/H1/S7/_cache/figures/<data_hash>/`, so repeated runs can skip rendering even when the new execution id is date-based.

## Result

- Figures requested: {len(figure_manifest)}
- Generated: {cache_counts.get("generated", 0)}
- Skipped by cache hit: {cache_counts.get("skipped_cache_hit", 0)}
- Data hash: `{summary["data_hash"]}`

## Strongest Absolute Run Slopes

{top[["case_id", "sensor_name", "segment_setting", "feature_name", "linear_slope_per_run", "pearson_run_corr", "n_runs"]].to_markdown(index=False)}
"""
    path = output_dir / "reports" / f"{PREFIX}_report.md"
    path.write_text(report, encoding="utf-8")
    html_text = "<!doctype html><html><head><meta charset='utf-8'><title>H1_S7</title></head><body>" + html.escape(report).replace("\n", "<br>\n") + "</body></html>"
    path.with_suffix(".html").write_text(html_text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    output_dir = make_output_dir(root)
    process = root / args.process_info_path
    signal = root / args.signal_data_path
    heuristic = root / args.heuristic_sequence_path

    data = load_dataset(process, signal, heuristic)
    sensors_df = identify_sensors(data)
    sensors = sensors_df.loc[sensors_df["used"], "sensor_name"].tolist()
    feature_long, construction = find_or_create_entry_steady_exit_features(data, sensors)
    data_hash = stable_feature_hash(feature_long)

    figure_manifest = build_figures(feature_long, sensors, output_dir, data_hash)
    progression = summarize_progression(feature_long)

    feature_long.to_csv(output_dir / "data" / f"{PREFIX}_feature_long.csv", index=False)
    sensors_df.to_csv(output_dir / "data" / f"{PREFIX}_sensor_mapping.csv", index=False)
    figure_manifest.to_csv(output_dir / "analysis" / f"{PREFIX}_figure_manifest.csv", index=False)
    progression.to_csv(output_dir / "analysis" / f"{PREFIX}_run_progression_summary.csv", index=False)

    summary = {
        "output_dir": str(output_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "process_info_path": str(process),
        "signal_data_path": str(signal),
        "heuristic_sequence_path": str(heuristic),
        "cases": sorted(int(x) for x in feature_long["case_id"].unique()),
        "sensors": sensors,
        "segments": SEGMENT_SETTINGS,
        "features": TARGET_FEATURES,
        "sensor_grid": SENSOR_GRID,
        "data_hash": data_hash,
        "plot_version": PLOT_VERSION,
        "num_feature_rows": int(len(feature_long)),
        "num_figures": int(len(figure_manifest)),
        "num_generated": int(figure_manifest["status"].eq("generated").sum()),
        "num_skipped_cache_hit": int(figure_manifest["status"].eq("skipped_cache_hit").sum()),
        "feature_construction": construction,
        "cache_root": str(output_dir.parents[0] / "_cache" / "figures" / data_hash),
    }
    (output_dir / "analysis" / f"{PREFIX}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    config = {
        "experiment_numbering": "H1_S7",
        "topic": TOPIC,
        "created_at": summary["created_at"],
        "skip_logic": "Stable cache key independent of date-based execution id; cache hits are hard-linked or copied into the current execution.",
        "sensor_grid": SENSOR_GRID,
        "normalization": "Each plotted feature, segment, and VB line is independently min-max normalized to 0-1 within the current case/axis.",
        "vb_style": "black line",
        "feature_group_colors": FEATURE_GROUP_COLORS,
        "inputs": {
            "process_info_path": str(process),
            "signal_data_path": str(signal),
            "heuristic_sequence_path": str(heuristic),
        },
        "plot_version": PLOT_VERSION,
        "data_hash": data_hash,
    }
    (output_dir / "configs" / f"{PREFIX}_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(output_dir, figure_manifest, progression, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H1_S7 feature run-progression visualization.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--process-info-path", default="datasets/processed/mill_process_info_enabled.csv")
    parser.add_argument("--signal-data-path", default="datasets/processed/mill_signal_data_enabled.csv")
    parser.add_argument("--heuristic-sequence-path", default="datasets/metadata/heuristic_sequence_peng2026.csv")
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

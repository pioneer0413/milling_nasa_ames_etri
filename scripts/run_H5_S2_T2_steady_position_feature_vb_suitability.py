#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.visualization.figure_export import save_figure_dual
from scripts import run_H5_S2_T1_steady_length_feature_vb_suitability as t1


PREFIX = "H5_S2_T2"
TOPIC = "steady_position_feature_sequence_vb_suitability"
SENSOR = t1.SENSOR
SEGMENT_SETTING = t1.SEGMENT_SETTING
TARGET_FEATURES = t1.TARGET_FEATURES
FEATURE_GROUPS = t1.FEATURE_GROUPS
DEFAULT_CASE_SCOPE = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
BASE_STEADY_LENGTH = 5000
SUBWINDOW_LENGTH = 2000
POSITIONS = ["start", "center", "end"]
BASE_METADATA_PATH = "datasets/metadata/heuristic_sequence_peng2026_steady5000.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H5_S2_T2 steady-position feature-sequence/VB suitability analysis.")
    parser.add_argument("--signal-path", default="datasets/processed/mill_signal_data.csv")
    parser.add_argument("--process-info-path", default="datasets/processed/mill_process_info.csv")
    parser.add_argument("--metadata-path", default=BASE_METADATA_PATH)
    parser.add_argument("--case-scope", type=int, nargs="+", default=DEFAULT_CASE_SCOPE)
    parser.add_argument("--output-root", default="experiments/executions")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    return t1.to_builtin(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        path.write_text(yaml.safe_dump(to_builtin(payload), sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        write_json(path.with_suffix(".json"), payload)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def execution_dir(output_root: Path, timestamp: str) -> Path:
    return output_root / "H5" / "S2" / "T2" / f"{timestamp}_{TOPIC}"


def make_dirs(output_dir: Path) -> None:
    for rel in ["configs", "data", "analysis", "figures", "logs", "reports"]:
        (output_dir / rel).mkdir(parents=True, exist_ok=True)
    for feature in TARGET_FEATURES:
        (output_dir / "figures" / feature).mkdir(parents=True, exist_ok=True)


def build_config(args: argparse.Namespace, timestamp: str, output_dir: Path) -> dict[str, Any]:
    return {
        "experiment": {
            "experiment_number": PREFIX,
            "name": f"{PREFIX}_{TOPIC}",
            "experiment_id": f"{timestamp}_{PREFIX}_{TOPIC}",
            "timestamp": timestamp,
            "execution_dir": str(output_dir),
            "analysis_type": "steady_position_feature_sequence_vb_suitability",
        },
        "data": {
            "signal_path": args.signal_path,
            "process_info_path": args.process_info_path,
            "metadata_path": args.metadata_path,
            "case_scope": args.case_scope,
            "sensor": SENSOR,
            "segment_setting": SEGMENT_SETTING,
        },
        "windowing": {
            "base_steady_length": BASE_STEADY_LENGTH,
            "subwindow_length": SUBWINDOW_LENGTH,
            "positions": POSITIONS,
            "position_definition": "start, center, and end 2000-sample windows inside the 5000-sample steady-cut interval",
        },
        "features": {
            "source": "H1 nine feature set",
            "feature_names": TARGET_FEATURES,
            "feature_groups": FEATURE_GROUPS,
        },
        "suitability": {
            "primary_method": t1.PRIMARY_SUITABILITY_METHOD,
            "components": ["monotonicity", "trendability", "prognosability"],
            "epsilon": t1.SUITABILITY_EPSILON,
            "prognosability_epsilon": t1.PROGNOSABILITY_EPSILON,
        },
    }


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(resolve(args.metadata_path))
    signal = pd.read_csv(resolve(args.signal_path), usecols=["case", "run", SENSOR])
    process = pd.read_csv(resolve(args.process_info_path))
    required_metadata = {"case", "run", "idx_start", "idx_end", "signal_length"}
    required_process = {"case", "run", "VB"}
    missing_metadata = required_metadata.difference(metadata.columns)
    missing_process = required_process.difference(process.columns)
    if missing_metadata:
        raise ValueError(f"metadata missing columns: {sorted(missing_metadata)}")
    if missing_process:
        raise ValueError(f"process info missing columns: {sorted(missing_process)}")
    return metadata, signal, process


def position_bounds(idx_start: int, idx_end: int, signal_size: int) -> dict[str, tuple[int, int]]:
    idx_start = int(np.clip(idx_start, 0, signal_size))
    idx_end = int(np.clip(idx_end, idx_start, signal_size))
    available = max(idx_end - idx_start, 0)
    if available < SUBWINDOW_LENGTH:
        raise ValueError(f"Steady interval shorter than {SUBWINDOW_LENGTH}: start={idx_start}, end={idx_end}")
    center_start = idx_start + (available - SUBWINDOW_LENGTH) // 2
    return {
        "start": (idx_start, idx_start + SUBWINDOW_LENGTH),
        "center": (center_start, center_start + SUBWINDOW_LENGTH),
        "end": (idx_end - SUBWINDOW_LENGTH, idx_end),
    }


def build_feature_long(metadata: pd.DataFrame, signal: pd.DataFrame, process: pd.DataFrame, case_scope: list[int]) -> pd.DataFrame:
    keys = metadata[["case", "run", "idx_start", "idx_end", "signal_length"]].drop_duplicates().copy()
    keys = keys.loc[keys["case"].isin(case_scope)].copy()
    merged = keys.merge(signal, on=["case", "run"], how="left", validate="one_to_one")
    merged = merged.merge(process, on=["case", "run"], how="left", validate="one_to_one")
    if merged[SENSOR].isna().any():
        missing = merged.loc[merged[SENSOR].isna(), ["case", "run"]].to_dict(orient="records")
        raise ValueError(f"Missing signal rows: {missing[:5]}")
    if merged["VB"].isna().any():
        missing = merged.loc[merged["VB"].isna(), ["case", "run"]].to_dict(orient="records")
        raise ValueError(f"Missing VB rows: {missing[:5]}")

    rows: list[dict[str, Any]] = []
    for source_row in merged.itertuples(index=False):
        case = int(source_row.case)
        run_id = int(source_row.run)
        signal_arr = t1.parse_signal(getattr(source_row, SENSOR))
        bounds = position_bounds(int(source_row.idx_start), int(source_row.idx_end), signal_arr.size)
        for position in POSITIONS:
            window_start, window_end = bounds[position]
            values = t1.compute_features(signal_arr[window_start:window_end])
            for feature_name, feature_value in values.items():
                rows.append(
                    {
                        "case_id": case,
                        "run": run_id,
                        "steady_position": position,
                        "position_order": POSITIONS.index(position),
                        "base_steady_length": BASE_STEADY_LENGTH,
                        "subwindow_length": SUBWINDOW_LENGTH,
                        "sensor_name": SENSOR,
                        "segment_setting": SEGMENT_SETTING,
                        "feature_name": feature_name,
                        "feature_group": FEATURE_GROUPS[feature_name],
                        "feature_value": feature_value,
                        "VB": float(source_row.VB),
                        "time": getattr(source_row, "time", np.nan),
                        "DOC": getattr(source_row, "DOC", np.nan),
                        "feed": getattr(source_row, "feed", np.nan),
                        "material_name": getattr(source_row, "material_name", np.nan),
                        "base_idx_start": int(source_row.idx_start),
                        "base_idx_end": int(source_row.idx_end),
                        "subwindow_idx_start": int(window_start),
                        "subwindow_idx_end": int(window_end),
                    }
                )
    return pd.DataFrame(rows).sort_values(["feature_name", "position_order", "case_id", "run"]).reset_index(drop=True)


def compute_suitability_tables(feature_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    group_cols = ["case_id", "steady_position", "position_order", "sensor_name", "segment_setting", "feature_name", "feature_group"]
    for keys, group in feature_long.groupby(group_cols, dropna=False):
        case_id, position, position_order, sensor_name, segment_setting, feature_name, feature_group = keys
        group = group.sort_values(["run"], kind="mergesort")
        x_raw = group["feature_value"].to_numpy(dtype=np.float64)
        t_raw = group["run"].to_numpy(dtype=np.float64)
        finite = np.isfinite(x_raw) & np.isfinite(t_raw)
        x = x_raw[finite]
        t = t_raw[finite]
        status = "ok"
        if len(x) < 2:
            monotonicity = trendability = x_start = x_final = float("nan")
            status = "skipped_n_less_than_2"
        else:
            monotonicity = t1.compute_monotonicity(x)
            trendability = t1.compute_trendability(x, t)
            x_start = float(x[0])
            x_final = float(x[-1])
            if not np.isfinite(trendability):
                status = "skipped_zero_trendability_denominator"
            if not np.isfinite(monotonicity):
                status = "skipped_monotonicity_not_computable"
        rows.append(
            {
                "case_id": int(case_id),
                "steady_position": position,
                "position_order": int(position_order),
                "sensor_name": sensor_name,
                "segment_setting": segment_setting,
                "feature_name": feature_name,
                "feature_group": feature_group,
                "monotonicity": monotonicity,
                "trendability": trendability,
                "x_start": x_start,
                "x_final": x_final,
                "sample_count": int(len(x)),
                "ordering_column": "run",
                "calculation_status": status,
                "primary_suitability_method": t1.PRIMARY_SUITABILITY_METHOD,
            }
        )

    case_level = pd.DataFrame(rows)
    case_level["prognosability"] = np.nan
    for _, group in case_level.groupby(["steady_position", "sensor_name", "segment_setting", "feature_name"], dropna=False):
        ok = group.loc[group["calculation_status"].eq("ok")]
        prognosability = t1.compute_prognosability(ok["x_start"].to_numpy(dtype=np.float64), ok["x_final"].to_numpy(dtype=np.float64))
        case_level.loc[group.index, "prognosability"] = prognosability
    case_level["suitability"] = case_level.apply(
        lambda row: t1.compute_suitability_harmonic(row["monotonicity"], row["trendability"], row["prognosability"]),
        axis=1,
    )
    invalid = case_level["suitability"].notna() & ~case_level["suitability"].between(0, 1)
    case_level.loc[invalid, "calculation_status"] = "invalid_suitability_range"

    ok = case_level.loc[case_level["calculation_status"].eq("ok")].copy()
    summary = (
        ok.groupby(["steady_position", "position_order", "sensor_name", "segment_setting", "feature_name", "feature_group"], dropna=False)
        .agg(
            mean_monotonicity=("monotonicity", "mean"),
            mean_trendability=("trendability", "mean"),
            mean_prognosability=("prognosability", "mean"),
            mean_suitability=("suitability", "mean"),
            median_suitability=("suitability", "median"),
            std_suitability=("suitability", "std"),
            min_suitability=("suitability", "min"),
            max_suitability=("suitability", "max"),
            num_cases=("case_id", "nunique"),
            total_sample_count=("sample_count", "sum"),
        )
        .reset_index()
    )
    summary["std_suitability"] = summary["std_suitability"].fillna(0.0)
    summary["rank_by_suitability_within_position"] = summary.groupby("steady_position")["mean_suitability"].rank(ascending=False, method="first").astype(int)
    summary["rank_by_suitability_within_feature"] = summary.groupby("feature_name")["mean_suitability"].rank(ascending=False, method="first").astype(int)
    summary = summary.sort_values(["feature_name", "position_order"]).reset_index(drop=True)
    return case_level.sort_values(["feature_name", "position_order", "case_id"]).reset_index(drop=True), summary


def case_suitability_text(case_level: pd.DataFrame, feature_name: str, position: str, case: int) -> str:
    row = case_level.loc[
        case_level["feature_name"].eq(feature_name)
        & case_level["steady_position"].eq(position)
        & case_level["case_id"].eq(case)
    ]
    if row.empty or not np.isfinite(row.iloc[0]["suitability"]):
        return ""
    return f"S={float(row.iloc[0]['suitability']):.2f}"


def plot_feature_grid(output_dir: Path, feature_long: pd.DataFrame, case_level: pd.DataFrame, summary: pd.DataFrame, feature_name: str, dpi: int) -> Path:
    feature_df = feature_long.loc[feature_long["feature_name"].eq(feature_name)].copy()
    cases = sorted(feature_df["case_id"].unique().tolist())
    row_labels = ["All cases mean"] + [f"Case {case}" for case in cases]
    fig, axes = plt.subplots(len(row_labels), len(POSITIONS), figsize=(len(POSITIONS) * 3.3, len(row_labels) * 1.45), squeeze=False)
    feature_ylim = t1.robust_ylim(feature_df["feature_value"])
    vb_ylim = t1.robust_ylim(feature_df["VB"])

    for col_idx, position in enumerate(POSITIONS):
        position_df = feature_df.loc[feature_df["steady_position"].eq(position)]
        avg = (
            position_df.groupby("run", as_index=False)
            .agg(feature_value=("feature_value", "mean"), VB=("VB", "mean"), n_cases=("case_id", "nunique"))
            .sort_values("run")
        )
        s_row = summary.loc[(summary["feature_name"].eq(feature_name)) & (summary["steady_position"].eq(position))]
        s_text = f"S={float(s_row.iloc[0]['mean_suitability']):.3f}" if not s_row.empty else ""
        t1.draw_sequence_ax(axes[0, col_idx], avg, feature_ylim, vb_ylim, title=f"{position} ({s_text})", row_label=row_labels[0])
        for row_idx, case in enumerate(cases, start=1):
            case_df = position_df.loc[position_df["case_id"].eq(case)].sort_values("run")
            t1.draw_sequence_ax(
                axes[row_idx, col_idx],
                case_df,
                feature_ylim,
                vb_ylim,
                title="",
                row_label=row_labels[row_idx],
                annotation=case_suitability_text(case_level, feature_name, position, case),
            )

    for ax in axes[-1, :]:
        ax.set_xlabel("run", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel(feature_name, fontsize=7)
    handles = [
        Line2D([0], [0], color="#2563eb", linewidth=1.2, marker="o", markersize=2.5, label=f"{feature_name}"),
        Line2D([0], [0], color="#dc2626", linewidth=1.0, linestyle="--", marker="s", markersize=2.5, label="VB trend"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"{PREFIX} {feature_name}: 2000-sample steady-position feature sequence with VB trend", fontsize=13, y=0.997)
    fig.tight_layout(rect=(0.01, 0.025, 0.99, 0.982), h_pad=0.45, w_pad=0.22)
    path = output_dir / "figures" / feature_name / f"{PREFIX}_{feature_name}_position_sequence_vb_grid.png"
    save_figure_dual(fig, path, dpi=dpi)
    plt.close(fig)
    return path


def plot_suitability_heatmap(output_dir: Path, summary: pd.DataFrame, dpi: int) -> Path:
    matrix = summary.pivot(index="feature_name", columns="steady_position", values="mean_suitability").reindex(index=TARGET_FEATURES, columns=POSITIONS)
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel("position inside 5000-sample steady-cut")
    ax.set_title(f"{PREFIX} mean suitability by feature and steady position")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value < 0.55 else "black")
    fig.colorbar(im, ax=ax, label="mean suitability")
    fig.tight_layout()
    path = output_dir / "figures" / f"{PREFIX}_suitability_heatmap.png"
    save_figure_dual(fig, path, dpi=dpi)
    plt.close(fig)
    return path


def write_report(output_dir: Path, summary_payload: dict[str, Any], suitability_summary: pd.DataFrame) -> None:
    top = suitability_summary.sort_values("mean_suitability", ascending=False).head(12)
    lines = [
        f"# {PREFIX} Steady-position Feature Sequence and VB Suitability",
        "",
        "## Scope",
        "",
        f"- Sensor: `{SENSOR}`",
        f"- Segment: `{SEGMENT_SETTING}` only",
        f"- Base steady-cut length: `{BASE_STEADY_LENGTH}`",
        f"- Sub-window length: `{SUBWINDOW_LENGTH}`",
        f"- Positions: `{POSITIONS}`",
        f"- Cases: `{summary_payload['case_count']}`",
        f"- Case/run rows: `{summary_payload['case_run_count']}`",
        "",
        "## Outputs",
        "",
        "- Feature sequence table: `data/H5_S2_T2_feature_sequence_long.csv`",
        "- Case-level suitability: `analysis/H5_S2_T2_case_level_suitability.csv`",
        "- Feature-position suitability summary: `analysis/H5_S2_T2_feature_position_suitability_summary.csv`",
        "- Per-feature figures: `figures/{feature}/H5_S2_T2_{feature}_position_sequence_vb_grid.{png,svg}`",
        "- Suitability heatmap: `figures/H5_S2_T2_suitability_heatmap.{png,svg}`",
        "",
        "## Top Suitability",
        "",
        "| feature | position | mean_suitability | mean_monotonicity | mean_trendability | mean_prognosability |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.feature_name} | {row.steady_position} | {row.mean_suitability:.4f} | "
            f"{row.mean_monotonicity:.4f} | {row.mean_trendability:.4f} | {row.mean_prognosability:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Method Note",
            "",
            "The 5000-sample steady-cut interval is taken from `heuristic_sequence_peng2026_steady5000.csv`. "
            "Each feature is recomputed on 2000-sample start, center, and end sub-windows, then evaluated by the same monotonicity, trendability, and prognosability suitability formulation used in T1.",
        ]
    )
    (output_dir / "reports" / "H5_S2_T2_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = args.timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = execution_dir(resolve(args.output_root), timestamp)
    if output_dir.exists():
        raise FileExistsError(f"Execution directory already exists: {output_dir}")
    make_dirs(output_dir)

    config = build_config(args, timestamp, output_dir)
    write_yaml(output_dir / "configs" / "H5_S2_T2_input_config.yaml", config)
    write_json(output_dir / "logs" / "H5_S2_T2_environment.json", collect_environment())

    metadata, signal, process = load_inputs(args)
    scoped = metadata.loc[metadata["case"].isin(args.case_scope)].copy()
    summary_payload: dict[str, Any] = {
        "experiment_id": config["experiment"]["experiment_id"],
        "execution_dir": str(output_dir),
        "sensor": SENSOR,
        "segment_setting": SEGMENT_SETTING,
        "base_steady_length": BASE_STEADY_LENGTH,
        "subwindow_length": SUBWINDOW_LENGTH,
        "positions": POSITIONS,
        "features": TARGET_FEATURES,
        "case_scope": sorted(args.case_scope),
        "case_count": int(scoped["case"].nunique()),
        "case_run_count": int(len(scoped[["case", "run"]].drop_duplicates())),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        write_json(output_dir / "analysis" / "H5_S2_T2_summary.json", summary_payload)
        return summary_payload

    feature_long = build_feature_long(metadata, signal, process, args.case_scope)
    feature_long.to_csv(output_dir / "data" / "H5_S2_T2_feature_sequence_long.csv", index=False)
    case_level, suitability_summary = compute_suitability_tables(feature_long)
    case_level.to_csv(output_dir / "analysis" / "H5_S2_T2_case_level_suitability.csv", index=False)
    suitability_summary.to_csv(output_dir / "analysis" / "H5_S2_T2_feature_position_suitability_summary.csv", index=False)

    figure_manifest: list[dict[str, Any]] = []
    for feature_name in TARGET_FEATURES:
        feature_dir = output_dir / "figures" / feature_name
        feature_long.loc[feature_long["feature_name"].eq(feature_name)].to_csv(feature_dir / f"{PREFIX}_{feature_name}_feature_sequence.csv", index=False)
        suitability_summary.loc[suitability_summary["feature_name"].eq(feature_name)].to_csv(feature_dir / f"{PREFIX}_{feature_name}_suitability_summary.csv", index=False)
        fig_path = plot_feature_grid(output_dir, feature_long, case_level, suitability_summary, feature_name, args.dpi)
        figure_manifest.append({"feature_name": feature_name, "figure_path": str(fig_path.relative_to(output_dir))})
    heatmap_path = plot_suitability_heatmap(output_dir, suitability_summary, args.dpi)
    figure_manifest.append({"feature_name": "__suitability_heatmap__", "figure_path": str(heatmap_path.relative_to(output_dir))})
    pd.DataFrame(figure_manifest).to_csv(output_dir / "analysis" / "H5_S2_T2_figure_manifest.csv", index=False)

    summary_payload.update(
        {
            "feature_sequence_rows": int(len(feature_long)),
            "case_level_suitability_rows": int(len(case_level)),
            "feature_position_summary_rows": int(len(suitability_summary)),
            "figure_count": int(len(figure_manifest)),
            "best_feature_position": suitability_summary.sort_values("mean_suitability", ascending=False).head(1).to_dict(orient="records"),
        }
    )
    write_json(output_dir / "analysis" / "H5_S2_T2_summary.json", summary_payload)
    write_report(output_dir, summary_payload, suitability_summary)
    return summary_payload


def main() -> None:
    output_dir: Path | None = None
    try:
        summary = run(parse_args())
        output_dir = Path(summary["execution_dir"])
        print(json.dumps(to_builtin(summary), indent=2, ensure_ascii=False))
    except Exception:
        if output_dir is not None:
            (output_dir / "logs" / "H5_S2_T2_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()

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


PREFIX = "H6_S2"
TOPIC = "signal_prefix_segment_coverage_visualization"
DEFAULT_METADATA_PATH = "datasets/metadata/heuristic_sequence_peng2026.csv"
DEFAULT_PREFIX_PERCENTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SEGMENTS = ["no_load", "entry", "steady", "exit"]
SEGMENT_LABELS = {
    "no_load": "No-load",
    "entry": "Entry",
    "steady": "Steady",
    "exit": "Exit",
}
STATUS_LABELS = {
    "none": "Not included",
    "partial": "Partially included",
    "full": "Fully included",
}
STATUS_COLORS = {
    "none": "#e5e7eb",
    "partial": "#f59e0b",
    "full": "#2563eb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H6_S2 signal-prefix segment coverage visualization.")
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--case-scope", type=int, nargs="+", default=None)
    parser.add_argument("--prefix-percents", type=float, nargs="+", default=DEFAULT_PREFIX_PERCENTS)
    parser.add_argument("--output-root", default="experiments/executions")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


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
    return output_root / "H6" / "S2" / f"{timestamp}_{TOPIC}"


def make_dirs(output_dir: Path) -> None:
    for rel in ["configs", "data", "analysis", "figures", "logs", "reports"]:
        (output_dir / rel).mkdir(parents=True, exist_ok=True)


def log_progress(message: str) -> None:
    print(f"[{PREFIX}] {message}", file=sys.stderr, flush=True)


def normalize_prefix_percent(value: float) -> float | int:
    value = float(value)
    return int(value) if value.is_integer() else value


def validate_prefix_percents(values: list[float]) -> list[float | int]:
    percents = sorted({normalize_prefix_percent(value) for value in values}, key=float)
    invalid = [value for value in percents if float(value) <= 0.0 or float(value) > 100.0]
    if invalid:
        raise ValueError(f"prefix percents must be in (0, 100], got {invalid}")
    return percents


def build_config(args: argparse.Namespace, timestamp: str, output_dir: Path) -> dict[str, Any]:
    return {
        "experiment": {
            "experiment_number": PREFIX,
            "name": f"{PREFIX}_{TOPIC}",
            "experiment_id": f"{timestamp}_{PREFIX}_{TOPIC}",
            "timestamp": timestamp,
            "execution_dir": str(output_dir),
            "analysis_type": "signal_prefix_segment_coverage_visualization",
        },
        "data": {
            "metadata_path": args.metadata_path,
            "case_scope": args.case_scope,
            "segment_boundaries": {
                "no_load": "[0, idx_noload_end)",
                "entry": "[idx_noload_end, idx_start)",
                "steady": "[idx_start, idx_end)",
                "exit": "[idx_end, signal_length)",
            },
        },
        "prefix": {
            "percents": validate_prefix_percents(args.prefix_percents),
            "rule": "prefix_end_idx = ceil(signal_length * prefix_percent / 100), clipped to [0, signal_length]",
            "start_idx": 0,
        },
        "outputs": {
            "coverage_long": "data/H6_S2_prefix_segment_coverage_long.csv",
            "coverage_summary": "analysis/H6_S2_segment_coverage_summary.csv",
            "dominant_segments": "analysis/H6_S2_dominant_segments.csv",
            "heatmap": "figures/H6_S2_prefix_segment_mean_inclusion_heatmap.{png,svg}",
            "stacked_bar": "figures/H6_S2_prefix_segment_status_stacked_bar.{png,svg}",
        },
    }


def load_metadata(args: argparse.Namespace) -> pd.DataFrame:
    path = resolve(args.metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata path: {path}")
    metadata = pd.read_csv(path)
    required = {"case", "run", "idx_noload_end", "idx_start", "idx_end", "signal_length"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    metadata = metadata.copy()
    if args.case_scope is not None:
        metadata = metadata.loc[metadata["case"].isin(args.case_scope)].copy()
    metadata = metadata.dropna(subset=list(required)).copy()
    for col in ["case", "run", "idx_noload_end", "idx_start", "idx_end", "signal_length"]:
        metadata[col] = metadata[col].astype(int)
    metadata = metadata.loc[metadata["signal_length"] > 0].copy()
    if metadata.empty:
        raise ValueError("No metadata rows after filtering.")
    invalid = metadata.loc[
        (metadata["idx_noload_end"] < 0)
        | (metadata["idx_start"] < metadata["idx_noload_end"])
        | (metadata["idx_end"] < metadata["idx_start"])
        | (metadata["signal_length"] < metadata["idx_end"])
    ]
    if not invalid.empty:
        raise ValueError(f"Invalid segment boundary rows: {invalid[['case', 'run']].head().to_dict(orient='records')}")
    return metadata.sort_values(["case", "run"]).reset_index(drop=True)


def segment_bounds(row: pd.Series) -> dict[str, tuple[int, int]]:
    return {
        "no_load": (0, int(row["idx_noload_end"])),
        "entry": (int(row["idx_noload_end"]), int(row["idx_start"])),
        "steady": (int(row["idx_start"]), int(row["idx_end"])),
        "exit": (int(row["idx_end"]), int(row["signal_length"])),
    }


def classify_ratio(ratio: float) -> str:
    if ratio <= 0:
        return "none"
    if ratio >= 1:
        return "full"
    return "partial"


def build_coverage_long(metadata: pd.DataFrame, prefix_percents: list[float | int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in metadata.itertuples(index=False):
        series = pd.Series(row._asdict())
        bounds = segment_bounds(series)
        signal_length = int(series["signal_length"])
        for percent in prefix_percents:
            prefix_end = int(np.ceil(signal_length * float(percent) / 100.0))
            prefix_end = int(np.clip(prefix_end, 0, signal_length))
            for order, segment in enumerate(SEGMENTS):
                seg_start, seg_end = bounds[segment]
                seg_len = int(seg_end - seg_start)
                overlap = int(max(0, min(prefix_end, seg_end) - seg_start))
                ratio = float(overlap / seg_len) if seg_len > 0 else float("nan")
                status = "zero_length" if seg_len <= 0 else classify_ratio(ratio)
                rows.append(
                    {
                        "case": int(series["case"]),
                        "run": int(series["run"]),
                        "dataset_run_id": f"case_{int(series['case']):02d}_run_{int(series['run']):03d}",
                        "signal_prefix_percent": float(percent),
                        "prefix_start_idx": 0,
                        "prefix_end_idx": prefix_end,
                        "signal_length": signal_length,
                        "segment": segment,
                        "segment_order": order,
                        "segment_start_idx": seg_start,
                        "segment_end_idx": seg_end,
                        "segment_length": seg_len,
                        "overlap_length": overlap,
                        "inclusion_ratio": ratio,
                        "coverage_status": status,
                    }
                )
    return pd.DataFrame(rows).sort_values(["signal_prefix_percent", "segment_order", "case", "run"]).reset_index(drop=True)


def aggregate_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    valid = coverage.loc[coverage["coverage_status"].ne("zero_length")].copy()
    summary = (
        valid.groupby(["signal_prefix_percent", "segment", "segment_order"], as_index=False)
        .agg(
            run_count=("dataset_run_id", "nunique"),
            mean_inclusion_ratio=("inclusion_ratio", "mean"),
            median_inclusion_ratio=("inclusion_ratio", "median"),
            segment_length_mean=("segment_length", "mean"),
            segment_length_median=("segment_length", "median"),
            none_count=("coverage_status", lambda s: int((s == "none").sum())),
            partial_count=("coverage_status", lambda s: int((s == "partial").sum())),
            full_count=("coverage_status", lambda s: int((s == "full").sum())),
        )
        .sort_values(["signal_prefix_percent", "segment_order"])
        .reset_index(drop=True)
    )
    for status in ["none", "partial", "full"]:
        summary[f"{status}_rate"] = summary[f"{status}_count"] / summary["run_count"]
    return summary


def build_dominant_segments(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for percent, group in summary.groupby("signal_prefix_percent"):
        by_ratio = group.sort_values(["mean_inclusion_ratio", "full_rate"], ascending=False).iloc[0]
        by_full = group.sort_values(["full_rate", "mean_inclusion_ratio"], ascending=False).iloc[0]
        mainly_partial = group.loc[group["partial_rate"].eq(group["partial_rate"].max())].sort_values("segment_order").iloc[0]
        rows.append(
            {
                "signal_prefix_percent": float(percent),
                "dominant_by_mean_inclusion": by_ratio["segment"],
                "dominant_mean_inclusion_ratio": float(by_ratio["mean_inclusion_ratio"]),
                "dominant_by_full_rate": by_full["segment"],
                "dominant_full_rate": float(by_full["full_rate"]),
                "most_partial_segment": mainly_partial["segment"],
                "most_partial_rate": float(mainly_partial["partial_rate"]),
                "interpretation": interpretation_for_percent(group),
            }
        )
    return pd.DataFrame(rows)


def interpretation_for_percent(group: pd.DataFrame) -> str:
    percent = float(group["signal_prefix_percent"].iloc[0])
    full_segments = group.loc[group["full_rate"] >= 0.90, "segment"].tolist()
    partial_segments = group.loc[group["partial_rate"] >= 0.50, "segment"].tolist()
    if full_segments:
        full_text = ", ".join(SEGMENT_LABELS[s] for s in full_segments)
    else:
        full_text = "no segment"
    if partial_segments:
        partial_text = ", ".join(SEGMENT_LABELS[s] for s in partial_segments)
    else:
        partial_text = "no segment"
    leader = group.sort_values(["mean_inclusion_ratio", "full_rate"], ascending=False).iloc[0]
    return (
        f"At {percent:g}%, {SEGMENT_LABELS[str(leader['segment'])]} has the highest mean inclusion "
        f"({float(leader['mean_inclusion_ratio']):.2f}); >=90% full inclusion: {full_text}; "
        f">=50% partial inclusion: {partial_text}."
    )


def plot_heatmap(output_dir: Path, summary: pd.DataFrame, dpi: int) -> Path:
    pivot = (
        summary.pivot(index="segment", columns="signal_prefix_percent", values="mean_inclusion_ratio")
        .reindex(SEGMENTS)
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{float(x):g}%" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([SEGMENT_LABELS[segment] for segment in pivot.index])
    ax.set_xlabel("Input length percent from run start")
    ax.set_ylabel("Segment")
    ax.set_title(f"{PREFIX} Mean segment inclusion ratio by signal-prefix length")
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            value = float(pivot.iloc[y, x])
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else "#111827", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, shrink=0.88)
    cbar.set_label("Mean inclusion ratio")
    fig.tight_layout()
    path = output_dir / "figures" / "H6_S2_prefix_segment_mean_inclusion_heatmap.png"
    save_figure_dual(fig, path, dpi=dpi)
    plt.close(fig)
    return path


def plot_status_stacked_bar(output_dir: Path, summary: pd.DataFrame, dpi: int) -> Path:
    percents = sorted(summary["signal_prefix_percent"].unique().tolist())
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, segment in zip(axes_flat, SEGMENTS):
        group = summary.loc[summary["segment"].eq(segment)].sort_values("signal_prefix_percent")
        x = np.arange(len(percents))
        none = group["none_rate"].to_numpy(dtype=float)
        partial = group["partial_rate"].to_numpy(dtype=float)
        full = group["full_rate"].to_numpy(dtype=float)
        ax.bar(x, none, color=STATUS_COLORS["none"], edgecolor="#9ca3af", linewidth=0.4, label=STATUS_LABELS["none"])
        ax.bar(
            x,
            partial,
            bottom=none,
            color=STATUS_COLORS["partial"],
            edgecolor="#92400e",
            linewidth=0.5,
            hatch="///",
            label=STATUS_LABELS["partial"],
        )
        ax.bar(
            x,
            full,
            bottom=none + partial,
            color=STATUS_COLORS["full"],
            edgecolor="#1e3a8a",
            linewidth=0.5,
            label=STATUS_LABELS["full"],
        )
        ax.set_title(SEGMENT_LABELS[segment])
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{float(p):g}%" for p in percents], rotation=45, ha="right")
    axes_flat[0].set_ylabel("Run share")
    axes_flat[2].set_ylabel("Run share")
    axes_flat[2].set_xlabel("Input length percent from run start")
    axes_flat[3].set_xlabel("Input length percent from run start")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle(f"{PREFIX} Segment coverage status by prefix length", y=0.99)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        title="Coverage status",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path = output_dir / "figures" / "H6_S2_prefix_segment_status_stacked_bar.png"
    save_figure_dual(fig, path, dpi=dpi)
    plt.close(fig)
    return path


def write_report(output_dir: Path, summary: dict[str, Any], dominant: pd.DataFrame) -> None:
    lines = [
        f"# {PREFIX} Signal Prefix Segment Coverage Visualization",
        "",
        "## Scope",
        "",
        f"- Metadata path: `{summary['metadata_path']}`",
        f"- Run count: `{summary['run_count']}`",
        f"- Case count: `{summary['case_count']}`",
        f"- Prefix percents: `{summary['prefix_percents']}`",
        "- Prefix rule: start at index 0 and end at `ceil(signal_length * percent / 100)`.",
        "- Segment definitions: no-load, entry, steady, exit from the run-specific boundary columns.",
        "",
        "## Outputs",
        "",
        "- Coverage long table: `data/H6_S2_prefix_segment_coverage_long.csv`",
        "- Aggregated coverage summary: `analysis/H6_S2_segment_coverage_summary.csv`",
        "- Dominant segment interpretation: `analysis/H6_S2_dominant_segments.csv`",
        "- Heatmap: `figures/H6_S2_prefix_segment_mean_inclusion_heatmap.{png,svg}`",
        "- Stacked bar chart: `figures/H6_S2_prefix_segment_status_stacked_bar.{png,svg}`",
        "",
        "## Interpretation By Prefix",
        "",
        "| prefix_percent | dominant_by_mean_inclusion | dominant_by_full_rate | interpretation |",
        "|---:|---|---|---|",
    ]
    for row in dominant.itertuples(index=False):
        lines.append(
            f"| {row.signal_prefix_percent:g} | {row.dominant_by_mean_inclusion} | "
            f"{row.dominant_by_full_rate} | {row.interpretation} |"
        )
    (output_dir / "reports" / "H6_S2_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = args.timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    args.prefix_percents = validate_prefix_percents(args.prefix_percents)
    output_dir = execution_dir(resolve(args.output_root), timestamp)
    if output_dir.exists():
        raise FileExistsError(f"Execution directory already exists: {output_dir}")
    make_dirs(output_dir)
    log_progress(f"Using execution directory: {output_dir}")

    config = build_config(args, timestamp, output_dir)
    write_yaml(output_dir / "configs" / "H6_S2_input_config.yaml", config)
    write_json(output_dir / "logs" / "H6_S2_environment.json", collect_environment())

    log_progress("Loading run segment boundary metadata.")
    metadata = load_metadata(args)
    log_progress(f"Computing coverage for {len(metadata)} runs x {len(args.prefix_percents)} prefix lengths x {len(SEGMENTS)} segments.")
    coverage = build_coverage_long(metadata, args.prefix_percents)
    coverage_summary = aggregate_coverage(coverage)
    dominant = build_dominant_segments(coverage_summary)

    coverage.to_csv(output_dir / "data" / "H6_S2_prefix_segment_coverage_long.csv", index=False)
    coverage_summary.to_csv(output_dir / "analysis" / "H6_S2_segment_coverage_summary.csv", index=False)
    dominant.to_csv(output_dir / "analysis" / "H6_S2_dominant_segments.csv", index=False)

    log_progress("Rendering heatmap and stacked bar figures.")
    heatmap_path = plot_heatmap(output_dir, coverage_summary, args.dpi)
    stacked_path = plot_status_stacked_bar(output_dir, coverage_summary, args.dpi)

    summary: dict[str, Any] = {
        "experiment_id": config["experiment"]["experiment_id"],
        "execution_dir": str(output_dir),
        "metadata_path": args.metadata_path,
        "case_scope": args.case_scope,
        "case_count": int(metadata["case"].nunique()),
        "run_count": int(metadata[["case", "run"]].drop_duplicates().shape[0]),
        "prefix_percents": [float(p) for p in args.prefix_percents],
        "coverage_rows": int(len(coverage)),
        "summary_rows": int(len(coverage_summary)),
        "dominant_rows": int(len(dominant)),
        "heatmap_path": str(heatmap_path.relative_to(output_dir)),
        "stacked_bar_path": str(stacked_path.relative_to(output_dir)),
    }
    write_json(output_dir / "analysis" / "H6_S2_summary.json", summary)
    write_report(output_dir, summary, dominant)
    log_progress("Completed.")
    return summary


def main() -> None:
    output_dir: Path | None = None
    try:
        summary = run(parse_args())
        output_dir = Path(summary["execution_dir"])
        print(json.dumps(to_builtin(summary), indent=2, ensure_ascii=False))
    except Exception:
        if output_dir is not None:
            (output_dir / "logs" / "H6_S2_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()

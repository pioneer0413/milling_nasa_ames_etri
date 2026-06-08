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


PREFIX = "H5_S1"
TOPIC = "steady_cut_length_visualization"
SENSOR = "smcDC"
STEADY_LENGTH_PATHS = {
    1000: "datasets/metadata/heuristic_sequence_peng2026_steady1000.csv",
    2000: "datasets/metadata/heuristic_sequence_peng2026_steady2000.csv",
    3000: "datasets/metadata/heuristic_sequence_peng2026_steady3000.csv",
    4000: "datasets/metadata/heuristic_sequence_peng2026.csv",
    5000: "datasets/metadata/heuristic_sequence_peng2026_steady5000.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H5_S1 steady-cut length visualization EDA.")
    parser.add_argument("--signal-path", default="datasets/processed/mill_signal_data.csv")
    parser.add_argument("--output-root", default="experiments/executions")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--max-plot-points", type=int, default=9000)
    parser.add_argument("--dry-run", action="store_true")
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


def parse_signal(value: Any) -> np.ndarray:
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return arr[np.isfinite(arr)]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def execution_dir(output_root: Path, timestamp: str) -> Path:
    return output_root / "H5" / "S1" / f"{timestamp}_{TOPIC}"


def make_dirs(output_dir: Path) -> None:
    for rel in ["configs", "data", "analysis", "figures", "logs", "reports"]:
        (output_dir / rel).mkdir(parents=True, exist_ok=True)


def build_config(args: argparse.Namespace, timestamp: str, output_dir: Path) -> dict[str, Any]:
    return {
        "experiment": {
            "experiment_number": PREFIX,
            "name": f"{PREFIX}_{TOPIC}",
            "experiment_id": f"{timestamp}_{PREFIX}_{TOPIC}",
            "timestamp": timestamp,
            "execution_dir": str(output_dir),
            "analysis_type": "steady_cut_length_visual_eda",
        },
        "data": {
            "signal_path": args.signal_path,
            "sensor": SENSOR,
            "metadata_paths_by_steady_length": STEADY_LENGTH_PATHS,
        },
        "visualization": {
            "layout": "one row x five columns per case/run",
            "figure_path_pattern": "figures/Case_{case}/Run_{run}.{png,svg}",
            "steady_lengths": sorted(STEADY_LENGTH_PATHS),
            "dpi": args.dpi,
            "max_plot_points": args.max_plot_points,
        },
    }


def load_metadata() -> dict[int, pd.DataFrame]:
    metadata: dict[int, pd.DataFrame] = {}
    for length, path in STEADY_LENGTH_PATHS.items():
        resolved = resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Missing metadata for steady length {length}: {resolved}")
        df = pd.read_csv(resolved)
        required = {"case", "run", "idx_noload_end", "idx_start", "idx_end", "signal_length"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{resolved} missing columns: {sorted(missing)}")
        df = df.copy()
        df["steady_length_requested"] = int(length)
        df["steady_length_actual"] = df["idx_end"].astype(int) - df["idx_start"].astype(int)
        metadata[length] = df
    return metadata


def load_signals(signal_path: Path) -> pd.DataFrame:
    signal = pd.read_csv(signal_path, usecols=["case", "run", SENSOR])
    if SENSOR not in signal.columns:
        raise ValueError(f"{SENSOR} column not found in {signal_path}")
    return signal


def metadata_key_frame(metadata: dict[int, pd.DataFrame]) -> pd.DataFrame:
    key_sets = []
    for length, df in metadata.items():
        keys = df[["case", "run"]].drop_duplicates().assign(steady_length_requested=length)
        key_sets.append(keys)
    combined = pd.concat(key_sets, ignore_index=True)
    counts = combined.groupby(["case", "run"], as_index=False)["steady_length_requested"].nunique()
    missing = counts.loc[counts["steady_length_requested"] != len(STEADY_LENGTH_PATHS)]
    if not missing.empty:
        raise ValueError(f"Some case/run keys do not exist in every metadata file: {missing.head().to_dict(orient='records')}")
    return counts[["case", "run"]].sort_values(["case", "run"]).reset_index(drop=True)


def downsample(values: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or values.size <= max_points:
        return np.arange(values.size), values
    idx = np.linspace(0, values.size - 1, max_points, dtype=int)
    return idx, values[idx]


def y_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.quantile(finite, 0.01))
    hi = float(np.quantile(finite, 0.99))
    if np.isclose(lo, hi):
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1.0)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def row_for(metadata: dict[int, pd.DataFrame], length: int, case: int, run: int) -> pd.Series:
    rows = metadata[length].loc[(metadata[length]["case"] == case) & (metadata[length]["run"] == run)]
    if len(rows) != 1:
        raise ValueError(f"Expected one metadata row for length={length}, case={case}, run={run}; got {len(rows)}")
    return rows.iloc[0]


def draw_panel(ax: plt.Axes, x_plot: np.ndarray, y_plot: np.ndarray, signal_length: int, row: pd.Series, length: int) -> None:
    idx_noload_end = int(row["idx_noload_end"])
    idx_start = int(row["idx_start"])
    idx_end = int(row["idx_end"])

    ax.axvspan(0, idx_noload_end, color="#9ca3af", alpha=0.14, linewidth=0)
    ax.axvspan(idx_noload_end, idx_start, color="#f59e0b", alpha=0.12, linewidth=0)
    ax.axvspan(idx_start, idx_end, color="#10b981", alpha=0.30, linewidth=0)
    ax.axvspan(idx_end, signal_length, color="#ef4444", alpha=0.10, linewidth=0)
    ax.plot(x_plot, y_plot, color="#1f2937", linewidth=0.65, alpha=0.90)

    ax.axvline(idx_noload_end, color="#d97706", linewidth=0.9, alpha=0.75)
    ax.axvline(idx_start, color="#047857", linewidth=1.1, alpha=0.85)
    ax.axvline(idx_end, color="#047857", linewidth=1.1, alpha=0.85)
    ax.set_title(f"steady {length}", fontsize=10, pad=6)
    ax.set_xlabel("sample index", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, color="#e5e7eb", linewidth=0.45, alpha=0.65)
    ax.text(
        0.02,
        0.96,
        f"{idx_start}:{idx_end}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="#065f46",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "boxstyle": "round,pad=0.22", "alpha": 0.86},
    )


def save_run_figure(
    output_dir: Path,
    metadata: dict[int, pd.DataFrame],
    signal_row: pd.Series,
    case: int,
    run: int,
    dpi: int,
    max_plot_points: int,
) -> Path:
    signal = parse_signal(signal_row[SENSOR])
    x_plot, y_plot = downsample(signal, max_plot_points)
    ylim = y_limits(signal)
    lengths = sorted(STEADY_LENGTH_PATHS)

    first_row = row_for(metadata, lengths[0], case, run)
    process_bits = []
    for col in ["time", "DOC", "feed", "material_name"]:
        if col in first_row.index:
            process_bits.append(f"{col}={first_row[col]}")
    subtitle = ", ".join(process_bits)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8), sharex=True, sharey=True, squeeze=False)
    for ax, length in zip(axes[0], lengths):
        meta_row = row_for(metadata, length, case, run)
        signal_length = int(meta_row.get("signal_length", signal.size))
        draw_panel(ax, x_plot, y_plot, signal_length, meta_row, length)
        ax.set_xlim(0, max(signal.size, signal_length))
        ax.set_ylim(*ylim)
    axes[0, 0].set_ylabel(f"{SENSOR}", fontsize=9)

    handles = [
        Line2D([0], [0], color="#9ca3af", linewidth=8, alpha=0.35, label="no-load"),
        Line2D([0], [0], color="#f59e0b", linewidth=8, alpha=0.35, label="entry"),
        Line2D([0], [0], color="#10b981", linewidth=8, alpha=0.45, label="steady"),
        Line2D([0], [0], color="#ef4444", linewidth=8, alpha=0.25, label="exit/tail"),
        Line2D([0], [0], color="#1f2937", linewidth=1.2, label=SENSOR),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"{PREFIX} Case {case} / Run {run} steady-cut length comparison", fontsize=13, y=0.99)
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center", va="center", fontsize=8, color="#4b5563")
    fig.tight_layout(rect=(0.01, 0.08, 0.99, 0.90))

    figure_path = output_dir / "figures" / f"Case_{case}" / f"Run_{run}.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_dual(fig, figure_path, dpi=dpi)
    plt.close(fig)
    return figure_path


def build_boundary_table(metadata: dict[int, pd.DataFrame]) -> pd.DataFrame:
    tables = []
    for length, df in metadata.items():
        cols = [
            "case",
            "run",
            "steady_length_requested",
            "steady_length_actual",
            "idx_noload_end",
            "idx_start",
            "idx_end",
            "index_fixed",
            "signal_length",
            "status",
            "note",
            "time",
            "DOC",
            "feed",
            "material_name",
        ]
        existing = [c for c in cols if c in df.columns]
        tables.append(df[existing].copy())
    return pd.concat(tables, ignore_index=True).sort_values(["case", "run", "steady_length_requested"])


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    report = f"""# H5_S1 Steady-cut Length Visualization

## Scope

- Sensor: `{SENSOR}`
- Steady-cut lengths: `{summary["steady_lengths"]}`
- Case/run figures: `{summary["figure_count"]}`
- Figure layout: one row with five columns, one column per steady-cut length.
- Figure path pattern: `figures/Case_#/Run_#.{png,svg}`

## Output

- Boundary table: `data/H5_S1_steady_cut_boundaries.csv`
- Figure manifest: `analysis/H5_S1_figure_manifest.csv`
- Summary: `analysis/H5_S1_summary.json`

## Notes

Each panel plots the full `{SENSOR}` signal and shades no-load, entry, steady, and exit/tail regions. The green shaded interval is the steady-cut segment for the requested length.
"""
    (output_dir / "reports" / "H5_S1_report.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = args.timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_root = resolve(args.output_root)
    output_dir = execution_dir(output_root, timestamp)
    if output_dir.exists():
        raise FileExistsError(f"Execution directory already exists: {output_dir}")
    make_dirs(output_dir)

    config = build_config(args, timestamp, output_dir)
    write_yaml(output_dir / "configs" / "H5_S1_input_config.yaml", config)
    write_json(output_dir / "logs" / "H5_S1_environment.json", collect_environment())

    metadata = load_metadata()
    signal = load_signals(resolve(args.signal_path))
    keys = metadata_key_frame(metadata)
    signal_keys = signal[["case", "run"]].drop_duplicates()
    missing_signal = keys.merge(signal_keys, on=["case", "run"], how="left", indicator=True)
    missing_signal = missing_signal.loc[missing_signal["_merge"].eq("left_only"), ["case", "run"]]
    if not missing_signal.empty:
        raise ValueError(f"Missing signal rows: {missing_signal.head().to_dict(orient='records')}")

    boundary_table = build_boundary_table(metadata)
    boundary_table.to_csv(output_dir / "data" / "H5_S1_steady_cut_boundaries.csv", index=False)

    summary = {
        "experiment_id": config["experiment"]["experiment_id"],
        "execution_dir": str(output_dir),
        "sensor": SENSOR,
        "steady_lengths": sorted(STEADY_LENGTH_PATHS),
        "case_count": int(keys["case"].nunique()),
        "case_run_count": int(len(keys)),
        "dry_run": bool(args.dry_run),
        "figure_count": 0,
    }
    if args.dry_run:
        write_json(output_dir / "analysis" / "H5_S1_summary.json", summary)
        write_report(output_dir, summary)
        return summary

    signal_indexed = signal.set_index(["case", "run"], drop=False)
    manifest: list[dict[str, Any]] = []
    for item in keys.itertuples(index=False):
        case = int(item.case)
        run_id = int(item.run)
        signal_row = signal_indexed.loc[(case, run_id)]
        figure_path = save_run_figure(output_dir, metadata, signal_row, case, run_id, args.dpi, args.max_plot_points)
        manifest.append({"case": case, "run": run_id, "figure_path": str(figure_path.relative_to(output_dir))})

    manifest_df = pd.DataFrame(manifest).sort_values(["case", "run"])
    manifest_df.to_csv(output_dir / "analysis" / "H5_S1_figure_manifest.csv", index=False)
    case_counts = manifest_df.groupby("case", as_index=False).size().rename(columns={"size": "figure_count"})
    case_counts.to_csv(output_dir / "analysis" / "H5_S1_case_figure_counts.csv", index=False)
    summary["figure_count"] = int(len(manifest_df))
    summary["figures_by_case"] = {int(row.case): int(row.figure_count) for row in case_counts.itertuples(index=False)}
    write_json(output_dir / "analysis" / "H5_S1_summary.json", summary)
    write_report(output_dir, summary)
    return summary


def main() -> None:
    args = parse_args()
    output_dir: Path | None = None
    try:
        summary = run(args)
        output_dir = Path(summary["execution_dir"])
        print(json.dumps(to_builtin(summary), indent=2, ensure_ascii=False))
    except Exception:
        if output_dir is not None:
            (output_dir / "logs" / "H5_S1_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()

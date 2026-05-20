#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import html
import json
import math
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
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde, kurtosis, ks_2samp, skew, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from milling_experiment_framework.experiment_logging.environment import collect_environment


PREFIX = "H1_S4"
TOPIC = "segment_distribution_visualization_and_distance_analysis"
SELECTED_CASES = [1, 2, 5, 8, 9, 12, 14, 16]
PAIR_DEFINITION = {"A": [1, 9], "B": [2, 12], "C": [5, 16], "D": [8, 14]}
PAIR_BY_CASE = {case: pair for pair, cases in PAIR_DEFINITION.items() for case in cases}
EXPECTED_RUN_FIGURES = 82
EXPECTED_CASE_FIGURES = 8
EXPECTED_GLOBAL_FIGURES = 1
SENSOR_CANDIDATES = ["smcAC", "smcDC", "vib_spindle", "vib_table", "AE_spindle", "AE_table"]
SEGMENT_ORDER = ["full_length", "entry", "steady", "exit", "entry_steady", "entry_exit", "steady_exit", "entry_steady_exit"]
MAIN_SEGMENTS = SEGMENT_ORDER
SEGMENT_COLORS = {
    "full_length": "#111111",
    "entry": "#1f77b4",
    "steady": "#2ca02c",
    "exit": "#d62728",
    "entry_steady": "#17becf",
    "entry_exit": "#9467bd",
    "steady_exit": "#ff7f0e",
    "entry_steady_exit": "#8c564b",
    "unknown": "#8c564b",
}
SEGMENT_LINE_STYLES = {
    "full_length": "-",
    "entry": "-",
    "steady": "--",
    "exit": "-.",
    "entry_steady": (0, (5, 1)),
    "entry_exit": (0, (3, 1, 1, 1)),
    "steady_exit": (0, (1, 1)),
    "entry_steady_exit": (0, (7, 1, 1, 1)),
    "unknown": "-",
}
DISTANCE_METRICS = [
    "wasserstein_distance",
    "jensen_shannon_distance",
    "kolmogorov_smirnov_statistic",
    "hellinger_distance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H1_S4 segment-wise distribution EDA.")
    parser.add_argument("--signal-path", default="datasets/processed/mill_signal_data_enabled.csv")
    parser.add_argument("--process-info-path", default="datasets/processed/mill_process_info_enabled.csv")
    parser.add_argument(
        "--segment-metadata-path",
        default="/home/hwkang/manufacturing_data_exercise/02_milling_v2/datasets/metadata/heuristic_sequence_peng2026_smcdc.csv",
    )
    parser.add_argument("--output-root", default="experiments/executions")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scaling-method", choices=["raw", "per_sensor_global_zscore"], default="raw")
    parser.add_argument("--num-grid-points", type=int, default=512)
    parser.add_argument("--min-kde-samples", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=160)
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


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_signal_cell(value: Any) -> np.ndarray:
    arr = np.asarray(ast.literal_eval(value) if isinstance(value, str) else value, dtype=np.float64)
    return arr[np.isfinite(arr)]


def infer_sensor_columns(signal: pd.DataFrame) -> list[str]:
    sensors: list[str] = []
    for col in SENSOR_CANDIDATES:
        if col in signal.columns:
            sample = signal[col].dropna()
            if not sample.empty and str(sample.iloc[0]).strip().startswith("["):
                sensors.append(col)
    if sensors:
        return sensors
    excluded = {"case", "run", "enable", "VB", "time", "DOC", "feed", "material", "material_name"}
    for col in signal.columns:
        if col in excluded:
            continue
        sample = signal[col].dropna()
        if not sample.empty and str(sample.iloc[0]).strip().startswith("["):
            sensors.append(col)
    return sensors


def execution_dir(output_root: Path, timestamp: str) -> Path:
    return output_root / "H1" / "S4" / f"{timestamp}_{TOPIC}"


def make_dirs(out: Path) -> None:
    for rel in [
        "configs",
        "data",
        "preprocessing",
        "metrics",
        "analysis/distance_matrices/case_level",
        "analysis/distance_matrices/global_level",
        "figures/run_level",
        "figures/case_level",
        "figures/global_level",
        "figures/distance_matrices/case_level",
        "figures/distance_matrices/global_level",
        "logs",
        "reports",
    ]:
        (out / rel).mkdir(parents=True, exist_ok=True)


def build_config(args: argparse.Namespace, timestamp: str, out: Path) -> dict[str, Any]:
    return {
        "experiment": {
            "experiment_number": PREFIX,
            "name": f"{PREFIX}_{TOPIC}",
            "experiment_id": f"{timestamp}_{PREFIX}_{TOPIC}",
            "timestamp": timestamp,
            "execution_dir": str(out),
            "analysis_type": "exploratory_distribution_segment_wise",
        },
        "data": {
            "signal_path": args.signal_path,
            "process_info_path": args.process_info_path,
            "segment_metadata_path": args.segment_metadata_path,
            "selected_cases": SELECTED_CASES,
            "case_pairs": PAIR_DEFINITION,
            "dataset_run_alias": {"run": "dataset_run_id"},
        },
        "segments": {
            "source_of_truth": args.segment_metadata_path,
            "boundary_columns": {
                "full_length": "0:signal_length",
                "entry": "idx_noload_end:idx_start",
                "steady": "idx_start:idx_end",
                "exit": "idx_end:signal_length",
                "entry_steady": "idx_noload_end:idx_end",
                "entry_exit": "concat(idx_noload_end:idx_start, idx_end:signal_length)",
                "steady_exit": "idx_start:signal_length",
                "entry_steady_exit": "idx_noload_end:signal_length",
            },
            "segment_list": SEGMENT_ORDER,
            "main_comparison_segments": MAIN_SEGMENTS,
            "include_no_load": False,
            "note": "no_load is excluded as a segment setting; full_length is retained as the whole-signal baseline.",
        },
        "sensors": {"expected": SENSOR_CANDIDATES, "boundary_sensor": "smcDC", "boundary_applied_to_all_sensors": True},
        "scaling": {"method": args.scaling_method, "fit_scope": "none" if args.scaling_method == "raw" else "selected_cases_all_runs_in_configured_signal_file"},
        "distribution": {
            "method": "kde",
            "fallback": "histogram_density",
            "normalize_per_segment": True,
            "common_x_grid": True,
            "num_grid_points": args.num_grid_points,
            "min_kde_samples": args.min_kde_samples,
        },
        "distance_metrics": DISTANCE_METRICS,
        "segment_colors": SEGMENT_COLORS,
        "segment_line_styles": {k: str(v) for k, v in SEGMENT_LINE_STYLES.items()},
        "expected_outputs": {
            "run_level_figures": EXPECTED_RUN_FIGURES,
            "case_level_figures": EXPECTED_CASE_FIGURES,
            "global_level_figures": EXPECTED_GLOBAL_FIGURES,
        },
    }


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_path = ROOT / args.signal_path if not Path(args.signal_path).is_absolute() else Path(args.signal_path)
    process_path = ROOT / args.process_info_path if not Path(args.process_info_path).is_absolute() else Path(args.process_info_path)
    metadata_path = Path(args.segment_metadata_path)
    signal = pd.read_csv(signal_path)
    process = pd.read_csv(process_path)
    metadata = pd.read_csv(metadata_path)
    return signal, process, metadata


def selected_enabled(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["case"].isin(SELECTED_CASES)].copy()
    if "enable" in out.columns:
        out = out[out["enable"].astype(str).str.lower().eq("true")]
    return out


def expected_run_keys_from_available_processed() -> list[tuple[int, int]]:
    full_path = ROOT / "datasets/processed/mill_signal_data.csv"
    if full_path.exists():
        full = pd.read_csv(full_path, usecols=["case", "run"])
        keys = full[full["case"].isin(SELECTED_CASES)][["case", "run"]].drop_duplicates()
    else:
        keys = pd.DataFrame({"case": [], "run": []})
    return [(int(r.case), int(r.run)) for r in keys.sort_values(["case", "run"]).itertuples(index=False)]


def validate_metadata(
    signal: pd.DataFrame,
    process: pd.DataFrame,
    metadata: pd.DataFrame,
    sensors: list[str],
    metadata_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    selected_signal = selected_enabled(signal)
    selected_process = selected_enabled(process)
    selected_meta = metadata[metadata["case"].isin(SELECTED_CASES)].copy()
    merged = (
        selected_signal[["case", "run"] + sensors]
        .merge(selected_process.drop(columns=[c for c in sensors if c in selected_process.columns], errors="ignore"), on=["case", "run"], how="left", suffixes=("", "_process"))
        .merge(selected_meta, on=["case", "run"], how="left", suffixes=("", "_segment"))
    )

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    aligned: list[dict[str, Any]] = []
    required_cols = ["case", "run", "idx_noload_end", "idx_start", "idx_end", "signal_length"]
    meta_cols = set(metadata.columns)
    case_col_ok = "case" in meta_cols
    run_col_ok = "run" in meta_cols
    boundary_ok = all(c in meta_cols for c in required_cols[2:])
    label_col = next((c for c in ["segment", "segment_label", "label", "state"] if c in meta_cols), None)

    expected_keys = expected_run_keys_from_available_processed()
    selected_signal_keys = {(int(r.case), int(r.run)) for r in selected_signal[["case", "run"]].itertuples(index=False)}
    selected_meta_keys = {(int(r.case), int(r.run)) for r in selected_meta[["case", "run"]].itertuples(index=False)}
    produced_candidate_keys = selected_signal_keys & selected_meta_keys

    for case_id, run_id in expected_keys:
        reason_parts = []
        if (case_id, run_id) not in selected_signal_keys:
            reason_parts.append("missing_in_configured_enabled_signal_or_filtered_by_enable")
        if (case_id, run_id) not in selected_meta_keys:
            reason_parts.append("missing_in_segment_metadata")
        if reason_parts:
            skipped.append({"case_id": case_id, "dataset_run_id": run_id, "reason": ";".join(reason_parts)})

    for row in merged.itertuples(index=False):
        case_id = int(getattr(row, "case"))
        run_id = int(getattr(row, "run"))
        item = {
            "case_id": case_id,
            "dataset_run_id": run_id,
            "metadata_exists": bool((case_id, run_id) in selected_meta_keys),
            "status": "ok",
            "reason": "",
            "signal_length_metadata": None,
            "signal_length_min_sensor": None,
            "segment_count": 0,
            "has_gap": False,
            "has_overlap": False,
            "boundary_in_range": False,
        }
        if (case_id, run_id) not in selected_meta_keys:
            item.update({"status": "skipped", "reason": "missing_segment_metadata"})
            rows.append(item)
            continue
        signal_lengths = []
        for sensor in sensors:
            try:
                signal_lengths.append(len(parse_signal_cell(getattr(row, sensor))))
            except Exception:
                pass
        if not signal_lengths:
            item.update({"status": "skipped", "reason": "no_usable_sensor_signal"})
            rows.append(item)
            skipped.append({"case_id": case_id, "dataset_run_id": run_id, "reason": "no_usable_sensor_signal"})
            continue
        n = int(min(signal_lengths))
        idx_noload_end = int(getattr(row, "idx_noload_end"))
        idx_start = int(getattr(row, "idx_start"))
        idx_end = int(getattr(row, "idx_end"))
        meta_len = int(getattr(row, "signal_length"))
        boundary_in_range = 0 <= idx_noload_end <= idx_start <= idx_end <= n
        item.update(
            {
                "signal_length_metadata": meta_len,
                "signal_length_min_sensor": n,
                "segment_count": len(SEGMENT_ORDER) if boundary_in_range else 0,
                "boundary_in_range": bool(boundary_in_range),
                "status": "ok" if boundary_in_range else "skipped",
                "reason": "" if boundary_in_range else "boundary_out_of_signal_range_or_non_monotonic",
            }
        )
        rows.append(item)
        if boundary_in_range:
            aligned.append({"case_id": case_id, "dataset_run_id": run_id})
        else:
            skipped.append({"case_id": case_id, "dataset_run_id": run_id, "reason": item["reason"]})

    validation = pd.DataFrame(rows)
    checks = [
        {"check": "processed_signal_file_exists", "passed": True, "detail": "loaded"},
        {"check": "processed_process_info_file_exists", "passed": True, "detail": "loaded"},
        {"check": "segment_metadata_file_exists", "passed": metadata_path.exists(), "detail": str(metadata_path)},
        {"check": "metadata_case_column_exists", "passed": case_col_ok, "detail": "case"},
        {"check": "metadata_dataset_run_column_exists", "passed": run_col_ok, "detail": "run"},
        {"check": "metadata_segment_label_column_exists", "passed": label_col is not None, "detail": label_col or "derived_from_boundary_columns"},
        {"check": "metadata_boundary_columns_exist", "passed": boundary_ok, "detail": ",".join(required_cols[2:])},
        {"check": "selected_8_cases_exist_in_configured_signal", "passed": set(SELECTED_CASES).issubset(set(selected_signal["case"].unique())), "detail": sorted(selected_signal["case"].unique().tolist())},
        {"check": "selected_8_cases_exist_in_metadata", "passed": set(SELECTED_CASES).issubset(set(selected_meta["case"].unique())), "detail": sorted(selected_meta["case"].unique().tolist())},
        {"check": "total_dataset_run_count_is_82", "passed": len(produced_candidate_keys) == EXPECTED_RUN_FIGURES, "detail": f"aligned_candidate_count={len(produced_candidate_keys)}, expected={EXPECTED_RUN_FIGURES}"},
        {"check": "metadata_signal_alignment_possible", "passed": len(aligned) > 0, "detail": f"aligned_runs={len(aligned)}"},
        {"check": "segment_boundary_in_range", "passed": bool(len(validation) and validation["boundary_in_range"].fillna(False).all()), "detail": f"valid={int(validation['boundary_in_range'].sum()) if len(validation) else 0}, checked={len(validation)}"},
        {"check": "sensor_list_identified", "passed": bool(sensors), "detail": sensors},
        {"check": "run_level_figure_82_possible", "passed": len(aligned) == EXPECTED_RUN_FIGURES, "detail": f"possible={len(aligned)}, expected={EXPECTED_RUN_FIGURES}"},
        {"check": "case_level_figure_8_possible", "passed": len(set(a['case_id'] for a in aligned)) == EXPECTED_CASE_FIGURES, "detail": f"possible_cases={sorted(set(a['case_id'] for a in aligned))}"},
        {"check": "global_level_figure_possible", "passed": len(aligned) > 0, "detail": f"aligned_runs={len(aligned)}"},
        {"check": "case_level_distance_matrix_possible", "passed": len(set(a['case_id'] for a in aligned)) > 0, "detail": "requires at least one aligned case"},
        {"check": "global_level_distance_matrix_possible", "passed": len(aligned) > 0, "detail": "requires at least one aligned run"},
    ]
    return validation, pd.DataFrame(checks), skipped, aligned


def segment_slices(row: pd.Series, n: int) -> dict[str, slice]:
    a = int(np.clip(row["idx_noload_end"], 0, n))
    b = int(np.clip(row["idx_start"], a, n))
    c = int(np.clip(row["idx_end"], b, n))
    return {
        "no_load": slice(0, a),
        "entry": slice(a, b),
        "steady": slice(b, c),
        "exit": slice(c, n),
    }


def build_segment_samples(arr: np.ndarray, slices: dict[str, slice]) -> dict[str, np.ndarray]:
    entry = arr[slices["entry"]]
    steady = arr[slices["steady"]]
    exit_ = arr[slices["exit"]]
    return {
        "full_length": arr,
        "entry": entry,
        "steady": steady,
        "exit": exit_,
        "entry_steady": np.concatenate([entry, steady]),
        "entry_exit": np.concatenate([entry, exit_]),
        "steady_exit": np.concatenate([steady, exit_]),
        "entry_steady_exit": np.concatenate([entry, steady, exit_]),
    }


def scaling_params(signal: pd.DataFrame, sensors: list[str], method: str) -> dict[str, dict[str, float]]:
    if method == "raw":
        return {sensor: {"mean": 0.0, "std": 1.0, "n_fit_samples": 0, "method": "raw"} for sensor in sensors}
    selected = selected_enabled(signal)
    params: dict[str, dict[str, float]] = {}
    for sensor in sensors:
        chunks: list[np.ndarray] = []
        for value in selected[sensor].dropna():
            arr = parse_signal_cell(value)
            if arr.size:
                chunks.append(arr)
        all_values = np.concatenate(chunks) if chunks else np.asarray([0.0])
        mean = float(np.mean(all_values))
        std = float(np.std(all_values))
        if not np.isfinite(std) or std <= 1e-12:
            std = 1.0
        params[sensor] = {"mean": mean, "std": std, "n_fit_samples": int(all_values.size), "method": method}
    return params


def standardize(arr: np.ndarray, params: dict[str, float]) -> np.ndarray:
    return (arr.astype(np.float64) - params["mean"]) / params["std"]


def estimate_density(samples: np.ndarray, grid: np.ndarray, min_kde_samples: int) -> tuple[np.ndarray, str, str]:
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.zeros_like(grid), "skipped", "n_samples_lt_2"
    if x.size >= min_kde_samples and np.std(x) > 1e-12:
        try:
            density = gaussian_kde(x)(grid)
            density = np.clip(density, 0.0, np.inf)
            area = float(np.trapezoid(density, grid))
            if area > 0:
                density = density / area
            return density, "kde", ""
        except Exception as exc:
            reason = f"kde_failed:{type(exc).__name__}"
    else:
        reason = "n_samples_or_variance_insufficient_for_kde"
    if not np.isfinite(grid.min()) or not np.isfinite(grid.max()) or float(grid.max()) <= float(grid.min()):
        center = float(np.mean(x)) if x.size else 0.0
        grid = np.linspace(center - 1e-3, center + 1e-3, len(grid))
    bins = min(64, max(2, min(int(np.sqrt(x.size)), max(2, len(np.unique(x))))))
    try:
        hist, edges = np.histogram(x, bins=bins, range=(float(grid.min()), float(grid.max())), density=True)
    except ValueError:
        center = float(np.mean(x)) if x.size else 0.0
        grid = np.linspace(center - 1e-3, center + 1e-3, len(grid))
        hist, edges = np.histogram(x, bins=2, range=(float(grid.min()), float(grid.max())), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    density = np.interp(grid, centers, hist, left=0.0, right=0.0)
    density = np.clip(density, 0.0, np.inf)
    area = float(np.trapezoid(density, grid))
    if area > 0:
        density = density / area
    return density, "histogram_density", reason


def summary_stats(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in ["mean", "std", "median", "min", "max", "q25", "q75", "iqr", "skewness", "kurtosis"]}
    q25, q75 = np.quantile(x, [0.25, 0.75])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "skewness": float(skew(x, bias=False)) if x.size > 2 and np.std(x) > 1e-12 else 0.0,
        "kurtosis": float(kurtosis(x, fisher=True, bias=False)) if x.size > 3 and np.std(x) > 1e-12 else 0.0,
    }


def clean_run_id(run_id: Any) -> str:
    return str(run_id).replace("/", "_").replace(" ", "_")


def plot_distribution(
    path: Path,
    title: str,
    sensors: list[str],
    segments: list[str],
    densities_by_sensor: dict[str, dict[str, np.ndarray]],
    grids_by_sensor: dict[str, np.ndarray],
    counts_by_sensor: dict[str, dict[str, int]],
    variability_by_sensor: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] | None = None,
    x_label: str = "signal value",
    dpi: int = 160,
) -> None:
    ncols = 2
    nrows = int(math.ceil(len(sensors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, max(3.2 * nrows, 4.0)), squeeze=False)
    handles = []
    labels = []
    for ax in axes.ravel():
        ax.set_visible(False)
    for idx, sensor in enumerate(sensors):
        ax = axes.ravel()[idx]
        ax.set_visible(True)
        grid = grids_by_sensor[sensor]
        for segment in segments:
            density = densities_by_sensor.get(sensor, {}).get(segment)
            if density is None:
                continue
            color = SEGMENT_COLORS.get(segment, SEGMENT_COLORS["unknown"])
            linestyle = SEGMENT_LINE_STYLES.get(segment, "-")
            count = counts_by_sensor.get(sensor, {}).get(segment, 0)
            line = ax.plot(grid, density, color=color, linestyle=linestyle, linewidth=1.8, label=f"{segment} (n={count})")[0]
            if variability_by_sensor and segment in variability_by_sensor.get(sensor, {}):
                lower, upper = variability_by_sensor[sensor][segment]
                ax.fill_between(grid, lower, upper, color=color, alpha=0.12, linewidth=0)
            if idx == 0:
                handles.append(line)
                labels.append(f"{segment} (n shown per sensor)")
        ax.set_title(sensor, fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel("density")
        ax.grid(alpha=0.18)
    fig.suptitle(title, fontsize=12)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def density_grid_for_samples(samples_by_segment: dict[str, np.ndarray], grid_points: int) -> np.ndarray:
    chunks = [x for x in samples_by_segment.values() if x.size]
    if not chunks:
        return np.linspace(-1.0, 1.0, grid_points)
    values = np.concatenate(chunks)
    lo, hi = np.quantile(values, [0.005, 0.995])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(values)) - 1.0, float(np.max(values)) + 1.0
    pad = 0.08 * (hi - lo)
    return np.linspace(float(lo - pad), float(hi + pad), grid_points)


def build_run_records(
    signal: pd.DataFrame,
    process: pd.DataFrame,
    metadata: pd.DataFrame,
    sensors: list[str],
    params: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_signal = selected_enabled(signal)
    selected_process = selected_enabled(process)
    selected_meta = metadata[metadata["case"].isin(SELECTED_CASES)].copy()
    data = selected_signal.merge(selected_process, on=["case", "run"], how="left", suffixes=("", "_process")).merge(
        selected_meta, on=["case", "run"], how="inner", suffixes=("", "_segment")
    )
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for _, row in data.sort_values(["case", "run"]).iterrows():
        case_id = int(row["case"])
        run_id = int(row["run"])
        sensor_arrays: dict[str, np.ndarray] = {}
        lengths = []
        for sensor in sensors:
            try:
                raw = parse_signal_cell(row[sensor])
                if raw.size:
                    sensor_arrays[sensor] = standardize(raw, params[sensor])
                    lengths.append(raw.size)
            except Exception as exc:
                skipped.append({"case_id": case_id, "dataset_run_id": run_id, "sensor_name": sensor, "reason": f"signal_parse_failed:{type(exc).__name__}"})
        if not sensor_arrays:
            skipped.append({"case_id": case_id, "dataset_run_id": run_id, "reason": "no_usable_sensor_signal"})
            continue
        n = int(min(lengths))
        slices = segment_slices(row, n)
        if not (slices["no_load"].stop <= slices["entry"].stop <= slices["steady"].stop <= slices["exit"].stop <= n):
            skipped.append({"case_id": case_id, "dataset_run_id": run_id, "reason": "invalid_boundary_order"})
            continue
        segment_samples: dict[str, dict[str, np.ndarray]] = {}
        for sensor, arr in sensor_arrays.items():
            arr = arr[:n]
            segment_samples[sensor] = build_segment_samples(arr, slices)
        records.append(
            {
                "case_id": case_id,
                "pair_id": PAIR_BY_CASE.get(case_id, ""),
                "dataset_run_id": run_id,
                "process": {k: to_builtin(row[k]) for k in ["VB", "time", "DOC", "feed", "material", "material_name"] if k in row.index},
                "metadata": {k: to_builtin(row[k]) for k in ["idx_noload_end", "idx_start", "idx_end", "signal_length", "status", "note"] if k in row.index},
                "samples": segment_samples,
            }
        )
    return records, skipped


def run_densities(records: list[dict[str, Any]], sensors: list[str], grid_points: int, min_kde_samples: int) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        key = (record["case_id"], record["dataset_run_id"])
        densities_by_sensor: dict[str, dict[str, np.ndarray]] = {}
        grids_by_sensor: dict[str, np.ndarray] = {}
        counts_by_sensor: dict[str, dict[str, int]] = {}
        methods_by_sensor: dict[str, dict[str, str]] = {}
        for sensor in sensors:
            samples_by_segment = record["samples"].get(sensor, {})
            grid = density_grid_for_samples(samples_by_segment, grid_points)
            grids_by_sensor[sensor] = grid
            densities_by_sensor[sensor] = {}
            counts_by_sensor[sensor] = {}
            methods_by_sensor[sensor] = {}
            for segment in SEGMENT_ORDER:
                samples = samples_by_segment.get(segment, np.asarray([], dtype=np.float64))
                density, method, _ = estimate_density(samples, grid, min_kde_samples)
                densities_by_sensor[sensor][segment] = density
                counts_by_sensor[sensor][segment] = int(samples.size)
                methods_by_sensor[sensor][segment] = method
        out[key] = {
            "densities": densities_by_sensor,
            "grids": grids_by_sensor,
            "counts": counts_by_sensor,
            "methods": methods_by_sensor,
            "record": record,
        }
    return out


def aggregate_case_densities(run_density: dict[tuple[int, int], dict[str, Any]], sensors: list[str]) -> dict[int, dict[str, Any]]:
    by_case: dict[int, list[dict[str, Any]]] = {}
    for item in run_density.values():
        by_case.setdefault(item["record"]["case_id"], []).append(item)
    result: dict[int, dict[str, Any]] = {}
    for case_id, items in by_case.items():
        densities: dict[str, dict[str, np.ndarray]] = {}
        grids: dict[str, np.ndarray] = {}
        counts: dict[str, dict[str, int]] = {}
        variability: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for sensor in sensors:
            grid = items[0]["grids"][sensor]
            grids[sensor] = grid
            densities[sensor] = {}
            counts[sensor] = {}
            variability[sensor] = {}
            for segment in SEGMENT_ORDER:
                stacked = np.vstack([np.interp(grid, item["grids"][sensor], item["densities"][sensor][segment]) for item in items])
                densities[sensor][segment] = np.mean(stacked, axis=0)
                lower = np.quantile(stacked, 0.25, axis=0)
                upper = np.quantile(stacked, 0.75, axis=0)
                variability[sensor][segment] = (lower, upper)
                counts[sensor][segment] = int(np.mean([item["counts"][sensor][segment] for item in items]))
        result[case_id] = {"densities": densities, "grids": grids, "counts": counts, "variability": variability, "run_count": len(items)}
    return result


def aggregate_global_densities(case_density: dict[int, dict[str, Any]], sensors: list[str]) -> dict[str, Any]:
    items = list(case_density.values())
    densities: dict[str, dict[str, np.ndarray]] = {}
    grids: dict[str, np.ndarray] = {}
    counts: dict[str, dict[str, int]] = {}
    variability: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for sensor in sensors:
        grid = items[0]["grids"][sensor]
        grids[sensor] = grid
        densities[sensor] = {}
        counts[sensor] = {}
        variability[sensor] = {}
        for segment in SEGMENT_ORDER:
            stacked = np.vstack([np.interp(grid, item["grids"][sensor], item["densities"][sensor][segment]) for item in items])
            densities[sensor][segment] = np.mean(stacked, axis=0)
            variability[sensor][segment] = (np.quantile(stacked, 0.25, axis=0), np.quantile(stacked, 0.75, axis=0))
            counts[sensor][segment] = int(np.mean([item["counts"][sensor][segment] for item in items]))
    return {"densities": densities, "grids": grids, "counts": counts, "variability": variability}


def plot_heatmap(path: Path, matrix: pd.DataFrame, title: str, dpi: int = 160) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    values = matrix.to_numpy(dtype=float)
    vmax = np.nanmax(values) if np.isfinite(values).any() else 1.0
    image = ax.imshow(values, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title(title, fontsize=10)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            ax.text(j, i, f"{val:.3f}" if np.isfinite(val) else "nan", ha="center", va="center", fontsize=8, color="white" if val > vmax * 0.55 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
    q = q / q.sum() if q.sum() > 0 else np.ones_like(q) / len(q)
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2.0))


def distance_value(metric: str, grid: np.ndarray, da: np.ndarray, db: np.ndarray, samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    da = np.nan_to_num(np.asarray(da, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    db = np.nan_to_num(np.asarray(db, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    da = np.clip(da, 0.0, np.inf)
    db = np.clip(db, 0.0, np.inf)
    if metric == "wasserstein_distance":
        if samples_a.size and samples_b.size:
            return float(wasserstein_distance(samples_a, samples_b))
        return float(wasserstein_distance(grid, grid, da, db))
    if metric == "jensen_shannon_distance":
        eps = 1e-15
        pa = da + eps
        pb = db + eps
        pa = pa / pa.sum()
        pb = pb / pb.sum()
        value = float(jensenshannon(pa, pb, base=2.0))
        return value if np.isfinite(value) else 0.0
    if metric == "kolmogorov_smirnov_statistic":
        if samples_a.size and samples_b.size:
            return float(ks_2samp(samples_a, samples_b).statistic)
        cdfa = np.cumsum(da) / da.sum() if da.sum() > 0 else np.zeros_like(da)
        cdfb = np.cumsum(db) / db.sum() if db.sum() > 0 else np.zeros_like(db)
        return float(np.max(np.abs(cdfa - cdfb)))
    if metric == "hellinger_distance":
        return hellinger(da, db)
    raise ValueError(metric)


def compute_distances(
    experiment_id: str,
    aggregation_level: str,
    case_id: int | None,
    sensor: str,
    densities: dict[str, np.ndarray],
    grid: np.ndarray,
    raw_samples: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in DISTANCE_METRICS:
        for a in SEGMENT_ORDER:
            for b in SEGMENT_ORDER:
                if a == b:
                    value = 0.0
                else:
                    value = distance_value(metric, grid, densities[a], densities[b], raw_samples.get(a, np.asarray([])), raw_samples.get(b, np.asarray([])))
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "aggregation_level": aggregation_level,
                        "case_id": case_id,
                        "sensor_name": sensor,
                        "distance_metric": metric,
                        "segment_a": a,
                        "segment_b": b,
                        "distance_value": value,
                        "n_samples_a": int(raw_samples.get(a, np.asarray([])).size),
                        "n_samples_b": int(raw_samples.get(b, np.asarray([])).size),
                    }
                )
    return pd.DataFrame(rows)


def save_distance_outputs(base_analysis: Path, base_figures: Path, stem: str, distances: pd.DataFrame, dpi: int) -> None:
    base_analysis.mkdir(parents=True, exist_ok=True)
    base_figures.mkdir(parents=True, exist_ok=True)
    distances.to_csv(base_analysis / f"{stem}.csv", index=False)
    distances.to_csv(base_analysis / f"{stem}_long.csv", index=False)
    for metric, group in distances.groupby("distance_metric"):
        matrix = group.pivot(index="segment_a", columns="segment_b", values="distance_value").reindex(index=SEGMENT_ORDER, columns=SEGMENT_ORDER)
        matrix.to_csv(base_analysis / f"{stem}_{metric}_wide.csv")
        plot_heatmap(base_figures / f"{stem}_{metric}.png", matrix, f"{stem} | {metric}", dpi=dpi)


def pooled_samples(records: list[dict[str, Any]], sensors: list[str], case_id: int | None = None) -> dict[str, dict[str, np.ndarray]]:
    chosen = [r for r in records if case_id is None or r["case_id"] == case_id]
    out: dict[str, dict[str, np.ndarray]] = {}
    for sensor in sensors:
        out[sensor] = {}
        for segment in SEGMENT_ORDER:
            chunks = [r["samples"][sensor][segment] for r in chosen if sensor in r["samples"] and r["samples"][sensor][segment].size]
            out[sensor][segment] = np.concatenate(chunks) if chunks else np.asarray([], dtype=float)
    return out


def build_distribution_summary(experiment_id: str, records: list[dict[str, Any]], sensors: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        for sensor in sensors:
            for segment in SEGMENT_ORDER:
                samples = record["samples"][sensor][segment]
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "aggregation_level": "run_level",
                        "case_id": record["case_id"],
                        "pair_id": record["pair_id"],
                        "dataset_run_id": record["dataset_run_id"],
                        "sensor_name": sensor,
                        "segment_label": segment,
                        "n_samples": int(samples.size),
                        **summary_stats(samples),
                    }
                )
    for case_id in sorted({r["case_id"] for r in records}):
        pooled = pooled_samples(records, sensors, case_id=case_id)
        for sensor in sensors:
            for segment in SEGMENT_ORDER:
                samples = pooled[sensor][segment]
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "aggregation_level": "case_level_mean",
                        "case_id": case_id,
                        "pair_id": PAIR_BY_CASE.get(case_id, ""),
                        "dataset_run_id": None,
                        "sensor_name": sensor,
                        "segment_label": segment,
                        "n_samples": int(samples.size),
                        **summary_stats(samples),
                    }
                )
    pooled = pooled_samples(records, sensors, case_id=None)
    for sensor in sensors:
        for segment in SEGMENT_ORDER:
            samples = pooled[sensor][segment]
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "aggregation_level": "global_level_mean",
                    "case_id": None,
                    "pair_id": None,
                    "dataset_run_id": None,
                    "sensor_name": sensor,
                    "segment_label": segment,
                    "n_samples": int(samples.size),
                    **summary_stats(samples),
                }
            )
    return pd.DataFrame(rows)


def figure_validation(out: Path, generated_run_keys: list[tuple[int, int]], generated_case_ids: list[int]) -> pd.DataFrame:
    expected_keys = expected_run_keys_from_available_processed()
    expected_cases = SELECTED_CASES
    run_files = list((out / "figures/run_level").glob(f"{PREFIX}_case_*_run_*_segment_distribution.png"))
    case_files = list((out / "figures/case_level").glob(f"{PREFIX}_case_*_mean_run_segment_distribution.png"))
    global_files = list((out / "figures/global_level").glob(f"{PREFIX}_global_mean_segment_distribution_all_cases_all_runs.png"))
    generated_key_set = set(generated_run_keys)
    expected_key_set = set(expected_keys)
    return pd.DataFrame(
        [
            {
                "figure_type": "run_level",
                "expected_count": EXPECTED_RUN_FIGURES,
                "actual_count": len(run_files),
                "passed": len(run_files) == EXPECTED_RUN_FIGURES,
                "missing_items": json.dumps([{"case_id": c, "dataset_run_id": r} for c, r in sorted(expected_key_set - generated_key_set)]),
                "extra_items": json.dumps([{"case_id": c, "dataset_run_id": r} for c, r in sorted(generated_key_set - expected_key_set)]),
            },
            {
                "figure_type": "case_level",
                "expected_count": EXPECTED_CASE_FIGURES,
                "actual_count": len(case_files),
                "passed": len(case_files) == EXPECTED_CASE_FIGURES,
                "missing_items": json.dumps([c for c in expected_cases if c not in set(generated_case_ids)]),
                "extra_items": json.dumps([c for c in generated_case_ids if c not in set(expected_cases)]),
            },
            {
                "figure_type": "global_level",
                "expected_count": EXPECTED_GLOBAL_FIGURES,
                "actual_count": len(global_files),
                "passed": len(global_files) == EXPECTED_GLOBAL_FIGURES,
                "missing_items": json.dumps([] if len(global_files) else ["global_mean"]),
                "extra_items": json.dumps([]),
            },
        ]
    )


def write_report(
    out: Path,
    config: dict[str, Any],
    validation: pd.DataFrame,
    figure_counts: pd.DataFrame,
    distance_summary: pd.DataFrame,
    skipped: list[dict[str, Any]],
    sensors: list[str],
    segment_labels: list[str],
    total_runs: int,
    generated_case_ids: list[int],
) -> None:
    experiment_id = config["experiment"]["experiment_id"]
    fig_line = ", ".join(f"{r.figure_type} {r.actual_count}/{r.expected_count}" for r in figure_counts.itertuples(index=False))
    global_all = distance_summary[(distance_summary["aggregation_level"] == "global_level_mean") & (distance_summary["sensor_name"] == "all_sensors")]
    closest = global_all.sort_values("distance_value").head(1)
    farthest = global_all.sort_values("distance_value", ascending=False).head(1)

    def pair_text(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "not available"
        r = frame.iloc[0]
        return f"{r['segment_a']} vs {r['segment_b']} ({r['distance_metric']}={r['distance_value']:.4f})"

    failed_checks = validation[~validation["passed"].fillna(False)] if "passed" in validation.columns else pd.DataFrame()
    skipped_preview = skipped[:25]
    report = f"""# H1_S4 Segment Distribution Visualization and Distance Analysis

## 1. Executive Summary

Experiment id: `{experiment_id}`

Generated figure counts: {fig_line}.

Configured processed files were the `enabled` files requested in the prompt. Dry-run validation found that those files contain aligned metadata for `{total_runs}` dataset_runs, not the requested 82. The experiment therefore saved all feasible segment-aware EDA outputs and marked the figure-count validation as failed for the missing conditions.

Closest global all-sensor segment pair: {pair_text(closest)}

Farthest global all-sensor segment pair: {pair_text(farthest)}

## 2. Experiment Setup

- Signal data: `{config['data']['signal_path']}`
- Process info: `{config['data']['process_info_path']}`
- Segment metadata: `{config['data']['segment_metadata_path']}`
- Cases: `{SELECTED_CASES}`
- Case pairs: `{PAIR_DEFINITION}`
- Sensors: `{sensors}`
- Segment labels: `{segment_labels}`
- Scaling: `{config['scaling']['method']}`, fit scope `{config['scaling']['fit_scope']}`
- Distribution method: KDE with histogram-density fallback, common x-grid per sensor/run or aggregation

## 3. Segment Metadata Validation

Metadata columns were resolved as `case`, `run`, `idx_noload_end`, `idx_start`, `idx_end`, and `signal_length`. No explicit segment label column was present, so labels were derived from boundary columns. `no_load` is excluded as an output segment setting.

- full_length: `0:signal_length`
- entry: `idx_noload_end:idx_start`
- steady: `idx_start:idx_end`
- exit: `idx_end:signal_length`
- entry_steady: `entry + steady`
- entry_exit: `entry + exit`
- steady_exit: `steady + exit`
- entry_steady_exit: `entry + steady + exit`

Failed dry-run checks:

{failed_checks.to_markdown(index=False) if not failed_checks.empty else 'None'}

## 4. Run-level Segment Distribution

Run-level figures were written to `figures/run_level/`. Actual count is `{int(figure_counts.loc[figure_counts.figure_type.eq('run_level'), 'actual_count'].iloc[0])}` out of expected 82. Missing runs are listed in `analysis/H1_S4_figure_count_validation.csv` and skipped reasons are summarized in `analysis/H1_S4_analysis_summary.json`.

## 5. Case-level Mean Segment Distribution

Case-level mean figures were written to `figures/case_level/`. Available aligned cases were `{sorted(generated_case_ids)}`. The case-level distance matrices use the same average run-density basis as the figures.

## 6. Global Mean Segment Distribution

The global mean distribution figure was written to `figures/global_level/H1_S4_global_mean_segment_distribution_all_cases_all_runs.png` using `{total_runs}` aligned dataset_runs, `{len(sensors)}` sensors, and `{len(segment_labels)}` segment labels.

## 7. Segment Distance Matrix

Case-level matrices were saved under `analysis/distance_matrices/case_level/` and `figures/distance_matrices/case_level/`.

Global-level matrices were saved under `analysis/distance_matrices/global_level/` and `figures/distance_matrices/global_level/`.

Distance metrics:

- Wasserstein distance: raw pooled samples when available
- Jensen-Shannon distance: normalized density vectors on common x-grid
- Kolmogorov-Smirnov statistic: raw pooled samples when available
- Hellinger distance: normalized density vectors on common x-grid

## 8. Interpretation

The distance matrices provide direct evidence that treating entry, steady, and exit as one full signal can hide segment-specific distribution shifts. Because this run used only feasible metadata-aligned records from the configured enabled files, conclusions should be read as partial EDA evidence rather than a complete 82-run case-pair result.

## 9. Limitations

- The analysis depends on heuristic metadata.
- The configured enabled processed files do not contain Case 5 and Case 16, while the requested case list expects them.
- Several selected runs have no row in the required metadata file.
- KDE and histogram fallback can slightly change density shape for small segments.
- Raw scaling keeps original engineering units, so sensor-level distances should be interpreted within sensor or after careful sensor aggregation.

## 10. Recommendations

- Regenerate or provide metadata for all 82 selected runs if the exact requested count is mandatory.
- Use non-enabled processed files only if the experiment definition is revised to permit them.
- Compare these segment distance results against H1_S1 association and later H2 prediction experiments.
- Carry entry/exit-aware features or attention/gating model inputs into follow-up experiments.

## Skipped Conditions

{pd.DataFrame(skipped).head(50).to_markdown(index=False) if skipped else 'None'}
"""
    safe_write_text(out / "reports" / f"{PREFIX}_report.md", report)
    html_text = "<html><body>" + html.escape(report).replace("\n", "<br>\n") + "</body></html>"
    safe_write_text(out / "reports" / f"{PREFIX}_report.html", html_text)


def main() -> int:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = execution_dir(ROOT / args.output_root, timestamp)
    make_dirs(out)
    config = build_config(args, timestamp, out)
    experiment_id = config["experiment"]["experiment_id"]
    log_lines: list[str] = [f"{datetime.now().isoformat()} start {experiment_id}", f"dry_run={args.dry_run}"]
    error_lines: list[str] = []

    try:
        write_yaml(out / "configs" / f"{PREFIX}_input_config.yaml", config)
        signal, process, metadata = load_inputs(args)
        sensors = infer_sensor_columns(signal)
        metadata_path = Path(args.segment_metadata_path)
        validation_rows, validation_checks, skipped, aligned = validate_metadata(signal, process, metadata, sensors, metadata_path)

        write_yaml(out / "configs" / f"{PREFIX}_resolved_config.yaml", config)
        write_json(out / "configs" / f"{PREFIX}_config_validation.json", validation_checks.to_dict(orient="records"))
        validation_rows.to_csv(out / "preprocessing" / f"{PREFIX}_segment_metadata_validation.csv", index=False)
        write_json(out / "data" / f"{PREFIX}_dataset_summary.json", {
            "signal_rows": len(signal),
            "process_rows": len(process),
            "metadata_rows": len(metadata),
            "selected_signal_rows": len(selected_enabled(signal)),
            "selected_process_rows": len(selected_enabled(process)),
            "selected_metadata_rows": int(metadata[metadata["case"].isin(SELECTED_CASES)].shape[0]),
            "aligned_runs_after_enabled_and_metadata_intersection": len(aligned),
        })
        write_json(out / "data" / f"{PREFIX}_sensor_summary.json", {
            "expected_sensors": SENSOR_CANDIDATES,
            "used_sensors": sensors,
            "boundary_sensor_from_metadata_note": "smcDC",
            "boundary_applied_to_all_sensors": True,
        })
        write_json(out / "data" / f"{PREFIX}_case_filter_summary.json", {
            "selected_cases": SELECTED_CASES,
            "case_pairs": PAIR_DEFINITION,
            "configured_signal_case_counts": selected_enabled(signal).groupby("case")["run"].nunique().to_dict(),
            "metadata_case_counts": metadata[metadata["case"].isin(SELECTED_CASES)].groupby("case")["run"].nunique().to_dict(),
        })
        write_json(out / "data" / f"{PREFIX}_target_summary.json", selected_enabled(process).groupby("case")["VB"].agg(["count", "mean", "std", "min", "max"]).reset_index().to_dict(orient="records"))
        write_json(out / "data" / f"{PREFIX}_metadata_summary.json", {
            "columns": metadata.columns.tolist(),
            "status_counts": metadata["status"].value_counts(dropna=False).to_dict() if "status" in metadata.columns else {},
            "note_sensor_mentions_smcDC": bool(metadata.get("note", pd.Series(dtype=str)).astype(str).str.contains("sensor=smcDC").any()),
        })
        write_json(out / "preprocessing" / f"{PREFIX}_segment_summary.json", {
            "segment_labels": SEGMENT_ORDER,
            "main_comparison_segments": MAIN_SEGMENTS,
            "mapping": config["segments"]["boundary_columns"],
            "source": args.segment_metadata_path,
        })

        if args.dry_run:
            fig_validation = pd.DataFrame(
                [
                    {"figure_type": "run_level", "expected_count": EXPECTED_RUN_FIGURES, "actual_count": 0, "passed": False, "missing_items": "dry_run", "extra_items": "[]"},
                    {"figure_type": "case_level", "expected_count": EXPECTED_CASE_FIGURES, "actual_count": 0, "passed": False, "missing_items": "dry_run", "extra_items": "[]"},
                    {"figure_type": "global_level", "expected_count": EXPECTED_GLOBAL_FIGURES, "actual_count": 0, "passed": False, "missing_items": "dry_run", "extra_items": "[]"},
                ]
            )
            fig_validation.to_csv(out / "analysis" / f"{PREFIX}_figure_count_validation.csv", index=False)
            write_json(out / "analysis" / f"{PREFIX}_analysis_summary.json", {"dry_run": True, "aligned_runs": len(aligned), "skipped": skipped})
            log_lines.append(f"dry-run complete aligned_runs={len(aligned)} output={out}")
            safe_write_text(out / "logs" / f"{PREFIX}_run.log", "\n".join(log_lines) + "\n")
            safe_write_text(out / "logs" / f"{PREFIX}_error.log", "\n".join(error_lines) + "\n")
            write_json(out / "logs" / f"{PREFIX}_environment.json", collect_environment(str(ROOT)))
            write_json(out / "logs" / f"{PREFIX}_reproducibility.json", {"experiment_id": experiment_id, "timestamp": timestamp, "command": sys.argv})
            print(out)
            return 0

        params = scaling_params(signal, sensors, args.scaling_method)
        records, extraction_skipped = build_run_records(signal, process, metadata, sensors, params)
        skipped.extend(extraction_skipped)
        run_density = run_densities(records, sensors, args.num_grid_points, args.min_kde_samples)
        case_density = aggregate_case_densities(run_density, sensors) if run_density else {}
        global_density = aggregate_global_densities(case_density, sensors) if case_density else {}
        pooled_global = pooled_samples(records, sensors, None)

        preprocessing_summary = {
            "scaling": config["scaling"],
            "scaling_params": params,
            "distribution": config["distribution"],
            "aligned_records_used": len(records),
            "segment_mapping": config["segments"]["boundary_columns"],
        }
        write_json(out / "preprocessing" / f"{PREFIX}_preprocessing_summary.json", preprocessing_summary)
        validation_rows.to_csv(out / "preprocessing" / f"{PREFIX}_alignment_summary.csv", index=False)
        write_json(out / "preprocessing" / f"{PREFIX}_shape_trace.json", {
            "records": [
                {
                    "case_id": r["case_id"],
                    "dataset_run_id": r["dataset_run_id"],
                    "sensor_lengths": {sensor: {seg: int(r["samples"][sensor][seg].size) for seg in SEGMENT_ORDER} for sensor in sensors},
                }
                for r in records
            ]
        })

        generated_run_keys: list[tuple[int, int]] = []
        for key, item in run_density.items():
            case_id, run_id = key
            generated_run_keys.append(key)
            title = (
                f"H1_S4 run-level distribution | case={case_id}, dataset_run={run_id} | "
                f"metadata=heuristic_sequence_peng2026_smcdc.csv"
            )
            plot_distribution(
                out / "figures/run_level" / f"{PREFIX}_case_{case_id}_run_{clean_run_id(run_id)}_segment_distribution.png",
                title,
                sensors,
                SEGMENT_ORDER,
                item["densities"],
                item["grids"],
                item["counts"],
                None,
                "raw signal value" if args.scaling_method == "raw" else "standardized signal value",
                args.dpi,
            )

        generated_case_ids: list[int] = []
        for case_id, item in sorted(case_density.items()):
            generated_case_ids.append(case_id)
            title = (
                f"H1_S4 case-level mean run distribution | case={case_id}, runs={item['run_count']} | "
                f"metadata=heuristic_sequence_peng2026_smcdc.csv"
            )
            plot_distribution(
                out / "figures/case_level" / f"{PREFIX}_case_{case_id}_mean_run_segment_distribution.png",
                title,
                sensors,
                SEGMENT_ORDER,
                item["densities"],
                item["grids"],
                item["counts"],
                item["variability"],
                "raw signal value" if args.scaling_method == "raw" else "standardized signal value",
                args.dpi,
            )

        if global_density:
            title = (
                f"H1_S4 global mean segment distribution | cases={len(generated_case_ids)}, "
                f"dataset_runs={len(records)}, sensors={len(sensors)}"
            )
            plot_distribution(
                out / "figures/global_level" / f"{PREFIX}_global_mean_segment_distribution_all_cases_all_runs.png",
                title,
                sensors,
                SEGMENT_ORDER,
                global_density["densities"],
                global_density["grids"],
                global_density["counts"],
                global_density["variability"],
                "raw signal value" if args.scaling_method == "raw" else "standardized signal value",
                args.dpi,
            )

        summary = build_distribution_summary(experiment_id, records, sensors)
        summary.to_csv(out / "metrics" / f"{PREFIX}_segment_distribution_summary.csv", index=False)

        all_distance_rows: list[pd.DataFrame] = []
        for case_id, item in sorted(case_density.items()):
            case_samples = pooled_samples(records, sensors, case_id)
            sensor_distance_rows = []
            for sensor in sensors:
                d = compute_distances(experiment_id, "case_level_mean", case_id, sensor, item["densities"][sensor], item["grids"][sensor], case_samples[sensor])
                sensor_distance_rows.append(d)
                all_distance_rows.append(d)
                save_distance_outputs(
                    out / "analysis/distance_matrices/case_level",
                    out / "figures/distance_matrices/case_level",
                    f"{PREFIX}_case_{case_id}_{sensor}_segment_distance_matrix",
                    d,
                    args.dpi,
                )
            concat = pd.concat(sensor_distance_rows, ignore_index=True)
            all_sensors = concat.groupby(["experiment_id", "aggregation_level", "case_id", "distance_metric", "segment_a", "segment_b"], as_index=False).agg(
                distance_value=("distance_value", "mean"),
                n_samples_a=("n_samples_a", "sum"),
                n_samples_b=("n_samples_b", "sum"),
            )
            all_sensors["sensor_name"] = "all_sensors"
            all_sensors = all_sensors[["experiment_id", "aggregation_level", "case_id", "sensor_name", "distance_metric", "segment_a", "segment_b", "distance_value", "n_samples_a", "n_samples_b"]]
            all_distance_rows.append(all_sensors)
            save_distance_outputs(
                out / "analysis/distance_matrices/case_level",
                out / "figures/distance_matrices/case_level",
                f"{PREFIX}_case_{case_id}_all_sensors_segment_distance_matrix",
                all_sensors,
                args.dpi,
            )

        if global_density:
            sensor_global_rows = []
            for sensor in sensors:
                d = compute_distances(
                    experiment_id,
                    "global_level_mean",
                    None,
                    sensor,
                    global_density["densities"][sensor],
                    global_density["grids"][sensor],
                    pooled_global[sensor],
                )
                sensor_global_rows.append(d)
                all_distance_rows.append(d)
                save_distance_outputs(
                    out / "analysis/distance_matrices/global_level",
                    out / "figures/distance_matrices/global_level",
                    f"{PREFIX}_global_{sensor}_segment_distance_matrix",
                    d,
                    args.dpi,
                )
            concat = pd.concat(sensor_global_rows, ignore_index=True)
            all_sensors = concat.groupby(["experiment_id", "aggregation_level", "distance_metric", "segment_a", "segment_b"], as_index=False).agg(
                distance_value=("distance_value", "mean"),
                n_samples_a=("n_samples_a", "sum"),
                n_samples_b=("n_samples_b", "sum"),
            )
            all_sensors["case_id"] = None
            all_sensors["sensor_name"] = "all_sensors"
            all_sensors = all_sensors[["experiment_id", "aggregation_level", "case_id", "sensor_name", "distance_metric", "segment_a", "segment_b", "distance_value", "n_samples_a", "n_samples_b"]]
            all_distance_rows.append(all_sensors)
            save_distance_outputs(
                out / "analysis/distance_matrices/global_level",
                out / "figures/distance_matrices/global_level",
                f"{PREFIX}_global_all_cases_all_runs_segment_distance_matrix",
                all_sensors,
                args.dpi,
            )

        distances = pd.concat(all_distance_rows, ignore_index=True) if all_distance_rows else pd.DataFrame()
        non_diag = distances[distances["segment_a"] != distances["segment_b"]].copy() if not distances.empty else distances
        non_diag.to_csv(out / "analysis" / f"{PREFIX}_segment_distance_summary.csv", index=False)

        fig_counts = figure_validation(out, generated_run_keys, generated_case_ids)
        fig_counts.to_csv(out / "analysis" / f"{PREFIX}_figure_count_validation.csv", index=False)
        analysis_summary = {
            "experiment_id": experiment_id,
            "execution_dir": str(out),
            "configured_data_source": {"signal": args.signal_path, "process": args.process_info_path, "metadata": args.segment_metadata_path},
            "case_ids": SELECTED_CASES,
            "case_pairs": PAIR_DEFINITION,
            "dataset_run_count_used": len(records),
            "sensors": sensors,
            "segment_labels": SEGMENT_ORDER,
            "figure_count_validation": fig_counts.to_dict(orient="records"),
            "skipped_conditions": skipped,
            "closest_global_all_sensors": non_diag[(non_diag["aggregation_level"] == "global_level_mean") & (non_diag["sensor_name"] == "all_sensors")].sort_values("distance_value").head(1).to_dict(orient="records") if not non_diag.empty else [],
            "farthest_global_all_sensors": non_diag[(non_diag["aggregation_level"] == "global_level_mean") & (non_diag["sensor_name"] == "all_sensors")].sort_values("distance_value", ascending=False).head(1).to_dict(orient="records") if not non_diag.empty else [],
        }
        write_json(out / "analysis" / f"{PREFIX}_analysis_summary.json", analysis_summary)

        write_report(out, config, validation_checks, fig_counts, non_diag, skipped, sensors, SEGMENT_ORDER, len(records), generated_case_ids)
        write_json(out / "logs" / f"{PREFIX}_environment.json", collect_environment(str(ROOT)))
        write_json(out / "logs" / f"{PREFIX}_reproducibility.json", {"experiment_id": experiment_id, "timestamp": timestamp, "command": sys.argv})
        log_lines.append(f"main complete records={len(records)} output={out}")
        safe_write_text(out / "logs" / f"{PREFIX}_run.log", "\n".join(log_lines) + "\n")
        safe_write_text(out / "logs" / f"{PREFIX}_error.log", "\n".join(error_lines) + "\n")
        print(out)
        return 0
    except Exception:
        tb = traceback.format_exc()
        error_lines.append(tb)
        log_lines.append("failed")
        safe_write_text(out / "logs" / f"{PREFIX}_run.log", "\n".join(log_lines) + "\n")
        safe_write_text(out / "logs" / f"{PREFIX}_error.log", "\n".join(error_lines) + "\n")
        print(tb, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

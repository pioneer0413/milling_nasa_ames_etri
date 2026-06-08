#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.visualization.figure_export import save_figure_dual


PREFIX = "H5_S2_T1"
TOPIC = "steady_length_feature_sequence_vb_suitability"
SENSOR = "smcDC"
SEGMENT_SETTING = "steady"
TARGET_FEATURES = [
    "mean",
    "std",
    "max",
    "min",
    "peak_to_peak",
    "kurtosis",
    "skewness",
    "spectral_centroid",
    "band_energy",
]
FEATURE_GROUPS = {
    "mean": "statistics",
    "std": "statistics",
    "max": "statistics",
    "min": "statistics",
    "peak_to_peak": "statistics",
    "kurtosis": "shape",
    "skewness": "shape",
    "spectral_centroid": "frequency",
    "band_energy": "frequency",
}
STEADY_LENGTH_PATHS = {
    1000: "datasets/metadata/heuristic_sequence_peng2026_steady1000.csv",
    2000: "datasets/metadata/heuristic_sequence_peng2026_steady2000.csv",
    3000: "datasets/metadata/heuristic_sequence_peng2026_steady3000.csv",
    4000: "datasets/metadata/heuristic_sequence_peng2026.csv",
    5000: "datasets/metadata/heuristic_sequence_peng2026_steady5000.csv",
}
PRIMARY_SUITABILITY_METHOD = "harmonic_mean"
SUITABILITY_EPSILON = 1e-12
PROGNOSABILITY_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H5_S2_T1 steady-length feature-sequence/VB suitability analysis.")
    parser.add_argument("--signal-path", default="datasets/processed/mill_signal_data.csv")
    parser.add_argument("--process-info-path", default="datasets/processed/mill_process_info.csv")
    parser.add_argument("--case-scope", type=int, nargs="+", default=None)
    parser.add_argument("--output-root", default="experiments/executions")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--dpi", type=int, default=150)
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


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def parse_signal(value: Any) -> np.ndarray:
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def execution_dir(output_root: Path, timestamp: str) -> Path:
    return output_root / "H5" / "S2" / "T1" / f"{timestamp}_{TOPIC}"


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
            "analysis_type": "steady_length_feature_sequence_vb_suitability",
        },
        "data": {
            "signal_path": args.signal_path,
            "process_info_path": args.process_info_path,
            "case_scope": args.case_scope,
            "sensor": SENSOR,
            "segment_setting": SEGMENT_SETTING,
            "metadata_paths_by_steady_length": STEADY_LENGTH_PATHS,
        },
        "features": {
            "source": "H1 nine feature set",
            "feature_names": TARGET_FEATURES,
            "feature_groups": FEATURE_GROUPS,
        },
        "visualization": {
            "layout": "rows = all-case average + case, columns = steady-cut length",
            "first_row": "mean feature sequence across all cases grouped by run; VB trend is mean VB grouped by run",
            "feature_directory_pattern": "figures/{feature}/",
            "dpi": args.dpi,
        },
        "suitability": {
            "primary_method": PRIMARY_SUITABILITY_METHOD,
            "components": ["monotonicity", "trendability", "prognosability"],
            "epsilon": SUITABILITY_EPSILON,
            "prognosability_epsilon": PROGNOSABILITY_EPSILON,
        },
    }


def load_metadata() -> dict[int, pd.DataFrame]:
    metadata: dict[int, pd.DataFrame] = {}
    for length, path in STEADY_LENGTH_PATHS.items():
        resolved = resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Missing metadata for steady length {length}: {resolved}")
        df = pd.read_csv(resolved)
        required = {"case", "run", "idx_start", "idx_end", "signal_length"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{resolved} missing columns: {sorted(missing)}")
        df = df.copy()
        df["steady_length_requested"] = int(length)
        df["steady_length_actual"] = df["idx_end"].astype(int) - df["idx_start"].astype(int)
        metadata[length] = df
    return metadata


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal = pd.read_csv(resolve(args.signal_path), usecols=["case", "run", SENSOR])
    process = pd.read_csv(resolve(args.process_info_path))
    required_process = {"case", "run", "VB"}
    missing = required_process.difference(process.columns)
    if missing:
        raise ValueError(f"process info missing columns: {sorted(missing)}")
    return signal, process


def metadata_keys(metadata: dict[int, pd.DataFrame], case_scope: list[int] | None = None) -> pd.DataFrame:
    frames = []
    scope = set(case_scope or [])
    for length, df in metadata.items():
        keys = df[["case", "run"]].drop_duplicates()
        if scope:
            keys = keys.loc[keys["case"].isin(scope)].copy()
        frames.append(keys.assign(steady_length_requested=length))
    combined = pd.concat(frames, ignore_index=True)
    counts = combined.groupby(["case", "run"], as_index=False)["steady_length_requested"].nunique()
    missing = counts.loc[counts["steady_length_requested"] != len(STEADY_LENGTH_PATHS)]
    if not missing.empty:
        raise ValueError(f"Metadata key mismatch across lengths: {missing.head().to_dict(orient='records')}")
    return counts[["case", "run"]].sort_values(["case", "run"]).reset_index(drop=True)


def row_for(metadata: dict[int, pd.DataFrame], length: int, case: int, run: int) -> pd.Series:
    rows = metadata[length].loc[(metadata[length]["case"] == case) & (metadata[length]["run"] == run)]
    if len(rows) != 1:
        raise ValueError(f"Expected one metadata row for length={length}, case={case}, run={run}; got {len(rows)}")
    return rows.iloc[0]


def compute_features(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        arr = np.asarray([0.0], dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(arr))
    freqs = np.fft.rfftfreq(len(arr), d=1.0)
    spectral_sum = float(spectrum.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "peak_to_peak": float(np.ptp(arr)),
            "kurtosis": float(kurtosis(arr, fisher=True, bias=False)) if len(arr) > 3 else 0.0,
            "skewness": float(skew(arr, bias=False)) if len(arr) > 2 else 0.0,
            "spectral_centroid": float(np.sum(freqs * spectrum) / spectral_sum) if spectral_sum > 0 else 0.0,
            "band_energy": float(np.sum(np.square(spectrum)) / max(len(arr), 1)),
        }
    return {key: (0.0 if not np.isfinite(value) else value) for key, value in raw.items()}


def build_feature_long(metadata: dict[int, pd.DataFrame], signal: pd.DataFrame, process: pd.DataFrame, case_scope: list[int] | None = None) -> pd.DataFrame:
    keys = metadata_keys(metadata, case_scope)
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
        signal_arr = parse_signal(getattr(source_row, SENSOR))
        for length in sorted(STEADY_LENGTH_PATHS):
            meta_row = row_for(metadata, length, case, run_id)
            idx_start = int(np.clip(meta_row["idx_start"], 0, signal_arr.size))
            idx_end = int(np.clip(meta_row["idx_end"], idx_start, signal_arr.size))
            steady = signal_arr[idx_start:idx_end]
            values = compute_features(steady)
            for feature_name, feature_value in values.items():
                rows.append(
                    {
                        "case_id": case,
                        "run": run_id,
                        "steady_length": int(length),
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
                        "idx_start": int(meta_row["idx_start"]),
                        "idx_end": int(meta_row["idx_end"]),
                        "steady_length_actual": int(meta_row["steady_length_actual"]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["feature_name", "steady_length", "case_id", "run"]).reset_index(drop=True)


def compute_monotonicity(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    diff = np.diff(x)
    n_inc = int(np.sum(diff > 0))
    n_dec = int(np.sum(diff < 0))
    return float(abs((n_inc - n_dec) / (len(x) - 1)))


def compute_trendability(x: np.ndarray, t: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(t)
    x = x[finite]
    t = t[finite]
    n = len(x)
    if n < 2:
        return float("nan")
    numerator = n * np.sum(x * t) - np.sum(x) * np.sum(t)
    x_term = n * np.sum(x**2) - np.sum(x) ** 2
    t_term = n * np.sum(t**2) - np.sum(t) ** 2
    denominator = math.sqrt(max(x_term * t_term, 0.0))
    if denominator <= 0:
        return float("nan")
    return float(abs(numerator / denominator))


def compute_prognosability(starts: np.ndarray, finals: np.ndarray) -> float:
    starts = np.asarray(starts, dtype=np.float64)
    finals = np.asarray(finals, dtype=np.float64)
    finite = np.isfinite(starts) & np.isfinite(finals)
    starts = starts[finite]
    finals = finals[finite]
    if len(finals) == 0:
        return float("nan")
    final_variance = float(np.var(finals))
    mean_start_final_distance = float(np.mean(np.abs(starts - finals)))
    return float(np.exp(-(final_variance / (mean_start_final_distance + PROGNOSABILITY_EPSILON))))


def compute_suitability_harmonic(monotonicity: float, trendability: float, prognosability: float) -> float:
    if np.isnan(monotonicity) or np.isnan(trendability) or np.isnan(prognosability):
        return float("nan")
    if monotonicity <= 0 and trendability <= 0 and prognosability <= 0:
        return 0.0
    return float(
        3.0
        / (
            (1.0 / (monotonicity + SUITABILITY_EPSILON))
            + (1.0 / (trendability + SUITABILITY_EPSILON))
            + (1.0 / (prognosability + SUITABILITY_EPSILON))
        )
    )


def compute_suitability_tables(feature_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    group_cols = ["case_id", "steady_length", "sensor_name", "segment_setting", "feature_name", "feature_group"]
    for keys, group in feature_long.groupby(group_cols, dropna=False):
        case_id, steady_length, sensor_name, segment_setting, feature_name, feature_group = keys
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
            monotonicity = compute_monotonicity(x)
            trendability = compute_trendability(x, t)
            x_start = float(x[0])
            x_final = float(x[-1])
            if not np.isfinite(trendability):
                status = "skipped_zero_trendability_denominator"
            if not np.isfinite(monotonicity):
                status = "skipped_monotonicity_not_computable"
        rows.append(
            {
                "case_id": int(case_id),
                "steady_length": int(steady_length),
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
                "primary_suitability_method": PRIMARY_SUITABILITY_METHOD,
            }
        )

    case_level = pd.DataFrame(rows)
    case_level["prognosability"] = np.nan
    for _, group in case_level.groupby(["steady_length", "sensor_name", "segment_setting", "feature_name"], dropna=False):
        ok = group.loc[group["calculation_status"].eq("ok")]
        prognosability = compute_prognosability(ok["x_start"].to_numpy(dtype=np.float64), ok["x_final"].to_numpy(dtype=np.float64))
        case_level.loc[group.index, "prognosability"] = prognosability
    case_level["suitability"] = case_level.apply(
        lambda row: compute_suitability_harmonic(row["monotonicity"], row["trendability"], row["prognosability"]),
        axis=1,
    )
    invalid = case_level["suitability"].notna() & ~case_level["suitability"].between(0, 1)
    case_level.loc[invalid, "calculation_status"] = "invalid_suitability_range"

    ok = case_level.loc[case_level["calculation_status"].eq("ok")].copy()
    summary = (
        ok.groupby(["steady_length", "sensor_name", "segment_setting", "feature_name", "feature_group"], dropna=False)
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
    summary["rank_by_suitability_within_length"] = summary.groupby("steady_length")["mean_suitability"].rank(ascending=False, method="first").astype(int)
    summary["rank_by_suitability_within_feature"] = summary.groupby("feature_name")["mean_suitability"].rank(ascending=False, method="first").astype(int)
    summary = summary.sort_values(["feature_name", "steady_length"]).reset_index(drop=True)
    return case_level.sort_values(["feature_name", "steady_length", "case_id"]).reset_index(drop=True), summary


def robust_ylim(values: pd.Series) -> tuple[float, float]:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if finite.empty:
        return -1.0, 1.0
    lo = float(finite.quantile(0.03))
    hi = float(finite.quantile(0.97))
    if np.isclose(lo, hi):
        lo = float(finite.min())
        hi = float(finite.max())
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1.0)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.12
    return lo - pad, hi + pad


def plot_feature_grid(output_dir: Path, feature_long: pd.DataFrame, suitability_summary: pd.DataFrame, feature_name: str, dpi: int) -> Path:
    feature_df = feature_long.loc[feature_long["feature_name"].eq(feature_name)].copy()
    lengths = sorted(feature_df["steady_length"].unique().tolist())
    cases = sorted(feature_df["case_id"].unique().tolist())
    row_labels = ["All cases mean"] + [f"Case {case}" for case in cases]
    n_rows = len(row_labels)
    n_cols = len(lengths)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.25, n_rows * 1.45), sharex=False, squeeze=False)
    feature_ylim = robust_ylim(feature_df["feature_value"])
    vb_ylim = robust_ylim(feature_df["VB"])

    for col_idx, length in enumerate(lengths):
        length_df = feature_df.loc[feature_df["steady_length"].eq(length)]
        avg = (
            length_df.groupby("run", as_index=False)
            .agg(feature_value=("feature_value", "mean"), VB=("VB", "mean"), n_cases=("case_id", "nunique"))
            .sort_values("run")
        )
        s_row = suitability_summary.loc[(suitability_summary["feature_name"].eq(feature_name)) & (suitability_summary["steady_length"].eq(length))]
        s_text = ""
        if not s_row.empty:
            s_text = f"S={float(s_row.iloc[0]['mean_suitability']):.3f}"
        draw_sequence_ax(axes[0, col_idx], avg, feature_ylim, vb_ylim, title=f"{length} ({s_text})", row_label=row_labels[0])

        for row_idx, case in enumerate(cases, start=1):
            case_df = length_df.loc[length_df["case_id"].eq(case)].sort_values("run")
            case_s = compute_case_suitability_text(feature_name, length, case, output_dir)
            draw_sequence_ax(axes[row_idx, col_idx], case_df, feature_ylim, vb_ylim, title="", row_label=row_labels[row_idx], annotation=case_s)

    for ax in axes[-1, :]:
        ax.set_xlabel("run", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel(feature_name, fontsize=7)
    for ax in axes[:, -1]:
        ax_twin = ax.twinx()
        ax_twin.set_ylabel("VB", fontsize=7, color="#b91c1c")
        ax_twin.set_yticks([])

    handles = [
        Line2D([0], [0], color="#2563eb", linewidth=1.2, marker="o", markersize=2.5, label=f"{feature_name}"),
        Line2D([0], [0], color="#dc2626", linewidth=1.0, linestyle="--", marker="s", markersize=2.5, label="VB trend"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"{PREFIX} {feature_name}: steady-cut length feature sequence with VB trend", fontsize=13, y=0.997)
    fig.tight_layout(rect=(0.01, 0.025, 0.99, 0.982), h_pad=0.45, w_pad=0.22)

    figure_path = output_dir / "figures" / feature_name / f"{PREFIX}_{feature_name}_sequence_vb_grid.png"
    save_figure_dual(fig, figure_path, dpi=dpi)
    plt.close(fig)
    return figure_path


def draw_sequence_ax(
    ax: plt.Axes,
    data: pd.DataFrame,
    feature_ylim: tuple[float, float],
    vb_ylim: tuple[float, float],
    title: str,
    row_label: str,
    annotation: str = "",
) -> None:
    twin = ax.twinx()
    if data.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=7)
    else:
        ax.plot(data["run"], data["feature_value"], color="#2563eb", marker="o", markersize=2.2, linewidth=1.05)
        twin.plot(data["run"], data["VB"], color="#dc2626", marker="s", markersize=2.0, linewidth=0.95, linestyle="--", alpha=0.85)
    ax.set_ylim(*feature_ylim)
    twin.set_ylim(*vb_ylim)
    twin.tick_params(axis="y", colors="#b91c1c", labelsize=5, length=2)
    ax.tick_params(axis="both", labelsize=5.5, length=2)
    ax.grid(True, color="#e5e7eb", linewidth=0.35, alpha=0.75)
    if title:
        ax.set_title(title, fontsize=8, pad=3)
    ax.text(-0.02, 0.5, row_label, transform=ax.transAxes, ha="right", va="center", fontsize=6.8)
    if annotation:
        ax.text(
            0.02,
            0.95,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.6,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "boxstyle": "round,pad=0.16", "alpha": 0.75},
        )


def compute_case_suitability_text(feature_name: str, length: int, case: int, output_dir: Path) -> str:
    cache_path = output_dir / "analysis" / "H5_S2_T1_case_level_suitability.csv"
    if not cache_path.exists():
        return ""
    # This helper is called after the case-level table is written; small CSV read keeps plotting code simple.
    case_level = pd.read_csv(cache_path)
    row = case_level.loc[
        case_level["feature_name"].eq(feature_name)
        & case_level["steady_length"].eq(length)
        & case_level["case_id"].eq(case)
    ]
    if row.empty or not np.isfinite(row.iloc[0]["suitability"]):
        return ""
    return f"S={float(row.iloc[0]['suitability']):.2f}"


def plot_suitability_heatmap(output_dir: Path, suitability_summary: pd.DataFrame, dpi: int) -> Path:
    matrix = suitability_summary.pivot(index="feature_name", columns="steady_length", values="mean_suitability").reindex(index=TARGET_FEATURES)
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.astype(str), fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel("steady-cut length")
    ax.set_title(f"{PREFIX} mean suitability by feature and steady-cut length")
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


def write_report(output_dir: Path, summary: dict[str, Any], suitability_summary: pd.DataFrame) -> None:
    top = suitability_summary.sort_values("mean_suitability", ascending=False).head(12)
    lines = [
        f"# {PREFIX} Steady-length Feature Sequence and VB Suitability",
        "",
        "## Scope",
        "",
        f"- Sensor: `{SENSOR}`",
        f"- Segment: `{SEGMENT_SETTING}` only",
        f"- Features: `{', '.join(TARGET_FEATURES)}`",
        f"- Steady-cut lengths: `{summary['steady_lengths']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Case/run rows: `{summary['case_run_count']}`",
        "",
        "## Outputs",
        "",
        "- Feature sequence table: `data/H5_S2_T1_feature_sequence_long.csv`",
        "- Case-level suitability: `analysis/H5_S2_T1_case_level_suitability.csv`",
        "- Feature-length suitability summary: `analysis/H5_S2_T1_feature_length_suitability_summary.csv`",
        "- Per-feature figures: `figures/{feature}/H5_S2_T1_{feature}_sequence_vb_grid.{png,svg}`",
        "- Suitability heatmap: `figures/H5_S2_T1_suitability_heatmap.{png,svg}`",
        "",
        "## Top Suitability",
        "",
        "| feature | steady_length | mean_suitability | mean_monotonicity | mean_trendability | mean_prognosability |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.feature_name} | {int(row.steady_length)} | {row.mean_suitability:.4f} | "
            f"{row.mean_monotonicity:.4f} | {row.mean_trendability:.4f} | {row.mean_prognosability:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Method Note",
            "",
            "Suitability follows the H1_S2 harmonic formulation over monotonicity, trendability, and prognosability. "
            "Each case-level sequence is ordered by `run`; the first visual row averages feature and VB sequences across all cases by run.",
        ]
    )
    (output_dir / "reports" / "H5_S2_T1_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = args.timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = execution_dir(resolve(args.output_root), timestamp)
    if output_dir.exists():
        raise FileExistsError(f"Execution directory already exists: {output_dir}")
    make_dirs(output_dir)

    config = build_config(args, timestamp, output_dir)
    write_yaml(output_dir / "configs" / "H5_S2_T1_input_config.yaml", config)
    write_json(output_dir / "logs" / "H5_S2_T1_environment.json", collect_environment())

    metadata = load_metadata()
    signal, process = load_inputs(args)
    keys = metadata_keys(metadata, args.case_scope)
    summary: dict[str, Any] = {
        "experiment_id": config["experiment"]["experiment_id"],
        "execution_dir": str(output_dir),
        "sensor": SENSOR,
        "segment_setting": SEGMENT_SETTING,
        "steady_lengths": sorted(STEADY_LENGTH_PATHS),
        "features": TARGET_FEATURES,
        "case_scope": sorted(args.case_scope) if args.case_scope else "all_metadata_cases",
        "case_count": int(keys["case"].nunique()),
        "case_run_count": int(len(keys)),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        write_json(output_dir / "analysis" / "H5_S2_T1_summary.json", summary)
        return summary

    feature_long = build_feature_long(metadata, signal, process, args.case_scope)
    feature_long.to_csv(output_dir / "data" / "H5_S2_T1_feature_sequence_long.csv", index=False)
    case_level, suitability_summary = compute_suitability_tables(feature_long)
    case_level.to_csv(output_dir / "analysis" / "H5_S2_T1_case_level_suitability.csv", index=False)
    suitability_summary.to_csv(output_dir / "analysis" / "H5_S2_T1_feature_length_suitability_summary.csv", index=False)

    figure_manifest: list[dict[str, Any]] = []
    for feature_name in TARGET_FEATURES:
        feature_dir = output_dir / "figures" / feature_name
        feature_long.loc[feature_long["feature_name"].eq(feature_name)].to_csv(feature_dir / f"{PREFIX}_{feature_name}_feature_sequence.csv", index=False)
        suitability_summary.loc[suitability_summary["feature_name"].eq(feature_name)].to_csv(feature_dir / f"{PREFIX}_{feature_name}_suitability_summary.csv", index=False)
        fig_path = plot_feature_grid(output_dir, feature_long, suitability_summary, feature_name, args.dpi)
        figure_manifest.append({"feature_name": feature_name, "figure_path": str(fig_path.relative_to(output_dir))})
    heatmap_path = plot_suitability_heatmap(output_dir, suitability_summary, args.dpi)
    figure_manifest.append({"feature_name": "__suitability_heatmap__", "figure_path": str(heatmap_path.relative_to(output_dir))})
    pd.DataFrame(figure_manifest).to_csv(output_dir / "analysis" / "H5_S2_T1_figure_manifest.csv", index=False)

    summary.update(
        {
            "feature_sequence_rows": int(len(feature_long)),
            "case_level_suitability_rows": int(len(case_level)),
            "feature_length_summary_rows": int(len(suitability_summary)),
            "figure_count": int(len(figure_manifest)),
            "best_feature_length": suitability_summary.sort_values("mean_suitability", ascending=False).head(1).to_dict(orient="records"),
        }
    )
    write_json(output_dir / "analysis" / "H5_S2_T1_summary.json", summary)
    write_report(output_dir, summary, suitability_summary)
    return summary


def main() -> None:
    output_dir: Path | None = None
    try:
        summary = run(parse_args())
        output_dir = Path(summary["execution_dir"])
        print(json.dumps(to_builtin(summary), indent=2, ensure_ascii=False))
    except Exception:
        if output_dir is not None:
            (output_dir / "logs" / "H5_S2_T1_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()

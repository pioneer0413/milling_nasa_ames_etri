#!/usr/bin/env python3
"""H16_S1: compare cutting-segment boundaries from alternative source sensors.

The legacy v2 table uses a reverse-kurtosis smcDC exit boundary to override
the Peng2026 steady5 ``idx_end``. This script applies the same idea as a
sensor-source axis and prioritizes visual diagnostics:

  1. detect an exit-start candidate for each requested sensor
  2. merge the detected exit boundary into the Peng2026 steady5 schema
  3. write sensor-specific segment CSVs for downstream H13/H14/H15 scripts
  4. generate boundary heatmaps, shift plots, and representative overlays

Outputs:
  experiments/analysis/segment_sensor_source/{timestamp}_sensor_source_visualization/
  datasets/nasa/cutting_segment_v2/sensor_source/
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_H4_S5_feature_gru_hpo import CASE_SCOPE, SENSORS, THRESH, parse_signal  # noqa: E402

SIGNAL_CSV = ROOT / "datasets/processed/mill_signal_data.csv"
PROCESS_CSV = ROOT / "datasets/processed/mill_process_info.csv"
BASE_SEG_CSV = ROOT / "datasets/nasa/cutting_segment/seg_peng2026_steady5.csv"
LEGACY_V2_CSV = ROOT / "datasets/nasa/cutting_segment_v2/seg_peng2026_steady5_exitfix_reverse_kurtosis.csv"
LEGACY_EXIT_CSV = ROOT / "datasets/nasa/cutting_segment_v2/seg_reverse_kurtosis_smcDC.csv"
DATASET_OUT = ROOT / "datasets/nasa/cutting_segment_v2/sensor_source"
ANALYSIS_ROOT = ROOT / "experiments/analysis/segment_sensor_source"


@dataclass(frozen=True)
class DetectorConfig:
    window: int = 256
    hop: int = 16
    min_fraction: float = 0.65
    max_fraction: float = 0.985
    threshold_quantile: float = 0.97
    threshold_mad_multiplier: float = 5.0
    min_tail: int = 384
    rms_weight: float = 0.40


def robust_z(values: np.ndarray) -> np.ndarray:
    med = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - med)))
    scale = 1.4826 * mad if mad > 1e-12 else float(np.nanstd(values))
    if scale <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - med) / scale


def rolling_features(x: np.ndarray, cfg: DetectorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(0, max(1, len(x) - cfg.window + 1), cfg.hop, dtype=int)
    if len(starts) == 0:
        starts = np.array([0], dtype=int)
    rms = []
    kur = []
    for start in starts:
        w = x[start : start + cfg.window]
        if len(w) < cfg.window:
            w = np.pad(w, (0, cfg.window - len(w)), mode="edge")
        rms.append(float(np.sqrt(np.mean(np.square(w)))))
        k = float(kurtosis(w, fisher=False, bias=False)) if len(w) > 3 else 3.0
        kur.append(k if np.isfinite(k) else 3.0)
    return starts, np.asarray(rms), np.asarray(kur)


def detect_exit_start(x: np.ndarray, cfg: DetectorConfig) -> dict[str, float | int | bool | str]:
    n = int(len(x))
    if n < cfg.window * 4:
        return {
            "idx_exit_start": n,
            "has_exit": False,
            "exit_confidence": 0.0,
            "score": 0.0,
            "threshold": 0.0,
            "segment_status": "too_short",
        }

    starts, rms, kur = rolling_features(x, cfg)
    zrms = robust_z(rms)
    zkur = robust_z(kur)
    score = np.zeros_like(zkur, dtype=float)
    score[1:] = np.abs(np.diff(zkur)) + cfg.rms_weight * np.abs(np.diff(zrms))
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    min_idx = int(round(n * cfg.min_fraction))
    max_idx = min(int(round(n * cfg.max_fraction)), n - cfg.min_tail)
    mask = (starts >= min_idx) & (starts <= max_idx)
    if not bool(mask.any()):
        return {
            "idx_exit_start": n,
            "has_exit": False,
            "exit_confidence": 0.0,
            "score": float(np.nanmax(score)) if len(score) else 0.0,
            "threshold": 0.0,
            "segment_status": "no_search_region",
        }

    candidate_scores = score[mask]
    local_indices = np.flatnonzero(mask)
    best_local = int(np.argmax(candidate_scores))
    best_i = int(local_indices[best_local])
    idx = int(starts[best_i])

    global_threshold = float(np.nanquantile(score, cfg.threshold_quantile))
    local_median = float(np.nanmedian(candidate_scores))
    local_mad = float(np.nanmedian(np.abs(candidate_scores - local_median)))
    robust_threshold = local_median + cfg.threshold_mad_multiplier * 1.4826 * max(local_mad, 1e-12)
    threshold = max(global_threshold, robust_threshold)
    best_score = float(score[best_i])
    confidence = best_score / (threshold + 1e-12)
    has_exit = bool(best_score >= threshold and idx < n - cfg.min_tail)

    return {
        "idx_exit_start": idx if has_exit else n,
        "has_exit": has_exit,
        "exit_confidence": float(confidence if has_exit else min(confidence, 0.999)),
        "score": best_score,
        "threshold": float(threshold),
        "segment_status": "ok" if has_exit else "no_exit",
    }


def generate_sensor_table(signal_df: pd.DataFrame, process_df: pd.DataFrame, sensor: str, cfg: DetectorConfig) -> pd.DataFrame:
    proc_cols = ["case", "run", "time", "DOC", "feed", "material", "material_name", "VB"]
    proc = process_df[proc_cols].copy()
    rows = []
    for row in signal_df.itertuples(index=False):
        case_id = int(row.case)
        run_id = int(row.run)
        x = parse_signal(getattr(row, sensor))
        n = int(len(x))
        if np.abs(x).max(initial=0.0) > THRESH:
            det = {
                "idx_exit_start": n,
                "has_exit": False,
                "exit_confidence": 0.0,
                "score": 0.0,
                "threshold": 0.0,
                "segment_status": "threshold_exceeded",
            }
        else:
            det = detect_exit_start(x, cfg)
        rows.append(
            {
                "case": case_id,
                "run": run_id,
                "sensor": sensor,
                "idx_exit_start": int(det["idx_exit_start"]),
                "idx_end": int(det["idx_exit_start"]),
                "signal_length": n,
                "has_exit": bool(det["has_exit"]),
                "exit_confidence": float(det["exit_confidence"]),
                "score": float(det["score"]),
                "threshold": float(det["threshold"]),
                "segment_status": str(det["segment_status"]),
                "method": "reverse_kurtosis_sensor_source",
                "detection_window": cfg.window,
                "detection_hop": cfg.hop,
            }
        )
    out = pd.DataFrame(rows)
    out = out.merge(proc, on=["case", "run"], how="left", validate="one_to_one")
    return out.sort_values(["case", "run"]).reset_index(drop=True)


def merge_with_base(base: pd.DataFrame, sensor_exit: pd.DataFrame, sensor: str) -> pd.DataFrame:
    cols = ["case", "run", "idx_exit_start", "has_exit", "exit_confidence"]
    merged = base.merge(sensor_exit[cols], on=["case", "run"], how="left", validate="one_to_one")
    merged["idx_end_peng"] = merged["idx_end"].astype(int)
    merged["idx_exit_start"] = merged["idx_exit_start"].fillna(merged["signal_length"]).astype(int)
    merged["has_exit"] = merged["has_exit"].fillna(False).astype(bool)
    merged["exit_confidence"] = merged["exit_confidence"].fillna(0.0).astype(float)
    merged["idx_end"] = np.where(merged["has_exit"], merged["idx_exit_start"], merged["signal_length"]).astype(int)
    merged["idx_end"] = np.maximum(merged["idx_start"].astype(int) + 1, merged["idx_end"].astype(int))
    merged["idx_end"] = np.minimum(merged["idx_end"].astype(int), merged["signal_length"].astype(int))
    merged["exit_source"] = f"reverse_kurtosis_{sensor}"
    return merged


def summarize_tables(sensor_tables: dict[str, pd.DataFrame], legacy_exit: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    long_rows = []
    legacy_idx: dict[tuple[int, int], int] = {}
    legacy_has: dict[tuple[int, int], bool] = {}
    if legacy_exit is not None:
        legacy_idx = {(int(r.case), int(r.run)): int(r.idx_exit_start) for r in legacy_exit.itertuples(index=False)}
        legacy_has = {(int(r.case), int(r.run)): bool(r.has_exit) for r in legacy_exit.itertuples(index=False)}

    for sensor, df in sensor_tables.items():
        tmp = df.copy()
        tmp["exit_frac"] = tmp["idx_exit_start"] / tmp["signal_length"]
        if legacy_idx:
            tmp["smcDC_idx_exit_start"] = [legacy_idx.get((int(c), int(r)), np.nan) for c, r in zip(tmp.case, tmp.run)]
            tmp["smcDC_has_exit"] = [legacy_has.get((int(c), int(r)), False) for c, r in zip(tmp.case, tmp.run)]
            tmp["shift_vs_smcDC"] = tmp["idx_exit_start"] - tmp["smcDC_idx_exit_start"]
        else:
            tmp["shift_vs_smcDC"] = np.nan
        summaries.append(
            {
                "sensor": sensor,
                "n_runs": int(len(tmp)),
                "has_exit_runs": int(tmp["has_exit"].sum()),
                "has_exit_rate": float(tmp["has_exit"].mean()),
                "mean_exit_frac": float(tmp["exit_frac"].mean()),
                "median_exit_frac": float(tmp["exit_frac"].median()),
                "mean_shift_vs_smcDC": float(tmp["shift_vs_smcDC"].mean(skipna=True)),
                "median_shift_vs_smcDC": float(tmp["shift_vs_smcDC"].median(skipna=True)),
                "mean_confidence": float(tmp["exit_confidence"].mean()),
            }
        )
        long_rows.extend(tmp[["case", "run", "sensor", "idx_exit_start", "signal_length", "exit_frac", "has_exit", "exit_confidence", "shift_vs_smcDC"]].to_dict("records"))
    return pd.DataFrame(summaries), pd.DataFrame(long_rows)


def normalized_trace(x: np.ndarray) -> np.ndarray:
    z = robust_z(np.asarray(x, dtype=float))
    return np.clip(z, -6, 6)


def plot_heatmap(long_df: pd.DataFrame, fig_dir: Path) -> None:
    pivot = long_df.pivot_table(index=["case", "run"], columns="sensor", values="exit_frac", aggfunc="first")
    pivot = pivot.sort_index()
    fig_h = max(7.0, len(pivot) * 0.045)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0.60, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    tick_step = max(1, len(pivot) // 30)
    y_idx = np.arange(0, len(pivot), tick_step)
    ax.set_yticks(y_idx)
    ax.set_yticklabels([f"{c}-{r}" for c, r in pivot.index[y_idx]])
    ax.set_title("Exit-start fraction by source sensor")
    ax.set_ylabel("case-run")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("idx_exit_start / signal_length")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_exit_fraction_heatmap.png", dpi=180)
    fig.savefig(fig_dir / "01_exit_fraction_heatmap.svg")
    plt.close(fig)


def plot_shift_box(long_df: pd.DataFrame, fig_dir: Path) -> None:
    data = [g["shift_vs_smcDC"].dropna().to_numpy() for _, g in long_df.groupby("sensor")]
    labels = [s for s, _ in long_df.groupby("sensor")]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axhline(0, color="black", linewidth=1, alpha=0.7)
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title("Boundary shift against legacy smcDC")
    ax.set_ylabel("idx_exit_start shift (samples)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_shift_vs_smcDC_boxplot.png", dpi=180)
    fig.savefig(fig_dir / "02_shift_vs_smcDC_boxplot.svg")
    plt.close(fig)


def plot_exit_rate(summary: pd.DataFrame, fig_dir: Path) -> None:
    ordered = summary.sort_values("has_exit_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(ordered["sensor"], ordered["has_exit_rate"], color="#3973ac")
    ax.set_ylim(0, 1)
    ax.set_title("Detected exit rate by source sensor")
    ax.set_ylabel("has_exit ratio")
    ax.tick_params(axis="x", rotation=35)
    for i, row in enumerate(ordered.itertuples(index=False)):
        ax.text(i, float(row.has_exit_rate) + 0.02, f"{int(row.has_exit_runs)}/{int(row.n_runs)}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_exit_rate_by_sensor.png", dpi=180)
    fig.savefig(fig_dir / "03_exit_rate_by_sensor.svg")
    plt.close(fig)


def plot_case_overlays(
    signal_df: pd.DataFrame,
    long_df: pd.DataFrame,
    sensors: list[str],
    fig_dir: Path,
    cases: list[int],
    runs_per_case: int,
) -> None:
    for case_id in cases:
        available = sorted(signal_df.loc[signal_df["case"] == case_id, "run"].astype(int).tolist())
        for run_id in available[:runs_per_case]:
            row = signal_df[(signal_df.case == case_id) & (signal_df.run == run_id)]
            if row.empty:
                continue
            row = row.iloc[0]
            fig, axes = plt.subplots(len(sensors), 1, figsize=(12, 1.8 * len(sensors)), sharex=True)
            if len(sensors) == 1:
                axes = [axes]
            for ax, sensor in zip(axes, sensors):
                x = parse_signal(row[sensor])
                t = np.arange(len(x))
                ax.plot(t, normalized_trace(x), color="#3f5f7f", linewidth=0.7)
                b = long_df[(long_df.case == case_id) & (long_df.run == run_id) & (long_df.sensor == sensor)]
                if not b.empty:
                    idx = int(b.iloc[0].idx_exit_start)
                    has_exit = bool(b.iloc[0].has_exit)
                    ax.axvline(idx, color="#c93f32" if has_exit else "#888888", linewidth=1.4, linestyle="-" if has_exit else "--")
                    ax.text(idx, 5.2, "exit" if has_exit else "no-exit", color="#c93f32" if has_exit else "#555555", fontsize=8, rotation=90, va="top")
                ax.set_ylabel(sensor)
                ax.set_ylim(-6.5, 6.5)
                ax.grid(True, axis="x", alpha=0.2)
            axes[-1].set_xlabel("sample index")
            fig.suptitle(f"Sensor-source boundary overlay: case {case_id}, run {run_id}", y=0.995)
            fig.tight_layout()
            fig.savefig(fig_dir / f"04_overlay_case{case_id:02d}_run{run_id:02d}.png", dpi=180)
            plt.close(fig)


def plot_case_mean_shift(long_df: pd.DataFrame, fig_dir: Path) -> None:
    case_shift = long_df.groupby(["case", "sensor"], as_index=False)["shift_vs_smcDC"].mean()
    pivot = case_shift.pivot(index="case", columns="sensor", values="shift_vs_smcDC").sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1500, vmax=1500)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Mean boundary shift by case")
    ax.set_ylabel("case")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("mean shift vs smcDC (samples)")
    fig.tight_layout()
    fig.savefig(fig_dir / "05_case_mean_shift_heatmap.png", dpi=180)
    fig.savefig(fig_dir / "05_case_mean_shift_heatmap.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensors", nargs="+", default=["smcDC", "smcAC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"], choices=SENSORS)
    parser.add_argument("--overlay-cases", nargs="+", type=int, default=[1, 8, 12, 13, 14])
    parser.add_argument("--runs-per-case", type=int, default=2)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--hop", type=int, default=16)
    parser.add_argument("--threshold-quantile", type=float, default=0.97)
    parser.add_argument("--threshold-mad-multiplier", type=float, default=5.0)
    args = parser.parse_args()

    cfg = DetectorConfig(
        window=args.window,
        hop=args.hop,
        threshold_quantile=args.threshold_quantile,
        threshold_mad_multiplier=args.threshold_mad_multiplier,
    )
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ANALYSIS_ROOT / f"{ts}_sensor_source_visualization"
    fig_dir = out / "figures"
    data_dir = out / "data"
    for d in [fig_dir, data_dir, DATASET_OUT]:
        d.mkdir(parents=True, exist_ok=True)

    signal_df = pd.read_csv(SIGNAL_CSV, usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(PROCESS_CSV)
    base = pd.read_csv(BASE_SEG_CSV)
    signal_df = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    base = base[base["case"].isin(CASE_SCOPE)].copy()
    legacy_exit = pd.read_csv(LEGACY_EXIT_CSV) if LEGACY_EXIT_CSV.exists() else None

    sensor_tables: dict[str, pd.DataFrame] = {}
    merged_tables: dict[str, pd.DataFrame] = {}
    for sensor in args.sensors:
        table = generate_sensor_table(signal_df, process_df, sensor, cfg)
        merged = merge_with_base(base, table, sensor)
        table.to_csv(DATASET_OUT / f"seg_reverse_kurtosis_{sensor}.csv", index=False)
        merged.to_csv(DATASET_OUT / f"seg_peng2026_steady5_exitfix_reverse_kurtosis_{sensor}.csv", index=False)
        table.to_csv(data_dir / f"seg_reverse_kurtosis_{sensor}.csv", index=False)
        merged.to_csv(data_dir / f"seg_peng2026_steady5_exitfix_reverse_kurtosis_{sensor}.csv", index=False)
        sensor_tables[sensor] = table
        merged_tables[sensor] = merged

    summary, long_df = summarize_tables(sensor_tables, legacy_exit)
    summary.to_csv(data_dir / "sensor_source_summary.csv", index=False)
    long_df.to_csv(data_dir / "sensor_source_boundaries_long.csv", index=False)

    plot_heatmap(long_df, fig_dir)
    plot_shift_box(long_df, fig_dir)
    plot_exit_rate(summary, fig_dir)
    plot_case_mean_shift(long_df, fig_dir)
    plot_case_overlays(signal_df, long_df, args.sensors, fig_dir, args.overlay_cases, args.runs_per_case)

    metadata = {
        "created_at": ts,
        "detector_config": asdict(cfg),
        "sensors": args.sensors,
        "dataset_outputs": str(DATASET_OUT),
        "analysis_output": str(out),
        "legacy_reference": str(LEGACY_EXIT_CSV),
        "base_segment": str(BASE_SEG_CSV),
        "note": "idx_end in merged files is overwritten by each source sensor's detected idx_exit_start when has_exit=True.",
    }
    (out / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[done] analysis: {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

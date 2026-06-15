#!/usr/bin/env python3
"""Plot smoothed per-run sensor overlays with cutting-segment boundaries.

This complements ``plot_segment_sensor_overlays.py``. It keeps smc sensors raw
and applies one selected smoothing method only to vib_* and AE_* sensors:

  - hampel_median: Hampel spike replacement followed by rolling median
  - savgol: Savitzky-Golay smoothing
  - butterworth: zero-phase low-pass Butterworth filtering
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, medfilt, savgol_filter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_segment_sensor_overlays import (  # noqa: E402
    DEFAULT_SEG_CSV,
    DEFAULT_SIGNAL_CSV,
    SEGMENT_COLORS,
    SENSORS,
    parse_signal,
    robust_normalize,
    safe_bound,
)

SMOOTHED_SENSORS = {"vib_table", "vib_spindle", "AE_table", "AE_spindle"}
DEFAULT_OUTPUT_ROOT = ROOT / "eda/result/segment_sensor_overlay_smoothed_by_run"


def odd_at_most(value: int, n: int) -> int:
    value = max(3, int(value))
    if value % 2 == 0:
        value += 1
    max_odd = n if n % 2 == 1 else n - 1
    return max(3, min(value, max_odd))


def hampel_replace(x: np.ndarray, window: int, n_sigma: float) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=np.float64))
    rolling_median = s.rolling(window=window, center=True, min_periods=1).median()
    abs_dev = (s - rolling_median).abs()
    mad = abs_dev.rolling(window=window, center=True, min_periods=1).median()
    threshold = n_sigma * 1.4826 * mad.replace(0, np.nan)
    outlier = abs_dev > threshold
    y = s.copy()
    y[outlier.fillna(False)] = rolling_median[outlier.fillna(False)]
    return y.to_numpy(dtype=np.float64)


def smooth_hampel_median(x: np.ndarray, sensor: str) -> np.ndarray:
    if sensor.startswith("vib_"):
        window, sigma, median_kernel = 51, 3.5, 51
    else:
        window, sigma, median_kernel = 101, 5.0, 101
    window = odd_at_most(window, len(x))
    median_kernel = odd_at_most(median_kernel, len(x))
    y = hampel_replace(x, window=window, n_sigma=sigma)
    return medfilt(y, kernel_size=median_kernel)


def smooth_savgol(x: np.ndarray, sensor: str) -> np.ndarray:
    window = 151 if sensor.startswith("vib_") else 251
    window = odd_at_most(window, len(x))
    if window <= 3:
        return x.copy()
    return savgol_filter(x, window_length=window, polyorder=2, mode="interp")


def smooth_butterworth(x: np.ndarray, sensor: str) -> np.ndarray:
    cutoff = 0.035 if sensor.startswith("vib_") else 0.025
    order = 4
    if len(x) < 64:
        return x.copy()
    sos = butter(order, cutoff, btype="lowpass", output="sos")
    return sosfiltfilt(sos, x)


def apply_smoothing(x: np.ndarray, sensor: str, method: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if sensor not in SMOOTHED_SENSORS:
        return x
    if method == "hampel_median":
        return smooth_hampel_median(x, sensor)
    if method == "savgol":
        return smooth_savgol(x, sensor)
    if method == "butterworth":
        return smooth_butterworth(x, sensor)
    raise ValueError(f"Unsupported method={method}")


def plot_run_overlay(
    signal_row: pd.Series,
    seg_row: pd.Series,
    output_path: Path,
    method: str,
    dpi: int,
) -> dict[str, object]:
    raw_arrays = {sensor: parse_signal(signal_row[sensor]) for sensor in SENSORS}
    signal_length = min(len(raw_arrays[sensor]) for sensor in SENSORS)
    arrays = {
        sensor: apply_smoothing(raw_arrays[sensor][:signal_length], sensor, method)
        for sensor in SENSORS
    }

    idx_noload_end = safe_bound(seg_row.idx_noload_end, 0, signal_length)
    idx_start = safe_bound(seg_row.idx_start, idx_noload_end, signal_length)
    idx_end = safe_bound(seg_row.idx_end, idx_start, signal_length)

    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    spans = [
        ("no-load", 0, idx_noload_end),
        ("entry", idx_noload_end, idx_start),
        ("steady", idx_start, idx_end),
        ("exit", idx_end, signal_length),
    ]
    for label, start, end in spans:
        if end > start:
            ax.axvspan(start, end, color=SEGMENT_COLORS[label], alpha=0.65, linewidth=0)
            ax.text((start + end) / 2, 6.35, label, ha="center", va="bottom", fontsize=9, color="#333333")

    colors = plt.get_cmap("tab10").colors
    t = np.arange(signal_length)
    for i, sensor in enumerate(SENSORS):
        y = robust_normalize(arrays[sensor])
        linewidth = 0.85 if sensor in SMOOTHED_SENSORS else 0.65
        ax.plot(t, y, linewidth=linewidth, alpha=0.9, label=sensor, color=colors[i % len(colors)])

    for label, idx, color in [
        ("no-load end", idx_noload_end, "#606060"),
        ("steady start", idx_start, "#2e7d32"),
        ("exit start", idx_end, "#b71c1c"),
    ]:
        ax.axvline(idx, color=color, linewidth=1.5, linestyle="--")
        ax.text(idx, -6.35, label, rotation=90, va="bottom", ha="right", fontsize=8, color=color)

    has_exit = bool(seg_row.has_exit) if "has_exit" in seg_row.index else idx_end < signal_length
    title = f"Case {int(seg_row.case)} Run {int(seg_row.run)} | {method} overlay | has_exit={has_exit}"
    subtitle = (
        f"smoothed sensors=vib_*+AE_* | idx_noload_end={idx_noload_end}, "
        f"idx_start={idx_start}, idx_end={idx_end}, source={seg_row.get('exit_source', '-')}"
    )
    ax.set_title(title, fontsize=13, pad=18)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#555555")
    ax.set_xlabel("sample index")
    ax.set_ylabel("robust-normalized amplitude (clipped z)")
    ax.set_ylim(-6.6, 6.75)
    ax.set_xlim(0, signal_length)
    ax.grid(True, axis="x", alpha=0.18)
    ax.legend(loc="upper right", ncol=3, frameon=True, fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    return {
        "case": int(seg_row.case),
        "run": int(seg_row.run),
        "method": method,
        "path": str(output_path),
        "idx_noload_end": idx_noload_end,
        "idx_start": idx_start,
        "idx_end": idx_end,
        "signal_length": signal_length,
        "has_exit": has_exit,
    }


def write_report(output_dir: Path, index_df: pd.DataFrame, args: argparse.Namespace) -> None:
    method = args.method
    report = f"""# Segment Sensor Overlay Smoothed By Run: {method}

Generated: {datetime.now().isoformat(timespec="seconds")}

## Source

- Segment CSV: `{args.segment_csv}`
- Signal CSV: `{args.signal_csv}`
- Segment boundaries are unchanged from the source CSV.
- Raw sensors kept unchanged: `smcAC`, `smcDC`
- Smoothed sensors: `vib_table`, `vib_spindle`, `AE_table`, `AE_spindle`

## Method

- `hampel_median`: Hampel spike replacement + rolling median
- `savgol`: Savitzky-Golay smoothing
- `butterworth`: zero-phase low-pass Butterworth filtering

This directory uses: `{method}`

## Output

- Figures: `figures/case_*/case_*_run_*_{method}_overlay.png`
- Index: `index.csv`
- Metadata: `metadata.json`

## Summary

- Runs plotted: {len(index_df)}
- Runs with exit segment: {int(index_df['has_exit'].sum()) if not index_df.empty else 0}
- Runs without exit segment: {len(index_df) - int(index_df['has_exit'].sum()) if not index_df.empty else 0}
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")


def run_method(args: argparse.Namespace, method: str) -> dict[str, object]:
    seg_df = pd.read_csv(args.segment_csv)
    signal_df = pd.read_csv(args.signal_csv, usecols=["case", "run"] + SENSORS)
    if args.case is not None:
        seg_df = seg_df[seg_df["case"] == args.case].copy()
        signal_df = signal_df[signal_df["case"] == args.case].copy()
    if args.run is not None:
        seg_df = seg_df[seg_df["run"] == args.run].copy()
        signal_df = signal_df[signal_df["run"] == args.run].copy()

    signal_lookup = {(int(row.case), int(row.run)): row for row in signal_df.itertuples(index=False)}
    output_dir = args.output_root / method
    figure_root = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    missing: list[dict[str, int]] = []
    for seg_row in seg_df.sort_values(["case", "run"]).itertuples(index=False):
        key = (int(seg_row.case), int(seg_row.run))
        signal_row = signal_lookup.get(key)
        if signal_row is None:
            missing.append({"case": key[0], "run": key[1]})
            continue
        path = figure_root / f"case_{key[0]:02d}" / f"case_{key[0]:02d}_run_{key[1]:02d}_{method}_overlay.png"
        records.append(
            plot_run_overlay(
                pd.Series(signal_row._asdict()),
                pd.Series(seg_row._asdict()),
                path,
                method,
                args.dpi,
            )
        )

    index_df = pd.DataFrame(records)
    index_df.to_csv(output_dir / "index.csv", index=False)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": method,
        "segment_csv": str(args.segment_csv),
        "signal_csv": str(args.signal_csv),
        "output_dir": str(output_dir),
        "n_plotted": len(records),
        "n_missing_signals": len(missing),
        "missing_signals": missing,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    args_for_report = argparse.Namespace(**{**vars(args), "method": method})
    write_report(output_dir, index_df, args_for_report)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-csv", type=Path, default=DEFAULT_SEG_CSV)
    parser.add_argument("--signal-csv", type=Path, default=DEFAULT_SIGNAL_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--method",
        choices=["hampel_median", "savgol", "butterworth", "all"],
        default="all",
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--case", type=int, default=None)
    parser.add_argument("--run", type=int, default=None)
    args = parser.parse_args()

    methods = ["hampel_median", "savgol", "butterworth"] if args.method == "all" else [args.method]
    summaries = [run_method(args, method) for method in methods]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "run_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    for summary in summaries:
        print(f"[done] method={summary['method']} plotted={summary['n_plotted']} missing={summary['n_missing_signals']} output={summary['output_dir']}")


if __name__ == "__main__":
    main()

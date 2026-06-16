#!/usr/bin/env python3
"""Plot per-run multi-sensor overlays with cutting-segment boundaries.

Uses the v2 segment table:
  datasets/nasa/cutting_segment_v2/seg_peng2026_steady5_exitfix_reverse_kurtosis.csv

For each run in the segment table, all six sensors are robust-normalized and
overlaid in one figure. Segment regions are shaded and boundaries are marked:
no-load, entry, steady, exit.
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

ROOT = Path(__file__).resolve().parents[1]

SENSORS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
DEFAULT_SEG_CSV = ROOT / "datasets/nasa/cutting_segment_v2/seg_peng2026_steady5_exitfix_reverse_kurtosis.csv"
DEFAULT_SIGNAL_CSV = ROOT / "datasets/nasa/raw_signal.csv"
DEFAULT_OUTPUT_DIR = ROOT / "eda/result/segment_sensor_overlay_by_run"

SEGMENT_COLORS = {
    "no-load": "#e8edf3",
    "entry": "#fdebd3",
    "steady": "#e3f2df",
    "exit": "#f8d7da",
}


def parse_signal(value: object) -> np.ndarray:
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    arr = np.fromstring(text, sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def robust_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = float(np.std(x))
    if scale <= 1e-12:
        return np.zeros_like(x)
    z = (x - med) / scale
    return np.clip(z, -6.0, 6.0)


def safe_bound(value: object, lower: int, upper: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = lower
    return int(np.clip(v, lower, upper))


def plot_run_overlay(
    signal_row: pd.Series,
    seg_row: pd.Series,
    output_path: Path,
    dpi: int,
) -> dict[str, object]:
    arrays = {sensor: parse_signal(signal_row[sensor]) for sensor in SENSORS}
    signal_length = min(len(arrays[sensor]) for sensor in SENSORS)
    if signal_length <= 0:
        raise ValueError(f"empty signal for case={seg_row.case}, run={seg_row.run}")

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
            ax.text(
                (start + end) / 2,
                6.35,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#333333",
            )

    colors = plt.get_cmap("tab10").colors
    t = np.arange(signal_length)
    for i, sensor in enumerate(SENSORS):
        y = robust_normalize(arrays[sensor][:signal_length])
        ax.plot(t, y, linewidth=0.75, alpha=0.85, label=sensor, color=colors[i % len(colors)])

    boundaries = [
        ("no-load end", idx_noload_end, "#606060"),
        ("steady start", idx_start, "#2e7d32"),
        ("exit start", idx_end, "#b71c1c"),
    ]
    for label, idx, color in boundaries:
        ax.axvline(idx, color=color, linewidth=1.5, linestyle="--")
        ax.text(idx, -6.35, label, rotation=90, va="bottom", ha="right", fontsize=8, color=color)

    has_exit = bool(seg_row.has_exit) if "has_exit" in seg_row.index else idx_end < signal_length
    title = (
        f"Case {int(seg_row.case)} Run {int(seg_row.run)} | "
        f"sensor overlay with segment boundaries | has_exit={has_exit}"
    )
    subtitle = (
        f"idx_noload_end={idx_noload_end}, idx_start={idx_start}, "
        f"idx_end={idx_end}, signal_length={signal_length}, source={seg_row.get('exit_source', '-')}"
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
        "path": str(output_path),
        "idx_noload_end": idx_noload_end,
        "idx_start": idx_start,
        "idx_end": idx_end,
        "signal_length": signal_length,
        "has_exit": has_exit,
    }


def write_report(output_dir: Path, index_df: pd.DataFrame, args: argparse.Namespace) -> None:
    n_cases = int(index_df["case"].nunique()) if not index_df.empty else 0
    n_runs = int(len(index_df))
    exit_runs = int(index_df["has_exit"].sum()) if not index_df.empty else 0
    report = f"""# Segment Sensor Overlay By Run

Generated: {datetime.now().isoformat(timespec="seconds")}

## Source

- Segment CSV: `{args.segment_csv}`
- Signal CSV: `{args.signal_csv}`
- Sensors: `{', '.join(SENSORS)}`

## Output

- Figures: `figures/case_*/case_*_run_*_segment_overlay.png`
- Index: `index.csv`
- Metadata: `metadata.json`

## Summary

- Cases: {n_cases}
- Runs plotted: {n_runs}
- Runs with exit segment: {exit_runs}
- Runs without exit segment: {n_runs - exit_runs}

Each figure overlays all six robust-normalized sensor traces in one axis and marks
`idx_noload_end`, `idx_start`, and `idx_end` from the provided segment table.
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-csv", type=Path, default=DEFAULT_SEG_CSV)
    parser.add_argument("--signal-csv", type=Path, default=DEFAULT_SIGNAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--case", type=int, default=None, help="Optional single case filter for quick checks.")
    parser.add_argument("--run", type=int, default=None, help="Optional single run filter for quick checks.")
    args = parser.parse_args()

    seg_df = pd.read_csv(args.segment_csv)
    signal_df = pd.read_csv(args.signal_csv, usecols=["case", "run"] + SENSORS)
    if args.case is not None:
        seg_df = seg_df[seg_df["case"] == args.case].copy()
        signal_df = signal_df[signal_df["case"] == args.case].copy()
    if args.run is not None:
        seg_df = seg_df[seg_df["run"] == args.run].copy()
        signal_df = signal_df[signal_df["run"] == args.run].copy()

    signal_lookup = {
        (int(row.case), int(row.run)): row
        for row in signal_df.itertuples(index=False)
    }

    output_dir = args.output_dir
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
        signal_series = pd.Series(signal_row._asdict())
        seg_series = pd.Series(seg_row._asdict())
        path = figure_root / f"case_{key[0]:02d}" / f"case_{key[0]:02d}_run_{key[1]:02d}_segment_overlay.png"
        records.append(plot_run_overlay(signal_series, seg_series, path, args.dpi))

    index_df = pd.DataFrame(records)
    index_df.to_csv(output_dir / "index.csv", index=False)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "segment_csv": str(args.segment_csv),
        "signal_csv": str(args.signal_csv),
        "output_dir": str(output_dir),
        "n_plotted": len(records),
        "n_missing_signals": len(missing),
        "missing_signals": missing,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(output_dir, index_df, args)

    print(f"[done] plotted={len(records)} missing={len(missing)} output={output_dir}")


if __name__ == "__main__":
    main()

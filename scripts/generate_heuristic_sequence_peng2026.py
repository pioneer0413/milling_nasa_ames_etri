#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from milling_experiment_framework.preprocessing.segmentation import CuttingStateSegmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate peng2026-inspired milling segmentation metadata.")
    parser.add_argument("--signal-path", default="datasets/processed/mill_signal_data.csv")
    parser.add_argument("--process-info-path", default="datasets/processed/mill_process_info.csv")
    parser.add_argument("--base-heuristic-path", default="datasets/metadata/heuristic_sequence.csv")
    parser.add_argument("--output-path", default="datasets/metadata/heuristic_sequence_peng2026.csv")
    parser.add_argument("--sensor", default="smcDC")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--hop-size", type=int, default=1)
    parser.add_argument("--target-length", type=int, default=4000)
    parser.add_argument("--idle-delta-ratio", type=float, default=0.35)
    parser.add_argument("--idle-delta", type=float, default=None)
    parser.add_argument("--statistic", default="rms", choices=["rms", "mean_abs", "std", "energy"])
    parser.add_argument("--reference-mode", default="middle", choices=["middle", "whole", "local"])
    parser.add_argument("--min-steady-length", type=int, default=4000)
    parser.add_argument("--candidate-merge-gap", type=int, default=32)
    parser.add_argument("--no-tail-idle", action="store_true")
    parser.add_argument("--middle-window-mode", default="center", choices=["center", "lowest_std"])
    return parser.parse_args()


def load_signal_cell(value: Any) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=float)
    return np.asarray(value, dtype=float)


def build_note(args: argparse.Namespace, diag: dict[str, Any]) -> str:
    fallback = diag.get("fallback_used")
    return (
        "peng2026_inspired("
        f"sensor={args.sensor},window={args.window_size},hop={args.hop_size},"
        f"stat={args.statistic},idle_delta_ratio={args.idle_delta_ratio},"
        f"reference={args.reference_mode},xi={diag.get('selected_xi')},"
        f"target_length={args.target_length},merge_gap={args.candidate_merge_gap},"
        f"fallback={fallback})"
    )


def main() -> None:
    args = parse_args()
    signal_path = Path(args.signal_path)
    process_info_path = Path(args.process_info_path)
    base_heuristic_path = Path(args.base_heuristic_path)
    output_path = Path(args.output_path)

    signals = pd.read_csv(signal_path)
    process = pd.read_csv(process_info_path)
    base = pd.read_csv(base_heuristic_path)
    if args.sensor not in signals.columns:
        raise ValueError(f"sensor={args.sensor} is not in {signal_path}")

    data = base[["case", "run"]].merge(signals, on=["case", "run"], how="left").merge(process, on=["case", "run"], how="left", suffixes=("", "_process"))
    missing = data[args.sensor].isna()
    if missing.any():
        missing_keys = data.loc[missing, ["case", "run"]].to_dict(orient="records")
        raise ValueError(f"Missing signal rows for {missing_keys[:5]}")

    segmenter = CuttingStateSegmenter(
        window_size=args.window_size,
        hop_size=args.hop_size,
        target_length=args.target_length,
        idle_delta_ratio=args.idle_delta_ratio,
        idle_delta=args.idle_delta,
        statistic=args.statistic,
        reference_mode=args.reference_mode,
        min_steady_length=args.min_steady_length,
        candidate_merge_gap=args.candidate_merge_gap,
        detect_tail_idle=not args.no_tail_idle,
        channel_axis="auto",
        padding_mode="reflect",
        middle_window_mode=args.middle_window_mode,
    )

    rows: list[dict[str, Any]] = []
    now = datetime.now().astimezone().isoformat()
    for row in data.itertuples(index=False):
        raw = load_signal_cell(getattr(row, args.sensor))
        result = segmenter.segment(raw)
        diag = result["diagnostics"]
        segments = result["segments"]
        idx_noload_end = int(segments["no_load"][1])
        idx_start = int(segments["steady"][0])
        idx_end = int(segments["steady"][1])
        signal_length = int(len(raw))
        index_fixed = int(min(idx_start + args.target_length, idx_end))
        if index_fixed <= idx_start:
            index_fixed = int(min(signal_length, idx_start + max(1, args.target_length)))
        rows.append(
            {
                "case": int(row.case),
                "run": int(row.run),
                "idx_noload_end": idx_noload_end,
                "idx_start": idx_start,
                "idx_end": idx_end,
                "index_fixed": index_fixed,
                "signal_length": signal_length,
                "status": "labeled",
                "note": build_note(args, diag),
                "time": getattr(row, "time"),
                "DOC": getattr(row, "DOC"),
                "feed": getattr(row, "feed"),
                "material_name": getattr(row, "material_name"),
                "updated_at": now,
            }
        )

    output = pd.DataFrame(rows, columns=base.columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"wrote {output_path} rows={len(output)}")
    print(output[["idx_noload_end", "idx_start", "idx_end", "index_fixed"]].describe().to_string())


if __name__ == "__main__":
    main()

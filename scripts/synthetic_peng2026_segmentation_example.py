#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from milling_experiment_framework.preprocessing.segmentation import CuttingStateSegmenter


def make_synthetic_signal(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    no_load = rng.normal(0.0, 0.03, 900)
    entry = np.linspace(1.45, 0.9, 700) * np.sin(np.linspace(0, 70, 700)) + rng.normal(0.0, 0.22, 700)
    steady = 0.82 * np.sin(np.linspace(0, 360, 4300)) + rng.normal(0.0, 0.045, 4300)
    exit_cut = np.linspace(1.6, 0.6, 850) * np.sin(np.linspace(0, 85, 850)) + rng.normal(0.0, 0.25, 850)
    tail_idle = rng.normal(0.0, 0.03, 700)
    return np.concatenate([no_load, entry, steady, exit_cut, tail_idle])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic peng2026-inspired segmentation example.")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal = make_synthetic_signal()
    segmenter = CuttingStateSegmenter(
        window_size=64,
        hop_size=4,
        target_length=4000,
        idle_delta_ratio=0.5,
        statistic="rms",
        reference_mode="middle",
        min_steady_length=3000,
        candidate_merge_gap=8,
        detect_tail_idle=True,
    )
    result = segmenter.segment(signal)
    print("segment indices")
    for name, bounds in result["segments"].items():
        print(f"{name}: {bounds}")
    print("segment lengths")
    for name, (start, end) in result["segments"].items():
        print(f"{name}: {end - start}")
    print("diagnostics")
    for key in ["wf", "sf", "delta", "selected_xi", "num_candidates", "fallback_used", "tail_no_load"]:
        print(f"{key}: {result['diagnostics'].get(key)}")
    if args.plot:
        import matplotlib.pyplot as plt

        segmenter.plot(signal, result)
        plt.show()


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Segment = tuple[int, int]


@dataclass(frozen=True)
class CandidateRegion:
    start: int
    end: int
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start


class CuttingStateSegmenter:
    """peng2026-inspired milling process state segmentation.

    The paper explicitly gives the xi sweep values and uses a 4000-sample
    steady extraction length. Window sizes, thresholds, references, merging,
    and padding are implementation hyperparameters, not reported paper values.
    """

    def __init__(
        self,
        window_size: int,
        hop_size: int,
        target_length: int = 4000,
        idle_delta_ratio: float | None = 0.25,
        idle_delta: float | None = None,
        xi_values: list[float] | tuple[float, ...] | None = None,
        statistic: str = "rms",
        reference_mode: str = "middle",
        min_steady_length: int | None = None,
        candidate_merge_gap: int = 0,
        detect_tail_idle: bool = True,
        channel_axis: str | int = "auto",
        padding_mode: str = "reflect",
        middle_window_mode: str = "center",
    ) -> None:
        self.window_size = int(window_size)
        self.hop_size = int(hop_size)
        self.target_length = int(target_length)
        self.idle_delta_ratio = idle_delta_ratio
        self.idle_delta = idle_delta
        self.xi_values = list(xi_values or [2, 2.5, 3, 3.5, 4, 4.5, 5])
        self.statistic = statistic
        self.reference_mode = reference_mode
        self.min_steady_length = int(min_steady_length) if min_steady_length is not None else int(target_length)
        self.candidate_merge_gap = int(candidate_merge_gap)
        self.detect_tail_idle = bool(detect_tail_idle)
        self.channel_axis = channel_axis
        self.padding_mode = padding_mode
        self.middle_window_mode = middle_window_mode
        self._validate()

    def segment(self, signal: np.ndarray | list[float]) -> dict[str, Any]:
        x = self._as_channels_time(signal)
        signal_length = int(x.shape[1])
        if signal_length == 0:
            raise ValueError("signal must contain at least one sample")

        work = self._pad_if_needed(x, min_length=max(self.window_size * 3, 1))
        frame_starts, stats = self._window_statistics(work)
        wf = float(np.mean(stats[:3]))
        delta = self._resolve_delta(wf)

        cutting_start_idx = self._detect_leading_idle_end(frame_starts, stats, wf, delta, signal_length)
        if self.detect_tail_idle:
            cutting_end_idx = self._detect_trailing_idle_start(frame_starts, stats, wf, delta, signal_length, cutting_start_idx)
        else:
            cutting_end_idx = signal_length
        if cutting_end_idx <= cutting_start_idx:
            cutting_start_idx = 0
            cutting_end_idx = signal_length

        steady_start_idx, steady_end_idx, steady_diag = self._detect_steady_region(
            x,
            cutting_start_idx,
            cutting_end_idx,
        )

        segments: dict[str, Segment] = {
            "no_load": (0, cutting_start_idx),
            "entry": (cutting_start_idx, steady_start_idx),
            "steady": (steady_start_idx, steady_end_idx),
            "exit": (steady_end_idx, cutting_end_idx),
        }
        diagnostics_tail: Segment | None = None
        if self.detect_tail_idle and cutting_end_idx < signal_length:
            diagnostics_tail = (cutting_end_idx, signal_length)

        signals = {name: self._slice_like_input(signal, start, end) for name, (start, end) in segments.items()}
        diagnostics: dict[str, Any] = {
            "wf": wf,
            "sf": steady_diag["sf"],
            "delta": delta,
            "selected_xi": steady_diag["selected_xi"],
            "num_candidates": steady_diag["num_candidates"],
            "fallback_used": steady_diag["fallback_used"],
            "cutting_start_idx": cutting_start_idx,
            "cutting_end_idx": cutting_end_idx,
            "steady_start_idx": steady_start_idx,
            "steady_end_idx": steady_end_idx,
            "tail_no_load": diagnostics_tail,
            "params": {
                "window_size": self.window_size,
                "hop_size": self.hop_size,
                "target_length": self.target_length,
                "idle_delta_ratio": self.idle_delta_ratio,
                "idle_delta": self.idle_delta,
                "xi_values": self.xi_values,
                "statistic": self.statistic,
                "reference_mode": self.reference_mode,
                "min_steady_length": self.min_steady_length,
                "candidate_merge_gap": self.candidate_merge_gap,
                "detect_tail_idle": self.detect_tail_idle,
                "channel_axis": self.channel_axis,
                "padding_mode": self.padding_mode,
                "middle_window_mode": self.middle_window_mode,
            },
        }
        diagnostics.update({k: v for k, v in steady_diag.items() if k not in diagnostics})
        return {"segments": segments, "signals": signals, "diagnostics": diagnostics}

    def plot(self, signal: np.ndarray | list[float], result: dict[str, Any] | None = None):
        result = result or self.segment(signal)
        return plot_segmentation(signal, result["segments"], result.get("diagnostics"))

    def _validate(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.hop_size <= 0:
            raise ValueError("hop_size must be positive")
        if self.target_length <= 0:
            raise ValueError("target_length must be positive")
        if self.min_steady_length <= 0:
            raise ValueError("min_steady_length must be positive")
        if self.statistic not in {"rms", "mean_abs", "std", "energy"}:
            raise ValueError(f"Unsupported statistic={self.statistic}")
        if self.reference_mode not in {"middle", "whole", "local"}:
            raise ValueError(f"Unsupported reference_mode={self.reference_mode}")
        if self.middle_window_mode not in {"center", "lowest_std"}:
            raise ValueError(f"Unsupported middle_window_mode={self.middle_window_mode}")
        if self.candidate_merge_gap < 0:
            raise ValueError("candidate_merge_gap must be non-negative")
        if not self.xi_values:
            raise ValueError("xi_values must not be empty")
        if self.idle_delta is None and self.idle_delta_ratio is None:
            raise ValueError("Either idle_delta or idle_delta_ratio must be provided")

    def _as_channels_time(self, signal: np.ndarray | list[float]) -> np.ndarray:
        arr = np.asarray(signal, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("signal must be 1D, (C, T), or (T, C)")
        axis = self.channel_axis
        if axis == "auto":
            if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= 32:
                return arr
            if arr.shape[1] <= 32:
                return arr.T
            return arr
        if axis in (0, "0", "channels_first"):
            return arr
        if axis in (1, "1", "channels_last"):
            return arr.T
        raise ValueError(f"Unsupported channel_axis={self.channel_axis}")

    def _pad_if_needed(self, x: np.ndarray, min_length: int) -> np.ndarray:
        if x.shape[1] >= min_length:
            return x
        pad_width = min_length - x.shape[1]
        mode = self.padding_mode
        if mode == "none":
            mode = "edge"
        if mode == "reflect" and x.shape[1] < 2:
            mode = "edge"
        return np.pad(x, ((0, 0), (0, pad_width)), mode=mode)

    def _window_statistics(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = x.shape[1]
        if n < self.window_size:
            x = self._pad_if_needed(x, self.window_size)
            n = x.shape[1]
        starts = np.arange(0, n - self.window_size + 1, self.hop_size, dtype=int)
        if len(starts) == 0:
            starts = np.array([0], dtype=int)
        stats = np.asarray([self._stat(x[:, start : start + self.window_size]) for start in starts], dtype=float)
        return starts, stats

    def _stat(self, window: np.ndarray) -> float:
        if self.statistic == "rms":
            return float(np.sqrt(np.mean(np.square(window))))
        if self.statistic == "mean_abs":
            return float(np.mean(np.abs(window)))
        if self.statistic == "std":
            return float(np.std(window))
        if self.statistic == "energy":
            return float(np.sum(np.square(window)) / max(window.size, 1))
        raise ValueError(f"Unsupported statistic={self.statistic}")

    def _std(self, window: np.ndarray) -> float:
        return float(np.std(window))

    def _resolve_delta(self, wf: float) -> float:
        if self.idle_delta is not None:
            return float(self.idle_delta)
        ratio = float(self.idle_delta_ratio or 0.0)
        return max(abs(wf) * ratio, np.finfo(float).eps)

    def _detect_leading_idle_end(
        self,
        starts: np.ndarray,
        stats: np.ndarray,
        wf: float,
        delta: float,
        signal_length: int,
    ) -> int:
        for start, stat in zip(starts, stats):
            if abs(float(stat) - wf) >= delta:
                return int(np.clip(start, 0, signal_length))
        return 0

    def _detect_trailing_idle_start(
        self,
        starts: np.ndarray,
        stats: np.ndarray,
        wf: float,
        delta: float,
        signal_length: int,
        cutting_start_idx: int,
    ) -> int:
        for start, stat in zip(starts[::-1], stats[::-1]):
            if abs(float(stat) - wf) >= delta:
                return int(np.clip(start + self.window_size, cutting_start_idx, signal_length))
        return signal_length

    def _detect_steady_region(self, x: np.ndarray, cutting_start_idx: int, cutting_end_idx: int) -> tuple[int, int, dict[str, Any]]:
        cut = x[:, cutting_start_idx:cutting_end_idx]
        cut_length = cut.shape[1]
        if cut_length <= 0:
            end = min(x.shape[1], cutting_start_idx + self.target_length)
            return cutting_start_idx, end, self._steady_diag(0.0, None, 0, "empty_cutting")

        cut = self._pad_if_needed(cut, min_length=max(self.window_size * 3, self.min_steady_length))
        starts, stats = self._window_statistics(cut)
        stds = np.asarray([self._std(cut[:, start : start + self.window_size]) for start in starts], dtype=float)

        middle_indices = self._middle_reference_indices(starts, stds)
        sf = float(np.mean([self._std(cut[:, starts[i] : starts[i] + self.window_size]) for i in middle_indices]))
        wg = self._reference_value(cut, starts, stats, middle_indices)

        candidates: list[CandidateRegion] = []
        selected_xi: float | None = None
        for xi in self.xi_values:
            mask = self._candidate_mask(stats, wg, sf, float(xi))
            mask = self._merge_candidate_gaps(mask)
            candidates = self._candidate_regions(mask, starts, cut, cutting_start_idx)
            if candidates:
                selected_xi = float(xi)
                break

        fallback_used: str | bool = False
        if not candidates:
            fallback_used = "rolling_std"
            region = self._rolling_std_fallback(x, cutting_start_idx, cutting_end_idx)
            if region is None:
                fallback_used = "center"
                region = self._center_fallback(cutting_start_idx, cutting_end_idx)
            candidates = [region]

        best = min(candidates, key=lambda c: (c.score, -c.length, c.start))
        start = int(np.clip(best.start, cutting_start_idx, cutting_end_idx))
        end = int(np.clip(best.end, start, cutting_end_idx))
        if end <= start:
            start, end = self._center_fallback(cutting_start_idx, cutting_end_idx).start, self._center_fallback(cutting_start_idx, cutting_end_idx).end
            fallback_used = fallback_used or "center"

        return start, end, self._steady_diag(sf, selected_xi, len(candidates), fallback_used, wg)

    def _middle_reference_indices(self, starts: np.ndarray, stds: np.ndarray) -> list[int]:
        if len(starts) <= 3:
            return list(range(len(starts)))
        if self.middle_window_mode == "lowest_std":
            mid = len(starts) // 2
            radius = max(3, len(starts) // 4)
            lo = max(0, mid - radius)
            hi = min(len(starts), mid + radius + 1)
            local = np.argsort(stds[lo:hi])[:3] + lo
            return sorted(int(i) for i in local)
        center = len(starts) // 2
        first = int(np.clip(center - 1, 0, len(starts) - 3))
        return [first, first + 1, first + 2]

    def _reference_value(self, cut: np.ndarray, starts: np.ndarray, stats: np.ndarray, middle_indices: list[int]) -> float:
        if self.reference_mode == "middle":
            return float(np.mean(stats[middle_indices]))
        if self.reference_mode == "whole":
            return self._stat(cut)
        if self.reference_mode == "local":
            return float(np.median(stats))
        raise ValueError(f"Unsupported reference_mode={self.reference_mode}")

    def _candidate_mask(self, stats: np.ndarray, wg: float, sf: float, xi: float) -> np.ndarray:
        threshold = max(xi * sf, np.finfo(float).eps)
        return np.abs(stats - wg) < threshold

    def _merge_candidate_gaps(self, mask: np.ndarray) -> np.ndarray:
        if self.candidate_merge_gap == 0 or len(mask) == 0:
            return mask
        merged = mask.copy()
        false_start: int | None = None
        for i, value in enumerate(np.r_[mask, True]):
            if not value and false_start is None:
                false_start = i
            elif value and false_start is not None:
                if false_start > 0 and i < len(mask) and i - false_start <= self.candidate_merge_gap:
                    merged[false_start:i] = True
                false_start = None
        return merged

    def _candidate_regions(
        self,
        mask: np.ndarray,
        starts: np.ndarray,
        cut: np.ndarray,
        cutting_start_idx: int,
    ) -> list[CandidateRegion]:
        regions: list[CandidateRegion] = []
        run_start: int | None = None
        for i, value in enumerate(np.r_[mask, False]):
            if value and run_start is None:
                run_start = i
            elif not value and run_start is not None:
                start = int(starts[run_start])
                end = int(starts[i - 1] + self.window_size)
                length = end - start
                if length >= self.min_steady_length:
                    steady_length = min(self.target_length, length)
                    sub_start, sub_end, score = self._best_subregion(cut, start, end, steady_length)
                    regions.append(
                        CandidateRegion(
                            cutting_start_idx + sub_start,
                            cutting_start_idx + min(sub_end, cut.shape[1]),
                            score,
                        )
                    )
                run_start = None
        return regions

    def _best_subregion(self, cut: np.ndarray, start: int, end: int, length: int) -> tuple[int, int, float]:
        if end - start <= length:
            return start, end, self._std(cut[:, start:end])
        best_start = start
        best_score = float("inf")
        for sub_start in range(start, end - length + 1, self.hop_size):
            sub_end = sub_start + length
            score = self._std(cut[:, sub_start:sub_end])
            if score < best_score:
                best_score = score
                best_start = sub_start
        return best_start, best_start + length, best_score

    def _rolling_std_fallback(self, x: np.ndarray, cutting_start_idx: int, cutting_end_idx: int) -> CandidateRegion | None:
        cut_length = cutting_end_idx - cutting_start_idx
        length = min(self.target_length, cut_length)
        if length <= 0:
            return None
        starts = range(cutting_start_idx, max(cutting_start_idx + 1, cutting_end_idx - length + 1), self.hop_size)
        best: CandidateRegion | None = None
        for start in starts:
            end = min(start + length, cutting_end_idx)
            score = self._std(x[:, start:end])
            region = CandidateRegion(start, end, score)
            if best is None or (region.score, region.start) < (best.score, best.start):
                best = region
        return best

    def _center_fallback(self, cutting_start_idx: int, cutting_end_idx: int) -> CandidateRegion:
        cut_length = max(0, cutting_end_idx - cutting_start_idx)
        length = min(self.target_length, cut_length)
        start = cutting_start_idx + max(0, (cut_length - length) // 2)
        end = min(cutting_end_idx, start + length)
        return CandidateRegion(start, end, float("inf"))

    def _steady_diag(
        self,
        sf: float,
        selected_xi: float | None,
        num_candidates: int,
        fallback_used: str | bool,
        wg: float | None = None,
    ) -> dict[str, Any]:
        return {
            "sf": float(sf),
            "wg": None if wg is None else float(wg),
            "selected_xi": selected_xi,
            "num_candidates": int(num_candidates),
            "fallback_used": fallback_used,
        }

    def _slice_like_input(self, signal: np.ndarray | list[float], start: int, end: int) -> np.ndarray:
        arr = np.asarray(signal)
        if arr.ndim == 1:
            return arr[start:end]
        if self.channel_axis in (1, "1", "channels_last") or (self.channel_axis == "auto" and arr.ndim == 2 and arr.shape[1] <= 32 < arr.shape[0]):
            return arr[start:end, :]
        return arr[:, start:end]


def plot_segmentation(signal: np.ndarray | list[float], segments: dict[str, Segment], diagnostics: dict[str, Any] | None = None):
    import matplotlib.pyplot as plt

    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 2:
        if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= 32:
            y = np.mean(arr, axis=0)
        else:
            y = np.mean(arr, axis=1)
    else:
        y = arr

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(y)), y, color="black", linewidth=0.8, alpha=0.8)
    colors = {
        "no_load": "#9ca3af",
        "entry": "#f59e0b",
        "steady": "#10b981",
        "exit": "#ef4444",
        "tail_no_load": "#6b7280",
    }
    labels = {
        "no_load": "no-load",
        "entry": "entry cut",
        "steady": "steady cut",
        "exit": "exit cut",
        "tail_no_load": "tail no-load",
    }
    for name, (start, end) in segments.items():
        if end > start:
            ax.axvspan(start, end, color=colors.get(name, "#60a5fa"), alpha=0.22, label=labels.get(name, name))
    tail = (diagnostics or {}).get("tail_no_load")
    if tail and tail[1] > tail[0]:
        ax.axvspan(tail[0], tail[1], color=colors["tail_no_load"], alpha=0.18, label=labels["tail_no_load"])
    ax.set_xlabel("sample index")
    ax.set_ylabel("signal")
    ax.legend(loc="upper right", ncols=3, fontsize=8)
    ax.margins(x=0)
    fig.tight_layout()
    return fig, ax

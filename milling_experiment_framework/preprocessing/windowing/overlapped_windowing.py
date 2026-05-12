from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OverlappedWindowResult:
    windows: np.ndarray  # [num_windows, window_size, num_sensors]
    masks: np.ndarray  # [num_windows, window_size]
    original_length: int
    target_length: int
    window_size: int
    stride: int
    window_names: list[str]
    start_indices: list[int]
    end_indices: list[int]
    valid_lengths: list[int]
    was_padded: bool
    was_truncated: bool
    padding_length: int
    status: str


def window_run(
    run_signal: np.ndarray,
    run_metadata: dict[str, Any] | None = None,
    window_config: dict[str, Any] | None = None,
) -> OverlappedWindowResult:
    """Split one run signal into fixed overlapped temporal windows.

    Parameters
    ----------
    run_signal:
        Array with shape ``[time_length, num_sensors]``.
    run_metadata:
        Accepted for API symmetry and logging callers; not used to set boundaries.
    window_config:
        Supports ``window_size``, ``stride``, ``num_windows``, ``target_length``,
        ``padding_value``, ``if_longer``, and ``if_shorter``.
    """

    del run_metadata
    cfg = window_config or {}
    x = np.asarray(run_signal, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"run_signal must be [time_length, num_sensors], got {x.shape}")

    window_size = int(cfg.get("window_size", 3000))
    stride = int(cfg.get("stride", 2000))
    num_windows = int(cfg.get("num_windows", 4))
    target_length = int(cfg.get("target_length", 9000))
    padding_value = float(cfg.get("padding_value", 0.0))
    if window_size <= 0 or stride <= 0 or num_windows <= 0 or target_length <= 0:
        raise ValueError("window_size, stride, num_windows, and target_length must be positive.")
    required_length = (num_windows - 1) * stride + window_size
    if required_length != target_length:
        raise ValueError(
            f"Window config inconsistent: (num_windows-1)*stride+window_size={required_length}, target_length={target_length}"
        )

    original_length = int(x.shape[0])
    was_padded = original_length < target_length
    was_truncated = original_length > target_length
    padding_length = max(target_length - original_length, 0)
    valid_run_length = min(original_length, target_length)

    if was_truncated:
        x_fixed = x[:target_length]
    elif was_padded:
        pad = np.full((padding_length, x.shape[1]), padding_value, dtype=np.float32)
        x_fixed = np.vstack([x, pad])
    else:
        x_fixed = x

    starts = [idx * stride for idx in range(num_windows)]
    ends = [start + window_size for start in starts]
    windows = []
    masks = []
    valid_lengths = []
    for start, end in zip(starts, ends):
        valid_start = min(start, valid_run_length)
        valid_end = min(end, valid_run_length)
        valid_length = max(valid_end - valid_start, 0)
        mask = np.zeros(window_size, dtype=bool)
        if valid_length:
            mask[:valid_length] = True
        windows.append(x_fixed[start:end])
        masks.append(mask)
        valid_lengths.append(int(valid_length))

    out = np.stack(windows, axis=0).astype(np.float32)
    mask_out = np.stack(masks, axis=0)
    expected_shape = (num_windows, window_size, x.shape[1])
    if out.shape != expected_shape:
        raise RuntimeError(f"Windowing produced shape {out.shape}, expected {expected_shape}")

    status = "ok"
    if was_padded:
        status = "padded_to_target_length"
    if was_truncated:
        status = "truncated_to_target_length"
    return OverlappedWindowResult(
        windows=out,
        masks=mask_out,
        original_length=original_length,
        target_length=target_length,
        window_size=window_size,
        stride=stride,
        window_names=[f"window_{idx + 1}" for idx in range(num_windows)],
        start_indices=starts,
        end_indices=ends,
        valid_lengths=valid_lengths,
        was_padded=was_padded,
        was_truncated=was_truncated,
        padding_length=int(padding_length),
        status=status,
    )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunWindowingResult:
    windows: np.ndarray
    original_time_length: int
    window_length: int
    num_windows: int
    stride: int
    window_start_indices: list[int]
    window_end_indices: list[int]
    short_run_policy: str
    padded: bool
    padding_length: int
    windowing_status: str


def _right_pad_channel_first(signal: np.ndarray, target_length: int) -> tuple[np.ndarray, int]:
    if signal.shape[1] >= target_length:
        return signal, 0
    pad_width = target_length - signal.shape[1]
    padded = np.pad(signal, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
    return padded, int(pad_width)


def _window_starts(length: int, window_length: int, num_windows: int) -> tuple[list[int], int]:
    if num_windows <= 0:
        raise ValueError("num_windows must be positive.")
    if window_length <= 0:
        raise ValueError("window_length must be positive.")
    if num_windows == 1:
        return [0], 0
    stride = int(np.floor((length - window_length) / (num_windows - 1)))
    stride = max(stride, 0)
    starts = [int(i * stride) for i in range(num_windows - 1)]
    starts.append(int(length - window_length))
    return starts, stride


def window_signal_channel_first(
    signal: np.ndarray,
    window_length: int = 3000,
    num_windows: int = 5,
    short_run_policy: str = "pad_to_window_length",
) -> RunWindowingResult:
    """Create fixed-count overlapped windows from a channel-first run signal.

    Parameters
    ----------
    signal:
        Array with shape ``[num_sensors, time_length]``.
    window_length:
        Temporal length of each window.
    num_windows:
        Number of windows to return. H4_S1 defaults to five.
    short_run_policy:
        Currently supports ``pad_to_window_length``. Short runs are right-zero
        padded to one window and then repeated, which keeps max pooling
        deterministic while preserving the required output shape.

    Returns
    -------
    RunWindowingResult
        ``windows`` has shape ``[num_windows, num_sensors, window_length]``.
    """

    array = np.asarray(signal, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"signal must have shape [num_sensors, time_length], got {array.shape}")
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    original_length = int(array.shape[1])
    if original_length <= 0:
        raise ValueError("time_length must be positive.")
    window_length = int(window_length)
    num_windows = int(num_windows)

    if original_length < window_length:
        if short_run_policy != "pad_to_window_length":
            raise ValueError(f"Unsupported short_run_policy for short runs: {short_run_policy}")
        padded_signal, padding_length = _right_pad_channel_first(array, window_length)
        windows = np.repeat(padded_signal[None, :, :], num_windows, axis=0)
        starts = [0 for _ in range(num_windows)]
        ends = [window_length for _ in range(num_windows)]
        stride = 0
        padded = True
        status = "padded_repeated_short_run"
    else:
        starts, stride = _window_starts(original_length, window_length, num_windows)
        windows = np.stack([array[:, start : start + window_length] for start in starts], axis=0)
        ends = [int(start + window_length) for start in starts]
        padding_length = 0
        padded = False
        status = "ok"

    expected_shape = (num_windows, array.shape[0], window_length)
    if windows.shape != expected_shape:
        raise RuntimeError(f"Windowing produced shape {windows.shape}, expected {expected_shape}")
    return RunWindowingResult(
        windows=windows.astype(np.float32),
        original_time_length=original_length,
        window_length=window_length,
        num_windows=num_windows,
        stride=int(stride),
        window_start_indices=starts,
        window_end_indices=ends,
        short_run_policy=short_run_policy,
        padded=padded,
        padding_length=int(padding_length),
        windowing_status=status,
    )

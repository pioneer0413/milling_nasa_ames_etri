from __future__ import annotations

import numpy as np


FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "peak_to_peak",
    "rms",
    "skewness",
    "kurtosis",
    "slope",
    "energy",
    "spectral_centroid",
    "band_energy",
    "dominant_frequency",
]


def _safe_series(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    arr = arr[np.isfinite(arr)]
    return arr


def extract_1d_features(x: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    values = _safe_series(x, mask)
    if values.size == 0:
        return {name: 0.0 for name in FEATURE_NAMES}
    mean = float(np.mean(values))
    std = float(np.std(values))
    centered = values - mean
    if std < 1e-12:
        skewness = 0.0
        kurtosis = 0.0
    else:
        z = centered / std
        skewness = float(np.mean(z**3))
        kurtosis = float(np.mean(z**4))
    if values.size >= 2:
        t = np.arange(values.size, dtype=np.float64)
        denom = float(np.sum((t - t.mean()) ** 2))
        slope = float(np.sum((t - t.mean()) * (values - mean)) / denom) if denom > 0 else 0.0
    else:
        slope = 0.0
    fft_mag = np.abs(np.fft.rfft(values))
    freqs = np.fft.rfftfreq(values.size, d=1.0)
    mag_sum = float(np.sum(fft_mag))
    spectral_centroid = float(np.sum(freqs * fft_mag) / (mag_sum + 1e-12))
    dominant_frequency = float(freqs[int(np.argmax(fft_mag))]) if fft_mag.size else 0.0
    energy = float(np.sum(values**2))
    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "peak_to_peak": float(np.max(values) - np.min(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "slope": slope,
        "energy": energy,
        "spectral_centroid": spectral_centroid,
        "band_energy": float(np.mean(values**2)),
        "dominant_frequency": dominant_frequency,
    }


def extract_multisensor_features(
    signal: np.ndarray,
    sensor_columns: list[str],
    mask: np.ndarray | None = None,
    prefix: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract handcrafted features from ``[time, sensors]`` signal."""
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"signal must be [time, sensors], got {x.shape}")
    if x.shape[1] != len(sensor_columns):
        raise ValueError(f"sensor column count {len(sensor_columns)} does not match signal shape {x.shape}")
    row = []
    names = []
    name_prefix = f"{prefix}__" if prefix else ""
    for sensor_idx, sensor in enumerate(sensor_columns):
        feats = extract_1d_features(x[:, sensor_idx], mask)
        for feature_name in FEATURE_NAMES:
            row.append(feats[feature_name])
            names.append(f"{name_prefix}{sensor}__{feature_name}")
    row = np.nan_to_num(np.asarray(row, dtype=np.float64), nan=0.0, posinf=1e12, neginf=-1e12)
    row = np.clip(row, -1e12, 1e12)
    return row, names

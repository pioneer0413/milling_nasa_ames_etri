#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import html
import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, pearsonr, skew, spearmanr
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from milling_experiment_framework.experiment_logging.environment import collect_environment


PREFIX = "H1_S1"
TOPIC = "feture_VB_segment_association_eda_full_length_with_entry_steady_exit"
CASE_SCOPE = [1, 2, 8, 9, 12, 14]
PAIR_DEFINITION = {"A": [1, 9], "B": [2, 12], "C": [8, 14]}
PAIR_BY_CASE = {case: pair for pair, cases in PAIR_DEFINITION.items() for case in cases}
SEGMENT_SETTINGS = ["full_length", "steady", "entry", "exit", "entry_steady", "entry_exit", "steady_exit", "entry_steady_exit"]
FEATURE_NAMES = ["mean", "std", "max", "min", "peak_to_peak", "kurtosis", "skewness", "spectral_centroid", "band_energy"]
EXCLUDED_SEGMENTS = [
    "no_load",
    "no_load_entry",
    "no_load_steady",
    "no_load_exit",
    "no_load_entry_steady",
    "no_load_entry_exit",
    "no_load_steady_exit",
    "no_load_entry_steady_exit",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        path.write_text(yaml.safe_dump(to_builtin(payload), sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        write_json(path.with_suffix(".json"), payload)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value


def parse_signal_cell(value: Any) -> np.ndarray:
    arr = np.asarray(ast.literal_eval(value) if isinstance(value, str) else value, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def infer_sensor_columns(signal: pd.DataFrame) -> list[str]:
    excluded = {"case", "run", "enable", "VB", "time", "DOC", "feed", "material", "material_name"}
    sensors: list[str] = []
    for col in signal.columns:
        if col in excluded:
            continue
        sample = signal[col].dropna()
        if not sample.empty and isinstance(sample.iloc[0], str) and sample.iloc[0].strip().startswith("["):
            sensors.append(col)
    return sensors


def robust_scale_abs(arr: np.ndarray) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    scale = mad * 1.4826 if mad > 1e-12 else np.std(finite)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return np.abs((arr - med) / scale)


def infer_cut_bounds(sensor_arrays: dict[str, np.ndarray]) -> tuple[int, int, dict[str, Any]]:
    min_len = min(len(x) for x in sensor_arrays.values())
    if min_len < 9:
        return 0, min_len, {"method": "fallback_full_length_short_signal"}
    stacked = np.vstack([robust_scale_abs(x[:min_len]) for x in sensor_arrays.values()])
    activity = np.nanmedian(stacked, axis=0)
    smooth_window = max(25, min_len // 100)
    kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
    smooth = np.convolve(activity, kernel, mode="same")
    threshold = max(float(np.quantile(smooth, 0.65)), float(np.median(smooth) + 0.1 * (np.quantile(smooth, 0.95) - np.median(smooth))))
    active = np.flatnonzero(smooth >= threshold)
    if active.size < max(10, min_len // 20):
        start, end = int(0.1 * min_len), int(0.9 * min_len)
        return start, max(start + 3, end), {"method": "fallback_middle_80_percent", "threshold": threshold}
    start = max(0, int(active[0]))
    end = min(min_len, int(active[-1]) + 1)
    if end - start < 9:
        start, end = int(0.1 * min_len), int(0.9 * min_len)
    return start, end, {"method": "robust_activity_threshold", "threshold": threshold, "signal_length": min_len}


def segment_arrays(arr: np.ndarray, start: int, end: int) -> dict[str, np.ndarray]:
    full_length = arr
    active = arr[start:end]
    if len(active) < 3:
        active = arr
    n = len(active)
    i1 = max(1, n // 3)
    i2 = max(i1 + 1, 2 * n // 3)
    entry = active[:i1]
    steady = active[i1:i2]
    exit_ = active[i2:]
    return {
        "full_length": full_length,
        "entry": entry,
        "steady": steady,
        "exit": exit_,
        "entry_steady": np.concatenate([entry, steady]),
        "entry_exit": np.concatenate([entry, exit_]),
        "steady_exit": np.concatenate([steady, exit_]),
        "entry_steady_exit": np.concatenate([entry, steady, exit_]),
    }


def compute_features(arr: np.ndarray) -> dict[str, float]:
    x = np.asarray(arr, dtype=np.float64)
    if x.size == 0:
        x = np.asarray([0.0], dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0)
    spectral_sum = float(spectrum.sum())
    raw = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
        "peak_to_peak": float(np.ptp(x)),
        "kurtosis": float(kurtosis(x, fisher=True, bias=False)) if len(x) > 3 else 0.0,
        "skewness": float(skew(x, bias=False)) if len(x) > 2 else 0.0,
        "spectral_centroid": float(np.sum(freqs * spectrum) / spectral_sum) if spectral_sum > 0 else 0.0,
        "band_energy": float(np.mean(np.square(x))),
    }
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in raw.items()}


def top3_mean(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    return float(vals.head(3).mean()) if not vals.empty else float("nan")


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan"), float("nan")
    try:
        if method == "pearson":
            r, p = pearsonr(x, y)
        else:
            r, p = spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        return float("nan"), float("nan")


def safe_mi(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan")
    try:
        return float(mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0])
    except Exception:
        return float("nan")


def build_execution_dir(root: Path, timestamp: str) -> Path:
    return root / "H1" / "S1" / f"{timestamp}_{TOPIC}"


def make_dirs(output_dir: Path) -> None:
    for name in ["configs", "data", "preprocessing", "metrics", "analysis", "figures", "logs", "reports"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def load_data(process_path: Path, signal_path: Path) -> tuple[pd.DataFrame, list[str]]:
    process = pd.read_csv(process_path)
    signal = pd.read_csv(signal_path)
    sensors = infer_sensor_columns(signal)
    process = process.loc[process["enable"].astype(bool) & process["case"].isin(CASE_SCOPE)].copy()
    signal = signal.loc[signal["enable"].astype(bool) & signal["case"].isin(CASE_SCOPE)].copy()
    data = process.merge(signal, on=["case", "run"], suffixes=("", "_signal"), validate="one_to_one")
    data = data.sort_values(["case", "run"]).reset_index(drop=True)
    data["case_id"] = data["case"].astype(int)
    data["pair_id"] = data["case_id"].map(PAIR_BY_CASE)
    data["dataset_run_id"] = data.apply(lambda r: f"case_{int(r.case):02d}_run_{int(r.run):03d}", axis=1)
    return data, sensors


def build_feature_table(data: pd.DataFrame, sensors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        sensor_arrays = {sensor: parse_signal_cell(row[sensor]) for sensor in sensors}
        start, end, info = infer_cut_bounds(sensor_arrays)
        for sensor, arr in sensor_arrays.items():
            segments = segment_arrays(arr, start, end)
            for segment_setting in SEGMENT_SETTINGS:
                vector = segments[segment_setting]
                segment_rows.append(
                    {
                        "case_id": int(row["case_id"]),
                        "dataset_run_id": row["dataset_run_id"],
                        "sensor_name": sensor,
                        "segment_setting": segment_setting,
                        "segment_length": int(len(vector)),
                        "cut_start": int(start),
                        "cut_end": int(end),
                        "construction_method": info["method"],
                    }
                )
                features = compute_features(vector)
                for feature_name, feature_value in features.items():
                    rows.append(
                        {
                            "case_id": int(row["case_id"]),
                            "pair_id": row["pair_id"],
                            "dataset_run_id": row["dataset_run_id"],
                            "run": int(row["run"]),
                            "VB": float(row["VB"]) if pd.notna(row["VB"]) else np.nan,
                            "sensor_name": sensor,
                            "segment_setting": segment_setting,
                            "feature_name": feature_name,
                            "feature_value": feature_value,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(segment_rows)


def feature_level_association(feature_table: pd.DataFrame, experiment_id: str, skipped: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = feature_table.groupby(["case_id", "pair_id", "sensor_name", "segment_setting", "feature_name"], dropna=False)
    for (case_id, pair_id, sensor, segment, feature), group in grouped:
        g = group[["feature_value", "VB"]].replace([np.inf, -np.inf], np.nan).dropna()
        x = g["feature_value"].to_numpy(dtype=np.float64)
        y = g["VB"].to_numpy(dtype=np.float64)
        n = len(g)
        if n < 3:
            skipped.append({"case_id": int(case_id), "sensor_name": sensor, "segment_setting": segment, "feature_name": feature, "reason": "n_samples_lt_3"})
        pearson_r, pearson_p = safe_corr(x, y, "pearson")
        spearman_r, spearman_p = safe_corr(x, y, "spearman")
        mi = safe_mi(x, y)
        rows.append(
            {
                "experiment_id": experiment_id,
                "case_id": int(case_id),
                "pair_id": pair_id,
                "dataset_run_count": int(n),
                "sensor_name": sensor,
                "segment_setting": segment,
                "feature_name": feature,
                "pearson_r": pearson_r,
                "pearson_abs": abs(pearson_r) if np.isfinite(pearson_r) else np.nan,
                "pearson_p_value": pearson_p,
                "spearman_r": spearman_r,
                "spearman_abs": abs(spearman_r) if np.isfinite(spearman_r) else np.nan,
                "spearman_p_value": spearman_p,
                "mutual_information": mi,
                "n_samples": int(n),
                "deferred_distance_correlation": "deferred",
                "deferred_kendall_tau": "deferred",
            }
        )
    return pd.DataFrame(rows)


def segment_level_summary(feature_level: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    agg = (
        feature_level.groupby(["experiment_id", "case_id", "pair_id", "sensor_name", "segment_setting"], as_index=False)
        .agg(
            mean_abs_pearson=("pearson_abs", "mean"),
            median_abs_pearson=("pearson_abs", "median"),
            max_abs_pearson=("pearson_abs", "max"),
            top3_mean_abs_pearson=("pearson_abs", top3_mean),
            mean_abs_spearman=("spearman_abs", "mean"),
            median_abs_spearman=("spearman_abs", "median"),
            max_abs_spearman=("spearman_abs", "max"),
            top3_mean_abs_spearman=("spearman_abs", top3_mean),
            mean_mutual_information=("mutual_information", "mean"),
            median_mutual_information=("mutual_information", "median"),
            max_mutual_information=("mutual_information", "max"),
            top3_mean_mutual_information=("mutual_information", top3_mean),
        )
        .reset_index(drop=True)
    )
    for metric, rank_col in [
        ("top3_mean_abs_pearson", "rank_by_pearson"),
        ("top3_mean_abs_spearman", "rank_by_spearman"),
        ("top3_mean_mutual_information", "rank_by_mi"),
    ]:
        agg[rank_col] = agg.groupby(["case_id", "sensor_name"])[metric].rank(ascending=False, method="min")
    return agg


def baseline_comparison(segment_level: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (case_id, pair_id, sensor), group in segment_level.groupby(["case_id", "pair_id", "sensor_name"]):
        by_seg = group.set_index("segment_setting")
        for segment in SEGMENT_SETTINGS:
            if segment not in by_seg.index:
                continue
            for baseline in ["full_length", "steady"]:
                if baseline not in by_seg.index:
                    continue
                row = by_seg.loc[segment]
                base = by_seg.loc[baseline]
                dp = float(row["top3_mean_abs_pearson"] - base["top3_mean_abs_pearson"])
                ds = float(row["top3_mean_abs_spearman"] - base["top3_mean_abs_spearman"])
                dm = float(row["top3_mean_mutual_information"] - base["top3_mean_mutual_information"])
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "case_id": int(case_id),
                        "pair_id": pair_id,
                        "sensor_name": sensor,
                        "segment_setting": segment,
                        "baseline": baseline,
                        "delta_top3_mean_abs_pearson": dp,
                        "delta_top3_mean_abs_spearman": ds,
                        "delta_top3_mean_mutual_information": dm,
                        "improved_vs_baseline_pearson": bool(dp > 0),
                        "improved_vs_baseline_spearman": bool(ds > 0),
                        "improved_vs_baseline_mi": bool(dm > 0),
                    }
                )
    return pd.DataFrame(rows)


def entry_steady_exit_focus(segment_level: pd.DataFrame, baseline: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ese = segment_level[segment_level["segment_setting"].eq("entry_steady_exit")]
    for _, row in ese.iterrows():
        b = baseline[
            (baseline["case_id"].eq(row["case_id"]))
            & (baseline["sensor_name"].eq(row["sensor_name"]))
            & (baseline["segment_setting"].eq("entry_steady_exit"))
        ]
        full = b[b["baseline"].eq("full_length")]
        steady = b[b["baseline"].eq("steady")]
        ds_full = float(full["delta_top3_mean_abs_spearman"].iloc[0]) if not full.empty else np.nan
        ds_steady = float(steady["delta_top3_mean_abs_spearman"].iloc[0]) if not steady.empty else np.nan
        rows.append(
            {
                "experiment_id": experiment_id,
                "case_id": int(row["case_id"]),
                "pair_id": row["pair_id"],
                "sensor_name": row["sensor_name"],
                "entry_steady_exit_top3_mean_abs_pearson": row["top3_mean_abs_pearson"],
                "entry_steady_exit_top3_mean_abs_spearman": row["top3_mean_abs_spearman"],
                "entry_steady_exit_top3_mean_mutual_information": row["top3_mean_mutual_information"],
                "entry_steady_exit_rank_by_pearson": row["rank_by_pearson"],
                "entry_steady_exit_rank_by_spearman": row["rank_by_spearman"],
                "entry_steady_exit_rank_by_mi": row["rank_by_mi"],
                "entry_steady_exit_delta_vs_full_length_spearman": ds_full,
                "entry_steady_exit_delta_vs_steady_spearman": ds_steady,
                "entry_steady_exit_better_than_full_length": bool(ds_full > 0) if np.isfinite(ds_full) else False,
                "entry_steady_exit_better_than_steady": bool(ds_steady > 0) if np.isfinite(ds_steady) else False,
                "interpretation": "improved over both baselines" if ds_full > 0 and ds_steady > 0 else "not improved over both baselines",
            }
        )
    return pd.DataFrame(rows)


def sensor_consistency(segment_level: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (case_id, pair_id, sensor), group in segment_level.groupby(["case_id", "pair_id", "sensor_name"]):
        by_seg = group.set_index("segment_setting")
        row = {
            "experiment_id": experiment_id,
            "case_id": int(case_id),
            "pair_id": pair_id,
            "sensor_name": sensor,
            "best_segment_by_pearson": group.sort_values("rank_by_pearson").iloc[0]["segment_setting"],
            "best_segment_by_spearman": group.sort_values("rank_by_spearman").iloc[0]["segment_setting"],
            "best_segment_by_mi": group.sort_values("rank_by_mi").iloc[0]["segment_setting"],
            "entry_steady_exit_rank_by_pearson": by_seg.loc["entry_steady_exit", "rank_by_pearson"] if "entry_steady_exit" in by_seg.index else np.nan,
            "entry_steady_exit_rank_by_spearman": by_seg.loc["entry_steady_exit", "rank_by_spearman"] if "entry_steady_exit" in by_seg.index else np.nan,
            "entry_steady_exit_rank_by_mi": by_seg.loc["entry_steady_exit", "rank_by_mi"] if "entry_steady_exit" in by_seg.index else np.nan,
        }
        for seg in SEGMENT_SETTINGS:
            row[f"average_rank_{seg}"] = float(by_seg.loc[seg, ["rank_by_pearson", "rank_by_spearman", "rank_by_mi"]].mean()) if seg in by_seg.index else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pair_level_summary(segment_level: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    pair = (
        segment_level.groupby(["experiment_id", "pair_id", "sensor_name", "segment_setting"], as_index=False)
        .agg(
            mean_top3_abs_pearson=("top3_mean_abs_pearson", "mean"),
            mean_top3_abs_spearman=("top3_mean_abs_spearman", "mean"),
            mean_top3_mutual_information=("top3_mean_mutual_information", "mean"),
        )
        .reset_index(drop=True)
    )
    pair["cases"] = pair["pair_id"].map(lambda p: ",".join(str(c) for c in PAIR_DEFINITION[p]))
    for metric, rank_col in [
        ("mean_top3_abs_spearman", "segment_rank_by_spearman"),
        ("mean_top3_abs_pearson", "segment_rank_by_pearson"),
        ("mean_top3_mutual_information", "segment_rank_by_mi"),
    ]:
        pair[rank_col] = pair.groupby(["pair_id", "sensor_name"])[metric].rank(ascending=False, method="min")
    cols = ["experiment_id", "pair_id", "cases", "sensor_name", "segment_setting", "mean_top3_abs_pearson", "mean_top3_abs_spearman", "mean_top3_mutual_information", "segment_rank_by_spearman", "segment_rank_by_pearson", "segment_rank_by_mi"]
    return pair[cols]


def segment_ranking(segment_level: pd.DataFrame, baseline: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in segment_level.iterrows():
        b = baseline[
            (baseline["case_id"].eq(row["case_id"]))
            & (baseline["sensor_name"].eq(row["sensor_name"]))
            & (baseline["segment_setting"].eq(row["segment_setting"]))
        ]
        rows.append(
            {
                "experiment_id": experiment_id,
                "aggregation_level": "case_sensor",
                "case_id": int(row["case_id"]),
                "pair_id": row["pair_id"],
                "sensor_name": row["sensor_name"],
                "segment_setting": row["segment_setting"],
                "rank_by_pearson": row["rank_by_pearson"],
                "rank_by_spearman": row["rank_by_spearman"],
                "rank_by_mi": row["rank_by_mi"],
                "mean_rank": float(np.nanmean([row["rank_by_pearson"], row["rank_by_spearman"], row["rank_by_mi"]])),
                "is_better_than_full_length": bool(b.loc[b["baseline"].eq("full_length"), "improved_vs_baseline_spearman"].any()),
                "is_better_than_steady": bool(b.loc[b["baseline"].eq("steady"), "improved_vs_baseline_spearman"].any()),
            }
        )
    return pd.DataFrame(rows)


def create_figures(output_dir: Path, segment_level: pd.DataFrame, baseline: pd.DataFrame, consistency: pd.DataFrame) -> None:
    figdir = output_dir / "figures"
    avg = segment_level.groupby(["sensor_name", "segment_setting"], as_index=False)[["top3_mean_abs_spearman", "top3_mean_abs_pearson", "top3_mean_mutual_information"]].mean()
    for metric, filename, ylabel in [
        ("top3_mean_abs_spearman", "H1_S1_segment_association_by_sensor_spearman.png", "Top3 abs Spearman"),
        ("top3_mean_abs_pearson", "H1_S1_segment_association_by_sensor_pearson.png", "Top3 abs Pearson"),
        ("top3_mean_mutual_information", "H1_S1_segment_association_by_sensor_mi.png", "Top3 MI"),
    ]:
        plt.figure(figsize=(13, 5))
        for sensor, g in avg.groupby("sensor_name"):
            g = g.set_index("segment_setting").reindex(SEGMENT_SETTINGS).reset_index()
            plt.plot(g["segment_setting"], g[metric], marker="o", label=sensor)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel(ylabel)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figdir / filename, dpi=180)
        plt.close()

    plt.figure(figsize=(12, 5))
    bavg = baseline.groupby(["segment_setting", "baseline"], as_index=False)["delta_top3_mean_abs_spearman"].mean()
    width = 0.35
    x = np.arange(len(SEGMENT_SETTINGS))
    for i, base in enumerate(["full_length", "steady"]):
        vals = bavg[bavg["baseline"].eq(base)].set_index("segment_setting").reindex(SEGMENT_SETTINGS)["delta_top3_mean_abs_spearman"].fillna(0)
        plt.bar(x + (i - 0.5) * width, vals, width=width, label=f"vs {base}")
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, SEGMENT_SETTINGS, rotation=35, ha="right")
    plt.ylabel("Delta top3 abs Spearman")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "H1_S1_baseline_comparison_spearman.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4))
    counts = consistency["best_segment_by_spearman"].value_counts().reindex(SEGMENT_SETTINGS).fillna(0)
    plt.bar(counts.index, counts.values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Best segment count by sensor/case")
    plt.tight_layout()
    plt.savefig(figdir / "H1_S1_sensor_best_segment_summary.png", dpi=180)
    plt.close()

    heat = segment_level.groupby(["pair_id", "segment_setting"], as_index=False)["top3_mean_abs_spearman"].mean().pivot(index="pair_id", columns="segment_setting", values="top3_mean_abs_spearman").reindex(columns=SEGMENT_SETTINGS)
    plt.figure(figsize=(11, 4))
    plt.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    plt.colorbar(label="Top3 abs Spearman")
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figdir / "H1_S1_case_pair_heatmap.png", dpi=180)
    plt.close()

    comp_targets = ["full_length", "steady", "entry_exit", "entry_steady", "steady_exit"]
    comp_rows = []
    by_group = segment_level.set_index(["case_id", "sensor_name", "segment_setting"])
    for (case_id, sensor), group in segment_level.groupby(["case_id", "sensor_name"]):
        if (case_id, sensor, "entry_steady_exit") not in by_group.index:
            continue
        ese_val = by_group.loc[(case_id, sensor, "entry_steady_exit"), "top3_mean_abs_spearman"]
        for target in comp_targets:
            if (case_id, sensor, target) in by_group.index:
                comp_rows.append({"comparison": f"entry_steady_exit - {target}", "delta": ese_val - by_group.loc[(case_id, sensor, target), "top3_mean_abs_spearman"]})
    comp_df = pd.DataFrame(comp_rows)
    plt.figure(figsize=(10, 4.5))
    comp_mean = comp_df.groupby("comparison")["delta"].mean().sort_values(ascending=False)
    plt.bar(comp_mean.index, comp_mean.values)
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean delta top3 abs Spearman")
    plt.tight_layout()
    plt.savefig(figdir / "H1_S1_entry_steady_exit_comparison.png", dpi=180)
    plt.close()


def write_report(output_dir: Path, context: dict[str, Any]) -> None:
    report = f"""# H1_S1 feture_VB_segment_association_eda with entry_steady_exit

## 실험 목적

절삭 구간별 sensor feature와 VB 사이 association을 비교했다. 모델 학습은 수행하지 않았다.

## 설정

- Experiment ID: `{context['experiment_id']}`
- Data files: `{context['process_info_path']}`, `{context['signal_data_path']}`
- Cases: `{CASE_SCOPE}`
- full_length에 no-load 포함: `true`
- entry_steady_exit: `entry + steady + exit` (no-load 제외)
- Segment settings: `{', '.join(SEGMENT_SETTINGS)}`
- Sensors: `{', '.join(context['sensors'])}`
- Features: `{', '.join(FEATURE_NAMES)}`
- Association measures: Pearson, Spearman, Mutual information
- Deferred: distance correlation, Kendall tau
- 본 실험에서 run은 실험 실행 run이 아니라 데이터셋 내부의 `dataset_run_id`를 의미한다.

## Main Findings

- 계산 condition 수: `{context['condition_count']}`
- skipped condition 수: `{context['skipped_count']}`
- Full-length보다 Spearman association이 높은 segment: `{', '.join(context['better_than_full_length']) or 'none'}`
- Steady보다 Spearman association이 높은 segment: `{', '.join(context['better_than_steady']) or 'none'}`
- entry_steady_exit 평균 Spearman rank: `{context['entry_steady_exit_mean_rank_spearman']:.3f}`
- entry_steady_exit가 full_length보다 개선된 비율: `{context['entry_steady_exit_better_full_ratio']:.3f}`
- entry_steady_exit가 steady보다 개선된 비율: `{context['entry_steady_exit_better_steady_ratio']:.3f}`

## Sensor별 Best Segment

{context['sensor_best_markdown']}

## Pair/Domain 요약

{context['pair_summary_markdown']}

## Research Questions

RQ1. `{context['rq1']}`

RQ2. `{context['rq2']}`

RQ3. `{context['rq3']}`

RQ4. sensor별 best segment는 위 표를 참조한다. `{context['rq4']}`

RQ5. `{context['rq5']}`

## 한계

두 processed 파일만 사용한다는 제약 때문에 no-load/entry/steady/exit 경계는 신호 activity 기반 휴리스틱으로 추정했다. Association은 예측 성능 또는 인과성을 보장하지 않는다.

## 추가 실험 제안

entry_steady_exit, entry_exit, steady_exit 후보를 H2/H3 예측 실험의 명시적 segment grid에 포함하고 domain shift 성능과 함께 검증한다.
"""
    (output_dir / "reports" / "H1_S1_report.md").write_text(report, encoding="utf-8")
    (output_dir / "reports" / "H1_S1_report.html").write_text(html.escape(report).replace("\n", "<br>\n"), encoding="utf-8")


def run(output_root: Path, dry_run: bool = False) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    experiment_id = f"{timestamp}_H1_S1_{TOPIC}"
    output_dir = build_execution_dir(output_root, timestamp)
    if output_dir.exists():
        raise FileExistsError(f"Execution directory already exists: {output_dir}")
    make_dirs(output_dir)
    process_path = Path("datasets/processed/mill_process_info_enabled.csv")
    signal_path = Path("datasets/processed/mill_signal_data_enabled.csv")
    raw_config = {
        "experiment": {"hypothesis_id": "H1", "scenario_id": "S1", "experiment_id": experiment_id, "experiment_topic": TOPIC},
        "data": {"process_info_path": str(process_path), "signal_data_path": str(signal_path), "selected_cases": CASE_SCOPE, "target_column": "VB"},
        "segments": {"used": SEGMENT_SETTINGS, "excluded": EXCLUDED_SEGMENTS, "entry_steady_exit_added": True, "full_length_includes_no_load": True},
        "features": FEATURE_NAMES,
        "association_measures": ["pearson", "spearman", "mutual_information"],
        "deferred_association_measures": ["distance_correlation", "kendall_tau"],
    }
    write_yaml(output_dir / "configs" / "H1_S1_input_config.yaml", raw_config)
    run_log = output_dir / "logs" / "H1_S1_run.log"
    error_log = output_dir / "logs" / "H1_S1_error.log"
    run_log.write_text(f"[{datetime.now().isoformat(timespec='seconds')}] started {experiment_id}\n", encoding="utf-8")
    error_log.write_text("", encoding="utf-8")
    try:
        data, sensors = load_data(process_path, signal_path)
        feature_table, segment_summary = build_feature_table(data, sensors)
        validation = {
            "data_files_exist": process_path.exists() and signal_path.exists(),
            "case_filtering_ok": sorted(data["case_id"].unique().tolist()) == CASE_SCOPE,
            "no_load_segment_labels_excluded": not any("no_load" in s for s in SEGMENT_SETTINGS),
            "full_length_includes_no_load": True,
            "entry_steady_exit_included": "entry_steady_exit" in SEGMENT_SETTINGS,
            "sensor_list": sensors,
            "segment_settings": SEGMENT_SETTINGS,
            "target_column_exists": "VB" in data.columns,
            "dataset_run_id_exists": "dataset_run_id" in data.columns,
            "vb_alignment_rows": int(feature_table[["dataset_run_id", "VB"]].drop_duplicates()["VB"].notna().sum()),
            "ok": True,
        }
        if not validation["data_files_exist"] or not validation["target_column_exists"] or not sensors:
            validation["ok"] = False
        write_json(output_dir / "configs" / "H1_S1_config_validation.json", validation)
        resolved = dict(raw_config)
        resolved["resolved_target_column"] = "VB"
        resolved["resolved_sensors"] = sensors
        resolved["execution_dir"] = str(output_dir)
        write_yaml(output_dir / "configs" / "H1_S1_resolved_config.yaml", resolved)
        write_json(output_dir / "logs" / "H1_S1_environment.json", collect_environment())
        write_json(output_dir / "logs" / "H1_S1_reproducibility.json", {"experiment_id": experiment_id, "timestamp": timestamp, "script": str(Path(__file__))})
        write_json(output_dir / "data" / "H1_S1_dataset_summary.json", {"rows": len(data), "cases": sorted(data["case_id"].unique().tolist()), "dataset_run_count": int(data["dataset_run_id"].nunique())})
        write_json(output_dir / "data" / "H1_S1_sensor_summary.json", {"sensors": sensors, "sensor_count": len(sensors)})
        write_json(output_dir / "data" / "H1_S1_case_filter_summary.json", {"selected_cases": CASE_SCOPE, "rows_by_case": data.groupby("case_id").size().to_dict()})
        write_json(output_dir / "data" / "H1_S1_target_summary.json", {"target_column": "VB", "non_null_count": int(data["VB"].notna().sum()), "by_case": data.groupby("case_id")["VB"].count().to_dict()})
        write_json(
            output_dir / "preprocessing" / "H1_S1_preprocessing_summary.json",
            {"output_type": "features", "no_load_segment_labels_excluded": True, "full_length_includes_no_load": True, "segment_boundary_method": "robust_activity_threshold_from_signal_only"},
        )
        write_json(output_dir / "preprocessing" / "H1_S1_feature_summary.json", {"feature_names": FEATURE_NAMES, "feature_count": len(FEATURE_NAMES)})
        write_json(output_dir / "preprocessing" / "H1_S1_segment_summary.json", {"segment_settings": SEGMENT_SETTINGS, "excluded_segments": EXCLUDED_SEGMENTS})
        write_json(output_dir / "preprocessing" / "H1_S1_shape_trace.json", {"data_shape": list(data.shape), "feature_long_shape": list(feature_table.shape), "segment_summary_shape": list(segment_summary.shape)})
        segment_summary.to_csv(output_dir / "preprocessing" / "H1_S1_segment_lengths.csv", index=False)
        if dry_run:
            run_log.write_text(run_log.read_text(encoding="utf-8") + "dry-run validation finished\n", encoding="utf-8")
            return {"experiment_id": experiment_id, "execution_dir": str(output_dir), "dry_run": True, "validation": validation}

        skipped: list[dict[str, Any]] = []
        feature_level = feature_level_association(feature_table, experiment_id, skipped)
        segment_level = segment_level_summary(feature_level, experiment_id)
        case_level = segment_level.copy()
        pair_assoc = pair_level_summary(segment_level, experiment_id)
        baseline = baseline_comparison(segment_level, experiment_id)
        focus = entry_steady_exit_focus(segment_level, baseline, experiment_id)
        consistency = sensor_consistency(segment_level, experiment_id)
        ranking = segment_ranking(segment_level, baseline, experiment_id)
        feature_level.to_csv(output_dir / "metrics" / "H1_S1_association_feature_level.csv", index=False)
        segment_level.to_csv(output_dir / "metrics" / "H1_S1_association_segment_level.csv", index=False)
        case_level.to_csv(output_dir / "metrics" / "H1_S1_association_case_level.csv", index=False)
        pair_assoc.to_csv(output_dir / "metrics" / "H1_S1_association_pair_level.csv", index=False)
        ranking.to_csv(output_dir / "analysis" / "H1_S1_segment_ranking.csv", index=False)
        baseline.to_csv(output_dir / "analysis" / "H1_S1_baseline_comparison.csv", index=False)
        consistency.to_csv(output_dir / "analysis" / "H1_S1_sensor_consistency.csv", index=False)
        pair_assoc.to_csv(output_dir / "analysis" / "H1_S1_pair_level_association_summary.csv", index=False)
        focus.to_csv(output_dir / "analysis" / "H1_S1_entry_steady_exit_focus_summary.csv", index=False)
        skipped_df = pd.DataFrame(skipped)
        skipped_df.to_csv(output_dir / "analysis" / "H1_S1_skipped_conditions.csv", index=False)
        create_figures(output_dir, segment_level, baseline, consistency)

        better_full = sorted(baseline.loc[baseline["baseline"].eq("full_length") & baseline["improved_vs_baseline_spearman"], "segment_setting"].unique().tolist())
        better_steady = sorted(baseline.loc[baseline["baseline"].eq("steady") & baseline["improved_vs_baseline_spearman"], "segment_setting"].unique().tolist())
        sensor_best = consistency.groupby(["sensor_name", "best_segment_by_spearman"], as_index=False).size().sort_values(["sensor_name", "size"], ascending=[True, False])
        entry_rank = float(focus["entry_steady_exit_rank_by_spearman"].mean())
        entry_full_ratio = float(focus["entry_steady_exit_better_than_full_length"].mean())
        entry_steady_ratio = float(focus["entry_steady_exit_better_than_steady"].mean())
        rq1 = "존재함" if better_full or better_steady else "확인되지 않음"
        rq2 = "entry_steady_exit가 새로운 best segment로 등장함" if (consistency["best_segment_by_spearman"] == "entry_steady_exit").any() else ("entry_steady_exit는 상위권이나 best는 아님" if entry_rank <= 3 else "entry_steady_exit 추가 후 기존 ranking 변화는 제한적임")
        top_counts = consistency["best_segment_by_spearman"].value_counts()
        rq3 = "sensor 전반에서 일관적 경향이 있음" if not top_counts.empty and top_counts.iloc[0] >= 6 else "sensor별로 유리한 segment setting이 다름"
        rq4 = "sensor별 best segment setting이 서로 다르므로 sensor-specific segment 선택이 필요하다."
        rq5 = "특정 절삭 구간 또는 조합을 고려할 필요가 있음" if better_full or better_steady else "Full-length/Steady cut 기반 feature extraction이 대체로 충분할 가능성이 있음"
        context = {
            "experiment_id": experiment_id,
            "process_info_path": process_path,
            "signal_data_path": signal_path,
            "sensors": sensors,
            "condition_count": int(len(feature_level)),
            "skipped_count": int(len(skipped_df)),
            "better_than_full_length": better_full,
            "better_than_steady": better_steady,
            "entry_steady_exit_mean_rank_spearman": entry_rank,
            "entry_steady_exit_better_full_ratio": entry_full_ratio,
            "entry_steady_exit_better_steady_ratio": entry_steady_ratio,
            "sensor_best_markdown": sensor_best.to_markdown(index=False),
            "pair_summary_markdown": pair_assoc.groupby(["pair_id", "segment_setting"], as_index=False)["mean_top3_abs_spearman"].mean().sort_values(["pair_id", "mean_top3_abs_spearman"], ascending=[True, False]).head(12).to_markdown(index=False),
            "rq1": rq1,
            "rq2": rq2,
            "rq3": rq3,
            "rq4": rq4,
            "rq5": rq5,
        }
        write_json(output_dir / "analysis" / "H1_S1_analysis_summary.json", {k: v for k, v in context.items() if not str(k).endswith("markdown")})
        write_report(output_dir, context)
        run_log.write_text(run_log.read_text(encoding="utf-8") + "full execution finished\n", encoding="utf-8")
        return {"experiment_id": experiment_id, "execution_dir": str(output_dir), "dry_run": False, "summary": context, "skipped": skipped}
    except Exception as exc:
        error_log.write_text(traceback.format_exc(), encoding="utf-8")
        run_log.write_text(run_log.read_text(encoding="utf-8") + f"failed: {exc}\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="experiments/executions")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(Path(args.output_root), dry_run=args.dry_run)
    print(json.dumps(to_builtin(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

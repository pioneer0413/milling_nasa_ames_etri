#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import html
import json
import math
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, kurtosis, pearsonr, skew, spearmanr
from sklearn.feature_selection import mutual_info_regression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework import __version__
from milling_experiment_framework.core.config import SCHEMA_VERSION, stable_hash
from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.experiments.execution_path import create_execution_dir


PREFIX = "H1_S6"
CASE_SCOPE = [1, 2, 8, 9, 12, 14]
PAIR_DEFINITION = {"A": [1, 9], "B": [2, 12], "C": [8, 14]}
PAIR_BY_CASE = {case: pair for pair, cases in PAIR_DEFINITION.items() for case in cases}
SEGMENT_SETTINGS = ["full_length", "steady", "entry", "exit", "entry_steady", "entry_exit", "steady_exit", "entry_steady_exit"]
TARGET_FEATURES = ["mean", "std", "max", "min", "peak_to_peak", "kurtosis", "skewness", "spectral_centroid", "band_energy"]
FEATURE_GROUPS = {
    "mean": "statistics",
    "std": "statistics",
    "max": "statistics",
    "min": "statistics",
    "peak_to_peak": "statistics",
    "kurtosis": "shape",
    "skewness": "shape",
    "spectral_centroid": "frequency",
    "band_energy": "frequency",
}
SENSOR_GROUPS = {
    "smcAC": "Current",
    "smcDC": "Current",
    "vib_spindle": "Vibration",
    "vib_table": "Vibration",
    "AE_spindle": "Acoustic",
    "AE_table": "Acoustic",
}
SUITABILITY_EPSILON = 1e-12


def infer_sensor_group(sensor: str) -> str:
    s = sensor.lower()
    if "ae" in s:
        return "Acoustic"
    if "vib" in s:
        return "Vibration"
    if "smc" in s or "current" in s:
        return "Current"
    return "Unknown"


def feature_group(feature: str) -> str:
    return FEATURE_GROUPS.get(feature, "unknown")


def load_dataset(process_info_path: Path, signal_data_path: Path, heuristic_sequence_path: Path) -> pd.DataFrame:
    process = pd.read_csv(process_info_path)
    signal = pd.read_csv(signal_data_path)
    heuristic = pd.read_csv(heuristic_sequence_path)
    process = process.loc[process["enable"].astype(bool) & process["case"].isin(CASE_SCOPE)].copy()
    signal = signal.loc[signal["enable"].astype(bool) & signal["case"].isin(CASE_SCOPE)].copy()
    heuristic = heuristic.loc[heuristic["case"].isin(CASE_SCOPE)].copy()
    data = process.merge(signal, on=["case", "run"], suffixes=("", "_signal"), validate="one_to_one")
    data = data.merge(heuristic[["case", "run", "idx_noload_end", "idx_start", "idx_end", "signal_length", "status"]], on=["case", "run"], how="inner", validate="one_to_one")
    data = data.sort_values(["case", "run"]).reset_index(drop=True)
    data["case_id"] = data["case"].astype(int)
    data["sample_id"] = data.apply(lambda row: f"case_{int(row['case']):02d}_run_{int(row['run']):03d}", axis=1)
    data["dataset_run_id"] = data["sample_id"]
    data["pair_id"] = data["case_id"].map(PAIR_BY_CASE)
    return data


def identify_sensors(data: pd.DataFrame) -> pd.DataFrame:
    excluded = {
        "case",
        "run",
        "VB",
        "time",
        "DOC",
        "feed",
        "material",
        "material_name",
        "enable",
        "enable_signal",
        "idx_noload_end",
        "idx_start",
        "idx_end",
        "signal_length",
        "status",
        "case_id",
        "sample_id",
        "dataset_run_id",
        "pair_id",
    }
    rows = []
    for col in data.columns:
        if col in excluded:
            continue
        non_null = data[col].dropna()
        first = non_null.iloc[0] if not non_null.empty else None
        if isinstance(first, str) and first.strip().startswith("["):
            rows.append({"sensor_name": col, "sensor_group": SENSOR_GROUPS.get(col, infer_sensor_group(col)), "source_column": col, "used": True, "reason_if_excluded": ""})
    return pd.DataFrame(rows)


def segments_for_row(arr: np.ndarray, row: pd.Series) -> dict[str, np.ndarray]:
    n = len(arr)
    idx_noload_end = int(np.clip(row["idx_noload_end"], 0, n))
    idx_start = int(np.clip(row["idx_start"], idx_noload_end, n))
    idx_end = int(np.clip(row["idx_end"], idx_start, n))
    return {
        "full_length": arr[:n],
        "entry": arr[idx_noload_end:idx_start],
        "steady": arr[idx_start:idx_end],
        "exit": arr[idx_end:n],
    }


def segment_vector(segments: dict[str, np.ndarray], segment_setting: str) -> np.ndarray:
    if segment_setting in {"full_length", "steady", "entry", "exit"}:
        vector = segments[segment_setting]
    elif segment_setting == "entry_steady":
        vector = np.concatenate([segments["entry"], segments["steady"]])
    elif segment_setting == "entry_exit":
        vector = np.concatenate([segments["entry"], segments["exit"]])
    elif segment_setting == "steady_exit":
        vector = np.concatenate([segments["steady"], segments["exit"]])
    elif segment_setting == "entry_steady_exit":
        vector = np.concatenate([segments["entry"], segments["steady"], segments["exit"]])
    else:
        raise ValueError(f"Unknown segment setting: {segment_setting}")
    if len(vector) == 0:
        return np.asarray([0.0], dtype="float64")
    return vector


def compute_features(arr: np.ndarray) -> dict[str, float]:
    spectrum = np.abs(np.fft.rfft(arr))
    freqs = np.fft.rfftfreq(len(arr), d=1.0)
    spectral_sum = spectrum.sum()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "peak_to_peak": float(np.ptp(arr)),
            "kurtosis": float(kurtosis(arr, fisher=True, bias=False)) if len(arr) > 3 else 0.0,
            "skewness": float(skew(arr, bias=False)) if len(arr) > 2 else 0.0,
            "spectral_centroid": float(np.sum(freqs * spectrum) / spectral_sum) if spectral_sum > 0 else 0.0,
            "band_energy": float(np.sum(np.square(spectrum)) / max(len(arr), 1)),
        }
    return {key: (0.0 if not np.isfinite(value) else value) for key, value in raw.items()}


def find_or_create_entry_steady_exit_features(data: pd.DataFrame, sensors: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    segment_lengths = []
    for _, row in data.iterrows():
        for sensor in sensors:
            try:
                arr = np.nan_to_num(np.asarray(ast.literal_eval(row[sensor]), dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Failed to parse sensor sequence: case={row['case']} run={row['run']} sensor={sensor}") from exc
            segments = segments_for_row(arr, row)
            for segment_setting in SEGMENT_SETTINGS:
                vector = segment_vector(segments, segment_setting)
                if segment_setting == "entry_steady_exit":
                    segment_lengths.append({"sample_id": row["sample_id"], "sensor_name": sensor, "length": int(len(vector))})
                values = compute_features(vector)
                for feature_name, feature_value in values.items():
                    rows.append(
                        {
                            "sample_id": row["sample_id"],
                            "dataset_run_id": row["dataset_run_id"],
                            "case_id": int(row["case_id"]),
                            "case": int(row["case_id"]),
                            "pair_id": row["pair_id"],
                            "run": int(row["run"]),
                            "time": row.get("time", np.nan),
                            "VB": float(row["VB"]),
                            "sensor_name": sensor,
                            "sensor_group": SENSOR_GROUPS.get(sensor, infer_sensor_group(sensor)),
                            "segment_setting": segment_setting,
                            "feature_name": feature_name,
                            "feature_group": feature_group(feature_name),
                            "feature_value": feature_value,
                        }
                    )
    summary = {
        "entry_steady_exit_source": "generated_from_raw_sensor_sequence",
        "construction": "time_axis_concatenation(entry, steady, exit); no-load excluded",
        "num_segment_length_records": len(segment_lengths),
        "min_length": int(min(x["length"] for x in segment_lengths)) if segment_lengths else None,
        "max_length": int(max(x["length"] for x in segment_lengths)) if segment_lengths else None,
    }
    return pd.DataFrame(rows), summary


def identify_feature_columns(feature_long: pd.DataFrame) -> pd.DataFrame:
    available = set(feature_long["feature_name"].unique())
    return pd.DataFrame(
        [
            {
                "feature_name": feature,
                "feature_group": feature_group(feature),
                "found": feature in available,
                "matched_columns": f"feature_name={feature}" if feature in available else "",
            }
            for feature in TARGET_FEATURES
        ]
    )


def identify_segment_settings(feature_long: pd.DataFrame) -> pd.DataFrame:
    available = set(feature_long["segment_setting"].unique())
    definitions = {
        "full_length": "whole signal, including no-load portion if present",
        "steady": "steady cutting segment",
        "entry": "tool-workpiece entry cutting segment",
        "exit": "tool-workpiece exit segment",
        "entry_steady": "entry + steady concatenated",
        "entry_exit": "entry + exit concatenated",
        "steady_exit": "steady + exit concatenated",
        "entry_steady_exit": "entry + steady + exit concatenated; no-load excluded cutting-only combination",
    }
    return pd.DataFrame(
        [
            {"segment_setting": s, "used": s in available, "source_available": s in available, "definition": definitions[s], "excluded_reason": ""}
            for s in SEGMENT_SETTINGS
        ]
    )


def compute_association_metrics(feature_long: pd.DataFrame, experiment_id: str) -> dict[str, pd.DataFrame]:
    feature_level = association_feature_level(feature_long, experiment_id)
    segment_level = association_segment_level(feature_level, experiment_id)
    baseline = baseline_comparison(segment_level, experiment_id)
    consistency = sensor_consistency(segment_level, experiment_id)
    pair_summary = pair_level_summary(segment_level, experiment_id)
    ranking = segment_ranking(segment_level, baseline, experiment_id)
    return {
        "association_feature_level": feature_level,
        "association_segment_level": segment_level,
        "association_case_level": segment_level.copy(),
        "association_pair_level": pair_summary.copy(),
        "baseline_comparison": baseline,
        "sensor_consistency": consistency,
        "pair_level_association_summary": pair_summary,
        "segment_ranking": ranking,
    }


def association_feature_level(feature_long: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    rows = []
    for (case_id, sensor, segment, feature_name), group in feature_long.groupby(["case_id", "sensor_name", "segment_setting", "feature_name"], dropna=False):
        x = group.sort_values("run")["feature_value"].to_numpy(dtype="float64")
        y = group.sort_values("run")["VB"].to_numpy(dtype="float64")
        pair_id = PAIR_BY_CASE[int(case_id)]
        row = {
            "experiment_id": experiment_id,
            "case_id": int(case_id),
            "pair_id": pair_id,
            "dataset_run_count": int(len(group)),
            "sensor_name": sensor,
            "segment_setting": segment,
            "feature_name": feature_name,
            "pearson_r": np.nan,
            "pearson_abs": np.nan,
            "pearson_p_value": np.nan,
            "spearman_r": np.nan,
            "spearman_abs": np.nan,
            "spearman_p_value": np.nan,
            "mutual_information": np.nan,
            "n_samples": int(len(group)),
            "status": "ok",
            "skip_reason": None,
        }
        finite = np.isfinite(x) & np.isfinite(y)
        x_f, y_f = x[finite], y[finite]
        if len(x_f) < 3:
            row["status"] = "skipped"
            row["skip_reason"] = "n_samples_lt_3"
        elif len(np.unique(x_f)) < 2:
            row["status"] = "skipped"
            row["skip_reason"] = "constant_feature"
            row["mutual_information"] = 0.0
        elif len(np.unique(y_f)) < 2:
            row["status"] = "skipped"
            row["skip_reason"] = "constant_target"
            row["mutual_information"] = 0.0
        else:
            pearson = pearsonr(x_f, y_f)
            spearman = spearmanr(x_f, y_f)
            mi = mutual_info_regression(x_f.reshape(-1, 1), y_f, n_neighbors=max(1, min(3, len(x_f) - 1)), random_state=0)[0]
            row.update(
                {
                    "pearson_r": float(pearson.statistic),
                    "pearson_abs": float(abs(pearson.statistic)),
                    "pearson_p_value": float(pearson.pvalue),
                    "spearman_r": float(spearman.statistic),
                    "spearman_abs": float(abs(spearman.statistic)),
                    "spearman_p_value": float(spearman.pvalue),
                    "mutual_information": float(mi),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def topk_mean(series: pd.Series, k: int = 3) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna().sort_values(ascending=False)
    return float(series.head(k).mean()) if len(series) else float("nan")


def association_segment_level(feature_level: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    rows = []
    for keys, group in feature_level.groupby(["case_id", "pair_id", "sensor_name", "segment_setting"], dropna=False):
        case_id, pair_id, sensor, segment = keys
        pearson = group["pearson_abs"].dropna()
        spearman = group["spearman_abs"].dropna()
        mi = group["mutual_information"].dropna()
        rows.append(
            {
                "experiment_id": experiment_id,
                "case_id": case_id,
                "pair_id": pair_id,
                "sensor_name": sensor,
                "segment_setting": segment,
                "mean_abs_pearson": float(pearson.mean()) if len(pearson) else np.nan,
                "median_abs_pearson": float(pearson.median()) if len(pearson) else np.nan,
                "max_abs_pearson": float(pearson.max()) if len(pearson) else np.nan,
                "top3_mean_abs_pearson": topk_mean(pearson),
                "mean_abs_spearman": float(spearman.mean()) if len(spearman) else np.nan,
                "median_abs_spearman": float(spearman.median()) if len(spearman) else np.nan,
                "max_abs_spearman": float(spearman.max()) if len(spearman) else np.nan,
                "top3_mean_abs_spearman": topk_mean(spearman),
                "mean_mutual_information": float(mi.mean()) if len(mi) else np.nan,
                "median_mutual_information": float(mi.median()) if len(mi) else np.nan,
                "max_mutual_information": float(mi.max()) if len(mi) else np.nan,
                "top3_mean_mutual_information": topk_mean(mi),
            }
        )
    frame = pd.DataFrame(rows)
    frame["rank_by_pearson"] = frame.groupby(["case_id", "sensor_name"])["top3_mean_abs_pearson"].rank(method="min", ascending=False)
    frame["rank_by_spearman"] = frame.groupby(["case_id", "sensor_name"])["top3_mean_abs_spearman"].rank(method="min", ascending=False)
    frame["rank_by_mi"] = frame.groupby(["case_id", "sensor_name"])["top3_mean_mutual_information"].rank(method="min", ascending=False)
    return frame


def baseline_comparison(segment_level: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    rows = []
    for keys, group in segment_level.groupby(["case_id", "pair_id", "sensor_name"], dropna=False):
        case_id, pair_id, sensor = keys
        for baseline in ["full_length", "steady"]:
            base = group.loc[group["segment_setting"] == baseline]
            if base.empty:
                continue
            b = base.iloc[0]
            for _, row in group.iterrows():
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "case_id": case_id,
                        "pair_id": pair_id,
                        "sensor_name": sensor,
                        "segment_setting": row["segment_setting"],
                        "baseline": baseline,
                        "delta_top3_mean_abs_pearson": row["top3_mean_abs_pearson"] - b["top3_mean_abs_pearson"],
                        "delta_top3_mean_abs_spearman": row["top3_mean_abs_spearman"] - b["top3_mean_abs_spearman"],
                        "delta_top3_mean_mutual_information": row["top3_mean_mutual_information"] - b["top3_mean_mutual_information"],
                        "improved_vs_baseline_pearson": bool(row["top3_mean_abs_pearson"] > b["top3_mean_abs_pearson"]),
                        "improved_vs_baseline_spearman": bool(row["top3_mean_abs_spearman"] > b["top3_mean_abs_spearman"]),
                        "improved_vs_baseline_mi": bool(row["top3_mean_mutual_information"] > b["top3_mean_mutual_information"]),
                    }
                )
    return pd.DataFrame(rows)


def sensor_consistency(segment_level: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    rows = []
    for keys, group in segment_level.groupby(["case_id", "pair_id", "sensor_name"], dropna=False):
        case_id, pair_id, sensor = keys
        best_p = group.loc[group["top3_mean_abs_pearson"].idxmax()]
        best_s = group.loc[group["top3_mean_abs_spearman"].idxmax()]
        best_m = group.loc[group["top3_mean_mutual_information"].idxmax()]
        row = {
            "experiment_id": experiment_id,
            "case_id": case_id,
            "pair_id": pair_id,
            "sensor_name": sensor,
            "best_segment_by_pearson": best_p["segment_setting"],
            "best_segment_by_spearman": best_s["segment_setting"],
            "best_segment_by_mi": best_m["segment_setting"],
        }
        rank_map = dict(zip(group["segment_setting"], group["rank_by_spearman"]))
        for segment in SEGMENT_SETTINGS:
            row[f"average_rank_{segment}"] = rank_map.get(segment, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def pair_level_summary(segment_level: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    rows = []
    for keys, group in segment_level.groupby(["pair_id", "sensor_name", "segment_setting"], dropna=False):
        pair_id, sensor, segment = keys
        rows.append(
            {
                "experiment_id": experiment_id,
                "pair_id": pair_id,
                "cases": ",".join(map(str, PAIR_DEFINITION[pair_id])),
                "sensor_name": sensor,
                "segment_setting": segment,
                "mean_top3_abs_pearson": group["top3_mean_abs_pearson"].mean(),
                "mean_top3_abs_spearman": group["top3_mean_abs_spearman"].mean(),
                "mean_top3_mutual_information": group["top3_mean_mutual_information"].mean(),
            }
        )
    frame = pd.DataFrame(rows)
    frame["segment_rank_by_spearman"] = frame.groupby(["pair_id", "sensor_name"])["mean_top3_abs_spearman"].rank(method="min", ascending=False)
    frame["segment_rank_by_pearson"] = frame.groupby(["pair_id", "sensor_name"])["mean_top3_abs_pearson"].rank(method="min", ascending=False)
    frame["segment_rank_by_mi"] = frame.groupby(["pair_id", "sensor_name"])["mean_top3_mutual_information"].rank(method="min", ascending=False)
    return frame


def segment_ranking(segment_level: pd.DataFrame, baseline: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    better_full = baseline.loc[baseline["baseline"] == "full_length", ["case_id", "sensor_name", "segment_setting", "improved_vs_baseline_spearman"]].rename(columns={"improved_vs_baseline_spearman": "is_better_than_full_length"})
    better_steady = baseline.loc[baseline["baseline"] == "steady", ["case_id", "sensor_name", "segment_setting", "improved_vs_baseline_spearman"]].rename(columns={"improved_vs_baseline_spearman": "is_better_than_steady"})
    ranking = segment_level[["experiment_id", "case_id", "pair_id", "sensor_name", "segment_setting", "rank_by_pearson", "rank_by_spearman", "rank_by_mi"]].copy()
    ranking["aggregation_level"] = "case_sensor"
    ranking["mean_rank"] = ranking[["rank_by_pearson", "rank_by_spearman", "rank_by_mi"]].mean(axis=1)
    ranking = ranking.merge(better_full, on=["case_id", "sensor_name", "segment_setting"], how="left").merge(better_steady, on=["case_id", "sensor_name", "segment_setting"], how="left")
    ranking["is_better_than_full_length"] = ranking["is_better_than_full_length"].fillna(False)
    ranking["is_better_than_steady"] = ranking["is_better_than_steady"].fillna(False)
    return ranking


def compute_monotonicity(x: np.ndarray) -> float:
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    diff = np.diff(x)
    return float(abs((int(np.sum(diff > 0)) - int(np.sum(diff < 0))) / (len(x) - 1)))


def compute_trendability(x: np.ndarray, t: np.ndarray) -> float:
    x = np.asarray(x, dtype="float64")
    t = np.asarray(t, dtype="float64")
    finite = np.isfinite(x) & np.isfinite(t)
    x, t = x[finite], t[finite]
    n = len(x)
    if n < 2:
        return float("nan")
    numerator = n * np.sum(x * t) - np.sum(x) * np.sum(t)
    x_term = n * np.sum(x**2) - np.sum(x) ** 2
    t_term = n * np.sum(t**2) - np.sum(t) ** 2
    denominator = math.sqrt(max(x_term * t_term, 0.0))
    return float("nan") if denominator <= 0 else float(abs(numerator / denominator))


def compute_suitability_sum(monotonicity: float, trendability: float) -> float:
    return float("nan") if np.isnan(monotonicity) or np.isnan(trendability) else float(monotonicity + trendability)


def compute_suitability_harmonic(monotonicity: float, trendability: float, epsilon: float = SUITABILITY_EPSILON) -> float:
    if np.isnan(monotonicity) or np.isnan(trendability):
        return float("nan")
    denominator = monotonicity + trendability
    return 0.0 if denominator == 0 else float(2 * monotonicity * trendability / (denominator + epsilon))


def compute_case_level_suitability(feature_long: pd.DataFrame, experiment_id: str, ordering_column: str) -> pd.DataFrame:
    rows = []
    group_cols = ["case_id", "sensor_name", "sensor_group", "segment_setting", "feature_name", "feature_group"]
    for keys, group in feature_long.groupby(group_cols, dropna=False):
        case_id, sensor, sensor_group, segment, feature, fgroup = keys
        group = group.sort_values([ordering_column, "run"], kind="mergesort")
        x_raw = group["feature_value"].to_numpy(dtype="float64")
        t_raw = group[ordering_column].to_numpy(dtype="float64")
        missing = int(np.sum(~np.isfinite(x_raw) | ~np.isfinite(t_raw)))
        finite = np.isfinite(x_raw) & np.isfinite(t_raw)
        x, t = x_raw[finite], t_raw[finite]
        status = "ok"
        if len(x) < 2:
            mono = trend = primary = legacy = harmonic = np.nan
            status = "skipped_n_lt_2"
        else:
            mono = compute_monotonicity(x)
            trend = compute_trendability(x, t)
            legacy = compute_suitability_sum(mono, trend)
            harmonic = compute_suitability_harmonic(mono, trend)
            primary = harmonic
            if np.isnan(trend):
                status = "skipped_trendability_denominator_zero"
        rows.append(
            {
                "experiment_id": experiment_id,
                "case_id": int(case_id),
                "sensor_name": sensor,
                "sensor_group": sensor_group,
                "feature_name": feature,
                "feature_group": fgroup,
                "segment_setting": segment,
                "monotonicity": mono,
                "trendability": trend,
                "suitability": primary,
                "suitability_sum_legacy": legacy,
                "suitability_harmonic_mean": harmonic,
                "primary_suitability_method": "harmonic_mean",
                "suitability_epsilon": SUITABILITY_EPSILON,
                "sample_count": int(len(x)),
                "missing_count": missing,
                "missing_rate": float(missing / len(group)) if len(group) else np.nan,
                "ordering_column": ordering_column,
                "calculation_status": status,
            }
        )
    return pd.DataFrame(rows)


def add_suitability_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["rank_by_sum_legacy"] = out["suitability_sum_legacy"].rank(ascending=False, method="first").astype(int)
    out["rank_by_harmonic_mean"] = out["suitability_harmonic_mean"].rank(ascending=False, method="first").astype(int)
    out["rank_by_suitability"] = out["rank_by_harmonic_mean"]
    return out.sort_values("rank_by_suitability").reset_index(drop=True)


def aggregate_suitability_results(case_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ok = case_results[case_results["calculation_status"].eq("ok")].copy()
    group_cols = ["experiment_id", "sensor_name", "sensor_group", "feature_name", "feature_group", "segment_setting"]
    overall = (
        ok.groupby(group_cols, dropna=False)
        .agg(
            mean_monotonicity=("monotonicity", "mean"),
            mean_trendability=("trendability", "mean"),
            mean_suitability=("suitability", "mean"),
            suitability_sum_legacy=("suitability_sum_legacy", "mean"),
            suitability_harmonic_mean=("suitability_harmonic_mean", "mean"),
            std_suitability=("suitability", "std"),
            median_suitability=("suitability", "median"),
            min_suitability=("suitability", "min"),
            max_suitability=("suitability", "max"),
            total_sample_count=("sample_count", "sum"),
            num_cases_available=("case_id", "nunique"),
            primary_suitability_method=("primary_suitability_method", "first"),
            suitability_epsilon=("suitability_epsilon", "first"),
        )
        .reset_index()
    )
    overall["std_suitability"] = overall["std_suitability"].fillna(0.0)
    overall = add_suitability_ranks(overall)

    pair_ok = ok.assign(pair_id=ok["case_id"].map(PAIR_BY_CASE), cases_in_pair=lambda x: x["pair_id"].map(lambda p: ",".join(map(str, PAIR_DEFINITION[p]))))
    pair = (
        pair_ok.groupby(["experiment_id", "pair_id", "cases_in_pair", "sensor_name", "sensor_group", "feature_name", "feature_group", "segment_setting"], dropna=False)
        .agg(
            mean_monotonicity=("monotonicity", "mean"),
            mean_trendability=("trendability", "mean"),
            mean_suitability=("suitability", "mean"),
            suitability_sum_legacy=("suitability_sum_legacy", "mean"),
            suitability_harmonic_mean=("suitability_harmonic_mean", "mean"),
            std_suitability=("suitability", "std"),
            total_sample_count=("sample_count", "sum"),
            primary_suitability_method=("primary_suitability_method", "first"),
            suitability_epsilon=("suitability_epsilon", "first"),
        )
        .reset_index()
    )
    pair["std_suitability"] = pair["std_suitability"].fillna(0.0)
    pair["rank_by_sum_legacy"] = pair.groupby("pair_id")["suitability_sum_legacy"].rank(ascending=False, method="first").astype(int)
    pair["rank_by_harmonic_mean"] = pair.groupby("pair_id")["suitability_harmonic_mean"].rank(ascending=False, method="first").astype(int)
    pair["rank_by_suitability"] = pair["rank_by_harmonic_mean"]

    fs = (
        overall.groupby(["experiment_id", "feature_name", "feature_group", "segment_setting"], dropna=False)
        .agg(
            mean_monotonicity=("mean_monotonicity", "mean"),
            mean_trendability=("mean_trendability", "mean"),
            mean_suitability=("mean_suitability", "mean"),
            suitability_sum_legacy=("suitability_sum_legacy", "mean"),
            suitability_harmonic_mean=("suitability_harmonic_mean", "mean"),
            std_suitability=("mean_suitability", "std"),
            num_sensors=("sensor_name", "nunique"),
            num_cases_available=("num_cases_available", "sum"),
            sample_count=("total_sample_count", "sum"),
            primary_suitability_method=("primary_suitability_method", "first"),
            suitability_epsilon=("suitability_epsilon", "first"),
        )
        .reset_index()
    )
    fs["std_suitability"] = fs["std_suitability"].fillna(0.0)
    fs = add_suitability_ranks(fs)
    simplified = fs.rename(columns={"feature_name": "feature", "segment_setting": "segment_combination", "mean_monotonicity": "monotonicity", "mean_trendability": "trendability", "mean_suitability": "suitability"})[
        ["feature", "segment_combination", "monotonicity", "trendability", "suitability", "suitability_sum_legacy", "suitability_harmonic_mean", "primary_suitability_method", "suitability_epsilon", "rank_by_sum_legacy", "rank_by_harmonic_mean", "rank_by_suitability", "sample_count"]
    ]
    method = overall[["experiment_id", "sensor_name", "sensor_group", "feature_name", "feature_group", "segment_setting", "mean_monotonicity", "mean_trendability", "suitability_sum_legacy", "suitability_harmonic_mean", "rank_by_sum_legacy", "rank_by_harmonic_mean"]].rename(columns={"mean_monotonicity": "monotonicity", "mean_trendability": "trendability"}).copy()
    method["rank_change"] = method["rank_by_harmonic_mean"] - method["rank_by_sum_legacy"]
    method["rank_change_abs"] = method["rank_change"].abs()
    method["score_gap_sum_minus_harmonic"] = method["suitability_sum_legacy"] - method["suitability_harmonic_mean"]
    method["interpretation"] = np.where((method["monotonicity"] >= method["monotonicity"].quantile(0.75)) & (method["trendability"] >= method["trendability"].quantile(0.75)), "balanced_high_m_and_t", "similar_or_mixed")
    return overall, pair, fs, simplified, method.sort_values("rank_by_harmonic_mean")


def top_bottom_suitability_summary(overall: pd.DataFrame, fs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for section, frame in [
        ("top_10_feature_segment_sensor_combinations", overall.sort_values("mean_suitability", ascending=False).head(10)),
        ("bottom_10_feature_segment_sensor_combinations", overall.sort_values("mean_suitability").head(10)),
        ("top_10_feature_segment_average", fs.assign(sensor_name="sensor_average", sensor_group="sensor_average").sort_values("mean_suitability", ascending=False).head(10)),
        ("bottom_10_feature_segment_average", fs.assign(sensor_name="sensor_average", sensor_group="sensor_average").sort_values("mean_suitability").head(10)),
    ]:
        for _, row in frame.iterrows():
            rows.append({"section": section, **row.to_dict()})
    return pd.DataFrame(rows)


def standardized_association(feature_level: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    agg = (
        feature_level.groupby(["sensor_name", "segment_setting", "feature_name"], dropna=False)
        .agg(
            top3_mean_abs_spearman=("spearman_abs", topk_mean),
            top3_mean_abs_pearson=("pearson_abs", topk_mean),
            top3_mean_mutual_information=("mutual_information", topk_mean),
            mean_abs_spearman=("spearman_abs", "mean"),
            mean_abs_pearson=("pearson_abs", "mean"),
            mean_mutual_information=("mutual_information", "mean"),
            sample_count=("n_samples", "sum"),
        )
        .reset_index()
    )
    agg["experiment_id"] = experiment_id
    agg["sensor_group"] = agg["sensor_name"].map(lambda s: SENSOR_GROUPS.get(s, infer_sensor_group(s)))
    agg["feature_group"] = agg["feature_name"].map(feature_group)
    agg["primary_association_score"] = agg["top3_mean_abs_spearman"]
    return agg


def join_association_and_suitability(association: pd.DataFrame, suitability: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    suit = suitability.rename(columns={"mean_monotonicity": "monotonicity", "mean_trendability": "trendability", "total_sample_count": "suitability_sample_count"}).copy()
    suit["primary_suitability_score"] = suit["suitability_harmonic_mean"]
    joined = association.merge(suit, on=["sensor_name", "sensor_group", "feature_name", "feature_group", "segment_setting"], how="inner", suffixes=("_association", "_suitability"))
    joined["experiment_id"] = experiment_id
    joined["association_rank"] = joined["primary_association_score"].rank(ascending=False, method="average")
    joined["suitability_rank"] = joined["primary_suitability_score"].rank(ascending=False, method="average")
    joined["rank_difference"] = joined["association_rank"] - joined["suitability_rank"]
    joined["rank_difference_abs"] = joined["rank_difference"].abs()
    joined = compute_alignment_categories(joined)
    joined = compute_combined_robust_score(joined)
    joined["interpretation"] = joined["alignment_category"].map(
        {
            "high_association_high_suitability": "target-relevant and degradation-aware robust candidate",
            "high_association_low_suitability": "VB-related but weak progression trend",
            "low_association_high_suitability": "progression-like trend but weak VB relation",
            "low_association_low_suitability": "low priority under both EDA views",
            "middle_or_mixed": "mixed or middle-ranked evidence",
        }
    )
    return joined.sort_values("combined_score", ascending=False).reset_index(drop=True)


def compute_alignment_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ah, al = out["primary_association_score"].quantile(0.75), out["primary_association_score"].quantile(0.25)
    sh, sl = out["primary_suitability_score"].quantile(0.75), out["primary_suitability_score"].quantile(0.25)
    out["alignment_category"] = "middle_or_mixed"
    out.loc[(out["primary_association_score"] >= ah) & (out["primary_suitability_score"] >= sh), "alignment_category"] = "high_association_high_suitability"
    out.loc[(out["primary_association_score"] >= ah) & (out["primary_suitability_score"] <= sl), "alignment_category"] = "high_association_low_suitability"
    out.loc[(out["primary_association_score"] <= al) & (out["primary_suitability_score"] >= sh), "alignment_category"] = "low_association_high_suitability"
    out.loc[(out["primary_association_score"] <= al) & (out["primary_suitability_score"] <= sl), "alignment_category"] = "low_association_low_suitability"
    return out


def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def compute_combined_robust_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["normalized_association_score"] = minmax(out["primary_association_score"])
    out["normalized_suitability_score"] = minmax(out["primary_suitability_score"])
    denom = out["normalized_association_score"] + out["normalized_suitability_score"]
    out["combined_score"] = np.where(denom > 0, 2 * out["normalized_association_score"] * out["normalized_suitability_score"] / (denom + 1e-12), 0.0)
    return out


def correlation_summary(joined: pd.DataFrame) -> pd.DataFrame:
    levels = {
        "sensor_feature_segment": joined,
        "feature_segment_mean": joined.groupby(["feature_name", "segment_setting"], as_index=False)[["primary_association_score", "primary_suitability_score"]].mean(),
        "sensor_group_segment_mean": joined.groupby(["sensor_group", "segment_setting"], as_index=False)[["primary_association_score", "primary_suitability_score"]].mean(),
    }
    rows = []
    for level, frame in levels.items():
        valid = frame[["primary_association_score", "primary_suitability_score"]].dropna()
        rows.append(
            {
                "aggregation_level": level,
                "association_score": "top3_mean_abs_spearman",
                "suitability_score": "suitability_harmonic_mean",
                "pearson_corr": float(pearsonr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic) if len(valid) > 2 else np.nan,
                "spearman_corr": float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic) if len(valid) > 2 else np.nan,
                "kendall_tau": float(kendalltau(valid.iloc[:, 0], valid.iloc[:, 1]).statistic) if len(valid) > 2 else np.nan,
                "n_combinations": int(len(valid)),
                "interpretation": "aligned rankings" if len(valid) > 2 and abs(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic) >= 0.5 else "weak or mixed alignment",
            }
        )
    return pd.DataFrame(rows)


def alignment_category_summary(joined: pd.DataFrame) -> pd.DataFrame:
    total = len(joined)
    rows = []
    for category, group in joined.groupby("alignment_category"):
        reps = "; ".join(f"{r.sensor_name}/{r.feature_name}/{r.segment_setting}" for r in group.sort_values("combined_score", ascending=False).head(5).itertuples())
        rows.append({"alignment_category": category, "num_combinations": int(len(group)), "ratio": float(len(group) / total) if total else 0.0, "representative_feature_segment_sensor_combinations": reps})
    return pd.DataFrame(rows).sort_values("num_combinations", ascending=False)


def robust_candidates(joined: pd.DataFrame) -> pd.DataFrame:
    data = joined[joined["alignment_category"].eq("high_association_high_suitability")].copy()
    if data.empty:
        data = joined.head(30).copy()
    data = data.sort_values("combined_score", ascending=False).head(40)
    data["reason_selected"] = np.where(data["alignment_category"].eq("high_association_high_suitability"), "top quartile in both association and harmonic suitability", "selected by combined robust score")
    data["manufacturing_interpretation"] = data.apply(lambda r: f"{r.sensor_group} sensor, {r.segment_setting} segment, {r.feature_name} feature selected by association+suitability evidence", axis=1)
    return data[["sensor_name", "sensor_group", "feature_name", "feature_group", "segment_setting", "primary_association_score", "primary_suitability_score", "combined_score", "association_rank", "suitability_rank", "reason_selected", "manufacturing_interpretation"]]


def disagreement_table(joined: pd.DataFrame) -> pd.DataFrame:
    data = joined[joined["alignment_category"].isin(["high_association_low_suitability", "low_association_high_suitability"])].copy()
    data["disagreement_type"] = data["alignment_category"].map({"high_association_low_suitability": "association_high_suitability_low", "low_association_high_suitability": "association_low_suitability_high"})
    data["possible_reason"] = data["disagreement_type"].map({"association_high_suitability_low": "non-monotonic or condition-specific VB relation", "association_low_suitability_high": "time/order proxy or weak target relevance"})
    data["recommendation"] = data["disagreement_type"].map({"association_high_suitability_low": "validate in prediction and domain shift", "association_low_suitability_high": "use cautiously; inspect target relevance"})
    return data[["sensor_name", "sensor_group", "feature_name", "feature_group", "segment_setting", "primary_association_score", "primary_suitability_score", "association_rank", "suitability_rank", "disagreement_type", "possible_reason", "recommendation"]].sort_values("association_rank")


def compare_entry_steady_exit_with_other_segments(joined: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "primary_association_score": "association",
        "top3_mean_abs_pearson": "association_pearson",
        "top3_mean_mutual_information": "association_mi",
        "primary_suitability_score": "suitability_harmonic",
        "suitability_sum_legacy": "suitability_sum_legacy",
        "combined_score": "combined_score",
    }
    rows = []
    keys = ["sensor_name", "sensor_group", "feature_name", "feature_group"]
    for key_vals, group in joined.groupby(keys, dropna=False):
        target = group[group["segment_setting"].eq("entry_steady_exit")]
        if target.empty:
            continue
        target = target.iloc[0]
        for _, comp in group[group["segment_setting"].ne("entry_steady_exit")].iterrows():
            for col, metric_name in metrics.items():
                delta = target[col] - comp[col]
                rows.append(
                    {
                        "comparison_target": comp["segment_setting"],
                        "sensor_name": target["sensor_name"],
                        "sensor_group": target["sensor_group"],
                        "feature_name": target["feature_name"],
                        "feature_group": target["feature_group"],
                        "metric_name": metric_name,
                        "entry_steady_exit_value": target[col],
                        "comparison_value": comp[col],
                        "delta": delta,
                        "improved": bool(delta > 0),
                        "interpretation": "entry_steady_exit higher" if delta > 0 else "comparison segment higher or tied",
                    }
                )
    return pd.DataFrame(rows)


def save_heatmap(df: pd.DataFrame, index: str, columns: str, values: str, path: Path, title: str, cmap: str = "viridis", annot: bool = True) -> None:
    pivot = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    if index == "feature_name":
        pivot = pivot.reindex(index=TARGET_FEATURES)
    pivot = pivot.reindex(columns=SEGMENT_SETTINGS)
    plt.figure(figsize=(12, max(4, len(pivot.index) * 0.45)))
    sns.heatmap(pivot, cmap=cmap, annot=annot, fmt=".2f", linewidths=0.3)
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def create_figures(output_dir: Path, assoc_std: pd.DataFrame, segment_level: pd.DataFrame, pair_assoc: pd.DataFrame, overall: pd.DataFrame, pair_suit: pd.DataFrame, fs: pd.DataFrame, method: pd.DataFrame, joined: pd.DataFrame, ese_comp: pd.DataFrame) -> None:
    figdir = output_dir / "figures"
    save_heatmap(assoc_std, "feature_name", "segment_setting", "primary_association_score", figdir / f"{PREFIX}_feature_segment_association_heatmap.png", "Association: feature x segment")
    save_heatmap(assoc_std, "sensor_name", "segment_setting", "primary_association_score", figdir / f"{PREFIX}_sensor_segment_association_heatmap.png", "Association: sensor x segment")
    save_heatmap(assoc_std, "sensor_group", "segment_setting", "primary_association_score", figdir / f"{PREFIX}_sensor_group_segment_association_heatmap.png", "Association: sensor group x segment")
    pair_a = pair_assoc.rename(columns={"mean_top3_abs_spearman": "primary_association_score"})
    save_heatmap(pair_a, "pair_id", "segment_setting", "primary_association_score", figdir / f"{PREFIX}_pair_segment_association_heatmap.png", "Association: pair x segment")

    fig, ax = plt.subplots(figsize=(8, 4))
    segment_level.groupby("segment_setting")["rank_by_spearman"].apply(lambda s: (s == 1).sum()).reindex(SEGMENT_SETTINGS).plot(kind="bar", ax=ax)
    ax.set_title("Best segment frequency by association")
    ax.set_ylabel("rank_by_spearman == 1 count")
    plt.tight_layout()
    plt.savefig(figdir / f"{PREFIX}_best_segment_frequency_by_association.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=assoc_std, x="top3_mean_abs_spearman", y="top3_mean_abs_pearson", hue="segment_setting", ax=axes[0], s=35)
    sns.scatterplot(data=assoc_std, x="top3_mean_abs_spearman", y="top3_mean_mutual_information", hue="segment_setting", ax=axes[1], s=35, legend=False)
    axes[0].set_title("Spearman vs Pearson")
    axes[1].set_title("Spearman vs MI")
    plt.tight_layout()
    plt.savefig(figdir / f"{PREFIX}_association_metric_agreement.png", dpi=180)
    plt.close()

    save_heatmap(fs, "feature_name", "segment_setting", "suitability_harmonic_mean", figdir / f"{PREFIX}_feature_segment_suitability_heatmap_harmonic_mean.png", "Suitability harmonic: feature x segment")
    save_heatmap(fs, "feature_name", "segment_setting", "suitability_sum_legacy", figdir / f"{PREFIX}_feature_segment_suitability_heatmap_sum_legacy.png", "Suitability sum legacy: feature x segment", cmap="mako")
    save_heatmap(overall, "sensor_name", "segment_setting", "mean_suitability", figdir / f"{PREFIX}_sensor_segment_suitability_heatmap.png", "Suitability: sensor x segment")
    save_heatmap(overall, "sensor_group", "segment_setting", "mean_suitability", figdir / f"{PREFIX}_sensor_group_segment_suitability_heatmap.png", "Suitability: sensor group x segment")
    p = pair_suit.rename(columns={"mean_suitability": "primary_suitability_score"})
    save_heatmap(p, "pair_id", "segment_setting", "primary_suitability_score", figdir / f"{PREFIX}_pair_segment_suitability_heatmap.png", "Suitability: pair x segment")
    save_heatmap(method, "feature_name", "segment_setting", "rank_change_abs", figdir / f"{PREFIX}_suitability_rank_change_heatmap.png", "Suitability method rank change", cmap="magma")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=overall, x="mean_monotonicity", y="mean_trendability", hue="segment_setting", style="sensor_group", s=40)
    plt.title("Monotonicity vs trendability")
    plt.tight_layout()
    plt.savefig(figdir / f"{PREFIX}_monotonicity_vs_trendability_scatter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=overall, x="suitability_sum_legacy", y="suitability_harmonic_mean", hue="segment_setting", style="sensor_group", s=40)
    plt.title("Sum legacy vs harmonic suitability")
    plt.tight_layout()
    plt.savefig(figdir / f"{PREFIX}_sum_vs_harmonic_suitability_scatter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=joined, x="primary_association_score", y="primary_suitability_score", hue="segment_setting", style="sensor_group", s=40)
    plt.title("Association vs suitability")
    plt.tight_layout()
    plt.savefig(figdir / f"{PREFIX}_association_vs_suitability_scatter.png", dpi=180)
    plt.close()
    save_heatmap(joined, "feature_name", "segment_setting", "rank_difference_abs", figdir / f"{PREFIX}_association_suitability_rank_difference_heatmap.png", "Association-suitability rank difference", cmap="magma")
    codes = {"low_association_low_suitability": 0, "high_association_low_suitability": 1, "low_association_high_suitability": 2, "middle_or_mixed": 3, "high_association_high_suitability": 4}
    save_heatmap(joined.assign(category_code=joined["alignment_category"].map(codes)), "feature_name", "segment_setting", "category_code", figdir / f"{PREFIX}_alignment_category_heatmap.png", "Alignment category code", cmap="tab10")
    save_heatmap(joined, "feature_name", "segment_setting", "combined_score", figdir / f"{PREFIX}_robust_feature_segment_candidate_heatmap.png", "Combined robust score")
    save_heatmap(joined, "sensor_group", "segment_setting", "combined_score", figdir / f"{PREFIX}_sensor_group_robust_score_heatmap.png", "Sensor group robust score")

    for metric, filename, title in [
        ("primary_association_score", f"{PREFIX}_entry_steady_exit_vs_other_segments_association.png", "entry_steady_exit vs other segments: association"),
        ("primary_suitability_score", f"{PREFIX}_entry_steady_exit_vs_other_segments_suitability.png", "entry_steady_exit vs other segments: suitability"),
        ("combined_score", f"{PREFIX}_entry_steady_exit_robust_score_comparison.png", "entry_steady_exit vs other segments: robust score"),
    ]:
        comp = ese_comp[ese_comp["metric_name"].str.contains(metric.replace("primary_", "").replace("_score", ""), regex=False)]
        if comp.empty:
            comp = ese_comp[ese_comp["metric_name"].isin(["association", "suitability_harmonic", "combined_score"])]
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_data = ese_comp[ese_comp["metric_name"].eq({"primary_association_score": "association", "primary_suitability_score": "suitability_harmonic", "combined_score": "combined_score"}[metric])]
        sns.barplot(data=plot_data, x="comparison_target", y="delta", hue="sensor_group", errorbar=None, ax=ax)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        plt.tight_layout()
        plt.savefig(figdir / filename, dpi=180)
        plt.close()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_yaml_like(path: Path, value: Any) -> None:
    try:
        import yaml

        path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        write_json(path, value)


def write_report(output_dir: Path, context: dict[str, Any], assoc_std: pd.DataFrame, overall: pd.DataFrame, joined: pd.DataFrame, robust: pd.DataFrame, ese_comp: pd.DataFrame, corr: pd.DataFrame, skipped: list[dict[str, str]]) -> None:
    top_assoc = assoc_std.sort_values("primary_association_score", ascending=False).head(10)
    top_suit = overall.sort_values("suitability_harmonic_mean", ascending=False).head(10)
    top_sum = overall.sort_values("suitability_sum_legacy", ascending=False).head(10)
    ese_assoc_rank = assoc_std.groupby("segment_setting")["primary_association_score"].mean().rank(ascending=False).get("entry_steady_exit", np.nan)
    ese_suit_rank = overall.groupby("segment_setting")["suitability_harmonic_mean"].mean().rank(ascending=False).get("entry_steady_exit", np.nan)
    ese_combined_rank = joined.groupby("segment_setting")["combined_score"].mean().rank(ascending=False).get("entry_steady_exit", np.nan)
    comp_summary = ese_comp.groupby(["comparison_target", "metric_name"]).agg(mean_delta=("delta", "mean"), improvement_ratio=("improved", "mean")).reset_index()
    acoustic = joined[joined["sensor_group"].eq("Acoustic") & joined["segment_setting"].isin(["entry_exit", "entry_steady_exit"])].sort_values("combined_score", ascending=False).head(10)
    text = f"""# H1_S6 Association/Suitability Cross-analysis with entry_steady_exit

## 1. Executive Summary

1. H1_S6 recomputed H1_S1-style association, H1_S4-style suitability, and H1_S5-style cross-analysis with `entry_steady_exit` included.
2. `entry_steady_exit` was generated as `entry + steady + exit`, excluding no-load, from raw sensor sequences.
3. Association/suitability join produced `{len(joined)}` sensor-feature-segment combinations.
4. Mean-rank positions: entry_steady_exit association rank `{ese_assoc_rank:.2f}`, suitability rank `{ese_suit_rank:.2f}`, combined robust rank `{ese_combined_rank:.2f}` among 8 segments.
5. Suitability uses harmonic mean as primary; `M + T` is retained as legacy/reference.

Top association combinations:

{top_assoc[['sensor_name','feature_name','segment_setting','primary_association_score']].to_string(index=False)}

Top harmonic suitability combinations:

{top_suit[['sensor_name','feature_name','segment_setting','suitability_harmonic_mean']].to_string(index=False)}

Top robust combinations:

{robust.head(12).to_string(index=False)}

## 2. Motivation for Re-analysis

Previous H1 analyses used seven no-load-excluded candidate segments and omitted `entry_steady_exit`. H1_S6 adds this cutting-only combined segment to evaluate whether using entry, steady, and exit together improves association, degradation-awareness, or robust combined evidence.

## 3. Data and Scope

- Process info: `{context['process_info_path']}`
- Signal data: `{context['signal_data_path']}`
- Heuristic sequence: `{context['heuristic_sequence_path']}`
- Cases: `{CASE_SCOPE}`
- Sensors: `{context['sensors']}`
- Features: `{TARGET_FEATURES}`
- Segments: `{SEGMENT_SETTINGS}`
- Ordering column: `{context['ordering_column']}`
- no-load rows excluded as separate segment labels: `{context['no_load_rows']}`
- entry_steady_exit construction: `entry + steady + exit`, time-axis concatenation, no-load excluded.

## 4. H1_S1-style Association Results

Primary association score is `top3_mean_abs_spearman`. `entry_steady_exit` is compared against full_length, steady, entry, exit, entry_steady, entry_exit, and steady_exit.

Segment mean association:

{assoc_std.groupby('segment_setting')['primary_association_score'].mean().sort_values(ascending=False).to_string()}

## 5. H1_S4-style Suitability Results

Monotonicity and trendability follow the paper formulas. Primary suitability is harmonic mean; legacy suitability is `M + T`.

Segment mean harmonic suitability:

{overall.groupby('segment_setting')['suitability_harmonic_mean'].mean().sort_values(ascending=False).to_string()}

Top legacy M+T combinations:

{top_sum[['sensor_name','feature_name','segment_setting','suitability_sum_legacy']].to_string(index=False)}

## 6. Association vs Suitability Cross-analysis

Correlation summary:

{corr.to_string(index=False)}

Alignment category counts:

{joined['alignment_category'].value_counts().to_string()}

## 7. entry_steady_exit Focus Analysis

Mean delta and improvement ratio versus other segments:

{comp_summary.to_string(index=False)}

`entry_steady_exit` versus `entry_exit` should be interpreted as the effect of adding the steady cutting interval to entry/exit transients. `entry_steady_exit` versus `full_length` should be interpreted as cutting-only combined signal versus whole signal that may include no-load.

Acoustic entry_exit / entry_steady_exit candidates:

{acoustic[['sensor_name','feature_name','segment_setting','primary_association_score','primary_suitability_score','combined_score','alignment_category']].to_string(index=False)}

## 8. Manufacturing Interpretation

`entry_steady_exit` represents the full cutting contact cycle without no-load. If it performs well, it suggests that combining transient contact/exit behavior with stable cutting load can preserve both target relevance and degradation trend. Current features often reflect load progression, vibration features reflect dynamics and instability, and acoustic features may reflect contact/exit impacts, friction, chip events, and micro events. These are physical hypotheses, not causal claims.

## 9. Implications for H2/H3 Modeling

H2/H3 should include `entry_steady_exit` as an explicit candidate segment. Reduced searches should prioritize robust H1_S6 candidates and compare `entry_steady_exit` against `entry_exit`, `steady_exit`, and `full_length` under domain shift.

## 10. Limitations

Association and suitability are EDA diagnostics and do not guarantee prediction performance. `entry_steady_exit` construction depends on heuristic segment boundaries. Segment length, operating condition differences, and run-order assumptions can affect results.

## 11. Recommendations

- Add `entry_steady_exit` to future H2/H3 segment grids.
- Use robust candidates as reduced candidate priors.
- Keep disagreement cases for ablation and diagnostics.
- Validate acoustic entry/exit/entry_steady_exit patterns in prediction experiments.

## Execution Path

- experiment_id: `{context['experiment_id']}`
- hypothesis_id: `H1`
- scenario_id: `S6`
- execution_dir: `{output_dir}`
- path_schema_version: `hierarchical_v1`
- skipped: `{skipped}`
"""
    report = output_dir / "reports" / f"{PREFIX}_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    html_text = "<!doctype html><html><head><meta charset='utf-8'><title>H1_S6 Report</title></head><body>" + html.escape(text).replace("\n", "<br>\n") + "</body></html>"
    (output_dir / "reports" / f"{PREFIX}_report.html").write_text(html_text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    experiment_id = f"{timestamp}_H1_S6_association_suitability_cross_analysis_with_entry_steady_exit"
    path_config = {"experiment": {"experiment_id": experiment_id, "timestamp": timestamp, "hypothesis_id": "H1", "scenario_id": "S6", "experiment_topic": "association_suitability_cross_analysis_with_entry_steady_exit"}}
    output_dir = Path(create_execution_dir(path_config, root=root / "experiments" / "executions"))
    for d in ["configs", "data", "analysis", "figures", "reports", "logs"]:
        (output_dir / d).mkdir(parents=True, exist_ok=True)
    error_log = output_dir / "logs" / f"{PREFIX}_error.log"
    error_log.touch()
    run_log = output_dir / "logs" / f"{PREFIX}_run.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line)
        with run_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    try:
        log(f"H1_S6 started: {experiment_id}")
        process_info_path = root / args.process_info_path
        signal_data_path = root / args.signal_data_path
        heuristic_sequence_path = root / args.heuristic_sequence_path
        data = load_dataset(process_info_path, signal_data_path, heuristic_sequence_path)
        sensors_df = identify_sensors(data)
        sensors = sensors_df.loc[sensors_df["used"], "sensor_name"].tolist()
        feature_long, ese_summary = find_or_create_entry_steady_exit_features(data, sensors)
        ordering_column = "run" if "run" in feature_long.columns else "time"

        association = compute_association_metrics(feature_long, experiment_id)
        case_suit = compute_case_level_suitability(feature_long, experiment_id, ordering_column)
        overall_suit, pair_suit, fs_suit, simplified_suit, method_comp = aggregate_suitability_results(case_suit)
        top_bottom = top_bottom_suitability_summary(overall_suit, fs_suit)
        assoc_std = standardized_association(association["association_feature_level"], experiment_id)
        joined = join_association_and_suitability(assoc_std, overall_suit, experiment_id)
        align_summary = alignment_category_summary(joined)
        corr = correlation_summary(joined)
        robust = robust_candidates(joined)
        disagreement = disagreement_table(joined)
        ese_comp = compare_entry_steady_exit_with_other_segments(joined)

        feature_def = identify_feature_columns(feature_long)
        segment_def = identify_segment_settings(feature_long)
        no_load_rows = int(feature_long["segment_setting"].astype(str).str.contains("no_load|no-load", case=False, na=False).sum())
        validation = {
            "ok": bool("entry_steady_exit" in feature_long["segment_setting"].unique() and len(joined) > 0),
            "features_found": feature_def["found"].to_dict(),
            "segments_found": segment_def.set_index("segment_setting")["used"].to_dict(),
            "entry_steady_exit_created": True,
            "no_load_rows_in_feature_long": no_load_rows,
            "case_count": int(feature_long["case_id"].nunique()),
            "joined_rows": int(len(joined)),
            "suitability_harmonic_in_range": bool(overall_suit["suitability_harmonic_mean"].between(0, 1).all()),
            "suitability_sum_in_range": bool(overall_suit["suitability_sum_legacy"].between(0, 2).all()),
        }
        if not validation["ok"]:
            raise ValueError(f"H1_S6 validation failed: {validation}")

        # Data/config outputs
        write_json(output_dir / "data" / f"{PREFIX}_dataset_summary.json", {"rows": len(data), "cases": sorted(data["case_id"].unique().tolist()), "sensors": sensors})
        write_json(output_dir / "data" / f"{PREFIX}_ordering_summary.json", {"ordering_column": ordering_column, "rule": "run index within case"})
        write_json(output_dir / "data" / f"{PREFIX}_entry_steady_exit_construction_summary.json", ese_summary)
        feature_def.to_csv(output_dir / "data" / f"{PREFIX}_feature_definition.csv", index=False)
        segment_def.to_csv(output_dir / "data" / f"{PREFIX}_segment_definition.csv", index=False)
        sensors_df.to_csv(output_dir / "data" / f"{PREFIX}_sensor_mapping.csv", index=False)
        write_json(output_dir / "configs" / f"{PREFIX}_config_validation.json", validation)
        config = {"experiment": path_config["experiment"], "process_info_path": str(process_info_path), "signal_data_path": str(signal_data_path), "heuristic_sequence_path": str(heuristic_sequence_path), "segments": SEGMENT_SETTINGS, "features": TARGET_FEATURES, "schema_version": SCHEMA_VERSION}
        config["config_hash"] = stable_hash(config)
        write_yaml_like(output_dir / "configs" / f"{PREFIX}_input_config.yaml", config)
        write_yaml_like(output_dir / "configs" / f"{PREFIX}_resolved_config.yaml", {**config, "validation": validation})

        # Analysis outputs
        association["association_feature_level"].to_csv(output_dir / "analysis" / f"{PREFIX}_association_feature_level.csv", index=False)
        association["association_segment_level"].to_csv(output_dir / "analysis" / f"{PREFIX}_association_segment_level.csv", index=False)
        association["association_case_level"].to_csv(output_dir / "analysis" / f"{PREFIX}_association_case_level.csv", index=False)
        association["association_pair_level"].to_csv(output_dir / "analysis" / f"{PREFIX}_association_pair_level.csv", index=False)
        association["segment_ranking"].to_csv(output_dir / "analysis" / f"{PREFIX}_segment_ranking.csv", index=False)
        association["baseline_comparison"].to_csv(output_dir / "analysis" / f"{PREFIX}_baseline_comparison.csv", index=False)
        association["sensor_consistency"].to_csv(output_dir / "analysis" / f"{PREFIX}_sensor_consistency.csv", index=False)
        association["pair_level_association_summary"].to_csv(output_dir / "analysis" / f"{PREFIX}_pair_level_association_summary.csv", index=False)
        case_suit.to_csv(output_dir / "analysis" / f"{PREFIX}_case_level_suitability_results.csv", index=False)
        overall_suit.to_csv(output_dir / "analysis" / f"{PREFIX}_suitability_results.csv", index=False)
        pair_suit.to_csv(output_dir / "analysis" / f"{PREFIX}_pair_level_suitability_results.csv", index=False)
        fs_suit.to_csv(output_dir / "analysis" / f"{PREFIX}_feature_segment_summary.csv", index=False)
        simplified_suit.to_csv(output_dir / "analysis" / f"{PREFIX}_feature_segment_simplified_table.csv", index=False)
        method_comp.to_csv(output_dir / "analysis" / f"{PREFIX}_suitability_method_comparison.csv", index=False)
        top_bottom.to_csv(output_dir / "analysis" / f"{PREFIX}_top_bottom_suitability_summary.csv", index=False)
        joined.to_csv(output_dir / "analysis" / f"{PREFIX}_association_vs_suitability_joined_table.csv", index=False)
        align_summary.to_csv(output_dir / "analysis" / f"{PREFIX}_alignment_category_summary.csv", index=False)
        corr.to_csv(output_dir / "analysis" / f"{PREFIX}_cross_score_correlation_summary.csv", index=False)
        robust.to_csv(output_dir / "analysis" / f"{PREFIX}_robust_feature_segment_candidates.csv", index=False)
        disagreement.to_csv(output_dir / "analysis" / f"{PREFIX}_association_suitability_disagreement.csv", index=False)
        ese_comp.to_csv(output_dir / "analysis" / f"{PREFIX}_entry_steady_exit_comparison_summary.csv", index=False)

        create_figures(output_dir, assoc_std, association["association_segment_level"], association["association_pair_level"], overall_suit, pair_suit, fs_suit, method_comp, joined, ese_comp)

        skipped: list[dict[str, str]] = []
        context = {
            "experiment_id": experiment_id,
            "process_info_path": str(process_info_path),
            "signal_data_path": str(signal_data_path),
            "heuristic_sequence_path": str(heuristic_sequence_path),
            "sensors": sensors,
            "ordering_column": ordering_column,
            "no_load_rows": no_load_rows,
        }
        write_report(output_dir, context, assoc_std, overall_suit, joined, robust, ese_comp, corr, skipped)
        summary = {
            "experiment_id": experiment_id,
            "execution_dir": str(output_dir),
            "case_count": int(feature_long["case_id"].nunique()),
            "num_feature_sensor_segment_combinations": int(len(joined)),
            "top_association": assoc_std.sort_values("primary_association_score", ascending=False).head(10).to_dict(orient="records"),
            "top_suitability": overall_suit.sort_values("suitability_harmonic_mean", ascending=False).head(10).to_dict(orient="records"),
            "top_robust": robust.head(10).to_dict(orient="records"),
            "entry_steady_exit_comparison": ese_comp.groupby(["comparison_target", "metric_name"]).agg(mean_delta=("delta", "mean"), improvement_ratio=("improved", "mean")).reset_index().to_dict(orient="records"),
            "validation": validation,
            "skipped": skipped,
        }
        write_json(output_dir / "analysis" / f"{PREFIX}_analysis_summary.json", summary)
        write_json(output_dir / "logs" / f"{PREFIX}_environment.json", collect_environment(str(root)))
        log(f"H1_S6 finished: {experiment_id}")
        return summary
    except Exception:
        error_log.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H1_S6 association/suitability analysis with entry_steady_exit.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--process-info-path", default="datasets/processed/mill_process_info_enabled.csv")
    parser.add_argument("--signal-data-path", default="datasets/processed/mill_signal_data_enabled.csv")
    parser.add_argument("--heuristic-sequence-path", default="datasets/metadata/heuristic_sequence.csv")
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

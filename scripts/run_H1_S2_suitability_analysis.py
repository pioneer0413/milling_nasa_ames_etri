#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
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
from scipy.stats import kurtosis, skew

try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - environment dependent fallback
    sns = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework import __version__
from milling_experiment_framework.core.config import SCHEMA_VERSION, stable_hash
from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.utils.io import write_csv, write_json, write_yaml
from milling_experiment_framework.utils.paths import ExperimentPaths


CASE_SCOPE = [1, 2, 8, 9, 12, 14]
PAIR_DEFINITION = {"A": [1, 9], "B": [2, 12], "C": [8, 14]}
PAIR_BY_CASE = {case: pair for pair, cases in PAIR_DEFINITION.items() for case in cases}
SEGMENT_SETTINGS = ["full_length", "steady", "entry", "exit", "entry_steady", "entry_exit", "steady_exit", "entry_steady_exit"]
EXCLUDED_SEGMENTS = ["no_load", "no_load_entry", "no_load_steady", "no_load_exit"]
TARGET_FEATURES = [
    "mean",
    "std",
    "max",
    "min",
    "peak_to_peak",
    "kurtosis",
    "skewness",
    "spectral_centroid",
    "band_energy",
]
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
PRIMARY_SUITABILITY_METHOD = "harmonic_mean"
COMPUTE_LEGACY_SUM = True
SUITABILITY_EPSILON = 1e-12
PROGNOSABILITY_EPSILON = 1e-12


def identify_feature_columns(df: pd.DataFrame, target_features: list[str]) -> pd.DataFrame:
    rows = []
    long_features = set(df["feature_name"].dropna().astype(str).unique()) if "feature_name" in df.columns else set()
    for feature in target_features:
        matched = []
        if feature in long_features:
            matched.append(f"feature_name={feature}")
        if feature in df.columns:
            matched.append(feature)
        suffix = f"__{feature}"
        matched.extend([col for col in df.columns if col.endswith(suffix)])
        rows.append(
            {
                "feature_name": feature,
                "feature_group": FEATURE_GROUPS.get(feature, "unknown"),
                "found": bool(matched),
                "matched_columns": "|".join(sorted(set(matched))),
                "definition": feature_definition_text(feature),
            }
        )
    return pd.DataFrame(rows)


def identify_segment_settings(df: pd.DataFrame) -> pd.DataFrame:
    if "segment_setting" in df.columns:
        available = sorted(df["segment_setting"].dropna().astype(str).unique().tolist())
    else:
        available = []
        for col in df.columns:
            parts = str(col).split("__")
            if len(parts) >= 3:
                available.append(parts[-2])
        available = sorted(set(available))
    rows = []
    for segment in SEGMENT_SETTINGS:
        rows.append(
            {
                "segment_setting": segment,
                "used": segment in available or not available,
                "definition": segment_definition_text(segment),
                "source_available": segment in available,
                "excluded_reason": "",
            }
        )
    for segment in available:
        if "no_load" in segment:
            rows.append(
                {
                    "segment_setting": segment,
                    "used": False,
                    "definition": "excluded no-load related segment",
                    "source_available": True,
                    "excluded_reason": "no-load segments are out of H1_S2 scope",
                }
            )
    return pd.DataFrame(rows)


def compute_monotonicity(x: np.ndarray) -> float:
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    diff = np.diff(x)
    n_inc = int(np.sum(diff > 0))
    n_dec = int(np.sum(diff < 0))
    return float(abs((n_inc - n_dec) / (n - 1)))


def compute_trendability(x: np.ndarray, t: np.ndarray) -> float:
    x = np.asarray(x, dtype="float64")
    t = np.asarray(t, dtype="float64")
    finite = np.isfinite(x) & np.isfinite(t)
    x = x[finite]
    t = t[finite]
    n = len(x)
    if n < 2:
        return float("nan")
    numerator = n * np.sum(x * t) - np.sum(x) * np.sum(t)
    x_term = n * np.sum(x**2) - np.sum(x) ** 2
    t_term = n * np.sum(t**2) - np.sum(t) ** 2
    denominator = math.sqrt(max(x_term * t_term, 0.0))
    if denominator <= 0:
        return float("nan")
    return float(abs(numerator / denominator))


def compute_suitability_sum(monotonicity: float, trendability: float) -> float:
    if np.isnan(monotonicity) or np.isnan(trendability):
        return float("nan")
    return float(monotonicity + trendability)


def compute_suitability_baseline(monotonicity: float, trendability: float, prognosability: float) -> float:
    if np.isnan(monotonicity) or np.isnan(trendability) or np.isnan(prognosability):
        return float("nan")
    return float(monotonicity + trendability + prognosability)


def compute_suitability_harmonic(
    monotonicity: float,
    trendability: float,
    prognosability: float,
    epsilon: float = SUITABILITY_EPSILON,
) -> float:
    if np.isnan(monotonicity) or np.isnan(trendability) or np.isnan(prognosability):
        return float("nan")
    if monotonicity <= 0 and trendability <= 0 and prognosability <= 0:
        return 0.0
    return float(
        3.0
        / (
            (1.0 / (monotonicity + epsilon))
            + (1.0 / (trendability + epsilon))
            + (1.0 / (prognosability + epsilon))
        )
    )


def compute_suitability(
    monotonicity: float,
    trendability: float,
    prognosability: float,
    method: str = PRIMARY_SUITABILITY_METHOD,
    epsilon: float = SUITABILITY_EPSILON,
) -> float:
    if method == "sum_legacy":
        return compute_suitability_sum(monotonicity, trendability)
    if method in {"baseline", "additive"}:
        return compute_suitability_baseline(monotonicity, trendability, prognosability)
    if method == "harmonic_mean":
        return compute_suitability_harmonic(monotonicity, trendability, prognosability, epsilon)
    raise ValueError(f"Unknown suitability method: {method}")


def compute_prognosability(starts: np.ndarray, finals: np.ndarray, epsilon: float = PROGNOSABILITY_EPSILON) -> float:
    starts = np.asarray(starts, dtype="float64")
    finals = np.asarray(finals, dtype="float64")
    finite = np.isfinite(starts) & np.isfinite(finals)
    starts = starts[finite]
    finals = finals[finite]
    if len(finals) == 0:
        return float("nan")
    final_variance = float(np.var(finals))
    mean_start_final_distance = float(np.mean(np.abs(starts - finals)))
    return float(np.exp(-(final_variance / (mean_start_final_distance + epsilon))))


def add_prognosability_and_suitability_scores(
    case_results: pd.DataFrame,
    primary_method: str = PRIMARY_SUITABILITY_METHOD,
    epsilon: float = SUITABILITY_EPSILON,
) -> pd.DataFrame:
    out = case_results.copy()
    out["prognosability"] = np.nan
    group_cols = ["sensor_name", "feature_name", "segment_setting"]
    for _, group in out.groupby(group_cols, dropna=False):
        ok = group.loc[group["calculation_status"].eq("ok")]
        p = compute_prognosability(ok["x_start"].to_numpy(dtype="float64"), ok["x_final"].to_numpy(dtype="float64"), PROGNOSABILITY_EPSILON)
        out.loc[group.index, "prognosability"] = p

    out["suitability_mt_legacy"] = out.apply(
        lambda row: compute_suitability_sum(row["monotonicity"], row["trendability"]),
        axis=1,
    )
    out["suitability_sum_legacy"] = out["suitability_mt_legacy"]
    out["suitability_baseline"] = out.apply(
        lambda row: compute_suitability_baseline(row["monotonicity"], row["trendability"], row["prognosability"]),
        axis=1,
    )
    out["suitability_harmonic"] = out.apply(
        lambda row: compute_suitability_harmonic(row["monotonicity"], row["trendability"], row["prognosability"], epsilon),
        axis=1,
    )
    out["suitability_harmonic_mean"] = out["suitability_harmonic"]
    out["suitability"] = out.apply(
        lambda row: compute_suitability(row["monotonicity"], row["trendability"], row["prognosability"], primary_method, epsilon),
        axis=1,
    )
    return out


def compute_case_level_suitability(
    feature_long: pd.DataFrame,
    experiment_id: str,
    ordering_column: str,
    primary_method: str = PRIMARY_SUITABILITY_METHOD,
    epsilon: float = SUITABILITY_EPSILON,
) -> pd.DataFrame:
    rows = []
    group_cols = ["case_id", "sensor_name", "sensor_group", "segment_setting", "feature_name", "feature_group"]
    for keys, group in feature_long.groupby(group_cols, dropna=False):
        case_id, sensor_name, sensor_group, segment_setting, feature_name, feature_group = keys
        group = group.sort_values([ordering_column, "run"], kind="mergesort")
        x_raw = group["feature_value"].to_numpy(dtype="float64")
        t_raw = group[ordering_column].to_numpy(dtype="float64")
        missing = int(np.sum(~np.isfinite(x_raw) | ~np.isfinite(t_raw)))
        finite = np.isfinite(x_raw) & np.isfinite(t_raw)
        x = x_raw[finite]
        t = t_raw[finite]
        status = "ok"
        if len(x) < 2:
            status = "skipped_n_less_than_2"
            monotonicity = trendability = float("nan")
            x_start = x_final = float("nan")
        else:
            monotonicity = compute_monotonicity(x)
            trendability = compute_trendability(x, t)
            x_start = float(x[0])
            x_final = float(x[-1])
            if not np.isfinite(trendability):
                status = "skipped_zero_trendability_denominator"
            if not np.isfinite(monotonicity):
                status = "skipped_monotonicity_not_computable"
        rows.append(
            {
                "experiment_id": experiment_id,
                "case_id": int(case_id),
                "sensor_name": sensor_name,
                "sensor_group": sensor_group,
                "feature_name": feature_name,
                "feature_group": feature_group,
                "segment_setting": segment_setting,
                "monotonicity": monotonicity,
                "trendability": trendability,
                "x_start": x_start,
                "x_final": x_final,
                "primary_suitability_method": primary_method,
                "suitability_epsilon": epsilon,
                "prognosability_epsilon": PROGNOSABILITY_EPSILON,
                "sample_count": int(len(x)),
                "missing_count": missing,
                "missing_rate": float(missing / len(group)) if len(group) else float("nan"),
                "ordering_column": ordering_column,
                "calculation_status": status,
            }
        )
    out = add_prognosability_and_suitability_scores(pd.DataFrame(rows), primary_method, epsilon)
    if primary_method == "sum_legacy":
        max_allowed = 2
    elif primary_method in {"baseline", "additive"}:
        max_allowed = 3
    else:
        max_allowed = 1
    invalid = out["suitability"].notna() & ~out["suitability"].between(0, max_allowed)
    out.loc[invalid, "calculation_status"] = "invalid_suitability_range"
    return out


def aggregate_overall_suitability(case_results: pd.DataFrame) -> pd.DataFrame:
    ok = case_results.loc[case_results["calculation_status"].eq("ok")].copy()
    group_cols = ["experiment_id", "sensor_name", "sensor_group", "feature_name", "feature_group", "segment_setting"]
    agg = (
        ok.groupby(group_cols, dropna=False)
        .agg(
            mean_monotonicity=("monotonicity", "mean"),
            mean_trendability=("trendability", "mean"),
            mean_prognosability=("prognosability", "mean"),
            mean_suitability=("suitability", "mean"),
            suitability_mt_legacy=("suitability_mt_legacy", "mean"),
            suitability_sum_legacy=("suitability_sum_legacy", "mean"),
            suitability_baseline=("suitability_baseline", "mean"),
            suitability_harmonic=("suitability_harmonic", "mean"),
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
    agg["std_suitability"] = agg["std_suitability"].fillna(0.0)
    agg = add_suitability_ranks(agg)
    return agg


def aggregate_pair_level_suitability(case_results: pd.DataFrame, pair_definition: dict[str, list[int]]) -> pd.DataFrame:
    ok = case_results.loc[case_results["calculation_status"].eq("ok")].copy()
    ok["pair_id"] = ok["case_id"].map(PAIR_BY_CASE)
    ok["cases_in_pair"] = ok["pair_id"].map(lambda pair: ",".join(map(str, pair_definition[pair])) if pair in pair_definition else "")
    group_cols = ["experiment_id", "pair_id", "cases_in_pair", "sensor_name", "sensor_group", "feature_name", "feature_group", "segment_setting"]
    out = (
        ok.groupby(group_cols, dropna=False)
        .agg(
            mean_monotonicity=("monotonicity", "mean"),
            mean_trendability=("trendability", "mean"),
            mean_prognosability=("prognosability", "mean"),
            mean_suitability=("suitability", "mean"),
            suitability_mt_legacy=("suitability_mt_legacy", "mean"),
            suitability_sum_legacy=("suitability_sum_legacy", "mean"),
            suitability_baseline=("suitability_baseline", "mean"),
            suitability_harmonic=("suitability_harmonic", "mean"),
            suitability_harmonic_mean=("suitability_harmonic_mean", "mean"),
            std_suitability=("suitability", "std"),
            total_sample_count=("sample_count", "sum"),
            primary_suitability_method=("primary_suitability_method", "first"),
            suitability_epsilon=("suitability_epsilon", "first"),
        )
        .reset_index()
    )
    out["std_suitability"] = out["std_suitability"].fillna(0.0)
    out["rank_by_baseline"] = out.groupby("pair_id")["suitability_baseline"].rank(ascending=False, method="first").astype(int)
    out["rank_by_harmonic"] = out.groupby("pair_id")["suitability_harmonic"].rank(ascending=False, method="first").astype(int)
    out["rank_by_sum_legacy"] = out.groupby("pair_id")["suitability_sum_legacy"].rank(ascending=False, method="first").astype(int)
    out["rank_by_harmonic_mean"] = out.groupby("pair_id")["suitability_harmonic_mean"].rank(ascending=False, method="first").astype(int)
    primary_method = out["primary_suitability_method"].dropna().iloc[0] if not out["primary_suitability_method"].dropna().empty else PRIMARY_SUITABILITY_METHOD
    if primary_method == "sum_legacy":
        out["rank_by_suitability"] = out["rank_by_sum_legacy"]
    elif primary_method in {"baseline", "additive"}:
        out["rank_by_suitability"] = out["rank_by_baseline"]
    else:
        out["rank_by_suitability"] = out["rank_by_harmonic"]
    out = out.sort_values(["pair_id", "rank_by_suitability"], ascending=[True, True]).reset_index(drop=True)
    return out


def add_suitability_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "suitability_baseline" in out:
        out["rank_by_baseline"] = out["suitability_baseline"].rank(ascending=False, method="first").astype(int)
    if "suitability_harmonic" in out:
        out["rank_by_harmonic"] = out["suitability_harmonic"].rank(ascending=False, method="first").astype(int)
    out["rank_by_sum_legacy"] = out["suitability_sum_legacy"].rank(ascending=False, method="first").astype(int)
    out["rank_by_harmonic_mean"] = out["suitability_harmonic_mean"].rank(ascending=False, method="first").astype(int)
    primary_method = out["primary_suitability_method"].dropna().iloc[0] if "primary_suitability_method" in out and not out["primary_suitability_method"].dropna().empty else PRIMARY_SUITABILITY_METHOD
    if primary_method == "sum_legacy":
        out["rank_by_suitability"] = out["rank_by_sum_legacy"]
    elif primary_method in {"baseline", "additive"}:
        out["rank_by_suitability"] = out["rank_by_baseline"]
    else:
        out["rank_by_suitability"] = out["rank_by_harmonic"] if "rank_by_harmonic" in out else out["rank_by_harmonic_mean"]
    return out.sort_values("rank_by_suitability", ascending=True).reset_index(drop=True)


def create_suitability_heatmaps(
    overall: pd.DataFrame,
    feature_segment: pd.DataFrame,
    pair_level: pd.DataFrame,
    method_comparison: pd.DataFrame,
    figure_dir: Path,
) -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid")
    _heatmap(
        feature_segment.pivot_table(index="feature_name", columns="segment_setting", values="mean_suitability", aggfunc="mean").reindex(index=TARGET_FEATURES, columns=SEGMENT_SETTINGS),
        "H1_S2 Feature x Segment Primary Suitability (harmonic mean)",
        figure_dir / "H1_S2_feature_segment_suitability_heatmap.png",
        vmax=1,
        label="Primary suitability (harmonic mean)",
    )
    _heatmap(
        feature_segment.pivot_table(index="feature_name", columns="segment_setting", values="suitability_harmonic_mean", aggfunc="mean").reindex(index=TARGET_FEATURES, columns=SEGMENT_SETTINGS),
        "H1_S2 Feature x Segment Suitability Harmonic Mean",
        figure_dir / "H1_S2_feature_segment_suitability_heatmap_harmonic_mean.png",
        vmax=1,
        label="Suitability harmonic mean",
    )
    _heatmap(
        feature_segment.pivot_table(index="feature_name", columns="segment_setting", values="suitability_baseline", aggfunc="mean").reindex(index=TARGET_FEATURES, columns=SEGMENT_SETTINGS),
        "H1_S2 Feature x Segment Suitability Baseline",
        figure_dir / "H1_S2_feature_segment_suitability_heatmap_baseline.png",
        vmax=3,
        label="Suitability baseline = M + T + P",
    )
    _heatmap(
        method_comparison.assign(rank_change_abs=method_comparison["rank_change_abs"]).pivot_table(index="feature_name", columns="segment_setting", values="rank_change_abs", aggfunc="mean").reindex(index=TARGET_FEATURES, columns=SEGMENT_SETTINGS),
        "H1_S2 Mean Rank Change Abs: Sum Legacy vs Harmonic",
        figure_dir / "H1_S2_suitability_rank_change_heatmap.png",
        vmax=None,
        label="Mean absolute rank change",
    )
    _heatmap(
        overall.pivot_table(index="sensor_name", columns="segment_setting", values="mean_suitability", aggfunc="mean").reindex(columns=SEGMENT_SETTINGS),
        "H1_S2 Sensor x Segment Suitability",
        figure_dir / "H1_S2_sensor_segment_suitability_heatmap.png",
        vmax=1,
        label="Primary suitability (harmonic mean)",
    )
    _heatmap(
        overall.pivot_table(index="sensor_group", columns="segment_setting", values="mean_suitability", aggfunc="mean").reindex(columns=SEGMENT_SETTINGS),
        "H1_S2 Sensor Group x Segment Suitability",
        figure_dir / "H1_S2_sensor_group_segment_suitability_heatmap.png",
        vmax=1,
        label="Primary suitability (harmonic mean)",
    )
    _heatmap(
        pair_level.pivot_table(index="pair_id", columns="segment_setting", values="mean_suitability", aggfunc="mean").reindex(index=["A", "B", "C"], columns=SEGMENT_SETTINGS),
        "H1_S2 Pair x Segment Suitability",
        figure_dir / "H1_S2_pair_segment_suitability_heatmap.png",
        vmax=1,
        label="Primary suitability (harmonic mean)",
    )
    plt.figure(figsize=(9, 6))
    if sns is not None:
        sns.scatterplot(
            data=overall,
            x="mean_monotonicity",
            y="mean_trendability",
            hue="segment_setting",
            style="feature_group",
            s=70,
        )
    else:
        segments = list(dict.fromkeys(overall["segment_setting"].tolist()))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(segments), 1)))
        color_map = dict(zip(segments, colors))
        for segment, group in overall.groupby("segment_setting"):
            plt.scatter(
                group["mean_monotonicity"],
                group["mean_trendability"],
                s=55,
                label=segment,
                color=color_map.get(segment),
                alpha=0.85,
            )
        plt.legend(fontsize=8, loc="best")
    plt.title("H1_S2 Monotonicity vs Trendability")
    plt.xlabel("Mean monotonicity")
    plt.ylabel("Mean trendability")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(figure_dir / "H1_S2_monotonicity_vs_trendability_scatter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    if sns is not None:
        sns.scatterplot(
            data=feature_segment,
            x="suitability_baseline",
            y="suitability_harmonic",
            hue="segment_setting",
            style="feature_group",
            s=80,
        )
    else:
        for segment, group in feature_segment.groupby("segment_setting"):
            plt.scatter(group["suitability_baseline"], group["suitability_harmonic"], s=60, label=segment, alpha=0.85)
        plt.legend(fontsize=8)
    plt.title("H1_S2 Baseline vs Harmonic Suitability")
    plt.xlabel("Suitability baseline = M + T + P")
    plt.ylabel("Suitability harmonic = HM(M,T,P)")
    plt.tight_layout()
    plt.savefig(figure_dir / "H1_S2_sum_vs_harmonic_suitability_scatter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    data = overall.copy()
    sizes = 35 + 140 * (data["suitability_harmonic_mean"] / max(data["suitability_harmonic_mean"].max(), 1e-12))
    scatter = plt.scatter(
        data["mean_monotonicity"],
        data["mean_trendability"],
        c=data["suitability_harmonic_mean"],
        s=sizes,
        cmap="viridis",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.2,
    )
    plt.colorbar(scatter, label="Suitability harmonic mean")
    plt.title("H1_S2 Monotonicity vs Trendability with Harmonic Suitability")
    plt.xlabel("Mean monotonicity")
    plt.ylabel("Mean trendability")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(figure_dir / "H1_S2_monotonicity_vs_trendability_with_suitability.png", dpi=180)
    plt.close()


def write_H1_S2_report(
    output_dir: Path,
    context: dict[str, Any],
    feature_definition: pd.DataFrame,
    segment_definition: pd.DataFrame,
    overall: pd.DataFrame,
    feature_segment: pd.DataFrame,
    simplified: pd.DataFrame,
    pair_level: pd.DataFrame,
    top_bottom: pd.DataFrame,
) -> None:
    report_path = output_dir / "reports" / "H1_S2_report.md"
    top10 = simplified.sort_values("suitability", ascending=False).head(10)
    bottom10 = simplified.sort_values("suitability", ascending=True).head(10)
    top10_sum = simplified.sort_values("suitability_baseline", ascending=False).head(10)
    top10_harmonic = simplified.sort_values("suitability_harmonic", ascending=False).head(10)
    top_sum_keys = set(zip(top10_sum["feature"], top10_sum["segment_combination"]))
    top_harmonic_keys = set(zip(top10_harmonic["feature"], top10_harmonic["segment_combination"]))
    robust_common = top10_harmonic.loc[
        [key in top_sum_keys for key in zip(top10_harmonic["feature"], top10_harmonic["segment_combination"])]
    ].head(10)
    method_comparison = suitability_method_comparison(overall)
    rank_change_big = method_comparison.sort_values("rank_change_abs", ascending=False).head(10)
    sum_high_harmonic_drop = method_comparison.sort_values(["rank_change", "rank_by_baseline"], ascending=[False, True]).head(10)
    balanced = method_comparison.loc[method_comparison["interpretation"].eq("balanced_high_m_and_t")].head(10)
    steady_best = feature_segment.loc[feature_segment["segment_setting"].eq("steady")].sort_values("mean_suitability", ascending=False).head(5)
    entry_exit_segments = ["entry", "exit", "entry_steady", "entry_exit", "steady_exit"]
    entry_exit_best = feature_segment.loc[feature_segment["segment_setting"].isin(entry_exit_segments)].sort_values("mean_suitability", ascending=False).head(8)
    high_t_low_m = top_bottom.loc[top_bottom["section"].eq("high_trendability_low_monotonicity_combinations")].head(8)
    high_m_low_t = top_bottom.loc[top_bottom["section"].eq("high_monotonicity_low_trendability_combinations")].head(8)
    missing_features = feature_definition.loc[~feature_definition["found"], "feature_name"].tolist()

    text = f"""# H1_S2 feature_VB_segment_suitability_eda

## 1. Executive Summary

1. 총 `{context['num_case_level_ok']}`개 case-level sequence에서 monotonicity, trendability, suitability를 계산했습니다.
2. Primary score는 harmonic mean이며, Feature x segment 평균 기준 최고 조합은 `{top10.iloc[0]['feature']} / {top10.iloc[0]['segment_combination']}`이고 primary suitability는 `{top10.iloc[0]['suitability']:.4f}`입니다.
3. Sensor까지 포함한 최고 조합은 `{overall.iloc[0]['sensor_name']} / {overall.iloc[0]['feature_name']} / {overall.iloc[0]['segment_setting']}`이며 mean harmonic suitability는 `{overall.iloc[0]['mean_suitability']:.4f}`입니다.
4. Steady 구간에서는 `{', '.join(steady_best['feature_name'].head(3).tolist())}` feature가 상대적으로 높게 나타났습니다.
5. Entry/exit 포함 구간에서는 `{', '.join(entry_exit_best['feature_name'].head(3).tolist())}` feature가 높은 degradation trend 후보로 나타났습니다.

## 2. Data and Scope

- Process file: `{context['process_info_path']}`
- Signal file: `{context['signal_data_path']}`
- Heuristic segment file: `{context['heuristic_sequence_path']}`
- 사용 case: `{context['case_scope']}`
- 사용 sensor: `{context['sensors']}`
- 사용 segment setting: `{SEGMENT_SETTINGS}`
- 사용 feature: `{TARGET_FEATURES}`
- no-load 제외 row 수: `{context['no_load_excluded_rows']}`
- 계산 가능한 sensor x feature x segment 조합 수: `{len(overall)}`
- heuristic segment 정의와 signal이 모두 존재해 계산에 사용한 run 수: `{context['num_runs']}`

## 3. Feature Definition

9개 feature는 time-domain statistics, shape, frequency group으로 나누었습니다. 누락 feature: `{missing_features if missing_features else '없음'}`.

{markdown_table(feature_definition[['feature_name','feature_group','found']])}

## 4. Segment Definition

{markdown_table(segment_definition[['segment_setting','used','definition']])}

Segment 조합은 framework의 heuristic sequence 기준으로 entry, steady, exit raw signal 구간을 만들고, 조합 segment는 time axis 방향 concatenate 후 run-level feature로 집계했습니다.

## 5. Metric Formula

- Monotonicity: `M = |(N_inc - N_dec) / (n - 1)|`
- Trendability: feature value `x`와 ordering index `t` 사이 Pearson correlation coefficient의 절댓값
	- Prognosability: `P = exp(-var(x_final)/(mean(abs(x_start - x_final)) + epsilon))`
	- Suitability baseline: `S_baseline = M + T + P`
	- Suitability harmonic: `S_hm = 3 / (1/(M+epsilon) + 1/(T+epsilon) + 1/(P+epsilon))`, epsilon = `{SUITABILITY_EPSILON}`
- Primary suitability: `{PRIMARY_SUITABILITY_METHOD}`

Zero difference는 monotonicity의 increase/decrease count에 포함하지 않았습니다. Missing value는 제거했고, 제거 후 `n < 2`이거나 trendability denominator가 0인 조건은 skipped 처리했습니다.

## Suitability Metric Update

	기존 `M + T`는 `suitability_mt_legacy`와 `suitability_sum_legacy`에 참고값으로 유지했습니다. 본 수정 분석의 baseline은 `M + T + P`이며, primary suitability는 `M`, `T`, `P`의 3항 조화평균인 `suitability_harmonic`으로 설정했습니다.

	`M + T + P`는 세 구성요소의 additive baseline입니다. 반면 harmonic mean은 M, T, P가 모두 높아야 높은 점수를 갖기 때문에 degradation-aware feature를 더 보수적으로 평가합니다. 두 방식의 ranking이 다르면 harmonic ranking을 primary로 해석하고, baseline ranking은 reference로 해석합니다.

## 6. Preprocessing Choices

- Normalization: 계산에는 raw run-level feature value 사용
- Missing value: sequence 내 finite 값만 사용
- Outlier handling: 제거하지 않음
- Segment length: raw segment를 직접 이어붙이지 않고 run-level feature로 요약 후 trend 계산
- Wear progression order: `{context['ordering_column']}` 사용. `time`은 보조 metadata로 기록했습니다.

## 7. Overall Suitability Results

### Top 10 feature x segment averages

{markdown_table(top10)}

### Bottom 10 feature x segment averages

{markdown_table(bottom10)}

전체 상세 결과는 `analysis/H1_S2_suitability_results.csv`와 `analysis/H1_S2_feature_segment_summary.csv`에 저장했습니다.

### Suitability Method Comparison

	#### Baseline 기준 top 10

	{markdown_table(top10_sum[['feature','segment_combination','monotonicity','trendability','prognosability','suitability_baseline','suitability_harmonic','rank_by_baseline','rank_by_harmonic']])}

	#### Harmonic 기준 top 10

	{markdown_table(top10_harmonic[['feature','segment_combination','monotonicity','trendability','prognosability','suitability_baseline','suitability_harmonic','rank_by_baseline','rank_by_harmonic']])}

#### 두 방식에서 공통으로 top에 남은 조합

	{markdown_table(robust_common[['feature','segment_combination','suitability_baseline','suitability_harmonic','rank_by_baseline','rank_by_harmonic']])}

	#### Baseline에서는 높지만 harmonic에서 하락한 sensor-level 조합

	{markdown_table(sum_high_harmonic_drop[['sensor_name','feature_name','segment_setting','monotonicity','trendability','prognosability','suitability_baseline','suitability_harmonic','rank_by_baseline','rank_by_harmonic','rank_change']])}

#### Rank 변화가 큰 sensor-level 조합

	{markdown_table(rank_change_big[['sensor_name','feature_name','segment_setting','monotonicity','trendability','prognosability','suitability_baseline','suitability_harmonic','rank_by_baseline','rank_by_harmonic','rank_change_abs','interpretation']])}

#### M과 T가 균형 있게 높은 조합

	{markdown_table(balanced[['sensor_name','feature_name','segment_setting','monotonicity','trendability','prognosability','suitability_baseline','suitability_harmonic','rank_by_harmonic']])}

## 8. Case-level and Operating-condition-level Results

Case-level 결과는 각 case 내부 run progression에서 먼저 계산한 뒤 평균했습니다. Pair/domain A/B/C 결과는 `analysis/H1_S2_pair_level_suitability_results.csv`에 저장했습니다. 이 방식은 서로 다른 operating condition의 case를 하나의 artificial time series로 이어붙이는 문제를 피합니다.

## 9. Monotonicity vs Trendability Interpretation

### High trendability, low monotonicity

{markdown_table(high_t_low_m[['sensor_name','feature_name','segment_setting','mean_monotonicity','mean_trendability','mean_suitability']])}

### High monotonicity, low trendability

{markdown_table(high_m_low_t[['sensor_name','feature_name','segment_setting','mean_monotonicity','mean_trendability','mean_suitability']])}

Trendability가 높고 monotonicity가 낮은 조합은 전체 방향성은 있으나 run-to-run noise가 큰 후보로 해석할 수 있습니다. 반대로 monotonicity가 높고 trendability가 낮은 조합은 인접 변화 방향은 비교적 일관되지만 선형 wear progression과의 정렬은 약한 후보입니다.

## 10. Segment-wise Interpretation

Steady segment는 안정 절삭 중 load/energy 변화가 누적 wear trend를 드러낼 수 있습니다. Entry/exit 포함 segment는 공구-소재 접촉/이탈 순간의 transient dynamics를 반영하므로 acoustic/vibration feature에서 degradation-aware signal이 나타날 가능성이 있습니다. 다만 suitability는 prediction 성능을 보장하지 않으므로 H1_S1 association 및 H2/H3 prediction 결과와 함께 해석해야 합니다.

## 11. Manufacturing Interpretation

- Current: spindle/motor load 및 절삭 부하 변화와 연결됩니다.
- Vibration: chatter, 불안정 절삭, 공구 상태 변화와 연결됩니다.
- Acoustic: 접촉/이탈 충격, 마찰, 칩 형성, 미세 파손 이벤트와 연결될 수 있습니다.
- Statistics feature는 평균 부하와 amplitude 변화를, shape feature는 impulsiveness/asymmetry를, frequency feature는 spectral energy 및 중심 주파수 이동을 요약합니다.

## 12. Paper Formula Fidelity

본 H1_S2 분석에서는 Monotonicity와 Trendability를 논문 원식과 동일하게 계산하고, case별 start/final feature value로 Prognosability를 추가 계산했다. 논문식 additive baseline은 `suitability_baseline = M + T + P`로 저장했고, primary suitability score는 `M`, `T`, `P`의 3항 조화평균인 `suitability_harmonic`으로 저장했다. 기존 2항 참고값 `M + T`는 `suitability_mt_legacy`와 `suitability_sum_legacy`에 유지했다.

단, 논문과 달리 현재 분석은 NASA Ames milling의 case/run/sensor/segment 구조에 맞춰 case별 sequence에서 먼저 계산한 뒤 평균하는 실무용 적용입니다.

## 13. Limitations

- Suitability가 prediction 성능을 보장하지 않습니다.
- Case 수와 run 수가 작고 operating condition 차이가 큽니다.
- Wear progression order는 `run` 기준이라는 가정에 의존합니다.
- Monotonicity는 run-to-run noise와 outlier에 민감합니다.
- Segment extraction heuristic의 품질이 결과에 영향을 줄 수 있습니다.

## 14. Recommendations

- H1_S1 association 결과와 함께 feature 후보를 좁히십시오.
- H2/H3 예측 실험에서는 suitability 상위 feature/segment를 reduced candidate로 우선 검증하십시오.
- acoustic + entry/exit 계열이 높게 나타나는 경우 confirmatory prediction 및 domain-shift robustness 검증을 권장합니다.
- cumulative descriptor 또는 sensor-wise segment attention 모델에서 segment별 feature weighting prior로 활용할 수 있습니다.

## Execution Path

- experiment_id: `{context.get('experiment_id')}`
- hypothesis_id: `{context.get('hypothesis_id')}`
- scenario_id: `{context.get('scenario_id')}`
- experiment_topic: `{context.get('experiment_topic')}`
- execution_dir: `{context.get('execution_dir')}`
- path_schema_version: `{context.get('path_schema_version')}`
"""
    report_path.write_text(text, encoding="utf-8")
    html = "<!doctype html><html><head><meta charset='utf-8'><title>H1_S2 Report</title></head><body>" + text.replace("\n", "<br>\n") + "</body></html>"
    (output_dir / "reports" / "H1_S2_report.html").write_text(html, encoding="utf-8")


def load_dataset(process_info_path: Path, signal_data_path: Path, heuristic_sequence_path: Path) -> pd.DataFrame:
    process = pd.read_csv(process_info_path)
    signal = pd.read_csv(signal_data_path)
    heuristic = pd.read_csv(heuristic_sequence_path)
    process = process.loc[process["enable"].astype(bool) & process["case"].isin(CASE_SCOPE)].copy()
    signal = signal.loc[signal["enable"].astype(bool) & signal["case"].isin(CASE_SCOPE)].copy()
    heuristic = heuristic.loc[heuristic["case"].isin(CASE_SCOPE)].copy()
    data = process.merge(signal, on=["case", "run"], suffixes=("", "_signal"), validate="one_to_one")
    data = data.merge(
        heuristic[["case", "run", "idx_noload_end", "idx_start", "idx_end", "signal_length", "status"]],
        on=["case", "run"],
        how="inner",
        validate="one_to_one",
    )
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
            rows.append(
                {
                    "sensor_name": col,
                    "sensor_group": SENSOR_GROUPS.get(col, infer_sensor_group(col)),
                    "source_column": col,
                    "used": True,
                    "reason_if_excluded": "",
                }
            )
    return pd.DataFrame(rows)


def build_feature_long_table(data: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        for sensor in sensors:
            try:
                arr = np.nan_to_num(np.asarray(ast.literal_eval(row[sensor]), dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Failed to parse sensor sequence: case={row['case']} run={row['run']} sensor={sensor}") from exc
            segments = segments_for_row(arr, row)
            for segment_setting in SEGMENT_SETTINGS:
                vector = segment_vector(segments, segment_setting)
                values = compute_features(vector)
                for feature_name, feature_value in values.items():
                    rows.append(
                        {
                            "sample_id": row["sample_id"],
                            "dataset_run_id": row["dataset_run_id"],
                            "case_id": int(row["case_id"]),
                            "pair_id": row["pair_id"],
                            "run": int(row["run"]),
                            "time": row.get("time", np.nan),
                            "sensor_name": sensor,
                            "sensor_group": SENSOR_GROUPS.get(sensor, infer_sensor_group(sensor)),
                            "segment_setting": segment_setting,
                            "feature_name": feature_name,
                            "feature_group": FEATURE_GROUPS[feature_name],
                            "feature_value": feature_value,
                        }
                    )
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


def feature_segment_summary(overall: pd.DataFrame) -> pd.DataFrame:
    out = (
        overall.groupby(["experiment_id", "feature_name", "feature_group", "segment_setting"], dropna=False)
        .agg(
            mean_monotonicity=("mean_monotonicity", "mean"),
            mean_trendability=("mean_trendability", "mean"),
            mean_prognosability=("mean_prognosability", "mean"),
            mean_suitability=("mean_suitability", "mean"),
            suitability_mt_legacy=("suitability_mt_legacy", "mean"),
            suitability_sum_legacy=("suitability_sum_legacy", "mean"),
            suitability_baseline=("suitability_baseline", "mean"),
            suitability_harmonic=("suitability_harmonic", "mean"),
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
    out["std_suitability"] = out["std_suitability"].fillna(0.0)
    return add_suitability_ranks(out)


def simplified_feature_segment_table(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.rename(
        columns={
            "feature_name": "feature",
            "segment_setting": "segment_combination",
            "mean_monotonicity": "monotonicity",
            "mean_trendability": "trendability",
            "mean_prognosability": "prognosability",
            "mean_suitability": "suitability",
        }
    )[
        [
            "feature",
            "segment_combination",
            "monotonicity",
            "trendability",
            "prognosability",
            "suitability",
            "suitability_mt_legacy",
            "suitability_sum_legacy",
            "suitability_baseline",
            "suitability_harmonic",
            "suitability_harmonic_mean",
            "primary_suitability_method",
            "suitability_epsilon",
            "rank_by_baseline",
            "rank_by_harmonic",
            "rank_by_sum_legacy",
            "rank_by_harmonic_mean",
            "rank_by_suitability",
            "sample_count",
        ]
    ]


def top_bottom_summary(overall: pd.DataFrame, feature_segment: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(section: str, frame: pd.DataFrame) -> None:
        for _, row in frame.iterrows():
            rows.append({"section": section, **row.to_dict()})

    add("top_10_feature_segment_sensor_combinations", overall.sort_values("mean_suitability", ascending=False).head(10))
    add("bottom_10_feature_segment_sensor_combinations", overall.sort_values("mean_suitability", ascending=True).head(10))
    fs_for_rows = feature_segment.assign(sensor_name="sensor_average", sensor_group="sensor_average")
    add("top_10_feature_segment_average", fs_for_rows.sort_values("mean_suitability", ascending=False).head(10))
    add("bottom_10_feature_segment_average", fs_for_rows.sort_values("mean_suitability", ascending=True).head(10))
    mono_q25 = overall["mean_monotonicity"].quantile(0.25)
    mono_q75 = overall["mean_monotonicity"].quantile(0.75)
    trend_q25 = overall["mean_trendability"].quantile(0.25)
    trend_q75 = overall["mean_trendability"].quantile(0.75)
    high_t_low_m = overall.loc[(overall["mean_trendability"] >= trend_q75) & (overall["mean_monotonicity"] <= mono_q25)].sort_values("mean_trendability", ascending=False).head(10)
    if high_t_low_m.empty:
        high_t_low_m = overall.assign(delta=overall["mean_trendability"] - overall["mean_monotonicity"]).sort_values("delta", ascending=False).head(10).drop(columns="delta")
    high_m_low_t = overall.loc[(overall["mean_monotonicity"] >= mono_q75) & (overall["mean_trendability"] <= trend_q25)].sort_values("mean_monotonicity", ascending=False).head(10)
    if high_m_low_t.empty:
        high_m_low_t = overall.assign(delta=overall["mean_monotonicity"] - overall["mean_trendability"]).sort_values("delta", ascending=False).head(10).drop(columns="delta")
    add("high_trendability_low_monotonicity_combinations", high_t_low_m)
    add("high_monotonicity_low_trendability_combinations", high_m_low_t)
    return pd.DataFrame(rows)


def suitability_method_comparison(overall: pd.DataFrame) -> pd.DataFrame:
    out = overall[
        [
            "experiment_id",
            "sensor_name",
            "sensor_group",
            "feature_name",
            "feature_group",
            "segment_setting",
            "mean_monotonicity",
            "mean_trendability",
            "mean_prognosability",
            "suitability_baseline",
            "suitability_harmonic",
            "suitability_sum_legacy",
            "suitability_harmonic_mean",
            "rank_by_baseline",
            "rank_by_harmonic",
            "rank_by_sum_legacy",
            "rank_by_harmonic_mean",
        ]
    ].rename(columns={"mean_monotonicity": "monotonicity", "mean_trendability": "trendability", "mean_prognosability": "prognosability"}).copy()
    out["rank_change"] = out["rank_by_harmonic"] - out["rank_by_baseline"]
    out["rank_change_abs"] = out["rank_change"].abs()
    out["score_gap_baseline_minus_harmonic"] = out["suitability_baseline"] - out["suitability_harmonic"]
    mono_q75 = out["monotonicity"].quantile(0.75)
    trend_q75 = out["trendability"].quantile(0.75)
    mono_q25 = out["monotonicity"].quantile(0.25)
    trend_q25 = out["trendability"].quantile(0.25)

    def interpret(row: pd.Series) -> str:
        if row["monotonicity"] >= mono_q75 and row["trendability"] >= trend_q75:
            return "balanced_high_m_and_t"
        if row["trendability"] >= trend_q75 and row["monotonicity"] <= mono_q25:
            return "high_trendability_low_monotonicity"
        if row["monotonicity"] >= mono_q75 and row["trendability"] <= trend_q25:
            return "high_monotonicity_low_trendability"
        if row["rank_change"] > 20:
            return "rank_drops_under_harmonic_mean"
        if row["rank_change"] < -20:
            return "rank_improves_under_harmonic_mean"
        return "similar_rank_between_methods"

    out["interpretation"] = out.apply(interpret, axis=1)
    return out.sort_values("rank_by_harmonic").reset_index(drop=True)


def case_pair_consistency_summary(case_results: pd.DataFrame) -> dict[str, Any]:
    ok = case_results.loc[case_results["calculation_status"].eq("ok")].copy()
    best_by_case = (
        ok.groupby(["case_id", "feature_name", "segment_setting"], dropna=False)["suitability"]
        .mean()
        .reset_index()
        .sort_values(["case_id", "suitability"], ascending=[True, False])
        .groupby("case_id")
        .head(1)
    )
    best_by_pair = ok.assign(pair_id=ok["case_id"].map(PAIR_BY_CASE))
    best_by_pair = (
        best_by_pair.groupby(["pair_id", "feature_name", "segment_setting"], dropna=False)["suitability"]
        .mean()
        .reset_index()
        .sort_values(["pair_id", "suitability"], ascending=[True, False])
        .groupby("pair_id")
        .head(1)
    )
    return {
        "best_by_case": best_by_case.to_dict(orient="records"),
        "best_by_pair": best_by_pair.to_dict(orient="records"),
    }


def _heatmap(data: pd.DataFrame, title: str, path: Path, vmax: float | None = 2, label: str = "Suitability") -> None:
    plt.figure(figsize=(11, max(4, 0.42 * len(data.index) + 1.5)))
    if sns is not None:
        sns.heatmap(data, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=vmax, linewidths=0.5, cbar_kws={"label": label})
    else:
        matrix = data.to_numpy(dtype="float64")
        image = plt.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        plt.colorbar(image, label=label)
        plt.xticks(range(len(data.columns)), data.columns, rotation=35, ha="right")
        plt.yticks(range(len(data.index)), data.index)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isfinite(value):
                    threshold = (vmax or np.nanmax(matrix)) * 0.5
                    plt.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value > threshold else "black", fontsize=7)
    plt.title(title)
    plt.xlabel("Segment setting")
    plt.ylabel(data.index.name or "")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    header = "| " + " | ".join(map(str, out.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in out.astype(str).to_numpy()]
    return "\n".join([header, separator, *rows])


def infer_sensor_group(sensor: str) -> str:
    lower = sensor.lower()
    if "smc" in lower or "current" in lower:
        return "Current"
    if "vib" in lower:
        return "Vibration"
    if "ae" in lower or "acoustic" in lower:
        return "Acoustic"
    return "Unknown"


def feature_definition_text(feature: str) -> str:
    return {
        "mean": "run-level arithmetic mean",
        "std": "run-level standard deviation",
        "max": "maximum value",
        "min": "minimum value",
        "peak_to_peak": "max - min",
        "kurtosis": "Fisher kurtosis",
        "skewness": "sample skewness",
        "spectral_centroid": "FFT magnitude weighted frequency centroid",
        "band_energy": "sum squared FFT magnitude divided by segment length",
    }.get(feature, "")


def segment_definition_text(segment: str) -> str:
    return {
        "full_length": "entire signal from index 0 to signal length",
        "entry": "idx_noload_end to idx_start",
        "steady": "idx_start to idx_end",
        "exit": "idx_end to signal length",
        "entry_steady": "entry concatenated with steady",
        "entry_exit": "entry concatenated with exit",
        "steady_exit": "steady concatenated with exit",
        "entry_steady_exit": "entry + steady + exit concatenated; no-load excluded",
    }.get(segment, "")


def make_dirs(paths: ExperimentPaths) -> None:
    paths.prepare_standard_dirs()


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_H1_S2_feature_VB_segment_suitability_eda")
    root = Path(args.root).resolve()
    paths = ExperimentPaths(root, experiment_id)
    make_dirs(paths)
    run_log = paths.execution_dir / "logs" / "H1_S2_run.log"
    error_log = paths.execution_dir / "logs" / "H1_S2_error.log"

    def log(message: str) -> None:
        with run_log.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")

    process_info_path = Path(args.process_info_path)
    signal_data_path = Path(args.signal_data_path)
    heuristic_sequence_path = Path(args.heuristic_sequence_path)
    if not process_info_path.is_absolute():
        process_info_path = root / process_info_path
    if not signal_data_path.is_absolute():
        signal_data_path = root / signal_data_path
    if not heuristic_sequence_path.is_absolute():
        heuristic_sequence_path = root / heuristic_sequence_path

    try:
        log(f"H1_S2 started: {experiment_id}")
        config = {
            "experiment": {
                "experiment_id": experiment_id,
                "experiment_name": "H1_S2_feature_VB_segment_suitability_eda",
            },
            "experiment_name": "H1_S2_feature_VB_segment_suitability_eda",
            "experiment_id": experiment_id,
            "analysis_type": "Exploratory / Feature quality analysis / Degradation-awareness analysis",
            "process_info_path": str(process_info_path),
            "signal_data_path": str(signal_data_path),
            "heuristic_sequence_path": str(heuristic_sequence_path),
            "case_scope": CASE_SCOPE,
            "pair_definition": PAIR_DEFINITION,
            "segment_settings": SEGMENT_SETTINGS,
            "features": TARGET_FEATURES,
            "analysis": {
                "primary_suitability_method": PRIMARY_SUITABILITY_METHOD,
                "compute_legacy_sum": COMPUTE_LEGACY_SUM,
                "suitability_epsilon": SUITABILITY_EPSILON,
                "prognosability_epsilon": PROGNOSABILITY_EPSILON,
            },
            "formula": {
                "monotonicity": "abs((N_inc - N_dec)/(n-1))",
                "trendability": "abs(Pearson(x,t))",
                "prognosability": "exp(-var(x_final)/(mean(abs(x_start-x_final))+epsilon))",
                "suitability_mt_legacy": "M + T",
                "suitability_baseline": "M + T + P",
                "suitability_harmonic": "3/(1/(M+epsilon)+1/(T+epsilon)+1/(P+epsilon))",
                "primary_suitability": PRIMARY_SUITABILITY_METHOD,
            },
            "framework_version": __version__,
            "schema_version": SCHEMA_VERSION,
        }
        paths.apply_to_config(config)
        config["config_hash"] = stable_hash(config)
        write_yaml(paths.execution_dir / "configs" / "H1_S2_input_config.yaml", config)
        write_yaml(paths.execution_dir / "configs" / "H1_S2_resolved_config.yaml", config)

        data = load_dataset(process_info_path, signal_data_path, heuristic_sequence_path)
        sensors_df = identify_sensors(data)
        sensors = sensors_df.loc[sensors_df["used"], "sensor_name"].tolist()
        log(f"loaded data rows={len(data)} sensors={sensors}")
        feature_long = build_feature_long_table(data, sensors)
        ordering_column = "run" if "run" in feature_long.columns else "time"
        feature_definition = identify_feature_columns(feature_long, TARGET_FEATURES)
        segment_definition = identify_segment_settings(feature_long)
        case_results = compute_case_level_suitability(feature_long, experiment_id, ordering_column, PRIMARY_SUITABILITY_METHOD, SUITABILITY_EPSILON)
        overall = aggregate_overall_suitability(case_results)
        pair_level = aggregate_pair_level_suitability(case_results, PAIR_DEFINITION)
        fs_summary = feature_segment_summary(overall)
        simplified = simplified_feature_segment_table(fs_summary)
        method_comparison = suitability_method_comparison(overall)
        top_bottom = top_bottom_summary(overall, fs_summary)

        no_load_rows = int(feature_long["segment_setting"].astype(str).str.contains("no_load", case=False, na=False).sum())
        zero_denominator_count = int(case_results["calculation_status"].eq("skipped_zero_trendability_denominator").sum())
        validation = {
            "ok": True,
            "data_files_exist": all(path.exists() for path in [process_info_path, signal_data_path, heuristic_sequence_path]),
            "case_scope_found": sorted(data["case_id"].unique().tolist()),
            "feature_definition_all_found": bool(feature_definition["found"].all()),
            "segments_used": SEGMENT_SETTINGS,
            "no_load_rows_excluded": no_load_rows,
            "case_id_exists": "case_id" in data.columns,
            "ordering_column": ordering_column,
            "sequence_length_min": int(case_results["sample_count"].min()),
            "sequence_length_lt_2_count": int(case_results["calculation_status"].eq("skipped_n_less_than_2").sum()),
            "trendability_zero_denominator_count": zero_denominator_count,
            "missing_uncomputable_count": int(case_results["calculation_status"].ne("ok").sum()),
            "primary_suitability_method": PRIMARY_SUITABILITY_METHOD,
            "suitability_epsilon": SUITABILITY_EPSILON,
            "suitability_in_range": bool(overall["mean_suitability"].between(0, 1).all()) if PRIMARY_SUITABILITY_METHOD == "harmonic_mean" else bool(overall["mean_suitability"].between(0, 3).all()),
            "suitability_sum_legacy_in_range": bool(overall["suitability_sum_legacy"].between(0, 2).all()),
            "suitability_baseline_in_range": bool(overall["suitability_baseline"].between(0, 3).all()),
            "suitability_harmonic_in_range": bool(overall["suitability_harmonic"].between(0, 1).all()),
        }
        if not validation["feature_definition_all_found"]:
            validation["ok"] = False
        write_json(paths.execution_dir / "configs" / "H1_S2_config_validation.json", validation)

        ordering_summary = {
            "ordering_column": ordering_column,
            "fallback_order": ["dataset_run_id/run index", "time", "row order"],
            "reason": "numeric run column is available and represents machining run order within each case",
            "time_column_available": "time" in data.columns,
            "vb_used_as_time_index": False,
            "rows_by_case": data.groupby("case_id").size().to_dict(),
            "run_range_by_case": data.groupby("case_id")["run"].agg(["min", "max"]).reset_index().to_dict(orient="records"),
        }
        write_json(paths.execution_dir / "data" / "H1_S2_ordering_summary.json", ordering_summary)
        dataset_summary = {
            "process_info_path": str(process_info_path),
            "signal_data_path": str(signal_data_path),
            "heuristic_sequence_path": str(heuristic_sequence_path),
            "num_runs": int(len(data)),
            "case_scope": CASE_SCOPE,
            "rows_by_case": data.groupby("case_id").size().to_dict(),
            "sensors": sensors,
            "num_feature_long_rows": int(len(feature_long)),
            "num_case_level_rows": int(len(case_results)),
            "num_case_level_ok": int(case_results["calculation_status"].eq("ok").sum()),
        }
        write_json(paths.execution_dir / "data" / "H1_S2_dataset_summary.json", dataset_summary)
        write_csv(paths.execution_dir / "data" / "H1_S2_feature_definition.csv", feature_definition)
        write_csv(paths.execution_dir / "data" / "H1_S2_segment_definition.csv", segment_definition)
        write_csv(paths.execution_dir / "data" / "H1_S2_sensor_mapping.csv", sensors_df)
        write_csv(paths.execution_dir / "analysis" / "H1_S2_case_level_suitability_results.csv", case_results)
        write_csv(paths.execution_dir / "analysis" / "H1_S2_suitability_results.csv", overall)
        write_csv(paths.execution_dir / "analysis" / "H1_S2_pair_level_suitability_results.csv", pair_level)
        write_csv(paths.execution_dir / "analysis" / "H1_S2_feature_segment_summary.csv", fs_summary)
        write_csv(paths.execution_dir / "analysis" / "H1_S2_feature_segment_simplified_table.csv", simplified)
        write_csv(paths.execution_dir / "analysis" / "H1_S2_suitability_method_comparison.csv", method_comparison)
        write_csv(paths.execution_dir / "analysis" / "H1_S2_top_bottom_suitability_summary.csv", top_bottom)

        analysis_summary = {
            "top_feature_segment": simplified.sort_values("suitability", ascending=False).head(10).to_dict(orient="records"),
            "bottom_feature_segment": simplified.sort_values("suitability", ascending=True).head(10).to_dict(orient="records"),
            "top_feature_segment_sum_legacy": simplified.sort_values("suitability_sum_legacy", ascending=False).head(10).to_dict(orient="records"),
            "top_feature_segment_harmonic_mean": simplified.sort_values("suitability_harmonic_mean", ascending=False).head(10).to_dict(orient="records"),
            "top_sensor_feature_segment": overall.head(10).to_dict(orient="records"),
            "largest_rank_changes": method_comparison.sort_values("rank_change_abs", ascending=False).head(10).to_dict(orient="records"),
            "case_pair_consistency": case_pair_consistency_summary(case_results),
            "status_counts": case_results["calculation_status"].value_counts().to_dict(),
        }
        write_json(paths.execution_dir / "analysis" / "H1_S2_analysis_summary.json", analysis_summary)

        create_suitability_heatmaps(overall, fs_summary, pair_level, method_comparison, paths.execution_dir / "figures")

        context = {
            **dataset_summary,
            "process_info_path": str(process_info_path),
            "signal_data_path": str(signal_data_path),
            "heuristic_sequence_path": str(heuristic_sequence_path),
            "no_load_excluded_rows": no_load_rows,
            "ordering_column": ordering_column,
            **paths.path_metadata,
        }
        write_H1_S2_report(paths.execution_dir, context, feature_definition, segment_definition, overall, fs_summary, simplified, pair_level, top_bottom)
        write_json(paths.execution_dir / "logs" / "H1_S2_environment.json", collect_environment())
        error_log.touch()
        log(f"H1_S2 finished: {experiment_id}")
        return {"experiment_id": experiment_id, "execution_dir": str(paths.execution_dir), "summary": analysis_summary}
    except Exception as exc:
        error_log.write_text(traceback.format_exc(), encoding="utf-8")
        log(f"ERROR: {exc}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H1_S2 feature_VB_segment_suitability_eda.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--process-info-path", default="datasets/processed/mill_process_info_enabled.csv")
    parser.add_argument("--signal-data-path", default="datasets/processed/mill_signal_data_enabled.csv")
    parser.add_argument("--heuristic-sequence-path", default="datasets/metadata/heuristic_sequence.csv")
    return parser.parse_args()


def main() -> None:
    result = run_analysis(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

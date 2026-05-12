#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.experiments.execution_path import create_execution_dir, find_experiment_dirs


PREFIX = "H1_S5"
FEATURES = ["mean", "std", "max", "min", "peak_to_peak", "kurtosis", "skewness", "spectral_centroid", "band_energy"]
SEGMENTS = ["full_length", "steady", "entry", "exit", "entry_steady", "entry_exit", "steady_exit"]
SENSOR_GROUPS = {
    "Current": ["smcAC", "smcDC"],
    "Vibration": ["vib_spindle", "vib_table"],
    "Acoustic": ["AE_spindle", "AE_table"],
}
FEATURE_GROUPS = {
    "statistics": ["mean", "std", "max", "min", "peak_to_peak"],
    "shape": ["kurtosis", "skewness"],
    "frequency": ["spectral_centroid", "band_energy"],
}
CATEGORY_CODES = {
    "low_association_low_suitability": 0,
    "high_association_low_suitability": 1,
    "low_association_high_suitability": 2,
    "middle_or_mixed": 3,
    "high_association_high_suitability": 4,
}


def sensor_group(sensor: str) -> str:
    for group, sensors in SENSOR_GROUPS.items():
        if sensor in sensors:
            return group
    return "Unknown"


def feature_group(feature: str) -> str:
    for group, features in FEATURE_GROUPS.items():
        if feature in features:
            return group
    return "unknown"


def find_source_experiment_dirs(root: Path, hypothesis_id: str, scenario_id: str, keyword: str | None = None) -> list[Path]:
    dirs = [Path(p) for p in find_experiment_dirs(root, hypothesis_id=hypothesis_id, scenario_id=scenario_id, keyword=keyword, include_legacy=True)]
    return sorted([p for p in dirs if p.is_dir()])


def first_existing(base: Path, candidates: list[str]) -> Path | None:
    for rel in candidates:
        path = base / rel
        if path.exists():
            return path
    return None


def glob_first(base: Path, pattern: str) -> Path | None:
    matches = sorted(base.glob(pattern))
    return matches[0] if matches else None


def load_H1_S1_association_results(path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    files = {
        "feature_level": first_existing(path, ["metrics/H1_S1_association_feature_level.csv", "metrics/association_feature_level.csv"]),
        "segment_level": first_existing(path, ["metrics/H1_S1_association_segment_level.csv", "metrics/association_segment_level.csv"]),
        "case_level": first_existing(path, ["metrics/H1_S1_association_case_level.csv", "metrics/association_case_level.csv"]),
        "pair_level": first_existing(path, ["metrics/H1_S1_association_pair_level.csv", "metrics/association_pair_level.csv", "analysis/pair_level_association_summary.csv"]),
        "report": first_existing(path, ["reports/H1_S1_report.md", "reports/report.md"]),
    }
    if files["feature_level"] is None:
        raise FileNotFoundError(f"H1_S1 feature-level association table not found under {path}")
    tables = {key: pd.read_csv(value) for key, value in files.items() if value is not None and value.suffix == ".csv"}
    return tables, {key: str(value) for key, value in files.items() if value is not None}


def load_H1_S4_suitability_results(path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    files = {
        "suitability": first_existing(path, ["analysis/H1_S4_suitability_results.csv", "analysis/suitability_results.csv"]),
        "feature_segment": first_existing(path, ["analysis/H1_S4_feature_segment_summary.csv", "analysis/feature_segment_summary.csv"]),
        "pair_level": first_existing(path, ["analysis/H1_S4_pair_level_suitability_results.csv", "analysis/pair_level_suitability_results.csv"]),
        "method_comparison": first_existing(path, ["analysis/H1_S4_suitability_method_comparison.csv", "analysis/suitability_method_comparison.csv"]),
        "report": first_existing(path, ["reports/H1_S4_report.md", "reports/report.md"]),
    }
    if files["suitability"] is None:
        raise FileNotFoundError(f"H1_S4 suitability table not found under {path}")
    tables = {key: pd.read_csv(value) for key, value in files.items() if value is not None and value.suffix == ".csv"}
    return tables, {key: str(value) for key, value in files.items() if value is not None}


def _top3_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    if values.empty:
        return float("nan")
    return float(values.head(3).mean())


def standardize_association_table(df: pd.DataFrame, source_experiment_id: str) -> pd.DataFrame:
    data = df.copy()
    data = data[data["segment_setting"].isin(SEGMENTS)]
    data = data[data["feature_name"].isin(FEATURES)]
    data = data[~data["segment_setting"].astype(str).str.contains("no_load|no-load", case=False, na=False)]
    grouped = data.groupby(["sensor_name", "segment_setting", "feature_name"], dropna=False)
    out = grouped.agg(
        top3_mean_abs_spearman=("spearman_abs", _top3_mean),
        top3_mean_abs_pearson=("pearson_abs", _top3_mean),
        top3_mean_mutual_information=("mutual_information", _top3_mean),
        mean_abs_spearman=("spearman_abs", "mean"),
        mean_abs_pearson=("pearson_abs", "mean"),
        mean_mutual_information=("mutual_information", "mean"),
        sample_count=("n_samples", "sum"),
    ).reset_index()
    out["experiment_id"] = np.nan
    if "experiment_id" in data.columns and len(data):
        out["experiment_id"] = data["experiment_id"].iloc[0]
    out["source_experiment_id"] = source_experiment_id
    out["sensor_group"] = out["sensor_name"].map(sensor_group)
    out["feature_group"] = out["feature_name"].map(feature_group)
    out["primary_association_score"] = out["top3_mean_abs_spearman"]
    out["aggregation_level"] = "case_mean_by_sensor_feature_segment"
    cols = [
        "experiment_id",
        "source_experiment_id",
        "sensor_name",
        "sensor_group",
        "feature_name",
        "feature_group",
        "segment_setting",
        "primary_association_score",
        "top3_mean_abs_spearman",
        "top3_mean_abs_pearson",
        "top3_mean_mutual_information",
        "mean_abs_spearman",
        "mean_abs_pearson",
        "mean_mutual_information",
        "sample_count",
        "aggregation_level",
    ]
    return out[cols].sort_values("primary_association_score", ascending=False).reset_index(drop=True)


def standardize_suitability_table(df: pd.DataFrame, source_experiment_id: str) -> pd.DataFrame:
    data = df.copy()
    data = data[data["segment_setting"].isin(SEGMENTS)]
    data = data[data["feature_name"].isin(FEATURES)]
    data = data[~data["segment_setting"].astype(str).str.contains("no_load|no-load", case=False, na=False)]
    if "suitability_harmonic_mean" not in data.columns and "mean_suitability" in data.columns:
        data["suitability_harmonic_mean"] = data["mean_suitability"]
    if "suitability_sum_legacy" not in data.columns:
        data["suitability_sum_legacy"] = data.get("mean_monotonicity", np.nan) + data.get("mean_trendability", np.nan)
    out = data.rename(
        columns={
            "mean_monotonicity": "monotonicity",
            "mean_trendability": "trendability",
            "total_sample_count": "sample_count",
        }
    ).copy()
    out["source_experiment_id"] = source_experiment_id
    out["sensor_group"] = out["sensor_group"].fillna(out["sensor_name"].map(sensor_group)) if "sensor_group" in out else out["sensor_name"].map(sensor_group)
    out["feature_group"] = out["feature_group"].fillna(out["feature_name"].map(feature_group)) if "feature_group" in out else out["feature_name"].map(feature_group)
    out["primary_suitability_score"] = out["suitability_harmonic_mean"]
    out["aggregation_level"] = "case_mean_by_sensor_feature_segment"
    cols = [
        "experiment_id",
        "source_experiment_id",
        "sensor_name",
        "sensor_group",
        "feature_name",
        "feature_group",
        "segment_setting",
        "primary_suitability_score",
        "suitability_harmonic_mean",
        "suitability_sum_legacy",
        "monotonicity",
        "trendability",
        "sample_count",
        "aggregation_level",
    ]
    return out[cols].sort_values("primary_suitability_score", ascending=False).reset_index(drop=True)


def metric_agreement(association: pd.DataFrame) -> pd.DataFrame:
    df = association.copy()
    metrics = {
        "spearman": "top3_mean_abs_spearman",
        "pearson": "top3_mean_abs_pearson",
        "mutual_information": "top3_mean_mutual_information",
    }
    for name, col in metrics.items():
        df[f"rank_{name}"] = df[col].rank(ascending=False, method="average")
    rows = []
    pairs = [("spearman", "pearson"), ("spearman", "mutual_information"), ("pearson", "mutual_information")]
    for a, b in pairs:
        valid = df[[metrics[a], metrics[b], f"rank_{a}", f"rank_{b}"]].dropna()
        rank_corr = _safe_corr(valid[f"rank_{a}"], valid[f"rank_{b}"], method="spearman")
        score_corr = _safe_corr(valid[metrics[a]], valid[metrics[b]], method="pearson")
        interp = "high agreement" if abs(rank_corr) >= 0.7 else "moderate agreement" if abs(rank_corr) >= 0.4 else "weak agreement"
        rows.append({"metric_pair": f"{a}_vs_{b}", "rank_correlation": rank_corr, "score_correlation": score_corr, "interpretation": interp})
    return pd.DataFrame(rows)


def join_association_and_suitability(association: pd.DataFrame, suitability: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    joined = association.merge(
        suitability,
        on=["sensor_name", "sensor_group", "feature_name", "feature_group", "segment_setting"],
        how="inner",
        suffixes=("_association", "_suitability"),
    )
    joined["experiment_id"] = experiment_id
    joined["association_rank"] = joined["primary_association_score"].rank(ascending=False, method="average").astype(float)
    joined["suitability_rank"] = joined["primary_suitability_score"].rank(ascending=False, method="average").astype(float)
    joined["rank_difference"] = joined["association_rank"] - joined["suitability_rank"]
    joined["rank_difference_abs"] = joined["rank_difference"].abs()
    joined = compute_alignment_categories(joined)
    joined = compute_combined_robust_score(joined)
    joined["interpretation"] = joined["alignment_category"].map(
        {
            "high_association_high_suitability": "target-relevant and degradation-aware robust candidate",
            "high_association_low_suitability": "VB-related but weak progression trend; inspect condition dependence",
            "low_association_high_suitability": "degradation-like trend but weak VB relation; possible time/order proxy",
            "low_association_low_suitability": "low priority under both EDA views",
            "middle_or_mixed": "mixed or middle-ranked evidence",
        }
    )
    return joined.sort_values("combined_score", ascending=False).reset_index(drop=True)


def compute_alignment_categories(joined: pd.DataFrame) -> pd.DataFrame:
    out = joined.copy()
    assoc_high = out["primary_association_score"].quantile(0.75)
    assoc_low = out["primary_association_score"].quantile(0.25)
    suit_high = out["primary_suitability_score"].quantile(0.75)
    suit_low = out["primary_suitability_score"].quantile(0.25)
    conditions = [
        (out["primary_association_score"] >= assoc_high) & (out["primary_suitability_score"] >= suit_high),
        (out["primary_association_score"] >= assoc_high) & (out["primary_suitability_score"] <= suit_low),
        (out["primary_association_score"] <= assoc_low) & (out["primary_suitability_score"] >= suit_high),
        (out["primary_association_score"] <= assoc_low) & (out["primary_suitability_score"] <= suit_low),
    ]
    choices = [
        "high_association_high_suitability",
        "high_association_low_suitability",
        "low_association_high_suitability",
        "low_association_low_suitability",
    ]
    out["alignment_category"] = np.select(conditions, choices, default="middle_or_mixed")
    return out


def _minmax(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    lo, hi = series.min(), series.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def compute_combined_robust_score(joined: pd.DataFrame) -> pd.DataFrame:
    out = joined.copy()
    out["normalized_association_score"] = _minmax(out["primary_association_score"])
    out["normalized_suitability_score"] = _minmax(out["primary_suitability_score"])
    denom = out["normalized_association_score"] + out["normalized_suitability_score"]
    out["combined_score"] = np.where(denom > 0, 2 * out["normalized_association_score"] * out["normalized_suitability_score"] / (denom + 1e-12), 0.0)
    return out


def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    valid = pd.concat([a, b], axis=1).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)
    if method == "spearman":
        return float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)
    if method == "kendall":
        return float(kendalltau(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)
    raise ValueError(method)


def compute_cross_score_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    levels = {
        "sensor_feature_segment": joined,
        "feature_segment_mean": joined.groupby(["feature_name", "feature_group", "segment_setting"], as_index=False)[["primary_association_score", "primary_suitability_score"]].mean(),
        "sensor_group_segment_mean": joined.groupby(["sensor_group", "segment_setting"], as_index=False)[["primary_association_score", "primary_suitability_score"]].mean(),
    }
    rows = []
    for level, df in levels.items():
        rows.append(
            {
                "aggregation_level": level,
                "association_score": "top3_mean_abs_spearman",
                "suitability_score": "suitability_harmonic_mean",
                "pearson_corr": _safe_corr(df["primary_association_score"], df["primary_suitability_score"], "pearson"),
                "spearman_corr": _safe_corr(df["primary_association_score"], df["primary_suitability_score"], "spearman"),
                "kendall_tau": _safe_corr(df["primary_association_score"], df["primary_suitability_score"], "kendall"),
                "n_combinations": int(len(df)),
            }
        )
    for row in rows:
        corr = row["spearman_corr"]
        row["interpretation"] = "aligned rankings" if abs(corr) >= 0.5 else "weak or condition-dependent alignment"
    return pd.DataFrame(rows)


def alignment_category_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(joined)
    for category, group in joined.groupby("alignment_category"):
        reps = group.sort_values("combined_score", ascending=False).head(5)
        rep_text = "; ".join(f"{r.sensor_name}/{r.feature_name}/{r.segment_setting}" for r in reps.itertuples())
        rows.append(
            {
                "alignment_category": category,
                "num_combinations": int(len(group)),
                "ratio": float(len(group) / total) if total else 0.0,
                "representative_feature_segment_sensor_combinations": rep_text,
                "interpretation": group["interpretation"].iloc[0] if "interpretation" in group else "",
            }
        )
    return pd.DataFrame(rows).sort_values("num_combinations", ascending=False)


def manufacturing_interpretation(row: pd.Series) -> str:
    sensor = row["sensor_group"]
    segment = row["segment_setting"]
    feature = row["feature_name"]
    base = {
        "Acoustic": "acoustic response may capture contact/exit impacts, friction, chip events, or micro-fracture signatures",
        "Vibration": "vibration response may reflect dynamics, chatter, unstable cutting, and tool-condition changes",
        "Current": "current response may reflect cutting load and motor/spindle load changes",
    }.get(sensor, "sensor response requires domain review")
    seg_note = "entry/exit transient segment" if any(x in segment for x in ["entry", "exit"]) else "steady/full cut segment"
    return f"{base}; {seg_note}; {feature} is selected by both association and suitability evidence"


def robust_candidates(joined: pd.DataFrame) -> pd.DataFrame:
    robust = joined[joined["alignment_category"].eq("high_association_high_suitability")].copy()
    if robust.empty:
        robust = joined.sort_values("combined_score", ascending=False).head(10).copy()
    else:
        robust = robust.sort_values("combined_score", ascending=False).head(30)
    robust["reason_selected"] = np.where(
        robust["alignment_category"].eq("high_association_high_suitability"),
        "top quartile in both primary association and harmonic suitability",
        "selected by combined robust score",
    )
    robust["manufacturing_interpretation"] = robust.apply(manufacturing_interpretation, axis=1)
    cols = [
        "sensor_name",
        "sensor_group",
        "feature_name",
        "feature_group",
        "segment_setting",
        "primary_association_score",
        "primary_suitability_score",
        "combined_score",
        "association_rank",
        "suitability_rank",
        "reason_selected",
        "manufacturing_interpretation",
    ]
    return robust[cols]


def disagreement_table(joined: pd.DataFrame) -> pd.DataFrame:
    data = joined[joined["alignment_category"].isin(["high_association_low_suitability", "low_association_high_suitability"])].copy()
    data["disagreement_type"] = data["alignment_category"].map(
        {
            "high_association_low_suitability": "association_high_suitability_low",
            "low_association_high_suitability": "association_low_suitability_high",
        }
    )
    data["possible_reason"] = data["disagreement_type"].map(
        {
            "association_high_suitability_low": "VB relation may be non-monotonic, case-specific, or condition-dependent",
            "association_low_suitability_high": "feature may track time/order but weakly explain VB variation",
        }
    )
    data["recommendation"] = data["disagreement_type"].map(
        {
            "association_high_suitability_low": "keep as exploratory predictor but validate domain-shift robustness",
            "association_low_suitability_high": "use cautiously; combine with target association or process checks",
        }
    )
    cols = [
        "sensor_name",
        "sensor_group",
        "feature_name",
        "feature_group",
        "segment_setting",
        "primary_association_score",
        "primary_suitability_score",
        "association_rank",
        "suitability_rank",
        "disagreement_type",
        "possible_reason",
        "recommendation",
    ]
    return data.sort_values("rank_difference_abs", ascending=False)[cols]


def save_heatmap(df: pd.DataFrame, index: str, columns: str, values: str, path: Path, title: str, cmap: str = "viridis", annot: bool = True) -> None:
    pivot = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    pivot = pivot.reindex(index=[x for x in FEATURES + list(SENSOR_GROUPS) if x in pivot.index] or None)
    pivot = pivot.reindex(columns=[x for x in SEGMENTS if x in pivot.columns])
    plt.figure(figsize=(max(8, len(pivot.columns) * 1.2), max(4, len(pivot.index) * 0.45)))
    sns.heatmap(pivot, cmap=cmap, annot=annot, fmt=".2f", linewidths=0.3, cbar_kws={"label": values})
    plt.title(title)
    plt.xlabel(columns)
    plt.ylabel(index)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def create_H1_S1_aligned_visualizations(association: pd.DataFrame, pair_df: pd.DataFrame | None, metric_summary: pd.DataFrame, figures_dir: Path) -> list[str]:
    figures = []
    save_heatmap(
        association,
        "feature_name",
        "segment_setting",
        "primary_association_score",
        figures_dir / f"{PREFIX}_H1_S1_feature_segment_association_heatmap.png",
        "H1_S1 feature x segment association (top3 mean abs Spearman)",
    )
    figures.append(f"{PREFIX}_H1_S1_feature_segment_association_heatmap.png")
    save_heatmap(
        association,
        "sensor_name",
        "segment_setting",
        "primary_association_score",
        figures_dir / f"{PREFIX}_H1_S1_sensor_segment_association_heatmap.png",
        "H1_S1 sensor x segment association",
    )
    figures.append(f"{PREFIX}_H1_S1_sensor_segment_association_heatmap.png")
    save_heatmap(
        association,
        "sensor_group",
        "segment_setting",
        "primary_association_score",
        figures_dir / f"{PREFIX}_H1_S1_sensor_group_segment_association_heatmap.png",
        "H1_S1 sensor group x segment association",
    )
    figures.append(f"{PREFIX}_H1_S1_sensor_group_segment_association_heatmap.png")
    if pair_df is not None and not pair_df.empty:
        pair = pair_df.copy()
        pair = pair[pair["segment_setting"].isin(SEGMENTS)]
        if "mean_top3_abs_spearman" in pair.columns:
            pair["primary_association_score"] = pair["mean_top3_abs_spearman"]
            save_heatmap(
                pair,
                "pair_id",
                "segment_setting",
                "primary_association_score",
                figures_dir / f"{PREFIX}_H1_S1_pair_segment_association_heatmap.png",
                "H1_S1 pair x segment association",
            )
            figures.append(f"{PREFIX}_H1_S1_pair_segment_association_heatmap.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=association, x="top3_mean_abs_spearman", y="top3_mean_abs_pearson", hue="segment_setting", ax=axes[0], s=35)
    axes[0].set_title("Spearman vs Pearson")
    sns.scatterplot(data=association, x="top3_mean_abs_spearman", y="top3_mean_mutual_information", hue="segment_setting", ax=axes[1], s=35, legend=False)
    axes[1].set_title("Spearman vs MI")
    for ax in axes:
        ax.set_xlabel("top3 mean abs Spearman")
    plt.suptitle("H1_S1 association metric agreement")
    plt.tight_layout()
    out = figures_dir / f"{PREFIX}_H1_S1_association_metric_agreement.png"
    plt.savefig(out, dpi=180)
    plt.close()
    figures.append(out.name)
    return figures


def copy_or_regenerate_h1_s4_figures(h1s4_dir: Path, suitability: pd.DataFrame, pair_suitability: pd.DataFrame | None, figures_dir: Path) -> tuple[list[str], list[str]]:
    copied, skipped = [], []
    mapping = {
        "H1_S4_feature_segment_suitability_heatmap.png": f"{PREFIX}_H1_S4_feature_segment_suitability_heatmap.png",
        "H1_S4_sensor_segment_suitability_heatmap.png": f"{PREFIX}_H1_S4_sensor_segment_suitability_heatmap.png",
        "H1_S4_sensor_group_segment_suitability_heatmap.png": f"{PREFIX}_H1_S4_sensor_group_segment_suitability_heatmap.png",
        "H1_S4_pair_segment_suitability_heatmap.png": f"{PREFIX}_H1_S4_pair_segment_suitability_heatmap.png",
        "H1_S4_monotonicity_vs_trendability_scatter.png": f"{PREFIX}_H1_S4_monotonicity_vs_trendability_scatter.png",
    }
    for src_name, dst_name in mapping.items():
        src = h1s4_dir / "figures" / src_name
        dst = figures_dir / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(dst_name)
        else:
            skipped.append(src_name)
    if f"{PREFIX}_H1_S4_feature_segment_suitability_heatmap.png" not in copied:
        save_heatmap(suitability, "feature_name", "segment_setting", "primary_suitability_score", figures_dir / f"{PREFIX}_H1_S4_feature_segment_suitability_heatmap.png", "H1_S4 feature x segment suitability")
        copied.append(f"{PREFIX}_H1_S4_feature_segment_suitability_heatmap.png")
    if f"{PREFIX}_H1_S4_sensor_segment_suitability_heatmap.png" not in copied:
        save_heatmap(suitability, "sensor_name", "segment_setting", "primary_suitability_score", figures_dir / f"{PREFIX}_H1_S4_sensor_segment_suitability_heatmap.png", "H1_S4 sensor x segment suitability")
        copied.append(f"{PREFIX}_H1_S4_sensor_segment_suitability_heatmap.png")
    if f"{PREFIX}_H1_S4_sensor_group_segment_suitability_heatmap.png" not in copied:
        save_heatmap(suitability, "sensor_group", "segment_setting", "primary_suitability_score", figures_dir / f"{PREFIX}_H1_S4_sensor_group_segment_suitability_heatmap.png", "H1_S4 sensor group x segment suitability")
        copied.append(f"{PREFIX}_H1_S4_sensor_group_segment_suitability_heatmap.png")
    if pair_suitability is not None and not pair_suitability.empty and f"{PREFIX}_H1_S4_pair_segment_suitability_heatmap.png" not in copied:
        p = pair_suitability.rename(columns={"mean_suitability": "primary_suitability_score"})
        if "suitability_harmonic_mean" in p:
            p["primary_suitability_score"] = p["suitability_harmonic_mean"]
        save_heatmap(p, "pair_id", "segment_setting", "primary_suitability_score", figures_dir / f"{PREFIX}_H1_S4_pair_segment_suitability_heatmap.png", "H1_S4 pair x segment suitability")
        copied.append(f"{PREFIX}_H1_S4_pair_segment_suitability_heatmap.png")
    return sorted(set(copied)), skipped


def create_cross_analysis_figures(joined: pd.DataFrame, figures_dir: Path) -> list[str]:
    figures = []
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=joined, x="primary_association_score", y="primary_suitability_score", hue="segment_setting", style="sensor_group", s=45)
    plt.title("Association vs harmonic suitability")
    plt.xlabel("H1_S1 primary association (top3 abs Spearman)")
    plt.ylabel("H1_S4 primary suitability (harmonic mean)")
    plt.tight_layout()
    out = figures_dir / f"{PREFIX}_association_vs_suitability_scatter.png"
    plt.savefig(out, dpi=180)
    plt.close()
    figures.append(out.name)

    save_heatmap(joined, "feature_name", "segment_setting", "rank_difference_abs", figures_dir / f"{PREFIX}_association_suitability_rank_difference_heatmap.png", "Mean absolute rank difference", cmap="magma")
    figures.append(f"{PREFIX}_association_suitability_rank_difference_heatmap.png")

    category = joined.copy()
    category["category_code"] = category["alignment_category"].map(CATEGORY_CODES)
    save_heatmap(category, "feature_name", "segment_setting", "category_code", figures_dir / f"{PREFIX}_alignment_category_heatmap.png", "Most common alignment category code", cmap="tab10", annot=True)
    figures.append(f"{PREFIX}_alignment_category_heatmap.png")

    save_heatmap(joined, "feature_name", "segment_setting", "combined_score", figures_dir / f"{PREFIX}_robust_feature_segment_candidate_heatmap.png", "Combined robust score")
    figures.append(f"{PREFIX}_robust_feature_segment_candidate_heatmap.png")
    save_heatmap(joined, "sensor_group", "segment_setting", "combined_score", figures_dir / f"{PREFIX}_sensor_group_robust_score_heatmap.png", "Sensor group robust score")
    figures.append(f"{PREFIX}_sensor_group_robust_score_heatmap.png")
    return figures


def write_mapping_summaries(output_dir: Path, association: pd.DataFrame, suitability: pd.DataFrame) -> None:
    feature_map = pd.DataFrame({"feature_name": FEATURES, "feature_group": [feature_group(f) for f in FEATURES]})
    feature_map["in_H1_S1"] = feature_map["feature_name"].isin(association["feature_name"].unique())
    feature_map["in_H1_S4"] = feature_map["feature_name"].isin(suitability["feature_name"].unique())
    feature_map.to_csv(output_dir / "data" / f"{PREFIX}_feature_mapping_summary.csv", index=False)
    pd.DataFrame({"segment_setting": SEGMENTS, "in_H1_S1": [s in set(association["segment_setting"]) for s in SEGMENTS], "in_H1_S4": [s in set(suitability["segment_setting"]) for s in SEGMENTS]}).to_csv(output_dir / "data" / f"{PREFIX}_segment_mapping_summary.csv", index=False)
    sensors = sorted(set(association["sensor_name"]) | set(suitability["sensor_name"]))
    pd.DataFrame({"sensor_name": sensors, "sensor_group": [sensor_group(s) for s in sensors], "in_H1_S1": [s in set(association["sensor_name"]) for s in sensors], "in_H1_S4": [s in set(suitability["sensor_name"]) for s in sensors]}).to_csv(output_dir / "data" / f"{PREFIX}_sensor_mapping_summary.csv", index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_yaml_like(path: Path, value: Any) -> None:
    try:
        import yaml

        path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        write_json(path, value)


def write_H1_S5_report(
    output_dir: Path,
    experiment_id: str,
    path_meta: dict[str, Any],
    h1s1_dir: Path,
    h1s4_dir: Path,
    loaded_h1s1: dict[str, str],
    loaded_h1s4: dict[str, str],
    joined: pd.DataFrame,
    category_summary: pd.DataFrame,
    corr: pd.DataFrame,
    robust: pd.DataFrame,
    disagreement: pd.DataFrame,
    metric_agreement_df: pd.DataFrame,
    h1s1_figures: list[str],
    h1s4_figures: list[str],
    cross_figures: list[str],
    skipped: list[dict[str, str]],
) -> None:
    top_robust = robust.head(10)
    high_assoc_low = disagreement[disagreement["disagreement_type"].eq("association_high_suitability_low")].head(10)
    low_assoc_high = disagreement[disagreement["disagreement_type"].eq("association_low_suitability_high")].head(10)
    acoustic_entry_exit = joined[
        joined["sensor_group"].eq("Acoustic") & joined["segment_setting"].astype(str).str.contains("entry|exit", case=False, na=False)
    ].sort_values("combined_score", ascending=False).head(10)
    corr_text = corr.to_string(index=False)
    text = f"""# H1_S5 Association vs Suitability Cross Analysis

## 1. Executive Summary

1. H1_S1 association and H1_S4 harmonic suitability were joined on `{len(joined)}` sensor-feature-segment combinations.
2. Primary association uses `top3_mean_abs_spearman`; primary suitability uses `suitability_harmonic_mean`.
3. Score/rank alignment is a cross-check, not a proof of predictive performance.
4. Robust candidates are combinations that are high in both target association and degradation-awareness.
5. Disagreements are informative: association-high/suitability-low suggests condition-specific or non-monotonic VB relation, while association-low/suitability-high suggests progression-like but target-weak behavior.

Top robust candidates:

{top_robust.to_string(index=False)}

Acoustic entry/exit candidates:

{acoustic_entry_exit[['sensor_name','feature_name','segment_setting','primary_association_score','primary_suitability_score','combined_score','alignment_category']].to_string(index=False)}

## 2. Source Experiments

- H1_S1 source directory: `{h1s1_dir}`
- H1_S4 source directory: `{h1s4_dir}`
- H1_S1 loaded files: `{loaded_h1s1}`
- H1_S4 loaded files: `{loaded_h1s4}`
- Skipped/missing items: `{skipped}`

## 3. Metric Meaning

- H1_S1 asks whether a feature is associated with VB. The primary score is `top3_mean_abs_spearman`.
- H1_S4 asks whether a feature follows degradation progression over run/time order. The primary score is `suitability_harmonic_mean`.
- These metrics measure different qualities; disagreement is expected in some combinations.
- H1_S4 uses harmonic suitability as primary and keeps `M + T` as legacy/reference.

## 4. H1_S1 Visualization Extension

Generated aligned H1_S1 figures:

{chr(10).join(f'- `{x}`' for x in h1s1_figures)}

Association metric agreement:

{metric_agreement_df.to_string(index=False)}

## 5. H1_S4 Visualization Summary

Copied or regenerated H1_S4 reference figures:

{chr(10).join(f'- `{x}`' for x in h1s4_figures)}

## 6. Association vs Suitability Alignment

Cross-score correlation summary:

{corr_text}

Alignment category summary:

{category_summary.to_string(index=False)}

## 7. Robust Feature-Segment Candidates

`combined_score` is the harmonic mean of min-max normalized association and min-max normalized harmonic suitability.

{top_robust.to_string(index=False)}

## 8. Disagreement Analysis

High association / low suitability:

{high_assoc_low.to_string(index=False)}

Low association / high suitability:

{low_assoc_high.to_string(index=False)}

Possible reasons include non-monotonic VB relationships, operating-condition dependency, time/order proxy behavior, sample-size instability, and metric definition differences.

## 9. Manufacturing Interpretation

Current sensors are interpreted as cutting-load and motor-load indicators. Vibration sensors can reflect dynamics, chatter, unstable cutting, and tool state changes. Acoustic sensors may capture contact/exit impacts, friction, chip events, and micro-fracture-like transient signatures. Entry/exit segment evidence should be interpreted as physically plausible but not causal without confirmatory prediction and controlled validation.

## 10. Implications for H2/H3 Modeling

- Prioritize robust candidates in reduced feature/segment searches.
- Use disagreement cases as ablations rather than immediate removals.
- Acoustic entry/exit candidates should be checked in H2/H3 prediction under domain shift.
- Sensor-wise segment attention/gating models can use robust and disagreement maps as diagnostic expectations.

## 11. Limitations

- Association does not guarantee prediction performance.
- Suitability does not guarantee prediction performance.
- H1_S1 and H1_S4 measure different feature quality aspects.
- Case count, operating-condition variation, order definition, and segment extraction quality limit interpretation.

## 12. Recommendations

- Confirm top robust candidates in H2/H3 prediction experiments.
- Investigate high-association/low-suitability features for non-monotonic or condition-specific behavior.
- Investigate high-suitability/low-association features for time/order proxy behavior.
- Run acoustic entry/exit confirmatory analysis before treating it as a stable modeling prior.

## Execution Path

- experiment_id: `{experiment_id}`
- hypothesis_id: `{path_meta.get('hypothesis_id')}`
- scenario_id: `{path_meta.get('scenario_id')}`
- experiment_topic: `{path_meta.get('experiment_topic')}`
- execution_dir: `{path_meta.get('execution_dir')}`
- path_schema_version: `{path_meta.get('path_schema_version')}`
"""
    report = output_dir / "reports" / f"{PREFIX}_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    html_text = "<!doctype html><html><head><meta charset='utf-8'><title>H1_S5 Report</title></head><body>" + html.escape(text).replace("\n", "<br>\n") + "</body></html>"
    (output_dir / "reports" / f"{PREFIX}_report.html").write_text(html_text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    experiment_id = f"{timestamp}_H1_S5_association_vs_suitability_cross_analysis"
    path_config = {"experiment": {"experiment_id": experiment_id, "timestamp": timestamp, "hypothesis_id": "H1", "scenario_id": "S5", "experiment_topic": "association_vs_suitability_cross_analysis"}}
    output_dir = Path(create_execution_dir(path_config, root=root / "experiments" / "executions"))
    for name in ["configs", "data", "analysis", "figures", "reports", "logs"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "logs" / f"{PREFIX}_run.log"
    error_log = output_dir / "logs" / f"{PREFIX}_error.log"
    error_log.touch()

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line)
        with run_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    skipped: list[dict[str, str]] = []
    try:
        log(f"H1_S5 started: {experiment_id}")
        h1s1_candidates = find_source_experiment_dirs(root / "experiments" / "executions", "H1", "S1")
        h1s4_candidates = find_source_experiment_dirs(root / "experiments" / "executions", "H1", "S4")
        if not h1s1_candidates:
            raise FileNotFoundError("No H1_S1 source directory found.")
        if not h1s4_candidates:
            raise FileNotFoundError("No H1_S4 source directory found.")
        h1s1_dir = h1s1_candidates[-1]
        h1s4_dir = h1s4_candidates[-1]
        h1s1_tables, h1s1_files = load_H1_S1_association_results(h1s1_dir)
        h1s4_tables, h1s4_files = load_H1_S4_suitability_results(h1s4_dir)
        write_json(output_dir / "data" / f"{PREFIX}_source_H1_S1_files_summary.json", {"source_dir": str(h1s1_dir), "loaded_files": h1s1_files})
        write_json(output_dir / "data" / f"{PREFIX}_source_H1_S4_files_summary.json", {"source_dir": str(h1s4_dir), "loaded_files": h1s4_files})

        h1s1_source_id = str(h1s1_tables["feature_level"]["experiment_id"].iloc[0]) if "experiment_id" in h1s1_tables["feature_level"] else h1s1_dir.name
        h1s4_source_id = str(h1s4_tables["suitability"]["experiment_id"].iloc[0]) if "experiment_id" in h1s4_tables["suitability"] else h1s4_dir.name
        association = standardize_association_table(h1s1_tables["feature_level"], h1s1_source_id)
        suitability = standardize_suitability_table(h1s4_tables["suitability"], h1s4_source_id)
        association.to_csv(output_dir / "analysis" / f"{PREFIX}_H1_S1_standardized_association_table.csv", index=False)
        suitability.to_csv(output_dir / "analysis" / f"{PREFIX}_H1_S4_standardized_suitability_table.csv", index=False)
        write_mapping_summaries(output_dir, association, suitability)

        metric_agreement_df = metric_agreement(association)
        metric_agreement_df.to_csv(output_dir / "analysis" / f"{PREFIX}_H1_S1_association_metric_agreement.csv", index=False)
        h1s1_figures = create_H1_S1_aligned_visualizations(association, h1s1_tables.get("pair_level"), metric_agreement_df, output_dir / "figures")
        h1s4_figures, h1s4_skipped = copy_or_regenerate_h1_s4_figures(h1s4_dir, suitability, h1s4_tables.get("pair_level"), output_dir / "figures")
        skipped.extend({"item": item, "reason": "source H1_S4 figure missing; regenerated when possible"} for item in h1s4_skipped)

        joined = join_association_and_suitability(association, suitability, experiment_id)
        joined.to_csv(output_dir / "analysis" / f"{PREFIX}_association_vs_suitability_joined_table.csv", index=False)
        category_summary = alignment_category_summary(joined)
        category_summary.to_csv(output_dir / "analysis" / f"{PREFIX}_alignment_category_summary.csv", index=False)
        corr = compute_cross_score_correlations(joined)
        corr.to_csv(output_dir / "analysis" / f"{PREFIX}_cross_score_correlation_summary.csv", index=False)
        robust = robust_candidates(joined)
        robust.to_csv(output_dir / "analysis" / f"{PREFIX}_robust_feature_segment_candidates.csv", index=False)
        disagreement = disagreement_table(joined)
        disagreement.to_csv(output_dir / "analysis" / f"{PREFIX}_association_suitability_disagreement.csv", index=False)
        cross_figures = create_cross_analysis_figures(joined, output_dir / "figures")

        validation = {
            "ok": bool(len(joined)),
            "h1_s1_dir_found": True,
            "h1_s4_dir_found": True,
            "association_rows": int(len(association)),
            "suitability_rows": int(len(suitability)),
            "joined_rows": int(len(joined)),
            "join_coverage_ratio": float(len(joined) / min(len(association), len(suitability))) if min(len(association), len(suitability)) else 0.0,
            "primary_association_score": "top3_mean_abs_spearman",
            "primary_suitability_score": "suitability_harmonic_mean",
            "skipped": skipped,
        }
        write_json(output_dir / "configs" / f"{PREFIX}_config_validation.json", validation)
        input_config = {
            "experiment": path_config["experiment"],
            "source_h1_s1": str(h1s1_dir),
            "source_h1_s4": str(h1s4_dir),
            "feature_scope": FEATURES,
            "segment_scope": SEGMENTS,
            "primary_association_score": "top3_mean_abs_spearman",
            "primary_suitability_score": "suitability_harmonic_mean",
        }
        write_yaml_like(output_dir / "configs" / f"{PREFIX}_input_config.yaml", input_config)
        write_yaml_like(output_dir / "configs" / f"{PREFIX}_resolved_config.yaml", {**input_config, "validation": validation})
        analysis_summary = {
            "experiment_id": experiment_id,
            "source_h1_s1": str(h1s1_dir),
            "source_h1_s4": str(h1s4_dir),
            "joined_rows": int(len(joined)),
            "top_robust": robust.head(10).to_dict(orient="records"),
            "alignment_counts": joined["alignment_category"].value_counts().to_dict(),
            "cross_score_correlation": corr.to_dict(orient="records"),
            "figures": h1s1_figures + h1s4_figures + cross_figures,
            "skipped": skipped,
        }
        write_json(output_dir / "analysis" / f"{PREFIX}_analysis_summary.json", analysis_summary)
        write_H1_S5_report(
            output_dir,
            experiment_id,
            path_config["experiment"],
            h1s1_dir,
            h1s4_dir,
            h1s1_files,
            h1s4_files,
            joined,
            category_summary,
            corr,
            robust,
            disagreement,
            metric_agreement_df,
            h1s1_figures,
            h1s4_figures,
            cross_figures,
            skipped,
        )
        log(f"H1_S5 finished: {experiment_id}")
        return {
            "experiment_id": experiment_id,
            "execution_dir": str(output_dir),
            "h1s1_dir": str(h1s1_dir),
            "h1s4_dir": str(h1s4_dir),
            "joined_rows": int(len(joined)),
            "top_robust": robust.head(10).to_dict(orient="records"),
            "correlation": corr.to_dict(orient="records"),
            "skipped": skipped,
        }
    except Exception:
        error_log.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H1_S5 association vs suitability cross-EDA analysis.")
    parser.add_argument("--root", default=".")
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

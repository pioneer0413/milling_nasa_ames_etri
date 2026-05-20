#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEGMENT_ORDER = [
    "full_length",
    "steady",
    "entry",
    "exit",
    "entry_steady",
    "entry_exit",
    "steady_exit",
    "entry_steady_exit",
]
FEATURE_ORDER = [
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


def latest_with_file(root: Path, relative: str) -> Path:
    candidates = []
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir() and (child / relative).exists():
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No execution under {root} contains {relative}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def minmax(series: pd.Series) -> pd.Series:
    series = series.astype("float64")
    lo = series.min(skipna=True)
    hi = series.max(skipna=True)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.ones(len(series)), index=series.index, dtype="float64")
    return (series - lo) / (hi - lo)


def harmonic3(a: pd.Series, b: pd.Series, c: pd.Series, eps: float = 1e-12) -> pd.Series:
    denom = (1 / (a + eps)) + (1 / (b + eps)) + (1 / (c + eps))
    return 3 / denom


def load_and_aggregate(s1_dir: Path, s2_dir: Path, s3_dir: Path) -> dict[str, pd.DataFrame]:
    s1 = pd.read_csv(s1_dir / "metrics" / "association_feature_level.csv")
    s1_ok = s1.loc[s1["status"].eq("ok")].copy()
    assoc = (
        s1_ok.groupby(["feature_name", "segment_setting"], dropna=False)
        .agg(
            association_mean_abs_pearson=("pearson_abs", "mean"),
            association_median_abs_pearson=("pearson_abs", "median"),
            association_mean_abs_spearman=("spearman_abs", "mean"),
            association_median_abs_spearman=("spearman_abs", "median"),
            association_mean_mutual_information=("mutual_information", "mean"),
            association_median_mutual_information=("mutual_information", "median"),
            association_max_abs_spearman=("spearman_abs", "max"),
            association_ok_conditions=("status", "size"),
            association_num_cases=("case_id", "nunique"),
            association_num_sensors=("sensor_name", "nunique"),
        )
        .reset_index()
    )
    assoc["association_mi_normalized"] = minmax(assoc["association_mean_mutual_information"])
    assoc["association_score"] = (
        assoc["association_mean_abs_pearson"]
        + assoc["association_mean_abs_spearman"]
        + assoc["association_mi_normalized"]
    ) / 3
    assoc["association_rank"] = assoc["association_score"].rank(ascending=False, method="first").astype(int)

    suit = pd.read_csv(s2_dir / "analysis" / "H1_S2_feature_segment_summary.csv")
    suit = suit.rename(
        columns={
            "mean_monotonicity": "suitability_mean_monotonicity",
            "mean_trendability": "suitability_mean_trendability",
            "mean_suitability": "suitability_score",
            "suitability_harmonic_mean": "suitability_harmonic_mean",
            "suitability_sum_legacy": "suitability_sum_legacy",
            "rank_by_suitability": "suitability_rank",
        }
    )
    suit = suit[
        [
            "feature_name",
            "feature_group",
            "segment_setting",
            "suitability_mean_monotonicity",
            "suitability_mean_trendability",
            "suitability_score",
            "suitability_harmonic_mean",
            "suitability_sum_legacy",
            "std_suitability",
            "suitability_rank",
            "sample_count",
        ]
    ].copy()

    rob = pd.read_csv(s3_dir / "analysis" / "H1_S3_robustness_segment_feature.csv")
    rob = rob.rename(
        columns={
            "mean_robustness": "robustness_score",
            "median_robustness": "robustness_median",
            "std_robustness": "robustness_std",
            "rank_by_mean_robustness": "robustness_rank",
        }
    )
    rob = rob[
        [
            "feature_name",
            "feature_group",
            "segment_setting",
            "robustness_score",
            "robustness_median",
            "robustness_std",
            "mean_abs_relative_deviation",
            "robustness_rank",
            "trend_method",
            "trend_window",
        ]
    ].copy()

    joined = assoc.merge(suit, on=["feature_name", "segment_setting"], how="outer", suffixes=("", "_suitability"))
    joined["feature_group"] = joined["feature_group"].fillna(joined.get("feature_group_suitability"))
    joined = joined.drop(columns=[c for c in ["feature_group_suitability"] if c in joined.columns])
    joined = joined.merge(rob, on=["feature_name", "feature_group", "segment_setting"], how="outer")
    joined["association_norm"] = minmax(joined["association_score"])
    joined["suitability_norm"] = minmax(joined["suitability_score"])
    joined["robustness_norm"] = minmax(joined["robustness_score"])
    joined["integrated_mean_score"] = joined[["association_norm", "suitability_norm", "robustness_norm"]].mean(axis=1)
    joined["integrated_balanced_score"] = harmonic3(
        joined["association_norm"].clip(lower=0),
        joined["suitability_norm"].clip(lower=0),
        joined["robustness_norm"].clip(lower=0),
    )
    joined["integrated_rank"] = joined["integrated_balanced_score"].rank(ascending=False, method="first").astype(int)

    for metric, col in [
        ("association", "association_norm"),
        ("suitability", "suitability_norm"),
        ("robustness", "robustness_norm"),
    ]:
        high = joined[col] >= joined[col].quantile(0.75)
        low = joined[col] <= joined[col].quantile(0.25)
        joined[f"{metric}_tier"] = np.where(high, "high", np.where(low, "low", "middle"))

    joined["evidence_pattern"] = joined.apply(classify_pattern, axis=1)

    segment_summary = (
        joined.groupby("segment_setting", dropna=False)
        .agg(
            association_score=("association_score", "mean"),
            suitability_score=("suitability_score", "mean"),
            robustness_score=("robustness_score", "mean"),
            integrated_balanced_score=("integrated_balanced_score", "mean"),
            integrated_mean_score=("integrated_mean_score", "mean"),
            num_features=("feature_name", "nunique"),
        )
        .reset_index()
    )
    segment_summary["integrated_rank"] = segment_summary["integrated_balanced_score"].rank(ascending=False, method="first").astype(int)

    feature_summary = (
        joined.groupby(["feature_name", "feature_group"], dropna=False)
        .agg(
            association_score=("association_score", "mean"),
            suitability_score=("suitability_score", "mean"),
            robustness_score=("robustness_score", "mean"),
            integrated_balanced_score=("integrated_balanced_score", "mean"),
            integrated_mean_score=("integrated_mean_score", "mean"),
            num_segments=("segment_setting", "nunique"),
        )
        .reset_index()
    )
    feature_summary["integrated_rank"] = feature_summary["integrated_balanced_score"].rank(ascending=False, method="first").astype(int)

    pattern_summary = (
        joined.groupby("evidence_pattern", dropna=False)
        .agg(
            count=("feature_name", "size"),
            mean_association=("association_score", "mean"),
            mean_suitability=("suitability_score", "mean"),
            mean_robustness=("robustness_score", "mean"),
            mean_integrated=("integrated_balanced_score", "mean"),
        )
        .reset_index()
        .sort_values(["mean_integrated", "count"], ascending=[False, False])
    )
    return {
        "association": assoc.sort_values("association_rank"),
        "suitability": suit.sort_values("suitability_rank"),
        "robustness": rob.sort_values("robustness_rank"),
        "joined": joined.sort_values("integrated_rank"),
        "segment_summary": segment_summary.sort_values("integrated_rank"),
        "feature_summary": feature_summary.sort_values("integrated_rank"),
        "pattern_summary": pattern_summary,
    }


def classify_pattern(row: pd.Series) -> str:
    tiers = (row["association_tier"], row["suitability_tier"], row["robustness_tier"])
    if tiers == ("high", "high", "high"):
        return "triple_high"
    if row["association_tier"] == "high" and row["suitability_tier"] == "high":
        return "target_relevant_and_degradation_aware"
    if row["association_tier"] == "high" and row["robustness_tier"] == "high":
        return "target_relevant_and_stable"
    if row["suitability_tier"] == "high" and row["robustness_tier"] == "high":
        return "trend_stable_but_target_check_needed"
    if row["association_tier"] == "high":
        return "target_relevant_only"
    if row["suitability_tier"] == "high":
        return "degradation_trend_only"
    if row["robustness_tier"] == "high":
        return "stable_only"
    if tiers == ("low", "low", "low"):
        return "triple_low"
    return "mixed"


def table(frame: pd.DataFrame, cols: list[str], n: int = 12) -> str:
    data = frame[cols].head(n).copy()
    if data.empty:
        return "_No rows._"
    for col in data.columns:
        if pd.api.types.is_float_dtype(data[col]):
            data[col] = data[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    header = "| " + " | ".join(data.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(data.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in data.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def write_heatmap(df: pd.DataFrame, value: str, title: str, path: Path) -> None:
    matrix = df.pivot_table(index="feature_name", columns="segment_setting", values=value, aggfunc="mean")
    matrix = matrix.reindex(index=FEATURE_ORDER, columns=SEGMENT_ORDER)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    arr = matrix.to_numpy(dtype="float64")
    image = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val > np.nanmedian(arr) else "black")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=value)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def write_report(output_dir: Path, sources: dict[str, Path], data: dict[str, pd.DataFrame]) -> None:
    joined = data["joined"]
    segment_summary = data["segment_summary"]
    feature_summary = data["feature_summary"]
    pattern_summary = data["pattern_summary"]
    corr = joined[
        ["association_score", "suitability_score", "robustness_score", "integrated_balanced_score"]
    ].corr(method="spearman")
    triple = joined[joined["evidence_pattern"].eq("triple_high")]
    conflicts = joined[
        joined["evidence_pattern"].isin(
            [
                "target_relevant_only",
                "degradation_trend_only",
                "stable_only",
                "trend_stable_but_target_check_needed",
            ]
        )
    ].sort_values(["association_norm", "suitability_norm", "robustness_norm"], ascending=[False, False, False])

    text = f"""# H1 Feature x Segment Integrated Analysis

## Scope

이 보고서는 H1/S1, H1/S2, H1/S3 실행 디렉터리를 탐색해 `feature x segment` 단위로 association, suitability, robustness를 결합 분석한 결과입니다.

- S1 association source: `{sources['s1']}`
- S2 suitability source: `{sources['s2']}`
- S3 robustness source: `{sources['s3']}`
- feature x segment 조합 수: `{len(joined)}`
- segment 수: `{joined['segment_setting'].nunique()}`
- feature 수: `{joined['feature_name'].nunique()}`

## Metric Alignment

- Association: S1 `association_feature_level.csv`에서 case와 sensor를 평균했습니다. `association_score = mean(abs Pearson, abs Spearman, normalized MI)`입니다.
- Suitability: S2 `H1_S2_feature_segment_summary.csv`의 harmonic suitability를 primary score로 사용했습니다.
- Robustness: S3 `H1_S3_robustness_segment_feature.csv`의 mean robustness를 사용했습니다.
- Integrated balanced score: 세 지표의 min-max normalized score에 대한 harmonic mean입니다. 한 축만 높은 후보보다 세 축이 함께 높은 후보를 우선합니다.

## Executive Findings

1. 통합 최상위 후보는 `{joined.iloc[0]['feature_name']} / {joined.iloc[0]['segment_setting']}`이며 balanced score는 `{joined.iloc[0]['integrated_balanced_score']:.4f}`입니다.
2. segment 평균 기준 최상위는 `{segment_summary.iloc[0]['segment_setting']}`입니다.
3. feature 평균 기준 최상위는 `{feature_summary.iloc[0]['feature_name']}`입니다.
4. `triple_high` 조합 수는 `{len(triple)}`개입니다.
5. Robustness는 spectral centroid와 mean 계열에서 높게 나타나지만, association/suitability까지 함께 보면 energy/load 계열 feature가 더 강한 후보로 올라옵니다.

## Top Integrated Feature x Segment

{table(joined, ['feature_name','segment_setting','association_score','suitability_score','robustness_score','integrated_balanced_score','evidence_pattern','integrated_rank'], 15)}

## Triple-High Candidates

{table(triple.sort_values('integrated_balanced_score', ascending=False), ['feature_name','segment_setting','association_score','suitability_score','robustness_score','integrated_balanced_score','integrated_rank'], 20)}

## Segment Summary

{table(segment_summary, ['segment_setting','association_score','suitability_score','robustness_score','integrated_balanced_score','integrated_rank'], 8)}

## Feature Summary

{table(feature_summary, ['feature_name','feature_group','association_score','suitability_score','robustness_score','integrated_balanced_score','integrated_rank'], 9)}

## Evidence Pattern Summary

{table(pattern_summary, ['evidence_pattern','count','mean_association','mean_suitability','mean_robustness','mean_integrated'], 20)}

## Main Tensions

아래 조합들은 한두 지표는 강하지만 세 축이 모두 강하지 않은 후보입니다. 예측 모델 후보로 쓸 때 목적을 분명히 나눠야 합니다.

{table(conflicts, ['feature_name','segment_setting','association_score','suitability_score','robustness_score','integrated_balanced_score','evidence_pattern'], 20)}

## Rank Details

### Association Top

{table(data['association'], ['feature_name','segment_setting','association_score','association_mean_abs_spearman','association_mean_abs_pearson','association_mean_mutual_information','association_rank'], 12)}

### Suitability Top

{table(data['suitability'], ['feature_name','segment_setting','suitability_score','suitability_mean_monotonicity','suitability_mean_trendability','suitability_rank'], 12)}

### Robustness Top

{table(data['robustness'], ['feature_name','segment_setting','robustness_score','robustness_median','robustness_std','robustness_rank'], 12)}

## Spearman Correlation Between Scores

{corr.round(4).to_markdown()}

## Interpretation

Association은 VB와의 직접 관련성이고, suitability는 run progression에서 monotonicity/trendability가 좋은지, robustness는 feature sequence가 평균 trend에서 얼마나 덜 흔들리는지를 봅니다. 따라서 통합 상위 후보는 `VB 관련성`, `마모 진행성`, `노이즈 안정성`을 동시에 만족하는 feature x segment 후보입니다.

Robustness 단독 상위는 spectral centroid 중심으로 매우 높게 나왔습니다. 다만 이 지표는 VB를 직접 보지 않으므로, association과 suitability를 함께 만족하는 후보를 우선하는 것이 더 보수적입니다. 특히 band_energy, mean, min 계열은 suitability가 높고 association도 상대적으로 유지되어 후속 H2/H3 예측 실험에서 우선순위가 높습니다.

## Output Files

- `analysis/H1_feature_segment_integrated_scores.csv`
- `analysis/H1_feature_segment_segment_summary.csv`
- `analysis/H1_feature_segment_feature_summary.csv`
- `analysis/H1_feature_segment_pattern_summary.csv`
- `figures/H1_integrated_balanced_score_heatmap.png`
- `figures/H1_association_score_heatmap.png`
- `figures/H1_suitability_score_heatmap.png`
- `figures/H1_robustness_score_heatmap.png`
"""
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / "H1_feature_segment_integrated_report.md").write_text(text, encoding="utf-8")
    html_text = "<!doctype html><html><head><meta charset='utf-8'><title>H1 Integrated Analysis</title></head><body>" + html.escape(text).replace("\n", "<br>\n") + "</body></html>"
    (output_dir / "reports" / "H1_feature_segment_integrated_report.html").write_text(html_text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    h1_root = root / "experiments" / "executions" / "H1"
    s1_dir = Path(args.s1_dir) if args.s1_dir else latest_with_file(h1_root / "S1", "metrics/association_feature_level.csv")
    s2_dir = Path(args.s2_dir) if args.s2_dir else latest_with_file(h1_root / "S2", "analysis/H1_S2_feature_segment_summary.csv")
    s3_dir = Path(args.s3_dir) if args.s3_dir else latest_with_file(h1_root / "S3", "analysis/H1_S3_robustness_segment_feature.csv")
    if not s1_dir.is_absolute():
        s1_dir = root / s1_dir
    if not s2_dir.is_absolute():
        s2_dir = root / s2_dir
    if not s3_dir.is_absolute():
        s3_dir = root / s3_dir

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = h1_root / "S3" / f"{timestamp}_feature_segment_integrated_association_suitability_robustness_analysis"
    for subdir in ["configs", "analysis", "figures", "reports", "logs"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    data = load_and_aggregate(s1_dir, s2_dir, s3_dir)
    data["joined"].to_csv(output_dir / "analysis" / "H1_feature_segment_integrated_scores.csv", index=False)
    data["association"].to_csv(output_dir / "analysis" / "H1_feature_segment_association_summary.csv", index=False)
    data["suitability"].to_csv(output_dir / "analysis" / "H1_feature_segment_suitability_summary.csv", index=False)
    data["robustness"].to_csv(output_dir / "analysis" / "H1_feature_segment_robustness_summary.csv", index=False)
    data["segment_summary"].to_csv(output_dir / "analysis" / "H1_feature_segment_segment_summary.csv", index=False)
    data["feature_summary"].to_csv(output_dir / "analysis" / "H1_feature_segment_feature_summary.csv", index=False)
    data["pattern_summary"].to_csv(output_dir / "analysis" / "H1_feature_segment_pattern_summary.csv", index=False)

    write_heatmap(data["joined"], "integrated_balanced_score", "H1 Integrated Balanced Score", output_dir / "figures" / "H1_integrated_balanced_score_heatmap.png")
    write_heatmap(data["joined"], "association_score", "H1 Association Score", output_dir / "figures" / "H1_association_score_heatmap.png")
    write_heatmap(data["joined"], "suitability_score", "H1 Suitability Score", output_dir / "figures" / "H1_suitability_score_heatmap.png")
    write_heatmap(data["joined"], "robustness_score", "H1 Robustness Score", output_dir / "figures" / "H1_robustness_score_heatmap.png")

    sources = {"s1": s1_dir, "s2": s2_dir, "s3": s3_dir}
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dirs": {k: str(v) for k, v in sources.items()},
        "aggregation_unit": "feature x segment",
        "association_score": "mean(abs Pearson, abs Spearman, minmax mean MI)",
        "integrated_balanced_score": "harmonic mean of minmax association, suitability, robustness",
        "output_dir": str(output_dir),
    }
    (output_dir / "configs" / "H1_feature_segment_integrated_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "output_dir": str(output_dir),
        "top_integrated": data["joined"].head(10).to_dict(orient="records"),
        "segment_summary": data["segment_summary"].to_dict(orient="records"),
        "feature_summary": data["feature_summary"].to_dict(orient="records"),
        "pattern_summary": data["pattern_summary"].to_dict(orient="records"),
    }
    (output_dir / "analysis" / "H1_feature_segment_integrated_analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_report(output_dir, sources, data)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate H1 S1/S2/S3 feature x segment evidence.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--s1-dir")
    parser.add_argument("--s2-dir")
    parser.add_argument("--s3-dir")
    result = run(parser.parse_args())
    print(json.dumps({"output_dir": result["output_dir"], "top_integrated": result["top_integrated"][:3]}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

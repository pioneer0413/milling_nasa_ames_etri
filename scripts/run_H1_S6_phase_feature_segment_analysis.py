#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_H1_S3_association_suitability_segment_cross_analysis import (  # noqa: E402
    CASE_SCOPE,
    SEGMENT_SETTINGS,
    TARGET_FEATURES,
    find_or_create_entry_steady_exit_features,
    identify_sensors,
    load_dataset,
)
from run_H1_S3_robustness_analysis import robustness_for_sequence  # noqa: E402


PHASES = ["early", "middle", "late"]


def assign_run_phase(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["run_phase"] = ""
    out["phase_index"] = -1
    out["phase_position"] = -1
    for case_id, group in out.sort_values(["case_id", "run"]).groupby("case_id", sort=False):
        indices = group.index.to_numpy()
        for phase_idx, phase_indices in enumerate(np.array_split(indices, 3)):
            if len(phase_indices) == 0:
                continue
            out.loc[phase_indices, "run_phase"] = PHASES[phase_idx]
            out.loc[phase_indices, "phase_index"] = phase_idx
            out.loc[phase_indices, "phase_position"] = np.arange(1, len(phase_indices) + 1)
    return out


def minmax_by_group(frame: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.Series:
    def scale(s: pd.Series) -> pd.Series:
        lo = s.min(skipna=True)
        hi = s.max(skipna=True)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(np.ones(len(s)), index=s.index, dtype="float64")
        return (s - lo) / (hi - lo)

    return frame.groupby(group_cols, dropna=False)[value_col].transform(scale)


def compute_monotonicity(values: np.ndarray) -> float:
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    diff = np.diff(values)
    inc = np.sum(diff > 0)
    dec = np.sum(diff < 0)
    return float(abs((inc - dec) / (len(values) - 1)))


def compute_trendability(values: np.ndarray, t: np.ndarray) -> float:
    values = np.asarray(values, dtype="float64")
    t = np.asarray(t, dtype="float64")
    finite = np.isfinite(values) & np.isfinite(t)
    values = values[finite]
    t = t[finite]
    if len(values) < 2:
        return np.nan
    if np.std(values) <= 0 or np.std(t) <= 0:
        return np.nan
    return float(abs(np.corrcoef(values, t)[0, 1]))


def harmonic2(a: float, b: float, eps: float = 1e-12) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    if a + b <= 0:
        return 0.0
    return float(2 * a * b / (a + b + eps))


def harmonic3(a: pd.Series, b: pd.Series, c: pd.Series, eps: float = 1e-12) -> pd.Series:
    return 3 / (1 / (a + eps) + 1 / (b + eps) + 1 / (c + eps))


def compute_association_case_sensor(feature_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    group_cols = ["run_phase", "case_id", "sensor_name", "segment_setting", "feature_name"]
    for keys, group in feature_long.groupby(group_cols, dropna=False):
        phase, case_id, sensor, segment, feature = keys
        g = group.sort_values("run")
        finite = np.isfinite(g["feature_value"].to_numpy(dtype="float64")) & np.isfinite(g["VB"].to_numpy(dtype="float64"))
        x = g["feature_value"].to_numpy(dtype="float64")[finite]
        y = g["VB"].to_numpy(dtype="float64")[finite]
        if len(x) < 3 or np.std(x) <= 0 or np.std(y) <= 0:
            skipped.append({"run_phase": phase, "case_id": int(case_id), "sensor_name": sensor, "segment_setting": segment, "feature_name": feature, "n_samples": int(len(x)), "reason": "n_lt_3_or_zero_variance"})
            continue
        pear = pearsonr(x, y).statistic
        spear = spearmanr(x, y).statistic
        try:
            mi = float(mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0])
        except Exception:
            mi = np.nan
        rows.append(
            {
                "run_phase": phase,
                "case_id": int(case_id),
                "sensor_name": sensor,
                "sensor_group": g["sensor_group"].iloc[0],
                "segment_setting": segment,
                "feature_name": feature,
                "feature_group": g["feature_group"].iloc[0],
                "pearson_abs": abs(float(pear)),
                "spearman_abs": abs(float(spear)),
                "mutual_information": mi,
                "n_samples": int(len(x)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(skipped)


def compute_suitability_case_sensor(feature_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["run_phase", "case_id", "sensor_name", "segment_setting", "feature_name"]
    for keys, group in feature_long.groupby(group_cols, dropna=False):
        phase, case_id, sensor, segment, feature = keys
        g = group.sort_values("run")
        x = g["feature_value"].to_numpy(dtype="float64")
        t = g["run"].to_numpy(dtype="float64")
        finite = np.isfinite(x) & np.isfinite(t)
        x = x[finite]
        t = t[finite]
        if len(x) < 2:
            status = "skipped_n_less_than_2"
            mono = trend = score = np.nan
        else:
            mono = compute_monotonicity(x)
            trend = compute_trendability(x, t)
            score = harmonic2(mono, trend)
            status = "ok" if np.isfinite(score) else "skipped_zero_variance"
        rows.append(
            {
                "run_phase": phase,
                "case_id": int(case_id),
                "sensor_name": sensor,
                "sensor_group": g["sensor_group"].iloc[0],
                "segment_setting": segment,
                "feature_name": feature,
                "feature_group": g["feature_group"].iloc[0],
                "monotonicity": mono,
                "trendability": trend,
                "suitability_score": score,
                "n_samples": int(len(x)),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def compute_robustness_case_sensor(feature_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["run_phase", "case_id", "sensor_name", "segment_setting", "feature_name"]
    for keys, group in feature_long.groupby(group_cols, dropna=False):
        phase, case_id, sensor, segment, feature = keys
        g = group.sort_values("run")
        result = robustness_for_sequence(g["feature_value"].to_numpy(dtype="float64"), window=3, epsilon=1e-12)
        rows.append(
            {
                "run_phase": phase,
                "case_id": int(case_id),
                "sensor_name": sensor,
                "sensor_group": g["sensor_group"].iloc[0],
                "segment_setting": segment,
                "feature_name": feature,
                "feature_group": g["feature_group"].iloc[0],
                "robustness_score": result["robustness"],
                "mean_abs_relative_deviation": result["mean_abs_relative_deviation"],
                "n_samples": result["sequence_length"],
                "status": result["status"],
            }
        )
    return pd.DataFrame(rows)


def aggregate_phase_metric(df: pd.DataFrame, metric_name: str, score_col: str, extra: dict[str, tuple[str, str]]) -> pd.DataFrame:
    valid = df.loc[np.isfinite(df[score_col])].copy()
    agg: dict[str, tuple[str, str]] = {
        f"{metric_name}_score": (score_col, "mean"),
        f"{metric_name}_median": (score_col, "median"),
        f"{metric_name}_std": (score_col, "std"),
        "num_case_sensor_sequences": (score_col, "size"),
        "num_cases": ("case_id", "nunique"),
        "num_sensors": ("sensor_name", "nunique"),
    }
    agg.update(extra)
    out = (
        valid.groupby(["run_phase", "feature_name", "feature_group", "segment_setting"], dropna=False)
        .agg(**agg)
        .reset_index()
    )
    out[f"{metric_name}_std"] = out[f"{metric_name}_std"].fillna(0.0)
    out[f"{metric_name}_rank_within_phase"] = out.groupby("run_phase")[f"{metric_name}_score"].rank(ascending=False, method="first").astype(int)
    return out


def build_phase_summary(assoc_case: pd.DataFrame, suit_case: pd.DataFrame, rob_case: pd.DataFrame) -> pd.DataFrame:
    assoc_case = assoc_case.copy()
    assoc_case["mi_norm"] = minmax_by_group(assoc_case, ["run_phase"], "mutual_information")
    assoc_case["association_score"] = (assoc_case["pearson_abs"] + assoc_case["spearman_abs"] + assoc_case["mi_norm"]) / 3.0
    assoc = aggregate_phase_metric(
        assoc_case,
        "association",
        "association_score",
        {
            "association_abs_pearson": ("pearson_abs", "mean"),
            "association_abs_spearman": ("spearman_abs", "mean"),
            "association_mi": ("mutual_information", "mean"),
            "association_mean_n": ("n_samples", "mean"),
        },
    )
    suit = aggregate_phase_metric(
        suit_case.loc[suit_case["status"].eq("ok")],
        "suitability",
        "suitability_score",
        {
            "mean_monotonicity": ("monotonicity", "mean"),
            "mean_trendability": ("trendability", "mean"),
            "suitability_mean_n": ("n_samples", "mean"),
        },
    )
    rob = aggregate_phase_metric(
        rob_case.loc[rob_case["status"].eq("ok")],
        "robustness",
        "robustness_score",
        {
            "mean_abs_relative_deviation": ("mean_abs_relative_deviation", "mean"),
            "robustness_mean_n": ("n_samples", "mean"),
        },
    )
    key = ["run_phase", "feature_name", "feature_group", "segment_setting"]
    out = assoc.merge(suit, on=key, how="outer").merge(rob, on=key, how="outer")
    for col in ["association_score", "suitability_score", "robustness_score"]:
        out[f"{col}_norm"] = minmax_by_group(out, ["run_phase"], col)
    out["integrated_mean_score"] = out[["association_score_norm", "suitability_score_norm", "robustness_score_norm"]].mean(axis=1)
    out["integrated_balanced_score"] = harmonic3(
        out["association_score_norm"].fillna(0).clip(lower=0),
        out["suitability_score_norm"].fillna(0).clip(lower=0),
        out["robustness_score_norm"].fillna(0).clip(lower=0),
    )
    out["rank_average"] = out[["association_rank_within_phase", "suitability_rank_within_phase", "robustness_rank_within_phase"]].mean(axis=1)
    out["integrated_rank_within_phase"] = out.groupby("run_phase")["integrated_balanced_score"].rank(ascending=False, method="first").astype(int)
    out["rank_average_within_phase"] = out.groupby("run_phase")["rank_average"].rank(ascending=True, method="first").astype(int)
    out["phase_order"] = out["run_phase"].map({"early": 0, "middle": 1, "late": 2})
    return out.sort_values(["phase_order", "integrated_rank_within_phase"]).reset_index(drop=True)


def build_transition_table(phase_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature, segment), group in phase_summary.groupby(["feature_name", "segment_setting"], dropna=False):
        wide = group.set_index("run_phase")
        row = {"feature_name": feature, "segment_setting": segment}
        for phase in PHASES:
            if phase in wide.index:
                for col in ["association_score", "suitability_score", "robustness_score", "integrated_balanced_score", "integrated_rank_within_phase", "rank_average_within_phase"]:
                    row[f"{phase}_{col}"] = wide.loc[phase, col]
        if "early" in wide.index and "late" in wide.index:
            row["delta_integrated_late_minus_early"] = wide.loc["late", "integrated_balanced_score"] - wide.loc["early", "integrated_balanced_score"]
            row["delta_rank_late_minus_early"] = wide.loc["late", "integrated_rank_within_phase"] - wide.loc["early", "integrated_rank_within_phase"]
            row["delta_rank_average_late_minus_early"] = wide.loc["late", "rank_average_within_phase"] - wide.loc["early", "rank_average_within_phase"]
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("delta_integrated_late_minus_early", ascending=False, na_position="last").reset_index(drop=True)


def markdown_table(frame: pd.DataFrame, max_rows: int = 15) -> str:
    if frame.empty:
        return "_No rows._"
    out = frame.head(max_rows).copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    header = "| " + " | ".join(map(str, out.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in out.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def write_report(output_dir: Path, phase_summary: pd.DataFrame, transitions: pd.DataFrame, phase_counts: pd.DataFrame, skipped_assoc: pd.DataFrame, sources: dict[str, str]) -> None:
    phase_top = phase_summary.sort_values("integrated_rank_within_phase").groupby("run_phase").head(1)
    rank_top = phase_summary.sort_values("rank_average_within_phase").groupby("run_phase").head(1)
    top_by_phase = []
    for phase in PHASES:
        integrated = phase_top.loc[phase_top["run_phase"].eq(phase)].iloc[0]
        rank_avg = rank_top.loc[rank_top["run_phase"].eq(phase)].iloc[0]
        top_by_phase.append(
            "- "
            f"{phase}: integrated 1위는 `{integrated.feature_name}/{integrated.segment_setting}` "
            f"(score={integrated.integrated_balanced_score:.4f}), "
            f"rank-average 1위는 `{rank_avg.feature_name}/{rank_avg.segment_setting}` "
            f"(rank_average={rank_avg.rank_average:.2f})입니다."
        )
    score_increase = transitions.sort_values("delta_integrated_late_minus_early", ascending=False).head(3)
    rank_improvement = transitions.sort_values("delta_rank_late_minus_early", ascending=True).head(3)

    sections = ["# H1_S6 Run Phase Feature x Segment Analysis\n"]
    sections.append(
        "## Scope\n\n"
        "Runs were split into early/middle/late thirds within each case. Association, suitability, robustness, integrated score, and rankings were recomputed by phase.\n\n"
        f"- Process file: `{sources['process']}`\n"
        f"- Signal file: `{sources['signal']}`\n"
        f"- Heuristic segment file: `{sources['heuristic']}`\n"
    )
    sections.append(
        "## Korean Executive Summary\n\n"
        "각 case의 run을 시간 순서대로 3등분해 early/middle/late phase를 만들고, phase 안에서 association, suitability, robustness 및 feature x segment 랭킹을 다시 계산했습니다.\n\n"
        + "\n".join(top_by_phase)
        + "\n\n"
        "- 전체적으로 초반과 후반은 `band_energy` 계열이 integrated score 상위권을 유지하지만, 중반은 `max/entry_steady_exit`와 `max/full_length`가 association과 suitability에서 동시에 강해져 1-2위를 차지합니다.\n"
        "- rank-average까지 같이 보면 초반은 `min/steady_exit`, 중반은 `max/entry_steady_exit`, 후반은 `band_energy/steady`가 가장 균형적입니다. 즉, score 기준과 rank 균형 기준의 결론이 phase마다 완전히 같지는 않습니다.\n"
        "- early에서 late로 integrated score가 가장 크게 증가한 조합은 "
        + ", ".join([f"`{r.feature_name}/{r.segment_setting}` (+{r.delta_integrated_late_minus_early:.4f})" for _, r in score_increase.iterrows()])
        + "입니다.\n"
        "- early에서 late로 integrated rank가 가장 크게 개선된 조합은 "
        + ", ".join([f"`{r.feature_name}/{r.segment_setting}` ({int(r.early_integrated_rank_within_phase)} -> {int(r.late_integrated_rank_within_phase)})" for _, r in rank_improvement.iterrows()])
        + "입니다.\n"
        "- 단, 3등분 후 phase별 표본 수가 작아져 association은 `n < 3` 또는 zero variance 조건에서 많이 스킵됩니다. 따라서 S6 결과는 최종 feature 선택보다는 run 진행 단계별 민감도 변화를 보는 탐색 분석으로 해석하는 편이 안전합니다.\n"
    )
    sections.append("## Phase Counts\n\n" + markdown_table(phase_counts, 30) + "\n")
    sections.append(
        "## Calculation Notes\n\n"
        "- Association uses VB-labeled samples inside each case/sensor/phase and skips sequences with `n < 3` or zero variance.\n"
        "- Suitability and robustness use feature sequences inside each case/sensor/phase and require at least 2 samples.\n"
        f"- Association skipped groups: `{len(skipped_assoc)}`.\n"
    )
    for phase in PHASES:
        group = phase_summary.loc[phase_summary["run_phase"].eq(phase)]
        sections.append(
            f"## {phase.title()} Phase\n\n"
            "### Top By Integrated Balanced Score\n\n"
            + markdown_table(
                group.sort_values("integrated_rank_within_phase")[
                    [
                        "feature_name",
                        "segment_setting",
                        "association_score",
                        "suitability_score",
                        "robustness_score",
                        "integrated_balanced_score",
                        "association_rank_within_phase",
                        "suitability_rank_within_phase",
                        "robustness_rank_within_phase",
                        "integrated_rank_within_phase",
                    ]
                ],
                15,
            )
            + "\n\n### Top By Rank Average\n\n"
            + markdown_table(
                group.sort_values("rank_average_within_phase")[
                    [
                        "feature_name",
                        "segment_setting",
                        "association_rank_within_phase",
                        "suitability_rank_within_phase",
                        "robustness_rank_within_phase",
                        "rank_average",
                        "rank_average_within_phase",
                        "integrated_balanced_score",
                    ]
                ],
                15,
            )
            + "\n"
        )
    sections.append(
        "## Early To Late Changes\n\n"
        "Positive `delta_integrated_late_minus_early` means the phase-level integrated score increased from early to late. Negative `delta_rank_late_minus_early` means ranking improved by late phase.\n\n"
        "### Largest Integrated Score Increase\n\n"
        + markdown_table(
            transitions[
                [
                    "feature_name",
                    "segment_setting",
                    "early_integrated_balanced_score",
                    "middle_integrated_balanced_score",
                    "late_integrated_balanced_score",
                    "delta_integrated_late_minus_early",
                    "early_integrated_rank_within_phase",
                    "late_integrated_rank_within_phase",
                    "delta_rank_late_minus_early",
                ]
            ].sort_values("delta_integrated_late_minus_early", ascending=False),
            20,
        )
        + "\n\n### Largest Rank Improvement\n\n"
        + markdown_table(
            transitions[
                [
                    "feature_name",
                    "segment_setting",
                    "early_integrated_rank_within_phase",
                    "middle_integrated_rank_within_phase",
                    "late_integrated_rank_within_phase",
                    "delta_rank_late_minus_early",
                    "delta_integrated_late_minus_early",
                ]
            ].sort_values("delta_rank_late_minus_early", ascending=True),
            20,
        )
        + "\n"
    )
    sections.append(
        "## Interpretation\n\n"
        f"- Integrated-score phase leaders: {', '.join([f'{r.run_phase}: {r.feature_name}/{r.segment_setting}' for _, r in phase_top.iterrows()])}.\n"
        f"- Rank-average phase leaders: {', '.join([f'{r.run_phase}: {r.feature_name}/{r.segment_setting}' for _, r in rank_top.iterrows()])}.\n"
        "- Treat association in each phase cautiously because thirds reduce sample size. The skipped-association table records where phase windows were too small or had zero variance.\n"
    )
    sections.append(
        "## Outputs\n\n"
        "- `analysis/H1_S6_phase_feature_segment_scores.csv`\n"
        "- `analysis/H1_S6_phase_feature_segment_transition.csv`\n"
        "- `analysis/H1_S6_association_case_sensor_phase.csv`\n"
        "- `analysis/H1_S6_suitability_case_sensor_phase.csv`\n"
        "- `analysis/H1_S6_robustness_case_sensor_phase.csv`\n"
        "- `analysis/H1_S6_association_skipped.csv`\n"
    )
    text = "\n".join(sections)
    report_path = output_dir / "reports" / "H1_S6_run_phase_report.md"
    report_path.write_text(text, encoding="utf-8")
    html_text = "<!doctype html><html><head><meta charset='utf-8'><title>H1_S6 Run Phase</title></head><body>" + html.escape(text).replace("\n", "<br>\n") + "</body></html>"
    report_path.with_suffix(".html").write_text(html_text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    process = root / args.process_info_path
    signal = root / args.signal_data_path
    heuristic = root / args.heuristic_sequence_path
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = root / "experiments" / "executions" / "H1" / "S6" / f"{timestamp}_run_phase_feature_segment_evidence_analysis"
    for dirname in ["configs", "data", "analysis", "reports", "logs"]:
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)

    data = load_dataset(process, signal, heuristic)
    data = assign_run_phase(data)
    sensors_df = identify_sensors(data)
    sensors = sensors_df.loc[sensors_df["used"], "sensor_name"].tolist()
    feature_long, construction = find_or_create_entry_steady_exit_features(data, sensors)
    phase_cols = data[["sample_id", "run_phase", "phase_index", "phase_position"]].drop_duplicates()
    feature_long = feature_long.drop(columns=[c for c in ["run_phase", "phase_index", "phase_position"] if c in feature_long.columns])
    feature_long = feature_long.merge(phase_cols, on="sample_id", how="left", validate="many_to_one")

    assoc_case, skipped_assoc = compute_association_case_sensor(feature_long)
    suit_case = compute_suitability_case_sensor(feature_long)
    rob_case = compute_robustness_case_sensor(feature_long)
    phase_summary = build_phase_summary(assoc_case, suit_case, rob_case)
    transitions = build_transition_table(phase_summary)
    phase_counts = (
        data.groupby(["case_id", "run_phase"], dropna=False)
        .agg(num_runs=("run", "size"), run_min=("run", "min"), run_max=("run", "max"), vb_labeled_runs=("VB", lambda s: int(s.notna().sum())))
        .reset_index()
        .sort_values(["case_id", "run_phase"])
    )

    feature_long.to_csv(output_dir / "data" / "H1_S6_feature_long_with_phase.csv", index=False)
    phase_counts.to_csv(output_dir / "data" / "H1_S6_phase_counts.csv", index=False)
    sensors_df.to_csv(output_dir / "data" / "H1_S6_sensor_mapping.csv", index=False)
    assoc_case.to_csv(output_dir / "analysis" / "H1_S6_association_case_sensor_phase.csv", index=False)
    skipped_assoc.to_csv(output_dir / "analysis" / "H1_S6_association_skipped.csv", index=False)
    suit_case.to_csv(output_dir / "analysis" / "H1_S6_suitability_case_sensor_phase.csv", index=False)
    rob_case.to_csv(output_dir / "analysis" / "H1_S6_robustness_case_sensor_phase.csv", index=False)
    phase_summary.to_csv(output_dir / "analysis" / "H1_S6_phase_feature_segment_scores.csv", index=False)
    transitions.to_csv(output_dir / "analysis" / "H1_S6_phase_feature_segment_transition.csv", index=False)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "process_info_path": str(process),
        "signal_data_path": str(signal),
        "heuristic_sequence_path": str(heuristic),
        "phase_rule": "Within each case, sort by run and split rows into three equal-sized chunks with numpy.array_split.",
        "segments": SEGMENT_SETTINGS,
        "features": TARGET_FEATURES,
        "output_dir": str(output_dir),
        "feature_construction": construction,
    }
    (output_dir / "configs" / "H1_S6_run_phase_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    summary = {
        "output_dir": str(output_dir),
        "phase_counts": phase_counts.to_dict(orient="records"),
        "top_integrated_by_phase": phase_summary.sort_values("integrated_rank_within_phase").groupby("run_phase").head(5).to_dict(orient="records"),
        "top_rank_average_by_phase": phase_summary.sort_values("rank_average_within_phase").groupby("run_phase").head(5).to_dict(orient="records"),
        "association_skipped_count": int(len(skipped_assoc)),
    }
    (output_dir / "analysis" / "H1_S6_run_phase_analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_report(output_dir, phase_summary, transitions, phase_counts, skipped_assoc, {"process": str(process), "signal": str(signal), "heuristic": str(heuristic)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H1_S6 early/middle/late run-phase feature x segment analysis.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--process-info-path", default="datasets/processed/mill_process_info_enabled.csv")
    parser.add_argument("--signal-data-path", default="datasets/processed/mill_signal_data_enabled.csv")
    parser.add_argument("--heuristic-sequence-path", default="datasets/metadata/heuristic_sequence_peng2026.csv")
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

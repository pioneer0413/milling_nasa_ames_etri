#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PAIR_DEFINITION = {"A": [1, 9], "B": [2, 12], "C": [8, 14]}
PAIR_BY_CASE = {case: pair for pair, cases in PAIR_DEFINITION.items() for case in cases}
CASE_SCOPE = [1, 2, 8, 9, 12, 14]


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
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


def minmax_by_group(frame: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.Series:
    def scale(s: pd.Series) -> pd.Series:
        lo = s.min(skipna=True)
        hi = s.max(skipna=True)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(np.ones(len(s)), index=s.index, dtype="float64")
        return (s - lo) / (hi - lo)

    return frame.groupby(group_cols, dropna=False)[value_col].transform(scale)


def add_condition_columns(df: pd.DataFrame, case_meta: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(case_meta, on="case_id", how="left", validate="many_to_one")
    missing = out["DOC"].isna().sum()
    if missing:
        raise ValueError(f"Missing condition metadata for {missing} rows")
    return out


def load_case_metadata(process_path: Path) -> pd.DataFrame:
    process = pd.read_csv(process_path)
    process = process.loc[process["enable"].astype(bool) & process["case"].isin(CASE_SCOPE)].copy()
    meta = (
        process.groupby("case", as_index=False)
        .agg(
            DOC=("DOC", "first"),
            feed=("feed", "first"),
            material_name=("material_name", "first"),
            enabled_runs=("run", "count"),
            vb_labeled_runs=("VB", lambda s: int(s.notna().sum())),
            vb_min=("VB", "min"),
            vb_max=("VB", "max"),
        )
        .rename(columns={"case": "case_id"})
    )
    meta["pair_id"] = meta["case_id"].map(PAIR_BY_CASE)
    meta["DOC_label"] = meta["DOC"].map(lambda x: f"DOC={x:g}")
    meta["feed_label"] = meta["feed"].map(lambda x: f"feed={x:g}")
    meta["material_label"] = meta["material_name"].astype(str)
    meta["pair_label"] = meta["pair_id"].map(
        {
            "A": "Pair A: DOC=1.5/feed=0.5/cast_iron",
            "B": "Pair B: DOC=0.75/feed=0.5/cast_iron",
            "C": "Pair C: DOC=0.75/feed=0.5/steel",
        }
    )
    return meta


def load_association(path: Path, case_meta: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path / "metrics" / "association_feature_level.csv")
    df = df.loc[df["status"].eq("ok")].copy()
    df["mi_norm"] = minmax_by_group(df, ["case_id", "sensor_name"], "mutual_information")
    df["association_score"] = (df["pearson_abs"] + df["spearman_abs"] + df["mi_norm"]) / 3.0
    df = add_condition_columns(df, case_meta)
    return df


def load_suitability(path: Path, case_meta: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path / "analysis" / "H1_S2_case_level_suitability_results.csv")
    df = df.loc[df["calculation_status"].eq("ok")].copy()
    df["suitability_score"] = df["suitability_harmonic_mean"]
    df = add_condition_columns(df, case_meta)
    return df


def load_robustness(path: Path, case_meta: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path / "analysis" / "H1_S3_robustness_case_sensor_feature_segment.csv")
    df = df.loc[df["calculation_status"].eq("ok")].copy()
    df["robustness_score"] = df["robustness"]
    df = add_condition_columns(df, case_meta)
    return df


def aggregate_metric(
    df: pd.DataFrame,
    condition_col: str,
    metric_name: str,
    score_col: str,
    extra_aggs: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    group_cols = [condition_col, "feature_name", "segment_setting"]
    agg_spec: dict[str, tuple[str, str]] = {
        f"{metric_name}_score": (score_col, "mean"),
        f"{metric_name}_median": (score_col, "median"),
        f"{metric_name}_std": (score_col, "std"),
        "num_rows": (score_col, "size"),
        "num_cases": ("case_id", "nunique"),
        "num_sensors": ("sensor_name", "nunique"),
    }
    agg_spec.update(extra_aggs)
    out = df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
    out[f"{metric_name}_std"] = out[f"{metric_name}_std"].fillna(0.0)
    out[f"{metric_name}_rank_within_condition"] = out.groupby(condition_col)[f"{metric_name}_score"].rank(ascending=False, method="first").astype(int)
    return out


def build_condition_table(assoc: pd.DataFrame, suit: pd.DataFrame, rob: pd.DataFrame, condition_col: str) -> pd.DataFrame:
    a = aggregate_metric(
        assoc,
        condition_col,
        "association",
        "association_score",
        {
            "association_abs_spearman": ("spearman_abs", "mean"),
            "association_abs_pearson": ("pearson_abs", "mean"),
            "association_mi": ("mutual_information", "mean"),
        },
    )
    s = aggregate_metric(
        suit,
        condition_col,
        "suitability",
        "suitability_score",
        {
            "mean_monotonicity": ("monotonicity", "mean"),
            "mean_trendability": ("trendability", "mean"),
        },
    )
    r = aggregate_metric(
        rob,
        condition_col,
        "robustness",
        "robustness_score",
        {
            "mean_abs_relative_deviation": ("mean_abs_relative_deviation", "mean"),
        },
    )
    key = [condition_col, "feature_name", "segment_setting"]
    out = a.merge(s, on=key, how="outer", suffixes=("", "_suitability"))
    out = out.merge(r, on=key, how="outer", suffixes=("", "_robustness"))
    for col in ["association_score", "suitability_score", "robustness_score"]:
        out[f"{col}_norm"] = minmax_by_group(out, [condition_col], col)
    out["integrated_mean_score"] = out[["association_score_norm", "suitability_score_norm", "robustness_score_norm"]].mean(axis=1)
    denom = (
        1.0 / (out["association_score_norm"] + 1e-12)
        + 1.0 / (out["suitability_score_norm"] + 1e-12)
        + 1.0 / (out["robustness_score_norm"] + 1e-12)
    )
    out["integrated_balanced_score"] = 3.0 / denom
    out["rank_average"] = out[
        [
            "association_rank_within_condition",
            "suitability_rank_within_condition",
            "robustness_rank_within_condition",
        ]
    ].mean(axis=1)
    out["integrated_rank_within_condition"] = out.groupby(condition_col)["integrated_balanced_score"].rank(ascending=False, method="first").astype(int)
    out["rank_average_within_condition"] = out.groupby(condition_col)["rank_average"].rank(ascending=True, method="first").astype(int)
    return out.sort_values([condition_col, "integrated_rank_within_condition"]).reset_index(drop=True)


def summarize_condition_axis(table: pd.DataFrame, condition_col: str) -> pd.DataFrame:
    return (
        table.groupby(condition_col, dropna=False)
        .apply(lambda g: pd.Series({
            "top_integrated": f"{g.sort_values('integrated_rank_within_condition').iloc[0]['feature_name']} / {g.sort_values('integrated_rank_within_condition').iloc[0]['segment_setting']}",
            "top_rank_average": f"{g.sort_values('rank_average_within_condition').iloc[0]['feature_name']} / {g.sort_values('rank_average_within_condition').iloc[0]['segment_setting']}",
            "mean_association": g["association_score"].mean(),
            "mean_suitability": g["suitability_score"].mean(),
            "mean_robustness": g["robustness_score"].mean(),
            "mean_integrated": g["integrated_balanced_score"].mean(),
            "num_feature_segments": len(g),
        }))
        .reset_index()
    )


def write_condition_report(output_dir: Path, axis_name: str, condition_col: str, table: pd.DataFrame, summary: pd.DataFrame, note: str = "") -> None:
    sections = [f"# H1_S5 {axis_name} Condition Feature x Segment Report\n"]
    if note:
        sections.append(f"## Note\n\n{note}\n")
    sections.append("## Condition Summary\n\n" + markdown_table(summary, 20) + "\n")
    for value, group in table.groupby(condition_col, dropna=False):
        top_integrated = group.sort_values("integrated_rank_within_condition").head(12)
        top_rank = group.sort_values("rank_average_within_condition").head(12)
        sections.append(
            f"## {condition_col} = {value}\n\n"
            "### Top By Integrated Balanced Score\n\n"
            + markdown_table(
                top_integrated[
                    [
                        "feature_name",
                        "segment_setting",
                        "association_score",
                        "suitability_score",
                        "robustness_score",
                        "integrated_balanced_score",
                        "association_rank_within_condition",
                        "suitability_rank_within_condition",
                        "robustness_rank_within_condition",
                        "integrated_rank_within_condition",
                    ]
                ],
                12,
            )
            + "\n\n### Top By Rank Average\n\n"
            + markdown_table(
                top_rank[
                    [
                        "feature_name",
                        "segment_setting",
                        "association_rank_within_condition",
                        "suitability_rank_within_condition",
                        "robustness_rank_within_condition",
                        "rank_average",
                        "rank_average_within_condition",
                        "integrated_balanced_score",
                    ]
                ],
                12,
            )
            + "\n"
        )
    report_path = output_dir / "reports" / f"H1_S5_{axis_name.lower()}_condition_report.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")
    html_text = "<!doctype html><html><head><meta charset='utf-8'><title>H1_S5</title></head><body>" + html.escape("\n".join(sections)).replace("\n", "<br>\n") + "</body></html>"
    report_path.with_suffix(".html").write_text(html_text, encoding="utf-8")


def write_master_report(output_dir: Path, case_meta: pd.DataFrame, summaries: dict[str, pd.DataFrame], notes: dict[str, str], sources: dict[str, str]) -> None:
    text = f"""# H1_S5 Process Condition Feature x Segment Analysis

## Scope

Association, suitability, and robustness were re-aggregated by process-condition axes: DoC, Feed, Material, and Pair.

## Sources

- S1 association: `{sources['S1']}`
- S2 suitability: `{sources['S2']}`
- S3 robustness: `{sources['S3']}`
- Process metadata: `{sources['process']}`

## Case To Condition Mapping

{markdown_table(case_meta[['case_id','pair_id','DOC','feed','material_name','enabled_runs','vb_labeled_runs','vb_min','vb_max']], 20)}

## Axis Notes

- DoC: compares `0.75` vs `1.5`, but material is not fully balanced because steel appears only at DoC `0.75`.
- Feed: {notes['Feed']}
- Material: compares `cast_iron` vs `steel`, but steel appears only in Pair C and DoC `0.75`.
- Pair-wise: Pair A=`DOC 1.5/cast_iron`, Pair B=`DOC 0.75/cast_iron`, Pair C=`DOC 0.75/steel`; this is the cleanest condition-level view in this dataset.

## DoC Summary

{markdown_table(summaries['DoC'], 20)}

## Feed Summary

{markdown_table(summaries['Feed'], 20)}

## Material Summary

{markdown_table(summaries['Material'], 20)}

## Pair-wise Summary

{markdown_table(summaries['Pair'], 20)}

## Interpretation

Because Feed has only one level, no feed-effect conclusion should be drawn. DoC and Material are partially confounded, so the Pair-wise view is the most honest process-condition summary: it preserves the actual case-pair protocol and keeps the observed process/material combinations intact. Use DoC and Material views as descriptive slices, not causal isolation.

## Outputs

- `analysis/H1_S5_DoC_condition_feature_segment.csv`
- `analysis/H1_S5_Feed_condition_feature_segment.csv`
- `analysis/H1_S5_Material_condition_feature_segment.csv`
- `analysis/H1_S5_Pair_condition_feature_segment.csv`
- `reports/H1_S5_doc_condition_report.md`
- `reports/H1_S5_feed_condition_report.md`
- `reports/H1_S5_material_condition_report.md`
- `reports/H1_S5_pair_condition_report.md`
"""
    report_path = output_dir / "reports" / "H1_S5_process_condition_master_report.md"
    report_path.write_text(text, encoding="utf-8")
    html_text = "<!doctype html><html><head><meta charset='utf-8'><title>H1_S5 Process Conditions</title></head><body>" + html.escape(text).replace("\n", "<br>\n") + "</body></html>"
    report_path.with_suffix(".html").write_text(html_text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    s1 = Path(args.s1)
    s2 = Path(args.s2)
    s3 = Path(args.s3)
    process_path = Path(args.process_info_path)
    if not s1.is_absolute():
        s1 = root / s1
    if not s2.is_absolute():
        s2 = root / s2
    if not s3.is_absolute():
        s3 = root / s3
    if not process_path.is_absolute():
        process_path = root / process_path

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = root / "experiments" / "executions" / "H1" / "S5" / f"{timestamp}_process_condition_feature_segment_evidence_analysis"
    for dirname in ["configs", "analysis", "reports", "logs"]:
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)

    case_meta = load_case_metadata(process_path)
    assoc = load_association(s1, case_meta)
    suit = load_suitability(s2, case_meta)
    rob = load_robustness(s3, case_meta)

    axes = {
        "DoC": ("DOC_label", ""),
        "Feed": ("feed_label", "All selected H1 cases have feed=0.5, so this axis is descriptive only and cannot compare feed levels."),
        "Material": ("material_label", ""),
        "Pair": ("pair_label", ""),
    }
    summaries: dict[str, pd.DataFrame] = {}
    notes: dict[str, str] = {}
    for axis_name, (condition_col, note) in axes.items():
        table = build_condition_table(assoc, suit, rob, condition_col)
        summary = summarize_condition_axis(table, condition_col)
        table.to_csv(output_dir / "analysis" / f"H1_S5_{axis_name}_condition_feature_segment.csv", index=False)
        summary.to_csv(output_dir / "analysis" / f"H1_S5_{axis_name}_condition_summary.csv", index=False)
        write_condition_report(output_dir, axis_name, condition_col, table, summary, note)
        summaries[axis_name] = summary
        notes[axis_name] = note

    sources = {"S1": str(s1), "S2": str(s2), "S3": str(s3), "process": str(process_path)}
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sources": sources,
        "case_scope": CASE_SCOPE,
        "pair_definition": PAIR_DEFINITION,
        "axes": {name: col for name, (col, _) in axes.items()},
        "output_dir": str(output_dir),
    }
    (output_dir / "configs" / "H1_S5_process_condition_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    case_meta.to_csv(output_dir / "data_case_condition_mapping.csv", index=False)
    write_master_report(output_dir, case_meta, summaries, notes, sources)
    summary_json = {
        "output_dir": str(output_dir),
        "summaries": {key: value.to_dict(orient="records") for key, value in summaries.items()},
    }
    (output_dir / "analysis" / "H1_S5_process_condition_analysis_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H1_S5 process-condition feature x segment evidence analysis.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--s1", default="experiments/executions/H1/S1/2026-05-20_104038_589496_feture_VB_segment_association_eda")
    parser.add_argument("--s2", default="experiments/executions/H1/S2/20260520_104913_feature_VB_segment_suitability_eda")
    parser.add_argument("--s3", default="experiments/executions/H1/S3/2026-05-20_110004_robustness_segment_feature_analysis")
    parser.add_argument("--process-info-path", default="datasets/processed/mill_process_info_enabled.csv")
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

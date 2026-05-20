#!/usr/bin/env python3
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
S5_DIR = ROOT / "experiments" / "executions" / "H1" / "S5" / "2026-05-20_150438_process_condition_feature_segment_evidence_analysis"
ANALYSIS_DIR = S5_DIR / "analysis"
REPORT_DIR = S5_DIR / "reports"


def fmt(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        frame = frame.head(max_rows)
    if frame.empty:
        return "_No rows._"
    rows = []
    cols = list(frame.columns)
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(rows)


def load_axis(axis: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    label_col = {
        "DoC": "DOC_label",
        "Feed": "feed_label",
        "Material": "material_label",
        "Pair": "pair_label",
    }[axis]
    detail = pd.read_csv(ANALYSIS_DIR / f"H1_S5_{axis}_condition_feature_segment.csv")
    summary = pd.read_csv(ANALYSIS_DIR / f"H1_S5_{axis}_condition_summary.csv")
    return detail, summary, label_col


def condition_top(detail: pd.DataFrame, label_col: str, rank_col: str, cols: list[str], n: int = 5) -> pd.DataFrame:
    out = (
        detail.sort_values([label_col, rank_col])
        .groupby(label_col, as_index=False)
        .head(n)
        .loc[:, [label_col] + cols]
    )
    return out


def best_full_vs_nonfull(detail: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    for label, group in detail.groupby(label_col):
        full = group.loc[group["segment_setting"].eq("full_length")].sort_values("integrated_rank_within_condition").head(1)
        nonfull = group.loc[~group["segment_setting"].eq("full_length")].sort_values("integrated_rank_within_condition").head(1)
        if full.empty or nonfull.empty:
            continue
        full_row = full.iloc[0]
        nonfull_row = nonfull.iloc[0]
        rows.append(
            {
                label_col: label,
                "best_no_load_excluded": f"{nonfull_row.feature_name} / {nonfull_row.segment_setting}",
                "no_load_integrated": nonfull_row.integrated_balanced_score,
                "no_load_rank": int(nonfull_row.integrated_rank_within_condition),
                "best_full_length": f"{full_row.feature_name} / {full_row.segment_setting}",
                "full_length_integrated": full_row.integrated_balanced_score,
                "full_length_rank": int(full_row.integrated_rank_within_condition),
                "no_load_minus_full": nonfull_row.integrated_balanced_score - full_row.integrated_balanced_score,
            }
        )
    return pd.DataFrame(rows)


def top_pattern_counts(detail: pd.DataFrame, label_col: str, rank_col: str, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for label, group in detail.groupby(label_col):
        top = group.sort_values(rank_col).head(top_n)
        segment_counts = ", ".join(f"{k}:{v}" for k, v in top["segment_setting"].value_counts().items())
        feature_counts = ", ".join(f"{k}:{v}" for k, v in top["feature_name"].value_counts().items())
        rows.append(
            {
                label_col: label,
                "top_n": top_n,
                "segment_counts": segment_counts,
                "feature_counts": feature_counts,
            }
        )
    return pd.DataFrame(rows)


def summary_delta(summary: pd.DataFrame, label_col: str, first: str, second: str) -> pd.DataFrame:
    a = summary.loc[summary[label_col].eq(first)].iloc[0]
    b = summary.loc[summary[label_col].eq(second)].iloc[0]
    metrics = ["mean_association", "mean_suitability", "mean_robustness", "mean_integrated"]
    return pd.DataFrame(
        [
            {
                "comparison": f"{second} - {first}",
                **{f"delta_{m.replace('mean_', '')}": b[m] - a[m] for m in metrics},
            }
        ]
    )


def main() -> None:
    doc_detail, doc_summary, doc_label = load_axis("DoC")
    feed_detail, feed_summary, feed_label = load_axis("Feed")
    material_detail, material_summary, material_label = load_axis("Material")
    pair_detail, pair_summary, pair_label = load_axis("Pair")
    case_map = pd.read_csv(S5_DIR / "data_case_condition_mapping.csv")

    top_cols = [
        "feature_name",
        "segment_setting",
        "association_score",
        "suitability_score",
        "robustness_score",
        "integrated_balanced_score",
        "rank_average",
        "integrated_rank_within_condition",
        "rank_average_within_condition",
    ]
    rank_cols = [
        "feature_name",
        "segment_setting",
        "association_rank_within_condition",
        "suitability_rank_within_condition",
        "robustness_rank_within_condition",
        "rank_average",
        "rank_average_within_condition",
        "integrated_balanced_score",
    ]

    doc_delta = summary_delta(doc_summary, doc_label, "DOC=0.75", "DOC=1.5")
    material_delta = summary_delta(material_summary, material_label, "cast_iron", "steel")

    sections = [
        "# H1_S5 Process Condition Result Interpretation\n",
        "## 분석 목적\n\n"
        "S5 산출물의 DoC, Feed, Material, Pair-wise 조건축 결과를 다시 읽어 `feature x segment` 후보가 공정 조건에 따라 어떻게 달라지는지 해석 중심으로 정리했습니다. "
        "Association은 VB와의 관련성, suitability는 run 진행에 따른 monotonicity/trendability, robustness는 평균 trend 대비 흔들림 정도를 의미합니다.\n",
        "## 핵심 결론\n\n"
        "1. `band_energy`는 거의 모든 조건축에서 integrated score 기준 최상위입니다. 다만 어떤 segment가 붙는지는 조건에 따라 바뀝니다.\n"
        "2. rank-average 기준에서는 `mean` 계열이 더 자주 1위입니다. 즉, score 크기 기준은 `band_energy`, 세 지표 랭킹 균형 기준은 `mean`이 강합니다.\n"
        "3. Feed는 모든 H1 case가 `feed=0.5`라서 조건 비교가 불가능합니다. Feed 결과는 단일 조건의 descriptive summary로만 봐야 합니다.\n"
        "4. DoC와 Material은 완전히 독립적인 비교가 아닙니다. steel case가 DoC=0.75에만 있으므로, DoC 효과와 material 효과가 일부 얽혀 있습니다.\n"
        "5. 가장 정직한 조건 관점은 Pair-wise입니다. Pair A, B, C는 실제 관측된 공정 조합을 그대로 보존하기 때문입니다.\n"
        "6. full_length도 강한 경우가 있지만, no-load 제외 segment가 top integrated에 반복적으로 등장합니다. 특히 Pair A는 `entry_exit`, Pair C는 `steady`가 최상위입니다.\n",
        "## Case To Condition Mapping\n\n" + markdown_table(case_map) + "\n",
        "## Condition Summary\n\n"
        "### DoC\n\n"
        + markdown_table(doc_summary)
        + "\n\n### Material\n\n"
        + markdown_table(material_summary)
        + "\n\n### Pair-wise\n\n"
        + markdown_table(pair_summary)
        + "\n\n### Feed\n\n"
        + markdown_table(feed_summary)
        + "\n",
        "## Condition-Level Metric Differences\n\n"
        "DoC=1.5는 DoC=0.75보다 평균 association과 robustness가 높지만, 평균 suitability는 낮습니다. Material 기준으로는 steel이 suitability는 높고 association/robustness는 낮습니다. "
        "다만 이 차이는 조건이 완전 균형 설계가 아니므로 causal effect로 읽으면 안 됩니다.\n\n"
        "### DoC Delta\n\n"
        + markdown_table(doc_delta)
        + "\n\n### Material Delta\n\n"
        + markdown_table(material_delta)
        + "\n",
        "## DoC View\n\n"
        "DoC=0.75에서는 `band_energy / entry_steady_exit`가 integrated 1위이고, `band_energy / full_length`, `band_energy / steady`가 바로 뒤를 따릅니다. "
        "DoC=1.5에서는 `band_energy / entry_exit`가 integrated 1위입니다. 이는 높은 DoC 조건에서 steady-only보다 entry/exit를 포함한 transient 반응이 VB 관련성에 더 강하게 잡힐 가능성을 시사합니다.\n\n"
        "### Top Integrated By DoC\n\n"
        + markdown_table(condition_top(doc_detail, doc_label, "integrated_rank_within_condition", top_cols), 12)
        + "\n\n### Top Rank-Average By DoC\n\n"
        + markdown_table(condition_top(doc_detail, doc_label, "rank_average_within_condition", rank_cols), 12)
        + "\n",
        "## Material View\n\n"
        "cast_iron은 `band_energy / full_length`가 integrated 1위이고, steel은 `band_energy / steady`가 integrated 1위입니다. "
        "steel에서는 suitability 평균이 더 높지만 association과 robustness 평균은 낮습니다. 이 결과는 steel 조건에서 feature sequence의 진행성은 비교적 잘 보이지만 VB 직접 관련성은 cast_iron보다 약하게 집계되었음을 뜻합니다.\n\n"
        "### Top Integrated By Material\n\n"
        + markdown_table(condition_top(material_detail, material_label, "integrated_rank_within_condition", top_cols), 12)
        + "\n\n### Top Rank-Average By Material\n\n"
        + markdown_table(condition_top(material_detail, material_label, "rank_average_within_condition", rank_cols), 12)
        + "\n",
        "## Pair-wise View\n\n"
        "Pair-wise가 S5에서 가장 해석력이 좋은 관점입니다. Pair A는 DoC=1.5/cast_iron, Pair B는 DoC=0.75/cast_iron, Pair C는 DoC=0.75/steel입니다. "
        "Pair B는 평균 integrated가 가장 높고, Pair A는 association과 robustness가 높으며, Pair C는 suitability가 가장 높습니다.\n\n"
        "### Top Integrated By Pair\n\n"
        + markdown_table(condition_top(pair_detail, pair_label, "integrated_rank_within_condition", top_cols), 15)
        + "\n\n### Top Rank-Average By Pair\n\n"
        + markdown_table(condition_top(pair_detail, pair_label, "rank_average_within_condition", rank_cols), 15)
        + "\n",
        "## Full-Length vs No-Load Excluded Segments\n\n"
        "아래 표는 각 조건에서 integrated score 기준 최상위 no-load 제외 segment와 최상위 full_length 조합을 비교한 것입니다. "
        "`no_load_minus_full`이 양수면 no-load 제외 segment가 full_length보다 유리합니다.\n\n"
        "### DoC\n\n"
        + markdown_table(best_full_vs_nonfull(doc_detail, doc_label))
        + "\n\n### Material\n\n"
        + markdown_table(best_full_vs_nonfull(material_detail, material_label))
        + "\n\n### Pair-wise\n\n"
        + markdown_table(best_full_vs_nonfull(pair_detail, pair_label))
        + "\n",
        "## Top-10 Pattern Counts\n\n"
        "상위 10개 후보 안에서 어떤 segment와 feature가 반복되는지 집계했습니다. Integrated 기준에서는 `band_energy`가 반복되고, rank-average 기준에서는 `mean`, `min`, `max` 같은 통계 feature가 더 균형적으로 나타납니다.\n\n"
        "### Integrated Top-10 Patterns By Pair\n\n"
        + markdown_table(top_pattern_counts(pair_detail, pair_label, "integrated_rank_within_condition"))
        + "\n\n### Rank-Average Top-10 Patterns By Pair\n\n"
        + markdown_table(top_pattern_counts(pair_detail, pair_label, "rank_average_within_condition"))
        + "\n",
        "## 해석 및 추천\n\n"
        "- 예측 feature 후보를 뽑는다면 `band_energy`를 primary feature로 유지하는 것이 타당합니다. 단, 현재 프레임워크의 `band_energy`는 특정 협대역 에너지라기보다 full-spectrum FFT energy에 가까우므로 이름 해석에는 주의가 필요합니다.\n"
        "- 조건별 segment는 하나로 고정하기보다 Pair-wise로 다르게 가져가는 편이 자연스럽습니다. Pair A는 `entry_exit`, Pair B는 `full_length` 또는 `entry_steady_exit`, Pair C는 `steady`가 강합니다.\n"
        "- no-load 제외 segment의 이점은 S5에서도 반복됩니다. full_length는 baseline/control로 남기되, 모델 입력 후보의 우선순위는 `entry_steady_exit`, `steady`, `entry_exit`, `steady_exit` 쪽에 두는 것이 좋아 보입니다.\n"
        "- rank-average까지 고려하면 `mean` 계열을 버리면 안 됩니다. `band_energy`가 score를 크게 끌어올리는 후보라면, `mean`은 association/suitability/robustness 랭킹 균형이 좋은 보완 후보입니다.\n"
        "- Feed 효과는 현재 데이터로 판단할 수 없습니다. Feed 조건을 논하려면 feed level이 다른 case를 포함한 재실험 또는 별도 split이 필요합니다.\n",
        "## Source Files\n\n"
        "- `analysis/H1_S5_DoC_condition_feature_segment.csv`\n"
        "- `analysis/H1_S5_Feed_condition_feature_segment.csv`\n"
        "- `analysis/H1_S5_Material_condition_feature_segment.csv`\n"
        "- `analysis/H1_S5_Pair_condition_feature_segment.csv`\n"
        "- `reports/H1_S5_process_condition_master_report.md`\n",
    ]

    report_text = "\n".join(sections)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "H1_S5_process_condition_interpretive_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    html_text = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>H1_S5 Process Condition Interpretation</title></head><body>"
        + html.escape(report_text).replace("\n", "<br>\n")
        + "</body></html>"
    )
    report_path.with_suffix(".html").write_text(html_text, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()

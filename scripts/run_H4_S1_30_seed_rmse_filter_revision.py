#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

PREFIX = "H4_S1"
EXPERIMENT_NUMBER = "H4_S1"
DEFAULT_EXECUTION_DIR = (
    ROOT
    / "experiments"
    / "executions"
    / "H4"
    / "S1"
    / "2026-05-11_140944_cnn_gru_run_sequence_domain_generalization_NASA_Ames_extended_condition_pairs"
)
PAIRS = {
    "A": [1, 9],
    "B": [2, 12],
    "C": [8, 14],
    "D": [3, 11],
    "E": [4, 10],
    "F": [7, 13],
}
MODELS = ["feature_ridge", "feature_random_forest", "feature_svr", "feature_xgboost", "cnn_only", "cnn_lag_concat", "cnn_gru"]
SEEDS_30 = list(range(30))
THRESHOLD = 0.5


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, indent=2, ensure_ascii=False)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        write_json(path.with_suffix(".json"), payload)
        return
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_builtin(payload), f, sort_keys=False, allow_unicode=True)


def scenario_pairs() -> list[tuple[str, str]]:
    return [(source, target) for source in PAIRS for target in PAIRS if source != target]


def scenario_name(source: str, target: str) -> str:
    return f"{source}_to_{target}"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_seed_completion_status(output_dir: Path, target_metrics: pd.DataFrame) -> pd.DataFrame:
    existing = set()
    if not target_metrics.empty:
        existing = set(
            zip(
                target_metrics["scenario_name"].astype(str),
                target_metrics["source_domain"].astype(str),
                target_metrics["target_domain"].astype(str),
                target_metrics["model_name"].astype(str),
                target_metrics["seed"].astype(int),
            )
        )
    rows: list[dict[str, Any]] = []
    for source, target in scenario_pairs():
        scen = scenario_name(source, target)
        for model in MODELS:
            for seed in SEEDS_30:
                key = (scen, source, target, model, seed)
                exists = key in existing
                rows.append(
                    {
                        "experiment_id": target_metrics["experiment_id"].iloc[0] if not target_metrics.empty else "",
                        "experiment_number": EXPERIMENT_NUMBER,
                        "scenario_name": scen,
                        "source_domain": source,
                        "target_domain": target,
                        "model_name": model,
                        "sensor_subset": "not_applicable",
                        "seed": seed,
                        "expected_seed_count": 30,
                        "status": "reused" if exists else "missing",
                        "existing_result_path": str(output_dir) if exists else "",
                        "result_path": str(output_dir) if exists else "",
                        "needs_execution": not exists,
                        "reason": "existing result reused" if exists else "seed result missing for 30-seed revision",
                    }
                )
    return pd.DataFrame(rows)


def filter_rmse_for_visualization(
    df: pd.DataFrame,
    rmse_col: str,
    threshold: float = THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    if out.empty or rmse_col not in out.columns:
        summary = out.copy()
        summary["rmse_value"] = np.nan
        summary["included_in_visualization"] = False
        summary["reason"] = "rmse_missing"
        return out.iloc[0:0].copy(), summary
    out["rmse_value"] = pd.to_numeric(out[rmse_col], errors="coerce")
    out["included_in_visualization"] = out["rmse_value"].le(threshold)
    out.loc[out["rmse_value"].isna(), "included_in_visualization"] = False
    out["reason"] = np.where(out["included_in_visualization"], "included_rmse_le_threshold", "excluded_rmse_gt_threshold")
    out.loc[out["rmse_value"].isna(), "reason"] = "rmse_missing"
    return out.loc[out["included_in_visualization"]].copy(), out


def _filter_summary_rows(source: pd.DataFrame, figure_name: str, plot_type: str, filtering_level: str) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "experiment_number",
        "figure_name",
        "plot_type",
        "filtering_level",
        "scenario_name",
        "source_domain",
        "target_domain",
        "case_id",
        "model_name",
        "sensor_subset",
        "seed",
        "rmse_value",
        "threshold",
        "included_in_visualization",
        "reason",
    ]
    if source.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(index=source.index)
    out["experiment_id"] = source.get("experiment_id", "")
    out["experiment_number"] = EXPERIMENT_NUMBER
    out["figure_name"] = figure_name
    out["plot_type"] = plot_type
    out["filtering_level"] = filtering_level
    for col in ["scenario_name", "source_domain", "target_domain", "model_name", "seed"]:
        out[col] = source[col] if col in source.columns else np.nan
    out["case_id"] = source["case_id"] if "case_id" in source.columns else np.nan
    out["sensor_subset"] = source["sensor_subset"] if "sensor_subset" in source.columns else "not_applicable"
    out["rmse_value"] = source.get("rmse_value", np.nan)
    out["threshold"] = THRESHOLD
    out["included_in_visualization"] = source.get("included_in_visualization", False)
    out["reason"] = source.get("reason", "not_applicable")
    return out[columns]


def add_caption() -> None:
    plt.figtext(
        0.5,
        0.005,
        "Note: RMSE > 0.5 conditions are excluded from this visualization only; all metrics and rankings remain unfiltered.",
        ha="center",
        fontsize=7,
    )


def plot_model_comparison(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(9, 4.8))
    plt.bar(df["model_name"], df["mean_RMSE"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean RMSE, filtered for plot")
    add_caption()
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(path, dpi=180)
    plt.close()


def plot_heatmap(df: pd.DataFrame, value_col: str, index_col: str, column_col: str, path: Path, label: str) -> None:
    if df.empty:
        return
    pivot = df.pivot(index=index_col, columns=column_col, values=value_col)
    if pivot.empty:
        return
    plt.figure(figsize=(8, 5))
    masked = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    im = plt.imshow(masked, cmap="viridis")
    plt.colorbar(im, label=label)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    add_caption()
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(path, dpi=180)
    plt.close()


def plot_pair_difficulty(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(7, 4.5))
    plt.bar(df["target_domain"], df["mean_RMSE_when_target"])
    plt.ylabel("Mean RMSE as target, filtered for plot")
    add_caption()
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(path, dpi=180)
    plt.close()


def regenerate_filtered_visualizations(output_dir: Path) -> tuple[pd.DataFrame, dict[str, Path], dict[str, int]]:
    target_metrics = load_csv(output_dir / "metrics" / f"{PREFIX}_extended_target_domain_metrics.csv")
    model_summary = load_csv(output_dir / "analysis" / f"{PREFIX}_extended_ABCDEF_scope_summary.csv")
    source_summary = load_csv(output_dir / "analysis" / f"{PREFIX}_pair_as_source_summary.csv")
    target_summary = load_csv(output_dir / "analysis" / f"{PREFIX}_pair_as_target_summary.csv")
    matrix = load_csv(output_dir / "analysis" / f"{PREFIX}_transfer_scenario_rmse_matrix.csv")

    fig_dir = output_dir / "figures" / "revision_rmse_filtered_0p5"
    fig_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = output_dir / "analysis"

    all_filter_rows: list[pd.DataFrame] = []
    output_paths: dict[str, Path] = {}
    exclusion_counts: dict[str, int] = {}

    model_plot = model_summary.rename(columns={"mean_RMSE_over_scenarios": "mean_RMSE"}).copy()
    if "mean_RMSE" not in model_plot.columns and "mean_RMSE_over_6_shifts" in model_plot.columns:
        model_plot = model_plot.rename(columns={"mean_RMSE_over_6_shifts": "mean_RMSE"})
    filtered, summary = filter_rmse_for_visualization(model_plot, "mean_RMSE")
    filtered.to_csv(analysis_dir / f"{PREFIX}_extended_model_comparison_for_plot_filtered_0p5.csv", index=False)
    figure = f"{PREFIX}_extended_model_comparison_rmse.png"
    plot_model_comparison(filtered, fig_dir / figure)
    all_filter_rows.append(_filter_summary_rows(summary, figure, "bar", "model_aggregate"))
    output_paths[figure] = fig_dir / figure
    exclusion_counts[figure] = int((~summary["included_in_visualization"]).sum()) if not summary.empty else 0

    filtered, summary = filter_rmse_for_visualization(source_summary, "mean_RMSE_when_source")
    filtered.to_csv(analysis_dir / f"{PREFIX}_pair_as_source_summary_for_plot_filtered_0p5.csv", index=False)
    figure = f"{PREFIX}_pair_as_source_rmse_heatmap.png"
    plot_heatmap(filtered, "mean_RMSE_when_source", "model_name", "source_domain", fig_dir / figure, "Mean RMSE as source")
    all_filter_rows.append(_filter_summary_rows(summary, figure, "heatmap", "pair_model_aggregate"))
    output_paths[figure] = fig_dir / figure
    exclusion_counts[figure] = int((~summary["included_in_visualization"]).sum()) if not summary.empty else 0

    filtered, summary = filter_rmse_for_visualization(target_summary, "mean_RMSE_when_target")
    filtered.to_csv(analysis_dir / f"{PREFIX}_pair_as_target_summary_for_plot_filtered_0p5.csv", index=False)
    figure = f"{PREFIX}_pair_as_target_rmse_heatmap.png"
    plot_heatmap(filtered, "mean_RMSE_when_target", "model_name", "target_domain", fig_dir / figure, "Mean RMSE as target")
    all_filter_rows.append(_filter_summary_rows(summary, figure, "heatmap", "pair_model_aggregate"))
    output_paths[figure] = fig_dir / figure
    exclusion_counts[figure] = int((~summary["included_in_visualization"]).sum()) if not summary.empty else 0

    pair_diff = target_summary.groupby("target_domain", as_index=False)["mean_RMSE_when_target"].mean()
    filtered, summary = filter_rmse_for_visualization(pair_diff, "mean_RMSE_when_target")
    figure = f"{PREFIX}_pair_difficulty_summary.png"
    plot_pair_difficulty(filtered.sort_values("mean_RMSE_when_target", ascending=False), fig_dir / figure)
    all_filter_rows.append(_filter_summary_rows(summary, figure, "bar", "pair_aggregate"))
    output_paths[figure] = fig_dir / figure
    exclusion_counts[figure] = int((~summary["included_in_visualization"]).sum()) if not summary.empty else 0

    filtered, summary = filter_rmse_for_visualization(matrix, "mean_RMSE")
    filtered.to_csv(analysis_dir / f"{PREFIX}_transfer_scenario_rmse_matrix_for_plot_filtered_0p5.csv", index=False)
    best_model = filtered.groupby("model_name")["mean_RMSE"].mean().sort_values().index[0] if not filtered.empty else None
    figure = f"{PREFIX}_transfer_scenario_rmse_matrix.png"
    if best_model:
        plot_heatmap(
            filtered.loc[filtered["model_name"] == best_model],
            "mean_RMSE",
            "source_domain",
            "target_domain",
            fig_dir / figure,
            f"{best_model} mean RMSE",
        )
    all_filter_rows.append(_filter_summary_rows(summary, figure, "heatmap", "scenario_model_aggregate"))
    output_paths[figure] = fig_dir / figure
    exclusion_counts[figure] = int((~summary["included_in_visualization"]).sum()) if not summary.empty else 0

    filter_summary = pd.concat(all_filter_rows, ignore_index=True) if all_filter_rows else pd.DataFrame()
    filter_summary.to_csv(analysis_dir / f"{PREFIX}_rmse_visualization_filter_summary.csv", index=False)
    return filter_summary, output_paths, exclusion_counts


def update_config_and_validation(output_dir: Path, seed_status: pd.DataFrame, filter_summary: pd.DataFrame) -> None:
    config_path = output_dir / "configs" / f"{PREFIX}_resolved_config.yaml"
    config = read_yaml(config_path)
    config.setdefault("experiment", {})
    config["experiment"]["seed_list"] = SEEDS_30
    config["experiment"]["num_seeds"] = 30
    config.setdefault("visualization", {})
    config["visualization"].update(
        {
            "rmse_filter_enabled": True,
            "rmse_filter_threshold": THRESHOLD,
            "rmse_filter_metric": "RMSE",
            "rmse_filter_operator": ">",
            "exclude_filtered_from_metrics": False,
            "exclude_filtered_from_ranking": False,
            "exclude_filtered_from_report_tables": False,
        }
    )
    write_yaml(config_path, config)

    validation_path = output_dir / "configs" / f"{PREFIX}_config_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    validation.update(
        {
            "seed_list_requested_0_29": True,
            "requested_seed_count": 30,
            "completed_seed_count": int(seed_status.loc[seed_status["status"].eq("reused"), "seed"].nunique()),
            "missing_seed_condition_count": int(seed_status["needs_execution"].sum()),
            "metrics_include_rmse_gt_0p5": True,
            "ranking_include_rmse_gt_0p5": True,
            "rmse_visualization_filter_enabled": True,
            "rmse_visualization_filter_threshold": THRESHOLD,
            "rmse_visualization_filter_summary_rows": int(len(filter_summary)),
            "exclude_filtered_from_metrics": False,
            "exclude_filtered_from_ranking": False,
            "exclude_filtered_from_report_tables": False,
        }
    )
    write_json(validation_path, validation)


def write_revision_report(
    output_dir: Path,
    seed_status: pd.DataFrame,
    filter_summary: pd.DataFrame,
    figure_paths: dict[str, Path],
) -> None:
    reused_count = int(seed_status["status"].eq("reused").sum())
    missing_count = int(seed_status["status"].eq("missing").sum())
    executed_count = int(seed_status["status"].eq("executed").sum()) if "executed" in set(seed_status["status"]) else 0
    excluded_count = int((~filter_summary["included_in_visualization"]).sum()) if not filter_summary.empty else 0
    completed_seeds = sorted(seed_status.loc[seed_status["status"].eq("reused"), "seed"].unique().tolist())
    missing_seeds = sorted(seed_status.loc[seed_status["status"].eq("missing"), "seed"].unique().tolist())
    figure_list = "\n".join(f"- `{path.relative_to(output_dir)}`" for path in figure_paths.values() if path.exists())
    report = f"""# H4_S1 30 Seed and RMSE Visualization Filter Revision

## Summary

- Experiment number: `{EXPERIMENT_NUMBER}`
- Source execution directory: `{output_dir}`
- Requested seed list: `{SEEDS_30}`
- Existing completed seeds found in metrics: `{completed_seeds}`
- Missing seeds in current metrics: `{missing_seeds}`
- Reused seed conditions: {reused_count}
- Newly executed seed conditions in this revision step: {executed_count}
- Missing seed conditions still requiring model execution: {missing_count}

## RMSE Visualization Filtering

RMSE > {THRESHOLD} conditions were excluded from RMSE-based figures only. They were not removed from:

- metrics CSV files
- model ranking
- target-domain aggregate metrics
- case-level metrics
- seed-level metrics
- report numeric tables

The exclusion list is saved at `analysis/{PREFIX}_rmse_visualization_filter_summary.csv`.

Filtered figures regenerated in this revision:

{figure_list if figure_list else '- No filtered figure had remaining data after filtering.'}

Visualization-only excluded rows across regenerated figures: {excluded_count}

## Seed Completion

Seed completion status is saved at:

- `analysis/{PREFIX}_seed_completion_status.csv`
- `analysis/{PREFIX}_reuse_plan_for_30_seed_revision.csv`

The current directory contains complete 20-seed results. Seeds 20-29 require additional model execution to make the full 30-seed metric tables complete.

## Notes

This revision preserves original metric files. Filtered plot data are written to separate `*_for_plot_filtered_0p5.csv` files. The model ranking and all metric summaries remain unfiltered.
"""
    (output_dir / "reports" / f"{PREFIX}_30_seed_rmse_filter_revision.md").write_text(report, encoding="utf-8")
    html = "<html><body>" + report.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>\n") + "</body></html>"
    (output_dir / "reports" / f"{PREFIX}_30_seed_rmse_filter_revision.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-dir", default=str(DEFAULT_EXECUTION_DIR))
    args = parser.parse_args()

    output_dir = Path(args.execution_dir).resolve()
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)
    target_metrics = load_csv(output_dir / "metrics" / f"{PREFIX}_extended_target_domain_metrics.csv")
    if target_metrics.empty:
        raise FileNotFoundError(output_dir / "metrics" / f"{PREFIX}_extended_target_domain_metrics.csv")

    seed_status = build_seed_completion_status(output_dir, target_metrics)
    seed_status.to_csv(output_dir / "analysis" / f"{PREFIX}_seed_completion_status.csv", index=False)
    seed_status[
        [
            "scenario_name",
            "source_domain",
            "target_domain",
            "model_name",
            "seed",
            "status",
            "result_path",
            "needs_execution",
            "reason",
        ]
    ].to_csv(output_dir / "analysis" / f"{PREFIX}_reuse_plan_for_30_seed_revision.csv", index=False)

    filter_summary, figure_paths, _ = regenerate_filtered_visualizations(output_dir)
    update_config_and_validation(output_dir, seed_status, filter_summary)
    write_revision_report(output_dir, seed_status, filter_summary, figure_paths)

    run_log = output_dir / "logs" / f"{PREFIX}_run.log"
    with run_log.open("a", encoding="utf-8") as f:
        f.write(
            "30-seed revision metadata and RMSE>0.5 visualization-only filtering generated. "
            f"Missing seed conditions={int(seed_status['needs_execution'].sum())}.\n"
        )
    print(output_dir)


if __name__ == "__main__":
    main()

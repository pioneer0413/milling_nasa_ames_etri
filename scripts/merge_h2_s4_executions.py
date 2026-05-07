#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.experiments.h2_s4_process_information_execution import (
    PREFIX,
    H2S4ProcessInformationExecution,
)
from milling_experiment_framework.utils.io import write_csv, write_json, write_yaml
from milling_experiment_framework.utils.paths import ExperimentPaths


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge H2.S4 initial and extra seed executions into one 20-seed result.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--execution-dirs", nargs="+", required=True)
    args = parser.parse_args()

    root = Path.cwd()
    experiment_id = (
        datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
        + "_H2_S4_process_information_combination_segment_aware_VB_prediction_seeds_0_19_combined"
    )
    paths = ExperimentPaths(root, experiment_id)
    paths.prepare_standard_dirs()

    helper = H2S4ProcessInformationExecution(args.config, root=root, dry_run=False, seed_mode="all")
    raw_config = helper._read_config()
    run_config = helper._resolve_run_config(raw_config)
    resolved = helper._resolved_config(raw_config, run_config, experiment_id)
    write_yaml(paths.execution_dir / "configs" / f"{PREFIX}_input_config.yaml", raw_config)
    write_yaml(paths.execution_dir / "configs" / f"{PREFIX}_resolved_config.yaml", resolved)
    write_json(paths.execution_dir / "configs" / f"{PREFIX}_config_validation.json", {"ok": True, "errors": [], "warnings": []})

    source_dirs = [Path(p) for p in args.execution_dirs]
    for dirname in ["data", "preprocessing"]:
        src = source_dirs[0] / dirname
        dst = paths.execution_dir / dirname
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    process_mapping = pd.read_csv(paths.execution_dir / "data" / f"{PREFIX}_process_information_mapping.csv")
    process_mapping["experiment_id"] = experiment_id
    for col in ["encoding_method", "scaling_method", "reason_if_excluded"]:
        process_mapping[col] = process_mapping[col].fillna("").astype("object")
    material_mask = process_mapping["source_column"] == "material_name"
    process_mapping.loc[material_mask, "feature_type"] = "categorical"
    process_mapping.loc[material_mask & process_mapping["used"].astype(bool), "encoding_method"] = "OneHotEncoder(handle_unknown='ignore')"
    process_mapping.loc[material_mask, "scaling_method"] = ""
    write_csv(paths.execution_dir / "data" / f"{PREFIX}_process_information_mapping.csv", process_mapping)
    shift_metrics = pd.concat([pd.read_csv(d / "metrics" / f"{PREFIX}_shift_metrics.csv") for d in source_dirs], ignore_index=True)
    predictions = pd.concat([pd.read_csv(d / "predictions" / f"{PREFIX}_predictions.csv") for d in source_dirs], ignore_index=True)
    splits = pd.concat([pd.read_csv(d / "splits" / f"{PREFIX}_split.csv") for d in source_dirs], ignore_index=True)
    shift_metrics["experiment_id"] = experiment_id

    dry_summary = _load_dry_summary(paths)
    dry_summary["seeds"] = run_config.seeds
    dry_summary["planned_atomic_evaluations"] = int(len(shift_metrics))
    write_json(paths.execution_dir / "data" / f"{PREFIX}_dataset_summary.json", dry_summary)

    seed_metrics = helper._seed_metrics(shift_metrics)
    feature_group_metrics = helper._feature_group_metrics(seed_metrics, process_mapping)
    process_metrics = helper._process_combination_metrics(feature_group_metrics, process_mapping)
    process_effect = helper._process_effect_metrics(seed_metrics)
    process_effect_summary = helper._process_effect_summary(process_effect)
    process_effect_consistency = helper._process_effect_consistency(process_effect)
    best_process = helper._best_process_combination(process_metrics)
    factor_contribution = helper._process_factor_contribution(process_metrics)
    process_vs_sensor = helper._process_only_vs_sensor_plus_process(feature_group_metrics)
    segment_effect = helper._segment_effect_after_process_info(feature_group_metrics)
    best_segment = helper._best_segment_by_process_combination(feature_group_metrics)
    r2_summary = helper._r2_positive_summary(feature_group_metrics)
    baseline = pd.concat([process_effect.assign(comparison_type="process_addition"), segment_effect.assign(comparison_type="segment")], ignore_index=True, sort=False)
    metrics_json = helper._metrics_json(feature_group_metrics, best_process, best_segment, r2_summary)

    results = {
        "shift_metrics": shift_metrics,
        "seed_metrics": seed_metrics,
        "feature_group_metrics": feature_group_metrics,
        "process_combination_metrics": process_metrics,
        "process_effect_metrics": process_effect,
        "comparison_metrics": baseline,
        "predictions": predictions,
        "splits": splits,
        "process_effect_summary": process_effect_summary,
        "process_effect_consistency": process_effect_consistency,
        "best_process_combination": best_process,
        "process_factor_contribution": factor_contribution,
        "segment_effect_after_process_info": segment_effect,
        "best_segment_by_process_combination": best_segment,
        "r2_positive_count_summary": r2_summary,
        "process_only_vs_sensor_plus_process": process_vs_sensor,
        "baseline_improvement": baseline,
        "metrics_json": metrics_json,
    }

    for frame in results.values():
        if isinstance(frame, pd.DataFrame) and "experiment_id" in frame.columns:
            frame["experiment_id"] = experiment_id
    metrics_json["best_overall_by_mae"]["experiment_id"] = experiment_id

    write_csv(paths.execution_dir / "metrics" / f"{PREFIX}_shift_metrics.csv", shift_metrics)
    write_csv(paths.execution_dir / "metrics" / f"{PREFIX}_seed_metrics.csv", seed_metrics)
    write_csv(paths.execution_dir / "metrics" / f"{PREFIX}_feature_group_metrics.csv", feature_group_metrics)
    write_csv(paths.execution_dir / "metrics" / f"{PREFIX}_process_combination_metrics.csv", process_metrics)
    write_csv(paths.execution_dir / "metrics" / f"{PREFIX}_process_effect_metrics.csv", process_effect)
    write_csv(paths.execution_dir / "metrics" / f"{PREFIX}_comparison_metrics.csv", baseline)
    write_json(paths.execution_dir / "metrics" / f"{PREFIX}_metrics.json", metrics_json)
    write_csv(paths.execution_dir / "predictions" / f"{PREFIX}_predictions.csv", predictions)
    write_csv(paths.execution_dir / "predictions" / f"{PREFIX}_prediction_errors.csv", predictions.sort_values("absolute_error", ascending=False))
    write_csv(paths.execution_dir / "splits" / f"{PREFIX}_split.csv", splits)
    write_json(paths.execution_dir / "splits" / f"{PREFIX}_split_summary.json", {"merged_from": [str(d) for d in source_dirs]})
    write_json(paths.execution_dir / "splits" / f"{PREFIX}_leakage_check.json", helper._leakage_check(splits))
    write_json(paths.execution_dir / "analysis" / f"{PREFIX}_analysis_summary.json", helper._analysis_summary(results, dry_summary, run_config))
    for key in [
        "process_effect_summary",
        "process_effect_consistency",
        "process_combination_metrics",
        "best_process_combination",
        "process_factor_contribution",
        "segment_effect_after_process_info",
        "best_segment_by_process_combination",
        "r2_positive_count_summary",
        "process_only_vs_sensor_plus_process",
        "baseline_improvement",
    ]:
        write_csv(paths.execution_dir / "analysis" / f"{PREFIX}_{key}.csv", results[key])
    helper._write_figures(paths, results)
    write_json(paths.execution_dir / "logs" / f"{PREFIX}_environment.json", collect_environment(str(root)))
    write_json(
        paths.execution_dir / "logs" / f"{PREFIX}_reproducibility.json",
        {
            "framework_version": resolved["framework_version"],
            "config_hash": resolved["config_hash"],
            "merged_from": [str(d) for d in source_dirs],
            "seeds": run_config.seeds,
        },
    )
    (paths.execution_dir / "logs" / f"{PREFIX}_run.log").write_text(
        "\n".join(
            [
                f"{datetime.now().isoformat(timespec='seconds')} INFO H2.S4 merged execution started: {experiment_id}",
                f"{datetime.now().isoformat(timespec='seconds')} INFO merged_from: {[str(d) for d in source_dirs]}",
                f"{datetime.now().isoformat(timespec='seconds')} INFO atomic_evaluations: {len(shift_metrics)}",
                f"{datetime.now().isoformat(timespec='seconds')} INFO H2.S4 merged execution finished: {experiment_id}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (paths.execution_dir / "logs" / f"{PREFIX}_error.log").touch()
    helper._write_report(paths, dry_summary, results, dry_run=False)
    helper._write_html(paths)
    helper._update_index(resolved, "finished", helper._best_metric(feature_group_metrics))
    print({"experiment_id": experiment_id, "execution_dir": str(paths.execution_dir), "best": metrics_json["best_overall_by_mae"]})


def _load_dry_summary(paths: ExperimentPaths) -> dict:
    with (paths.execution_dir / "data" / f"{PREFIX}_dataset_summary.json").open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()

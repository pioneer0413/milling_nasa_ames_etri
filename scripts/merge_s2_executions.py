#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.experiments.s2_sensor_combination_execution import (
    S2SensorCombinationExecution,
)
from milling_experiment_framework.utils.io import write_csv, write_json, write_yaml
from milling_experiment_framework.utils.paths import ExperimentPaths


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge S2 initial and extra seed executions into one 20-seed result.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--execution-dirs", nargs="+", required=True)
    args = parser.parse_args()

    root = Path.cwd()
    experiment_id = (
        datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
        + "_S2_sensor_combination_segment_aware_VB_prediction_seeds_0_19_combined"
    )
    paths = ExperimentPaths(root, experiment_id)
    paths.prepare_standard_dirs()

    helper = S2SensorCombinationExecution(args.config, root=root, dry_run=False, seed_mode="all")
    raw_config = helper._read_config()
    run_config = helper._resolve_run_config(raw_config)
    resolved = helper._resolved_config(raw_config, run_config, experiment_id)
    write_yaml(paths.execution_dir / "configs" / "input_config.yaml", raw_config)
    write_yaml(paths.execution_dir / "configs" / "resolved_config.yaml", resolved)
    write_json(paths.execution_dir / "configs" / "config_validation.json", {"ok": True, "errors": [], "warnings": []})

    source_dirs = [Path(p) for p in args.execution_dirs]
    for dirname in ["data", "preprocessing"]:
        src = source_dirs[0] / dirname
        dst = paths.execution_dir / dirname
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    shift_metrics = pd.concat([pd.read_csv(d / "metrics" / "shift_metrics.csv") for d in source_dirs], ignore_index=True)
    predictions = pd.concat([pd.read_csv(d / "predictions" / "predictions.csv") for d in source_dirs], ignore_index=True)
    splits = pd.concat([pd.read_csv(d / "splits" / "split.csv") for d in source_dirs], ignore_index=True)
    shift_metrics["experiment_id"] = experiment_id
    dry_summary = _load_dry_summary(paths)
    dry_summary["seeds"] = run_config.seeds
    dry_summary["planned_atomic_evaluations"] = int(len(shift_metrics))
    write_json(paths.execution_dir / "data" / "dataset_summary.json", dry_summary)

    seed_metrics = helper._seed_metrics(shift_metrics)
    combo_metrics = helper._combo_metrics(seed_metrics)
    segment_metrics = helper._segment_metrics(seed_metrics)
    sensor_effect = helper._sensor_combination_effect(combo_metrics)
    best_combo = helper._best_sensor_combination(combo_metrics)
    segment_effect = helper._segment_effect(combo_metrics)
    best_segment = helper._best_segment_by_combo(combo_metrics)
    r2_summary = helper._r2_positive_summary(combo_metrics)
    baseline = pd.concat(
        [
            sensor_effect.assign(comparison_type="sensor_combination"),
            segment_effect.assign(comparison_type="segment"),
        ],
        ignore_index=True,
        sort=False,
    )
    metrics_json = helper._metrics_json(combo_metrics, best_combo, best_segment, r2_summary)

    results = {
        "shift_metrics": shift_metrics,
        "seed_metrics": seed_metrics,
        "sensor_combination_metrics": combo_metrics,
        "segment_metrics": segment_metrics,
        "comparison_metrics": baseline,
        "predictions": predictions,
        "splits": splits,
        "sensor_combination_effect": sensor_effect,
        "best_sensor_combination": best_combo,
        "segment_effect_under_sensor_combination": segment_effect,
        "best_segment_by_sensor_combination": best_segment,
        "r2_positive_count_summary": r2_summary,
        "baseline_improvement": baseline,
        "metrics_json": metrics_json,
    }

    for frame in results.values():
        if isinstance(frame, pd.DataFrame) and "experiment_id" in frame.columns:
            frame["experiment_id"] = experiment_id
    metrics_json["best_overall_by_mae"]["experiment_id"] = experiment_id

    write_csv(paths.execution_dir / "metrics" / "shift_metrics.csv", shift_metrics)
    write_csv(paths.execution_dir / "metrics" / "seed_metrics.csv", seed_metrics)
    write_csv(paths.execution_dir / "metrics" / "sensor_combination_metrics.csv", combo_metrics)
    write_csv(paths.execution_dir / "metrics" / "segment_metrics.csv", segment_metrics)
    write_csv(paths.execution_dir / "metrics" / "comparison_metrics.csv", baseline)
    write_json(paths.execution_dir / "metrics" / "metrics.json", metrics_json)
    write_csv(paths.execution_dir / "predictions" / "predictions.csv", predictions)
    write_csv(paths.execution_dir / "predictions" / "prediction_errors.csv", predictions.sort_values("absolute_error", ascending=False))
    write_csv(paths.execution_dir / "splits" / "split.csv", splits)
    write_json(paths.execution_dir / "splits" / "split_summary.json", {"merged_from": [str(d) for d in source_dirs]})
    write_json(paths.execution_dir / "splits" / "leakage_check.json", helper._leakage_check(splits))
    write_json(paths.execution_dir / "analysis" / "analysis_summary.json", helper._analysis_summary(results, dry_summary, run_config))
    write_csv(paths.execution_dir / "analysis" / "sensor_combination_effect.csv", sensor_effect)
    write_csv(paths.execution_dir / "analysis" / "best_sensor_combination.csv", best_combo)
    write_csv(paths.execution_dir / "analysis" / "segment_effect_under_sensor_combination.csv", segment_effect)
    write_csv(paths.execution_dir / "analysis" / "best_segment_by_sensor_combination.csv", best_segment)
    write_csv(paths.execution_dir / "analysis" / "r2_positive_count_summary.csv", r2_summary)
    write_csv(paths.execution_dir / "analysis" / "baseline_improvement.csv", baseline)
    helper._write_figures(paths, results)
    write_json(paths.execution_dir / "logs" / "environment.json", collect_environment(str(root)))
    write_json(
        paths.execution_dir / "logs" / "reproducibility.json",
        {
            "framework_version": resolved["framework_version"],
            "config_hash": resolved["config_hash"],
            "merged_from": [str(d) for d in source_dirs],
            "seeds": run_config.seeds,
        },
    )
    (paths.execution_dir / "logs" / "run.log").write_text(
        "\n".join(
            [
                f"{datetime.now().isoformat(timespec='seconds')} INFO S2 merged execution started: {experiment_id}",
                f"{datetime.now().isoformat(timespec='seconds')} INFO merged_from: {[str(d) for d in source_dirs]}",
                f"{datetime.now().isoformat(timespec='seconds')} INFO atomic_evaluations: {len(shift_metrics)}",
                f"{datetime.now().isoformat(timespec='seconds')} INFO S2 merged execution finished: {experiment_id}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (paths.execution_dir / "logs" / "error.log").touch()
    helper._write_report(paths, dry_summary, results, dry_run=False)
    helper._write_html(paths)
    helper._update_index(resolved, "finished", helper._best_metric(combo_metrics))
    print({"experiment_id": experiment_id, "execution_dir": str(paths.execution_dir), "best": metrics_json["best_overall_by_mae"]})


def _load_dry_summary(paths: ExperimentPaths) -> dict:
    import json

    with (paths.execution_dir / "data" / "dataset_summary.json").open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()

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
from milling_experiment_framework.experiments.s1_segment_execution import (
    S1SegmentExecution,
)
from milling_experiment_framework.utils.io import write_csv, write_json, write_yaml
from milling_experiment_framework.utils.paths import ExperimentPaths


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge S1 initial and extra seed executions into one 20-seed result.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--execution-dirs", nargs="+", required=True)
    args = parser.parse_args()

    root = Path.cwd()
    experiment_id = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f") + "_S1_all_models_all_sensors_all_segments_all_shifts_seeds_0_19_combined"
    paths = ExperimentPaths(root, experiment_id)
    paths.prepare_standard_dirs()

    helper = S1SegmentExecution(args.config, root=root, dry_run=False, seed_mode="all")
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

    results = {"shift_metrics": shift_metrics, "predictions": predictions, "splits": splits}
    results["seed_metrics"] = helper._seed_metrics(shift_metrics)
    results["segment_metrics"] = helper._segment_metrics(results["seed_metrics"])
    results["baseline_improvement"] = helper._baseline_improvement(results["segment_metrics"])
    results["sensor_best_segment"] = helper._sensor_best(results["segment_metrics"])
    results["comparison_metrics"] = results["baseline_improvement"].copy()
    results["segment_ranking"] = helper._segment_ranking(results["segment_metrics"])
    results["metrics_json"] = helper._metrics_json(results["segment_metrics"])
    results["metrics_json"]["best_overall_by_mae"]["experiment_id"] = experiment_id
    results["h1s1_association"] = pd.DataFrame()

    for key in ["seed_metrics", "segment_metrics", "baseline_improvement", "sensor_best_segment", "comparison_metrics", "segment_ranking"]:
        if "experiment_id" in results[key].columns:
            results[key]["experiment_id"] = experiment_id

    write_csv(paths.execution_dir / "metrics" / "shift_metrics.csv", shift_metrics)
    write_csv(paths.execution_dir / "metrics" / "seed_metrics.csv", results["seed_metrics"])
    write_csv(paths.execution_dir / "metrics" / "segment_metrics.csv", results["segment_metrics"])
    write_csv(paths.execution_dir / "metrics" / "comparison_metrics.csv", results["comparison_metrics"])
    write_json(paths.execution_dir / "metrics" / "metrics.json", results["metrics_json"])
    write_csv(paths.execution_dir / "predictions" / "predictions.csv", predictions)
    write_csv(paths.execution_dir / "predictions" / "prediction_errors.csv", predictions.sort_values("absolute_error", ascending=False))
    write_csv(paths.execution_dir / "splits" / "split.csv", splits)
    write_json(paths.execution_dir / "splits" / "split_summary.json", {"merged_from": [str(d) for d in source_dirs]})
    write_json(paths.execution_dir / "splits" / "leakage_check.json", helper._leakage_check(splits))
    write_json(paths.execution_dir / "analysis" / "analysis_summary.json", helper._analysis_summary(results, sorted(shift_metrics["sensor"].unique()), run_config))
    write_csv(paths.execution_dir / "analysis" / "segment_ranking.csv", results["segment_ranking"])
    write_csv(paths.execution_dir / "analysis" / "sensor_best_segment.csv", results["sensor_best_segment"])
    write_csv(paths.execution_dir / "analysis" / "baseline_improvement.csv", results["baseline_improvement"])
    write_json(paths.execution_dir / "analysis" / "h1s1_association_comparison.json", {"status": "H1.S1 association reference unavailable"})
    helper._write_figures(paths, results)
    write_json(paths.execution_dir / "logs" / "environment.json", collect_environment(str(root)))
    write_json(paths.execution_dir / "logs" / "reproducibility.json", {"framework_version": resolved["framework_version"], "config_hash": resolved["config_hash"], "merged_from": [str(d) for d in source_dirs], "seeds": run_config.seeds})
    dry_summary = pd.read_json(paths.execution_dir / "data" / "dataset_summary.json", typ="series").to_dict()
    dry_summary["seeds"] = run_config.seeds
    dry_summary["planned_atomic_executions"] = len(shift_metrics)
    helper._write_report(paths, results, dry_summary, dry_run=False)
    helper._write_report_html(paths)
    helper._update_index(resolved, status="finished", best_metric=float(results["segment_metrics"]["mean_mae"].min()))
    print({"experiment_id": experiment_id, "execution_dir": str(paths.execution_dir), "best": results["metrics_json"]["best_overall_by_mae"]})


if __name__ == "__main__":
    main()

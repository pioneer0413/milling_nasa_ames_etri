#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.experiments.estimator import estimate_experiment_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate experiment execution count, runtime, and resources.")
    parser.add_argument("--config", required=True, help="YAML/JSON experiment config path.")
    parser.add_argument("--output", default=None, help="Output directory. Defaults to experiments/estimations/{estimation_id}.")
    parser.add_argument("--runtime-per-atomic-sec", type=float, default=None, help="Override runtime per atomic execution for all models.")
    parser.add_argument("--compare-config", default=None, help="Optional second config to compare against.")
    parser.add_argument("--profile", action="store_true", help="Enable profiling mode hook. Current implementation records profiling as requested and uses configured/heuristic estimate.")
    parser.add_argument("--profile-n-batches", type=int, default=5)
    args = parser.parse_args()

    overrides = {"__default__": args.runtime_per_atomic_sec} if args.runtime_per_atomic_sec is not None else None
    result = estimate_experiment_from_config(args.config, output_dir=args.output, runtime_overrides=overrides, profile=args.profile)
    payload = {
        "estimation_id": result["estimation_id"],
        "experiment_name": result["experiment_name"],
        "total_atomic_executions": result["total_atomic_executions"],
        "total_sub_runs": result["total_sub_runs"],
        "estimated_wall_clock_hours": result["estimated_wall_clock_hours"],
        "warning_level": result["warning_level"],
        "output_dir": result["output_dir"],
        "report_path": result["report_path"],
    }
    if args.compare_config:
        compare_result = estimate_experiment_from_config(args.compare_config, output_dir=None, runtime_overrides=overrides, profile=args.profile)
        ratio = (
            compare_result["estimated_wall_clock_hours"] / result["estimated_wall_clock_hours"]
            if result["estimated_wall_clock_hours"] > 0
            else None
        )
        payload["comparison"] = {
            "base_config": args.config,
            "compare_config": args.compare_config,
            "base_atomic_executions": result["total_atomic_executions"],
            "compare_atomic_executions": compare_result["total_atomic_executions"],
            "atomic_execution_difference": compare_result["total_atomic_executions"] - result["total_atomic_executions"],
            "runtime_ratio_compare_over_base": ratio,
            "compare_output_dir": compare_result["output_dir"],
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


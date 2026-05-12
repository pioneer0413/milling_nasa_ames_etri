from __future__ import annotations

import csv
import itertools
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from milling_experiment_framework.experiments.estimator.cost_model import estimate_disk_mb, runtime_for_model
from milling_experiment_framework.experiments.estimator.report_writer import write_estimation_outputs
from milling_experiment_framework.experiments.estimator.schema import AXIS_NAMES, GridAxes
from milling_experiment_framework.utils.io import read_yaml_or_json


def estimate_experiment_from_config(
    config_path: str,
    output_dir: str | None = None,
    runtime_overrides: dict | None = None,
    use_historical_runtime: bool = True,
    profile: bool = False,
) -> dict:
    config = read_yaml_or_json(config_path)
    estimation_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f_EXPERIMENT_ESTIMATOR")
    axes = parse_grid_axes(config)
    phase_axes = build_phase_axes(config, axes)
    count_rows = build_count_rows(config, phase_axes)
    runtime_rows = build_runtime_rows(config, count_rows, runtime_overrides, use_historical_runtime, profile)
    resource = build_resource_estimate(estimation_id, config, count_rows, runtime_rows)
    warnings = build_warnings(config, resource, count_rows)
    resource["warning_level"] = max_warning_level(warnings)
    reduced = build_reduced_grid_suggestion(config, axes, runtime_rows, warnings)
    result = {
        "estimation_id": estimation_id,
        "experiment_name": experiment_name(config),
        "input_config": config,
        "resolved_grid": {"axes": {name: getattr(axes, name) for name in AXIS_NAMES}, "phases": {k: {name: getattr(v, name) for name in AXIS_NAMES} for k, v in phase_axes.items()}},
        "atomic_count_summary": count_rows,
        "runtime_estimate": runtime_rows,
        "resource_estimate": resource,
        "warnings": warnings,
        "reduced_grid_suggestion": reduced,
    }
    output_base = Path(output_dir) if output_dir else Path("experiments") / "estimations" / estimation_id
    paths = write_estimation_outputs(result, output_base)
    result.update(
        {
            "total_atomic_executions": resource["total_atomic_executions"],
            "total_sub_runs": resource["total_sub_runs"],
            "estimated_wall_clock_hours": resource["estimated_wall_clock_hours"],
            "warning_level": resource["warning_level"],
            "atomic_count_summary_path": paths["atomic_count_summary"],
            "runtime_estimate_path": paths["runtime_estimate"],
            "report_path": paths["report"],
            "output_dir": str(output_base),
        }
    )
    return result


def experiment_name(config: dict[str, Any]) -> str:
    return str(config.get("experiment", {}).get("name") or config.get("model", {}).get("name") or "experiment")


def listify(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return value if value else list(default)
    return [value]


def parse_grid_axes(config: dict[str, Any]) -> GridAxes:
    runtime = config.get("runtime", {})
    experiment = config.get("experiment", {})
    model_cfg = config.get("model", {})
    seeds_cfg = config.get("seeds", {})
    preprocessing = config.get("preprocessing", {})
    axes = GridAxes()
    axes.models = listify(config.get("models"), listify(model_cfg.get("name"), ["default_model"]))
    axes.seeds = listify(runtime.get("seeds"), listify(seeds_cfg.get("seed_list"), listify(experiment.get("seeds"), listify(config.get("seed"), [0]))))
    axes.shifts = normalize_shifts(runtime.get("shifts") or experiment.get("shifts") or config.get("shifts") or config.get("domain_shift_scenarios"))
    axes.sensor_combinations = listify(
        runtime.get("sensor_combinations") or config.get("sensor_combinations") or config.get("sensor_combination") or preprocessing.get("sensor_combinations") or model_cfg.get("sensor_setting") or preprocessing.get("sensor_setting"),
        ["default_sensor"],
    )
    axes.segment_settings = listify(
        runtime.get("segment_settings") or runtime.get("segment_modes") or config.get("segment_settings") or preprocessing.get("segment_settings") or model_cfg.get("sequence", {}).get("segment_mode"),
        ["default_segment"],
    )
    axes.feature_combinations = listify(runtime.get("feature_combinations") or config.get("feature_combinations") or config.get("feature_combination"), ["default_feature"])
    axes.process_combinations = listify(runtime.get("process_combinations") or config.get("process_combinations") or config.get("process_combination") or model_cfg.get("process_combination"), ["none"])
    axes.input_representations = listify(runtime.get("input_representations") or config.get("input_representations") or model_cfg.get("input_representation"), ["default_input"])
    axes.modes = listify(runtime.get("modes") or config.get("modes") or model_cfg.get("mode"), ["default_mode"])
    axes.base_learners = enabled_base_learners(model_cfg)
    axes.sequence_modes = listify(runtime.get("sequence_modes") or model_cfg.get("sequence", {}).get("construction_mode"), ["default_sequence"])
    axes.hyperparameter_grid = hyperparameter_grid_values(config.get("hyperparameter_grid") or config.get("hparam_grid") or runtime.get("hyperparameter_grid"))
    return axes


def normalize_shifts(value: Any) -> list[Any]:
    if value is None:
        return ["default_shift"]
    if isinstance(value, list):
        if not value:
            return ["default_shift"]
        return ["_to_".join(item) if isinstance(item, list) and len(item) == 2 else item for item in value]
    return [value]


def enabled_base_learners(model_cfg: dict[str, Any]) -> list[str]:
    learners = model_cfg.get("base_learners", {})
    if not learners:
        return []
    return [name for name, cfg in learners.items() if not isinstance(cfg, dict) or cfg.get("enabled", True)]


def hyperparameter_grid_values(grid: Any) -> list[Any]:
    if not grid:
        return ["default_hparam"]
    if isinstance(grid, list):
        return grid
    if isinstance(grid, dict):
        keys = list(grid)
        values = [listify(grid[k], [grid[k]]) for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)] or ["default_hparam"]
    return [grid]


def build_phase_axes(config: dict[str, Any], axes: GridAxes) -> dict[str, GridAxes]:
    seeds_cfg = config.get("seeds", {})
    initial_seeds = listify(seeds_cfg.get("initial_seeds"), axes.seeds[:3] or [0])
    extra_seeds = listify(seeds_cfg.get("extra_seeds"), axes.seeds[3:] or [])
    phases = {
        "full": axes,
        "initial": axes.copy_with(seeds=initial_seeds),
        "extra_seed": axes.copy_with(seeds=extra_seeds or ["none"]),
        "smoke": axes.copy_with(
            seeds=axes.seeds[:1],
            shifts=axes.shifts[:1],
            sensor_combinations=axes.sensor_combinations[:1],
            segment_settings=axes.segment_settings[:1],
            feature_combinations=axes.feature_combinations[:1],
            process_combinations=axes.process_combinations[:1],
            input_representations=axes.input_representations[:1],
            modes=axes.modes[:1],
        ),
    }
    reduced = suggest_reduced_axes(config, axes)
    phases["reduced"] = reduced
    optional = axes.copy_with(input_representations=axes.input_representations[:2], modes=axes.modes[:2])
    phases["optional_ablation"] = optional
    return phases


def suggest_reduced_axes(config: dict[str, Any], axes: GridAxes) -> GridAxes:
    def keep_preferred(values: list[Any], preferred: list[Any], max_n: int) -> list[Any]:
        kept = [v for v in preferred if v in values]
        for v in values:
            if v not in kept and len(kept) < max_n:
                kept.append(v)
        return kept or values[:max_n] or ["default"]

    initial = config.get("estimation", {}).get("reduced_grid_policy", {}).get("initial_seeds") or config.get("seeds", {}).get("initial_seeds") or axes.seeds[:3]
    return axes.copy_with(
        seeds=listify(initial, axes.seeds[:3] or [0]),
        sensor_combinations=keep_preferred(axes.sensor_combinations, ["current", "acoustic", "current_acoustic", "current_vibration_acoustic"], 4),
        segment_settings=keep_preferred(axes.segment_settings, ["full_length", "steady", "entry_exit"], 3),
        process_combinations=keep_preferred(axes.process_combinations, ["none", "doc", "doc_feed", "doc_feed_material_time"], 4),
        feature_combinations=axes.feature_combinations[:4],
        input_representations=keep_preferred(axes.input_representations, ["cumulative", "raw"], 2),
        modes=keep_preferred(axes.modes, ["base_only", "s_dnn"], 2),
    )


def build_count_rows(config: dict[str, Any], phase_axes: dict[str, GridAxes]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_groups = config.get("feature_groups")
    conditional = config.get("estimation", {}).get("conditional_grids")
    for phase, axes in phase_axes.items():
        if conditional:
            for group in conditional:
                group_axes = axes_for_condition_group(axes, group)
                rows.extend(rows_for_axes(config, phase, group.get("name", "conditional"), group_axes, group.get("notes", "")))
        elif feature_groups and set(feature_groups) & {"sensor_only", "process_only", "sensor_plus_process"}:
            specs = [
                {"name": "sensor_only", "segment": True, "process": False, "notes": "segment applies; process_combination ignored"},
                {"name": "process_only", "segment": False, "process": True, "notes": "segment-independent; process_combination applies"},
                {"name": "sensor_plus_process", "segment": True, "process": True, "notes": "segment and process_combination both apply"},
            ]
            for spec in specs:
                if spec["name"] in feature_groups:
                    group_axes = axes.copy_with(
                        segment_settings=axes.segment_settings if spec["segment"] else ["segment_independent"],
                        process_combinations=axes.process_combinations if spec["process"] else ["none"],
                        modes=[spec["name"]],
                    )
                    rows.extend(rows_for_axes(config, phase, spec["name"], group_axes, spec["notes"]))
        else:
            rows.extend(rows_for_axes(config, phase, "default", axes, ""))
    return rows


def axes_for_condition_group(axes: GridAxes, group: dict[str, Any]) -> GridAxes:
    updates = {}
    applies = group.get("applies", {})
    for axis in AXIS_NAMES:
        if axis in group:
            updates[axis] = listify(group[axis], [getattr(axes, axis)[0]])
        elif applies.get(axis) is False:
            updates[axis] = [group.get(f"{axis}_placeholder", "not_applicable")]
    return axes.copy_with(**updates)


def rows_for_axes(config: dict[str, Any], phase: str, condition_group: str, axes: GridAxes, notes: str) -> list[dict[str, Any]]:
    rows = []
    for model in axes.models:
        model_name = str(model)
        for mode in axes.modes:
            for input_representation in axes.input_representations:
                counts = axes.count_dict()
                # base_learners are not independent atomic axes; they are sub-runs for stacking.
                counts["base_learners"] = max(1, len(axes.base_learners))
                atomic = (
                    axes.count("seeds")
                    * axes.count("shifts")
                    * axes.count("sensor_combinations")
                    * axes.count("segment_settings")
                    * axes.count("feature_combinations")
                    * axes.count("process_combinations")
                    * axes.count("sequence_modes")
                    * axes.count("hyperparameter_grid")
                )
                sub_runs = sub_runs_per_atomic(config, model_name, str(mode), axes)
                rows.append(
                    {
                        "phase": phase,
                        "condition_group": condition_group,
                        "model": model_name,
                        "mode": str(mode),
                        "input_representation": str(input_representation),
                        "n_shifts": axes.count("shifts"),
                        "n_seeds": axes.count("seeds"),
                        "n_models": axes.count("models"),
                        "n_sensor_combinations": axes.count("sensor_combinations"),
                        "n_segment_settings": axes.count("segment_settings"),
                        "n_feature_combinations": axes.count("feature_combinations"),
                        "n_process_combinations": axes.count("process_combinations"),
                        "n_base_learners": len(axes.base_learners),
                        "n_folds": folds(config),
                        "atomic_executions": int(atomic),
                        "sub_runs_per_atomic": int(sub_runs),
                        "total_sub_runs": int(atomic * sub_runs),
                        "notes": notes,
                    }
                )
    return rows


def folds(config: dict[str, Any]) -> int:
    return int(config.get("model", {}).get("ensemble", {}).get("cv_folds") or config.get("ensemble", {}).get("cv_folds") or 1)


def sub_runs_per_atomic(config: dict[str, Any], model: str, mode: str, axes: GridAxes) -> int:
    is_stacking = model == "cumulative_s_dnn" or "s_dnn" in str(mode)
    if is_stacking:
        n_base = len(axes.base_learners) or 4
        return int(n_base * folds(config) + 1)
    return 1


def build_runtime_rows(config: dict[str, Any], count_rows: list[dict[str, Any]], runtime_overrides: dict | None, use_historical_runtime: bool, profile: bool) -> list[dict[str, Any]]:
    rows = []
    for row in count_rows:
        sec, method, assumption = runtime_for_model(row["model"], config, runtime_overrides)
        if use_historical_runtime and method == "heuristic_default":
            historical = historical_runtime_for_model(row["model"])
            if historical is not None:
                sec, source = historical
                method = "historical_runtime_lookup"
                assumption = source
        if profile:
            method = "profile_requested_not_executed"
            assumption += "; profiling hook enabled, estimator used configured/heuristic sec"
        total_sec = sec * row["atomic_executions"]
        rows.append(
            {
                "phase": row["phase"],
                "condition_group": row["condition_group"],
                "model": row["model"],
                "atomic_executions": row["atomic_executions"],
                "total_sub_runs": row["total_sub_runs"],
                "estimated_sec_per_atomic": sec,
                "estimated_total_sec": total_sec,
                "estimated_total_min": total_sec / 60.0,
                "estimated_total_hours": total_sec / 3600.0,
                "runtime_estimation_method": method,
                "assumption": assumption,
            }
        )
    return rows


def historical_runtime_for_model(model: str, root: str | Path = "experiments") -> tuple[float, str] | None:
    """Best-effort lookup for prior runtime summaries.

    Supported lightweight schemas:
    - runtime_summary.csv with columns model + runtime_sec_per_atomic
    - runtime_summary.csv with columns model + elapsed_sec + atomic_executions
    - experiments/index.csv with columns model + duration_sec + atomic_executions
    """
    root = Path(root)
    candidates = list(root.glob("executions/**/runtime_summary.csv"))
    candidates += list(root.glob("executions/**/metrics/*runtime_summary.csv"))
    candidates += list(root.glob("executions/**/analysis/*runtime_summary.csv"))
    index_path = root / "index.csv"
    if index_path.exists():
        candidates.append(index_path)

    values: list[float] = []
    used_paths: list[str] = []
    model_key = str(model).lower()
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for item in reader:
                    row_model = str(item.get("model") or item.get("model_name") or "").lower()
                    if row_model and row_model != model_key:
                        continue
                    direct = first_float(item, ["runtime_sec_per_atomic", "sec_per_atomic", "estimated_sec_per_atomic"])
                    if direct is not None and direct > 0:
                        values.append(direct)
                        used_paths.append(str(path))
                        continue
                    elapsed = first_float(item, ["elapsed_sec", "duration_sec", "runtime_sec", "total_sec"])
                    atomic = first_float(item, ["atomic_executions", "num_atomic", "n_atomic"])
                    if elapsed is not None and atomic and atomic > 0:
                        values.append(elapsed / atomic)
                        used_paths.append(str(path))
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
    if not values:
        return None
    return float(sum(values) / len(values)), f"mean historical runtime from {len(set(used_paths))} file(s)"


def first_float(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def selected_phase(config: dict[str, Any]) -> str:
    return str(config.get("estimation", {}).get("phase") or "full")


def build_resource_estimate(estimation_id: str, config: dict[str, Any], count_rows: list[dict[str, Any]], runtime_rows: list[dict[str, Any]]) -> dict[str, Any]:
    phase = selected_phase(config)
    phase_counts = [r for r in count_rows if r["phase"] == phase] or count_rows
    phase_runtime = [r for r in runtime_rows if r["phase"] == phase] or runtime_rows
    total_atomic = int(sum(r["atomic_executions"] for r in phase_counts))
    total_sub_runs = int(sum(r["total_sub_runs"] for r in phase_counts))
    hours = float(sum(r["estimated_total_hours"] for r in phase_runtime))
    checkpoint_cfg = config.get("estimation", {}).get("checkpoint_size_mb", {})
    checkpoint_size = max([float(v) for v in checkpoint_cfg.values()], default=5.0) if isinstance(checkpoint_cfg, dict) else float(checkpoint_cfg or 5.0)
    checkpoint_count = total_sub_runs * (2 if config.get("checkpoint", {}).get("save_best", True) else 1)
    prediction_rows = total_atomic * int(config.get("estimation", {}).get("expected_prediction_rows_per_atomic", 50))
    figure_count = int(min(200, max(4, total_atomic // 50)))
    disk_mb = estimate_disk_mb(checkpoint_count, checkpoint_size, prediction_rows, figure_count)
    uses_gpu = any(str(r["model"]).lower() in {"mlp", "cnn1d", "lstm", "hybrid_lstm_process", "cumulative_s_dnn"} for r in phase_counts)
    return {
        "estimation_id": estimation_id,
        "experiment_name": experiment_name(config),
        "total_atomic_executions": total_atomic,
        "total_sub_runs": total_sub_runs,
        "estimated_wall_clock_hours": hours,
        "estimated_gpu_hours": hours if uses_gpu else 0.0,
        "estimated_cpu_hours": hours if not uses_gpu else hours * 0.25,
        "estimated_memory_requirement_mb": heuristic_memory_mb(config),
        "estimated_disk_usage_mb": disk_mb,
        "estimated_checkpoint_count": int(checkpoint_count),
        "estimated_prediction_rows": int(prediction_rows),
        "estimated_metric_rows": int(total_atomic),
        "estimated_figure_count": int(figure_count),
        "warning_level": "INFO",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def heuristic_memory_mb(config: dict[str, Any]) -> float:
    sequence = config.get("model", {}).get("sequence", {})
    length = int(sequence.get("window_size") or sequence.get("resample_length") or 1024)
    channels = int(config.get("estimation", {}).get("num_channels", 6))
    batch = int(config.get("training", {}).get("batch_size") or 32)
    return float(max(512, batch * length * channels * 4 / (1024**2) * 12 + 512))


def build_warnings(config: dict[str, Any], resource: dict[str, Any], count_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    thresholds = {
        "atomic_executions_warning": 1000,
        "atomic_executions_critical": 10000,
        "runtime_hours_warning": 12,
        "runtime_hours_critical": 24,
        "disk_usage_mb_warning": 10000,
    }
    thresholds.update(config.get("estimation", {}).get("warning_thresholds", {}))
    warnings = []
    add_threshold_warning(warnings, resource["total_atomic_executions"], thresholds["atomic_executions_warning"], thresholds["atomic_executions_critical"], "ATOMIC_EXECUTIONS", "atomic execution count is large", "Run smoke/reduced/initial phases first.")
    add_threshold_warning(warnings, resource["estimated_wall_clock_hours"], thresholds["runtime_hours_warning"], thresholds["runtime_hours_critical"], "RUNTIME_HOURS", "estimated runtime is high", "Use reduced grid or fewer seeds.")
    if resource["estimated_disk_usage_mb"] > thresholds["disk_usage_mb_warning"]:
        warnings.append({"warning_level": "WARNING", "warning_code": "DISK_USAGE", "message": f"estimated disk usage {resource['estimated_disk_usage_mb']:.1f} MB exceeds threshold", "affected_phase": selected_phase(config), "suggested_action": "Reduce checkpoints or max_keep."})
    if any(r["sub_runs_per_atomic"] > 1 for r in count_rows):
        warnings.append({"warning_level": "HIGH", "warning_code": "STACKING_SUBRUNS", "message": "stacking/OOF model has many sub-runs per atomic execution", "affected_phase": "all", "suggested_action": "Profile one condition and run reduced grid first."})
    if "process_only" in str(config.get("feature_groups", [])):
        warnings.append({"warning_level": "INFO", "warning_code": "CONDITIONAL_GRID", "message": "conditional feature_group grid detected; process_only is treated as segment-independent", "affected_phase": "all", "suggested_action": "Check conditional grid rows in atomic_count_summary.csv."})
    return warnings


def add_threshold_warning(warnings: list[dict[str, Any]], value: float, warning_threshold: float, critical_threshold: float, code: str, message: str, action: str) -> None:
    if value >= critical_threshold:
        level = "CRITICAL"
    elif value >= warning_threshold:
        level = "HIGH"
    else:
        return
    warnings.append({"warning_level": level, "warning_code": code, "message": f"{message}: {value}", "affected_phase": "selected", "suggested_action": action})


def max_warning_level(warnings: list[dict[str, Any]]) -> str:
    order = {"INFO": 0, "WARNING": 1, "HIGH": 2, "CRITICAL": 3}
    if not warnings:
        return "INFO"
    return max((w["warning_level"] for w in warnings), key=lambda level: order.get(level, 0))


def build_reduced_grid_suggestion(config: dict[str, Any], axes: GridAxes, runtime_rows: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> dict[str, Any]:
    reduced_axes = suggest_reduced_axes(config, axes)
    temp_config = dict(config)
    reduced_rows = rows_for_axes(temp_config, "reduced", "suggested", reduced_axes, "suggested reduced grid")
    reduced_runtime = build_runtime_rows(temp_config, reduced_rows, None, False, False)
    return {
        "reason": "large grid or cautious initial execution" if warnings else "standard staged execution",
        "recommended_phase": "reduced",
        "recommended_grid": {name: getattr(reduced_axes, name) for name in AXIS_NAMES},
        "estimated_atomic_executions": int(sum(r["atomic_executions"] for r in reduced_rows)),
        "estimated_runtime_hours": float(sum(r["estimated_total_hours"] for r in reduced_runtime)),
        "excluded_axes": {"note": "axes truncated to preferred first candidates where applicable"},
        "notes": "Run reduced grid, inspect failures/runtime, then expand to initial/full.",
    }

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from milling_experiment_framework import __version__
from milling_experiment_framework.utils.io import read_yaml_or_json


SCHEMA_VERSION = "0.1.0"

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {"experiment_id": None, "name": "experiment", "seed": 42},
    "dataset": {"name": "example_milling", "dataset_version": "v1"},
    "task": {"type": "classification", "target_column": "label", "num_classes": 3, "positive_label": None},
    "preprocessing": {"output_type": "timeseries", "cache": {"enabled": False}, "steps": []},
    "split": {"strategy": "random", "validation_ratio": 0.2, "test_ratio": 0.2, "leakage_check": True},
    "model": {"name": "cnn1d", "model_type": "DL", "input_type": "timeseries-based", "params": {}},
    "training": {"epochs": 3, "batch_size": 16, "optimizer": "adam", "learning_rate": 0.001, "seed": 42},
    "checkpoint": {
        "enabled": True,
        "monitor": "val_loss",
        "mode": "min",
        "save_best": True,
        "save_last": True,
        "save_interval": None,
        "max_keep": 3,
        "resume_from": None,
    },
    "evaluation": {"metrics": ["accuracy", "f1_macro"], "group_metrics": []},
    "logging": {"save_environment": True, "save_git_state": True, "shape_trace": True},
    "report": {"enabled": True, "formats": ["md"]},
}

COMPATIBILITY = {
    "features": "feature-based",
    "timeseries": "timeseries-based",
    "hybrid": "hybrid",
}

REQUIRED_TOP_LEVEL = ["experiment", "dataset", "task", "preprocessing", "split", "model", "training", "evaluation"]
ALLOWED_TASKS = {"classification", "regression", "anomaly_detection", "rul_prediction", "representation_learning"}
ALLOWED_STEADY_CUT_MODES = {"full_signal", "sliding_window", "steady_cut_only", "air_cut_removal", "segmentation"}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def stable_hash(data: dict[str, Any]) -> str:
    payload = copy.deepcopy(data)
    payload.pop("config_hash", None)
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def generate_experiment_id(config: dict[str, Any]) -> str:
    now = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    name = str(config["experiment"].get("name") or "experiment").replace(" ", "_")
    model_name = config["model"].get("name", "model")
    steady_mode = get_steady_cut_mode(config)
    return f"{now}_{name}_{model_name}_{steady_mode}"


def get_steady_cut_mode(config: dict[str, Any]) -> str:
    for step in config.get("preprocessing", {}).get("steps", []):
        if step.get("name") == "steady_cut":
            return step.get("mode", "full_signal")
    return "full_signal"


def load_and_resolve_config(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    input_config = read_yaml_or_json(config_path)
    resolved = deep_merge(DEFAULT_CONFIG, input_config)
    seed = int(resolved["experiment"].get("seed") or resolved["training"].get("seed") or 42)
    resolved["training"]["seed"] = int(resolved["training"].get("seed") or seed)
    resolved["experiment"]["seed"] = seed
    resolved["experiment"]["created_at"] = datetime.now().isoformat(timespec="seconds")
    if not resolved["experiment"].get("experiment_id"):
        resolved["experiment"]["experiment_id"] = generate_experiment_id(resolved)
    resolved["framework_version"] = __version__
    resolved["config_schema_version"] = SCHEMA_VERSION
    resolved["dataset_schema_version"] = SCHEMA_VERSION
    resolved["output_schema_version"] = SCHEMA_VERSION
    resolved["steady_cut_mode"] = get_steady_cut_mode(resolved)
    resolved["config_hash"] = stable_hash(resolved)
    validation = validate_config(resolved)
    return input_config, resolved, validation


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    for key in REQUIRED_TOP_LEVEL:
        if key not in config:
            errors.append(f"Missing top-level config key: {key}")

    output_type = config.get("preprocessing", {}).get("output_type")
    input_type = config.get("model", {}).get("input_type")
    expected_input = COMPATIBILITY.get(output_type)
    if expected_input and expected_input != input_type:
        errors.append(f"model.input_type={input_type} is incompatible with preprocessing.output_type={output_type}")

    task_type = config.get("task", {}).get("type")
    if task_type not in ALLOWED_TASKS:
        errors.append(f"Unsupported task.type={task_type}")

    if task_type == "classification" and not config.get("task", {}).get("num_classes"):
        errors.append("classification task requires task.num_classes")

    strategy = config.get("split", {}).get("strategy")
    if strategy in {"dataset_run_wise", "condition_wise", "tool_wise", "machine_wise"} and not config.get("split", {}).get(
        "group_key"
    ):
        default_group = {
            "dataset_run_wise": "dataset_run_id",
            "condition_wise": "condition_id",
            "tool_wise": "tool_id",
            "machine_wise": "machine_id",
        }[strategy]
        config["split"]["group_key"] = default_group

    for step in config.get("preprocessing", {}).get("steps", []):
        if step.get("name") == "steady_cut":
            mode = step.get("mode", "full_signal")
            if mode not in ALLOWED_STEADY_CUT_MODES:
                errors.append(f"Unsupported steady_cut.mode={mode}")
            if mode not in {"full_signal", "sliding_window"}:
                warnings.append(f"steady_cut.mode={mode} is recorded but MVP processing implements full_signal/sliding_window only.")
            if mode in {"steady_cut_only", "air_cut_removal"} and not step.get("reference_signal"):
                errors.append(f"steady_cut.mode={mode} requires reference_signal")

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def validate_metadata_columns(config: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    errors: list[str] = []
    required = ["sample_id", "label", "dataset_run_id"]
    if "sequence_index" not in columns and "timestamp" not in columns:
        errors.append("metadata requires sequence_index or timestamp")
    for col in required:
        if col not in columns:
            errors.append(f"metadata missing required column: {col}")
    group_key = config.get("split", {}).get("group_key")
    if group_key and group_key not in columns:
        errors.append(f"split.strategy requires metadata column: {group_key}")
    return {"ok": not errors, "errors": errors}

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.experiments.h2_execution_utils import (
    effective_seeds_for_model,
    model_seed_value,
    seed_label,
)
from milling_experiment_framework.models.dl.hybrid_lstm_process import HybridLSTMProcessRegressor
from milling_experiment_framework.models.h2_regressors import canonical_model_name, create_h2_feature_pipeline
from milling_experiment_framework.preprocessing.vb_common import COMMON_VB_PREPROCESSING_RULES, apply_common_vb_prediction_preprocessing
from milling_experiment_framework.visualization.figure_export import save_figure_dual


PREFIX = "H4_S1"
TOPIC = "signal_prefix_length_model_prediction_effect"
DEFAULT_CASE_SCOPE = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
DEFAULT_SENSORS = ["smcDC", "smcAC", "vib_spindle", "vib_table", "AE_spindle", "AE_table"] #
DEFAULT_PREFIX_PERCENTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]#[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
DEFAULT_MODELS = ["stacking_ensemble"] # "random_forest", "extra_trees", "svr", "xgboost", "lightgbm", "feature_gru", "mlp", "hybrid_lstm_process", "stacking_ensemble"
DEFAULT_SEEDS = [0, 1, 2]
HYBRID_MODEL_NAME = "hybrid_lstm_process"
STACKING_MODEL_NAME = "stacking_ensemble"
HYBRID_MODEL_ALIASES = {
    "hybrid_lstm": HYBRID_MODEL_NAME,
    "hybrid-lstm": HYBRID_MODEL_NAME,
    "hybrid_lstm_process": HYBRID_MODEL_NAME,
    "hybrid-lstm-process": HYBRID_MODEL_NAME,
}
STACKING_MODEL_ALIASES = {
    "stack": STACKING_MODEL_NAME,
    "stacking": STACKING_MODEL_NAME,
    "stacking_ensemble": STACKING_MODEL_NAME,
    "stacking-ensemble": STACKING_MODEL_NAME,
}
H4_MODEL_ORDER = [
    "linear_regression",
    "svr",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "mlp",
    "feature_gru",
    HYBRID_MODEL_NAME,
    STACKING_MODEL_NAME,
]
DEFAULT_HYBRID_PROCESS_COLUMNS = ["DOC", "feed", "material_name", "time"]
DEFAULT_STACK_BASE_MODELS = ["random_forest", "xgboost"]
DEFAULT_STACK_PROCESS_COLUMNS = ["DOC", "feed", "material_name", "time"]
DEFAULT_FEATURES = [
    "mean",
    "std",
    "max",
    "min",
    "peak_to_peak",
    "kurtosis",
    "skewness",
    "spectral_centroid",
    "band_energy",
]
SIGNATURE_FILE = "H4_S1_resume_signature.json"
PROGRESS_FILE = "H4_S1_progress_state.json"
DEFAULT_MAX_SIGNAL_ABS = 1_000_000.0
FEATURE_CLIP_ABS = 1.0e30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H4_S1 signal-prefix length model prediction effect experiment.")
    parser.add_argument("--signal-path", default="datasets/processed/mill_signal_data.csv")
    parser.add_argument("--process-info-path", default="datasets/processed/mill_process_info.csv")
    parser.add_argument("--case-scope", type=int, nargs="+", default=DEFAULT_CASE_SCOPE)
    parser.add_argument("--sensors", nargs="+", default=DEFAULT_SENSORS)
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--prefix-percents", type=float, nargs="+", default=DEFAULT_PREFIX_PERCENTS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output-root", default="experiments/executions")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--max-signal-abs",
        type=float,
        default=DEFAULT_MAX_SIGNAL_ABS,
        help="Clip raw signal samples to +/- this value before feature extraction; use 0 to disable.",
    )
    parser.add_argument("--hybrid-sequence-length", type=int, default=256)
    parser.add_argument("--hybrid-hidden-size", type=int, default=64)
    parser.add_argument("--hybrid-num-layers", type=int, default=1)
    parser.add_argument("--hybrid-epochs", type=int, default=80)
    parser.add_argument("--hybrid-batch-size", type=int, default=16)
    parser.add_argument("--hybrid-learning-rate", type=float, default=0.001)
    parser.add_argument("--hybrid-weight-decay", type=float, default=0.0)
    parser.add_argument("--hybrid-lstm-dropout", type=float, default=0.0)
    parser.add_argument("--hybrid-head-dropout", type=float, default=0.0)
    parser.add_argument(
        "--hybrid-input-mode",
        choices=["sensor_only", "sensor_plus_process", "process_only"],
        default="sensor_only",
    )
    parser.add_argument("--hybrid-process-columns", nargs="+", default=DEFAULT_HYBRID_PROCESS_COLUMNS)
    parser.add_argument("--hybrid-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--stack-base-models", nargs="+", default=DEFAULT_STACK_BASE_MODELS)
    parser.add_argument(
        "--stack-meta-learner",
        choices=["logistic_regressor", "logistic_regression", "mlp", "linear_regression"],
        default="logistic_regressor",
    )
    parser.add_argument("--stack-use-process-info", default=False, action="store_true")
    parser.add_argument("--stack-process-columns", nargs="+", default=DEFAULT_STACK_PROCESS_COLUMNS)
    parser.add_argument("--stack-logistic-bins", type=int, default=5)
    parser.add_argument("--stack-mlp-hidden-layers", type=int, nargs="+", default=[32, 16])
    parser.add_argument("--stack-mlp-max-iter", type=int, default=500)
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        path.write_text(yaml.safe_dump(to_builtin(payload), sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        write_json(path.with_suffix(".json"), payload)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def parse_signal(value: Any) -> np.ndarray:
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def execution_dir(output_root: Path, timestamp: str) -> Path:
    return output_root / "H4" / "S1" / f"{timestamp}_{TOPIC}"


def timestamp_from_execution_dir(output_dir: Path) -> str:
    suffix = f"_{TOPIC}"
    if output_dir.name.endswith(suffix):
        return output_dir.name[: -len(suffix)]
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def make_dirs(output_dir: Path) -> None:
    for rel in ["configs", "data", "splits", "metrics", "predictions", "analysis", "figures", "logs", "reports"]:
        (output_dir / rel).mkdir(parents=True, exist_ok=True)


def log_progress(message: str) -> None:
    print(f"[{PREFIX}] {message}", file=sys.stderr, flush=True)


def append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)


def task_id(percent: float | int, model_name: str, seed: int, target_case: int) -> str:
    return f"prefix={float(percent):g}|model={model_name}|seed={int(seed)}|case={int(target_case)}"


def task_id_from_row(row: Any) -> str:
    return task_id(row.signal_prefix_percent, row.model, row.seed, row.target_case)


def build_resume_signature(args: argparse.Namespace, models: list[str]) -> dict[str, Any]:
    payload = {
        "script": Path(__file__).name,
        "experiment": PREFIX,
        "topic": TOPIC,
        "signal_path": args.signal_path,
        "process_info_path": args.process_info_path,
        "case_scope": sorted(int(case) for case in args.case_scope),
        "sensors": list(args.sensors),
        "features": list(args.features),
        "prefix_percents": [float(percent) for percent in args.prefix_percents],
        "models": list(models),
        "seeds": [int(seed) for seed in args.seeds],
        "max_signal_abs": float(args.max_signal_abs),
        "feature_clip_abs": float(FEATURE_CLIP_ABS),
        "hybrid_lstm_process": {
            "input_mode": args.hybrid_input_mode,
            "sequence_length": int(args.hybrid_sequence_length),
            "process_columns": list(args.hybrid_process_columns),
            "hidden_size": int(args.hybrid_hidden_size),
            "num_layers": int(args.hybrid_num_layers),
            "epochs": int(args.hybrid_epochs),
            "batch_size": int(args.hybrid_batch_size),
            "learning_rate": float(args.hybrid_learning_rate),
            "weight_decay": float(args.hybrid_weight_decay),
            "lstm_dropout": float(args.hybrid_lstm_dropout),
            "head_dropout": float(args.hybrid_head_dropout),
            "device": args.hybrid_device,
        },
        "stacking_ensemble": {
            "base_models": list(args.stack_base_models),
            "meta_learner": args.stack_meta_learner,
            "use_process_info": bool(args.stack_use_process_info),
            "process_columns": list(args.stack_process_columns),
            "logistic_bins": int(args.stack_logistic_bins),
            "mlp_hidden_layers": [int(value) for value in args.stack_mlp_hidden_layers],
            "mlp_max_iter": int(args.stack_mlp_max_iter),
        },
        "common_preprocessing": COMMON_VB_PREPROCESSING_RULES,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"signature_version": 1, "signature_sha256": digest, "payload": payload}


def progress_path(output_dir: Path) -> Path:
    return output_dir / "logs" / PROGRESS_FILE


def write_progress_state(
    output_dir: Path,
    *,
    status: str,
    total_tasks: int,
    completed_tasks: int,
    current_task: dict[str, Any] | None = None,
) -> None:
    previous = read_json(progress_path(output_dir))
    now = datetime.now().isoformat(timespec="seconds")
    state = {
        **previous,
        "status": status,
        "total_tasks": int(total_tasks),
        "completed_tasks": int(completed_tasks),
        "percent_complete": float(completed_tasks / total_tasks * 100.0) if total_tasks else 100.0,
        "current_task": current_task,
        "started_at": previous.get("started_at", now),
        "updated_at": now,
    }
    write_json(progress_path(output_dir), state)


def mark_interrupted(output_dir: Path) -> None:
    state = read_json(progress_path(output_dir))
    state.update({"status": "interrupted", "updated_at": datetime.now().isoformat(timespec="seconds")})
    write_json(progress_path(output_dir), state)


def completed_task_ids(metric_path: Path, resume: bool) -> set[str]:
    if not resume or not metric_path.exists() or metric_path.stat().st_size == 0:
        return set()
    existing = pd.read_csv(metric_path)
    required = {"signal_prefix_percent", "model", "seed", "target_case"}
    if not required.issubset(existing.columns):
        return set()
    return {task_id_from_row(row) for row in existing.itertuples(index=False)}


def find_resume_dir(output_root: Path, signature: dict[str, Any]) -> Path | None:
    root = output_root / "H4" / "S1"
    if not root.exists():
        return None
    candidates = sorted(root.glob(f"*_{TOPIC}"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        candidate_signature = read_json(candidate / "logs" / SIGNATURE_FILE)
        if candidate_signature != signature:
            continue
        state = read_json(progress_path(candidate))
        if state.get("status") != "completed":
            return candidate
    return None


def prepare_execution_dir(args: argparse.Namespace, output_root: Path, signature: dict[str, Any]) -> tuple[Path, str, bool]:
    if args.timestamp:
        output_dir = execution_dir(output_root, args.timestamp)
        if output_dir.exists() and not args.resume:
            raise FileExistsError(f"Execution directory already exists: {output_dir}")
        resumed = output_dir.exists()
        return output_dir, args.timestamp, resumed

    if args.resume:
        resume_dir = find_resume_dir(output_root, signature)
        if resume_dir is not None:
            return resume_dir, timestamp_from_execution_dir(resume_dir), True

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return execution_dir(output_root, timestamp), timestamp, False


def normalize_prefix_percent(value: float) -> float | int:
    value = float(value)
    return int(value) if value.is_integer() else value


def validate_prefix_percents(values: list[float]) -> list[float | int]:
    percents = sorted({normalize_prefix_percent(value) for value in values}, key=float)
    invalid = [value for value in percents if float(value) <= 0.0 or float(value) > 100.0]
    if invalid:
        raise ValueError(f"prefix percents must be in (0, 100], got {invalid}")
    return percents


def validate_features(features: list[str]) -> list[str]:
    selected = []
    for feature in features:
        name = str(feature).strip()
        if name and name not in selected:
            selected.append(name)
    missing = [name for name in selected if name not in DEFAULT_FEATURES]
    if missing:
        raise ValueError(f"Unsupported features: {missing}. Supported features: {DEFAULT_FEATURES}")
    if not selected:
        raise ValueError("At least one feature must be selected.")
    return selected


def canonical_h6_model_name(model_name: str) -> str:
    key = str(model_name).strip().lower()
    if key in HYBRID_MODEL_ALIASES:
        return HYBRID_MODEL_ALIASES[key]
    if key in STACKING_MODEL_ALIASES:
        return STACKING_MODEL_ALIASES[key]
    return canonical_model_name(key)


def ordered_h6_models(models: list[str]) -> list[str]:
    selected = []
    for model in models:
        name = canonical_h6_model_name(model)
        if name not in H4_MODEL_ORDER:
            raise ValueError(f"Unsupported H4_S1 model: {model}. Supported models: {H4_MODEL_ORDER}")
        if name not in selected:
            selected.append(name)
    order = {name: idx for idx, name in enumerate(H4_MODEL_ORDER)}
    return sorted(selected, key=lambda name: order[name])


def effective_seeds_for_h6_model(model_name: str, seeds: list[int]) -> list[int]:
    if model_name in {HYBRID_MODEL_NAME, STACKING_MODEL_NAME}:
        return [int(seed) for seed in seeds]
    return effective_seeds_for_model(model_name, seeds)


def validate_hybrid_args(args: argparse.Namespace) -> None:
    if int(args.hybrid_sequence_length) <= 1:
        raise ValueError("--hybrid-sequence-length must be > 1.")
    if int(args.hybrid_hidden_size) <= 0:
        raise ValueError("--hybrid-hidden-size must be > 0.")
    if int(args.hybrid_num_layers) <= 0:
        raise ValueError("--hybrid-num-layers must be > 0.")
    if int(args.hybrid_epochs) <= 0:
        raise ValueError("--hybrid-epochs must be > 0.")
    if int(args.hybrid_batch_size) <= 0:
        raise ValueError("--hybrid-batch-size must be > 0.")
    if float(args.hybrid_learning_rate) <= 0:
        raise ValueError("--hybrid-learning-rate must be > 0.")
    selected = []
    for column in args.hybrid_process_columns:
        name = str(column).strip()
        if name and name not in selected:
            selected.append(name)
    args.hybrid_process_columns = selected
    if args.hybrid_input_mode == "process_only" and not selected:
        raise ValueError("--hybrid-input-mode process_only requires --hybrid-process-columns.")


def validate_stacking_args(args: argparse.Namespace) -> None:
    allowed_base_models = {"random_forest", "xgboost", "lightgbm", "extra_trees"}
    selected_base_models = []
    for model in args.stack_base_models:
        name = canonical_model_name(str(model))
        if name not in allowed_base_models:
            raise ValueError(f"Unsupported stacking base model: {model}. Supported base models: {sorted(allowed_base_models)}")
        if name not in selected_base_models:
            selected_base_models.append(name)
    if not selected_base_models:
        raise ValueError("--stack-base-models must include at least one base model.")
    args.stack_base_models = selected_base_models
    if args.stack_meta_learner == "logistic_regression":
        args.stack_meta_learner = "logistic_regressor"
    if int(args.stack_logistic_bins) < 2:
        raise ValueError("--stack-logistic-bins must be >= 2.")
    if int(args.stack_mlp_max_iter) <= 0:
        raise ValueError("--stack-mlp-max-iter must be > 0.")
    args.stack_mlp_hidden_layers = [int(value) for value in args.stack_mlp_hidden_layers if int(value) > 0]
    if args.stack_meta_learner == "mlp" and not args.stack_mlp_hidden_layers:
        raise ValueError("--stack-mlp-hidden-layers must include at least one positive integer when --stack-meta-learner mlp.")
    selected_process_columns = []
    for column in args.stack_process_columns:
        name = str(column).strip()
        if name and name not in selected_process_columns:
            selected_process_columns.append(name)
    args.stack_process_columns = selected_process_columns


def build_config(args: argparse.Namespace, timestamp: str, output_dir: Path) -> dict[str, Any]:
    models = ordered_h6_models(args.models)
    prefix_percents = validate_prefix_percents(args.prefix_percents)
    selected_features = validate_features(args.features)
    return {
        "experiment": {
            "experiment_number": PREFIX,
            "name": f"{PREFIX}_{TOPIC}",
            "experiment_id": f"{timestamp}_{PREFIX}_{TOPIC}",
            "timestamp": timestamp,
            "execution_dir": str(output_dir),
            "analysis_type": "signal_prefix_length_model_prediction_effect",
        },
        "data": {
            "signal_path": args.signal_path,
            "process_info_path": args.process_info_path,
            "case_scope": args.case_scope,
            "case_6_excluded": 6 not in set(args.case_scope),
            "target": "VB",
            "sensors": args.sensors,
        },
        "common_preprocessing": COMMON_VB_PREPROCESSING_RULES,
        "windowing": {
            "mode": "signal_prefix_from_idx0",
            "no_load_included": True,
            "start_idx": 0,
            "prefix_percents": prefix_percents,
            "end_idx_rule": "ceil(min_selected_sensor_signal_length * prefix_percent / 100)",
        },
        "features": {
            "source": "H1/H2 nine feature set computed per sensor on each signal prefix",
            "base_feature_names": DEFAULT_FEATURES,
            "selected_feature_names": selected_features,
            "feature_naming": "{sensor}__{feature_name}",
            "raw_signal_sample_clip_abs": float(args.max_signal_abs),
            "feature_value_clip_abs": float(FEATURE_CLIP_ABS),
        },
        "modeling": {
            "task": "VB regression",
            "input": "tabular features for H2-style models; fixed-length prefix sensor sequences for hybrid_lstm_process",
            "models": models,
            "seeds": args.seeds,
            "seed_policy": "linear_regression and svr run once with seed=-1; seeded models use configured seeds",
            "hybrid_lstm_process": {
                "input_mode": args.hybrid_input_mode,
                "sequence_length": int(args.hybrid_sequence_length),
                "process_columns": args.hybrid_process_columns,
                "hidden_size": int(args.hybrid_hidden_size),
                "num_layers": int(args.hybrid_num_layers),
                "epochs": int(args.hybrid_epochs),
                "batch_size": int(args.hybrid_batch_size),
                "learning_rate": float(args.hybrid_learning_rate),
                "weight_decay": float(args.hybrid_weight_decay),
                "device": args.hybrid_device,
            },
            "stacking_ensemble": {
                "base_models": args.stack_base_models,
                "base_prediction_policy": "out-of-fold predictions on outer-train rows; final base models refit on full outer train for test predictions",
                "inner_split": "LeaveOneGroupOut by case_id when possible; otherwise seeded KFold fallback",
                "meta_learner": args.stack_meta_learner,
                "meta_learner_note": "logistic_regressor discretizes continuous VB into quantile bins and predicts expected VB from class probabilities",
                "use_process_info": bool(args.stack_use_process_info),
                "process_columns": args.stack_process_columns if args.stack_use_process_info else [],
                "logistic_bins": int(args.stack_logistic_bins),
                "mlp_hidden_layers": [int(value) for value in args.stack_mlp_hidden_layers],
                "mlp_max_iter": int(args.stack_mlp_max_iter),
            },
        },
        "split": {
            "strategy": "leave_one_case_out",
            "test_unit": "case",
            "validation": "none",
        },
    }


def compute_features(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        arr = np.asarray([0.0], dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(arr))
    freqs = np.fft.rfftfreq(len(arr), d=1.0)
    spectral_sum = float(spectrum.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "peak_to_peak": float(np.ptp(arr)),
            "kurtosis": float(kurtosis(arr, fisher=True, bias=False)) if len(arr) > 3 else 0.0,
            "skewness": float(skew(arr, bias=False)) if len(arr) > 2 else 0.0,
            "spectral_centroid": float(np.sum(freqs * spectrum) / spectral_sum) if spectral_sum > 0 else 0.0,
            "band_energy": float(np.sum(np.square(spectrum)) / max(len(arr), 1)),
        }
    return {key: (0.0 if not np.isfinite(value) else value) for key, value in raw.items()}


def sanitize_signal(arr: np.ndarray, max_abs: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if max_abs and max_abs > 0:
        arr = np.clip(arr, -float(max_abs), float(max_abs))
    return arr


def feature_columns(sensors: list[str], features: list[str]) -> list[str]:
    return [f"{sensor}__{feature}" for sensor in sensors for feature in features]


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    signal_path = resolve(args.signal_path)
    process_path = resolve(args.process_info_path)
    signal = pd.read_csv(signal_path, usecols=["case", "run", *args.sensors])
    process = pd.read_csv(process_path)
    missing_process = {"case", "run", "VB"}.difference(process.columns)
    if missing_process:
        raise ValueError(f"process info missing columns: {sorted(missing_process)}")
    missing_signal = set(args.sensors).difference(signal.columns)
    if missing_signal:
        raise ValueError(f"signal data missing sensor columns: {sorted(missing_signal)}")
    data = signal.merge(process, on=["case", "run"], how="inner", validate="one_to_one")
    data = data.loc[data["case"].isin(args.case_scope)].copy()
    data, preprocessing_report = apply_common_vb_prediction_preprocessing(data)
    if data.empty:
        raise ValueError("No rows after case/VB filtering.")
    return data, preprocessing_report


def sanitize_feature_matrix(feature_matrix: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = feature_matrix.copy()
    report: dict[str, Any] = {
        "feature_clip_abs": float(FEATURE_CLIP_ABS),
        "columns": {},
        "total_nonfinite_replaced": 0,
        "total_values_clipped": 0,
    }
    for col in features:
        values = out[col].to_numpy(dtype=np.float64)
        nonfinite_mask = ~np.isfinite(values)
        clipped_mask = np.isfinite(values) & (np.abs(values) > FEATURE_CLIP_ABS)
        cleaned = np.nan_to_num(values, nan=0.0, posinf=FEATURE_CLIP_ABS, neginf=-FEATURE_CLIP_ABS)
        cleaned = np.clip(cleaned, -FEATURE_CLIP_ABS, FEATURE_CLIP_ABS)
        out[col] = cleaned
        nonfinite_count = int(nonfinite_mask.sum())
        clipped_count = int(clipped_mask.sum())
        report["total_nonfinite_replaced"] += nonfinite_count
        report["total_values_clipped"] += clipped_count
        if nonfinite_count or clipped_count:
            report["columns"][col] = {
                "nonfinite_replaced": nonfinite_count,
                "values_clipped": clipped_count,
                "max_abs_before": float(np.nanmax(np.abs(values))) if len(values) else 0.0,
                "max_abs_after": float(np.max(np.abs(cleaned))) if len(cleaned) else 0.0,
            }
    return out, report


def build_feature_matrix(
    data: pd.DataFrame,
    sensors: list[str],
    features: list[str],
    prefix_percents: list[float | int],
    max_signal_abs: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_row in data.itertuples(index=False):
        case = int(source_row.case)
        run_id = int(source_row.run)
        arrays = {sensor: sanitize_signal(parse_signal(getattr(source_row, sensor)), max_signal_abs) for sensor in sensors}
        lengths = {sensor: int(len(arr)) for sensor, arr in arrays.items()}
        base_length = min(lengths.values())
        if base_length <= 0:
            raise ValueError(f"Empty signal for case={case}, run={run_id}")
        for percent in prefix_percents:
            prefix_end_idx = int(np.ceil(base_length * float(percent) / 100.0))
            prefix_end_idx = int(np.clip(prefix_end_idx, 1, base_length))
            row: dict[str, Any] = {
                "sample_id": f"case_{case:02d}_run_{run_id:03d}_prefix_{float(percent):g}pct",
                "dataset_run_id": f"case_{case:02d}_run_{run_id:03d}",
                "case_id": case,
                "run": run_id,
                "signal_prefix_percent": float(percent),
                "prefix_start_idx": 0,
                "prefix_end_idx": prefix_end_idx,
                "signal_length_min": base_length,
                "signal_length_max": max(lengths.values()),
                "no_load_included": True,
                "VB": float(source_row.VB),
                "time": getattr(source_row, "time", np.nan),
                "DOC": getattr(source_row, "DOC", np.nan),
                "feed": getattr(source_row, "feed", np.nan),
                "material_name": getattr(source_row, "material_name", np.nan),
            }
            for sensor, arr in arrays.items():
                values = compute_features(arr[:prefix_end_idx])
                for feature_name in features:
                    row[f"{sensor}__{feature_name}"] = values[feature_name]
            rows.append(row)
    out = pd.DataFrame(rows)
    missing = [col for col in feature_columns(sensors, features) if col not in out.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return out.sort_values(["signal_prefix_percent", "case_id", "run"]).reset_index(drop=True)


def resample_1d(arr: np.ndarray, sequence_length: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros(1, dtype=np.float64)
    if arr.size == 1:
        return np.repeat(arr.astype("float32"), int(sequence_length))
    old = np.linspace(0.0, 1.0, num=len(arr), dtype=np.float32)
    new = np.linspace(0.0, 1.0, num=int(sequence_length), dtype=np.float32)
    return np.interp(new, old, arr).astype("float32")


def build_hybrid_sequence_matrix(
    data: pd.DataFrame,
    sensors: list[str],
    prefix_percents: list[float | int],
    max_signal_abs: float,
    sequence_length: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_row in data.itertuples(index=False):
        case = int(source_row.case)
        run_id = int(source_row.run)
        arrays = {sensor: sanitize_signal(parse_signal(getattr(source_row, sensor)), max_signal_abs) for sensor in sensors}
        lengths = {sensor: int(len(arr)) for sensor, arr in arrays.items()}
        base_length = min(lengths.values())
        if base_length <= 0:
            raise ValueError(f"Empty signal for case={case}, run={run_id}")
        for percent in prefix_percents:
            prefix_end_idx = int(np.ceil(base_length * float(percent) / 100.0))
            prefix_end_idx = int(np.clip(prefix_end_idx, 1, base_length))
            channels = [resample_1d(arr[:prefix_end_idx], sequence_length) for arr in arrays.values()]
            rows.append(
                {
                    "sample_id": f"case_{case:02d}_run_{run_id:03d}_prefix_{float(percent):g}pct",
                    "dataset_run_id": f"case_{case:02d}_run_{run_id:03d}",
                    "case_id": case,
                    "run": run_id,
                    "signal_prefix_percent": float(percent),
                    "prefix_start_idx": 0,
                    "prefix_end_idx": prefix_end_idx,
                    "signal_length_min": base_length,
                    "signal_length_max": max(lengths.values()),
                    "sequence_length_after_resample": int(sequence_length),
                    "no_load_included": True,
                    "VB": float(source_row.VB),
                    "time": getattr(source_row, "time", np.nan),
                    "DOC": getattr(source_row, "DOC", np.nan),
                    "feed": getattr(source_row, "feed", np.nan),
                    "material_name": getattr(source_row, "material_name", np.nan),
                    "sequence": np.stack(channels, axis=1).astype("float32"),
                }
            )
    return pd.DataFrame(rows).sort_values(["signal_prefix_percent", "case_id", "run"]).reset_index(drop=True)


class HybridPrefixDataset(Dataset):
    def __init__(self, sequences: np.ndarray, process_features: np.ndarray, targets: np.ndarray | None = None) -> None:
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.process_features = torch.as_tensor(process_features, dtype=torch.float32)
        self.targets = None if targets is None else torch.as_tensor(targets, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return int(len(self.sequences))

    def __getitem__(self, index: int):
        if self.targets is None:
            return self.sequences[index], self.process_features[index]
        return self.sequences[index], self.process_features[index], self.targets[index]


class ProcessPreprocessor:
    def __init__(self, columns: list[str]) -> None:
        self.columns = list(columns)
        self.numeric_cols = [column for column in self.columns if column != "material_name"]
        self.categorical_cols = [column for column in self.columns if column == "material_name"]
        self.numeric_mean: pd.Series | None = None
        self.numeric_std: pd.Series | None = None
        self.categories: dict[str, list[str]] = {}

    def fit(self, frame: pd.DataFrame) -> "ProcessPreprocessor":
        missing = [column for column in self.columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Hybrid process columns missing from data: {missing}")
        if self.numeric_cols:
            numeric = frame[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
            self.numeric_mean = numeric.mean().fillna(0.0)
            self.numeric_std = numeric.std().replace(0, 1.0).fillna(1.0)
        for col in self.categorical_cols:
            self.categories[col] = sorted(frame[col].fillna("__missing__").astype(str).unique().tolist())
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.numeric_cols:
            assert self.numeric_mean is not None and self.numeric_std is not None
            numeric = frame[self.numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(self.numeric_mean)
            parts.append(((numeric - self.numeric_mean) / self.numeric_std).to_numpy(dtype="float32"))
        for col in self.categorical_cols:
            cats = self.categories[col]
            values = frame[col].fillna("__missing__").astype(str)
            one_hot = np.zeros((len(frame), len(cats)), dtype="float32")
            cat_to_idx = {cat: idx for idx, cat in enumerate(cats)}
            for row_idx, value in enumerate(values):
                if value in cat_to_idx:
                    one_hot[row_idx, cat_to_idx[value]] = 1.0
            parts.append(one_hot)
        if not parts:
            return np.zeros((len(frame), 0), dtype="float32")
        return np.concatenate(parts, axis=1).astype("float32")


def stack_sequences(frame: pd.DataFrame) -> np.ndarray:
    return np.stack(frame["sequence"].to_numpy()).astype("float32")


def normalize_sequences(train_seq: np.ndarray, test_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_seq.mean(axis=(0, 1), keepdims=True)
    std = train_seq.std(axis=(0, 1), keepdims=True)
    std = np.where(std <= 1e-8, 1.0, std)
    return ((train_seq - mean) / std).astype("float32"), ((test_seq - mean) / std).astype("float32")


def set_hybrid_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def predict_hybrid_loader(model: nn.Module, loader: DataLoader, device: torch.device, input_mode: str) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            sequences, process_features = batch
            sequences = sequences.to(device)
            process_features = process_features.to(device)
            sensor_input = None if input_mode == "process_only" else sequences
            process_input = process_features if input_mode in {"process_only", "sensor_plus_process"} else None
            pred = model(sensor_input, process_input).detach().cpu().numpy().reshape(-1)
            preds.append(pred)
    return np.concatenate(preds) if preds else np.array([], dtype="float32")


def fit_predict_hybrid_lstm(train: pd.DataFrame, test: pd.DataFrame, sensors: list[str], args: argparse.Namespace, seed: int) -> np.ndarray:
    set_hybrid_seed(model_seed_value(seed))
    device = torch.device(args.hybrid_device if args.hybrid_device == "cuda" and torch.cuda.is_available() else "cpu")
    train_seq, test_seq = normalize_sequences(stack_sequences(train), stack_sequences(test))

    process_columns = args.hybrid_process_columns if args.hybrid_input_mode in {"process_only", "sensor_plus_process"} else []
    preprocessor = ProcessPreprocessor(process_columns).fit(train)
    train_proc = preprocessor.transform(train)
    test_proc = preprocessor.transform(test)

    y_train = train["VB"].to_numpy(dtype="float32")
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std <= 1e-8:
        y_std = 1.0
    y_train_norm = ((y_train - y_mean) / y_std).astype("float32")

    process_dim = int(train_proc.shape[1])
    model = HybridLSTMProcessRegressor(
        num_sensors=len(sensors),
        process_feature_dim=process_dim,
        hidden_size=int(args.hybrid_hidden_size),
        num_layers=int(args.hybrid_num_layers),
        lstm_dropout=float(args.hybrid_lstm_dropout),
        bidirectional=False,
        temporal_output_mode="last_hidden",
        regression_hidden_dims=(32, 8),
        regression_dropout=float(args.hybrid_head_dropout),
        use_process_info=process_dim > 0 and args.hybrid_input_mode != "sensor_only",
        input_mode=args.hybrid_input_mode,
    ).to(device)

    generator = torch.Generator()
    generator.manual_seed(model_seed_value(seed))
    train_loader = DataLoader(
        HybridPrefixDataset(train_seq, train_proc, y_train_norm),
        batch_size=max(1, int(args.hybrid_batch_size)),
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(
        HybridPrefixDataset(test_seq, test_proc),
        batch_size=max(1, int(args.hybrid_batch_size)),
        shuffle=False,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.hybrid_learning_rate),
        weight_decay=float(args.hybrid_weight_decay),
    )
    loss_fn = nn.MSELoss()
    input_mode = args.hybrid_input_mode
    for _ in range(max(1, int(args.hybrid_epochs))):
        model.train()
        for sequences, process_features, targets in train_loader:
            sequences = sequences.to(device)
            process_features = process_features.to(device)
            targets = targets.to(device)
            sensor_input = None if input_mode == "process_only" else sequences
            process_input = process_features if input_mode in {"process_only", "sensor_plus_process"} else None
            optimizer.zero_grad()
            loss = loss_fn(model(sensor_input, process_input), targets)
            loss.backward()
            optimizer.step()

    pred_norm = predict_hybrid_loader(model, test_loader, device, input_mode)
    return pred_norm * y_std + y_mean


class QuantileLogisticVBRegressor:
    def __init__(self, bins: int = 5, random_state: int = 0) -> None:
        self.bins = int(bins)
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileLogisticVBRegressor":
        x = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64).reshape(-1)
        self.constant_prediction_ = float(np.mean(target)) if len(target) else 0.0
        unique_y = np.unique(target)
        if len(unique_y) < 2 or len(target) < 3:
            self.is_constant_ = True
            return self

        bins = min(int(self.bins), len(unique_y), len(target))
        edges = np.unique(np.quantile(target, np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) <= 2:
            self.is_constant_ = True
            return self

        labels = np.digitize(target, edges[1:-1], right=True)
        classes = np.unique(labels)
        if len(classes) < 2:
            self.is_constant_ = True
            return self

        self.bin_values_ = {int(label): float(np.mean(target[labels == label])) for label in classes}
        self.scaler_ = StandardScaler()
        x_scaled = self.scaler_.fit_transform(x)
        self.model_ = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self.model_.fit(x_scaled, labels)
        self.is_constant_ = False
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64)
        if getattr(self, "is_constant_", False):
            return np.full(len(x), self.constant_prediction_, dtype=np.float64)
        proba = self.model_.predict_proba(self.scaler_.transform(x))
        centers = np.asarray([self.bin_values_[int(label)] for label in self.model_.classes_], dtype=np.float64)
        return proba @ centers


def stacking_inner_splits(train: pd.DataFrame, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    row_indices = np.arange(len(train))
    groups = train["case_id"].to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        return [(train_idx, val_idx) for train_idx, val_idx in LeaveOneGroupOut().split(row_indices, groups=groups)]
    if len(train) >= 2:
        n_splits = min(5, len(train))
        return [(train_idx, val_idx) for train_idx, val_idx in KFold(n_splits=n_splits, shuffle=True, random_state=model_seed_value(seed)).split(row_indices)]
    return [(row_indices, row_indices)]


def build_stacking_meta_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    splits = stacking_inner_splits(train, seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    meta_names: list[str] = []
    for base_model_name in args.stack_base_models:
        oof_pred = np.zeros(len(train), dtype=np.float64)
        for inner_train_idx, inner_val_idx in splits:
            model = create_h2_feature_pipeline(base_model_name, model_seed_value(seed), model_defaults=None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train.iloc[inner_train_idx][features], train.iloc[inner_train_idx]["VB"])
                oof_pred[inner_val_idx] = model.predict(train.iloc[inner_val_idx][features])
        final_model = create_h2_feature_pipeline(base_model_name, model_seed_value(seed), model_defaults=None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_model.fit(train[features], train["VB"])
            test_pred = final_model.predict(test[features])
        train_parts.append(oof_pred.reshape(-1, 1))
        test_parts.append(np.asarray(test_pred, dtype=np.float64).reshape(-1, 1))
        meta_names.append(f"{base_model_name}__oof_prediction")

    if args.stack_use_process_info:
        preprocessor = ProcessPreprocessor(args.stack_process_columns).fit(train)
        train_proc = preprocessor.transform(train).astype(np.float64)
        test_proc = preprocessor.transform(test).astype(np.float64)
        train_parts.append(train_proc)
        test_parts.append(test_proc)
        meta_names.extend([f"process__{column}" for column in args.stack_process_columns])

    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1), meta_names


def fit_predict_stacking_ensemble(train: pd.DataFrame, test: pd.DataFrame, features: list[str], args: argparse.Namespace, seed: int) -> np.ndarray:
    meta_train, meta_test, _ = build_stacking_meta_features(train, test, features, args, seed)
    y_train = train["VB"].to_numpy(dtype=np.float64)
    meta_learner = args.stack_meta_learner
    if meta_learner == "logistic_regressor":
        return QuantileLogisticVBRegressor(args.stack_logistic_bins, model_seed_value(seed)).fit(meta_train, y_train).predict(meta_test)
    if meta_learner == "mlp":
        x_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(meta_train)
        x_test = x_scaler.transform(meta_test)
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train))
        if y_std <= 1e-8:
            y_std = 1.0
        y_norm = (y_train - y_mean) / y_std
        model = MLPRegressor(
            hidden_layer_sizes=tuple(int(value) for value in args.stack_mlp_hidden_layers),
            max_iter=int(args.stack_mlp_max_iter),
            random_state=model_seed_value(seed),
            learning_rate_init=0.001,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_norm)
        return model.predict(x_test) * y_std + y_mean
    if meta_learner == "linear_regression":
        x_scaler = StandardScaler()
        model = LinearRegression()
        model.fit(x_scaler.fit_transform(meta_train), y_train)
        return model.predict(x_scaler.transform(meta_test))
    raise ValueError(f"Unsupported stack meta learner: {meta_learner}")


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    case_vb_min = float(np.min(y_true))
    case_vb_max = float(np.max(y_true))
    case_vb_mean = float(np.mean(y_true))
    case_vb_mean_abs = float(np.mean(np.abs(y_true)))
    case_vb_std = float(np.std(y_true))
    case_vb_range = float(case_vb_max - case_vb_min)
    out = {
        "metric_mae": mae,
        "metric_rmse": rmse,
        "metric_r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "case_vb_min": case_vb_min,
        "case_vb_max": case_vb_max,
        "case_vb_mean": case_vb_mean,
        "case_vb_mean_abs": case_vb_mean_abs,
        "case_vb_std": case_vb_std,
        "case_vb_range": case_vb_range,
        "metric_nmae_case_range": mae / case_vb_range if case_vb_range > 0 else float("nan"),
        "metric_nrmse_case_range": rmse / case_vb_range if case_vb_range > 0 else float("nan"),
        "metric_nmae_case_mean_abs": mae / case_vb_mean_abs if case_vb_mean_abs > 0 else float("nan"),
        "metric_nrmse_case_mean_abs": rmse / case_vb_mean_abs if case_vb_mean_abs > 0 else float("nan"),
    }
    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        out["metric_pearson"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        out["metric_pearson"] = float("nan")
    return out


def split_for_case(data: pd.DataFrame, target_case: int) -> pd.DataFrame:
    split = data[
        [
            "sample_id",
            "dataset_run_id",
            "case_id",
            "run",
            "signal_prefix_percent",
            "prefix_start_idx",
            "prefix_end_idx",
            "VB",
        ]
    ].copy()
    split["split"] = np.where(split["case_id"].eq(target_case), "test", "train")
    return split


def run_grid(
    feature_matrix: pd.DataFrame,
    sequence_matrix: pd.DataFrame | None,
    features: list[str],
    models: list[str],
    seeds: list[int],
    sensors: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shift_path = output_dir / "metrics" / "H4_S1_shift_metrics.csv"
    prediction_path = output_dir / "predictions" / "H4_S1_predictions.csv"
    split_path = output_dir / "splits" / "H4_S1_splits.csv"
    cases = sorted(feature_matrix["case_id"].unique().tolist())
    percents = sorted(feature_matrix["signal_prefix_percent"].unique().tolist())
    total_tasks = int(sum(len(effective_seeds_for_h6_model(model, seeds)) * len(cases) for model in models) * len(percents))
    done = completed_task_ids(shift_path, resume)
    completed = len(done)
    write_progress_state(output_dir, status="running", total_tasks=total_tasks, completed_tasks=completed)
    if completed:
        log_progress(f"Resuming: {completed}/{total_tasks} atomic fits already completed.")
    else:
        log_progress(f"Training grid started: {total_tasks} atomic fits planned.")

    for percent in percents:
        for model_name in models:
            model_frame = sequence_matrix if model_name == HYBRID_MODEL_NAME else feature_matrix
            if model_frame is None:
                raise ValueError(f"{HYBRID_MODEL_NAME} was selected but no sequence matrix was built.")
            length_df = model_frame.loc[model_frame["signal_prefix_percent"].eq(percent)].copy()
            for seed in effective_seeds_for_h6_model(model_name, seeds):
                for target_case in cases:
                    current_id = task_id(percent, model_name, seed, target_case)
                    current_task = {
                        "task_id": current_id,
                        "signal_prefix_percent": float(percent),
                        "model": model_name,
                        "seed": int(seed),
                        "seed_label": seed_label(seed),
                        "target_case": int(target_case),
                    }
                    if current_id in done:
                        continue
                    write_progress_state(
                        output_dir,
                        status="running",
                        total_tasks=total_tasks,
                        completed_tasks=completed,
                        current_task=current_task,
                    )
                    split = split_for_case(length_df, target_case)
                    train = length_df.loc[length_df["case_id"].ne(target_case)].copy()
                    test = length_df.loc[length_df["case_id"].eq(target_case)].copy()
                    if train.empty or test.empty:
                        raise ValueError(f"Empty split for prefix={percent:g}%, target_case={target_case}")
                    if model_name == HYBRID_MODEL_NAME:
                        y_pred = fit_predict_hybrid_lstm(train, test, sensors, args, seed)
                    elif model_name == STACKING_MODEL_NAME:
                        y_pred = fit_predict_stacking_ensemble(train, test, features, args, seed)
                    else:
                        model = create_h2_feature_pipeline(model_name, model_seed_value(seed), model_defaults=None)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model.fit(train[features], train["VB"])
                            y_pred = model.predict(test[features])
                    metrics = metric_dict(test["VB"].to_numpy(), y_pred)
                    scenario = f"train_without_case_{target_case}_to_case_{target_case}"
                    shift_row = {
                        "signal_prefix_percent": float(percent),
                        "model": model_name,
                        "seed": int(seed),
                        "seed_label": seed_label(seed),
                        "target_case": int(target_case),
                        "source_cases": ",".join(map(str, [case for case in cases if case != target_case])),
                        "target_cases": str(int(target_case)),
                        "scenario": scenario,
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        **metrics,
                    }
                    shift_row["task_id"] = current_id
                    preds = test[
                        [
                            "sample_id",
                            "dataset_run_id",
                            "case_id",
                            "run",
                            "signal_prefix_percent",
                            "prefix_start_idx",
                            "prefix_end_idx",
                            "VB",
                        ]
                    ].copy()
                    preds["model"] = model_name
                    preds["seed"] = int(seed)
                    preds["seed_label"] = seed_label(seed)
                    preds["target_case"] = int(target_case)
                    preds["scenario"] = scenario
                    preds["y_true"] = preds["VB"]
                    preds["y_pred"] = y_pred
                    preds["residual"] = preds["y_true"] - preds["y_pred"]
                    preds["absolute_error"] = preds["residual"].abs()
                    preds["task_id"] = current_id
                    append_csv(preds, prediction_path)
                    split["model"] = model_name
                    split["seed"] = int(seed)
                    split["seed_label"] = seed_label(seed)
                    split["target_case"] = int(target_case)
                    split["scenario"] = scenario
                    split["task_id"] = current_id
                    append_csv(split, split_path)
                    append_csv(pd.DataFrame([shift_row]), shift_path)
                    done.add(current_id)
                    completed += 1
                    write_progress_state(
                        output_dir,
                        status="running",
                        total_tasks=total_tasks,
                        completed_tasks=completed,
                        current_task=current_task,
                    )
                    log_progress(
                        f"Progress {completed}/{total_tasks} ({completed / total_tasks * 100:.1f}%) "
                        f"prefix={float(percent):g}% model={model_name} seed={seed_label(seed)} case={target_case}"
                    )

    write_progress_state(output_dir, status="completed", total_tasks=total_tasks, completed_tasks=completed, current_task=None)
    shift_metrics = pd.read_csv(shift_path).drop_duplicates(["task_id"], keep="last")
    predictions = pd.read_csv(prediction_path).drop_duplicates(["task_id", "sample_id"], keep="last")
    splits = pd.read_csv(split_path).drop_duplicates(["task_id", "sample_id"], keep="last")
    return shift_metrics, predictions, splits


def aggregate_metrics(shift_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_metrics = (
        shift_metrics.groupby(["signal_prefix_percent", "model", "seed", "seed_label"], as_index=False)
        .agg(
            mean_mae_over_cases=("metric_mae", "mean"),
            mean_rmse_over_cases=("metric_rmse", "mean"),
            mean_nmae_case_range_over_cases=("metric_nmae_case_range", "mean"),
            mean_nrmse_case_range_over_cases=("metric_nrmse_case_range", "mean"),
            mean_nmae_case_mean_abs_over_cases=("metric_nmae_case_mean_abs", "mean"),
            mean_nrmse_case_mean_abs_over_cases=("metric_nrmse_case_mean_abs", "mean"),
            mean_r2_over_cases=("metric_r2", "mean"),
            mean_pearson_over_cases=("metric_pearson", "mean"),
            std_rmse_over_cases=("metric_rmse", "std"),
            test_case_count=("target_case", "nunique"),
            total_test_rows=("test_rows", "sum"),
        )
    )
    seed_metrics["std_rmse_over_cases"] = seed_metrics["std_rmse_over_cases"].fillna(0.0)
    model_metrics = (
        seed_metrics.groupby(["signal_prefix_percent", "model"], as_index=False)
        .agg(
            mean_mae=("mean_mae_over_cases", "mean"),
            std_mae=("mean_mae_over_cases", "std"),
            mean_rmse=("mean_rmse_over_cases", "mean"),
            std_rmse=("mean_rmse_over_cases", "std"),
            mean_nmae_case_range=("mean_nmae_case_range_over_cases", "mean"),
            mean_nrmse_case_range=("mean_nrmse_case_range_over_cases", "mean"),
            mean_nmae_case_mean_abs=("mean_nmae_case_mean_abs_over_cases", "mean"),
            mean_nrmse_case_mean_abs=("mean_nrmse_case_mean_abs_over_cases", "mean"),
            mean_r2=("mean_r2_over_cases", "mean"),
            std_r2=("mean_r2_over_cases", "std"),
            mean_pearson=("mean_pearson_over_cases", "mean"),
            seed_count=("seed", "nunique"),
            test_case_count=("test_case_count", "max"),
            total_test_rows=("total_test_rows", "max"),
        )
        .reset_index(drop=True)
    )
    for col in ["std_mae", "std_rmse", "std_r2"]:
        model_metrics[col] = model_metrics[col].fillna(0.0)
    model_metrics["rank_by_rmse_within_model"] = model_metrics.groupby("model")["mean_rmse"].rank(ascending=True, method="first").astype(int)
    model_metrics["rank_by_rmse_overall"] = model_metrics["mean_rmse"].rank(ascending=True, method="first").astype(int)
    best_by_model = model_metrics.sort_values(["model", "mean_rmse"]).groupby("model", as_index=False).head(1)
    model_metrics = model_metrics.sort_values(["model", "signal_prefix_percent"]).reset_index(drop=True)
    return seed_metrics, model_metrics, best_by_model.reset_index(drop=True)


def build_casewise_metrics(shift_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_columns = [
        "signal_prefix_percent",
        "model",
        "seed",
        "seed_label",
        "target_case",
        "scenario",
        "train_rows",
        "test_rows",
        "case_vb_min",
        "case_vb_max",
        "case_vb_mean",
        "case_vb_mean_abs",
        "case_vb_std",
        "case_vb_range",
        "metric_mae",
        "metric_rmse",
        "metric_nmae_case_range",
        "metric_nrmse_case_range",
        "metric_nmae_case_mean_abs",
        "metric_nrmse_case_mean_abs",
        "metric_r2",
        "metric_pearson",
    ]
    case_metrics = shift_metrics[[col for col in case_columns if col in shift_metrics.columns]].copy()
    case_model_metrics = (
        case_metrics.groupby(["signal_prefix_percent", "model", "target_case"], as_index=False)
        .agg(
            case_vb_min=("case_vb_min", "first"),
            case_vb_max=("case_vb_max", "first"),
            case_vb_mean=("case_vb_mean", "first"),
            case_vb_mean_abs=("case_vb_mean_abs", "first"),
            case_vb_std=("case_vb_std", "first"),
            case_vb_range=("case_vb_range", "first"),
            mean_mae=("metric_mae", "mean"),
            std_mae=("metric_mae", "std"),
            mean_rmse=("metric_rmse", "mean"),
            std_rmse=("metric_rmse", "std"),
            mean_nmae_case_range=("metric_nmae_case_range", "mean"),
            mean_nrmse_case_range=("metric_nrmse_case_range", "mean"),
            mean_nmae_case_mean_abs=("metric_nmae_case_mean_abs", "mean"),
            mean_nrmse_case_mean_abs=("metric_nrmse_case_mean_abs", "mean"),
            mean_r2=("metric_r2", "mean"),
            mean_pearson=("metric_pearson", "mean"),
            seed_count=("seed", "nunique"),
            test_rows=("test_rows", "max"),
        )
        .reset_index(drop=True)
    )
    for col in ["std_mae", "std_rmse"]:
        case_model_metrics[col] = case_model_metrics[col].fillna(0.0)
    return (
        case_metrics.sort_values(["model", "signal_prefix_percent", "target_case", "seed"]).reset_index(drop=True),
        case_model_metrics.sort_values(["model", "signal_prefix_percent", "target_case"]).reset_index(drop=True),
    )


def plot_prefix_effect(output_dir: Path, model_metrics: pd.DataFrame, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for model_name, group in model_metrics.groupby("model"):
        group = group.sort_values("signal_prefix_percent")
        x = group["signal_prefix_percent"].to_numpy(dtype=float)
        ax.plot(x, group["mean_rmse"], marker="o", linewidth=1.4, label=model_name)
        ax.fill_between(
            x,
            (group["mean_rmse"] - group["std_rmse"]).to_numpy(dtype=float),
            (group["mean_rmse"] + group["std_rmse"]).to_numpy(dtype=float),
            alpha=0.12,
        )
    ax.set_xlabel("signal prefix length from IDX=0 (%)")
    ax.set_ylabel("mean RMSE over leave-one-case-out folds")
    ax.set_title(f"{PREFIX} prediction error by input signal prefix length")
    ax.grid(True, color="#e5e7eb", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "figures" / "H4_S1_signal_prefix_length_effect_rmse.png"
    save_figure_dual(fig, path, dpi=dpi)
    plt.close(fig)
    return path


def write_report(output_dir: Path, summary: dict[str, Any], model_metrics: pd.DataFrame, best_by_model: pd.DataFrame) -> None:
    overall = model_metrics.sort_values("mean_rmse").head(12)
    lines = [
        f"# {PREFIX} Signal-prefix Length Model Prediction Effect",
        "",
        "## Scope",
        "",
        f"- Sensors: `{summary['sensors']}`",
        f"- Prefix percents: `{summary['prefix_percents']}`",
        "- Prefix rule: `IDX=0` to `ceil(signal_length * percent / 100)`",
        "- No-load included: `true`",
        f"- Selected features: `{summary['selected_features']}`",
        f"- Feature input columns: `{summary['input_feature_count']}` sensor-wise features",
        f"- Split: leave-one-case-out over `{summary['case_count']}` cases",
        f"- Models: `{summary['models']}`",
        "",
        "## Outputs",
        "",
        "- Feature matrix: `data/H4_S1_feature_matrix.csv`",
        "- Hybrid LSTM sequence index: `data/H4_S1_hybrid_sequence_index.csv` when `hybrid_lstm_process` is selected",
        "- Common preprocessing: `analysis/H4_S1_common_preprocessing.json`",
        "- Feature sanitization: `analysis/H4_S1_feature_sanitization.json`",
        "- Shift metrics: `metrics/H4_S1_shift_metrics.csv`",
        "- Case-wise metrics: `metrics/H4_S1_case_metrics.csv`",
        "- Case-wise model metrics: `metrics/H4_S1_case_model_metrics.csv`",
        "- Seed metrics: `metrics/H4_S1_seed_metrics.csv`",
        "- Prefix-model metrics: `metrics/H4_S1_prefix_model_metrics.csv`",
        "- Predictions: `predictions/H4_S1_predictions.csv`",
        "- Progress state: `logs/H4_S1_progress_state.json`",
        "- Prefix effect figure: `figures/H4_S1_signal_prefix_length_effect_rmse.{png,svg}`",
        "",
        "## Best Prefix By Model",
        "",
        "| model | best_prefix_percent | mean_rmse | mean_mae | mean_r2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in best_by_model.itertuples(index=False):
        lines.append(f"| {row.model} | {row.signal_prefix_percent:g} | {row.mean_rmse:.6f} | {row.mean_mae:.6f} | {row.mean_r2:.6f} |")
    lines.extend(["", "## Overall Top Rows", "", "| model | prefix_percent | mean_rmse | mean_mae | mean_r2 |", "|---|---:|---:|---:|---:|"])
    for row in overall.itertuples(index=False):
        lines.append(f"| {row.model} | {row.signal_prefix_percent:g} | {row.mean_rmse:.6f} | {row.mean_mae:.6f} | {row.mean_r2:.6f} |")
    lines.extend(
        [
            "",
            "## Method Note",
            "",
            "Each row uses the selected sensors from the beginning of the raw signal. "
            "No-load is intentionally retained because no offset such as `idx_noload_end` is applied.",
            "Common VB preprocessing is applied before feature construction: missing VB is interpolated within each case, then each case's first run is excluded.",
            "",
            "Case-wise metrics include raw MAE/RMSE plus normalized MAE/RMSE using each target case's VB range "
            "and mean absolute VB to make cases with different VB scales easier to compare.",
        ]
    )
    (output_dir / "reports" / "H4_S1_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.prefix_percents = validate_prefix_percents(args.prefix_percents)
    args.features = validate_features(args.features)
    validate_hybrid_args(args)
    validate_stacking_args(args)
    if float(args.max_signal_abs) < 0:
        raise ValueError("--max-signal-abs must be >= 0; use 0 to disable clipping.")
    models = ordered_h6_models(args.models)
    args.models = models
    signature = build_resume_signature(args, models)
    output_root = resolve(args.output_root)
    resume_enabled = bool(args.resume and not args.dry_run)
    output_dir, timestamp, resumed = prepare_execution_dir(args, output_root, signature) if resume_enabled else prepare_execution_dir(
        argparse.Namespace(**{**vars(args), "resume": False}),
        output_root,
        signature,
    )
    if output_dir.exists():
        existing_signature = read_json(output_dir / "logs" / SIGNATURE_FILE)
        if existing_signature and existing_signature != signature:
            raise ValueError(f"Execution directory has a different resume signature: {output_dir}")
        if not resumed and not args.dry_run:
            raise FileExistsError(f"Execution directory already exists: {output_dir}")
    make_dirs(output_dir)
    write_json(output_dir / "logs" / SIGNATURE_FILE, signature)
    if resumed:
        log_progress(f"Using existing execution directory: {output_dir}")
    else:
        log_progress(f"Using new execution directory: {output_dir}")

    config = build_config(args, timestamp, output_dir)
    write_yaml(output_dir / "configs" / "H4_S1_input_config.yaml", config)
    write_json(output_dir / "logs" / "H4_S1_environment.json", collect_environment())

    log_progress("Loading input data.")
    data, preprocessing_report = load_inputs(args)
    input_features = feature_columns(args.sensors, args.features)
    log_progress("Building signal-prefix feature matrix.")
    feature_matrix = build_feature_matrix(data, args.sensors, args.features, args.prefix_percents, args.max_signal_abs)
    feature_matrix, sanitization_report = sanitize_feature_matrix(feature_matrix, input_features)
    if sanitization_report["total_nonfinite_replaced"] or sanitization_report["total_values_clipped"]:
        log_progress(
            "Feature matrix sanitized: "
            f"{sanitization_report['total_nonfinite_replaced']} nonfinite replaced, "
            f"{sanitization_report['total_values_clipped']} values clipped."
        )
    sequence_matrix = None
    if HYBRID_MODEL_NAME in models:
        log_progress("Building Hybrid LSTM signal-prefix sequence matrix.")
        sequence_matrix = build_hybrid_sequence_matrix(
            data,
            args.sensors,
            args.prefix_percents,
            args.max_signal_abs,
            args.hybrid_sequence_length,
        )
    cases = sorted(feature_matrix["case_id"].unique().tolist())
    prefix_percents = sorted(feature_matrix["signal_prefix_percent"].unique().tolist())
    summary: dict[str, Any] = {
        "experiment_id": config["experiment"]["experiment_id"],
        "execution_dir": str(output_dir),
        "sensors": args.sensors,
        "prefix_percents": prefix_percents,
        "prefix_start_idx": 0,
        "no_load_included": True,
        "selected_features": args.features,
        "feature_columns": input_features,
        "input_feature_count": int(len(input_features)),
        "raw_signal_sample_clip_abs": float(args.max_signal_abs),
        "feature_value_clip_abs": float(FEATURE_CLIP_ABS),
        "common_preprocessing": preprocessing_report,
        "feature_sanitization": sanitization_report,
        "models": models,
        "seeds": args.seeds,
        "hybrid_lstm_process": {
            "input_mode": args.hybrid_input_mode,
            "sequence_length": int(args.hybrid_sequence_length),
            "process_columns": args.hybrid_process_columns,
            "hidden_size": int(args.hybrid_hidden_size),
            "num_layers": int(args.hybrid_num_layers),
            "epochs": int(args.hybrid_epochs),
            "batch_size": int(args.hybrid_batch_size),
            "learning_rate": float(args.hybrid_learning_rate),
            "weight_decay": float(args.hybrid_weight_decay),
            "lstm_dropout": float(args.hybrid_lstm_dropout),
            "head_dropout": float(args.hybrid_head_dropout),
            "device": args.hybrid_device,
        },
        "stacking_ensemble": {
            "base_models": args.stack_base_models,
            "meta_learner": args.stack_meta_learner,
            "use_process_info": bool(args.stack_use_process_info),
            "process_columns": args.stack_process_columns if args.stack_use_process_info else [],
            "logistic_bins": int(args.stack_logistic_bins),
            "mlp_hidden_layers": [int(value) for value in args.stack_mlp_hidden_layers],
            "mlp_max_iter": int(args.stack_mlp_max_iter),
        },
        "case_scope": sorted(args.case_scope),
        "case_count": int(len(cases)),
        "case_run_count": int(feature_matrix[["case_id", "run"]].drop_duplicates().shape[0]),
        "feature_matrix_rows": int(len(feature_matrix)),
        "hybrid_sequence_rows": int(len(sequence_matrix)) if sequence_matrix is not None else 0,
        "planned_atomic_fits": int(sum(len(effective_seeds_for_h6_model(model, args.seeds)) * len(prefix_percents) * len(cases) for model in models)),
        "dry_run": bool(args.dry_run),
    }
    feature_matrix.to_csv(output_dir / "data" / "H4_S1_feature_matrix.csv", index=False)
    if sequence_matrix is not None:
        sequence_matrix.drop(columns=["sequence"]).to_csv(output_dir / "data" / "H4_S1_hybrid_sequence_index.csv", index=False)
    write_json(output_dir / "analysis" / "H4_S1_common_preprocessing.json", preprocessing_report)
    write_json(output_dir / "analysis" / "H4_S1_feature_sanitization.json", sanitization_report)
    if args.dry_run:
        write_progress_state(
            output_dir,
            status="dry_run_completed",
            total_tasks=summary["planned_atomic_fits"],
            completed_tasks=0,
            current_task=None,
        )
        write_json(output_dir / "analysis" / "H4_S1_summary.json", summary)
        log_progress(f"Dry-run completed: {summary['planned_atomic_fits']} atomic fits planned; no models trained.")
        return summary

    try:
        shift_metrics, predictions, splits = run_grid(
            feature_matrix,
            sequence_matrix,
            input_features,
            models,
            args.seeds,
            args.sensors,
            args,
            output_dir,
            resume_enabled,
        )
    except KeyboardInterrupt:
        mark_interrupted(output_dir)
        log_progress(f"Interrupted. Re-run the same command to resume from: {output_dir}")
        raise
    seed_metrics, model_metrics, best_by_model = aggregate_metrics(shift_metrics)
    case_metrics, case_model_metrics = build_casewise_metrics(shift_metrics)
    shift_metrics.to_csv(output_dir / "metrics" / "H4_S1_shift_metrics.csv", index=False)
    case_metrics.to_csv(output_dir / "metrics" / "H4_S1_case_metrics.csv", index=False)
    case_model_metrics.to_csv(output_dir / "metrics" / "H4_S1_case_model_metrics.csv", index=False)
    seed_metrics.to_csv(output_dir / "metrics" / "H4_S1_seed_metrics.csv", index=False)
    model_metrics.to_csv(output_dir / "metrics" / "H4_S1_prefix_model_metrics.csv", index=False)
    best_by_model.to_csv(output_dir / "analysis" / "H4_S1_best_prefix_by_model.csv", index=False)
    predictions.to_csv(output_dir / "predictions" / "H4_S1_predictions.csv", index=False)
    splits.to_csv(output_dir / "splits" / "H4_S1_splits.csv", index=False)
    fig_path = plot_prefix_effect(output_dir, model_metrics, args.dpi)

    summary.update(
        {
            "shift_metric_rows": int(len(shift_metrics)),
            "case_metric_rows": int(len(case_metrics)),
            "case_model_metric_rows": int(len(case_model_metrics)),
            "prediction_rows": int(len(predictions)),
            "best_overall": model_metrics.sort_values("mean_rmse").head(1).to_dict(orient="records"),
            "best_by_model": best_by_model.to_dict(orient="records"),
            "figure_path": str(fig_path.relative_to(output_dir)),
        }
    )
    write_json(output_dir / "analysis" / "H4_S1_summary.json", summary)
    write_report(output_dir, summary, model_metrics, best_by_model)
    return summary


def main() -> None:
    output_dir: Path | None = None
    try:
        summary = run(parse_args())
        output_dir = Path(summary["execution_dir"])
        print(json.dumps(to_builtin(summary), indent=2, ensure_ascii=False))
    except Exception:
        if output_dir is not None:
            (output_dir / "logs" / "H4_S1_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()

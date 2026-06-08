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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.experiments.h2_execution_utils import effective_seeds_for_model, model_seed_value, ordered_h2_models, seed_label
from milling_experiment_framework.models.h2_regressors import canonical_model_name, create_h2_feature_pipeline
from milling_experiment_framework.preprocessing.vb_common import COMMON_VB_PREPROCESSING_RULES, apply_common_vb_prediction_preprocessing
from milling_experiment_framework.visualization.figure_export import save_figure_dual


PREFIX = "H6_S3"
TOPIC = "segment_input_model_prediction_effect"
DEFAULT_CASE_SCOPE = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
DEFAULT_SENSORS = ["smcAC", "smcDC"]#, "vib_spindle", "vib_table", "AE_spindle", "AE_table"]
SEGMENT_SETTINGS = ["entry", "entry_steady", "steady"]
DEFAULT_MODELS = ["random_forest", "feature_gru"]#"random_forest", "svr", "xgboost"]
DEFAULT_SEEDS = [0, 1, 2]
FEATURE_NAMES = [
    "mean",
    "std",
    "max",
    "min",
    "peak_to_peak",
    "kurtosis",
    "skewness",
    "band_energy",
    "spectral_centroid",
]
DEFAULT_MAX_SIGNAL_ABS = 1_000_000.0
FEATURE_CLIP_ABS = 1.0e30
SIGNATURE_FILE = "H6_S3_resume_signature.json"
PROGRESS_FILE = "H6_S3_progress_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H6_S3 segment-input model prediction effect experiment.")
    parser.add_argument("--signal-path", default="datasets/processed/mill_signal_data.csv")
    parser.add_argument("--process-info-path", default="datasets/processed/mill_process_info.csv")
    parser.add_argument("--heuristic-sequence-path", default="datasets/metadata/heuristic_sequence_peng2026.csv")
    parser.add_argument("--case-scope", type=int, nargs="+", default=DEFAULT_CASE_SCOPE)
    parser.add_argument("--sensors", nargs="+", default=DEFAULT_SENSORS)
    parser.add_argument("--segments", nargs="+", default=SEGMENT_SETTINGS, choices=SEGMENT_SETTINGS)
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


def execution_dir(output_root: Path, timestamp: str) -> Path:
    return output_root / "H6" / "S3" / f"{timestamp}_{TOPIC}"


def timestamp_from_execution_dir(output_dir: Path) -> str:
    suffix = f"_{TOPIC}"
    return output_dir.name[: -len(suffix)] if output_dir.name.endswith(suffix) else datetime.now().strftime("%Y-%m-%d_%H%M%S")


def make_dirs(output_dir: Path) -> None:
    for rel in ["configs", "data", "splits", "metrics", "predictions", "analysis", "figures", "logs", "reports"]:
        (output_dir / rel).mkdir(parents=True, exist_ok=True)


def log_progress(message: str) -> None:
    print(f"[{PREFIX}] {message}", file=sys.stderr, flush=True)


def append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)


def parse_signal(value: Any) -> np.ndarray:
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def sanitize_signal(arr: np.ndarray, max_abs: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if max_abs and max_abs > 0:
        arr = np.clip(arr, -float(max_abs), float(max_abs))
    return arr


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
            "band_energy": float(np.sum(np.square(spectrum)) / max(len(arr), 1)),
            "spectral_centroid": float(np.sum(freqs * spectrum) / spectral_sum) if spectral_sum > 0 else 0.0,
        }
    return {key: (0.0 if not np.isfinite(value) else value) for key, value in raw.items()}


def feature_columns(sensors: list[str], feature_names: list[str] = FEATURE_NAMES) -> list[str]:
    return [f"{sensor}__{feature}" for sensor in sensors for feature in feature_names]


def segment_vector(arr: np.ndarray, row: pd.Series, segment_setting: str) -> np.ndarray:
    n = int(len(arr))
    idx_noload_end = int(np.clip(row["idx_noload_end"], 0, n))
    idx_start = int(np.clip(row["idx_start"], idx_noload_end, n))
    idx_end = int(np.clip(row["idx_end"], idx_start, n))
    segments = {
        "entry": arr[idx_noload_end:idx_start],
        "steady": arr[idx_start:idx_end],
    }
    if segment_setting == "entry_steady":
        vector = np.concatenate([segments["entry"], segments["steady"]])
    elif segment_setting in segments:
        vector = segments[segment_setting]
    else:
        raise ValueError(f"Unsupported segment setting: {segment_setting}")
    if len(vector) == 0:
        vector = np.asarray([0.0], dtype=np.float64)
    return vector


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


def build_config(args: argparse.Namespace, timestamp: str, output_dir: Path) -> dict[str, Any]:
    return {
        "experiment": {
            "experiment_number": PREFIX,
            "name": f"{PREFIX}_{TOPIC}",
            "experiment_id": f"{timestamp}_{PREFIX}_{TOPIC}",
            "timestamp": timestamp,
            "execution_dir": str(output_dir),
            "analysis_type": "segment_input_model_prediction_effect",
        },
        "data": {
            "signal_path": args.signal_path,
            "process_info_path": args.process_info_path,
            "heuristic_sequence_path": args.heuristic_sequence_path,
            "case_scope": args.case_scope,
            "target": "VB",
            "sensors": args.sensors,
        },
        "common_preprocessing": COMMON_VB_PREPROCESSING_RULES,
        "segments": {
            "settings": args.segments,
            "entry": "[idx_noload_end, idx_start)",
            "steady": "[idx_start, idx_end)",
            "entry_steady": "concat(entry, steady)",
            "reference": "H2_S1 segment feature construction",
        },
        "features": {
            "source": "H2_S1 all-sensor all-feature segment feature table",
            "feature_names": FEATURE_NAMES,
            "feature_naming": "{sensor}__{feature_name}",
            "raw_signal_sample_clip_abs": float(args.max_signal_abs),
            "feature_value_clip_abs": float(FEATURE_CLIP_ABS),
        },
        "modeling": {
            "task": "VB regression",
            "models": args.models,
            "seeds": args.seeds,
            "seed_policy": "linear_regression and svr run once with seed=-1; seeded models use up to three configured seeds",
        },
        "split": {
            "strategy": "leave_one_case_out",
            "test_unit": "case",
            "validation": "none",
        },
    }


def build_resume_signature(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "script": Path(__file__).name,
        "experiment": PREFIX,
        "topic": TOPIC,
        "signal_path": args.signal_path,
        "process_info_path": args.process_info_path,
        "heuristic_sequence_path": args.heuristic_sequence_path,
        "case_scope": sorted(int(case) for case in args.case_scope),
        "sensors": list(args.sensors),
        "segments": list(args.segments),
        "feature_names": FEATURE_NAMES,
        "models": list(args.models),
        "seeds": [int(seed) for seed in args.seeds],
        "max_signal_abs": float(args.max_signal_abs),
        "feature_clip_abs": float(FEATURE_CLIP_ABS),
        "common_preprocessing": COMMON_VB_PREPROCESSING_RULES,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"signature_version": 1, "signature_sha256": digest, "payload": payload}


def progress_path(output_dir: Path) -> Path:
    return output_dir / "logs" / PROGRESS_FILE


def write_progress_state(output_dir: Path, *, status: str, total_tasks: int, completed_tasks: int, current_task: dict[str, Any] | None = None) -> None:
    previous = read_json(progress_path(output_dir))
    now = datetime.now().isoformat(timespec="seconds")
    write_json(
        progress_path(output_dir),
        {
            **previous,
            "status": status,
            "total_tasks": int(total_tasks),
            "completed_tasks": int(completed_tasks),
            "percent_complete": float(completed_tasks / total_tasks * 100.0) if total_tasks else 100.0,
            "current_task": current_task,
            "started_at": previous.get("started_at", now),
            "updated_at": now,
        },
    )


def mark_interrupted(output_dir: Path) -> None:
    state = read_json(progress_path(output_dir))
    state.update({"status": "interrupted", "updated_at": datetime.now().isoformat(timespec="seconds")})
    write_json(progress_path(output_dir), state)


def task_id(segment_setting: str, model_name: str, seed: int, target_case: int) -> str:
    return f"segment={segment_setting}|model={model_name}|seed={int(seed)}|case={int(target_case)}"


def task_id_from_row(row: Any) -> str:
    return task_id(row.segment_setting, row.model, row.seed, row.target_case)


def completed_task_ids(metric_path: Path, resume: bool) -> set[str]:
    if not resume or not metric_path.exists() or metric_path.stat().st_size == 0:
        return set()
    existing = pd.read_csv(metric_path)
    required = {"segment_setting", "model", "seed", "target_case"}
    if not required.issubset(existing.columns):
        return set()
    return {task_id_from_row(row) for row in existing.itertuples(index=False)}


def find_resume_dir(output_root: Path, signature: dict[str, Any]) -> Path | None:
    root = output_root / "H6" / "S3"
    if not root.exists():
        return None
    candidates = sorted(root.glob(f"*_{TOPIC}"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if read_json(candidate / "logs" / SIGNATURE_FILE) != signature:
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
        return output_dir, args.timestamp, output_dir.exists()
    if args.resume:
        resume_dir = find_resume_dir(output_root, signature)
        if resume_dir is not None:
            return resume_dir, timestamp_from_execution_dir(resume_dir), True
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return execution_dir(output_root, timestamp), timestamp, False


def load_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    signal = pd.read_csv(resolve(args.signal_path), usecols=["case", "run", *args.sensors])
    process = pd.read_csv(resolve(args.process_info_path))
    heuristic = pd.read_csv(resolve(args.heuristic_sequence_path))
    required_process = {"case", "run", "VB"}
    required_heuristic = {"case", "run", "idx_noload_end", "idx_start", "idx_end", "signal_length"}
    missing_process = required_process.difference(process.columns)
    missing_signal = set(args.sensors).difference(signal.columns)
    missing_heuristic = required_heuristic.difference(heuristic.columns)
    if missing_process:
        raise ValueError(f"process info missing columns: {sorted(missing_process)}")
    if missing_signal:
        raise ValueError(f"signal data missing sensor columns: {sorted(missing_signal)}")
    if missing_heuristic:
        raise ValueError(f"heuristic sequence missing columns: {sorted(missing_heuristic)}")
    data = signal.merge(process, on=["case", "run"], how="inner", validate="one_to_one")
    data = data.merge(
        heuristic[["case", "run", "idx_noload_end", "idx_start", "idx_end", "signal_length"]],
        on=["case", "run"],
        how="inner",
        validate="one_to_one",
    )
    data = data.loc[data["case"].isin(args.case_scope)].copy()
    data, preprocessing_report = apply_common_vb_prediction_preprocessing(data)
    if data.empty:
        raise ValueError("No rows after common preprocessing.")
    return data.reset_index(drop=True), preprocessing_report


def build_feature_matrix(data: pd.DataFrame, sensors: list[str], segments: list[str], max_signal_abs: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_row in data.itertuples(index=False):
        row = pd.Series(source_row._asdict())
        case = int(row["case"])
        run_id = int(row["run"])
        arrays = {sensor: sanitize_signal(parse_signal(row[sensor]), max_signal_abs) for sensor in sensors}
        for segment_setting in segments:
            out: dict[str, Any] = {
                "sample_id": f"case_{case:02d}_run_{run_id:03d}_segment_{segment_setting}",
                "dataset_run_id": f"case_{case:02d}_run_{run_id:03d}",
                "case_id": case,
                "run": run_id,
                "segment_setting": segment_setting,
                "idx_noload_end": int(row["idx_noload_end"]),
                "idx_start": int(row["idx_start"]),
                "idx_end": int(row["idx_end"]),
                "signal_length": int(row["signal_length"]),
                "VB": float(row["VB"]),
                "time": row.get("time", np.nan),
                "DOC": row.get("DOC", np.nan),
                "feed": row.get("feed", np.nan),
                "material_name": row.get("material_name", np.nan),
            }
            for sensor, arr in arrays.items():
                vector = segment_vector(arr, row, segment_setting)
                values = compute_features(vector)
                for feature_name in FEATURE_NAMES:
                    out[f"{sensor}__{feature_name}"] = values[feature_name]
            rows.append(out)
    feature_matrix = pd.DataFrame(rows)
    missing = [col for col in feature_columns(sensors) if col not in feature_matrix.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return feature_matrix.sort_values(["segment_setting", "case_id", "run"]).reset_index(drop=True)


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
    split = data[["sample_id", "dataset_run_id", "case_id", "run", "segment_setting", "VB"]].copy()
    split["split"] = np.where(split["case_id"].eq(target_case), "test", "train")
    return split


def run_grid(
    feature_matrix: pd.DataFrame,
    input_features: list[str],
    segments: list[str],
    models: list[str],
    seeds: list[int],
    output_dir: Path,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shift_path = output_dir / "metrics" / "H6_S3_shift_metrics.csv"
    prediction_path = output_dir / "predictions" / "H6_S3_predictions.csv"
    split_path = output_dir / "splits" / "H6_S3_splits.csv"
    cases = sorted(feature_matrix["case_id"].unique().tolist())
    total_tasks = int(len(segments) * len(cases) * sum(len(effective_seeds_for_model(model, seeds)) for model in models))
    done = completed_task_ids(shift_path, resume)
    completed = len(done)
    write_progress_state(output_dir, status="running", total_tasks=total_tasks, completed_tasks=completed)
    log_progress(f"{'Resuming' if completed else 'Training grid started'}: {completed}/{total_tasks} atomic fits completed.")
    for segment_setting in segments:
        segment_df = feature_matrix.loc[feature_matrix["segment_setting"].eq(segment_setting)].copy()
        for model_name in models:
            for seed in effective_seeds_for_model(model_name, seeds):
                for target_case in cases:
                    current_id = task_id(segment_setting, model_name, seed, target_case)
                    current_task = {
                        "task_id": current_id,
                        "segment_setting": segment_setting,
                        "model": model_name,
                        "seed": int(seed),
                        "seed_label": seed_label(seed),
                        "target_case": int(target_case),
                    }
                    if current_id in done:
                        continue
                    write_progress_state(output_dir, status="running", total_tasks=total_tasks, completed_tasks=completed, current_task=current_task)
                    split = split_for_case(segment_df, target_case)
                    train = segment_df.loc[segment_df["case_id"].ne(target_case)].copy()
                    test = segment_df.loc[segment_df["case_id"].eq(target_case)].copy()
                    if train.empty or test.empty:
                        raise ValueError(f"Empty split for segment={segment_setting}, model={model_name}, target_case={target_case}")
                    estimator = create_h2_feature_pipeline(model_name, model_seed_value(seed), model_defaults=None)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        estimator.fit(train[input_features], train["VB"])
                    y_pred = estimator.predict(test[input_features])
                    metrics = metric_dict(test["VB"].to_numpy(), y_pred)
                    scenario = f"train_without_case_{target_case}_to_case_{target_case}"
                    shift_row = {
                        "segment_setting": segment_setting,
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
                        "task_id": current_id,
                    }
                    preds = test[["sample_id", "dataset_run_id", "case_id", "run", "segment_setting", "VB"]].copy()
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
                    split["model"] = model_name
                    split["seed"] = int(seed)
                    split["seed_label"] = seed_label(seed)
                    split["target_case"] = int(target_case)
                    split["scenario"] = scenario
                    split["task_id"] = current_id
                    append_csv(preds, prediction_path)
                    append_csv(split, split_path)
                    append_csv(pd.DataFrame([shift_row]), shift_path)
                    done.add(current_id)
                    completed += 1
                    write_progress_state(output_dir, status="running", total_tasks=total_tasks, completed_tasks=completed, current_task=current_task)
                    log_progress(
                        f"Progress {completed}/{total_tasks} ({completed / total_tasks * 100:.1f}%) "
                        f"segment={segment_setting} model={model_name} seed={seed_label(seed)} case={target_case}"
                    )
    write_progress_state(output_dir, status="completed", total_tasks=total_tasks, completed_tasks=completed, current_task=None)
    return (
        pd.read_csv(shift_path).drop_duplicates(["task_id"], keep="last"),
        pd.read_csv(prediction_path).drop_duplicates(["task_id", "sample_id"], keep="last"),
        pd.read_csv(split_path).drop_duplicates(["task_id", "sample_id"], keep="last"),
    )


def aggregate_metrics(shift_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_metrics = (
        shift_metrics.groupby(["segment_setting", "model", "seed", "seed_label"], as_index=False)
        .agg(
            mean_mae_over_cases=("metric_mae", "mean"),
            mean_rmse_over_cases=("metric_rmse", "mean"),
            mean_nmae_case_range_over_cases=("metric_nmae_case_range", "mean"),
            mean_nrmse_case_range_over_cases=("metric_nrmse_case_range", "mean"),
            mean_r2_over_cases=("metric_r2", "mean"),
            mean_pearson_over_cases=("metric_pearson", "mean"),
            std_rmse_over_cases=("metric_rmse", "std"),
            test_case_count=("target_case", "nunique"),
            total_test_rows=("test_rows", "sum"),
        )
    )
    seed_metrics["std_rmse_over_cases"] = seed_metrics["std_rmse_over_cases"].fillna(0.0)
    segment_metrics = (
        seed_metrics.groupby(["segment_setting", "model"], as_index=False)
        .agg(
            mean_mae=("mean_mae_over_cases", "mean"),
            std_mae=("mean_mae_over_cases", "std"),
            mean_rmse=("mean_rmse_over_cases", "mean"),
            std_rmse=("mean_rmse_over_cases", "std"),
            mean_nmae_case_range=("mean_nmae_case_range_over_cases", "mean"),
            mean_nrmse_case_range=("mean_nrmse_case_range_over_cases", "mean"),
            mean_r2=("mean_r2_over_cases", "mean"),
            std_r2=("mean_r2_over_cases", "std"),
            mean_pearson=("mean_pearson_over_cases", "mean"),
            seed_count=("seed", "nunique"),
            test_case_count=("test_case_count", "max"),
            total_test_rows=("total_test_rows", "max"),
        )
    )
    for col in ["std_mae", "std_rmse", "std_r2"]:
        segment_metrics[col] = segment_metrics[col].fillna(0.0)
    segment_metrics["rank_by_rmse"] = segment_metrics["mean_rmse"].rank(ascending=True, method="first").astype(int)
    best_segment = segment_metrics.sort_values("mean_rmse").head(1).reset_index(drop=True)
    case_metrics, case_segment_metrics = build_casewise_metrics(shift_metrics)
    return seed_metrics, segment_metrics.sort_values("rank_by_rmse").reset_index(drop=True), best_segment, case_metrics, case_segment_metrics


def build_casewise_metrics(shift_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_columns = [
        "segment_setting",
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
    case_segment_metrics = (
        case_metrics.groupby(["segment_setting", "model", "target_case"], as_index=False)
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
            mean_r2=("metric_r2", "mean"),
            mean_pearson=("metric_pearson", "mean"),
            seed_count=("seed", "nunique"),
            test_rows=("test_rows", "max"),
        )
    )
    for col in ["std_mae", "std_rmse"]:
        case_segment_metrics[col] = case_segment_metrics[col].fillna(0.0)
    return case_metrics.sort_values(["segment_setting", "target_case", "seed"]).reset_index(drop=True), case_segment_metrics.sort_values(["segment_setting", "target_case"]).reset_index(drop=True)


def plot_segment_effect(output_dir: Path, segment_metrics: pd.DataFrame, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    segments = [segment for segment in SEGMENT_SETTINGS if segment in set(segment_metrics["segment_setting"])]
    models = sorted(segment_metrics["model"].unique().tolist())
    x = np.arange(len(segments))
    width = min(0.72 / max(len(models), 1), 0.28)
    for i, model_name in enumerate(models):
        group = segment_metrics.loc[segment_metrics["model"].eq(model_name)].set_index("segment_setting").reindex(segments)
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, group["mean_rmse"], yerr=group["std_rmse"], width=width, alpha=0.82, capsize=3, label=model_name)
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=20, ha="right")
    ax.set_xlabel("Segment input")
    ax.set_ylabel("mean RMSE over leave-one-case-out folds")
    ax.set_title(f"{PREFIX} model error by segment input")
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "figures" / "H6_S3_segment_model_effect_rmse.png"
    save_figure_dual(fig, path, dpi=dpi)
    plt.close(fig)
    return path


def write_report(output_dir: Path, summary: dict[str, Any], segment_metrics: pd.DataFrame, best_segment: pd.DataFrame) -> None:
    lines = [
        f"# {PREFIX} Segment Input Model Prediction Effect",
        "",
        "## Scope",
        "",
        f"- Sensors: `{summary['sensors']}`",
        f"- Features: `{summary['feature_names']}`",
        f"- Segment inputs: `{summary['segments']}`",
        f"- Split: leave-one-case-out over `{summary['case_count']}` cases",
        f"- Seeds: `{summary['seeds']}`",
        f"- Models: `{summary['models']}`",
        "",
        "## Outputs",
        "",
        "- Feature matrix: `data/H6_S3_feature_matrix.csv`",
        "- Common preprocessing: `analysis/H6_S3_common_preprocessing.json`",
        "- Feature sanitization: `analysis/H6_S3_feature_sanitization.json`",
        "- Shift metrics: `metrics/H6_S3_shift_metrics.csv`",
        "- Case-wise metrics: `metrics/H6_S3_case_metrics.csv`",
        "- Segment metrics: `metrics/H6_S3_segment_metrics.csv`",
        "- Predictions: `predictions/H6_S3_predictions.csv`",
        "- Progress state: `logs/H6_S3_progress_state.json`",
        "- Segment effect figure: `figures/H6_S3_segment_model_effect_rmse.{png,svg}`",
        "",
        "## Segment Ranking",
        "",
        "| rank | model | segment | mean_rmse | mean_mae | mean_r2 | mean_nrmse_case_range |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in segment_metrics.itertuples(index=False):
        lines.append(f"| {row.rank_by_rmse} | {row.model} | {row.segment_setting} | {row.mean_rmse:.6f} | {row.mean_mae:.6f} | {row.mean_r2:.6f} | {row.mean_nrmse_case_range:.6f} |")
    if not best_segment.empty:
        row = best_segment.iloc[0]
        lines.extend(["", "## Best Model-Segment", "", f"`{row['model']} / {row['segment_setting']}` has the lowest mean RMSE: `{row['mean_rmse']:.6f}`."])
    (output_dir / "reports" / "H6_S3_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if float(args.max_signal_abs) < 0:
        raise ValueError("--max-signal-abs must be >= 0; use 0 to disable clipping.")
    args.segments = [segment for segment in SEGMENT_SETTINGS if segment in set(args.segments)]
    args.models = ordered_h2_models([canonical_model_name(model) for model in args.models])
    args.seeds = [int(seed) for seed in args.seeds[:3]]
    signature = build_resume_signature(args)
    output_root = resolve(args.output_root)
    resume_enabled = bool(args.resume and not args.dry_run)
    output_dir, timestamp, resumed = prepare_execution_dir(args, output_root, signature) if resume_enabled else prepare_execution_dir(argparse.Namespace(**{**vars(args), "resume": False}), output_root, signature)
    if output_dir.exists():
        existing_signature = read_json(output_dir / "logs" / SIGNATURE_FILE)
        if existing_signature and existing_signature != signature:
            raise ValueError(f"Execution directory has a different resume signature: {output_dir}")
        if not resumed and not args.dry_run:
            raise FileExistsError(f"Execution directory already exists: {output_dir}")
    make_dirs(output_dir)
    write_json(output_dir / "logs" / SIGNATURE_FILE, signature)
    log_progress(f"Using {'existing' if resumed else 'new'} execution directory: {output_dir}")

    config = build_config(args, timestamp, output_dir)
    write_yaml(output_dir / "configs" / "H6_S3_input_config.yaml", config)
    write_json(output_dir / "logs" / "H6_S3_environment.json", collect_environment())

    log_progress("Loading input data and applying common VB preprocessing.")
    data, preprocessing_report = load_dataset(args)
    input_features = feature_columns(args.sensors)
    log_progress("Building H2_S1-style segment feature matrix.")
    feature_matrix = build_feature_matrix(data, args.sensors, args.segments, args.max_signal_abs)
    feature_matrix, sanitization_report = sanitize_feature_matrix(feature_matrix, input_features)
    cases = sorted(feature_matrix["case_id"].unique().tolist())
    summary: dict[str, Any] = {
        "experiment_id": config["experiment"]["experiment_id"],
        "execution_dir": str(output_dir),
        "sensors": args.sensors,
        "segments": args.segments,
        "feature_names": FEATURE_NAMES,
        "feature_columns": input_features,
        "input_feature_count": int(len(input_features)),
        "raw_signal_sample_clip_abs": float(args.max_signal_abs),
        "feature_value_clip_abs": float(FEATURE_CLIP_ABS),
        "common_preprocessing": preprocessing_report,
        "feature_sanitization": sanitization_report,
        "models": args.models,
        "seeds": args.seeds,
        "case_scope": sorted(args.case_scope),
        "case_count": int(len(cases)),
        "case_run_count": int(feature_matrix[["case_id", "run"]].drop_duplicates().shape[0]),
        "feature_matrix_rows": int(len(feature_matrix)),
        "planned_atomic_fits": int(len(args.segments) * len(cases) * sum(len(effective_seeds_for_model(model, args.seeds)) for model in args.models)),
        "dry_run": bool(args.dry_run),
    }
    feature_matrix.to_csv(output_dir / "data" / "H6_S3_feature_matrix.csv", index=False)
    write_json(output_dir / "analysis" / "H6_S3_common_preprocessing.json", preprocessing_report)
    write_json(output_dir / "analysis" / "H6_S3_feature_sanitization.json", sanitization_report)
    if args.dry_run:
        write_progress_state(output_dir, status="dry_run_completed", total_tasks=summary["planned_atomic_fits"], completed_tasks=0, current_task=None)
        write_json(output_dir / "analysis" / "H6_S3_summary.json", summary)
        log_progress(f"Dry-run completed: {summary['planned_atomic_fits']} atomic fits planned; no models trained.")
        return summary

    try:
        shift_metrics, predictions, splits = run_grid(feature_matrix, input_features, args.segments, args.models, args.seeds, output_dir, resume_enabled)
    except KeyboardInterrupt:
        mark_interrupted(output_dir)
        log_progress(f"Interrupted. Re-run the same command to resume from: {output_dir}")
        raise
    seed_metrics, segment_metrics, best_segment, case_metrics, case_segment_metrics = aggregate_metrics(shift_metrics)
    shift_metrics.to_csv(output_dir / "metrics" / "H6_S3_shift_metrics.csv", index=False)
    case_metrics.to_csv(output_dir / "metrics" / "H6_S3_case_metrics.csv", index=False)
    case_segment_metrics.to_csv(output_dir / "metrics" / "H6_S3_case_segment_metrics.csv", index=False)
    seed_metrics.to_csv(output_dir / "metrics" / "H6_S3_seed_metrics.csv", index=False)
    segment_metrics.to_csv(output_dir / "metrics" / "H6_S3_segment_metrics.csv", index=False)
    best_segment.to_csv(output_dir / "analysis" / "H6_S3_best_segment.csv", index=False)
    predictions.to_csv(output_dir / "predictions" / "H6_S3_predictions.csv", index=False)
    splits.to_csv(output_dir / "splits" / "H6_S3_splits.csv", index=False)
    fig_path = plot_segment_effect(output_dir, segment_metrics, args.dpi)
    summary.update(
        {
            "shift_metric_rows": int(len(shift_metrics)),
            "case_metric_rows": int(len(case_metrics)),
            "prediction_rows": int(len(predictions)),
            "best_segment": best_segment.to_dict(orient="records"),
            "figure_path": str(fig_path.relative_to(output_dir)),
        }
    )
    write_json(output_dir / "analysis" / "H6_S3_summary.json", summary)
    write_report(output_dir, summary, segment_metrics, best_segment)
    return summary


def main() -> None:
    output_dir: Path | None = None
    try:
        summary = run(parse_args())
        output_dir = Path(summary["execution_dir"])
        print(json.dumps(to_builtin(summary), indent=2, ensure_ascii=False))
    except Exception:
        if output_dir is not None:
            (output_dir / "logs" / "H6_S3_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()

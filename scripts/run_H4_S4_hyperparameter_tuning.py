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
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneGroupOut

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
from milling_experiment_framework.visualization.figure_export import save_figure_dual
from scripts import run_H4_S3_feature_gru_segment_input_prediction_effect as h4s3


PREFIX = "H4_S4"
TOPIC = "segment_input_hyperparameter_tuning"
DEFAULT_MODELS = ["random_forest"]
SUPPORTED_MODELS = ["random_forest", "svr", "xgboost", "feature_gru"]
SIGNATURE_FILE = "H4_S4_resume_signature.json"
PROGRESS_FILE = "H4_S4_progress_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H4_S4 segment-input hyperparameter tuning experiment.")
    parser.add_argument("--signal-path", default="datasets/processed/mill_signal_data.csv")
    parser.add_argument("--process-info-path", default="datasets/processed/mill_process_info.csv")
    parser.add_argument("--heuristic-sequence-path", default="datasets/metadata/heuristic_sequence_peng2026.csv")
    parser.add_argument("--case-scope", type=int, nargs="+", default=h4s3.DEFAULT_CASE_SCOPE)
    parser.add_argument("--sensors", nargs="+", default=h4s3.DEFAULT_SENSORS)
    parser.add_argument("--segments", nargs="+", default=h4s3.SEGMENT_SETTINGS, choices=h4s3.SEGMENT_SETTINGS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=h4s3.DEFAULT_SEEDS)
    parser.add_argument("--grid-profile", choices=["smoke", "small", "full"], default="small")
    parser.add_argument("--output-root", default="experiments/executions")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--max-signal-abs",
        type=float,
        default=h4s3.DEFAULT_MAX_SIGNAL_ABS,
        help="Clip raw signal samples to +/- this value before feature extraction; use 0 to disable.",
    )
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    return h4s3.to_builtin(value)


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
    return output_root / "H4" / "S4" / f"{timestamp}_{TOPIC}"


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


def param_grids(profile: str) -> dict[str, list[dict[str, Any]]]:
    grids = {
        "smoke": {
            "random_forest": [{"model__n_estimators": [20], "model__max_depth": [None], "model__min_samples_leaf": [1]}],
            "svr": [{"model__C": [1.0], "model__epsilon": [0.03], "model__gamma": ["scale"]}],
            "xgboost": [{"model__n_estimators": [30], "model__max_depth": [2], "model__learning_rate": [0.05]}],
            "feature_gru": [{"model__epochs": [5], "model__gru_hidden_size": [16], "model__learning_rate": [0.001]}],
        },
        "small": {
            "random_forest": [
                {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [None, 12],
                    "model__min_samples_leaf": [1, 3],
                    "model__max_features": ["sqrt"],
                }
            ],
            "svr": [
                {
                    "model__C": [1.0, 10.0, 50.0],
                    "model__epsilon": [0.01, 0.03],
                    "model__gamma": ["scale", "auto"],
                }
            ],
            "xgboost": [
                {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [2, 3],
                    "model__learning_rate": [0.03, 0.05],
                    "model__subsample": [0.9],
                    "model__colsample_bytree": [0.9],
                }
            ],
            "feature_gru": [
                {
                    "model__epochs": [60],
                    "model__gru_hidden_size": [16, 32],
                    "model__regression_head_hidden_dim": [16, 32],
                    "model__learning_rate": [0.001],
                    "model__batch_size": [16],
                }
            ],
        },
        "full": {
            "random_forest": [
                {
                    "model__n_estimators": [100, 200, 400],
                    "model__max_depth": [None, 8, 16],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__max_features": ["sqrt", None],
                }
            ],
            "svr": [
                {
                    "model__C": [1.0, 10.0, 50.0, 100.0],
                    "model__epsilon": [0.01, 0.03, 0.05],
                    "model__gamma": ["scale", "auto"],
                }
            ],
            "xgboost": [
                {
                    "model__n_estimators": [100, 200, 400],
                    "model__max_depth": [2, 3, 4],
                    "model__learning_rate": [0.03, 0.05, 0.08],
                    "model__subsample": [0.8, 0.95],
                    "model__colsample_bytree": [0.8, 0.95],
                }
            ],
            "feature_gru": [
                {
                    "model__epochs": [80, 120],
                    "model__gru_hidden_size": [16, 32, 64],
                    "model__regression_head_hidden_dim": [16, 32],
                    "model__learning_rate": [0.0005, 0.001],
                    "model__batch_size": [8, 16],
                }
            ],
        },
    }
    return grids[profile]


def grid_size(grid: list[dict[str, Any]]) -> int:
    size = 0
    for item in grid:
        combos = 1
        for values in item.values():
            combos *= len(values)
        size += combos
    return int(size)


def build_config(args: argparse.Namespace, timestamp: str, output_dir: Path, grids: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "experiment": {
            "experiment_number": PREFIX,
            "name": f"{PREFIX}_{TOPIC}",
            "experiment_id": f"{timestamp}_{PREFIX}_{TOPIC}",
            "timestamp": timestamp,
            "execution_dir": str(output_dir),
            "analysis_type": "hyperparameter_tuning",
        },
        "data": {
            "signal_path": args.signal_path,
            "process_info_path": args.process_info_path,
            "heuristic_sequence_path": args.heuristic_sequence_path,
            "case_scope": args.case_scope,
            "target": "VB",
            "sensors": args.sensors,
        },
        "segments": {"settings": args.segments, "reference": "H4_S3/H2_S1 segment feature construction"},
        "features": {
            "feature_names": h4s3.FEATURE_NAMES,
            "feature_naming": "{sensor}__{feature_name}",
            "raw_signal_sample_clip_abs": float(args.max_signal_abs),
            "feature_value_clip_abs": float(h4s3.FEATURE_CLIP_ABS),
        },
        "modeling": {
            "task": "VB regression",
            "models": args.models,
            "seeds": args.seeds,
            "grid_profile": args.grid_profile,
            "param_grids": grids,
            "grid_search": {
                "estimator": "StandardScaler -> model",
                "scoring": "neg_root_mean_squared_error",
                "inner_cv": "LeaveOneGroupOut by case on the training cases",
                "refit": True,
            },
        },
        "split": {"strategy": "outer_leave_one_case_out", "test_unit": "case", "validation": "inner_grid_search_cv"},
    }


def build_resume_signature(args: argparse.Namespace, grids: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
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
        "feature_names": h4s3.FEATURE_NAMES,
        "models": list(args.models),
        "seeds": [int(seed) for seed in args.seeds],
        "grid_profile": args.grid_profile,
        "param_grids": grids,
        "max_signal_abs": float(args.max_signal_abs),
        "feature_clip_abs": float(h4s3.FEATURE_CLIP_ABS),
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
    root = output_root / "H4" / "S4"
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


def inner_cv(train: pd.DataFrame):
    groups = train["case_id"].to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        return LeaveOneGroupOut(), groups
    splits = min(3, len(train))
    if splits < 2:
        raise ValueError("GridSearchCV requires at least two training rows.")
    return KFold(n_splits=splits, shuffle=True, random_state=0), None


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return h4s3.metric_dict(y_true, y_pred)


def split_for_case(data: pd.DataFrame, target_case: int) -> pd.DataFrame:
    return h4s3.split_for_case(data, target_case)


def run_grid(
    feature_matrix: pd.DataFrame,
    input_features: list[str],
    segments: list[str],
    models: list[str],
    seeds: list[int],
    grids: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shift_path = output_dir / "metrics" / "H4_S4_shift_metrics.csv"
    prediction_path = output_dir / "predictions" / "H4_S4_predictions.csv"
    split_path = output_dir / "splits" / "H4_S4_splits.csv"
    tuning_path = output_dir / "analysis" / "H4_S4_tuning_results.csv"
    cases = sorted(feature_matrix["case_id"].unique().tolist())
    total_tasks = int(len(segments) * len(cases) * sum(len(effective_seeds_for_model(model, seeds)) for model in models))
    done = completed_task_ids(shift_path, resume)
    completed = len(done)
    write_progress_state(output_dir, status="running", total_tasks=total_tasks, completed_tasks=completed)
    log_progress(f"{'Resuming' if completed else 'Grid search started'}: {completed}/{total_tasks} atomic tuning runs completed.")

    for segment_setting in segments:
        segment_df = feature_matrix.loc[feature_matrix["segment_setting"].eq(segment_setting)].copy()
        for model_name in models:
            grid = grids[model_name]
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
                        "grid_size": grid_size(grid),
                    }
                    if current_id in done:
                        continue
                    write_progress_state(output_dir, status="running", total_tasks=total_tasks, completed_tasks=completed, current_task=current_task)
                    train = segment_df.loc[segment_df["case_id"].ne(target_case)].copy()
                    test = segment_df.loc[segment_df["case_id"].eq(target_case)].copy()
                    if train.empty or test.empty:
                        raise ValueError(f"Empty split for segment={segment_setting}, model={model_name}, target_case={target_case}")
                    cv, groups = inner_cv(train)
                    estimator = create_h2_feature_pipeline(model_name, model_seed_value(seed), model_defaults=None)
                    search = GridSearchCV(
                        estimator=estimator,
                        param_grid=grid,
                        scoring="neg_root_mean_squared_error",
                        cv=cv,
                        refit=True,
                        n_jobs=1,
                        error_score="raise",
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if groups is None:
                            search.fit(train[input_features], train["VB"])
                        else:
                            search.fit(train[input_features], train["VB"], groups=groups)
                    y_pred = search.predict(test[input_features])
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
                        "best_cv_score_neg_rmse": float(search.best_score_),
                        "best_cv_rmse": float(-search.best_score_),
                        "best_params_json": json.dumps(to_builtin(search.best_params_), sort_keys=True),
                        "grid_size": int(grid_size(grid)),
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
                    split = split_for_case(segment_df, target_case)
                    split["model"] = model_name
                    split["seed"] = int(seed)
                    split["seed_label"] = seed_label(seed)
                    split["target_case"] = int(target_case)
                    split["scenario"] = scenario
                    split["task_id"] = current_id
                    tuning_rows = pd.DataFrame(search.cv_results_).copy()
                    tuning_rows["segment_setting"] = segment_setting
                    tuning_rows["model"] = model_name
                    tuning_rows["seed"] = int(seed)
                    tuning_rows["seed_label"] = seed_label(seed)
                    tuning_rows["target_case"] = int(target_case)
                    tuning_rows["scenario"] = scenario
                    tuning_rows["task_id"] = current_id
                    append_csv(preds, prediction_path)
                    append_csv(split, split_path)
                    append_csv(tuning_rows, tuning_path)
                    append_csv(pd.DataFrame([shift_row]), shift_path)
                    done.add(current_id)
                    completed += 1
                    write_progress_state(output_dir, status="running", total_tasks=total_tasks, completed_tasks=completed, current_task=current_task)
                    log_progress(
                        f"Progress {completed}/{total_tasks} ({completed / total_tasks * 100:.1f}%) "
                        f"segment={segment_setting} model={model_name} seed={seed_label(seed)} case={target_case} "
                        f"best_cv_rmse={-search.best_score_:.6f}"
                    )
    write_progress_state(output_dir, status="completed", total_tasks=total_tasks, completed_tasks=completed, current_task=None)
    return (
        pd.read_csv(shift_path).drop_duplicates(["task_id"], keep="last"),
        pd.read_csv(prediction_path).drop_duplicates(["task_id", "sample_id"], keep="last"),
        pd.read_csv(split_path).drop_duplicates(["task_id", "sample_id"], keep="last"),
        pd.read_csv(tuning_path).drop_duplicates(["task_id", "params"], keep="last"),
    )


def aggregate_metrics(shift_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_metrics, segment_metrics, best_segment, case_metrics, case_segment_metrics = h4s3.aggregate_metrics(shift_metrics)
    best_by_model = segment_metrics.sort_values(["model", "mean_rmse"]).groupby("model", as_index=False).head(1).reset_index(drop=True)
    return seed_metrics, segment_metrics, best_segment, best_by_model, case_metrics, case_segment_metrics


def plot_tuned_segment_effect(output_dir: Path, segment_metrics: pd.DataFrame, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    segments = [segment for segment in h4s3.SEGMENT_SETTINGS if segment in set(segment_metrics["segment_setting"])]
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
    ax.set_ylabel("mean test RMSE after inner GridSearchCV")
    ax.set_title(f"{PREFIX} tuned model error by segment input")
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "figures" / "H4_S4_tuned_segment_model_effect_rmse.png"
    save_figure_dual(fig, path, dpi=dpi)
    plt.close(fig)
    return path


def write_report(output_dir: Path, summary: dict[str, Any], segment_metrics: pd.DataFrame, best_by_model: pd.DataFrame) -> None:
    lines = [
        f"# {PREFIX} Hyperparameter Tuning",
        "",
        "## Scope",
        "",
        f"- Models: `{summary['models']}`",
        f"- Grid profile: `{summary['grid_profile']}`",
        f"- Segment inputs: `{summary['segments']}`",
        f"- Sensors: `{summary['sensors']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Split: outer leave-one-case-out over `{summary['case_count']}` cases; inner GridSearchCV by training case.",
        "",
        "## Outputs",
        "",
        "- Feature matrix: `data/H4_S4_feature_matrix.csv`",
        "- Parameter grids: `configs/H4_S4_param_grids.json`",
        "- Tuning CV results: `analysis/H4_S4_tuning_results.csv`",
        "- Shift metrics: `metrics/H4_S4_shift_metrics.csv`",
        "- Segment metrics: `metrics/H4_S4_segment_metrics.csv`",
        "- Case-wise metrics: `metrics/H4_S4_case_metrics.csv`",
        "- Predictions: `predictions/H4_S4_predictions.csv`",
        "- Progress state: `logs/H4_S4_progress_state.json`",
        "- Figure: `figures/H4_S4_tuned_segment_model_effect_rmse.{png,svg}`",
        "",
        "## Best Segment By Model",
        "",
        "| model | segment | mean_rmse | mean_mae | mean_r2 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in best_by_model.itertuples(index=False):
        lines.append(f"| {row.model} | {row.segment_setting} | {row.mean_rmse:.6f} | {row.mean_mae:.6f} | {row.mean_r2:.6f} |")
    lines.extend(["", "## Overall Ranking", "", "| rank | model | segment | mean_rmse | mean_nrmse_case_range |", "|---:|---|---|---:|---:|"])
    for row in segment_metrics.sort_values("rank_by_rmse").itertuples(index=False):
        lines.append(f"| {row.rank_by_rmse} | {row.model} | {row.segment_setting} | {row.mean_rmse:.6f} | {row.mean_nrmse_case_range:.6f} |")
    (output_dir / "reports" / "H4_S4_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if float(args.max_signal_abs) < 0:
        raise ValueError("--max-signal-abs must be >= 0; use 0 to disable clipping.")
    args.segments = [segment for segment in h4s3.SEGMENT_SETTINGS if segment in set(args.segments)]
    args.models = ordered_h2_models([canonical_model_name(model) for model in args.models])
    unsupported = [model for model in args.models if model not in SUPPORTED_MODELS]
    if unsupported:
        raise ValueError(f"H4_S4 supports {SUPPORTED_MODELS}, got unsupported models: {unsupported}")
    args.seeds = [int(seed) for seed in args.seeds[:3]]
    grids_all = param_grids(args.grid_profile)
    grids = {model: grids_all[model] for model in args.models}
    signature = build_resume_signature(args, grids)
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

    config = build_config(args, timestamp, output_dir, grids)
    write_yaml(output_dir / "configs" / "H4_S4_input_config.yaml", config)
    write_json(output_dir / "configs" / "H4_S4_param_grids.json", grids)
    write_json(output_dir / "logs" / "H4_S4_environment.json", collect_environment())

    log_progress("Loading input data and applying common VB preprocessing.")
    data, preprocessing_report = h4s3.load_dataset(args)
    input_features = h4s3.feature_columns(args.sensors)
    log_progress("Building H4_S3/H2_S1-style segment feature matrix.")
    feature_matrix = h4s3.build_feature_matrix(data, args.sensors, args.segments, args.max_signal_abs)
    feature_matrix, sanitization_report = h4s3.sanitize_feature_matrix(feature_matrix, input_features)
    cases = sorted(feature_matrix["case_id"].unique().tolist())
    planned_atomic = int(len(args.segments) * len(cases) * sum(len(effective_seeds_for_model(model, args.seeds)) for model in args.models))
    planned_grid_fits = int(
        sum(
            len(args.segments)
            * len(cases)
            * len(effective_seeds_for_model(model, args.seeds))
            * grid_size(grids[model])
            * max(len(cases) - 1, 1)
            for model in args.models
        )
    )
    summary: dict[str, Any] = {
        "experiment_id": config["experiment"]["experiment_id"],
        "execution_dir": str(output_dir),
        "models": args.models,
        "grid_profile": args.grid_profile,
        "grid_sizes": {model: grid_size(grids[model]) for model in args.models},
        "segments": args.segments,
        "sensors": args.sensors,
        "feature_names": h4s3.FEATURE_NAMES,
        "feature_columns": input_features,
        "input_feature_count": int(len(input_features)),
        "common_preprocessing": preprocessing_report,
        "feature_sanitization": sanitization_report,
        "seeds": args.seeds,
        "case_scope": sorted(args.case_scope),
        "case_count": int(len(cases)),
        "case_run_count": int(feature_matrix[["case_id", "run"]].drop_duplicates().shape[0]),
        "feature_matrix_rows": int(len(feature_matrix)),
        "planned_atomic_tuning_runs": planned_atomic,
        "estimated_inner_grid_fits": planned_grid_fits,
        "dry_run": bool(args.dry_run),
    }
    feature_matrix.to_csv(output_dir / "data" / "H4_S4_feature_matrix.csv", index=False)
    write_json(output_dir / "analysis" / "H4_S4_common_preprocessing.json", preprocessing_report)
    write_json(output_dir / "analysis" / "H4_S4_feature_sanitization.json", sanitization_report)
    if args.dry_run:
        write_progress_state(output_dir, status="dry_run_completed", total_tasks=planned_atomic, completed_tasks=0, current_task=None)
        write_json(output_dir / "analysis" / "H4_S4_summary.json", summary)
        log_progress(f"Dry-run completed: {planned_atomic} atomic tuning runs planned; estimated inner fits={planned_grid_fits}; no models trained.")
        return summary

    try:
        shift_metrics, predictions, splits, tuning_results = run_grid(feature_matrix, input_features, args.segments, args.models, args.seeds, grids, output_dir, resume_enabled)
    except KeyboardInterrupt:
        mark_interrupted(output_dir)
        log_progress(f"Interrupted. Re-run the same command to resume from: {output_dir}")
        raise
    seed_metrics, segment_metrics, best_segment, best_by_model, case_metrics, case_segment_metrics = aggregate_metrics(shift_metrics)
    shift_metrics.to_csv(output_dir / "metrics" / "H4_S4_shift_metrics.csv", index=False)
    case_metrics.to_csv(output_dir / "metrics" / "H4_S4_case_metrics.csv", index=False)
    case_segment_metrics.to_csv(output_dir / "metrics" / "H4_S4_case_segment_metrics.csv", index=False)
    seed_metrics.to_csv(output_dir / "metrics" / "H4_S4_seed_metrics.csv", index=False)
    segment_metrics.to_csv(output_dir / "metrics" / "H4_S4_segment_metrics.csv", index=False)
    best_segment.to_csv(output_dir / "analysis" / "H4_S4_best_overall.csv", index=False)
    best_by_model.to_csv(output_dir / "analysis" / "H4_S4_best_by_model.csv", index=False)
    tuning_results.to_csv(output_dir / "analysis" / "H4_S4_tuning_results.csv", index=False)
    predictions.to_csv(output_dir / "predictions" / "H4_S4_predictions.csv", index=False)
    splits.to_csv(output_dir / "splits" / "H4_S4_splits.csv", index=False)
    fig_path = plot_tuned_segment_effect(output_dir, segment_metrics, args.dpi)
    summary.update(
        {
            "shift_metric_rows": int(len(shift_metrics)),
            "case_metric_rows": int(len(case_metrics)),
            "prediction_rows": int(len(predictions)),
            "tuning_result_rows": int(len(tuning_results)),
            "best_overall": best_segment.to_dict(orient="records"),
            "best_by_model": best_by_model.to_dict(orient="records"),
            "figure_path": str(fig_path.relative_to(output_dir)),
        }
    )
    write_json(output_dir / "analysis" / "H4_S4_summary.json", summary)
    write_report(output_dir, summary, segment_metrics, best_by_model)
    return summary


def main() -> None:
    output_dir: Path | None = None
    try:
        summary = run(parse_args())
        output_dir = Path(summary["execution_dir"])
        print(json.dumps(to_builtin(summary), indent=2, ensure_ascii=False))
    except Exception:
        if output_dir is not None:
            (output_dir / "logs" / "H4_S4_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
from milling_experiment_framework.models.h2_regressors import canonical_model_name, create_h2_feature_pipeline
from milling_experiment_framework.preprocessing.vb_common import COMMON_VB_PREPROCESSING_RULES, apply_common_vb_prediction_preprocessing
from milling_experiment_framework.visualization.figure_export import save_figure_dual
from scripts import run_H5_S2_T1_steady_length_feature_vb_suitability as t1
from scripts import run_H5_S2_T2_steady_position_feature_vb_suitability as t2


PREFIX = "H5_S3_T2"
TOPIC = "steady_position_model_prediction_effect"
SEGMENT_SETTING = t2.SEGMENT_SETTING
DEFAULT_SENSORS = [t2.SENSOR]
DEFAULT_FEATURES = t2.TARGET_FEATURES
POSITIONS = t2.POSITIONS
BASE_STEADY_LENGTH = t2.BASE_STEADY_LENGTH
SUBWINDOW_LENGTH = t2.SUBWINDOW_LENGTH
DEFAULT_CASE_SCOPE = t2.DEFAULT_CASE_SCOPE
DEFAULT_MODELS = ["linear_regression", "svr", "random_forest", "xgboost", "mlp", "feature_gru"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H5_S3_T2 steady-position model prediction effect experiment.")
    parser.add_argument("--signal-path", default="datasets/processed/mill_signal_data.csv")
    parser.add_argument("--process-info-path", default="datasets/processed/mill_process_info.csv")
    parser.add_argument("--metadata-path", default=t2.BASE_METADATA_PATH)
    parser.add_argument("--case-scope", type=int, nargs="+", default=DEFAULT_CASE_SCOPE)
    parser.add_argument("--sensors", nargs="+", default=DEFAULT_SENSORS)
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output-root", default="experiments/executions")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    return t2.to_builtin(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2, ensure_ascii=False), encoding="utf-8")


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
    return output_root / "H5" / "S3" / "T2" / f"{timestamp}_{TOPIC}"


def make_dirs(output_dir: Path) -> None:
    for rel in ["configs", "data", "splits", "metrics", "predictions", "analysis", "figures", "logs", "reports"]:
        (output_dir / rel).mkdir(parents=True, exist_ok=True)


def build_config(args: argparse.Namespace, timestamp: str, output_dir: Path) -> dict[str, Any]:
    models = [canonical_model_name(model) for model in args.models]
    selected_features = validate_features(args.features)
    return {
        "experiment": {
            "experiment_number": PREFIX,
            "name": f"{PREFIX}_{TOPIC}",
            "experiment_id": f"{timestamp}_{PREFIX}_{TOPIC}",
            "timestamp": timestamp,
            "execution_dir": str(output_dir),
            "analysis_type": "steady_position_model_prediction_effect",
        },
        "data": {
            "signal_path": args.signal_path,
            "process_info_path": args.process_info_path,
            "metadata_path": args.metadata_path,
            "case_scope": args.case_scope,
            "sensors": args.sensors,
            "segment_setting": SEGMENT_SETTING,
        },
        "common_preprocessing": COMMON_VB_PREPROCESSING_RULES,
        "windowing": {
            "base_steady_length": BASE_STEADY_LENGTH,
            "subwindow_length": SUBWINDOW_LENGTH,
            "positions": POSITIONS,
            "source": "H5_S2_T2 position definition",
        },
        "features": {
            "source": "H5_S2_T2 position-specific steady-cut feature extraction",
            "base_feature_names": t2.TARGET_FEATURES,
            "selected_feature_names": selected_features,
            "feature_naming": "{sensor}__{feature_name}",
        },
        "modeling": {
            "task": "VB regression",
            "input": "one row per case/run/steady_position, columns = selected sensors x selected features",
            "models": models,
            "seeds": args.seeds,
            "seed_policy": "linear_regression and svr run once with seed=-1; seeded models use configured seeds",
        },
        "split": {
            "strategy": "leave_one_case_out",
            "test_unit": "case",
            "validation": "none",
        },
    }


def validate_features(features: list[str]) -> list[str]:
    selected = []
    for feature in features:
        name = str(feature).strip()
        if name and name not in selected:
            selected.append(name)
    missing = [name for name in selected if name not in t2.TARGET_FEATURES]
    if missing:
        raise ValueError(f"Unsupported features: {missing}. Supported features: {t2.TARGET_FEATURES}")
    if not selected:
        raise ValueError("At least one feature must be selected.")
    return selected


def feature_columns(sensors: list[str], features: list[str]) -> list[str]:
    return [f"{sensor}__{feature}" for sensor in sensors for feature in features]


def load_feature_wide(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_features = validate_features(args.features)
    metadata = pd.read_csv(resolve(args.metadata_path))
    signal = pd.read_csv(resolve(args.signal_path), usecols=["case", "run", *args.sensors])
    process = pd.read_csv(resolve(args.process_info_path))
    required_metadata = {"case", "run", "idx_start", "idx_end", "signal_length"}
    required_process = {"case", "run", "VB"}
    missing_metadata = required_metadata.difference(metadata.columns)
    missing_process = required_process.difference(process.columns)
    missing_signal = set(args.sensors).difference(signal.columns)
    if missing_metadata:
        raise ValueError(f"metadata missing columns: {sorted(missing_metadata)}")
    if missing_process:
        raise ValueError(f"process info missing columns: {sorted(missing_process)}")
    if missing_signal:
        raise ValueError(f"signal data missing sensor columns: {sorted(missing_signal)}")

    keys = metadata[["case", "run", "idx_start", "idx_end", "signal_length"]].drop_duplicates().copy()
    keys = keys.loc[keys["case"].isin(args.case_scope)].copy()
    merged = keys.merge(signal, on=["case", "run"], how="left", validate="one_to_one")
    merged = merged.merge(process, on=["case", "run"], how="left", validate="one_to_one")
    missing_any = merged[args.sensors].isna().any(axis=1)
    if missing_any.any():
        missing = merged.loc[missing_any, ["case", "run"]].to_dict(orient="records")
        raise ValueError(f"Missing signal rows: {missing[:5]}")
    merged, preprocessing_report = apply_common_vb_prediction_preprocessing(merged)

    rows: list[dict[str, Any]] = []
    for source_row in merged.itertuples(index=False):
        case = int(source_row.case)
        run_id = int(source_row.run)
        arrays = {sensor: t1.parse_signal(getattr(source_row, sensor)) for sensor in args.sensors}
        min_signal_size = min(arr.size for arr in arrays.values())
        bounds = t2.position_bounds(int(source_row.idx_start), int(source_row.idx_end), min_signal_size)
        for position in POSITIONS:
            window_start, window_end = bounds[position]
            row: dict[str, Any] = {
                "sample_id": f"case_{case:02d}_run_{run_id:03d}_position_{position}",
                "dataset_run_id": f"case_{case:02d}_run_{run_id:03d}",
                "case_id": case,
                "run": run_id,
                "steady_position": position,
                "position_order": POSITIONS.index(position),
                "base_steady_length": BASE_STEADY_LENGTH,
                "subwindow_length": SUBWINDOW_LENGTH,
                "segment_setting": SEGMENT_SETTING,
                "VB": float(source_row.VB),
                "time": getattr(source_row, "time", np.nan),
                "DOC": getattr(source_row, "DOC", np.nan),
                "feed": getattr(source_row, "feed", np.nan),
                "material_name": getattr(source_row, "material_name", np.nan),
                "base_idx_start": int(source_row.idx_start),
                "base_idx_end": int(source_row.idx_end),
                "subwindow_idx_start": int(window_start),
                "subwindow_idx_end": int(window_end),
            }
            for sensor, signal_arr in arrays.items():
                values = t1.compute_features(signal_arr[window_start:window_end])
                for feature_name in selected_features:
                    row[f"{sensor}__{feature_name}"] = values[feature_name]
            rows.append(row)
    feature_wide = pd.DataFrame(rows).sort_values(["position_order", "case_id", "run"]).reset_index(drop=True)
    missing_cols = [col for col in feature_columns(args.sensors, selected_features) if col not in feature_wide.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    return feature_wide, preprocessing_report


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    out = {
        "metric_mae": float(mean_absolute_error(y_true, y_pred)),
        "metric_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "metric_r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }
    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        out["metric_pearson"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        out["metric_pearson"] = float("nan")
    return out


def split_for_case(data: pd.DataFrame, target_case: int) -> pd.DataFrame:
    split = data[["sample_id", "dataset_run_id", "case_id", "run", "steady_position", "position_order", "VB"]].copy()
    split["split"] = np.where(split["case_id"].eq(target_case), "test", "train")
    return split


def run_grid(feature_wide: pd.DataFrame, input_features: list[str], models: list[str], seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shift_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    split_rows: list[pd.DataFrame] = []
    cases = sorted(feature_wide["case_id"].unique().tolist())
    positions = feature_wide[["steady_position", "position_order"]].drop_duplicates().sort_values("position_order")

    for pos_row in positions.itertuples(index=False):
        position = str(pos_row.steady_position)
        position_order = int(pos_row.position_order)
        position_df = feature_wide.loc[feature_wide["steady_position"].eq(position)].copy()
        for model_name in models:
            for seed in effective_seeds_for_model(model_name, seeds):
                for target_case in cases:
                    split = split_for_case(position_df, target_case)
                    train = position_df.loc[position_df["case_id"].ne(target_case)].copy()
                    test = position_df.loc[position_df["case_id"].eq(target_case)].copy()
                    if train.empty or test.empty:
                        raise ValueError(f"Empty split for position={position}, target_case={target_case}")
                    model = create_h2_feature_pipeline(model_name, model_seed_value(seed), model_defaults=None)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(train[input_features], train["VB"])
                    y_pred = model.predict(test[input_features])
                    metrics = metric_dict(test["VB"].to_numpy(), y_pred)
                    scenario = f"train_without_case_{target_case}_to_case_{target_case}"
                    shift_rows.append(
                        {
                            "steady_position": position,
                            "position_order": position_order,
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
                    )
                    preds = test[["sample_id", "dataset_run_id", "case_id", "run", "steady_position", "position_order", "VB"]].copy()
                    preds["model"] = model_name
                    preds["seed"] = int(seed)
                    preds["seed_label"] = seed_label(seed)
                    preds["target_case"] = int(target_case)
                    preds["scenario"] = scenario
                    preds["y_true"] = preds["VB"]
                    preds["y_pred"] = y_pred
                    preds["residual"] = preds["y_true"] - preds["y_pred"]
                    preds["absolute_error"] = preds["residual"].abs()
                    prediction_rows.append(preds)
                    split["model"] = model_name
                    split["seed"] = int(seed)
                    split["seed_label"] = seed_label(seed)
                    split["target_case"] = int(target_case)
                    split["scenario"] = scenario
                    split_rows.append(split)

    return pd.DataFrame(shift_rows), pd.concat(prediction_rows, ignore_index=True), pd.concat(split_rows, ignore_index=True)


def aggregate_metrics(shift_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_metrics = (
        shift_metrics.groupby(["steady_position", "position_order", "model", "seed", "seed_label"], as_index=False)
        .agg(
            mean_mae_over_cases=("metric_mae", "mean"),
            mean_rmse_over_cases=("metric_rmse", "mean"),
            mean_r2_over_cases=("metric_r2", "mean"),
            mean_pearson_over_cases=("metric_pearson", "mean"),
            std_rmse_over_cases=("metric_rmse", "std"),
            test_case_count=("target_case", "nunique"),
            total_test_rows=("test_rows", "sum"),
        )
    )
    seed_metrics["std_rmse_over_cases"] = seed_metrics["std_rmse_over_cases"].fillna(0.0)
    model_metrics = (
        seed_metrics.groupby(["steady_position", "position_order", "model"], as_index=False)
        .agg(
            mean_mae=("mean_mae_over_cases", "mean"),
            std_mae=("mean_mae_over_cases", "std"),
            mean_rmse=("mean_rmse_over_cases", "mean"),
            std_rmse=("mean_rmse_over_cases", "std"),
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
    model_metrics = model_metrics.sort_values(["model", "position_order"]).reset_index(drop=True)
    return seed_metrics, model_metrics, best_by_model.reset_index(drop=True)


def plot_position_effect(output_dir: Path, model_metrics: pd.DataFrame, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for model_name, group in model_metrics.groupby("model"):
        group = group.sort_values("position_order")
        x = group["position_order"].to_numpy(dtype=float)
        ax.plot(x, group["mean_rmse"], marker="o", linewidth=1.4, label=model_name)
        ax.fill_between(
            x,
            (group["mean_rmse"] - group["std_rmse"]).to_numpy(dtype=float),
            (group["mean_rmse"] + group["std_rmse"]).to_numpy(dtype=float),
            alpha=0.12,
        )
    ax.set_xticks(np.arange(len(POSITIONS)))
    ax.set_xticklabels(POSITIONS)
    ax.set_xlabel("2000-sample window position inside 5000-sample steady-cut")
    ax.set_ylabel("mean RMSE over leave-one-case-out folds")
    ax.set_title(f"{PREFIX} prediction error by steady-cut position")
    ax.grid(True, color="#e5e7eb", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "figures" / "H5_S3_T2_position_effect_rmse.png"
    save_figure_dual(fig, path, dpi=dpi)
    plt.close(fig)
    return path


def write_report(output_dir: Path, summary: dict[str, Any], model_metrics: pd.DataFrame, best_by_model: pd.DataFrame) -> None:
    overall = model_metrics.sort_values("mean_rmse").head(12)
    lines = [
        f"# {PREFIX} Steady-position Model Prediction Effect",
        "",
        "## Scope",
        "",
        f"- Sensors: `{summary['sensors']}`",
        f"- Segment: `{SEGMENT_SETTING}`",
        f"- Base steady-cut length: `{BASE_STEADY_LENGTH}`",
        f"- Sub-window length: `{SUBWINDOW_LENGTH}`",
        f"- Positions: `{summary['steady_positions']}`",
        f"- Selected features: `{summary['selected_features']}`",
        f"- Feature input column count: `{summary['input_feature_count']}`",
        f"- Split: leave-one-case-out over `{summary['case_count']}` cases",
        f"- Models: `{summary['models']}`",
        "",
        "## Outputs",
        "",
        "- Feature matrix: `data/H5_S3_T2_feature_matrix.csv`",
        "- Common preprocessing: `analysis/H5_S3_T2_common_preprocessing.json`",
        "- Shift metrics: `metrics/H5_S3_T2_shift_metrics.csv`",
        "- Seed metrics: `metrics/H5_S3_T2_seed_metrics.csv`",
        "- Position-model metrics: `metrics/H5_S3_T2_position_model_metrics.csv`",
        "- Predictions: `predictions/H5_S3_T2_predictions.csv`",
        "- Position effect figure: `figures/H5_S3_T2_position_effect_rmse.{png,svg}`",
        "",
        "## Best Position By Model",
        "",
        "| model | best_steady_position | mean_rmse | mean_mae | mean_r2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in best_by_model.itertuples(index=False):
        lines.append(f"| {row.model} | {row.steady_position} | {row.mean_rmse:.6f} | {row.mean_mae:.6f} | {row.mean_r2:.6f} |")
    lines.extend(["", "## Overall Top Rows", "", "| model | steady_position | mean_rmse | mean_mae | mean_r2 |", "|---|---:|---:|---:|---:|"])
    for row in overall.itertuples(index=False):
        lines.append(f"| {row.model} | {row.steady_position} | {row.mean_rmse:.6f} | {row.mean_mae:.6f} | {row.mean_r2:.6f} |")
    lines.extend(
        [
            "",
            "## Method Note",
            "",
            "The 5000-sample steady-cut interval is taken from the H5_S2_T2 metadata path. "
            "Selected features are recomputed per selected sensor on 2000-sample start, center, and end windows, then evaluated with leave-one-case-out VB regression.",
            "Common VB preprocessing is applied before feature construction: missing VB is interpolated within each case, then each case's first run is excluded.",
        ]
    )
    (output_dir / "reports" / "H5_S3_T2_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = args.timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = execution_dir(resolve(args.output_root), timestamp)
    if output_dir.exists():
        raise FileExistsError(f"Execution directory already exists: {output_dir}")
    make_dirs(output_dir)

    models = [canonical_model_name(model) for model in args.models]
    args.features = validate_features(args.features)
    config = build_config(args, timestamp, output_dir)
    write_yaml(output_dir / "configs" / "H5_S3_T2_input_config.yaml", config)
    write_json(output_dir / "logs" / "H5_S3_T2_environment.json", collect_environment())

    feature_wide, preprocessing_report = load_feature_wide(args)
    input_features = feature_columns(args.sensors, args.features)
    positions = feature_wide[["steady_position", "position_order"]].drop_duplicates().sort_values("position_order")
    position_names = positions["steady_position"].tolist()
    cases = sorted(feature_wide["case_id"].unique().tolist())
    summary: dict[str, Any] = {
        "experiment_id": config["experiment"]["experiment_id"],
        "execution_dir": str(output_dir),
        "sensors": args.sensors,
        "segment_setting": SEGMENT_SETTING,
        "base_steady_length": BASE_STEADY_LENGTH,
        "subwindow_length": SUBWINDOW_LENGTH,
        "steady_positions": position_names,
        "selected_features": args.features,
        "feature_columns": input_features,
        "input_feature_count": int(len(input_features)),
        "common_preprocessing": preprocessing_report,
        "models": models,
        "seeds": args.seeds,
        "case_scope": sorted(args.case_scope),
        "case_count": int(len(cases)),
        "case_run_count": int(feature_wide[["case_id", "run"]].drop_duplicates().shape[0]),
        "feature_matrix_rows": int(len(feature_wide)),
        "planned_atomic_fits": int(sum(len(effective_seeds_for_model(model, args.seeds)) * len(position_names) * len(cases) for model in models)),
        "dry_run": bool(args.dry_run),
    }
    feature_wide.to_csv(output_dir / "data" / "H5_S3_T2_feature_matrix.csv", index=False)
    write_json(output_dir / "analysis" / "H5_S3_T2_common_preprocessing.json", preprocessing_report)
    if args.dry_run:
        write_json(output_dir / "analysis" / "H5_S3_T2_summary.json", summary)
        return summary

    shift_metrics, predictions, splits = run_grid(feature_wide, input_features, models, args.seeds)
    seed_metrics, model_metrics, best_by_model = aggregate_metrics(shift_metrics)
    shift_metrics.to_csv(output_dir / "metrics" / "H5_S3_T2_shift_metrics.csv", index=False)
    seed_metrics.to_csv(output_dir / "metrics" / "H5_S3_T2_seed_metrics.csv", index=False)
    model_metrics.to_csv(output_dir / "metrics" / "H5_S3_T2_position_model_metrics.csv", index=False)
    best_by_model.to_csv(output_dir / "analysis" / "H5_S3_T2_best_position_by_model.csv", index=False)
    predictions.to_csv(output_dir / "predictions" / "H5_S3_T2_predictions.csv", index=False)
    splits.to_csv(output_dir / "splits" / "H5_S3_T2_splits.csv", index=False)
    fig_path = plot_position_effect(output_dir, model_metrics, args.dpi)

    summary.update(
        {
            "shift_metric_rows": int(len(shift_metrics)),
            "prediction_rows": int(len(predictions)),
            "best_overall": model_metrics.sort_values("mean_rmse").head(1).to_dict(orient="records"),
            "best_by_model": best_by_model.to_dict(orient="records"),
            "figure_path": str(fig_path.relative_to(output_dir)),
        }
    )
    write_json(output_dir / "analysis" / "H5_S3_T2_summary.json", summary)
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
            (output_dir / "logs" / "H5_S3_T2_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()

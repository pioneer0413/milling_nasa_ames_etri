from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import milling_experiment_framework.datasets.loaders  # noqa: F401
from milling_experiment_framework.analysis.error_analyzer import prediction_errors
from milling_experiment_framework.analysis.execution_analyzer import analysis_summary
from milling_experiment_framework.analysis.summary_tables import metrics_summary_table
from milling_experiment_framework.core.config import load_and_resolve_config, validate_metadata_columns
from milling_experiment_framework.core.seed import set_seed
from milling_experiment_framework.datasets.schema import dataset_fingerprint
from milling_experiment_framework.evaluation.evaluator import Evaluator
from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.experiment_logging.experiment_logger import ExperimentLogger
from milling_experiment_framework.experiment_logging.result_writer import write_metrics
from milling_experiment_framework.models.factory import create_model
from milling_experiment_framework.preprocessing.pipeline import PreprocessingPipeline
from milling_experiment_framework.registry.dataset_registry import DATASET_REGISTRY
from milling_experiment_framework.reports.report_generator import generate_report
from milling_experiment_framework.splits.split_manager import create_split
from milling_experiment_framework.training.trainer import DLTrainer
from milling_experiment_framework.utils.dataframe import label_distribution, metadata_summary
from milling_experiment_framework.utils.io import write_csv, write_json, write_yaml
from milling_experiment_framework.utils.paths import ExperimentPaths
from milling_experiment_framework.visualization.plot_manager import create_standard_plots


class ExperimentExecution:
    def __init__(self, config_path: str | Path, root: str | Path = ".", smoke: bool = False, dry_run: bool = False):
        self.config_path = Path(config_path)
        self.root = Path(root).resolve()
        self.smoke = smoke
        self.dry_run = dry_run

    def run(self) -> dict[str, Any]:
        input_config, config, validation = load_and_resolve_config(self.config_path)
        experiment_id = config["experiment"]["experiment_id"]
        paths = ExperimentPaths(self.root, experiment_id)
        paths.apply_to_config(config)
        paths.prepare_standard_dirs()
        logger = ExperimentLogger(paths.execution_dir / "logs" / "run.log")
        logger.info(f"experiment_execution started: {experiment_id}")
        if self.smoke:
            logger.info("smoke mode enabled")
        if self.dry_run:
            logger.info("dry-run mode enabled")
        start = datetime.now()
        self._write_initial_artifacts(paths, input_config, config, validation)
        self._update_index(config, status="running", created_at=config["experiment"]["created_at"])
        try:
            if not validation["ok"]:
                raise ValueError("Config validation failed: " + "; ".join(validation["errors"]))
            set_seed(int(config["experiment"]["seed"]))
            data = self._load_dataset(config)
            meta_validation = validate_metadata_columns(config, list(data.metadata.columns))
            write_json(paths.execution_dir / "configs" / "metadata_validation.json", meta_validation)
            if not meta_validation["ok"]:
                raise ValueError("Metadata validation failed: " + "; ".join(meta_validation["errors"]))
            if self.smoke:
                data = self._take_smoke_subset(data, int(config.get("smoke", {}).get("num_samples", 32)))
            split = create_split(data.metadata, config["split"], int(config["experiment"]["seed"]))
            data.metadata = split.assignments.reset_index(drop=True)
            data.y = np.asarray(data.metadata[config["task"]["target_column"]])
            self._write_data_artifacts(paths, data, split)
            preprocess = PreprocessingPipeline(config["preprocessing"]).fit_transform(data)
            data = preprocess.data
            self._write_preprocessing_artifacts(paths, preprocess)
            if self.dry_run:
                logger.info("dry-run completed before model training")
                self._update_index(config, status="finished", best_metric=None)
                return {"experiment_id": experiment_id, "dry_run": True, "execution_dir": str(paths.execution_dir)}
            model = create_model(config["model"], config["task"])
            history, predictions, model_artifact = self._train_and_predict(paths, config, model, data)
            metrics, domain_metrics = Evaluator(config["task"], config["evaluation"]).evaluate(predictions)
            metrics_row = self._metrics_row(config, metrics)
            write_metrics(paths.execution_dir / "metrics", metrics, metrics_row)
            write_csv(paths.execution_dir / "metrics" / "training_history.csv", history)
            write_csv(paths.execution_dir / "metrics" / "domain_metrics.csv", domain_metrics)
            write_json(paths.execution_dir / "metrics" / "threshold_summary.json", {"threshold": None})
            self._write_predictions_and_analysis(paths, config, predictions, metrics)
            self._write_model_artifacts(paths, config, model, model_artifact, preprocess)
            create_standard_plots(config["task"]["type"], predictions, history, paths.execution_dir / "figures")
            env = collect_environment(str(self.root))
            end = datetime.now()
            env["started_at"] = start.isoformat(timespec="seconds")
            env["finished_at"] = end.isoformat(timespec="seconds")
            env["duration_seconds"] = (end - start).total_seconds()
            write_json(paths.execution_dir / "logs" / "environment.json", env)
            write_json(
                paths.execution_dir / "logs" / "reproducibility.json",
                {
                    "seed": config["experiment"]["seed"],
                    "deterministic": True,
                    "config_hash": config["config_hash"],
                    "framework_version": config["framework_version"],
                    "git": env.get("git", {}),
                },
            )
            report_artifacts = {
                "resolved_config": paths.execution_dir / "configs" / "resolved_config.yaml",
                "split_csv": paths.execution_dir / "splits" / "split.csv",
                "predictions_csv": paths.execution_dir / "predictions" / "predictions.csv",
                "model_artifact": model_artifact,
                "environment_json": paths.execution_dir / "logs" / "environment.json",
            }
            if config.get("report", {}).get("enabled", True):
                generate_report(paths.execution_dir / "reports" / "report.md", config, metrics, report_artifacts)
            best_metric = self._best_metric(metrics_row, config["evaluation"].get("metrics", []))
            self._update_index(config, status="finished", best_metric=best_metric)
            logger.info(f"experiment_execution finished: {experiment_id}")
            return {"experiment_id": experiment_id, "metrics": metrics, "execution_dir": str(paths.execution_dir)}
        except Exception as exc:
            error_text = traceback.format_exc()
            logger.error(str(exc))
            (paths.execution_dir / "logs" / "error.log").write_text(error_text, encoding="utf-8")
            self._update_index(
                config,
                status="failed",
                error_message=str(exc),
                error_log=str(paths.execution_dir / "logs" / "error.log"),
            )
            raise

    def _load_dataset(self, config: dict) -> Any:
        dataset_cfg = dict(config["dataset"])
        dataset_cfg["seed"] = config["experiment"]["seed"]
        dataset_cfg["steady_cut_mode"] = config.get("steady_cut_mode")
        loader_cls = DATASET_REGISTRY.get(dataset_cfg.get("name"))
        return loader_cls(dataset_cfg).load()

    def _take_smoke_subset(self, data, n: int):
        n = min(n, len(data.metadata))
        data.X = data.X[:n] if data.X is not None else None
        data.y = data.y[:n]
        data.metadata = data.metadata.iloc[:n].reset_index(drop=True)
        return data

    def _write_initial_artifacts(self, paths, input_config, config, validation) -> None:
        write_yaml(paths.execution_dir / "configs" / "input_config.yaml", input_config)
        write_yaml(paths.execution_dir / "configs" / "resolved_config.yaml", config)
        write_json(paths.execution_dir / "configs" / "config_validation.json", validation)

    def _write_data_artifacts(self, paths, data, split) -> None:
        write_csv(paths.execution_dir / "splits" / "split.csv", split.assignments)
        write_json(paths.execution_dir / "splits" / "split_summary.json", split.summary)
        write_json(paths.execution_dir / "splits" / "leakage_check.json", split.leakage_check)
        write_json(paths.execution_dir / "data" / "dataset_fingerprint.json", dataset_fingerprint(data.metadata, data.dataset_info))
        write_json(
            paths.execution_dir / "data" / "dataset_summary.json",
            {
                **data.dataset_info,
                "num_samples": int(len(data.metadata)),
                "sensor_channels": data.sensor_channels,
                "input_shape": list(data.primary_X().shape),
                "split_counts": data.metadata["split"].value_counts().to_dict(),
            },
        )
        write_csv(paths.execution_dir / "data" / "label_distribution.csv", label_distribution(data.metadata, "split"))
        write_json(paths.execution_dir / "data" / "metadata_summary.json", metadata_summary(data.metadata))
        write_json(paths.execution_dir / "data" / "shape_summary.json", {"X": list(data.primary_X().shape), "y": list(data.y.shape)})

    def _write_preprocessing_artifacts(self, paths, preprocess) -> None:
        write_json(paths.execution_dir / "preprocessing" / "preprocessing_summary.json", preprocess.summary)
        write_json(paths.execution_dir / "preprocessing" / "normalization_params.json", preprocess.artifacts.get("normalization_params", {}))
        write_json(paths.execution_dir / "preprocessing" / "feature_summary.json", preprocess.artifacts.get("feature_summary", {}))
        write_json(paths.execution_dir / "preprocessing" / "steady_cut_summary.json", preprocess.artifacts.get("steady_cut_summary", {}))
        write_json(paths.execution_dir / "preprocessing" / "shape_trace.json", preprocess.shape_trace)

    def _train_and_predict(self, paths, config, model, data) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
        X = data.primary_X()
        y = np.asarray(data.y)
        task_type = config["task"]["type"]
        if config["model"]["model_type"] == "DL":
            trainer = DLTrainer(model, config["training"], config["checkpoint"], config["config_hash"], paths.execution_dir)
            history = trainer.fit(X, y, data.metadata)
            y_pred, probabilities = trainer.predict(X)
            model_artifact = str(paths.execution_dir / "checkpoints" / "best.pt")
        else:
            train_mask = data.metadata["split"].to_numpy() == "train"
            model.fit(X[train_mask], y[train_mask])
            y_pred = model.predict(X)
            probabilities = model.predict_proba(X) if task_type != "regression" else None
            model_artifact = str(paths.execution_dir / "artifacts" / "model.joblib")
            model.save(model_artifact)
            history = pd.DataFrame([{"epoch": 1, "train_loss": None, "val_loss": None, "best_metric": None}])
        predictions = data.metadata.copy()
        predictions["y_true"] = y
        predictions["y_pred"] = y_pred
        if task_type == "regression":
            predictions["residual"] = predictions["y_true"] - predictions["y_pred"]
            predictions["absolute_error"] = predictions["residual"].abs()
        elif probabilities is not None:
            for idx in range(probabilities.shape[1]):
                predictions[f"prob_class_{idx}"] = probabilities[:, idx]
        return history, predictions, model_artifact

    def _write_predictions_and_analysis(self, paths, config, predictions, metrics) -> None:
        write_csv(paths.execution_dir / "predictions" / "predictions.csv", predictions)
        errors = prediction_errors(predictions, config["task"]["type"])
        write_csv(paths.execution_dir / "predictions" / "prediction_errors.csv", errors)
        write_json(paths.execution_dir / "analysis" / "analysis_summary.json", analysis_summary(predictions, metrics))
        write_csv(paths.execution_dir / "analysis" / "error_summary.csv", errors)
        domain_cols = [c for c in ["domain_id", "condition_id", "tool_id", "machine_id"] if c in predictions.columns]
        if domain_cols:
            rows = []
            for col in domain_cols:
                rows.extend(predictions.groupby(col).size().reset_index(name="count").assign(group_column=col).to_dict("records"))
            write_csv(paths.execution_dir / "analysis" / "domain_analysis.csv", pd.DataFrame(rows))
        else:
            write_csv(paths.execution_dir / "analysis" / "domain_analysis.csv", pd.DataFrame())
        if "steady_cut_mode" in predictions:
            write_csv(paths.execution_dir / "analysis" / "steady_cut_analysis.csv", predictions.groupby("steady_cut_mode").size().reset_index(name="count"))
        else:
            write_csv(paths.execution_dir / "analysis" / "steady_cut_analysis.csv", pd.DataFrame())
        write_csv(paths.execution_dir / "analysis" / "summary_tables.csv", metrics_summary_table(metrics))

    def _write_model_artifacts(self, paths, config, model, model_artifact, preprocess) -> None:
        write_json(
            paths.execution_dir / "model" / "model_summary.json",
            {
                "name": config["model"]["name"],
                "model_type": config["model"]["model_type"],
                "input_type": config["model"]["input_type"],
                "params": config["model"].get("params", {}),
            },
        )
        write_yaml(paths.execution_dir / "model" / "model_config.yaml", config["model"])
        architecture = str(getattr(model, "module", getattr(model, "estimator", model)))
        (paths.execution_dir / "model" / "model_architecture.txt").write_text(architecture, encoding="utf-8")
        importance = getattr(model, "feature_importance", lambda: None)()
        feature_names = preprocess.artifacts.get("feature_summary", {}).get("feature_names", [])
        if importance is not None:
            write_csv(
                paths.execution_dir / "model" / "feature_importance.csv",
                pd.DataFrame({"feature": feature_names or [f"f{i}" for i in range(len(importance))], "importance": importance}),
            )
        else:
            write_csv(paths.execution_dir / "model" / "feature_importance.csv", pd.DataFrame())

    def _metrics_row(self, config, metrics: dict) -> dict:
        row = {
            "experiment_id": config["experiment"]["experiment_id"],
            "dataset": config["dataset"]["name"],
            "model": config["model"]["name"],
            "input_type": config["model"]["input_type"],
            "split_strategy": config["split"]["strategy"],
            "steady_cut_mode": config.get("steady_cut_mode", "full_signal"),
        }
        for key, value in metrics.get("overall", {}).items():
            row[key] = value
        for split, values in metrics.get("splits", {}).items():
            for key, value in values.items():
                row[f"{split}_{key}"] = value
        return row

    def _best_metric(self, row: dict, metric_names: list[str]) -> float | None:
        for name in metric_names:
            if name in row:
                return float(row[name])
        for key, value in row.items():
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _update_index(
        self,
        config: dict,
        status: str,
        created_at: str | None = None,
        best_metric: float | None = None,
        error_message: str | None = None,
        error_log: str | None = None,
    ) -> None:
        index_path = self.root / "experiments" / "index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "experiment_id": config["experiment"]["experiment_id"],
            "hypothesis_id": config["experiment"].get("hypothesis_id"),
            "scenario_id": config["experiment"].get("scenario_id"),
            "timestamp": config["experiment"].get("timestamp"),
            "experiment_topic": config["experiment"].get("experiment_topic"),
            "execution_dir": config["experiment"].get("execution_dir"),
            "legacy_execution_dir": config["experiment"].get("legacy_execution_dir"),
            "path_schema_version": config["experiment"].get("path_schema_version"),
            "experiment_name": config["experiment"].get("name"),
            "dataset": config["dataset"].get("name"),
            "model": config["model"].get("name"),
            "input_type": config["model"].get("input_type"),
            "split_strategy": config["split"].get("strategy"),
            "steady_cut_mode": config.get("steady_cut_mode", "full_signal"),
            "status": status,
            "best_metric": best_metric,
            "created_at": created_at or config["experiment"].get("created_at"),
            "error_message": error_message,
            "error_log": error_log,
        }
        if index_path.exists():
            frame = pd.read_csv(index_path)
            frame = frame.loc[frame["experiment_id"] != row["experiment_id"]]
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        else:
            frame = pd.DataFrame([row])
        frame.to_csv(index_path, index=False)

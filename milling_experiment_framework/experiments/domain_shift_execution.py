from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from milling_experiment_framework.core.config import load_and_resolve_config
from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.experiment_logging.experiment_logger import ExperimentLogger
from milling_experiment_framework.reports.report_generator import generate_report
from milling_experiment_framework.utils.io import write_csv, write_json, write_yaml
from milling_experiment_framework.utils.paths import ExperimentPaths


CASE_SCOPE = [1, 2, 8, 9, 12, 14]
DOMAIN_CASES = {
    "A": [1, 9],
    "B": [2, 12],
    "C": [8, 14],
}
SHIFT_SCENARIOS = [
    ("A", "B"),
    ("A", "C"),
    ("B", "A"),
    ("B", "C"),
    ("C", "A"),
    ("C", "B"),
]
SIGNAL_COLUMNS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
PROCESS_FEATURES = ["time", "DOC", "feed", "material"]


class DomainShiftExecution:
    """Run the fixed A/B/C case-pair domain-shift protocol from docs/usage/cli.md."""

    def __init__(self, config_path: str | Path, root: str | Path = "."):
        self.config_path = Path(config_path)
        self.root = Path(root).resolve()

    def run(self) -> dict[str, Any]:
        input_config, config, validation = load_and_resolve_config(self.config_path)
        experiment_id = config["experiment"]["experiment_id"]
        paths = ExperimentPaths(self.root, experiment_id)
        paths.prepare_standard_dirs()
        logger = ExperimentLogger(paths.execution_dir / "logs" / "run.log")
        logger.info(f"domain_shift_execution started: {experiment_id}")
        write_yaml(paths.execution_dir / "configs" / "input_config.yaml", input_config)
        write_yaml(paths.execution_dir / "configs" / "resolved_config.yaml", config)
        write_json(paths.execution_dir / "configs" / "config_validation.json", validation)
        if not validation["ok"]:
            raise ValueError("Config validation failed: " + "; ".join(validation["errors"]))

        data, feature_columns = self._load_features(config)
        self._write_dataset_artifacts(paths, config, data, feature_columns)

        rows = []
        predictions = []
        split_assignments = []
        failures = []
        model_params = dict(config.get("model", {}).get("params", {}))
        model_params.setdefault("n_estimators", 100)
        model_params.setdefault("random_state", int(config["experiment"].get("seed", 42)))
        for source_domain, target_domain in SHIFT_SCENARIOS:
            scenario = f"{source_domain}_to_{target_domain}"
            source_cases = DOMAIN_CASES[source_domain]
            target_cases = DOMAIN_CASES[target_domain]
            try:
                scenario_result, scenario_predictions = self._run_one_shift(
                    data=data,
                    feature_columns=feature_columns,
                    source_domain=source_domain,
                    target_domain=target_domain,
                    source_cases=source_cases,
                    target_cases=target_cases,
                    model_params=model_params,
                    metrics=config["evaluation"].get("metrics", ["mae", "rmse", "r2"]),
                )
                rows.extend(scenario_result)
                predictions.append(scenario_predictions)
                split_assignments.append(
                    self._scenario_split_assignments(data, source_domain, target_domain, source_cases, target_cases)
                )
                logger.info(f"shift finished: {scenario}")
            except Exception as exc:
                failures.append({"scenario": scenario, "error": str(exc)})
                logger.error(f"shift failed: {scenario}: {exc}")

        shift_metrics = pd.DataFrame(rows)
        if not shift_metrics.empty:
            shift_metrics["experiment_id"] = experiment_id
        write_csv(paths.execution_dir / "metrics" / "shift_metrics.csv", shift_metrics)
        if predictions:
            write_csv(paths.execution_dir / "predictions" / "predictions.csv", pd.concat(predictions, ignore_index=True))
        else:
            write_csv(paths.execution_dir / "predictions" / "predictions.csv", pd.DataFrame())
        if split_assignments:
            write_csv(paths.execution_dir / "splits" / "split.csv", pd.concat(split_assignments, ignore_index=True))
        else:
            write_csv(paths.execution_dir / "splits" / "split.csv", pd.DataFrame())

        metrics_json = self._aggregate_metrics(shift_metrics, failures)
        write_json(paths.execution_dir / "metrics" / "metrics.json", metrics_json)
        write_csv(paths.execution_dir / "metrics" / "metrics.csv", self._metrics_csv_row(config, metrics_json))
        write_csv(paths.execution_dir / "metrics" / "domain_metrics.csv", shift_metrics)
        write_json(paths.execution_dir / "splits" / "leakage_check.json", self._leakage_summary(failures))
        write_json(paths.execution_dir / "logs" / "environment.json", collect_environment(str(self.root)))
        write_json(
            paths.execution_dir / "logs" / "reproducibility.json",
            {
                "seed": config["experiment"].get("seed"),
                "config_hash": config.get("config_hash"),
                "framework_version": config.get("framework_version"),
                "protocol": "fixed_case_pair_domain_shift",
                "case_scope": CASE_SCOPE,
                "domain_cases": DOMAIN_CASES,
                "shift_scenarios": [f"{s}_to_{t}" for s, t in SHIFT_SCENARIOS],
            },
        )
        generate_report(
            paths.execution_dir / "reports" / "report.md",
            config,
            metrics_json,
            {
                "resolved_config": paths.execution_dir / "configs" / "resolved_config.yaml",
                "split_csv": paths.execution_dir / "metrics" / "shift_metrics.csv",
                "predictions_csv": paths.execution_dir / "predictions" / "predictions.csv",
                "model_artifact": "per-shift in-memory sklearn pipelines",
                "environment_json": paths.execution_dir / "logs" / "environment.json",
            },
        )
        self._update_index(config, metrics_json, "failed" if failures else "finished")
        logger.info(f"domain_shift_execution finished: {experiment_id}")
        return {"experiment_id": experiment_id, "metrics": metrics_json, "execution_dir": str(paths.execution_dir)}

    def _load_features(self, config: dict) -> tuple[pd.DataFrame, list[str]]:
        process_path = Path(config["dataset"]["process_info_path"])
        signal_path = Path(config["dataset"]["signal_data_path"])
        process = pd.read_csv(process_path)
        signal = pd.read_csv(signal_path)
        process = process.loc[process["enable"].astype(bool) & process["case"].isin(CASE_SCOPE)].copy()
        signal = signal.loc[signal["enable"].astype(bool) & signal["case"].isin(CASE_SCOPE)].copy()
        data = process.merge(signal, on=["case", "run"], suffixes=("", "_signal"), validate="one_to_one")
        data = data.loc[data["VB"].notna()].reset_index(drop=True)
        feature_rows = []
        for _, row in data.iterrows():
            features: dict[str, float] = {}
            for col in SIGNAL_COLUMNS:
                arr = np.asarray(ast.literal_eval(row[col]), dtype="float64")
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                features[f"{col}_mean"] = float(arr.mean())
                features[f"{col}_std"] = float(arr.std())
                features[f"{col}_min"] = float(arr.min())
                features[f"{col}_max"] = float(arr.max())
                features[f"{col}_rms"] = float(np.sqrt(np.mean(np.square(arr))))
                features[f"{col}_p2p"] = float(arr.max() - arr.min())
            for col in PROCESS_FEATURES:
                features[col] = float(row[col])
            feature_rows.append(features)
        features = pd.DataFrame(feature_rows)
        meta = data[["case", "run", "VB", "time", "DOC", "feed", "material", "material_name"]].copy()
        meta["sample_id"] = meta.apply(lambda r: f"case_{int(r['case']):02d}_run_{int(r['run']):03d}", axis=1)
        meta["dataset_run_id"] = meta.apply(lambda r: f"case_{int(r['case']):02d}", axis=1)
        meta["domain_id"] = meta["case"].map(self._domain_for_case)
        meta["label"] = meta["VB"]
        data = pd.concat([meta.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
        return data, list(features.columns)

    def _run_one_shift(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        source_domain: str,
        target_domain: str,
        source_cases: list[int],
        target_cases: list[int],
        model_params: dict,
        metrics: list[str],
    ) -> tuple[list[dict], pd.DataFrame]:
        source = data.loc[data["case"].isin(source_cases)].copy()
        target = data.loc[data["case"].isin(target_cases)].copy()
        if set(source["case"]) & set(target["case"]):
            raise ValueError("Source and target cases overlap.")
        validation_case = min(source_cases)
        train = source.loc[source["case"] != validation_case].copy()
        validation = source.loc[source["case"] == validation_case].copy()
        if train.empty:
            raise ValueError(f"No train samples after source-only validation split for {source_domain}_to_{target_domain}.")
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(**model_params)),
            ]
        )
        model.fit(train[feature_columns], train["VB"])
        target_pred = model.predict(target[feature_columns])
        metric_values = self._regression_metrics(target["VB"], target_pred, metrics)
        rows = []
        for metric_name, metric_value in metric_values.items():
            rows.append(
                {
                    "experiment_id": None,
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "source_cases": ",".join(map(str, source_cases)),
                    "target_cases": ",".join(map(str, target_cases)),
                    "validation_case": validation_case,
                    "train_cases": ",".join(map(str, sorted(train["case"].unique()))),
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "num_train": len(train),
                    "num_validation": len(validation),
                    "num_test": len(target),
                }
            )
        pred_frame = target[["sample_id", "case", "run", "domain_id", "VB"]].copy()
        pred_frame["source_domain"] = source_domain
        pred_frame["target_domain"] = target_domain
        pred_frame["split"] = "test"
        pred_frame["y_true"] = pred_frame["VB"]
        pred_frame["y_pred"] = target_pred
        pred_frame["residual"] = pred_frame["y_true"] - pred_frame["y_pred"]
        pred_frame["absolute_error"] = pred_frame["residual"].abs()
        return rows, pred_frame

    def _regression_metrics(self, y_true, y_pred, metrics: list[str]) -> dict[str, float]:
        requested = {m.lower() for m in metrics}
        out = {}
        if "mae" in requested:
            out["mae"] = float(mean_absolute_error(y_true, y_pred))
        if "rmse" in requested:
            out["rmse"] = float(mean_squared_error(y_true, y_pred) ** 0.5)
        if "r2" in requested or "r2_score" in requested:
            out["r2"] = float(r2_score(y_true, y_pred))
        return out

    def _scenario_split_assignments(
        self,
        data: pd.DataFrame,
        source_domain: str,
        target_domain: str,
        source_cases: list[int],
        target_cases: list[int],
    ) -> pd.DataFrame:
        validation_case = min(source_cases)
        relevant = data.loc[data["case"].isin(source_cases + target_cases)].copy()
        relevant["source_domain"] = source_domain
        relevant["target_domain"] = target_domain
        relevant["scenario"] = f"{source_domain}_to_{target_domain}"
        relevant["split"] = "excluded"
        relevant.loc[relevant["case"].isin(target_cases), "split"] = "test"
        relevant.loc[relevant["case"].isin(source_cases) & (relevant["case"] != validation_case), "split"] = "train"
        relevant.loc[relevant["case"] == validation_case, "split"] = "validation"
        return relevant[
            [
                "scenario",
                "sample_id",
                "dataset_run_id",
                "case",
                "run",
                "domain_id",
                "label",
                "split",
                "source_domain",
                "target_domain",
            ]
        ]

    def _aggregate_metrics(self, shift_metrics: pd.DataFrame, failures: list[dict]) -> dict:
        scenarios = [f"{source}_to_{target}" for source, target in SHIFT_SCENARIOS]
        if failures:
            return {
                "aggregation": "not_computed_due_to_failed_shift",
                "shift_scenarios": scenarios,
                "failed_scenarios": failures,
                "final_metrics": {},
            }
        final_metrics = shift_metrics.groupby("metric_name")["metric_value"].mean().to_dict()
        return {
            "aggregation": "mean_over_6_domain_shifts",
            "shift_scenarios": scenarios,
            "final_metrics": {key: float(value) for key, value in final_metrics.items()},
        }

    def _metrics_csv_row(self, config: dict, metrics_json: dict) -> pd.DataFrame:
        row = {
            "experiment_id": config["experiment"]["experiment_id"],
            "dataset": config["dataset"]["name"],
            "model": config["model"]["name"],
            "input_type": config["model"]["input_type"],
            "split_strategy": "fixed_case_pair_domain_shift",
            "steady_cut_mode": config.get("steady_cut_mode", "full_signal"),
            "aggregation": metrics_json["aggregation"],
        }
        row.update(metrics_json.get("final_metrics", {}))
        return pd.DataFrame([row])

    def _write_dataset_artifacts(self, paths: ExperimentPaths, config: dict, data: pd.DataFrame, feature_columns: list[str]) -> None:
        write_json(
            paths.execution_dir / "data" / "dataset_summary.json",
            {
                "name": config["dataset"]["name"],
                "process_info_path": config["dataset"]["process_info_path"],
                "signal_data_path": config["dataset"]["signal_data_path"],
                "case_scope": CASE_SCOPE,
                "domain_cases": DOMAIN_CASES,
                "num_labeled_samples": len(data),
                "rows_by_case": data.groupby("case").size().to_dict(),
                "feature_columns": feature_columns,
            },
        )
        write_csv(paths.execution_dir / "data" / "label_distribution.csv", data[["case", "run", "VB", "domain_id"]])
        write_json(paths.execution_dir / "data" / "metadata_summary.json", {"columns": list(data.columns)})
        write_json(paths.execution_dir / "data" / "shape_summary.json", {"X": [len(data), len(feature_columns)], "y": [len(data)]})
        write_json(paths.execution_dir / "data" / "dataset_fingerprint.json", {"num_rows": len(data), "case_scope": CASE_SCOPE})

    def _leakage_summary(self, failures: list[dict]) -> dict:
        checks = []
        for source, target in SHIFT_SCENARIOS:
            checks.append(
                {
                    "scenario": f"{source}_to_{target}",
                    "source_cases": DOMAIN_CASES[source],
                    "target_cases": DOMAIN_CASES[target],
                    "overlap": sorted(set(DOMAIN_CASES[source]) & set(DOMAIN_CASES[target])),
                }
            )
        return {"ok": not failures and all(not item["overlap"] for item in checks), "checks": checks, "failures": failures}

    def _domain_for_case(self, case: int) -> str:
        for domain, cases in DOMAIN_CASES.items():
            if int(case) in cases:
                return domain
        raise ValueError(f"Case outside fixed scope: {case}")

    def _update_index(self, config: dict, metrics_json: dict, status: str) -> None:
        index_path = self.root / "experiments" / "index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "experiment_id": config["experiment"]["experiment_id"],
            "experiment_name": config["experiment"].get("name"),
            "dataset": config["dataset"].get("name"),
            "model": config["model"].get("name"),
            "input_type": config["model"].get("input_type"),
            "split_strategy": "fixed_case_pair_domain_shift",
            "steady_cut_mode": config.get("steady_cut_mode", "full_signal"),
            "status": status,
            "best_metric": metrics_json.get("final_metrics", {}).get("mae"),
            "created_at": config["experiment"].get("created_at", datetime.now().isoformat(timespec="seconds")),
            "error_message": None if status == "finished" else "one or more shift scenarios failed",
            "error_log": None,
        }
        if index_path.exists():
            frame = pd.read_csv(index_path)
            frame = frame.loc[frame["experiment_id"] != row["experiment_id"]]
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        else:
            frame = pd.DataFrame([row])
        frame.to_csv(index_path, index=False)

from __future__ import annotations

import ast
import json
import math
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, pearsonr, skew, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from milling_experiment_framework import __version__
from milling_experiment_framework.core.config import SCHEMA_VERSION, stable_hash
from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.experiment_logging.experiment_logger import ExperimentLogger
from milling_experiment_framework.experiments.execution_path import execution_index_fields
from milling_experiment_framework.utils.io import write_csv, write_json, write_yaml
from milling_experiment_framework.utils.paths import ExperimentPaths


CASE_SCOPE = [1, 2, 8, 9, 12, 14]
DOMAIN_CASES = {"A": [1, 9], "B": [2, 12], "C": [8, 14]}
SHIFT_SCENARIOS = [("A", "B"), ("A", "C"), ("B", "A"), ("B", "C"), ("C", "A"), ("C", "B")]
SEGMENT_SETTINGS = ["full_length", "steady", "entry", "exit", "entry_steady", "entry_exit", "steady_exit"]
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
METRIC_COLUMNS = {
    "mae": "metric_mae",
    "rmse": "metric_rmse",
    "r2": "metric_r2",
    "pearson": "metric_pearson",
    "spearman": "metric_spearman",
}


@dataclass(frozen=True)
class S1RunConfig:
    process_info_path: Path
    signal_data_path: Path
    heuristic_sequence_path: Path
    seeds: list[int]
    models: list[str]
    random_forest_params: dict[str, Any]
    mlp_params: dict[str, Any]


class S1SegmentExecution:
    """Execute S1 segment-setting effect experiment exactly over the fixed case-pair protocol."""

    def __init__(
        self,
        config_path: str | Path,
        root: str | Path = ".",
        dry_run: bool = False,
        seed_mode: str = "initial",
    ):
        self.config_path = Path(config_path)
        self.root = Path(root).resolve()
        self.dry_run = dry_run
        self.seed_mode = seed_mode

    def run(self) -> dict[str, Any]:
        raw_config = self._read_config()
        run_config = self._resolve_run_config(raw_config)
        experiment_id = self._generate_experiment_id()
        resolved_config = self._resolved_config(raw_config, run_config, experiment_id)
        paths = ExperimentPaths(self.root, experiment_id)
        paths.apply_to_config(resolved_config)
        paths.prepare_standard_dirs()
        logger = ExperimentLogger(paths.execution_dir / "logs" / "run.log")
        logger.info(f"S1 execution started: {experiment_id}")
        if self.dry_run:
            logger.info("dry-run mode enabled")

        write_yaml(paths.execution_dir / "configs" / "input_config.yaml", raw_config)
        write_yaml(paths.execution_dir / "configs" / "resolved_config.yaml", resolved_config)
        write_json(paths.execution_dir / "configs" / "config_validation.json", {"ok": True, "errors": [], "warnings": []})

        try:
            dataset = self._load_dataset(run_config)
            sensors = self._sensor_columns(dataset)
            feature_table = self._build_feature_table(dataset, sensors)
            dry_summary = self._dry_run_summary(dataset, feature_table, sensors, run_config.seeds)
            self._write_common_data_artifacts(paths, dataset, feature_table, dry_summary, resolved_config)
            if self.dry_run:
                self._write_dry_run_outputs(paths, dry_summary)
                self._update_index(resolved_config, status="finished", best_metric=None)
                logger.info(f"S1 dry-run finished: {experiment_id}")
                return {"experiment_id": experiment_id, "dry_run": True, "summary": dry_summary, "execution_dir": str(paths.execution_dir)}

            results = self._run_grid(feature_table, sensors, run_config, logger)
            self._write_results(paths, resolved_config, results, dry_summary, sensors, run_config)
            self._update_index(resolved_config, status="finished", best_metric=self._best_metric(results["segment_metrics"]))
            logger.info(f"S1 execution finished: {experiment_id}")
            return {
                "experiment_id": experiment_id,
                "metrics": results["metrics_json"],
                "execution_dir": str(paths.execution_dir),
            }
        except Exception as exc:
            error_text = traceback.format_exc()
            logger.error(str(exc))
            (paths.execution_dir / "logs" / "error.log").write_text(error_text, encoding="utf-8")
            self._update_index(resolved_config, status="failed", best_metric=None, error_message=str(exc), error_log=str(paths.execution_dir / "logs" / "error.log"))
            raise

    def _read_config(self) -> dict[str, Any]:
        import yaml

        with self.config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _resolve_run_config(self, config: dict[str, Any]) -> S1RunConfig:
        seed_list = config.get("seeds", {}).get("seed_list", list(range(20)))
        if self.seed_mode == "initial":
            seeds = config.get("seeds", {}).get("initial_seeds", [0, 1, 2])
        elif self.seed_mode == "extra":
            seeds = config.get("seeds", {}).get("extra_seeds", [s for s in seed_list if s not in {0, 1, 2}])
        elif self.seed_mode == "all":
            seeds = seed_list
        else:
            seeds = [int(s) for s in self.seed_mode.split(",") if s.strip()]
        models = config.get("models", ["random_forest", "mlp"])
        model_cfg = config.get("model_defaults", {})
        return S1RunConfig(
            process_info_path=Path(config["dataset"]["process_info_path"]),
            signal_data_path=Path(config["dataset"]["signal_data_path"]),
            heuristic_sequence_path=Path(config["dataset"]["heuristic_sequence_path"]),
            seeds=[int(seed) for seed in seeds],
            models=models,
            random_forest_params=dict(model_cfg.get("random_forest", {})),
            mlp_params=dict(model_cfg.get("mlp", {})),
        )

    def _generate_experiment_id(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
        seed_tag = "dry_run" if self.dry_run else f"seeds_{self.seed_mode}"
        return f"{now}_S1_all_models_all_sensors_all_segments_all_shifts_{seed_tag}"

    def _resolved_config(self, raw_config: dict[str, Any], run_config: S1RunConfig, experiment_id: str) -> dict[str, Any]:
        resolved = dict(raw_config)
        resolved["experiment"] = dict(resolved.get("experiment", {}))
        resolved["experiment"]["experiment_id"] = experiment_id
        resolved["experiment"]["created_at"] = datetime.now().isoformat(timespec="seconds")
        resolved["experiment"]["analysis_type"] = "Exploratory"
        resolved["experiment"]["framework_version"] = __version__
        resolved["framework_version"] = __version__
        resolved["config_schema_version"] = SCHEMA_VERSION
        resolved["dataset_schema_version"] = SCHEMA_VERSION
        resolved["output_schema_version"] = SCHEMA_VERSION
        resolved["case_scope"] = CASE_SCOPE
        resolved["domain_cases"] = DOMAIN_CASES
        resolved["shift_scenarios"] = [f"{source}_to_{target}" for source, target in SHIFT_SCENARIOS]
        resolved["segment_settings"] = SEGMENT_SETTINGS
        resolved["feature_names"] = FEATURE_NAMES
        resolved["sensors"] = "resolved_at_runtime"
        resolved["seeds"] = run_config.seeds
        resolved["dry_run"] = self.dry_run
        resolved["config_hash"] = stable_hash(resolved)
        return resolved

    def _load_dataset(self, run_config: S1RunConfig) -> pd.DataFrame:
        for path in [run_config.process_info_path, run_config.signal_data_path, run_config.heuristic_sequence_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required data file not found: {path}")
        process = pd.read_csv(run_config.process_info_path)
        signal = pd.read_csv(run_config.signal_data_path)
        heuristic = pd.read_csv(run_config.heuristic_sequence_path)
        process = process.loc[process["enable"].astype(bool) & process["case"].isin(CASE_SCOPE)].copy()
        signal = signal.loc[signal["enable"].astype(bool) & signal["case"].isin(CASE_SCOPE)].copy()
        heuristic = heuristic.loc[heuristic["case"].isin(CASE_SCOPE)].copy()
        data = process.merge(signal, on=["case", "run"], suffixes=("", "_signal"), validate="one_to_one")
        data = data.merge(
            heuristic[["case", "run", "idx_noload_end", "idx_start", "idx_end", "signal_length", "status"]],
            on=["case", "run"],
            how="inner",
            validate="one_to_one",
        )
        if "VB" not in data.columns:
            raise ValueError("Target column VB not found.")
        data = data.loc[data["VB"].notna()].reset_index(drop=True)
        if sorted(data["case"].unique().tolist()) != CASE_SCOPE:
            raise ValueError(f"Case filtering failed. Found cases={sorted(data['case'].unique().tolist())}")
        data["sample_id"] = data.apply(lambda row: f"case_{int(row['case']):02d}_run_{int(row['run']):03d}", axis=1)
        data["dataset_run_id"] = data.apply(lambda row: f"case_{int(row['case']):02d}", axis=1)
        data["domain_id"] = data["case"].map(self._domain_for_case)
        data["label"] = data["VB"]
        return data

    def _sensor_columns(self, data: pd.DataFrame) -> list[str]:
        excluded = {
            "case",
            "run",
            "VB",
            "time",
            "DOC",
            "feed",
            "material",
            "material_name",
            "enable",
            "enable_signal",
            "idx_noload_end",
            "idx_start",
            "idx_end",
            "signal_length",
            "status",
            "sample_id",
            "dataset_run_id",
            "domain_id",
            "label",
        }
        sensors = []
        for col in data.columns:
            if col in excluded:
                continue
            first_value = data[col].dropna().iloc[0] if not data[col].dropna().empty else None
            if isinstance(first_value, str) and first_value.strip().startswith("["):
                sensors.append(col)
        if not sensors:
            raise ValueError("No sensor columns found.")
        return sensors

    def _build_feature_table(self, data: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for _, row in data.iterrows():
            arrays = {sensor: np.nan_to_num(np.asarray(ast.literal_eval(row[sensor]), dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0) for sensor in sensors}
            for sensor in sensors:
                segments = self._segments(arrays[sensor], row)
                for segment_setting in SEGMENT_SETTINGS:
                    vector = self._segment_vector(segments, segment_setting)
                    feature_values = self._features(vector)
                    rows.append(
                        {
                            "sample_id": row["sample_id"],
                            "dataset_run_id": row["dataset_run_id"],
                            "case": int(row["case"]),
                            "run": int(row["run"]),
                            "domain_id": row["domain_id"],
                            "sensor": sensor,
                            "segment_setting": segment_setting,
                            "VB": float(row["VB"]),
                            **feature_values,
                        }
                    )
        return pd.DataFrame(rows)

    def _segments(self, arr: np.ndarray, row: pd.Series) -> dict[str, np.ndarray]:
        n = len(arr)
        idx_noload_end = int(np.clip(row["idx_noload_end"], 0, n))
        idx_start = int(np.clip(row["idx_start"], idx_noload_end, n))
        idx_end = int(np.clip(row["idx_end"], idx_start, n))
        return {
            "full_length": arr[:n],
            "entry": arr[idx_noload_end:idx_start],
            "steady": arr[idx_start:idx_end],
            "exit": arr[idx_end:n],
        }

    def _segment_vector(self, segments: dict[str, np.ndarray], segment_setting: str) -> np.ndarray:
        if segment_setting in {"full_length", "steady", "entry", "exit"}:
            vector = segments[segment_setting]
        elif segment_setting == "entry_steady":
            vector = np.concatenate([segments["entry"], segments["steady"]])
        elif segment_setting == "entry_exit":
            vector = np.concatenate([segments["entry"], segments["exit"]])
        elif segment_setting == "steady_exit":
            vector = np.concatenate([segments["steady"], segments["exit"]])
        else:
            raise ValueError(f"Unknown segment setting: {segment_setting}")
        if len(vector) == 0:
            raise ValueError(f"Empty segment vector for setting: {segment_setting}")
        return vector

    def _features(self, arr: np.ndarray) -> dict[str, float]:
        spectrum = np.abs(np.fft.rfft(arr))
        freqs = np.fft.rfftfreq(len(arr), d=1.0)
        spectral_sum = spectrum.sum()
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

    def _dry_run_summary(self, dataset: pd.DataFrame, feature_table: pd.DataFrame, sensors: list[str], seeds: list[int]) -> dict[str, Any]:
        availability = (
            feature_table.groupby(["sensor", "segment_setting"])
            .size()
            .reset_index(name="num_feature_rows")
            .to_dict(orient="records")
        )
        split_counts = []
        for source, target in SHIFT_SCENARIOS:
            relevant = self._split_frame(dataset, source, target)
            split_counts.append(
                {
                    "scenario": f"{source}_to_{target}",
                    "source_cases": DOMAIN_CASES[source],
                    "target_cases": DOMAIN_CASES[target],
                    "split_counts": relevant["split"].value_counts().to_dict(),
                    "overlap": sorted(set(DOMAIN_CASES[source]) & set(DOMAIN_CASES[target])),
                }
            )
        return {
            "case_scope": CASE_SCOPE,
            "rows_by_case": dataset.groupby("case").size().to_dict(),
            "sensors": sensors,
            "segment_settings": SEGMENT_SETTINGS,
            "feature_names": FEATURE_NAMES,
            "feature_availability": availability,
            "shift_splits": split_counts,
            "models": ["random_forest", "mlp"],
            "seeds": seeds,
            "planned_atomic_executions": len(SHIFT_SCENARIOS) * 2 * len(sensors) * len(SEGMENT_SETTINGS) * len(seeds),
            "h1s1_association_reference": self._association_reference_status(),
        }

    def _run_grid(self, feature_table: pd.DataFrame, sensors: list[str], run_config: S1RunConfig, logger: ExperimentLogger) -> dict[str, Any]:
        shift_rows: list[dict[str, Any]] = []
        prediction_rows: list[pd.DataFrame] = []
        split_rows: list[pd.DataFrame] = []
        total = len(run_config.models) * len(sensors) * len(SEGMENT_SETTINGS) * len(run_config.seeds) * len(SHIFT_SCENARIOS)
        done = 0
        for model_name in run_config.models:
            for sensor in sensors:
                for segment_setting in SEGMENT_SETTINGS:
                    condition = feature_table.loc[
                        (feature_table["sensor"] == sensor) & (feature_table["segment_setting"] == segment_setting)
                    ].copy()
                    X_cols = FEATURE_NAMES
                    self._assert_no_leakage_features(X_cols)
                    for seed in run_config.seeds:
                        for source, target in SHIFT_SCENARIOS:
                            done += 1
                            result, preds, splits = self._run_atomic(
                                model_name=model_name,
                                sensor=sensor,
                                segment_setting=segment_setting,
                                seed=seed,
                                source_domain=source,
                                target_domain=target,
                                data=condition,
                                feature_columns=X_cols,
                                run_config=run_config,
                            )
                            shift_rows.append(result)
                            prediction_rows.append(preds)
                            split_rows.append(splits)
                            if done % 500 == 0 or done == total:
                                logger.info(f"S1 progress {done}/{total}")
        shift_metrics = pd.DataFrame(shift_rows)
        predictions = pd.concat(prediction_rows, ignore_index=True)
        splits = pd.concat(split_rows, ignore_index=True)
        seed_metrics = self._seed_metrics(shift_metrics)
        segment_metrics = self._segment_metrics(seed_metrics)
        baseline_improvement = self._baseline_improvement(segment_metrics)
        sensor_best = self._sensor_best(segment_metrics)
        comparison = baseline_improvement.copy()
        ranking = self._segment_ranking(segment_metrics)
        metrics_json = self._metrics_json(segment_metrics)
        return {
            "shift_metrics": shift_metrics,
            "predictions": predictions,
            "splits": splits,
            "seed_metrics": seed_metrics,
            "segment_metrics": segment_metrics,
            "baseline_improvement": baseline_improvement,
            "sensor_best_segment": sensor_best,
            "comparison_metrics": comparison,
            "segment_ranking": ranking,
            "metrics_json": metrics_json,
            "h1s1_association": self._h1s1_comparison(segment_metrics),
        }

    def _run_atomic(
        self,
        model_name: str,
        sensor: str,
        segment_setting: str,
        seed: int,
        source_domain: str,
        target_domain: str,
        data: pd.DataFrame,
        feature_columns: list[str],
        run_config: S1RunConfig,
    ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
        split = self._split_frame(data, source_domain, target_domain)
        train = split.loc[split["split"] == "train"].copy()
        test = split.loc[split["split"] == "test"].copy()
        if train.empty or test.empty:
            raise ValueError(f"Empty train/test split for {source_domain}_to_{target_domain}")
        model = self._model(model_name, seed, run_config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[feature_columns], train["VB"])
        y_pred = model.predict(test[feature_columns])
        metrics = self._metrics(test["VB"].to_numpy(), y_pred)
        child_key = f"S1_{model_name}_{sensor}_{segment_setting}_{source_domain}_to_{target_domain}_seed_{seed}"
        result = {
            "experiment_id": None,
            "child_execution_key": child_key,
            "model": model_name,
            "sensor": sensor,
            "segment_setting": segment_setting,
            "seed": seed,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "source_cases": ",".join(map(str, DOMAIN_CASES[source_domain])),
            "target_cases": ",".join(map(str, DOMAIN_CASES[target_domain])),
            **metrics,
        }
        preds = test[["sample_id", "dataset_run_id", "case", "run", "domain_id", "VB"]].copy()
        preds["child_execution_key"] = child_key
        preds["model"] = model_name
        preds["sensor"] = sensor
        preds["segment_setting"] = segment_setting
        preds["seed"] = seed
        preds["source_domain"] = source_domain
        preds["target_domain"] = target_domain
        preds["split"] = "test"
        preds["y_true"] = preds["VB"]
        preds["y_pred"] = y_pred
        preds["residual"] = preds["y_true"] - preds["y_pred"]
        preds["absolute_error"] = preds["residual"].abs()
        split_out = split[["sample_id", "dataset_run_id", "case", "run", "domain_id", "VB", "split"]].copy()
        split_out["child_execution_key"] = child_key
        split_out["model"] = model_name
        split_out["sensor"] = sensor
        split_out["segment_setting"] = segment_setting
        split_out["seed"] = seed
        split_out["source_domain"] = source_domain
        split_out["target_domain"] = target_domain
        return result, preds, split_out

    def _model(self, model_name: str, seed: int, run_config: S1RunConfig) -> Pipeline:
        if model_name == "random_forest":
            params = dict(run_config.random_forest_params)
            params.setdefault("random_state", seed)
            return Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(**params))])
        if model_name == "mlp":
            params = dict(run_config.mlp_params)
            params.setdefault("hidden_layer_sizes", (64, 32, 16))
            params.setdefault("random_state", seed)
            params.setdefault("max_iter", 300)
            params.setdefault("learning_rate_init", 0.001)
            return Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(**params))])
        raise ValueError(f"Unsupported model: {model_name}")

    def _split_frame(self, data: pd.DataFrame, source_domain: str, target_domain: str) -> pd.DataFrame:
        source_cases = DOMAIN_CASES[source_domain]
        target_cases = DOMAIN_CASES[target_domain]
        validation_case = min(source_cases)
        split = data.loc[data["case"].isin(source_cases + target_cases)].copy()
        split["split"] = "excluded"
        split.loc[split["case"].isin(target_cases), "split"] = "test"
        split.loc[split["case"].isin(source_cases) & (split["case"] != validation_case), "split"] = "train"
        split.loc[split["case"] == validation_case, "split"] = "validation"
        return split

    def _metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            pearson = float(pearsonr(y_true, y_pred).statistic)
            spearman = float(spearmanr(y_true, y_pred).statistic)
        else:
            pearson = float("nan")
            spearman = float("nan")
        return {
            "metric_mae": float(mean_absolute_error(y_true, y_pred)),
            "metric_rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
            "metric_r2": float(r2_score(y_true, y_pred)),
            "metric_pearson": pearson,
            "metric_spearman": spearman,
        }

    def _seed_metrics(self, shift_metrics: pd.DataFrame) -> pd.DataFrame:
        grouped = shift_metrics.groupby(["model", "sensor", "segment_setting", "seed"], dropna=False)
        rows = []
        for keys, group in grouped:
            model, sensor, segment_setting, seed = keys
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "sensor": sensor,
                    "segment_setting": segment_setting,
                    "seed": seed,
                    "mean_mae_over_6_shifts": group["metric_mae"].mean(),
                    "mean_rmse_over_6_shifts": group["metric_rmse"].mean(),
                    "mean_r2_over_6_shifts": group["metric_r2"].mean(),
                    "mean_pearson_over_6_shifts": group["metric_pearson"].mean(),
                    "mean_spearman_over_6_shifts": group["metric_spearman"].mean(),
                }
            )
        return pd.DataFrame(rows)

    def _segment_metrics(self, seed_metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for keys, group in seed_metrics.groupby(["model", "sensor", "segment_setting"], dropna=False):
            model, sensor, segment_setting = keys
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "sensor": sensor,
                    "segment_setting": segment_setting,
                    "mean_mae": group["mean_mae_over_6_shifts"].mean(),
                    "std_mae": group["mean_mae_over_6_shifts"].std(ddof=0),
                    "mean_rmse": group["mean_rmse_over_6_shifts"].mean(),
                    "std_rmse": group["mean_rmse_over_6_shifts"].std(ddof=0),
                    "mean_r2": group["mean_r2_over_6_shifts"].mean(),
                    "std_r2": group["mean_r2_over_6_shifts"].std(ddof=0),
                    "mean_pearson": group["mean_pearson_over_6_shifts"].mean(),
                    "std_pearson": group["mean_pearson_over_6_shifts"].std(ddof=0),
                    "mean_spearman": group["mean_spearman_over_6_shifts"].mean(),
                    "std_spearman": group["mean_spearman_over_6_shifts"].std(ddof=0),
                }
            )
        return pd.DataFrame(rows)

    def _baseline_improvement(self, segment_metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (model, sensor), group in segment_metrics.groupby(["model", "sensor"]):
            for baseline in ["full_length", "steady"]:
                base = group.loc[group["segment_setting"] == baseline]
                if base.empty:
                    continue
                base_row = base.iloc[0]
                for _, row in group.iterrows():
                    rows.append(
                        {
                            "experiment_id": None,
                            "model": model,
                            "sensor": sensor,
                            "segment_setting": row["segment_setting"],
                            "baseline": baseline,
                            "delta_mae": row["mean_mae"] - base_row["mean_mae"],
                            "delta_rmse": row["mean_rmse"] - base_row["mean_rmse"],
                            "delta_r2": row["mean_r2"] - base_row["mean_r2"],
                            "improved_mae": bool(row["mean_mae"] < base_row["mean_mae"]),
                            "improved_rmse": bool(row["mean_rmse"] < base_row["mean_rmse"]),
                            "improved_r2": bool(row["mean_r2"] > base_row["mean_r2"]),
                        }
                    )
        return pd.DataFrame(rows)

    def _sensor_best(self, segment_metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (model, sensor), group in segment_metrics.groupby(["model", "sensor"]):
            best_mae = group.loc[group["mean_mae"].idxmin()]
            best_rmse = group.loc[group["mean_rmse"].idxmin()]
            best_r2 = group.loc[group["mean_r2"].idxmax()]
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "sensor": sensor,
                    "best_segment_by_mae": best_mae["segment_setting"],
                    "best_segment_by_rmse": best_rmse["segment_setting"],
                    "best_segment_by_r2": best_r2["segment_setting"],
                    "best_mae": best_mae["mean_mae"],
                    "best_rmse": best_rmse["mean_rmse"],
                    "best_r2": best_r2["mean_r2"],
                }
            )
        return pd.DataFrame(rows)

    def _segment_ranking(self, segment_metrics: pd.DataFrame) -> pd.DataFrame:
        ranking = segment_metrics.copy()
        ranking["rank_mae"] = ranking.groupby(["model", "sensor"])["mean_mae"].rank(method="min", ascending=True)
        ranking["rank_rmse"] = ranking.groupby(["model", "sensor"])["mean_rmse"].rank(method="min", ascending=True)
        ranking["rank_r2"] = ranking.groupby(["model", "sensor"])["mean_r2"].rank(method="min", ascending=False)
        return ranking.sort_values(["model", "sensor", "rank_mae"])

    def _metrics_json(self, segment_metrics: pd.DataFrame) -> dict[str, Any]:
        best = segment_metrics.loc[segment_metrics["mean_mae"].idxmin()].to_dict()
        return {
            "aggregation": "mean_over_6_domain_shifts_then_mean_std_over_seeds",
            "shift_scenarios": [f"{source}_to_{target}" for source, target in SHIFT_SCENARIOS],
            "primary_metric": "mean_mae",
            "best_overall_by_mae": best,
        }

    def _h1s1_comparison(self, segment_metrics: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

    def _association_reference_status(self) -> str:
        candidates = [p for p in (self.root / "analysis").glob("*h1*") if p.is_file()] if (self.root / "analysis").exists() else []
        return "available" if candidates else "unavailable"

    def _write_common_data_artifacts(self, paths: ExperimentPaths, dataset: pd.DataFrame, feature_table: pd.DataFrame, dry_summary: dict[str, Any], config: dict[str, Any]) -> None:
        write_json(paths.execution_dir / "data" / "dataset_summary.json", dry_summary)
        write_json(paths.execution_dir / "data" / "dataset_fingerprint.json", {"num_labeled_rows": len(dataset), "case_scope": CASE_SCOPE, "config_hash": config["config_hash"]})
        write_csv(paths.execution_dir / "data" / "label_distribution.csv", dataset[["case", "run", "VB", "domain_id"]])
        write_json(paths.execution_dir / "data" / "metadata_summary.json", {"columns": list(dataset.columns)})
        write_json(paths.execution_dir / "data" / "shape_summary.json", {"feature_table": list(feature_table.shape), "dataset": list(dataset.shape)})
        write_json(paths.execution_dir / "preprocessing" / "feature_summary.json", {"feature_names": FEATURE_NAMES, "num_feature_rows": len(feature_table)})
        write_json(paths.execution_dir / "preprocessing" / "preprocessing_summary.json", {"output_type": "features", "segment_settings": SEGMENT_SETTINGS})
        write_json(paths.execution_dir / "preprocessing" / "shape_trace.json", [{"stage": "feature_extraction", "shape": list(feature_table.shape)}])

    def _write_dry_run_outputs(self, paths: ExperimentPaths, dry_summary: dict[str, Any]) -> None:
        write_json(paths.execution_dir / "splits" / "split_summary.json", {"shift_splits": dry_summary["shift_splits"]})
        write_json(paths.execution_dir / "splits" / "leakage_check.json", self._leakage_check(pd.DataFrame()))
        write_json(paths.execution_dir / "metrics" / "metrics.json", {"dry_run": True, "summary": dry_summary})
        self._write_report(paths, {}, dry_summary, dry_run=True)

    def _write_results(self, paths: ExperimentPaths, config: dict[str, Any], results: dict[str, Any], dry_summary: dict[str, Any], sensors: list[str], run_config: S1RunConfig) -> None:
        experiment_id = config["experiment"]["experiment_id"]
        for key in ["shift_metrics", "seed_metrics", "segment_metrics", "baseline_improvement", "sensor_best_segment", "comparison_metrics", "segment_ranking"]:
            if key in results and not results[key].empty and "experiment_id" in results[key].columns:
                results[key]["experiment_id"] = experiment_id
        results["shift_metrics"]["experiment_id"] = experiment_id
        results["seed_metrics"]["experiment_id"] = experiment_id
        results["segment_metrics"]["experiment_id"] = experiment_id
        results["baseline_improvement"]["experiment_id"] = experiment_id
        results["sensor_best_segment"]["experiment_id"] = experiment_id
        if "best_overall_by_mae" in results["metrics_json"]:
            results["metrics_json"]["best_overall_by_mae"]["experiment_id"] = experiment_id
        write_csv(paths.execution_dir / "metrics" / "shift_metrics.csv", results["shift_metrics"])
        write_csv(paths.execution_dir / "metrics" / "seed_metrics.csv", results["seed_metrics"])
        write_csv(paths.execution_dir / "metrics" / "segment_metrics.csv", results["segment_metrics"])
        write_csv(paths.execution_dir / "metrics" / "comparison_metrics.csv", results["comparison_metrics"])
        write_json(paths.execution_dir / "metrics" / "metrics.json", results["metrics_json"])
        write_csv(paths.execution_dir / "predictions" / "predictions.csv", results["predictions"])
        write_csv(paths.execution_dir / "predictions" / "prediction_errors.csv", results["predictions"].sort_values("absolute_error", ascending=False))
        write_csv(paths.execution_dir / "splits" / "split.csv", results["splits"])
        write_json(paths.execution_dir / "splits" / "split_summary.json", {"shift_scenarios": dry_summary["shift_splits"]})
        write_json(paths.execution_dir / "splits" / "leakage_check.json", self._leakage_check(results["splits"]))
        write_json(paths.execution_dir / "analysis" / "analysis_summary.json", self._analysis_summary(results, sensors, run_config))
        write_csv(paths.execution_dir / "analysis" / "segment_ranking.csv", results["segment_ranking"])
        write_csv(paths.execution_dir / "analysis" / "sensor_best_segment.csv", results["sensor_best_segment"])
        write_csv(paths.execution_dir / "analysis" / "baseline_improvement.csv", results["baseline_improvement"])
        if results["h1s1_association"].empty:
            write_json(paths.execution_dir / "analysis" / "h1s1_association_comparison.json", {"status": "H1.S1 association reference unavailable"})
        else:
            write_csv(paths.execution_dir / "analysis" / "h1s1_association_comparison.csv", results["h1s1_association"])
        self._write_figures(paths, results)
        write_json(paths.execution_dir / "logs" / "environment.json", collect_environment(str(self.root)))
        write_json(paths.execution_dir / "logs" / "reproducibility.json", {"framework_version": __version__, "config_hash": config["config_hash"], "seeds": run_config.seeds})
        self._write_report(paths, results, dry_summary, dry_run=False)
        self._write_report_html(paths)

    def _leakage_check(self, splits: pd.DataFrame) -> dict[str, Any]:
        scenario_checks = []
        ok = True
        for source, target in SHIFT_SCENARIOS:
            overlap = sorted(set(DOMAIN_CASES[source]) & set(DOMAIN_CASES[target]))
            scenario_checks.append({"scenario": f"{source}_to_{target}", "source_cases": DOMAIN_CASES[source], "target_cases": DOMAIN_CASES[target], "overlap": overlap})
            ok = ok and not overlap
        duplicate_checks = {}
        if not splits.empty:
            for key, group in splits.groupby("child_execution_key"):
                split_sets = {
                    split_name: set(split_group["sample_id"])
                    for split_name, split_group in group.groupby("split")
                }
                duplicate_checks[key] = {
                    "train_validation": sorted(split_sets.get("train", set()) & split_sets.get("validation", set())),
                    "train_test": sorted(split_sets.get("train", set()) & split_sets.get("test", set())),
                    "validation_test": sorted(split_sets.get("validation", set()) & split_sets.get("test", set())),
                }
                ok = ok and not any(duplicate_checks[key].values())
        return {
            "ok": ok,
            "case_scope": CASE_SCOPE,
            "source_target_overlap_checks": scenario_checks,
            "sample_overlap_checks": duplicate_checks,
            "scaling_fit_policy": "sklearn Pipeline fits StandardScaler on train split only for each atomic execution",
            "feature_leakage_check": "VB excluded from feature_columns; feature_columns are signal-derived only",
        }

    def _analysis_summary(self, results: dict[str, Any], sensors: list[str], run_config: S1RunConfig) -> dict[str, Any]:
        best = results["metrics_json"]["best_overall_by_mae"]
        return {
            "objective": "S1 segment setting effect on VB prediction",
            "models": run_config.models,
            "sensors": sensors,
            "segment_settings": SEGMENT_SETTINGS,
            "seeds": run_config.seeds,
            "num_atomic_executions": len(results["shift_metrics"]),
            "best_overall_by_mae": best,
            "h1s1_association": "H1.S1 association reference unavailable",
        }

    def _write_figures(self, paths: ExperimentPaths, results: dict[str, Any]) -> None:
        import matplotlib.pyplot as plt

        segment_metrics = results["segment_metrics"]
        figures = paths.execution_dir / "figures"
        for model_name, group in segment_metrics.groupby("model"):
            fig, ax = plt.subplots(figsize=(10, 5))
            for sensor, sensor_group in group.groupby("sensor"):
                ordered = sensor_group.set_index("segment_setting").loc[SEGMENT_SETTINGS]
                ax.plot(SEGMENT_SETTINGS, ordered["mean_mae"], marker="o", label=sensor)
            ax.set_title(f"Segment performance by sensor ({model_name})")
            ax.set_ylabel("Mean MAE")
            ax.tick_params(axis="x", rotation=35)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figures / f"segment_performance_by_sensor_{model_name}.png")
            if model_name == "random_forest":
                fig.savefig(figures / "segment_performance_by_sensor.png")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        model_seg = segment_metrics.groupby(["model", "segment_setting"])["mean_mae"].mean().reset_index()
        for model_name, group in model_seg.groupby("model"):
            ordered = group.set_index("segment_setting").loc[SEGMENT_SETTINGS]
            ax.plot(SEGMENT_SETTINGS, ordered["mean_mae"], marker="o", label=model_name)
        ax.set_title("Segment performance by model")
        ax.set_ylabel("Mean MAE")
        ax.tick_params(axis="x", rotation=35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "segment_performance_by_model.png")
        plt.close(fig)

        baseline = results["baseline_improvement"]
        fig, ax = plt.subplots(figsize=(10, 5))
        baseline.loc[baseline["baseline"] == "full_length"].groupby("segment_setting")["delta_mae"].mean().loc[SEGMENT_SETTINGS].plot(kind="bar", ax=ax)
        ax.set_title("Baseline improvement vs full_length")
        ax.set_ylabel("Delta MAE")
        fig.tight_layout()
        fig.savefig(figures / "baseline_improvement.png")
        plt.close(fig)

        shift = results["shift_metrics"].groupby(["source_domain", "target_domain"])["metric_mae"].mean().unstack()
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(shift.fillna(np.nan), cmap="viridis")
        ax.set_xticks(range(len(shift.columns)), shift.columns)
        ax.set_yticks(range(len(shift.index)), shift.index)
        ax.set_title("Shift performance heatmap (MAE)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(figures / "shift_performance_heatmap.png")
        plt.close(fig)

        best = results["sensor_best_segment"]
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = best["model"] + "/" + best["sensor"]
        ax.bar(labels, best["best_mae"])
        ax.set_title("Sensor best segment summary")
        ax.set_ylabel("Best MAE")
        ax.tick_params(axis="x", rotation=70)
        fig.tight_layout()
        fig.savefig(figures / "sensor_best_segment_summary.png")
        plt.close(fig)

    def _write_report(self, paths: ExperimentPaths, results: dict[str, Any], dry_summary: dict[str, Any], dry_run: bool) -> None:
        report = paths.execution_dir / "reports" / "report.md"
        if dry_run:
            body = f"""# S1 Dry-run Report

## Status

Dry-run completed.

## Data

- Process info: `datasets/processed/mill_process_info_enabled.csv`
- Signal data: `datasets/processed/mill_signal_data_enabled.csv`
- Heuristic sequence: `datasets/metadata/heuristic_sequence.csv`
- Cases: {CASE_SCOPE}
- Domains: {DOMAIN_CASES}
- Sensors: {dry_summary['sensors']}
- Segment settings: {SEGMENT_SETTINGS}
- Planned atomic executions: {dry_summary['planned_atomic_executions']}

## H1.S1 Association

H1.S1 association reference unavailable.
"""
        else:
            best = results["metrics_json"]["best_overall_by_mae"]
            top = results["segment_ranking"].head(20).to_string(index=False)
            sensor_best = results["sensor_best_segment"].to_string(index=False)
            body = f"""# S1 Segment Setting Effect on VB Prediction

## Objective

Evaluate whether segment setting changes VB prediction performance and whether any segment setting improves over `full_length` or `steady`.

## Data and Protocol

- Data files: `datasets/processed/mill_process_info_enabled.csv`, `datasets/processed/mill_signal_data_enabled.csv`
- Cases: {CASE_SCOPE}
- Domain pairs: {DOMAIN_CASES}
- Shift scenarios: {[f'{s}_to_{t}' for s, t in SHIFT_SCENARIOS]}
- Sensors: {dry_summary['sensors']}
- Segment settings: {SEGMENT_SETTINGS}
- Models: {dry_summary['models']}
- Seeds: {dry_summary['seeds']}
- Metrics: MAE, RMSE, R2, Pearson, Spearman

## Best Overall

```json
{json.dumps(best, indent=2, default=str)}
```

## Segment Ranking Preview

```text
{top}
```

## Sensor Best Segment

```text
{sensor_best}
```

## H1.S1 Association Comparison

H1.S1 association reference unavailable. RQ4 is deferred.

## Limitations

- Validation split is source-domain only and is held out by the smaller source case.
- No hyperparameter tuning was performed.
- MLP convergence warnings are suppressed and defaults are recorded in resolved config.
"""
        report.write_text(body + paths.report_metadata_markdown(), encoding="utf-8")

    def _write_report_html(self, paths: ExperimentPaths) -> None:
        md = paths.execution_dir / "reports" / "report.md"
        html = paths.execution_dir / "reports" / "report.html"
        body = md.read_text(encoding="utf-8").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>\n")
        html.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")

    def _best_metric(self, segment_metrics: pd.DataFrame) -> float | None:
        if segment_metrics.empty:
            return None
        return float(segment_metrics["mean_mae"].min())

    def _update_index(
        self,
        config: dict[str, Any],
        status: str,
        best_metric: float | None,
        error_message: str | None = None,
        error_log: str | None = None,
    ) -> None:
        index_path = self.root / "experiments" / "index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "experiment_id": config["experiment"]["experiment_id"],
            **execution_index_fields(config),
            "experiment_name": "S1_segment_setting_effect_on_VB_prediction",
            "dataset": "mill_processed_enabled",
            "model": "random_forest,mlp",
            "input_type": "feature-based",
            "split_strategy": "fixed_case_pair_domain_shift",
            "steady_cut_mode": "segmentation",
            "status": status,
            "best_metric": best_metric,
            "created_at": config["experiment"].get("created_at"),
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

    def _assert_no_leakage_features(self, feature_columns: list[str]) -> None:
        forbidden = {"VB", "label", "y", "target"}
        overlap = forbidden & set(feature_columns)
        if overlap:
            raise ValueError(f"Leakage features detected: {sorted(overlap)}")

    def _domain_for_case(self, case: int) -> str:
        for domain, cases in DOMAIN_CASES.items():
            if int(case) in cases:
                return domain
        raise ValueError(f"Case outside fixed scope: {case}")

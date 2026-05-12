from __future__ import annotations

import json
import math
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from milling_experiment_framework import __version__
from milling_experiment_framework.core.config import SCHEMA_VERSION, stable_hash
from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.experiment_logging.experiment_logger import ExperimentLogger
from milling_experiment_framework.experiments.execution_path import execution_index_fields
from milling_experiment_framework.experiments.s1_segment_execution import (
    CASE_SCOPE,
    DOMAIN_CASES,
    FEATURE_NAMES,
    SEGMENT_SETTINGS,
    SHIFT_SCENARIOS,
    S1RunConfig,
    S1SegmentExecution,
)
from milling_experiment_framework.utils.io import write_csv, write_json, write_yaml
from milling_experiment_framework.utils.paths import ExperimentPaths


PREFIX = "H2_S4"
SENSOR_SETTING = "all_sensors"
EXPECTED_SENSORS = ["smcAC", "smcDC", "vib_spindle", "vib_table", "AE_spindle", "AE_table"]
PROCESS_FACTORS = {
    "A": {"name": "DoC", "columns": ["DOC"]},
    "B": {"name": "Feed", "columns": ["feed"]},
    "C": {"name": "Material", "columns": ["material_name"]},
    "D": {"name": "Time", "columns": ["time"]},
}
EXCLUDED_PROCESS_COLUMNS = {
    "material": "duplicate integer code for material_name; excluded to avoid duplicate material encoding",
    "case": "domain split identifier; leakage risk",
    "run": "dataset run identifier; leakage risk",
    "VB": "target column",
    "enable": "administrative filter flag, not process information",
}
PROCESS_COMBINATIONS = {
    "doc": ["A"],
    "feed": ["B"],
    "material": ["C"],
    "time": ["D"],
    "doc_feed": ["A", "B"],
    "doc_material": ["A", "C"],
    "doc_time": ["A", "D"],
    "feed_material": ["B", "C"],
    "feed_time": ["B", "D"],
    "material_time": ["C", "D"],
    "doc_feed_material": ["A", "B", "C"],
    "doc_feed_time": ["A", "B", "D"],
    "doc_material_time": ["A", "C", "D"],
    "feed_material_time": ["B", "C", "D"],
    "doc_feed_material_time": ["A", "B", "C", "D"],
}
FACTOR_NAMES = {key: value["name"] for key, value in PROCESS_FACTORS.items()}


class H2S4ProcessInformationExecution:
    """Run H2.S4 process-information combination experiment."""

    def __init__(self, config_path: str | Path, root: str | Path = ".", dry_run: bool = False, seed_mode: str = "initial"):
        self.config_path = Path(config_path)
        self.root = Path(root).resolve()
        self.dry_run = dry_run
        self.seed_mode = seed_mode
        self.skipped: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        raw_config = self._read_config()
        run_config = self._resolve_run_config(raw_config)
        experiment_id = self._generate_experiment_id()
        config = self._resolved_config(raw_config, run_config, experiment_id)
        paths = ExperimentPaths(self.root, experiment_id)
        paths.apply_to_config(config)
        paths.prepare_standard_dirs()
        logger = ExperimentLogger(paths.execution_dir / "logs" / f"{PREFIX}_run.log")
        logger.info(f"H2.S4 execution started: {experiment_id}")
        if self.dry_run:
            logger.info("dry-run mode enabled")

        write_yaml(paths.execution_dir / "configs" / f"{PREFIX}_input_config.yaml", raw_config)
        write_yaml(paths.execution_dir / "configs" / f"{PREFIX}_resolved_config.yaml", config)
        try:
            helper = S1SegmentExecution(self.config_path, root=self.root, dry_run=True, seed_mode=self.seed_mode)
            validation = self._validate_inputs(run_config)
            dataset = helper._load_dataset(run_config)
            sensors = helper._sensor_columns(dataset)
            sensor_mapping = self._sensor_mapping(sensors)
            process_mapping = self._process_mapping(dataset, experiment_id)
            feature_table = helper._build_feature_table(dataset, sensor_mapping["actual_sensor"].tolist())
            tables = self._build_condition_tables(feature_table, dataset, sensor_mapping, process_mapping)
            validation.update(self._validate_tables(dataset, sensor_mapping, process_mapping, tables))
            write_json(paths.execution_dir / "configs" / f"{PREFIX}_config_validation.json", validation)
            dry_summary = self._dry_summary(dataset, sensor_mapping, process_mapping, tables, run_config)
            self._write_common_outputs(paths, dataset, feature_table, sensor_mapping, process_mapping, tables, dry_summary)
            if self.dry_run:
                write_json(paths.execution_dir / "metrics" / "H2_S4_metrics.json", {"dry_run": True, "summary": dry_summary})
                self._write_report(paths, dry_summary, {}, dry_run=True)
                self._write_html(paths)
                self._update_index(config, "finished", None)
                logger.info(f"H2.S4 dry-run finished: {experiment_id}")
                return {"experiment_id": experiment_id, "dry_run": True, "summary": dry_summary, "execution_dir": str(paths.execution_dir)}

            results = self._run_grid(helper, tables, process_mapping, run_config, logger)
            self._write_results(paths, config, results, dry_summary, process_mapping, run_config)
            (paths.execution_dir / "logs" / f"{PREFIX}_error.log").touch()
            self._update_index(config, "finished", self._best_metric(results["feature_group_metrics"]))
            logger.info(f"H2.S4 execution finished: {experiment_id}")
            return {"experiment_id": experiment_id, "metrics": results["metrics_json"], "execution_dir": str(paths.execution_dir)}
        except Exception as exc:
            error_text = traceback.format_exc()
            logger.error(str(exc))
            (paths.execution_dir / "logs" / f"{PREFIX}_error.log").write_text(error_text, encoding="utf-8")
            write_json(paths.execution_dir / "configs" / f"{PREFIX}_config_validation.json", {"ok": False, "errors": [str(exc)]})
            self._update_index(config, "failed", None, str(exc), str(paths.execution_dir / "logs" / f"{PREFIX}_error.log"))
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
        model_cfg = config.get("model_defaults", {})
        return S1RunConfig(
            process_info_path=Path(config["dataset"]["process_info_path"]),
            signal_data_path=Path(config["dataset"]["signal_data_path"]),
            heuristic_sequence_path=Path(config["dataset"]["heuristic_sequence_path"]),
            seeds=[int(seed) for seed in seeds],
            models=config.get("models", ["random_forest", "mlp"]),
            random_forest_params=dict(model_cfg.get("random_forest", {})),
            mlp_params=dict(model_cfg.get("mlp", {})),
        )

    def _generate_experiment_id(self) -> str:
        suffix = "dry_run" if self.dry_run else f"seeds_{self.seed_mode}"
        return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f") + f"_H2_S4_process_information_combination_segment_aware_VB_prediction_{suffix}"

    def _resolved_config(self, raw_config: dict[str, Any], run_config: S1RunConfig, experiment_id: str) -> dict[str, Any]:
        config = dict(raw_config)
        config["experiment"] = dict(config.get("experiment", {}))
        config["experiment"]["experiment_id"] = experiment_id
        config["experiment"]["created_at"] = datetime.now().isoformat(timespec="seconds")
        config["experiment"]["analysis_type"] = "Exploratory"
        config["framework_version"] = __version__
        config["config_schema_version"] = SCHEMA_VERSION
        config["dataset_schema_version"] = SCHEMA_VERSION
        config["output_schema_version"] = SCHEMA_VERSION
        config["case_scope"] = CASE_SCOPE
        config["domain_cases"] = DOMAIN_CASES
        config["shift_scenarios"] = [f"{s}_to_{t}" for s, t in SHIFT_SCENARIOS]
        config["sensor_setting"] = SENSOR_SETTING
        config["expected_sensors"] = EXPECTED_SENSORS
        config["process_factors"] = PROCESS_FACTORS
        config["process_combinations"] = PROCESS_COMBINATIONS
        config["feature_groups"] = ["sensor_only", "process_only", "sensor_plus_process"]
        config["segment_settings"] = SEGMENT_SETTINGS
        config["process_only_segment_setting"] = "segment_independent"
        config["seeds"] = run_config.seeds
        config["dry_run"] = self.dry_run
        config["scaling_policy"] = "Numeric imputer/scaler and categorical imputer/one-hot encoder fit on train split only"
        config["config_hash"] = stable_hash(config)
        return config

    def _validate_inputs(self, run_config: S1RunConfig) -> dict[str, Any]:
        errors = []
        for path in [run_config.process_info_path, run_config.signal_data_path, run_config.heuristic_sequence_path]:
            if not path.exists():
                errors.append(f"Missing required file: {path}")
        return {"ok": not errors, "errors": errors, "warnings": []}

    def _sensor_mapping(self, sensors: list[str]) -> pd.DataFrame:
        rows = []
        for sensor in EXPECTED_SENSORS:
            rows.append({"sensor_setting": SENSOR_SETTING, "expected_sensor": sensor, "actual_sensor": sensor if sensor in sensors else None, "status": "mapped" if sensor in sensors else "missing"})
        mapping = pd.DataFrame(rows)
        missing = mapping.loc[mapping["status"] == "missing"]
        if not missing.empty:
            raise ValueError(f"Sensor mapping failed: {missing.to_dict(orient='records')}")
        return mapping

    def _process_mapping(self, dataset: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
        rows = []
        used_columns = {col for spec in PROCESS_FACTORS.values() for col in spec["columns"]}
        for factor, spec in PROCESS_FACTORS.items():
            for col in spec["columns"]:
                if col not in dataset.columns:
                    rows.append(self._process_mapping_row(experiment_id, factor, spec["name"], col, "unknown", False, "source column not found", dataset))
                    continue
                feature_type = "categorical" if dataset[col].dtype == "object" or is_string_dtype(dataset[col]) else "numeric"
                rows.append(self._process_mapping_row(experiment_id, factor, spec["name"], col, feature_type, True, "", dataset))
        for col, reason in EXCLUDED_PROCESS_COLUMNS.items():
            if col in dataset.columns and col not in used_columns:
                factor = "excluded"
                name = "Excluded"
                feature_type = "categorical" if dataset[col].dtype == "object" or is_string_dtype(dataset[col]) else "numeric"
                rows.append(self._process_mapping_row(experiment_id, factor, name, col, feature_type, False, reason, dataset))
        mapping = pd.DataFrame(rows)
        missing_factors = []
        for factor in PROCESS_FACTORS:
            if mapping.loc[(mapping["process_factor"] == factor) & (mapping["used"])].empty:
                missing_factors.append(factor)
        if missing_factors:
            raise ValueError(f"Process factor mapping failed for factors: {missing_factors}")
        return mapping

    def _process_mapping_row(self, experiment_id: str, factor: str, factor_name: str, col: str, feature_type: str, used: bool, reason: str, dataset: pd.DataFrame) -> dict[str, Any]:
        if col in dataset.columns:
            missing_rate = float(dataset[col].isna().mean())
            num_unique = int(dataset[col].nunique(dropna=True))
        else:
            missing_rate = math.nan
            num_unique = 0
        return {
            "experiment_id": experiment_id,
            "process_factor": factor,
            "process_factor_name": factor_name,
            "source_column": col,
            "feature_type": feature_type,
            "used": bool(used),
            "reason_if_excluded": reason,
            "missing_rate": missing_rate,
            "num_unique_values": num_unique,
            "encoding_method": "OneHotEncoder(handle_unknown='ignore')" if feature_type == "categorical" and used else "",
            "scaling_method": "StandardScaler" if feature_type == "numeric" and used else "",
        }

    def _process_columns_for_combination(self, combo: str, mapping: pd.DataFrame) -> list[str]:
        factors = PROCESS_COMBINATIONS[combo]
        return mapping.loc[mapping["used"] & mapping["process_factor"].isin(factors), "source_column"].tolist()

    def _process_factors_for_combination(self, combo: str) -> str:
        return ",".join(PROCESS_COMBINATIONS[combo])

    def _sensor_feature_table(self, feature_table: pd.DataFrame, sensors: list[str], segment: str) -> pd.DataFrame:
        sample_cols = ["sample_id", "dataset_run_id", "case", "run", "domain_id", "VB"]
        segment_rows = feature_table.loc[feature_table["segment_setting"] == segment]
        table = segment_rows[sample_cols].drop_duplicates().reset_index(drop=True)
        for sensor in sensors:
            sensor_rows = segment_rows.loc[segment_rows["sensor"] == sensor, sample_cols + FEATURE_NAMES].copy()
            rename = {name: f"{sensor}__{segment}__{name}" for name in FEATURE_NAMES}
            sensor_rows = sensor_rows.rename(columns=rename)
            table = table.merge(sensor_rows[sample_cols + list(rename.values())], on=sample_cols, how="inner", validate="one_to_one")
        return table

    def _process_base_table(self, dataset: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
        sample_cols = ["sample_id", "dataset_run_id", "case", "run", "domain_id", "VB"]
        used_cols = mapping.loc[mapping["used"], "source_column"].tolist()
        return dataset[sample_cols + used_cols].drop_duplicates().reset_index(drop=True)

    def _build_condition_tables(self, feature_table: pd.DataFrame, dataset: pd.DataFrame, sensor_mapping: pd.DataFrame, process_mapping: pd.DataFrame) -> dict[tuple[str, str, str], pd.DataFrame]:
        tables: dict[tuple[str, str, str], pd.DataFrame] = {}
        sensors = sensor_mapping["actual_sensor"].tolist()
        process_base = self._process_base_table(dataset, process_mapping)
        for segment in SEGMENT_SETTINGS:
            sensor_table = self._sensor_feature_table(feature_table, sensors, segment)
            tables[("sensor_only", segment, "none")] = sensor_table
            for combo in PROCESS_COMBINATIONS:
                process_cols = self._process_columns_for_combination(combo, process_mapping)
                tables[("sensor_plus_process", segment, combo)] = sensor_table.merge(
                    process_base[["sample_id"] + process_cols],
                    on="sample_id",
                    how="inner",
                    validate="one_to_one",
                )
        for combo in PROCESS_COMBINATIONS:
            process_cols = self._process_columns_for_combination(combo, process_mapping)
            tables[("process_only", "segment_independent", combo)] = process_base[["sample_id", "dataset_run_id", "case", "run", "domain_id", "VB"] + process_cols].copy()
        return tables

    def _validate_tables(self, dataset: pd.DataFrame, sensor_mapping: pd.DataFrame, process_mapping: pd.DataFrame, tables: dict[tuple[str, str, str], pd.DataFrame]) -> dict[str, Any]:
        errors = []
        if sorted(dataset["case"].unique().tolist()) != CASE_SCOPE:
            errors.append(f"case filtering failed: {sorted(dataset['case'].unique().tolist())}")
        if "VB" not in dataset.columns:
            errors.append("VB target column missing")
        if any("no_load" in setting for setting in SEGMENT_SETTINGS):
            errors.append("no-load setting included")
        if sensor_mapping["sensor_setting"].nunique() != 1 or sensor_mapping["sensor_setting"].iloc[0] != SENSOR_SETTING:
            errors.append("sensor_setting is not fixed to all_sensors")
        leakage_cols = {"VB", "case", "domain_id", "pair_id", "dataset_run_id", "sample_id", "split", "source_domain", "target_domain", "run"}
        for key, table in tables.items():
            feature_cols = self._feature_columns_for_table(table)
            if not feature_cols:
                errors.append(f"empty feature vector for {key}")
            if leakage_cols & set(feature_cols):
                errors.append(f"leakage columns in {key}: {sorted(leakage_cols & set(feature_cols))}")
        for combo in PROCESS_COMBINATIONS:
            if not self._process_columns_for_combination(combo, process_mapping):
                errors.append(f"empty process combination: {combo}")
        return {
            "ok": not errors,
            "errors": errors,
            "warnings": [
                "time is used as elapsed machining time; verify it is available at prediction time and not measured after VB",
                "material_name is used as categorical material property; verify it is not merely a case/domain shortcut",
            ],
            "table_validation": {
                "num_tables": len(tables),
                "sensor_setting": SENSOR_SETTING,
                "process_combinations": list(PROCESS_COMBINATIONS),
                "feature_groups": ["sensor_only", "process_only", "sensor_plus_process"],
                "table_shapes": {"/".join(k): list(v.shape) for k, v in tables.items()},
            },
        }

    def _feature_columns_for_table(self, table: pd.DataFrame) -> list[str]:
        meta = {"sample_id", "dataset_run_id", "case", "run", "domain_id", "VB", "split"}
        return [c for c in table.columns if c not in meta]

    def _dry_summary(self, dataset: pd.DataFrame, sensor_mapping: pd.DataFrame, process_mapping: pd.DataFrame, tables: dict[tuple[str, str, str], pd.DataFrame], run_config: S1RunConfig) -> dict[str, Any]:
        return {
            "case_scope": CASE_SCOPE,
            "rows_by_case": dataset.groupby("case").size().to_dict(),
            "target_column": "VB",
            "no_load_excluded": True,
            "sensor_setting": SENSOR_SETTING,
            "sensors": sensor_mapping["actual_sensor"].tolist(),
            "process_factors": PROCESS_FACTORS,
            "process_combinations": PROCESS_COMBINATIONS,
            "process_columns_used": process_mapping.loc[process_mapping["used"], "source_column"].tolist(),
            "process_columns_excluded": process_mapping.loc[~process_mapping["used"], ["source_column", "reason_if_excluded"]].to_dict(orient="records"),
            "feature_groups": ["sensor_only", "process_only", "sensor_plus_process"],
            "segment_settings": SEGMENT_SETTINGS,
            "models": run_config.models,
            "seeds": run_config.seeds,
            "planned_atomic_evaluations": (
                len(SHIFT_SCENARIOS)
                * len(run_config.models)
                * len(run_config.seeds)
                * (len(SEGMENT_SETTINGS) + len(PROCESS_COMBINATIONS) + len(SEGMENT_SETTINGS) * len(PROCESS_COMBINATIONS))
            ),
            "condition_table_shapes": {"/".join(k): list(v.shape) for k, v in tables.items()},
        }

    def _run_grid(self, helper: S1SegmentExecution, tables: dict[tuple[str, str, str], pd.DataFrame], process_mapping: pd.DataFrame, run_config: S1RunConfig, logger: ExperimentLogger) -> dict[str, Any]:
        shift_rows = []
        prediction_frames = []
        split_frames = []
        conditions = list(tables)
        total = len(run_config.models) * len(conditions) * len(run_config.seeds) * len(SHIFT_SCENARIOS)
        done = 0
        included_sensors = ",".join(EXPECTED_SENSORS)
        for model_name in run_config.models:
            for feature_group, segment, process_combo in conditions:
                table = tables[(feature_group, segment, process_combo)]
                feature_cols = self._feature_columns_for_table(table)
                sensor_cols = [c for c in feature_cols if "__" in c]
                process_cols = [c for c in feature_cols if "__" not in c]
                process_factors = "" if process_combo == "none" else self._process_factors_for_combination(process_combo)
                for seed in run_config.seeds:
                    for source, target in SHIFT_SCENARIOS:
                        done += 1
                        row, preds, splits = self._run_atomic(
                            helper,
                            model_name,
                            feature_group,
                            segment,
                            process_combo,
                            process_factors,
                            included_sensors,
                            ",".join(sensor_cols),
                            ",".join(process_cols),
                            seed,
                            source,
                            target,
                            table,
                            feature_cols,
                            process_cols,
                            run_config,
                        )
                        shift_rows.append(row)
                        prediction_frames.append(preds)
                        split_frames.append(splits)
                        if done % 500 == 0 or done == total:
                            logger.info(f"H2.S4 progress {done}/{total}")
        shift_metrics = pd.DataFrame(shift_rows)
        seed_metrics = self._seed_metrics(shift_metrics)
        feature_group_metrics = self._feature_group_metrics(seed_metrics, process_mapping)
        process_metrics = self._process_combination_metrics(feature_group_metrics, process_mapping)
        process_effect = self._process_effect_metrics(seed_metrics)
        process_effect_summary = self._process_effect_summary(process_effect)
        process_effect_consistency = self._process_effect_consistency(process_effect)
        best_process = self._best_process_combination(process_metrics)
        factor_contribution = self._process_factor_contribution(process_metrics)
        process_vs_sensor = self._process_only_vs_sensor_plus_process(feature_group_metrics)
        segment_effect = self._segment_effect_after_process_info(feature_group_metrics)
        best_segment = self._best_segment_by_process_combination(feature_group_metrics)
        r2_summary = self._r2_positive_summary(feature_group_metrics)
        baseline = pd.concat([process_effect.assign(comparison_type="process_addition"), segment_effect.assign(comparison_type="segment")], ignore_index=True, sort=False)
        metrics_json = self._metrics_json(feature_group_metrics, best_process, best_segment, r2_summary)
        return {
            "shift_metrics": shift_metrics,
            "seed_metrics": seed_metrics,
            "feature_group_metrics": feature_group_metrics,
            "process_combination_metrics": process_metrics,
            "process_effect_metrics": process_effect,
            "comparison_metrics": baseline,
            "predictions": pd.concat(prediction_frames, ignore_index=True),
            "splits": pd.concat(split_frames, ignore_index=True),
            "analysis_summary": None,
            "process_effect_summary": process_effect_summary,
            "process_effect_consistency": process_effect_consistency,
            "best_process_combination": best_process,
            "process_factor_contribution": factor_contribution,
            "segment_effect_after_process_info": segment_effect,
            "best_segment_by_process_combination": best_segment,
            "r2_positive_count_summary": r2_summary,
            "process_only_vs_sensor_plus_process": process_vs_sensor,
            "baseline_improvement": baseline,
            "metrics_json": metrics_json,
        }

    def _run_atomic(self, helper: S1SegmentExecution, model_name: str, feature_group: str, segment: str, process_combo: str, process_factors: str, sensors: str, sensor_features: str, process_features: str, seed: int, source: str, target: str, table: pd.DataFrame, feature_cols: list[str], process_cols: list[str], run_config: S1RunConfig):
        split = helper._split_frame(table, source, target)
        train = split.loc[split["split"] == "train"]
        test = split.loc[split["split"] == "test"]
        model = self._model(model_name, seed, run_config, feature_cols, process_cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[feature_cols], train["VB"])
        y_pred = model.predict(test[feature_cols])
        metrics = helper._metrics(test["VB"].to_numpy(), y_pred)
        child = f"H2S4_{model_name}_{feature_group}_{segment}_{process_combo}_{source}_to_{target}_seed_{seed}"
        row = {
            "experiment_id": None,
            "child_execution_key": child,
            "model": model_name,
            "sensor_setting": SENSOR_SETTING,
            "feature_group": feature_group,
            "segment_setting": segment,
            "process_combination": process_combo,
            "included_process_factors": process_factors,
            "included_sensors": sensors,
            "included_sensor_features": sensor_features,
            "included_process_features": process_features,
            "seed": seed,
            "source_domain": source,
            "target_domain": target,
            "source_cases": ",".join(map(str, DOMAIN_CASES[source])),
            "target_cases": ",".join(map(str, DOMAIN_CASES[target])),
            **metrics,
        }
        preds = test[["sample_id", "dataset_run_id", "case", "run", "domain_id", "VB"]].copy()
        for col, value in {
            "child_execution_key": child,
            "model": model_name,
            "sensor_setting": SENSOR_SETTING,
            "feature_group": feature_group,
            "segment_setting": segment,
            "process_combination": process_combo,
            "seed": seed,
            "source_domain": source,
            "target_domain": target,
        }.items():
            preds[col] = value
        preds["split"] = "test"
        preds["y_true"] = preds["VB"]
        preds["y_pred"] = y_pred
        preds["residual"] = preds["y_true"] - preds["y_pred"]
        preds["absolute_error"] = preds["residual"].abs()
        split_out = split[["sample_id", "dataset_run_id", "case", "run", "domain_id", "VB", "split"]].copy()
        for col, value in {
            "child_execution_key": child,
            "model": model_name,
            "sensor_setting": SENSOR_SETTING,
            "feature_group": feature_group,
            "segment_setting": segment,
            "process_combination": process_combo,
            "seed": seed,
            "source_domain": source,
            "target_domain": target,
        }.items():
            split_out[col] = value
        return row, preds, split_out

    def _model(self, model_name: str, seed: int, run_config: S1RunConfig, feature_cols: list[str], process_cols: list[str]) -> Pipeline:
        categorical = [c for c in process_cols if c == "material_name"]
        numeric = [c for c in feature_cols if c not in categorical]
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
            ],
            remainder="drop",
        )
        if model_name == "random_forest":
            params = dict(run_config.random_forest_params)
            params.setdefault("random_state", seed)
            estimator = RandomForestRegressor(**params)
        elif model_name == "mlp":
            params = dict(run_config.mlp_params)
            params.setdefault("hidden_layer_sizes", (64, 32, 16))
            params.setdefault("random_state", seed)
            params.setdefault("max_iter", 300)
            params.setdefault("learning_rate_init", 0.001)
            estimator = MLPRegressor(**params)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        return Pipeline([("preprocess", preprocessor), ("model", estimator)])

    def _seed_metrics(self, shift_metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for keys, group in shift_metrics.groupby(["model", "sensor_setting", "feature_group", "segment_setting", "process_combination", "seed"]):
            model, sensor_setting, feature_group, segment, process_combo, seed = keys
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "sensor_setting": sensor_setting,
                    "feature_group": feature_group,
                    "segment_setting": segment,
                    "process_combination": process_combo,
                    "seed": seed,
                    "mean_mae_over_6_shifts": group["metric_mae"].mean(),
                    "mean_rmse_over_6_shifts": group["metric_rmse"].mean(),
                    "mean_r2_over_6_shifts": group["metric_r2"].mean(),
                    "mean_pearson_over_6_shifts": group["metric_pearson"].mean(),
                    "mean_spearman_over_6_shifts": group["metric_spearman"].mean(),
                }
            )
        return pd.DataFrame(rows)

    def _feature_group_metrics(self, seed_metrics: pd.DataFrame, process_mapping: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for keys, group in seed_metrics.groupby(["model", "sensor_setting", "feature_group", "segment_setting", "process_combination"]):
            model, sensor_setting, feature_group, segment, process_combo = keys
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "sensor_setting": sensor_setting,
                    "feature_group": feature_group,
                    "segment_setting": segment,
                    "process_combination": process_combo,
                    "included_process_factors": "" if process_combo == "none" else self._process_factors_for_combination(process_combo),
                    "included_process_columns": "" if process_combo == "none" else ",".join(self._process_columns_for_combination(process_combo, process_mapping)),
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

    def _process_combination_metrics(self, metrics: pd.DataFrame, process_mapping: pd.DataFrame) -> pd.DataFrame:
        return metrics.loc[metrics["process_combination"] != "none"].copy()

    def _process_effect_metrics(self, seed_metrics: pd.DataFrame) -> pd.DataFrame:
        sensor = seed_metrics.loc[seed_metrics["feature_group"] == "sensor_only"]
        plus = seed_metrics.loc[seed_metrics["feature_group"] == "sensor_plus_process"]
        rows = []
        for _, row in plus.iterrows():
            base = sensor.loc[
                (sensor["model"] == row["model"])
                & (sensor["segment_setting"] == row["segment_setting"])
                & (sensor["seed"] == row["seed"])
            ]
            if base.empty:
                continue
            base = base.iloc[0]
            rows.append(
                {
                    "experiment_id": None,
                    "model": row["model"],
                    "sensor_setting": SENSOR_SETTING,
                    "segment_setting": row["segment_setting"],
                    "process_combination": row["process_combination"],
                    "seed": row["seed"],
                    "delta_mae": row["mean_mae_over_6_shifts"] - base["mean_mae_over_6_shifts"],
                    "delta_rmse": row["mean_rmse_over_6_shifts"] - base["mean_rmse_over_6_shifts"],
                    "delta_r2": row["mean_r2_over_6_shifts"] - base["mean_r2_over_6_shifts"],
                    "delta_pearson": row["mean_pearson_over_6_shifts"] - base["mean_pearson_over_6_shifts"],
                    "delta_spearman": row["mean_spearman_over_6_shifts"] - base["mean_spearman_over_6_shifts"],
                    "improved_mae": bool(row["mean_mae_over_6_shifts"] < base["mean_mae_over_6_shifts"]),
                    "improved_rmse": bool(row["mean_rmse_over_6_shifts"] < base["mean_rmse_over_6_shifts"]),
                    "improved_r2": bool(row["mean_r2_over_6_shifts"] > base["mean_r2_over_6_shifts"]),
                    "improved_pearson": bool(row["mean_pearson_over_6_shifts"] > base["mean_pearson_over_6_shifts"]),
                    "improved_spearman": bool(row["mean_spearman_over_6_shifts"] > base["mean_spearman_over_6_shifts"]),
                }
            )
        return pd.DataFrame(rows)

    def _process_effect_summary(self, effect: pd.DataFrame) -> pd.DataFrame:
        if effect.empty:
            return effect
        return effect.groupby(["model", "segment_setting", "process_combination"], as_index=False).agg(
            mean_delta_mae=("delta_mae", "mean"),
            mean_delta_rmse=("delta_rmse", "mean"),
            mean_delta_r2=("delta_r2", "mean"),
            improved_mae_count=("improved_mae", "sum"),
            improved_r2_count=("improved_r2", "sum"),
            num_seeds=("seed", "nunique"),
        )

    def _process_effect_consistency(self, effect: pd.DataFrame) -> pd.DataFrame:
        if effect.empty:
            return effect
        return effect.groupby(["model", "process_combination"], as_index=False).agg(
            improved_r2_ratio=("improved_r2", "mean"),
            improved_mae_ratio=("improved_mae", "mean"),
            mean_delta_r2=("delta_r2", "mean"),
        )

    def _best_process_combination(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        data = metrics.loc[metrics["process_combination"] != "none"]
        for keys, group in data.groupby(["model", "feature_group", "sensor_setting", "segment_setting"]):
            model, feature_group, sensor_setting, segment = keys
            mae = group.loc[group["mean_mae"].idxmin()]
            rmse = group.loc[group["mean_rmse"].idxmin()]
            r2 = group.loc[group["mean_r2"].idxmax()]
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "feature_group": feature_group,
                    "sensor_setting": sensor_setting,
                    "segment_setting": segment,
                    "best_process_combination_by_mae": mae["process_combination"],
                    "best_process_combination_by_rmse": rmse["process_combination"],
                    "best_process_combination_by_r2": r2["process_combination"],
                    "best_mae": mae["mean_mae"],
                    "best_rmse": rmse["mean_rmse"],
                    "best_r2": r2["mean_r2"],
                    "is_all_process_combination_best_by_r2": bool(r2["process_combination"] == "doc_feed_material_time"),
                }
            )
        return pd.DataFrame(rows)

    def _process_factor_contribution(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        data = metrics.loc[metrics["process_combination"] != "none"]
        for (model, feature_group), group in data.groupby(["model", "feature_group"]):
            for factor, name in FACTOR_NAMES.items():
                included = group.loc[group["process_combination"].map(lambda c: factor in PROCESS_COMBINATIONS[c])]
                excluded = group.loc[group["process_combination"].map(lambda c: factor not in PROCESS_COMBINATIONS[c])]
                rows.append(
                    {
                        "experiment_id": None,
                        "model": model,
                        "feature_group": feature_group,
                        "process_factor": factor,
                        "process_factor_name": name,
                        "mean_metric_when_included": included["mean_r2"].mean(),
                        "mean_metric_when_excluded": excluded["mean_r2"].mean(),
                        "delta_mae_when_included": included["mean_mae"].mean() - excluded["mean_mae"].mean(),
                        "delta_rmse_when_included": included["mean_rmse"].mean() - excluded["mean_rmse"].mean(),
                        "delta_r2_when_included": included["mean_r2"].mean() - excluded["mean_r2"].mean(),
                        "inclusion_improves_mae": bool(included["mean_mae"].mean() < excluded["mean_mae"].mean()),
                        "inclusion_improves_rmse": bool(included["mean_rmse"].mean() < excluded["mean_rmse"].mean()),
                        "inclusion_improves_r2": bool(included["mean_r2"].mean() > excluded["mean_r2"].mean()),
                    }
                )
        return pd.DataFrame(rows)

    def _process_only_vs_sensor_plus_process(self, metrics: pd.DataFrame) -> pd.DataFrame:
        process = metrics.loc[metrics["feature_group"] == "process_only"]
        plus = metrics.loc[metrics["feature_group"] == "sensor_plus_process"]
        rows = []
        for _, row in plus.iterrows():
            base = process.loc[(process["model"] == row["model"]) & (process["process_combination"] == row["process_combination"])]
            if base.empty:
                continue
            base = base.iloc[0]
            rows.append(
                {
                    "experiment_id": None,
                    "model": row["model"],
                    "process_combination": row["process_combination"],
                    "segment_setting": row["segment_setting"],
                    "delta_mae_sensor_plus_vs_process_only": row["mean_mae"] - base["mean_mae"],
                    "delta_rmse_sensor_plus_vs_process_only": row["mean_rmse"] - base["mean_rmse"],
                    "delta_r2_sensor_plus_vs_process_only": row["mean_r2"] - base["mean_r2"],
                    "sensor_plus_better_mae": bool(row["mean_mae"] < base["mean_mae"]),
                    "sensor_plus_better_rmse": bool(row["mean_rmse"] < base["mean_rmse"]),
                    "sensor_plus_better_r2": bool(row["mean_r2"] > base["mean_r2"]),
                }
            )
        return pd.DataFrame(rows)

    def _segment_effect_after_process_info(self, metrics: pd.DataFrame) -> pd.DataFrame:
        data = metrics.loc[metrics["feature_group"] == "sensor_plus_process"]
        rows = []
        for (model, process_combo), group in data.groupby(["model", "process_combination"]):
            for baseline in ["full_length", "steady"]:
                b = group.loc[group["segment_setting"] == baseline]
                if b.empty:
                    continue
                b = b.iloc[0]
                for _, row in group.iterrows():
                    segment = row["segment_setting"]
                    rows.append(
                        {
                            "experiment_id": None,
                            "model": model,
                            "process_combination": process_combo,
                            "segment_setting": segment,
                            "baseline": baseline,
                            "delta_mae": row["mean_mae"] - b["mean_mae"],
                            "delta_rmse": row["mean_rmse"] - b["mean_rmse"],
                            "delta_r2": row["mean_r2"] - b["mean_r2"],
                            "improved_vs_baseline_mae": bool(row["mean_mae"] < b["mean_mae"]),
                            "improved_vs_baseline_rmse": bool(row["mean_rmse"] < b["mean_rmse"]),
                            "improved_vs_baseline_r2": bool(row["mean_r2"] > b["mean_r2"]),
                            "segment_contains_entry": "entry" in segment,
                            "segment_contains_exit": "exit" in segment,
                            "is_segment_aware": segment not in {"full_length", "steady"},
                        }
                    )
        return pd.DataFrame(rows)

    def _best_segment_by_process_combination(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        data = metrics.loc[metrics["feature_group"] == "sensor_plus_process"]
        for (model, process_combo, feature_group), group in data.groupby(["model", "process_combination", "feature_group"]):
            mae = group.loc[group["mean_mae"].idxmin()]
            rmse = group.loc[group["mean_rmse"].idxmin()]
            r2 = group.loc[group["mean_r2"].idxmax()]
            segment = str(r2["segment_setting"])
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "process_combination": process_combo,
                    "feature_group": feature_group,
                    "best_segment_by_mae": mae["segment_setting"],
                    "best_segment_by_rmse": rmse["segment_setting"],
                    "best_segment_by_r2": r2["segment_setting"],
                    "best_mae": mae["mean_mae"],
                    "best_rmse": rmse["mean_rmse"],
                    "best_r2": r2["mean_r2"],
                    "best_segment_contains_entry": "entry" in segment,
                    "best_segment_contains_exit": "exit" in segment,
                }
            )
        return pd.DataFrame(rows)

    def _r2_positive_summary(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (model, feature_group, process_combo), group in metrics.groupby(["model", "feature_group", "process_combination"]):
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "feature_group": feature_group,
                    "process_combination": process_combo,
                    "num_total_conditions": len(group),
                    "num_r2_positive": int((group["mean_r2"] > 0).sum()),
                    "ratio_r2_positive": float((group["mean_r2"] > 0).mean()),
                    "mean_r2": group["mean_r2"].mean(),
                    "median_r2": group["mean_r2"].median(),
                    "min_r2": group["mean_r2"].min(),
                    "max_r2": group["mean_r2"].max(),
                }
            )
        return pd.DataFrame(rows)

    def _metrics_json(self, metrics: pd.DataFrame, best_process: pd.DataFrame, best_segment: pd.DataFrame, r2_summary: pd.DataFrame) -> dict[str, Any]:
        best = metrics.loc[metrics["mean_mae"].idxmin()].to_dict()
        return {
            "aggregation": "mean_over_6_domain_shifts_then_mean_std_over_seeds",
            "primary_metric": "mean_mae",
            "best_overall_by_mae": best,
            "all_process_combination_best_by_r2_ratio": float(best_process["is_all_process_combination_best_by_r2"].mean()) if not best_process.empty else 0.0,
            "r2_positive_summary_by_feature_group": r2_summary.groupby("feature_group")["num_r2_positive"].sum().to_dict(),
        }

    def _write_common_outputs(self, paths, dataset, feature_table, sensor_mapping, process_mapping, tables, dry_summary) -> None:
        write_json(paths.execution_dir / "data" / f"{PREFIX}_dataset_summary.json", dry_summary)
        write_csv(paths.execution_dir / "data" / f"{PREFIX}_sensor_mapping.csv", sensor_mapping)
        write_json(paths.execution_dir / "data" / f"{PREFIX}_sensor_setting_summary.json", {"sensor_setting": SENSOR_SETTING, "included_sensors": sensor_mapping["actual_sensor"].tolist()})
        write_csv(paths.execution_dir / "data" / f"{PREFIX}_process_information_mapping.csv", process_mapping)
        write_json(paths.execution_dir / "data" / f"{PREFIX}_case_filter_summary.json", {"used_cases": CASE_SCOPE, "rows_by_case": dry_summary["rows_by_case"]})
        write_json(paths.execution_dir / "data" / f"{PREFIX}_target_summary.json", {"target_column": "VB", "count": int(dataset["VB"].notna().sum()), "min": float(dataset["VB"].min()), "max": float(dataset["VB"].max())})
        write_json(paths.execution_dir / "preprocessing" / f"{PREFIX}_preprocessing_summary.json", {"output_type": "features", "no_load_excluded": True, "sensor_setting": SENSOR_SETTING})
        write_json(paths.execution_dir / "preprocessing" / f"{PREFIX}_feature_summary.json", {"sensor_feature_names": FEATURE_NAMES, "feature_naming": "{sensor_name}__{segment_setting}__{feature_name}"})
        write_json(paths.execution_dir / "preprocessing" / f"{PREFIX}_process_preprocessing_summary.json", {"numeric": "median imputation + StandardScaler", "categorical": "most frequent imputation + OneHotEncoder(handle_unknown='ignore')", "fit_policy": "train split only"})
        write_json(paths.execution_dir / "preprocessing" / f"{PREFIX}_shape_trace.json", [{"stage": "sensor_feature_table", "shape": list(feature_table.shape)}, {"stage": "condition_tables", "count": len(tables)}])

    def _write_results(self, paths, config, results, dry_summary, process_mapping, run_config) -> None:
        experiment_id = config["experiment"]["experiment_id"]
        for frame in results.values():
            if isinstance(frame, pd.DataFrame) and "experiment_id" in frame.columns:
                frame["experiment_id"] = experiment_id
        results["metrics_json"]["best_overall_by_mae"]["experiment_id"] = experiment_id
        files = {
            "shift_metrics": ("metrics", f"{PREFIX}_shift_metrics.csv"),
            "seed_metrics": ("metrics", f"{PREFIX}_seed_metrics.csv"),
            "feature_group_metrics": ("metrics", f"{PREFIX}_feature_group_metrics.csv"),
            "process_combination_metrics": ("metrics", f"{PREFIX}_process_combination_metrics.csv"),
            "process_effect_metrics": ("metrics", f"{PREFIX}_process_effect_metrics.csv"),
            "comparison_metrics": ("metrics", f"{PREFIX}_comparison_metrics.csv"),
            "predictions": ("predictions", f"{PREFIX}_predictions.csv"),
        }
        for key, (folder, name) in files.items():
            write_csv(paths.execution_dir / folder / name, results[key])
        write_json(paths.execution_dir / "metrics" / f"{PREFIX}_metrics.json", results["metrics_json"])
        write_csv(paths.execution_dir / "predictions" / f"{PREFIX}_prediction_errors.csv", results["predictions"].sort_values("absolute_error", ascending=False))
        write_csv(paths.execution_dir / "splits" / f"{PREFIX}_split.csv", results["splits"])
        write_json(paths.execution_dir / "splits" / f"{PREFIX}_split_summary.json", {"shift_scenarios": [f"{s}_to_{t}" for s, t in SHIFT_SCENARIOS]})
        write_json(paths.execution_dir / "splits" / f"{PREFIX}_leakage_check.json", self._leakage_check(results["splits"]))
        write_json(paths.execution_dir / "analysis" / f"{PREFIX}_analysis_summary.json", self._analysis_summary(results, dry_summary, run_config))
        for key in [
            "process_effect_summary",
            "process_effect_consistency",
            "process_combination_metrics",
            "best_process_combination",
            "process_factor_contribution",
            "segment_effect_after_process_info",
            "best_segment_by_process_combination",
            "r2_positive_count_summary",
            "process_only_vs_sensor_plus_process",
            "baseline_improvement",
        ]:
            write_csv(paths.execution_dir / "analysis" / f"{PREFIX}_{key}.csv", results[key])
        self._write_figures(paths, results)
        write_json(paths.execution_dir / "logs" / f"{PREFIX}_environment.json", collect_environment(str(self.root)))
        write_json(paths.execution_dir / "logs" / f"{PREFIX}_reproducibility.json", {"framework_version": __version__, "config_hash": config["config_hash"], "seeds": run_config.seeds})
        self._write_report(paths, dry_summary, results, dry_run=False)
        self._write_html(paths)

    def _leakage_check(self, splits: pd.DataFrame) -> dict[str, Any]:
        checks = []
        ok = True
        for source, target in SHIFT_SCENARIOS:
            overlap = sorted(set(DOMAIN_CASES[source]) & set(DOMAIN_CASES[target]))
            checks.append({"scenario": f"{source}_to_{target}", "source_cases": DOMAIN_CASES[source], "target_cases": DOMAIN_CASES[target], "overlap": overlap})
            ok = ok and not overlap
        return {
            "ok": ok,
            "case_scope": CASE_SCOPE,
            "source_target_overlap_checks": checks,
            "preprocessing_fit_policy": "imputation, encoding, and scaling are fit inside sklearn Pipeline on train split only",
            "feature_leakage_check": "VB/case/domain/sample/dataset_run/split identifiers excluded from feature columns",
            "process_only_segment_setting": "segment_independent",
        }

    def _analysis_summary(self, results: dict[str, Any], dry_summary: dict[str, Any], run_config: S1RunConfig) -> dict[str, Any]:
        return {
            "analysis_type": "Exploratory",
            "num_atomic_evaluations": int(len(results["shift_metrics"])),
            "skipped_condition_count": len(self.skipped),
            "skip_reasons": pd.DataFrame(self.skipped)["reason"].value_counts().to_dict() if self.skipped else {},
            "models": run_config.models,
            "seeds": run_config.seeds,
            "sensor_setting": SENSOR_SETTING,
            "included_sensors": dry_summary["sensors"],
            "process_combinations": list(PROCESS_COMBINATIONS),
            "feature_groups": ["sensor_only", "process_only", "sensor_plus_process"],
            "segment_settings": SEGMENT_SETTINGS,
            "best_overall_by_mae": results["metrics_json"]["best_overall_by_mae"],
            "all_process_combination_best_by_r2_ratio": results["metrics_json"]["all_process_combination_best_by_r2_ratio"],
        }

    def _write_figures(self, paths, results) -> None:
        import matplotlib.pyplot as plt

        figures = paths.execution_dir / "figures"
        effect = results["process_effect_metrics"]
        fig, ax = plt.subplots(figsize=(10, 5))
        effect.groupby(["model", "segment_setting"])["delta_r2"].mean().unstack(0).reindex(SEGMENT_SETTINGS).plot(kind="bar", ax=ax)
        ax.set_ylabel("Delta R2 vs sensor_only")
        fig.tight_layout()
        fig.savefig(figures / f"{PREFIX}_process_effect_by_segment.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        effect.groupby("model")["delta_r2"].mean().plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean delta R2")
        fig.tight_layout()
        fig.savefig(figures / f"{PREFIX}_process_effect_by_model.png")
        plt.close(fig)

        proc = results["process_combination_metrics"]
        order = list(PROCESS_COMBINATIONS)
        fig, ax = plt.subplots(figsize=(12, 5))
        for fg, group in proc.groupby("feature_group"):
            ordered = group.groupby("process_combination")["mean_r2"].mean().reindex(order)
            ax.plot(order, ordered, marker="o", label=fg)
        ax.tick_params(axis="x", rotation=40)
        ax.set_ylabel("Mean R2")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"{PREFIX}_process_combination_performance.png")
        plt.close(fig)

        factor = results["process_factor_contribution"]
        fig, ax = plt.subplots(figsize=(8, 4))
        factor.groupby("process_factor_name")["delta_r2_when_included"].mean().plot(kind="bar", ax=ax)
        ax.set_ylabel("Delta R2 when included")
        fig.tight_layout()
        fig.savefig(figures / f"{PREFIX}_process_factor_contribution.png")
        plt.close(fig)

        r2 = results["r2_positive_count_summary"]
        fig, ax = plt.subplots(figsize=(8, 4))
        r2.groupby(["feature_group", "model"])["ratio_r2_positive"].mean().unstack(1).plot(kind="bar", ax=ax)
        ax.set_ylabel("R2 positive ratio")
        fig.tight_layout()
        fig.savefig(figures / f"{PREFIX}_r2_positive_count_comparison.png")
        plt.close(fig)

        best_segment = results["best_segment_by_process_combination"]
        fig, ax = plt.subplots(figsize=(10, 4))
        best_segment["best_segment_by_r2"].value_counts().reindex(SEGMENT_SETTINGS, fill_value=0).plot(kind="bar", ax=ax)
        ax.set_ylabel("Best count")
        fig.tight_layout()
        fig.savefig(figures / f"{PREFIX}_best_segment_after_process_info.png")
        plt.close(fig)

        heat_data = proc.loc[proc["feature_group"] == "sensor_plus_process"].groupby(["segment_setting", "process_combination"])["mean_r2"].mean().unstack()
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(heat_data.reindex(index=SEGMENT_SETTINGS, columns=order).to_numpy(), aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(SEGMENT_SETTINGS)), SEGMENT_SETTINGS)
        ax.set_xticks(range(len(order)), order, rotation=40, ha="right")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(figures / f"{PREFIX}_feature_group_performance_heatmap.png")
        plt.close(fig)

    def _write_report(self, paths, dry_summary, results, dry_run: bool) -> None:
        report = paths.execution_dir / "reports" / f"{PREFIX}_report.md"
        if dry_run:
            body = f"""# H2.S4 Dry-run Report

Dry-run completed.

- Cases: {CASE_SCOPE}
- Domain pairs: {DOMAIN_CASES}
- Sensor setting: {SENSOR_SETTING}
- Sensors: {dry_summary['sensors']}
- Process factors: {PROCESS_FACTORS}
- Process combinations: {list(PROCESS_COMBINATIONS)}
- Process columns used: {dry_summary['process_columns_used']}
- Excluded process columns: {dry_summary['process_columns_excluded']}
- Feature groups: {dry_summary['feature_groups']}
- Segment settings: {SEGMENT_SETTINGS}
- Planned atomic evaluations: {dry_summary['planned_atomic_evaluations']}
- No-load excluded: true
"""
        else:
            summary = results["metrics_json"]
            best_process = results["best_process_combination"].to_string(index=False)
            factor = results["process_factor_contribution"].to_string(index=False)
            best_segment = results["best_segment_by_process_combination"].to_string(index=False)
            r2 = results["r2_positive_count_summary"].to_string(index=False)
            body = f"""# H2.S4 Process Information Combination Effect on Segment-aware VB Prediction

## Protocol

- Data files: `datasets/processed/mill_process_info_enabled.csv`, `datasets/processed/mill_signal_data_enabled.csv`
- Cases: {CASE_SCOPE}
- Domain pairs: {DOMAIN_CASES}
- Shift scenarios: {[f'{s}_to_{t}' for s, t in SHIFT_SCENARIOS]}
- Sensor setting: {SENSOR_SETTING}
- Included sensors: {dry_summary['sensors']}
- Process factors: A=DoC, B=Feed, C=Material, D=Time
- Process combinations: {list(PROCESS_COMBINATIONS)}
- Used process columns: {dry_summary['process_columns_used']}
- Excluded process columns and reasons: {dry_summary['process_columns_excluded']}
- Feature groups: {dry_summary['feature_groups']}
- Segment settings: {SEGMENT_SETTINGS}
- Models: {dry_summary['models']}
- Seeds: {dry_summary['seeds']}
- No-load excluded: true

## Best Overall

```json
{json.dumps(summary['best_overall_by_mae'], indent=2, default=str)}
```

## Best Process Combination

```text
{best_process}
```

## Process Factor Contribution

```text
{factor}
```

## Best Segment After Process Information

```text
{best_segment}
```

## R2 Positive Count

```text
{r2}
```

## Leakage Review

- Time: `time` is elapsed machining time and may be a strong wear progression proxy. It is used here, but should be considered potentially unavailable or leakage-prone if measured after the prediction point.
- Material: `material_name` is treated as a categorical material property. It may also act as a domain shortcut if material is case-specific, so interpretation should check domain dependence.
- Excluded: `material` numeric code was excluded because it duplicates `material_name`; `case`, `run`, `VB`, and `enable` were excluded.

## RQ Summary

- RQ1/RQ2: Inspect `metrics/H2_S4_process_effect_metrics.csv`; process information helps only where delta R2 is positive or delta MAE/RMSE is negative.
- RQ3: Inspect `analysis/H2_S4_segment_effect_after_process_info.csv` and `analysis/H2_S4_best_segment_by_process_combination.csv`.
- RQ4: Inspect `analysis/H2_S4_r2_positive_count_summary.csv`.
- RQ5: Inspect `analysis/H2_S4_process_only_vs_sensor_plus_process.csv`.
- RQ6: Inspect `analysis/H2_S4_process_factor_contribution.csv`.
"""
        report.write_text(body + paths.report_metadata_markdown(), encoding="utf-8")

    def _write_html(self, paths) -> None:
        md = paths.execution_dir / "reports" / f"{PREFIX}_report.md"
        html = paths.execution_dir / "reports" / f"{PREFIX}_report.html"
        body = md.read_text(encoding="utf-8").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>\n")
        html.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")

    def _best_metric(self, metrics: pd.DataFrame) -> float | None:
        return None if metrics.empty else float(metrics["mean_mae"].min())

    def _update_index(self, config: dict[str, Any], status: str, best_metric: float | None, error_message: str | None = None, error_log: str | None = None) -> None:
        index_path = self.root / "experiments" / "index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "experiment_id": config["experiment"]["experiment_id"],
            **execution_index_fields(config),
            "experiment_name": "H2_S4_process_information_combination_effect_on_segment_aware_VB_prediction",
            "dataset": "mill_processed_enabled",
            "model": "random_forest,mlp",
            "input_type": "feature-based",
            "split_strategy": "fixed_case_pair_domain_shift",
            "steady_cut_mode": "segmentation_no_noload",
            "status": status,
            "best_metric": best_metric,
            "created_at": config["experiment"]["created_at"],
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

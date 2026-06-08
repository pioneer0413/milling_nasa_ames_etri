from __future__ import annotations

import json
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from milling_experiment_framework import __version__
from milling_experiment_framework.core.config import SCHEMA_VERSION, stable_hash
from milling_experiment_framework.datasets.mill_constants import (
    CASE_SCOPE,
    DOMAIN_CASES,
    SEGMENT_SETTINGS,
    SHIFT_SCENARIOS,
)
from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.experiment_logging.experiment_logger import ExperimentLogger
from milling_experiment_framework.experiments.base_execution import BaseH2Execution
from milling_experiment_framework.experiments.h2_execution_utils import (
    S1RunConfig,
    atomic_signature,
    concat_existing_new,
    effective_seeds_for_model,
    existing_run_signatures,
    model_seed_value,
    ModelProgressReporter,
    planned_atomic_count,
    print_runtime_estimate_and_confirm,
    read_existing_csv,
    seed_label,
)
from milling_experiment_framework.experiments.s1_segment_execution import S1SegmentExecution
from milling_experiment_framework.models.h2_regressors import h2_model_catalog
from milling_experiment_framework.utils.io import write_csv, write_json, write_yaml
from milling_experiment_framework.utils.paths import ExperimentPaths


SENSOR_SETTING = "all_sensors"
EXPECTED_SENSORS = ["smcAC", "smcDC", "vib_spindle", "vib_table", "AE_spindle", "AE_table"]
FEATURE_GROUPS = {
    "statistics": ["mean", "std", "max", "min", "peak_to_peak"],
    "shape": ["kurtosis", "skewness"],
    "frequency": ["spectral_centroid", "band_energy"],
}
FEATURE_COMBINATIONS = {
    "statistics": ["statistics"],
    "shape": ["shape"],
    "frequency": ["frequency"],
    "statistics_shape": ["statistics", "shape"],
    "statistics_frequency": ["statistics", "frequency"],
    "shape_frequency": ["shape", "frequency"],
    "statistics_shape_frequency": ["statistics", "shape", "frequency"],
}
SINGLE_FEATURE_GROUPS = {"statistics", "shape", "frequency"}


class H2S3FeatureCombinationExecution(BaseH2Execution):
    """Run H2.S3 segment-aware VB prediction over feature group combinations."""

    _scenario_id = "S3"
    _experiment_topic = "feature_combination_all_sensors_segment_aware_VB_prediction"
    _index_name = "H2_S3_feature_combination_effect_on_segment_aware_VB_prediction_all_sensors"
    _index_steady_cut = "segmentation_no_noload"

    def run(self) -> dict[str, Any]:
        raw_config = self._read_config()
        run_config = self._resolve_run_config(raw_config)
        experiment_id = self._generate_experiment_id()
        config = self._resolved_config(raw_config, run_config, experiment_id)
        paths = ExperimentPaths(self.root, experiment_id)
        paths.apply_to_config(config)
        paths.prepare_standard_dirs()
        logger = ExperimentLogger(paths.execution_dir / "logs" / "run.log")
        logger.info(f"H2.S3 execution started: {experiment_id}")
        if self.dry_run:
            logger.info("dry-run mode enabled")

        write_yaml(paths.execution_dir / "configs" / "input_config.yaml", raw_config)
        write_yaml(paths.execution_dir / "configs" / "resolved_config.yaml", config)
        try:
            helper = S1SegmentExecution(self.config_path, root=self.root, dry_run=True, seed_mode=self.seed_mode)
            validation = self._validate_inputs(run_config)
            dataset = helper._load_dataset(run_config)
            sensors = helper._sensor_columns(dataset)
            mapping = self._sensor_mapping(sensors)
            feature_table = helper._build_feature_table(dataset, mapping["actual_sensor"].tolist())
            combo_features = self._build_feature_tables(feature_table, mapping)
            validation.update(self._validate_features(dataset, mapping, combo_features))
            write_json(paths.execution_dir / "configs" / "config_validation.json", validation)
            dry_summary = self._dry_summary(dataset, mapping, combo_features, run_config)
            self._write_common_outputs(paths, dataset, feature_table, mapping, combo_features, dry_summary)
            if self.dry_run:
                write_json(paths.execution_dir / "metrics" / "metrics.json", {"dry_run": True, "summary": dry_summary})
                self._write_report(paths, dry_summary, {}, dry_run=True)
                self._write_html(paths)
                self._update_index(config, "finished", None)
                logger.info(f"H2.S3 dry-run finished: {experiment_id}")
                return {"experiment_id": experiment_id, "dry_run": True, "summary": dry_summary, "execution_dir": str(paths.execution_dir)}

            print_runtime_estimate_and_confirm(
                "H2.S3 feature combination effect",
                run_config.models,
                run_config.seeds,
                condition_count=len(FEATURE_COMBINATIONS) * len(SEGMENT_SETTINGS),
                test_count=len(SHIFT_SCENARIOS),
                config=raw_config,
                assume_yes=self.assume_yes,
            )
            results = self._run_grid(combo_features, mapping, run_config, logger, paths)
            self._write_results(paths, config, results, dry_summary, run_config)
            (paths.execution_dir / "logs" / "error.log").touch()
            self._update_index(config, "finished", self._best_metric(results["feature_combination_metrics"]))
            logger.info(f"H2.S3 execution finished: {experiment_id}")
            return {"experiment_id": experiment_id, "metrics": results["metrics_json"], "execution_dir": str(paths.execution_dir)}
        except Exception as exc:
            error_text = traceback.format_exc()
            logger.error(str(exc))
            (paths.execution_dir / "logs" / "error.log").write_text(error_text, encoding="utf-8")
            write_json(paths.execution_dir / "configs" / "config_validation.json", {"ok": False, "errors": [str(exc)]})
            self._update_index(config, "failed", None, str(exc), str(paths.execution_dir / "logs" / "error.log"))
            raise

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
        config["feature_groups"] = FEATURE_GROUPS
        config["feature_combinations"] = FEATURE_COMBINATIONS
        config["segment_settings"] = SEGMENT_SETTINGS
        config["seeds"] = run_config.seeds
        config["model_catalog"] = h2_model_catalog(run_config.model_params)
        config["dry_run"] = self.dry_run
        config["scaling_policy"] = "StandardScaler fit on train split only for all H2 feature-based model pipelines"
        config["config_hash"] = stable_hash(config)
        return config

    def _sensor_mapping(self, sensors: list[str]) -> pd.DataFrame:
        rows = []
        for sensor in EXPECTED_SENSORS:
            rows.append(
                {
                    "sensor_setting": SENSOR_SETTING,
                    "expected_sensor": sensor,
                    "actual_sensor": sensor if sensor in sensors else None,
                    "status": "mapped" if sensor in sensors else "missing",
                }
            )
        mapping = pd.DataFrame(rows)
        missing = mapping.loc[mapping["status"] == "missing"]
        if not missing.empty:
            raise ValueError(f"Sensor mapping failed: {missing.to_dict(orient='records')}")
        return mapping

    def _features_for_combination(self, combo: str) -> list[str]:
        names: list[str] = []
        for group in FEATURE_COMBINATIONS[combo]:
            for feature in FEATURE_GROUPS[group]:
                if feature not in names:
                    names.append(feature)
        return names

    def _build_feature_tables(self, feature_table: pd.DataFrame, mapping: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
        sample_cols = ["sample_id", "dataset_run_id", "case", "run", "domain_id", "VB"]
        sensors = mapping["actual_sensor"].tolist()
        tables: dict[tuple[str, str], pd.DataFrame] = {}
        for segment in SEGMENT_SETTINGS:
            segment_rows = feature_table.loc[feature_table["segment_setting"] == segment]
            base = segment_rows[sample_cols].drop_duplicates().reset_index(drop=True)
            for combo in FEATURE_COMBINATIONS:
                table = base.copy()
                selected = self._features_for_combination(combo)
                for sensor in sensors:
                    sensor_rows = segment_rows.loc[segment_rows["sensor"] == sensor, sample_cols + selected].copy()
                    rename = {name: f"{sensor}__{segment}__{name}" for name in selected}
                    sensor_rows = sensor_rows.rename(columns=rename)
                    table = table.merge(sensor_rows[sample_cols + list(rename.values())], on=sample_cols, how="inner", validate="one_to_one")
                tables[(combo, segment)] = table
        return tables

    def _validate_features(self, dataset: pd.DataFrame, mapping: pd.DataFrame, combo_features: dict[tuple[str, str], pd.DataFrame]) -> dict[str, Any]:
        errors = []
        if sorted(dataset["case"].unique().tolist()) != CASE_SCOPE:
            errors.append(f"case filtering failed: {sorted(dataset['case'].unique().tolist())}")
        if "VB" not in dataset.columns:
            errors.append("VB target column missing")
        if any("no_load" in setting for setting in SEGMENT_SETTINGS):
            errors.append("no-load setting included")
        if len(mapping["sensor_setting"].unique()) != 1 or mapping["sensor_setting"].iloc[0] != SENSOR_SETTING:
            errors.append("sensor_setting is not fixed to all_sensors")
        expected_features = {f for features in FEATURE_GROUPS.values() for f in features}
        leakage_names = {"VB", "case", "domain_id", "pair_id", "dataset_run_id", "split", "source_domain", "target_domain"}
        for key, table in combo_features.items():
            feature_cols = [c for c in table.columns if "__" in c]
            used_leaf_names = {c.rsplit("__", 1)[-1] for c in feature_cols}
            if not used_leaf_names.issubset(expected_features):
                errors.append(f"unknown feature in {key}: {sorted(used_leaf_names - expected_features)}")
            if leakage_names & set(feature_cols):
                errors.append(f"leakage feature in {key}: {sorted(leakage_names & set(feature_cols))}")
            if table.empty:
                errors.append(f"empty feature table for {key}")
        return {
            "ok": not errors,
            "errors": errors,
            "warnings": [],
            "feature_validation": {
                "sensor_setting": SENSOR_SETTING,
                "num_feature_combinations": len(FEATURE_COMBINATIONS),
                "num_segment_settings": len(SEGMENT_SETTINGS),
                "sensors": mapping["actual_sensor"].tolist(),
                "feature_tables": {f"{c}/{s}": list(t.shape) for (c, s), t in combo_features.items()},
            },
        }

    def _dry_summary(self, dataset: pd.DataFrame, mapping: pd.DataFrame, combo_features: dict[tuple[str, str], pd.DataFrame], run_config: S1RunConfig) -> dict[str, Any]:
        return {
            "case_scope": CASE_SCOPE,
            "rows_by_case": dataset.groupby("case").size().to_dict(),
            "target_column": "VB",
            "no_load_excluded": True,
            "sensor_setting": SENSOR_SETTING,
            "sensors": mapping["actual_sensor"].tolist(),
            "feature_groups": FEATURE_GROUPS,
            "feature_combinations": FEATURE_COMBINATIONS,
            "segment_settings": SEGMENT_SETTINGS,
            "models": run_config.models,
            "seeds": run_config.seeds,
            "effective_seed_policy": "linear_regression and svr run once with seed=-1; seeded models use the configured seeds",
            "planned_atomic_evaluations": planned_atomic_count(run_config.models, run_config.seeds, len(FEATURE_COMBINATIONS) * len(SEGMENT_SETTINGS), len(SHIFT_SCENARIOS)),
            "combination_feature_shapes": {f"{combo}/{segment}": list(table.shape) for (combo, segment), table in combo_features.items()},
            "sensor_mapping": mapping.to_dict(orient="records"),
        }

    def _run_grid(self, combo_features: dict[tuple[str, str], pd.DataFrame], mapping: pd.DataFrame, run_config: S1RunConfig, logger: ExperimentLogger, paths: ExperimentPaths) -> dict[str, Any]:
        helper = S1SegmentExecution(self.config_path, root=self.root, dry_run=False, seed_mode=self.seed_mode)
        shift_rows = []
        prediction_frames = []
        split_frames = []
        existing_shift = read_existing_csv(paths.execution_dir / "metrics" / "shift_metrics.csv")
        existing_predictions = read_existing_csv(paths.execution_dir / "predictions" / "predictions.csv")
        existing_splits = read_existing_csv(paths.execution_dir / "splits" / "split.csv")
        completed = existing_run_signatures(existing_shift)
        total = planned_atomic_count(run_config.models, run_config.seeds, len(FEATURE_COMBINATIONS) * len(SEGMENT_SETTINGS), len(SHIFT_SCENARIOS))
        done = 0
        included_sensors = ",".join(mapping["actual_sensor"].tolist())
        progress = ModelProgressReporter("H2.S3")
        for model_name in run_config.models:
            model_total = len(effective_seeds_for_model(model_name, run_config.seeds)) * len(FEATURE_COMBINATIONS) * len(SEGMENT_SETTINGS) * len(SHIFT_SCENARIOS)
            progress.start_model(model_name, model_total)
            for feature_combo, groups in FEATURE_COMBINATIONS.items():
                included_groups = ",".join(groups)
                included_features = ",".join(self._features_for_combination(feature_combo))
                for segment in SEGMENT_SETTINGS:
                    data = combo_features[(feature_combo, segment)]
                    feature_cols = [c for c in data.columns if "__" in c]
                    for seed in effective_seeds_for_model(model_name, run_config.seeds):
                        for source, target in SHIFT_SCENARIOS:
                            done += 1
                            child = f"H2S3_{model_name}_{SENSOR_SETTING}_{feature_combo}_{segment}_{source}_to_{target}_seed_{seed_label(seed)}"
                            signature = atomic_signature(
                                {
                                    "experiment": "H2_S3",
                                    "model": model_name,
                                    "model_params": run_config.model_params.get(model_name, {}),
                                    "sensor_setting": SENSOR_SETTING,
                                    "included_sensors": included_sensors,
                                    "feature_combination": feature_combo,
                                    "included_feature_groups": included_groups,
                                    "included_features": included_features,
                                    "segment_setting": segment,
                                    "feature_columns": feature_cols,
                                    "seed": seed,
                                    "source_domain": source,
                                    "target_domain": target,
                                }
                            )
                            if signature in completed:
                                self.skipped.append({"child_execution_key": child, "run_signature": signature, "model": model_name, "seed": seed, "reason": "existing_result_same_setting"})
                                progress.step(skipped=True)
                                continue
                            result, preds, splits = self._run_atomic(
                                helper,
                                model_name,
                                feature_combo,
                                included_groups,
                                included_features,
                                included_sensors,
                                segment,
                                seed,
                                source,
                                target,
                                data,
                                feature_cols,
                                run_config,
                                child,
                                signature,
                            )
                            shift_rows.append(result)
                            prediction_frames.append(preds)
                            split_frames.append(splits)
                            progress.step(skipped=False)
                            if done % 500 == 0 or done == total:
                                logger.info(f"H2.S3 progress {done}/{total}")
        shift_metrics = concat_existing_new(existing_shift, shift_rows)
        seed_metrics = self._seed_metrics(shift_metrics)
        combo_metrics = self._combo_metrics(seed_metrics)
        segment_metrics = self._segment_metrics(seed_metrics)
        contribution = self._feature_group_contribution(combo_metrics)
        combo_effect = self._feature_combination_effect(combo_metrics)
        best_combo = self._best_feature_combination(combo_metrics)
        segment_effect = self._segment_effect(combo_metrics)
        best_segment = self._best_segment_by_combo(combo_metrics)
        r2_summary = self._r2_positive_summary(combo_metrics)
        baseline = pd.concat(
            [combo_effect.assign(comparison_type="feature_combination"), segment_effect.assign(comparison_type="segment")],
            ignore_index=True,
            sort=False,
        )
        metrics_json = self._metrics_json(combo_metrics, best_combo, best_segment, r2_summary)
        return {
            "shift_metrics": shift_metrics,
            "seed_metrics": seed_metrics,
            "feature_combination_metrics": combo_metrics,
            "segment_metrics": segment_metrics,
            "comparison_metrics": baseline,
            "predictions": concat_existing_new(existing_predictions, prediction_frames),
            "splits": concat_existing_new(existing_splits, split_frames),
            "feature_group_contribution": contribution,
            "feature_combination_effect": combo_effect,
            "best_feature_combination": best_combo,
            "segment_effect_under_feature_combination": segment_effect,
            "best_segment_by_feature_combination": best_segment,
            "r2_positive_count_summary": r2_summary,
            "baseline_improvement": baseline,
            "metrics_json": metrics_json,
        }

    def _run_atomic(self, helper: S1SegmentExecution, model_name: str, feature_combo: str, groups: str, features: str, sensors: str, segment: str, seed: int, source: str, target: str, data: pd.DataFrame, feature_cols: list[str], run_config: S1RunConfig, child: str, run_signature: str):
        split = helper._split_frame(data, source, target)
        train = split.loc[split["split"] == "train"]
        test = split.loc[split["split"] == "test"]
        model = helper._model(model_name, model_seed_value(seed), run_config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[feature_cols], train["VB"])
        y_pred = model.predict(test[feature_cols])
        metrics = helper._metrics(test["VB"].to_numpy(), y_pred)
        row = {
            "experiment_id": None,
            "child_execution_key": child,
            "run_signature": run_signature,
            "model": model_name,
            "sensor_setting": SENSOR_SETTING,
            "feature_combination": feature_combo,
            "included_feature_groups": groups,
            "included_features": features,
            "included_sensors": sensors,
            "segment_setting": segment,
            "seed": seed,
            "source_domain": source,
            "target_domain": target,
            "source_cases": ",".join(map(str, DOMAIN_CASES[source])),
            "target_cases": ",".join(map(str, DOMAIN_CASES[target])),
            **metrics,
        }
        preds = test[["sample_id", "dataset_run_id", "case", "run", "domain_id", "VB"]].copy()
        preds["child_execution_key"] = child
        preds["run_signature"] = run_signature
        preds["model"] = model_name
        preds["sensor_setting"] = SENSOR_SETTING
        preds["feature_combination"] = feature_combo
        preds["included_feature_groups"] = groups
        preds["included_features"] = features
        preds["included_sensors"] = sensors
        preds["segment_setting"] = segment
        preds["seed"] = seed
        preds["source_domain"] = source
        preds["target_domain"] = target
        preds["split"] = "test"
        preds["y_true"] = preds["VB"]
        preds["y_pred"] = y_pred
        preds["residual"] = preds["y_true"] - preds["y_pred"]
        preds["absolute_error"] = preds["residual"].abs()
        split_out = split[["sample_id", "dataset_run_id", "case", "run", "domain_id", "VB", "split"]].copy()
        split_out["child_execution_key"] = child
        split_out["run_signature"] = run_signature
        split_out["model"] = model_name
        split_out["sensor_setting"] = SENSOR_SETTING
        split_out["feature_combination"] = feature_combo
        split_out["segment_setting"] = segment
        split_out["seed"] = seed
        split_out["source_domain"] = source
        split_out["target_domain"] = target
        return row, preds, split_out

    def _seed_metrics(self, shift_metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for keys, group in shift_metrics.groupby(["model", "sensor_setting", "feature_combination", "segment_setting", "seed"]):
            model, sensor_setting, combo, segment, seed = keys
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "sensor_setting": sensor_setting,
                    "feature_combination": combo,
                    "segment_setting": segment,
                    "seed": seed,
                    "mean_mae_over_6_shifts": group["metric_mae"].mean(),
                    "mean_rmse_over_6_shifts": group["metric_rmse"].mean(),
                    "mean_r2_over_6_shifts": group["metric_r2"].mean(),
                    "mean_pearson_over_6_shifts": group["metric_pearson"].mean(),
                    "mean_spearman_over_6_shifts": group["metric_spearman"].mean(),
                }
            )
        return pd.DataFrame(rows)

    def _combo_metrics(self, seed_metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for keys, group in seed_metrics.groupby(["model", "sensor_setting", "feature_combination", "segment_setting"]):
            model, sensor_setting, combo, segment = keys
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "sensor_setting": sensor_setting,
                    "feature_combination": combo,
                    "segment_setting": segment,
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

    def _segment_metrics(self, seed_metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for keys, group in seed_metrics.groupby(["model", "sensor_setting", "segment_setting"]):
            model, sensor_setting, segment = keys
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "sensor_setting": sensor_setting,
                    "segment_setting": segment,
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

    def _feature_group_contribution(self, metrics: pd.DataFrame) -> pd.DataFrame:
        singles = metrics.loc[metrics["feature_combination"].isin(SINGLE_FEATURE_GROUPS)].copy()
        singles = singles.rename(columns={"feature_combination": "feature_group"})
        singles["rank_by_mae"] = singles.groupby(["model", "sensor_setting", "segment_setting"])["mean_mae"].rank(method="min", ascending=True)
        singles["rank_by_rmse"] = singles.groupby(["model", "sensor_setting", "segment_setting"])["mean_rmse"].rank(method="min", ascending=True)
        singles["rank_by_r2"] = singles.groupby(["model", "sensor_setting", "segment_setting"])["mean_r2"].rank(method="min", ascending=False)
        singles["is_best_by_r2"] = singles["rank_by_r2"] == 1
        return singles[
            [
                "experiment_id",
                "model",
                "sensor_setting",
                "segment_setting",
                "feature_group",
                "mean_mae",
                "mean_rmse",
                "mean_r2",
                "rank_by_mae",
                "rank_by_rmse",
                "rank_by_r2",
                "is_best_by_r2",
            ]
        ]

    def _feature_combination_effect(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        baseline_map = {
            "statistics_shape": ["statistics", "shape"],
            "statistics_frequency": ["statistics", "frequency"],
            "shape_frequency": ["shape", "frequency"],
            "statistics_shape_frequency": ["statistics", "shape", "frequency"],
        }
        for (model, sensor_setting, segment), group in metrics.groupby(["model", "sensor_setting", "segment_setting"]):
            for combo, baselines in baseline_map.items():
                c = group.loc[group["feature_combination"] == combo]
                if c.empty:
                    continue
                c = c.iloc[0]
                for baseline in baselines:
                    b = group.loc[group["feature_combination"] == baseline]
                    if b.empty:
                        continue
                    b = b.iloc[0]
                    rows.append(
                        {
                            "experiment_id": None,
                            "model": model,
                            "sensor_setting": sensor_setting,
                            "segment_setting": segment,
                            "comparison_feature_combination": combo,
                            "baseline_feature_group": baseline,
                            "delta_mae": c["mean_mae"] - b["mean_mae"],
                            "delta_rmse": c["mean_rmse"] - b["mean_rmse"],
                            "delta_r2": c["mean_r2"] - b["mean_r2"],
                            "delta_pearson": c["mean_pearson"] - b["mean_pearson"],
                            "delta_spearman": c["mean_spearman"] - b["mean_spearman"],
                            "improved_mae": bool(c["mean_mae"] < b["mean_mae"]),
                            "improved_rmse": bool(c["mean_rmse"] < b["mean_rmse"]),
                            "improved_r2": bool(c["mean_r2"] > b["mean_r2"]),
                            "improved_pearson": bool(c["mean_pearson"] > b["mean_pearson"]),
                            "improved_spearman": bool(c["mean_spearman"] > b["mean_spearman"]),
                        }
                    )
        return pd.DataFrame(rows)

    def _best_feature_combination(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (model, sensor_setting, segment), group in metrics.groupby(["model", "sensor_setting", "segment_setting"]):
            mae = group.loc[group["mean_mae"].idxmin()]
            rmse = group.loc[group["mean_rmse"].idxmin()]
            r2 = group.loc[group["mean_r2"].idxmax()]
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "sensor_setting": sensor_setting,
                    "segment_setting": segment,
                    "best_feature_combination_by_mae": mae["feature_combination"],
                    "best_feature_combination_by_rmse": rmse["feature_combination"],
                    "best_feature_combination_by_r2": r2["feature_combination"],
                    "best_mae": mae["mean_mae"],
                    "best_rmse": rmse["mean_rmse"],
                    "best_r2": r2["mean_r2"],
                    "is_all_feature_combination_best_by_r2": bool(r2["feature_combination"] == "statistics_shape_frequency"),
                }
            )
        return pd.DataFrame(rows)

    def _segment_effect(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (model, sensor_setting, combo), group in metrics.groupby(["model", "sensor_setting", "feature_combination"]):
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
                            "sensor_setting": sensor_setting,
                            "feature_combination": combo,
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

    def _best_segment_by_combo(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (model, sensor_setting, combo), group in metrics.groupby(["model", "sensor_setting", "feature_combination"]):
            mae = group.loc[group["mean_mae"].idxmin()]
            rmse = group.loc[group["mean_rmse"].idxmin()]
            r2 = group.loc[group["mean_r2"].idxmax()]
            segment = str(r2["segment_setting"])
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "sensor_setting": sensor_setting,
                    "feature_combination": combo,
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
        for (model, sensor_setting, combo), group in metrics.groupby(["model", "sensor_setting", "feature_combination"]):
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "sensor_setting": sensor_setting,
                    "feature_combination_type": self._combo_type(combo),
                    "feature_combination": combo,
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

    def _metrics_json(self, metrics: pd.DataFrame, best_combo: pd.DataFrame, best_segment: pd.DataFrame, r2_summary: pd.DataFrame) -> dict[str, Any]:
        best = metrics.loc[metrics["mean_mae"].idxmin()].to_dict()
        all_best_ratio = float(best_combo["is_all_feature_combination_best_by_r2"].mean()) if not best_combo.empty else 0.0
        return {
            "aggregation": f"mean_over_{len(CASE_SCOPE)}_leave_one_case_tests_then_mean_std_over_seeds",
            "primary_metric": "mean_mae",
            "best_overall_by_mae": best,
            "all_feature_combination_best_by_r2_ratio": all_best_ratio,
            "r2_positive_summary_by_type": r2_summary.groupby("feature_combination_type")["num_r2_positive"].sum().to_dict(),
        }

    def _write_common_outputs(self, paths, dataset, feature_table, mapping, combo_features, dry_summary) -> None:
        write_json(paths.execution_dir / "data" / "dataset_summary.json", dry_summary)
        write_csv(paths.execution_dir / "data" / "sensor_mapping.csv", mapping)
        write_json(paths.execution_dir / "data" / "sensor_setting_summary.json", {"sensor_setting": SENSOR_SETTING, "included_sensors": mapping["actual_sensor"].tolist()})
        write_json(paths.execution_dir / "data" / "case_filter_summary.json", {"used_cases": CASE_SCOPE, "rows_by_case": dry_summary["rows_by_case"]})
        write_json(paths.execution_dir / "data" / "target_summary.json", {"target_column": "VB", "count": int(dataset["VB"].notna().sum()), "min": float(dataset["VB"].min()), "max": float(dataset["VB"].max())})
        write_json(paths.execution_dir / "preprocessing" / "preprocessing_summary.json", {"output_type": "features", "no_load_excluded": True, "scaling": "StandardScaler fit on train split only"})
        write_json(paths.execution_dir / "preprocessing" / "feature_summary.json", {"feature_groups": FEATURE_GROUPS, "feature_naming": "{sensor_name}__{segment_setting}__{feature_name}"})
        write_json(paths.execution_dir / "preprocessing" / "feature_combination_summary.json", {"feature_combinations": FEATURE_COMBINATIONS, "sensor_setting": SENSOR_SETTING})
        write_json(paths.execution_dir / "preprocessing" / "shape_trace.json", [{"stage": "sensor_feature_table", "shape": list(feature_table.shape)}, {"stage": "feature_combination_tables", "count": len(combo_features)}])

    def _write_results(self, paths, config, results, dry_summary, run_config) -> None:
        experiment_id = config["experiment"]["experiment_id"]
        for frame in results.values():
            if isinstance(frame, pd.DataFrame) and "experiment_id" in frame.columns:
                frame["experiment_id"] = experiment_id
        if "best_overall_by_mae" in results["metrics_json"]:
            results["metrics_json"]["best_overall_by_mae"]["experiment_id"] = experiment_id
        write_csv(paths.execution_dir / "metrics" / "shift_metrics.csv", results["shift_metrics"])
        write_csv(paths.execution_dir / "metrics" / "seed_metrics.csv", results["seed_metrics"])
        write_csv(paths.execution_dir / "metrics" / "feature_combination_metrics.csv", results["feature_combination_metrics"])
        write_csv(paths.execution_dir / "metrics" / "segment_metrics.csv", results["segment_metrics"])
        write_csv(paths.execution_dir / "metrics" / "comparison_metrics.csv", results["comparison_metrics"])
        write_json(paths.execution_dir / "metrics" / "metrics.json", results["metrics_json"])
        write_csv(paths.execution_dir / "predictions" / "predictions.csv", results["predictions"])
        write_csv(paths.execution_dir / "predictions" / "prediction_errors.csv", results["predictions"].sort_values("absolute_error", ascending=False))
        write_csv(paths.execution_dir / "splits" / "split.csv", results["splits"])
        write_json(paths.execution_dir / "splits" / "split_summary.json", {"shift_scenarios": [f"{s}_to_{t}" for s, t in SHIFT_SCENARIOS]})
        write_json(paths.execution_dir / "splits" / "leakage_check.json", self._leakage_check(results["splits"]))
        write_json(paths.execution_dir / "analysis" / "analysis_summary.json", self._analysis_summary(results, dry_summary, run_config))
        write_csv(paths.execution_dir / "analysis" / "feature_group_contribution.csv", results["feature_group_contribution"])
        write_csv(paths.execution_dir / "analysis" / "feature_combination_effect.csv", results["feature_combination_effect"])
        write_csv(paths.execution_dir / "analysis" / "best_feature_combination.csv", results["best_feature_combination"])
        write_csv(paths.execution_dir / "analysis" / "segment_effect_under_feature_combination.csv", results["segment_effect_under_feature_combination"])
        write_csv(paths.execution_dir / "analysis" / "best_segment_by_feature_combination.csv", results["best_segment_by_feature_combination"])
        write_csv(paths.execution_dir / "analysis" / "r2_positive_count_summary.csv", results["r2_positive_count_summary"])
        write_csv(paths.execution_dir / "analysis" / "baseline_improvement.csv", results["baseline_improvement"])
        self._write_figures(paths, results)
        write_json(paths.execution_dir / "logs" / "environment.json", collect_environment(str(self.root)))
        write_json(paths.execution_dir / "logs" / "reproducibility.json", {"framework_version": __version__, "config_hash": config["config_hash"], "seeds": run_config.seeds})
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
            "scaling_fit_policy": "StandardScaler fit on train split only in sklearn Pipeline",
            "feature_leakage_check": "feature columns are sensor__segment__feature only; VB/case/domain/split identifiers excluded",
            "sensor_setting_check": SENSOR_SETTING,
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
            "feature_combinations": list(FEATURE_COMBINATIONS),
            "segment_settings": SEGMENT_SETTINGS,
            "best_overall_by_mae": results["metrics_json"]["best_overall_by_mae"],
            "all_feature_combination_best_by_r2_ratio": results["metrics_json"]["all_feature_combination_best_by_r2_ratio"],
        }

    def _write_figures(self, paths, results) -> None:
        import matplotlib.pyplot as plt

        metrics = results["feature_combination_metrics"]
        figures = paths.execution_dir / "figures"
        combo_order = list(FEATURE_COMBINATIONS)
        fig, ax = plt.subplots(figsize=(11, 5))
        perf = metrics.groupby(["model", "feature_combination"])["mean_r2"].mean().reset_index()
        for model, group in perf.groupby("model"):
            ordered = group.set_index("feature_combination").reindex(combo_order)
            ax.plot(combo_order, ordered["mean_r2"], marker="o", label=model)
        ax.set_ylabel("Mean R2")
        ax.tick_params(axis="x", rotation=35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "feature_combination_performance_by_model.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5))
        perf = metrics.groupby(["segment_setting", "feature_combination"])["mean_r2"].mean().reset_index()
        for segment, group in perf.groupby("segment_setting"):
            ordered = group.set_index("feature_combination").reindex(combo_order)
            ax.plot(combo_order, ordered["mean_r2"], marker="o", label=segment)
        ax.set_ylabel("Mean R2")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "feature_combination_performance_by_segment.png")
        plt.close(fig)

        best = results["best_feature_combination"]
        heat = pd.crosstab(best["segment_setting"], best["best_feature_combination_by_r2"]).reindex(index=SEGMENT_SETTINGS, columns=combo_order, fill_value=0)
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(heat.index)), heat.index)
        ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=35, ha="right")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(figures / "best_feature_combination_summary.png")
        plt.close(fig)

        seg = results["segment_effect_under_feature_combination"]
        fig, ax = plt.subplots(figsize=(10, 5))
        seg.loc[seg["baseline"] == "full_length"].groupby("segment_setting")["delta_r2"].mean().reindex(SEGMENT_SETTINGS).plot(kind="bar", ax=ax)
        ax.set_ylabel("Delta R2 vs full_length")
        fig.tight_layout()
        fig.savefig(figures / "segment_effect_under_feature_combination.png")
        plt.close(fig)

        bestseg = results["best_segment_by_feature_combination"]
        fig, ax = plt.subplots(figsize=(9, 4))
        bestseg["best_segment_by_r2"].value_counts().reindex(SEGMENT_SETTINGS, fill_value=0).plot(kind="bar", ax=ax)
        ax.set_ylabel("Best count")
        fig.tight_layout()
        fig.savefig(figures / "best_segment_by_feature_combination.png")
        plt.close(fig)

        r2 = results["r2_positive_count_summary"]
        fig, ax = plt.subplots(figsize=(8, 4))
        r2.groupby(["model", "feature_combination_type"])["ratio_r2_positive"].mean().unstack(0).plot(kind="bar", ax=ax)
        ax.set_ylabel("R2 positive ratio")
        fig.tight_layout()
        fig.savefig(figures / "r2_positive_count_comparison.png")
        plt.close(fig)

    def _write_report(self, paths, dry_summary, results, dry_run: bool) -> None:
        report = paths.execution_dir / "reports" / "report.md"
        if dry_run:
            body = f"""# H2.S3 Dry-run Report

Dry-run completed.

- Cases: {CASE_SCOPE}
- Leave-one-case-out cases: {CASE_SCOPE}
- Sensor setting: {SENSOR_SETTING}
- Sensors: {dry_summary['sensors']}
- Feature groups: {FEATURE_GROUPS}
- Feature combinations: {list(FEATURE_COMBINATIONS)}
- Segment settings: {SEGMENT_SETTINGS}
- Planned atomic evaluations: {dry_summary['planned_atomic_evaluations']}
- No-load excluded: true
"""
        else:
            summary = results["metrics_json"]
            contribution = results["feature_group_contribution"].to_string(index=False)
            best_combo = results["best_feature_combination"].to_string(index=False)
            best_segment = results["best_segment_by_feature_combination"].to_string(index=False)
            r2 = results["r2_positive_count_summary"].to_string(index=False)
            body = f"""# H2.S3 Feature Combination Effect on Segment-aware VB Prediction

## Objective

Evaluate whether statistics, shape, frequency, and their combinations improve segment-aware VB prediction with `sensor_setting=all_sensors`.

## Protocol

- Data files: `datasets/processed/mill_process_info.csv`, `datasets/processed/mill_signal_data.csv`
- Cases: {CASE_SCOPE}
- Leave-one-case-out cases: {CASE_SCOPE}
- Cross-test scenarios: {[f'{s}_to_{t}' for s, t in SHIFT_SCENARIOS]}
- Validation: none
- Sensor setting: {SENSOR_SETTING}
- Included sensors: {dry_summary['sensors']}
- Feature groups: {FEATURE_GROUPS}
- Feature combinations: {list(FEATURE_COMBINATIONS)}
- Segment settings: {SEGMENT_SETTINGS}
- Models: {dry_summary['models']}
- Seeds: {dry_summary['seeds']}
- No-load excluded: true

## Best Overall

```json
{json.dumps(summary['best_overall_by_mae'], indent=2, default=str)}
```

## Feature Group Contribution

```text
{contribution}
```

## Best Feature Combination

```text
{best_combo}
```

## Best Segment By Feature Combination

```text
{best_segment}
```

## R2 Positive Count

```text
{r2}
```

## RQ Summary

- RQ1: Inspect `metrics/feature_combination_metrics.csv`; performance changes across feature combinations.
- RQ2: Inspect `analysis/feature_combination_effect.csv`; combination benefits are condition-dependent when only some comparison rows improve.
- RQ3: Segment-aware effect is supported when non-baseline segment settings improve over `full_length` or `steady` in `analysis/segment_effect_under_feature_combination.csv`.
- RQ4: Best segment varies by feature combination when `analysis/best_segment_by_feature_combination.csv` contains multiple best segment settings.
"""
        report.write_text(body + paths.report_metadata_markdown(), encoding="utf-8")

    def _write_html(self, paths) -> None:
        md = paths.execution_dir / "reports" / "report.md"
        html = paths.execution_dir / "reports" / "report.html"
        body = md.read_text(encoding="utf-8").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>\n")
        html.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")

    def _combo_type(self, combo: str) -> str:
        n = len(FEATURE_COMBINATIONS[combo])
        if n == 1:
            return "single_feature_group"
        if n == 2:
            return "two_group_combination"
        return "three_group_combination"

    def _best_metric(self, metrics: pd.DataFrame) -> float | None:
        return None if metrics.empty else float(metrics["mean_mae"].min())


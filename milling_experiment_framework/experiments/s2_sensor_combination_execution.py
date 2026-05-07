from __future__ import annotations

import json
import math
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from milling_experiment_framework import __version__
from milling_experiment_framework.core.config import SCHEMA_VERSION, stable_hash
from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.experiment_logging.experiment_logger import ExperimentLogger
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


SENSOR_GROUPS = {
    "current": ["smcAC", "smcDC"],
    "vibration": ["vib_spindle", "vib_table"],
    "acoustic": ["AE_spindle", "AE_table"],
}
SENSOR_COMBINATIONS = {
    "current": ["current"],
    "vibration": ["vibration"],
    "acoustic": ["acoustic"],
    "current_vibration": ["current", "vibration"],
    "current_acoustic": ["current", "acoustic"],
    "vibration_acoustic": ["vibration", "acoustic"],
    "current_vibration_acoustic": ["current", "vibration", "acoustic"],
}
SINGLE_GROUPS = {"current", "vibration", "acoustic"}


class S2SensorCombinationExecution:
    """Run S2 segment-aware VB prediction over fixed sensor group combinations."""

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
        paths.prepare_standard_dirs()
        logger = ExperimentLogger(paths.execution_dir / "logs" / "run.log")
        logger.info(f"S2 execution started: {experiment_id}")
        if self.dry_run:
            logger.info("dry-run mode enabled")

        write_yaml(paths.execution_dir / "configs" / "input_config.yaml", raw_config)
        write_yaml(paths.execution_dir / "configs" / "resolved_config.yaml", config)
        try:
            helper = S1SegmentExecution(self.config_path, root=self.root, dry_run=True, seed_mode=self.seed_mode)
            validation = self._validate_inputs(run_config)
            dataset = helper._load_dataset(run_config)
            sensors = helper._sensor_columns(dataset)
            mapping = self._sensor_group_mapping(sensors)
            feature_table = helper._build_feature_table(dataset, sensors)
            combo_features = self._build_combination_tables(feature_table, mapping)
            validation.update(self._validate_features(dataset, sensors, mapping, combo_features))
            write_json(paths.execution_dir / "configs" / "config_validation.json", validation)
            dry_summary = self._dry_summary(dataset, sensors, mapping, combo_features, run_config)
            self._write_common_outputs(paths, dataset, feature_table, mapping, combo_features, dry_summary)
            if self.dry_run:
                write_json(paths.execution_dir / "metrics" / "metrics.json", {"dry_run": True, "summary": dry_summary})
                self._write_report(paths, dry_summary, {}, dry_run=True)
                self._write_html(paths)
                self._update_index(config, "finished", None)
                logger.info(f"S2 dry-run finished: {experiment_id}")
                return {"experiment_id": experiment_id, "dry_run": True, "summary": dry_summary, "execution_dir": str(paths.execution_dir)}

            results = self._run_grid(combo_features, mapping, run_config, logger)
            self._write_results(paths, config, results, dry_summary, run_config)
            (paths.execution_dir / "logs" / "error.log").touch()
            self._update_index(config, "finished", self._best_metric(results["sensor_combination_metrics"]))
            logger.info(f"S2 execution finished: {experiment_id}")
            return {"experiment_id": experiment_id, "metrics": results["metrics_json"], "execution_dir": str(paths.execution_dir)}
        except Exception as exc:
            error_text = traceback.format_exc()
            logger.error(str(exc))
            (paths.execution_dir / "logs" / "error.log").write_text(error_text, encoding="utf-8")
            write_json(paths.execution_dir / "configs" / "config_validation.json", {"ok": False, "errors": [str(exc)]})
            self._update_index(config, "failed", None, str(exc), str(paths.execution_dir / "logs" / "error.log"))
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
        return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f") + f"_S2_sensor_combination_segment_aware_VB_prediction_{suffix}"

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
        config["sensor_groups"] = SENSOR_GROUPS
        config["sensor_combinations"] = SENSOR_COMBINATIONS
        config["segment_settings"] = SEGMENT_SETTINGS
        config["feature_names"] = FEATURE_NAMES
        config["seeds"] = run_config.seeds
        config["dry_run"] = self.dry_run
        config["scaling_policy"] = "StandardScaler fit on train split only for both RandomForest and MLP pipelines"
        config["config_hash"] = stable_hash(config)
        return config

    def _validate_inputs(self, run_config: S1RunConfig) -> dict[str, Any]:
        errors = []
        for path in [run_config.process_info_path, run_config.signal_data_path, run_config.heuristic_sequence_path]:
            if not path.exists():
                errors.append(f"Missing required file: {path}")
        return {"ok": not errors, "errors": errors, "warnings": []}

    def _sensor_group_mapping(self, sensors: list[str]) -> pd.DataFrame:
        rows = []
        for group, expected in SENSOR_GROUPS.items():
            for sensor in expected:
                rows.append(
                    {
                        "sensor_group": group,
                        "expected_sensor": sensor,
                        "actual_sensor": sensor if sensor in sensors else None,
                        "status": "mapped" if sensor in sensors else "missing",
                    }
                )
        mapping = pd.DataFrame(rows)
        missing = mapping.loc[mapping["status"] == "missing"]
        if not missing.empty:
            raise ValueError(f"Sensor group mapping failed: {missing.to_dict(orient='records')}")
        return mapping

    def _build_combination_tables(self, feature_table: pd.DataFrame, mapping: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
        sample_cols = ["sample_id", "dataset_run_id", "case", "run", "domain_id", "VB"]
        tables: dict[tuple[str, str], pd.DataFrame] = {}
        group_to_sensors = mapping.groupby("sensor_group")["actual_sensor"].apply(list).to_dict()
        for segment in SEGMENT_SETTINGS:
            segment_rows = feature_table.loc[feature_table["segment_setting"] == segment]
            base = segment_rows[sample_cols].drop_duplicates().reset_index(drop=True)
            for combo, groups in SENSOR_COMBINATIONS.items():
                sensors = [sensor for group in groups for sensor in group_to_sensors[group]]
                table = base.copy()
                for sensor in sensors:
                    sensor_rows = segment_rows.loc[segment_rows["sensor"] == sensor, sample_cols + FEATURE_NAMES].copy()
                    rename = {name: f"{sensor}__{segment}__{name}" for name in FEATURE_NAMES}
                    sensor_rows = sensor_rows.rename(columns=rename)
                    table = table.merge(sensor_rows[sample_cols + list(rename.values())], on=sample_cols, how="inner", validate="one_to_one")
                tables[(combo, segment)] = table
        return tables

    def _validate_features(self, dataset: pd.DataFrame, sensors: list[str], mapping: pd.DataFrame, combo_features: dict[tuple[str, str], pd.DataFrame]) -> dict[str, Any]:
        errors = []
        if sorted(dataset["case"].unique().tolist()) != CASE_SCOPE:
            errors.append(f"case filtering failed: {sorted(dataset['case'].unique().tolist())}")
        if "VB" not in dataset.columns:
            errors.append("VB target column missing")
        if any("no_load" in setting for setting in SEGMENT_SETTINGS):
            errors.append("no-load setting included")
        leakage_names = {"VB", "case", "domain_id", "pair_id", "dataset_run_id", "split", "source_domain", "target_domain"}
        for key, table in combo_features.items():
            feature_cols = [c for c in table.columns if "__" in c]
            if leakage_names & set(feature_cols):
                errors.append(f"leakage feature in {key}: {sorted(leakage_names & set(feature_cols))}")
            if table.empty:
                errors.append(f"empty feature table for {key}")
        return {
            "ok": not errors,
            "errors": errors,
            "warnings": [],
            "feature_validation": {
                "num_combinations": len(SENSOR_COMBINATIONS),
                "num_segment_settings": len(SEGMENT_SETTINGS),
                "sensors": sensors,
                "feature_tables": {f"{c}/{s}": list(t.shape) for (c, s), t in combo_features.items()},
            },
        }

    def _dry_summary(self, dataset: pd.DataFrame, sensors: list[str], mapping: pd.DataFrame, combo_features: dict[tuple[str, str], pd.DataFrame], run_config: S1RunConfig) -> dict[str, Any]:
        return {
            "case_scope": CASE_SCOPE,
            "rows_by_case": dataset.groupby("case").size().to_dict(),
            "target_column": "VB",
            "no_load_excluded": True,
            "sensors": sensors,
            "sensor_groups": SENSOR_GROUPS,
            "sensor_combinations": SENSOR_COMBINATIONS,
            "segment_settings": SEGMENT_SETTINGS,
            "feature_names": FEATURE_NAMES,
            "models": run_config.models,
            "seeds": run_config.seeds,
            "planned_atomic_evaluations": len(SHIFT_SCENARIOS) * len(run_config.models) * len(SENSOR_COMBINATIONS) * len(SEGMENT_SETTINGS) * len(run_config.seeds),
            "combination_feature_shapes": {f"{combo}/{segment}": list(table.shape) for (combo, segment), table in combo_features.items()},
            "sensor_group_mapping": mapping.to_dict(orient="records"),
        }

    def _run_grid(self, combo_features: dict[tuple[str, str], pd.DataFrame], mapping: pd.DataFrame, run_config: S1RunConfig, logger: ExperimentLogger) -> dict[str, Any]:
        helper = S1SegmentExecution(self.config_path, root=self.root, dry_run=False, seed_mode=self.seed_mode)
        shift_rows = []
        prediction_frames = []
        split_frames = []
        total = len(run_config.models) * len(SENSOR_COMBINATIONS) * len(SEGMENT_SETTINGS) * len(run_config.seeds) * len(SHIFT_SCENARIOS)
        done = 0
        for model_name in run_config.models:
            for sensor_combo, groups in SENSOR_COMBINATIONS.items():
                included_sensors = self._included_sensors(mapping, sensor_combo)
                included_groups = ",".join(groups)
                for segment in SEGMENT_SETTINGS:
                    data = combo_features[(sensor_combo, segment)]
                    feature_cols = [c for c in data.columns if "__" in c]
                    for seed in run_config.seeds:
                        for source, target in SHIFT_SCENARIOS:
                            done += 1
                            result, preds, splits = self._run_atomic(
                                helper,
                                model_name,
                                sensor_combo,
                                included_groups,
                                included_sensors,
                                segment,
                                seed,
                                source,
                                target,
                                data,
                                feature_cols,
                                run_config,
                            )
                            shift_rows.append(result)
                            prediction_frames.append(preds)
                            split_frames.append(splits)
                            if done % 500 == 0 or done == total:
                                logger.info(f"S2 progress {done}/{total}")
        shift_metrics = pd.DataFrame(shift_rows)
        seed_metrics = self._seed_metrics(shift_metrics)
        combo_metrics = self._combo_metrics(seed_metrics)
        segment_metrics = self._segment_metrics(seed_metrics)
        sensor_effect = self._sensor_combination_effect(combo_metrics)
        best_combo = self._best_sensor_combination(combo_metrics)
        segment_effect = self._segment_effect(combo_metrics)
        best_segment = self._best_segment_by_combo(combo_metrics)
        r2_summary = self._r2_positive_summary(combo_metrics)
        baseline = pd.concat([sensor_effect.assign(comparison_type="sensor_combination"), segment_effect.assign(comparison_type="segment")], ignore_index=True, sort=False)
        metrics_json = self._metrics_json(combo_metrics, best_combo, best_segment, r2_summary)
        return {
            "shift_metrics": shift_metrics,
            "seed_metrics": seed_metrics,
            "sensor_combination_metrics": combo_metrics,
            "segment_metrics": segment_metrics,
            "comparison_metrics": baseline,
            "predictions": pd.concat(prediction_frames, ignore_index=True),
            "splits": pd.concat(split_frames, ignore_index=True),
            "sensor_combination_effect": sensor_effect,
            "best_sensor_combination": best_combo,
            "segment_effect_under_sensor_combination": segment_effect,
            "best_segment_by_sensor_combination": best_segment,
            "r2_positive_count_summary": r2_summary,
            "baseline_improvement": baseline,
            "metrics_json": metrics_json,
        }

    def _run_atomic(self, helper: S1SegmentExecution, model_name: str, sensor_combo: str, groups: str, sensors: str, segment: str, seed: int, source: str, target: str, data: pd.DataFrame, feature_cols: list[str], run_config: S1RunConfig):
        split = helper._split_frame(data, source, target)
        train = split.loc[split["split"] == "train"]
        test = split.loc[split["split"] == "test"]
        model = helper._model(model_name, seed, run_config)
        model.fit(train[feature_cols], train["VB"])
        y_pred = model.predict(test[feature_cols])
        metrics = helper._metrics(test["VB"].to_numpy(), y_pred)
        child = f"S2_{model_name}_{sensor_combo}_{segment}_{source}_to_{target}_seed_{seed}"
        row = {
            "experiment_id": None,
            "child_execution_key": child,
            "model": model_name,
            "sensor_combination": sensor_combo,
            "included_sensor_groups": groups,
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
        preds["model"] = model_name
        preds["sensor_combination"] = sensor_combo
        preds["included_sensor_groups"] = groups
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
        split_out["model"] = model_name
        split_out["sensor_combination"] = sensor_combo
        split_out["segment_setting"] = segment
        split_out["seed"] = seed
        split_out["source_domain"] = source
        split_out["target_domain"] = target
        return row, preds, split_out

    def _seed_metrics(self, shift_metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for keys, group in shift_metrics.groupby(["model", "sensor_combination", "segment_setting", "seed"]):
            model, combo, segment, seed = keys
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "sensor_combination": combo,
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
        for keys, group in seed_metrics.groupby(["model", "sensor_combination", "segment_setting"]):
            model, combo, segment = keys
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "sensor_combination": combo,
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
        for keys, group in seed_metrics.groupby(["model", "segment_setting"]):
            model, segment = keys
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
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

    def _sensor_combination_effect(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        baseline_map = {
            "current_vibration": ["current", "vibration"],
            "current_acoustic": ["current", "acoustic"],
            "vibration_acoustic": ["vibration", "acoustic"],
            "current_vibration_acoustic": ["current", "vibration", "acoustic"],
        }
        for (model, segment), group in metrics.groupby(["model", "segment_setting"]):
            for combo, baselines in baseline_map.items():
                c = group.loc[group["sensor_combination"] == combo]
                if c.empty:
                    continue
                c = c.iloc[0]
                for baseline in baselines:
                    b = group.loc[group["sensor_combination"] == baseline]
                    if b.empty:
                        continue
                    b = b.iloc[0]
                    rows.append(
                        {
                            "experiment_id": None,
                            "model": model,
                            "segment_setting": segment,
                            "comparison_sensor_combination": combo,
                            "baseline_sensor_group": baseline,
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

    def _best_sensor_combination(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (model, segment), group in metrics.groupby(["model", "segment_setting"]):
            mae = group.loc[group["mean_mae"].idxmin()]
            rmse = group.loc[group["mean_rmse"].idxmin()]
            r2 = group.loc[group["mean_r2"].idxmax()]
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "segment_setting": segment,
                    "best_sensor_combination_by_mae": mae["sensor_combination"],
                    "best_sensor_combination_by_rmse": rmse["sensor_combination"],
                    "best_sensor_combination_by_r2": r2["sensor_combination"],
                    "best_mae": mae["mean_mae"],
                    "best_rmse": rmse["mean_rmse"],
                    "best_r2": r2["mean_r2"],
                    "is_all_sensor_combination_best_by_r2": bool(r2["sensor_combination"] == "current_vibration_acoustic"),
                }
            )
        return pd.DataFrame(rows)

    def _segment_effect(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (model, combo), group in metrics.groupby(["model", "sensor_combination"]):
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
                            "sensor_combination": combo,
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
        for (model, combo), group in metrics.groupby(["model", "sensor_combination"]):
            mae = group.loc[group["mean_mae"].idxmin()]
            rmse = group.loc[group["mean_rmse"].idxmin()]
            r2 = group.loc[group["mean_r2"].idxmax()]
            segment = str(r2["segment_setting"])
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "sensor_combination": combo,
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
        for (model, combo), group in metrics.groupby(["model", "sensor_combination"]):
            rows.append(
                {
                    "experiment_id": None,
                    "model": model,
                    "sensor_combination_type": self._combo_type(combo),
                    "sensor_combination": combo,
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
        all_best_ratio = float(best_combo["is_all_sensor_combination_best_by_r2"].mean()) if not best_combo.empty else 0.0
        return {
            "aggregation": "mean_over_6_domain_shifts_then_mean_std_over_seeds",
            "primary_metric": "mean_mae",
            "best_overall_by_mae": best,
            "all_sensor_combination_best_by_r2_ratio": all_best_ratio,
            "r2_positive_summary_by_type": r2_summary.groupby("sensor_combination_type")["num_r2_positive"].sum().to_dict(),
        }

    def _write_common_outputs(self, paths, dataset, feature_table, mapping, combo_features, dry_summary) -> None:
        write_json(paths.execution_dir / "data" / "dataset_summary.json", dry_summary)
        write_csv(paths.execution_dir / "data" / "sensor_group_mapping.csv", mapping)
        write_json(paths.execution_dir / "data" / "case_filter_summary.json", {"used_cases": CASE_SCOPE, "rows_by_case": dry_summary["rows_by_case"]})
        write_json(paths.execution_dir / "data" / "target_summary.json", {"target_column": "VB", "count": int(dataset["VB"].notna().sum()), "min": float(dataset["VB"].min()), "max": float(dataset["VB"].max())})
        write_json(paths.execution_dir / "preprocessing" / "preprocessing_summary.json", {"output_type": "features", "no_load_excluded": True, "scaling": "StandardScaler fit on train split only"})
        write_json(paths.execution_dir / "preprocessing" / "feature_summary.json", {"feature_names": FEATURE_NAMES, "feature_naming": "{sensor_name}__{segment_setting}__{feature_name}"})
        write_json(paths.execution_dir / "preprocessing" / "sensor_combination_summary.json", {"sensor_groups": SENSOR_GROUPS, "sensor_combinations": SENSOR_COMBINATIONS})
        write_json(paths.execution_dir / "preprocessing" / "shape_trace.json", [{"stage": "sensor_feature_table", "shape": list(feature_table.shape)}, {"stage": "combination_tables", "count": len(combo_features)}])

    def _write_results(self, paths, config, results, dry_summary, run_config) -> None:
        experiment_id = config["experiment"]["experiment_id"]
        for key, frame in results.items():
            if isinstance(frame, pd.DataFrame) and "experiment_id" in frame.columns:
                frame["experiment_id"] = experiment_id
        if "best_overall_by_mae" in results["metrics_json"]:
            results["metrics_json"]["best_overall_by_mae"]["experiment_id"] = experiment_id
        write_csv(paths.execution_dir / "metrics" / "shift_metrics.csv", results["shift_metrics"])
        write_csv(paths.execution_dir / "metrics" / "seed_metrics.csv", results["seed_metrics"])
        write_csv(paths.execution_dir / "metrics" / "sensor_combination_metrics.csv", results["sensor_combination_metrics"])
        write_csv(paths.execution_dir / "metrics" / "segment_metrics.csv", results["segment_metrics"])
        write_csv(paths.execution_dir / "metrics" / "comparison_metrics.csv", results["comparison_metrics"])
        write_json(paths.execution_dir / "metrics" / "metrics.json", results["metrics_json"])
        write_csv(paths.execution_dir / "predictions" / "predictions.csv", results["predictions"])
        write_csv(paths.execution_dir / "predictions" / "prediction_errors.csv", results["predictions"].sort_values("absolute_error", ascending=False))
        write_csv(paths.execution_dir / "splits" / "split.csv", results["splits"])
        write_json(paths.execution_dir / "splits" / "split_summary.json", {"shift_scenarios": [f"{s}_to_{t}" for s, t in SHIFT_SCENARIOS]})
        write_json(paths.execution_dir / "splits" / "leakage_check.json", self._leakage_check(results["splits"]))
        write_json(paths.execution_dir / "analysis" / "analysis_summary.json", self._analysis_summary(results, dry_summary, run_config))
        write_csv(paths.execution_dir / "analysis" / "sensor_combination_effect.csv", results["sensor_combination_effect"])
        write_csv(paths.execution_dir / "analysis" / "best_sensor_combination.csv", results["best_sensor_combination"])
        write_csv(paths.execution_dir / "analysis" / "segment_effect_under_sensor_combination.csv", results["segment_effect_under_sensor_combination"])
        write_csv(paths.execution_dir / "analysis" / "best_segment_by_sensor_combination.csv", results["best_segment_by_sensor_combination"])
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
        }

    def _analysis_summary(self, results: dict[str, Any], dry_summary: dict[str, Any], run_config: S1RunConfig) -> dict[str, Any]:
        return {
            "analysis_type": "Exploratory",
            "num_atomic_evaluations": int(len(results["shift_metrics"])),
            "skipped_condition_count": len(self.skipped),
            "skip_reasons": pd.DataFrame(self.skipped)["reason"].value_counts().to_dict() if self.skipped else {},
            "models": run_config.models,
            "seeds": run_config.seeds,
            "sensor_combinations": list(SENSOR_COMBINATIONS),
            "segment_settings": SEGMENT_SETTINGS,
            "best_overall_by_mae": results["metrics_json"]["best_overall_by_mae"],
            "all_sensor_combination_best_by_r2_ratio": results["metrics_json"]["all_sensor_combination_best_by_r2_ratio"],
        }

    def _write_figures(self, paths, results) -> None:
        import matplotlib.pyplot as plt

        metrics = results["sensor_combination_metrics"]
        figures = paths.execution_dir / "figures"
        combo_order = list(SENSOR_COMBINATIONS)
        fig, ax = plt.subplots(figsize=(11, 5))
        perf = metrics.groupby(["model", "sensor_combination"])["mean_r2"].mean().reset_index()
        for model, group in perf.groupby("model"):
            ordered = group.set_index("sensor_combination").reindex(combo_order)
            ax.plot(combo_order, ordered["mean_r2"], marker="o", label=model)
        ax.set_ylabel("Mean R2")
        ax.tick_params(axis="x", rotation=35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "sensor_combination_performance_by_model.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5))
        perf = metrics.groupby(["segment_setting", "sensor_combination"])["mean_r2"].mean().reset_index()
        for segment, group in perf.groupby("segment_setting"):
            ordered = group.set_index("sensor_combination").reindex(combo_order)
            ax.plot(combo_order, ordered["mean_r2"], marker="o", label=segment)
        ax.set_ylabel("Mean R2")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "sensor_combination_performance_by_segment.png")
        plt.close(fig)

        best = results["best_sensor_combination"]
        heat = pd.crosstab(best["segment_setting"], best["best_sensor_combination_by_r2"]).reindex(index=SEGMENT_SETTINGS, columns=combo_order, fill_value=0)
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(heat.index)), heat.index)
        ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=35, ha="right")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(figures / "best_sensor_combination_summary.png")
        plt.close(fig)

        seg = results["segment_effect_under_sensor_combination"]
        fig, ax = plt.subplots(figsize=(10, 5))
        seg.loc[seg["baseline"] == "full_length"].groupby("segment_setting")["delta_r2"].mean().reindex(SEGMENT_SETTINGS).plot(kind="bar", ax=ax)
        ax.set_ylabel("Delta R2 vs full_length")
        fig.tight_layout()
        fig.savefig(figures / "segment_effect_under_sensor_combination.png")
        plt.close(fig)

        bestseg = results["best_segment_by_sensor_combination"]
        fig, ax = plt.subplots(figsize=(9, 4))
        bestseg["best_segment_by_r2"].value_counts().reindex(SEGMENT_SETTINGS, fill_value=0).plot(kind="bar", ax=ax)
        ax.set_ylabel("Best count")
        fig.tight_layout()
        fig.savefig(figures / "best_segment_by_sensor_combination.png")
        plt.close(fig)

        r2 = results["r2_positive_count_summary"]
        fig, ax = plt.subplots(figsize=(8, 4))
        r2.groupby(["model", "sensor_combination_type"])["ratio_r2_positive"].mean().unstack(0).plot(kind="bar", ax=ax)
        ax.set_ylabel("R2 positive ratio")
        fig.tight_layout()
        fig.savefig(figures / "r2_positive_count_comparison.png")
        plt.close(fig)

    def _write_report(self, paths, dry_summary, results, dry_run: bool) -> None:
        report = paths.execution_dir / "reports" / "report.md"
        if dry_run:
            body = f"""# S2 Dry-run Report

Dry-run completed.

- Cases: {CASE_SCOPE}
- Domain pairs: {DOMAIN_CASES}
- Sensors: {dry_summary['sensors']}
- Sensor groups: {SENSOR_GROUPS}
- Sensor combinations: {list(SENSOR_COMBINATIONS)}
- Segment settings: {SEGMENT_SETTINGS}
- Planned atomic evaluations: {dry_summary['planned_atomic_evaluations']}
- No-load excluded: true
"""
        else:
            summary = results["metrics_json"]
            best_combo = results["best_sensor_combination"].to_string(index=False)
            best_segment = results["best_segment_by_sensor_combination"].to_string(index=False)
            r2 = results["r2_positive_count_summary"].to_string(index=False)
            body = f"""# S2 Sensor Combination Effect on Segment-aware VB Prediction

## Objective

Evaluate whether sensor group combinations improve segment-aware VB prediction under fixed A/B/C domain shifts.

## Protocol

- Data files: `datasets/processed/mill_process_info_enabled.csv`, `datasets/processed/mill_signal_data_enabled.csv`
- Cases: {CASE_SCOPE}
- Domain pairs: {DOMAIN_CASES}
- Shift scenarios: {[f'{s}_to_{t}' for s, t in SHIFT_SCENARIOS]}
- Sensor groups: {SENSOR_GROUPS}
- Sensor combinations: {list(SENSOR_COMBINATIONS)}
- Segment settings: {SEGMENT_SETTINGS}
- Models: {dry_summary['models']}
- Seeds: {dry_summary['seeds']}
- No-load excluded: true

## Best Overall

```json
{json.dumps(summary['best_overall_by_mae'], indent=2, default=str)}
```

## Best Sensor Combination

```text
{best_combo}
```

## Best Segment By Sensor Combination

```text
{best_segment}
```

## R2 Positive Count

```text
{r2}
```

## RQ Summary

- RQ1: Inspect `analysis/sensor_combination_effect.csv`; improvements are condition-dependent when only some combination-baseline rows improve.
- RQ2: `current_vibration_acoustic` is not always best unless `is_all_sensor_combination_best_by_r2` is true for most rows.
- RQ3: Segment-aware effect is supported when non-baseline segment settings improve over `full_length` or `steady` in `analysis/segment_effect_under_sensor_combination.csv`.
- RQ4: Best segment varies by sensor combination when `analysis/best_segment_by_sensor_combination.csv` contains multiple best segment settings.
"""
        report.write_text(body, encoding="utf-8")

    def _write_html(self, paths) -> None:
        md = paths.execution_dir / "reports" / "report.md"
        html = paths.execution_dir / "reports" / "report.html"
        body = md.read_text(encoding="utf-8").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>\n")
        html.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")

    def _included_sensors(self, mapping: pd.DataFrame, combo: str) -> str:
        groups = SENSOR_COMBINATIONS[combo]
        sensors = mapping.loc[mapping["sensor_group"].isin(groups), "actual_sensor"].tolist()
        return ",".join(sensors)

    def _combo_type(self, combo: str) -> str:
        n = len(SENSOR_COMBINATIONS[combo])
        if n == 1:
            return "single_group"
        if n == 2:
            return "two_group_combination"
        return "three_group_combination"

    def _best_metric(self, metrics: pd.DataFrame) -> float | None:
        return None if metrics.empty else float(metrics["mean_mae"].min())

    def _update_index(self, config: dict[str, Any], status: str, best_metric: float | None, error_message: str | None = None, error_log: str | None = None) -> None:
        index_path = self.root / "experiments" / "index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "experiment_id": config["experiment"]["experiment_id"],
            "experiment_name": "S2_sensor_combination_effect_on_segment_aware_VB_prediction",
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

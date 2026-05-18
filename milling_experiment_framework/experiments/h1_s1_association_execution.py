from __future__ import annotations

import json
import math
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

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
    S1RunConfig,
    S1SegmentExecution,
)
from milling_experiment_framework.utils.io import write_csv, write_json, write_yaml
from milling_experiment_framework.utils.paths import ExperimentPaths


PAIR_BY_CASE = {case: pair for pair, cases in DOMAIN_CASES.items() for case in cases}


class H1S1AssociationExecution:
    """EDA execution for segment feature to VB association, excluding no-load settings."""

    def __init__(self, config_path: str | Path, root: str | Path = ".", dry_run: bool = False):
        self.config_path = Path(config_path)
        self.root = Path(root).resolve()
        self.dry_run = dry_run
        self.skipped: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        raw_config = self._read_config()
        experiment_id = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f_H1_S1_feture_VB_segment_association_eda")
        paths = ExperimentPaths(self.root, experiment_id)
        config = self._resolved_config(raw_config, experiment_id)
        paths.apply_to_config(config)
        paths.prepare_standard_dirs()
        logger = ExperimentLogger(paths.execution_dir / "logs" / "run.log")
        logger.info(f"H1.S1 association execution started: {experiment_id}")
        if self.dry_run:
            logger.info("dry-run mode enabled")
        write_yaml(paths.execution_dir / "configs" / "input_config.yaml", raw_config)
        write_yaml(paths.execution_dir / "configs" / "resolved_config.yaml", config)

        try:
            helper, run_config = self._helper(raw_config)
            validation = self._validate_inputs(run_config)
            dataset = helper._load_dataset(run_config)
            sensors = helper._sensor_columns(dataset)
            feature_table = helper._build_feature_table(dataset, sensors)
            validation.update(self._validate_feature_table(dataset, feature_table, sensors))
            write_json(paths.execution_dir / "configs" / "config_validation.json", validation)
            dry_summary = self._dry_summary(dataset, feature_table, sensors, validation)
            self._write_common_outputs(paths, dataset, feature_table, sensors, dry_summary, validation)
            if self.dry_run:
                write_json(paths.execution_dir / "metrics" / "metrics.json", {"dry_run": True, "summary": dry_summary})
                self._write_report(paths, dry_summary, {}, dry_run=True)
                self._write_html(paths)
                self._update_index(config, "finished", None)
                logger.info(f"H1.S1 dry-run finished: {experiment_id}")
                return {"experiment_id": experiment_id, "dry_run": True, "summary": dry_summary, "execution_dir": str(paths.execution_dir)}

            results = self._analyze(feature_table, experiment_id, logger)
            self._write_results(paths, dataset, feature_table, sensors, dry_summary, validation, results)
            (paths.execution_dir / "logs" / "error.log").touch()
            self._update_index(config, "finished", self._best_metric(results["association_segment_level"]))
            logger.info(f"H1.S1 association execution finished: {experiment_id}")
            return {"experiment_id": experiment_id, "summary": results["analysis_summary"], "execution_dir": str(paths.execution_dir)}
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

    def _resolved_config(self, raw_config: dict[str, Any], experiment_id: str) -> dict[str, Any]:
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
        config["segment_settings"] = SEGMENT_SETTINGS
        config["excluded_segments"] = ["no_load", "no_load_entry", "no_load_steady", "no_load_exit", "all_no_load_related_combinations"]
        config["feature_names"] = FEATURE_NAMES
        config["association_measures"] = ["pearson", "spearman", "mutual_information"]
        config["deferred_association_measures"] = ["distance_correlation", "kendall_tau"]
        config["config_hash"] = stable_hash(config)
        return config

    def _helper(self, config: dict[str, Any]) -> tuple[S1SegmentExecution, S1RunConfig]:
        helper = S1SegmentExecution(self.config_path, root=self.root, dry_run=True, seed_mode="initial")
        dataset_cfg = config["dataset"]
        run_config = S1RunConfig(
            process_info_path=Path(dataset_cfg["process_info_path"]),
            signal_data_path=Path(dataset_cfg["signal_data_path"]),
            heuristic_sequence_path=Path(dataset_cfg["heuristic_sequence_path"]),
            seeds=[],
            models=[],
            random_forest_params={},
            mlp_params={},
        )
        return helper, run_config

    def _validate_inputs(self, run_config: S1RunConfig) -> dict[str, Any]:
        errors = []
        warnings = []
        for path in [run_config.process_info_path, run_config.signal_data_path, run_config.heuristic_sequence_path]:
            if not path.exists():
                errors.append(f"Missing required file: {path}")
        process_cols = pd.read_csv(run_config.process_info_path, nrows=0).columns.tolist() if run_config.process_info_path.exists() else []
        signal_cols = pd.read_csv(run_config.signal_data_path, nrows=0).columns.tolist() if run_config.signal_data_path.exists() else []
        heuristic_cols = pd.read_csv(run_config.heuristic_sequence_path, nrows=0).columns.tolist() if run_config.heuristic_sequence_path.exists() else []
        for col in ["case", "run", "VB"]:
            if col not in process_cols:
                errors.append(f"process info missing column: {col}")
        for col in ["case", "run"]:
            if col not in signal_cols:
                errors.append(f"signal data missing column: {col}")
        for col in ["case", "run", "idx_noload_end", "idx_start", "idx_end"]:
            if col not in heuristic_cols:
                errors.append(f"heuristic sequence missing column: {col}")
        if any("no_load" in setting for setting in SEGMENT_SETTINGS):
            errors.append("no-load setting included in active segment settings")
        return {"ok": not errors, "errors": errors, "warnings": warnings}

    def _validate_feature_table(self, dataset: pd.DataFrame, feature_table: pd.DataFrame, sensors: list[str]) -> dict[str, Any]:
        errors = []
        warnings = []
        found_cases = sorted(dataset["case"].unique().tolist())
        if found_cases != CASE_SCOPE:
            errors.append(f"case filtering failed: {found_cases}")
        if "VB" not in dataset.columns:
            errors.append("VB target column missing")
        if not sensors:
            errors.append("no sensors identified")
        expected_rows = len(dataset) * len(sensors) * len(SEGMENT_SETTINGS)
        if len(feature_table) != expected_rows:
            errors.append(f"feature table row mismatch: expected={expected_rows}, actual={len(feature_table)}")
        if feature_table[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).isna().any().any():
            warnings.append("feature table contains NaN/inf values after feature extraction")
        return {
            "feature_validation": {
                "expected_feature_rows": expected_rows,
                "actual_feature_rows": len(feature_table),
                "sensors": sensors,
                "segment_settings": SEGMENT_SETTINGS,
            },
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
        }

    def _dry_summary(self, dataset: pd.DataFrame, feature_table: pd.DataFrame, sensors: list[str], validation: dict[str, Any]) -> dict[str, Any]:
        labeled_by_case = dataset.groupby("case").size().to_dict()
        availability = (
            feature_table.groupby(["sensor", "segment_setting"])
            .size()
            .reset_index(name="num_feature_rows")
            .to_dict(orient="records")
        )
        n_samples_by_condition = (
            feature_table.groupby(["case", "sensor", "segment_setting"])
            .size()
            .reset_index(name="n_samples")
            .to_dict(orient="records")
        )
        insufficient = [row for row in n_samples_by_condition if row["n_samples"] < 3]
        return {
            "data_files_exist": validation["ok"] or not validation["errors"],
            "case_scope": CASE_SCOPE,
            "rows_by_case": labeled_by_case,
            "target_column": "VB",
            "no_load_excluded": True,
            "sensors": sensors,
            "segment_settings": SEGMENT_SETTINGS,
            "feature_names": FEATURE_NAMES,
            "association_measures": ["pearson", "spearman", "mutual_information"],
            "deferred": ["distance_correlation", "kendall_tau"],
            "feature_availability": availability,
            "n_samples_by_condition": n_samples_by_condition,
            "insufficient_conditions": insufficient,
            "planned_feature_level_conditions": len(CASE_SCOPE) * len(sensors) * len(SEGMENT_SETTINGS) * len(FEATURE_NAMES),
        }

    def _analyze(self, feature_table: pd.DataFrame, experiment_id: str, logger: ExperimentLogger) -> dict[str, Any]:
        feature_level = self._feature_level_association(feature_table, experiment_id, logger)
        segment_level = self._segment_level_summary(feature_level, experiment_id)
        baseline = self._baseline_comparison(segment_level, experiment_id)
        consistency = self._sensor_consistency(segment_level, experiment_id)
        pair_summary = self._pair_level_summary(segment_level, experiment_id)
        ranking = self._segment_ranking(segment_level, baseline, experiment_id)
        analysis_summary = self._analysis_summary(feature_level, segment_level, baseline, consistency)
        return {
            "association_feature_level": feature_level,
            "association_segment_level": segment_level,
            "association_case_level": segment_level.copy(),
            "association_pair_level": pair_summary.copy(),
            "baseline_comparison": baseline,
            "sensor_consistency": consistency,
            "pair_level_association_summary": pair_summary,
            "segment_ranking": ranking,
            "analysis_summary": analysis_summary,
        }

    def _feature_level_association(self, feature_table: pd.DataFrame, experiment_id: str, logger: ExperimentLogger) -> pd.DataFrame:
        rows = []
        for (case_id, sensor, segment), group in feature_table.groupby(["case", "sensor", "segment_setting"], dropna=False):
            y = group["VB"].to_numpy(dtype="float64")
            pair_id = PAIR_BY_CASE[int(case_id)]
            n_samples = len(group)
            for feature_name in FEATURE_NAMES:
                x = group[feature_name].to_numpy(dtype="float64")
                row = {
                    "experiment_id": experiment_id,
                    "case_id": int(case_id),
                    "pair_id": pair_id,
                    "dataset_run_count": int(n_samples),
                    "sensor_name": sensor,
                    "segment_setting": segment,
                    "feature_name": feature_name,
                    "pearson_r": np.nan,
                    "pearson_abs": np.nan,
                    "pearson_p_value": np.nan,
                    "spearman_r": np.nan,
                    "spearman_abs": np.nan,
                    "spearman_p_value": np.nan,
                    "mutual_information": np.nan,
                    "n_samples": int(n_samples),
                    "deferred_distance_correlation": "deferred",
                    "deferred_kendall_tau": "deferred",
                    "status": "ok",
                    "skip_reason": None,
                }
                finite = np.isfinite(x) & np.isfinite(y)
                x_f = x[finite]
                y_f = y[finite]
                if len(x_f) < 3:
                    row["status"] = "skipped"
                    row["skip_reason"] = "n_samples_lt_3"
                    self.skipped.append(row.copy())
                    rows.append(row)
                    continue
                if len(np.unique(x_f)) < 2:
                    row["status"] = "skipped"
                    row["skip_reason"] = "constant_feature"
                    row["mutual_information"] = 0.0
                    logger.info(f"constant feature skipped for correlation: case={case_id}, sensor={sensor}, segment={segment}, feature={feature_name}")
                    self.skipped.append(row.copy())
                    rows.append(row)
                    continue
                if len(np.unique(y_f)) < 2:
                    row["status"] = "skipped"
                    row["skip_reason"] = "constant_target"
                    row["mutual_information"] = 0.0
                    self.skipped.append(row.copy())
                    rows.append(row)
                    continue
                pearson = pearsonr(x_f, y_f)
                spearman = spearmanr(x_f, y_f)
                mi = mutual_info_regression(
                    x_f.reshape(-1, 1),
                    y_f,
                    n_neighbors=max(1, min(3, len(x_f) - 1)),
                    random_state=0,
                )[0]
                row.update(
                    {
                        "pearson_r": float(pearson.statistic),
                        "pearson_abs": float(abs(pearson.statistic)),
                        "pearson_p_value": float(pearson.pvalue),
                        "spearman_r": float(spearman.statistic),
                        "spearman_abs": float(abs(spearman.statistic)),
                        "spearman_p_value": float(spearman.pvalue),
                        "mutual_information": float(mi),
                    }
                )
                rows.append(row)
        return pd.DataFrame(rows)

    def _segment_level_summary(self, feature_level: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
        rows = []
        for keys, group in feature_level.groupby(["case_id", "pair_id", "sensor_name", "segment_setting"], dropna=False):
            case_id, pair_id, sensor, segment = keys
            pearson = group["pearson_abs"].dropna()
            spearman = group["spearman_abs"].dropna()
            mi = group["mutual_information"].dropna()
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "case_id": case_id,
                    "pair_id": pair_id,
                    "sensor_name": sensor,
                    "segment_setting": segment,
                    "mean_abs_pearson": self._mean(pearson),
                    "median_abs_pearson": self._median(pearson),
                    "max_abs_pearson": self._max(pearson),
                    "top3_mean_abs_pearson": self._topk_mean(pearson, 3),
                    "mean_abs_spearman": self._mean(spearman),
                    "median_abs_spearman": self._median(spearman),
                    "max_abs_spearman": self._max(spearman),
                    "top3_mean_abs_spearman": self._topk_mean(spearman, 3),
                    "mean_mutual_information": self._mean(mi),
                    "median_mutual_information": self._median(mi),
                    "max_mutual_information": self._max(mi),
                    "top3_mean_mutual_information": self._topk_mean(mi, 3),
                }
            )
        frame = pd.DataFrame(rows)
        frame["rank_by_pearson"] = frame.groupby(["case_id", "sensor_name"])["top3_mean_abs_pearson"].rank(method="min", ascending=False)
        frame["rank_by_spearman"] = frame.groupby(["case_id", "sensor_name"])["top3_mean_abs_spearman"].rank(method="min", ascending=False)
        frame["rank_by_mi"] = frame.groupby(["case_id", "sensor_name"])["top3_mean_mutual_information"].rank(method="min", ascending=False)
        return frame

    def _baseline_comparison(self, segment_level: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
        rows = []
        for keys, group in segment_level.groupby(["case_id", "pair_id", "sensor_name"], dropna=False):
            case_id, pair_id, sensor = keys
            for baseline in ["full_length", "steady"]:
                base = group.loc[group["segment_setting"] == baseline]
                if base.empty:
                    continue
                b = base.iloc[0]
                for _, row in group.iterrows():
                    d_pearson = row["top3_mean_abs_pearson"] - b["top3_mean_abs_pearson"]
                    d_spearman = row["top3_mean_abs_spearman"] - b["top3_mean_abs_spearman"]
                    d_mi = row["top3_mean_mutual_information"] - b["top3_mean_mutual_information"]
                    rows.append(
                        {
                            "experiment_id": experiment_id,
                            "case_id": case_id,
                            "pair_id": pair_id,
                            "sensor_name": sensor,
                            "segment_setting": row["segment_setting"],
                            "baseline": baseline,
                            "delta_top3_mean_abs_pearson": d_pearson,
                            "delta_top3_mean_abs_spearman": d_spearman,
                            "delta_top3_mean_mutual_information": d_mi,
                            "improved_vs_baseline_pearson": bool(d_pearson > 0),
                            "improved_vs_baseline_spearman": bool(d_spearman > 0),
                            "improved_vs_baseline_mi": bool(d_mi > 0),
                        }
                    )
        return pd.DataFrame(rows)

    def _sensor_consistency(self, segment_level: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
        rows = []
        for keys, group in segment_level.groupby(["case_id", "pair_id", "sensor_name"], dropna=False):
            case_id, pair_id, sensor = keys
            best_p = group.loc[group["top3_mean_abs_pearson"].idxmax()]
            best_s = group.loc[group["top3_mean_abs_spearman"].idxmax()]
            best_m = group.loc[group["top3_mean_mutual_information"].idxmax()]
            row = {
                "experiment_id": experiment_id,
                "case_id": case_id,
                "pair_id": pair_id,
                "sensor_name": sensor,
                "best_segment_by_pearson": best_p["segment_setting"],
                "best_segment_by_spearman": best_s["segment_setting"],
                "best_segment_by_mi": best_m["segment_setting"],
            }
            rank_map = dict(zip(group["segment_setting"], group["rank_by_spearman"]))
            for segment in SEGMENT_SETTINGS:
                row[f"average_rank_{segment}"] = rank_map.get(segment, np.nan)
            rows.append(row)
        return pd.DataFrame(rows)

    def _pair_level_summary(self, segment_level: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
        rows = []
        grouped = segment_level.groupby(["pair_id", "sensor_name", "segment_setting"], dropna=False)
        for keys, group in grouped:
            pair_id, sensor, segment = keys
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "pair_id": pair_id,
                    "cases": ",".join(map(str, DOMAIN_CASES[pair_id])),
                    "sensor_name": sensor,
                    "segment_setting": segment,
                    "mean_top3_abs_pearson": group["top3_mean_abs_pearson"].mean(),
                    "mean_top3_abs_spearman": group["top3_mean_abs_spearman"].mean(),
                    "mean_top3_mutual_information": group["top3_mean_mutual_information"].mean(),
                }
            )
        frame = pd.DataFrame(rows)
        frame["segment_rank_by_spearman"] = frame.groupby(["pair_id", "sensor_name"])["mean_top3_abs_spearman"].rank(method="min", ascending=False)
        frame["segment_rank_by_pearson"] = frame.groupby(["pair_id", "sensor_name"])["mean_top3_abs_pearson"].rank(method="min", ascending=False)
        frame["segment_rank_by_mi"] = frame.groupby(["pair_id", "sensor_name"])["mean_top3_mutual_information"].rank(method="min", ascending=False)
        return frame

    def _segment_ranking(self, segment_level: pd.DataFrame, baseline: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
        better_full = baseline.loc[baseline["baseline"] == "full_length", ["case_id", "sensor_name", "segment_setting", "improved_vs_baseline_spearman"]]
        better_full = better_full.rename(columns={"improved_vs_baseline_spearman": "is_better_than_full_length"})
        better_steady = baseline.loc[baseline["baseline"] == "steady", ["case_id", "sensor_name", "segment_setting", "improved_vs_baseline_spearman"]]
        better_steady = better_steady.rename(columns={"improved_vs_baseline_spearman": "is_better_than_steady"})
        ranking = segment_level[
            ["experiment_id", "case_id", "pair_id", "sensor_name", "segment_setting", "rank_by_pearson", "rank_by_spearman", "rank_by_mi"]
        ].copy()
        ranking["aggregation_level"] = "case_sensor"
        ranking["mean_rank"] = ranking[["rank_by_pearson", "rank_by_spearman", "rank_by_mi"]].mean(axis=1)
        ranking = ranking.merge(better_full, on=["case_id", "sensor_name", "segment_setting"], how="left")
        ranking = ranking.merge(better_steady, on=["case_id", "sensor_name", "segment_setting"], how="left")
        ranking["is_better_than_full_length"] = ranking["is_better_than_full_length"].fillna(False)
        ranking["is_better_than_steady"] = ranking["is_better_than_steady"].fillna(False)
        return ranking[
            [
                "experiment_id",
                "aggregation_level",
                "case_id",
                "pair_id",
                "sensor_name",
                "segment_setting",
                "rank_by_pearson",
                "rank_by_spearman",
                "rank_by_mi",
                "mean_rank",
                "is_better_than_full_length",
                "is_better_than_steady",
            ]
        ]

    def _analysis_summary(self, feature_level: pd.DataFrame, segment_level: pd.DataFrame, baseline: pd.DataFrame, consistency: pd.DataFrame) -> dict[str, Any]:
        best = segment_level.loc[segment_level["top3_mean_abs_spearman"].idxmax()].to_dict()
        best_counts = consistency["best_segment_by_spearman"].value_counts().to_dict()
        return {
            "analysis_type": "Exploratory",
            "num_feature_level_rows": int(len(feature_level)),
            "num_segment_level_rows": int(len(segment_level)),
            "skipped_condition_count": int((feature_level["status"] == "skipped").sum()),
            "skip_reasons": feature_level.loc[feature_level["status"] == "skipped", "skip_reason"].value_counts().to_dict(),
            "best_overall_by_top3_mean_abs_spearman": best,
            "best_segment_count_by_spearman": best_counts,
            "full_length_improvement_count_spearman": int(baseline.loc[baseline["baseline"] == "full_length", "improved_vs_baseline_spearman"].sum()),
            "steady_improvement_count_spearman": int(baseline.loc[baseline["baseline"] == "steady", "improved_vs_baseline_spearman"].sum()),
            "rq4": "H1.S1 association reference generated here; no-load excluded; compare this output with S1 prediction results.",
        }

    def _write_common_outputs(
        self,
        paths: ExperimentPaths,
        dataset: pd.DataFrame,
        feature_table: pd.DataFrame,
        sensors: list[str],
        dry_summary: dict[str, Any],
        validation: dict[str, Any],
    ) -> None:
        write_json(paths.execution_dir / "data" / "dataset_summary.json", dry_summary)
        write_json(paths.execution_dir / "data" / "sensor_summary.json", {"sensors": sensors, "num_sensors": len(sensors)})
        write_json(paths.execution_dir / "data" / "case_filter_summary.json", {"used_cases": CASE_SCOPE, "rows_by_case": dry_summary["rows_by_case"]})
        write_json(paths.execution_dir / "data" / "target_summary.json", {"target_column": "VB", "count": int(dataset["VB"].notna().sum()), "min": float(dataset["VB"].min()), "max": float(dataset["VB"].max())})
        write_json(paths.execution_dir / "preprocessing" / "preprocessing_summary.json", {"output_type": "features", "no_load_excluded": True})
        write_json(paths.execution_dir / "preprocessing" / "feature_summary.json", {"feature_names": FEATURE_NAMES, "feature_naming": "{sensor_name}__{segment_setting}__{feature_name}"})
        write_json(paths.execution_dir / "preprocessing" / "segment_summary.json", {"segment_settings": SEGMENT_SETTINGS, "excluded": ["no_load", "no_load_related_combinations"]})
        write_json(paths.execution_dir / "preprocessing" / "shape_trace.json", [{"stage": "feature_table", "shape": list(feature_table.shape)}])

    def _write_results(
        self,
        paths: ExperimentPaths,
        dataset: pd.DataFrame,
        feature_table: pd.DataFrame,
        sensors: list[str],
        dry_summary: dict[str, Any],
        validation: dict[str, Any],
        results: dict[str, Any],
    ) -> None:
        write_csv(paths.execution_dir / "metrics" / "association_feature_level.csv", results["association_feature_level"])
        write_csv(paths.execution_dir / "metrics" / "association_segment_level.csv", results["association_segment_level"])
        write_csv(paths.execution_dir / "metrics" / "association_case_level.csv", results["association_case_level"])
        write_csv(paths.execution_dir / "metrics" / "association_pair_level.csv", results["association_pair_level"])
        write_json(paths.execution_dir / "metrics" / "metrics.json", results["analysis_summary"])
        write_json(paths.execution_dir / "analysis" / "analysis_summary.json", results["analysis_summary"])
        write_csv(paths.execution_dir / "analysis" / "segment_ranking.csv", results["segment_ranking"])
        write_csv(paths.execution_dir / "analysis" / "baseline_comparison.csv", results["baseline_comparison"])
        write_csv(paths.execution_dir / "analysis" / "sensor_consistency.csv", results["sensor_consistency"])
        write_csv(paths.execution_dir / "analysis" / "pair_level_association_summary.csv", results["pair_level_association_summary"])
        write_json(paths.execution_dir / "logs" / "environment.json", collect_environment(str(self.root)))
        write_json(paths.execution_dir / "logs" / "reproducibility.json", {"framework_version": __version__, "validation": validation, "case_scope": CASE_SCOPE, "segment_settings": SEGMENT_SETTINGS})
        self._write_figures(paths, results)
        self._write_report(paths, dry_summary, results, dry_run=False)
        self._write_html(paths)

    def _write_figures(self, paths: ExperimentPaths, results: dict[str, Any]) -> None:
        import matplotlib.pyplot as plt

        segment = results["association_segment_level"]
        pair = results["pair_level_association_summary"]
        figures = paths.execution_dir / "figures"
        metric_map = [
            ("top3_mean_abs_spearman", "segment_association_by_sensor_spearman.png", "Top3 mean abs Spearman"),
            ("top3_mean_abs_pearson", "segment_association_by_sensor_pearson.png", "Top3 mean abs Pearson"),
            ("top3_mean_mutual_information", "segment_association_by_sensor_mi.png", "Top3 mean mutual information"),
        ]
        overall = segment.groupby(["sensor_name", "segment_setting"], as_index=False).mean(numeric_only=True)
        for metric, filename, ylabel in metric_map:
            fig, ax = plt.subplots(figsize=(10, 5))
            for sensor, group in overall.groupby("sensor_name"):
                ordered = group.set_index("segment_setting").reindex(SEGMENT_SETTINGS)
                ax.plot(SEGMENT_SETTINGS, ordered[metric], marker="o", label=sensor)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=35)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figures / filename)
            plt.close(fig)

        baseline = results["baseline_comparison"]
        fig, ax = plt.subplots(figsize=(10, 5))
        baseline.loc[baseline["baseline"] == "full_length"].groupby("segment_setting")["delta_top3_mean_abs_spearman"].mean().reindex(SEGMENT_SETTINGS).plot(kind="bar", ax=ax)
        ax.set_ylabel("Delta top3 abs Spearman vs full_length")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(figures / "baseline_comparison_spearman.png")
        plt.close(fig)

        consistency = results["sensor_consistency"]
        fig, ax = plt.subplots(figsize=(9, 4))
        consistency["best_segment_by_spearman"].value_counts().reindex(SEGMENT_SETTINGS, fill_value=0).plot(kind="bar", ax=ax)
        ax.set_ylabel("Best count")
        ax.set_title("Sensor best segment summary")
        fig.tight_layout()
        fig.savefig(figures / "sensor_best_segment_summary.png")
        plt.close(fig)

        heatmap = pair.groupby(["pair_id", "segment_setting"])["mean_top3_abs_spearman"].mean().unstack().reindex(columns=SEGMENT_SETTINGS)
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(heatmap.to_numpy(), cmap="viridis", aspect="auto")
        ax.set_yticks(range(len(heatmap.index)), heatmap.index)
        ax.set_xticks(range(len(heatmap.columns)), heatmap.columns, rotation=35, ha="right")
        ax.set_title("Case-pair heatmap")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(figures / "case_pair_heatmap.png")
        plt.close(fig)

    def _write_report(self, paths: ExperimentPaths, dry_summary: dict[str, Any], results: dict[str, Any], dry_run: bool) -> None:
        report = paths.execution_dir / "reports" / "report.md"
        if dry_run:
            body = f"""# H1.S1 Dry-run Report

Dry-run completed.

- Data files: `datasets/processed/mill_process_info_enabled.csv`, `datasets/processed/mill_signal_data_enabled.csv`
- Cases: {CASE_SCOPE}
- No-load excluded: true
- Sensors: {dry_summary['sensors']}
- Segment settings: {SEGMENT_SETTINGS}
- Feature names: {FEATURE_NAMES}
- Association measures: Pearson, Spearman, mutual information
- Deferred: distance correlation, Kendall's Tau
- Planned feature-level conditions: {dry_summary['planned_feature_level_conditions']}
- Insufficient conditions: {len(dry_summary['insufficient_conditions'])}
"""
        else:
            summary = results["analysis_summary"]
            best = summary["best_overall_by_top3_mean_abs_spearman"]
            best_counts = pd.Series(summary["best_segment_count_by_spearman"]).to_string()
            full_better = results["baseline_comparison"].query("baseline == 'full_length' and improved_vs_baseline_spearman").head(15)
            steady_better = results["baseline_comparison"].query("baseline == 'steady' and improved_vs_baseline_spearman").head(15)
            body = f"""# H1_S1 feture_VB_segment_association_eda

## Objective

Explore whether sensor feature and VB association changes by cutting segment while excluding no-load settings.

## Data

- Process info: `datasets/processed/mill_process_info_enabled.csv`
- Signal data: `datasets/processed/mill_signal_data_enabled.csv`
- Cases: {CASE_SCOPE}
- Domain pairs: {DOMAIN_CASES}
- No-load excluded: true

## Settings

- Sensors: {dry_summary['sensors']}
- Segment settings: {SEGMENT_SETTINGS}
- Features: {FEATURE_NAMES}
- Association measures: Pearson correlation, Spearman correlation, mutual information
- Deferred measures: distance correlation, Kendall's Tau

## Best Overall By Spearman

```json
{json.dumps(best, indent=2, default=str)}
```

## Sensor Best Segment Counts

```text
{best_counts}
```

## Full-length Improvements Preview

```text
{full_better[['case_id','sensor_name','segment_setting','delta_top3_mean_abs_spearman']].to_string(index=False)}
```

## Steady Improvements Preview

```text
{steady_better[['case_id','sensor_name','segment_setting','delta_top3_mean_abs_spearman']].to_string(index=False)}
```

## Research Questions

- RQ1: Segment settings with stronger association than full-length or steady exist when positive baseline deltas are present in `analysis/baseline_comparison.csv`.
- RQ2: Best segment consistency can be inspected in `analysis/sensor_consistency.csv`; best-count summary is shown above.
- RQ3: Sensor-specific best segment settings are stored in `analysis/sensor_consistency.csv`.
- RQ4: Entry/exit combinations should be considered when they repeatedly improve over full-length or steady.

## Limitations

- Association is not causal evidence.
- Cases with few labeled dataset runs can produce unstable correlations.
- Constant features are recorded as skipped for correlation.
"""
        report.write_text(body + paths.report_metadata_markdown(), encoding="utf-8")

    def _write_html(self, paths: ExperimentPaths) -> None:
        md = paths.execution_dir / "reports" / "report.md"
        html = paths.execution_dir / "reports" / "report.html"
        body = md.read_text(encoding="utf-8").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>\n")
        html.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")

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
            "experiment_name": "H1_S1_feture_VB_segment_association_eda",
            "dataset": "mill_processed_enabled",
            "model": "none",
            "input_type": "feature_table",
            "split_strategy": "case_pair_summary_no_model",
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

    def _best_metric(self, segment_level: pd.DataFrame) -> float | None:
        if segment_level.empty:
            return None
        return float(segment_level["top3_mean_abs_spearman"].max())

    @staticmethod
    def _mean(series: pd.Series) -> float:
        return float(series.mean()) if len(series) else float("nan")

    @staticmethod
    def _median(series: pd.Series) -> float:
        return float(series.median()) if len(series) else float("nan")

    @staticmethod
    def _max(series: pd.Series) -> float:
        return float(series.max()) if len(series) else float("nan")

    @staticmethod
    def _topk_mean(series: pd.Series, k: int) -> float:
        return float(series.sort_values(ascending=False).head(k).mean()) if len(series) else float("nan")

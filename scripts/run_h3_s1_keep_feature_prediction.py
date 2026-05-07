#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.experiments.s1_segment_execution import (
    DOMAIN_CASES,
    FEATURE_NAMES,
    SHIFT_SCENARIOS,
    S1RunConfig,
    S1SegmentExecution,
)


PREFIX = "H3_S1"
PAIR_BY_CASE = {case: pair for pair, cases in DOMAIN_CASES.items() for case in cases}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H3.S1 Keep feature subset VB prediction experiment.")
    parser.add_argument("--config", default="configs/experiments/H2_S4_process_information_combination_effect_on_segment_aware_VB_prediction.yaml")
    parser.add_argument("--h3-s0-dir", default=None)
    parser.add_argument("--seed-mode", default="all", help="initial, extra, all, or comma-separated seeds")
    parser.add_argument("--feature-subsets", default="keep_only,keep_plus_transform,keep_plus_domain_validation,all_features_reference")
    args = parser.parse_args()

    root = Path.cwd()
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_H3_S1_keep_feature_subset_VB_prediction")
    output = root / "experiments" / "executions" / experiment_id
    for dirname in ["configs", "data", "splits", "preprocessing", "metrics", "predictions", "analysis", "figures", "reports", "logs"]:
        (output / dirname).mkdir(parents=True, exist_ok=True)
    run_log = output / "logs" / f"{PREFIX}_run.log"
    error_log = output / "logs" / f"{PREFIX}_error.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line)
        with run_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    try:
        log(f"H3.S1 execution started: {experiment_id}")
        h3s0 = Path(args.h3_s0_dir) if args.h3_s0_dir else find_latest_h3_s0(root)
        rec_path = h3s0 / "analysis" / "H3_S0_feature_recommendation.csv"
        if not rec_path.exists():
            raise FileNotFoundError(f"H3_S0 recommendation file not found: {rec_path}")
        recommendation = read_recommendation(rec_path)
        run_config = resolve_run_config(Path(args.config), args.seed_mode)
        helper = S1SegmentExecution(args.config, root=root, dry_run=True, seed_mode="initial")
        dataset = helper._load_dataset(run_config)
        sensors = helper._sensor_columns(dataset)
        feature_table = helper._build_feature_table(dataset, sensors)
        wide = build_wide_feature_table(dataset, feature_table, sensors)
        feature_subsets = build_feature_subsets(recommendation, wide.columns.tolist(), args.feature_subsets.split(","))
        validation = validate_inputs(wide, feature_subsets)
        write_json(output / "splits" / f"{PREFIX}_leakage_check.json", validation)
        if not validation["ok"]:
            raise ValueError(f"validation failed: {validation['errors']}")
        log(
            "Dry-run passed: "
            f"h3_s0={h3s0.name}, samples={len(wide)}, subsets="
            + ", ".join(f"{k}:{len(v)}" for k, v in feature_subsets.items())
        )

        write_json(output / "data" / f"{PREFIX}_source_h3_s0_summary.json", {
            "source_h3_s0_dir": str(h3s0),
            "recommendation_file": str(rec_path),
            "source_report": str(h3s0 / "reports" / "H3_S0_feature_quality_report.md"),
            "recommendation_counts": recommendation["final_recommendation"].value_counts().to_dict(),
        })
        write_json(output / "data" / f"{PREFIX}_target_summary.json", {
            "target_column": "VB",
            "sample_count": int(len(wide)),
            "min": float(wide["VB"].min()),
            "mean": float(wide["VB"].mean()),
            "median": float(wide["VB"].median()),
            "max": float(wide["VB"].max()),
        })
        write_feature_lists(output, recommendation, feature_subsets)
        composition = feature_subset_composition(recommendation, feature_subsets["keep_only"])
        composition.to_csv(output / "data" / f"{PREFIX}_feature_subset_composition.csv", index=False)
        split_artifacts(wide).to_csv(output / "splits" / f"{PREFIX}_split.csv", index=False)
        write_json(output / "splits" / f"{PREFIX}_split_summary.json", split_summary(wide))
        write_json(output / "preprocessing" / f"{PREFIX}_preprocessing_summary.json", {
            "numeric": "SimpleImputer(median) + StandardScaler fit on source train split only",
            "categorical": "SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown=ignore) fit on source train split only",
            "models": ["random_forest", "mlp"],
        })
        write_json(output / "preprocessing" / f"{PREFIX}_shape_trace.json", {
            "wide_feature_table_shape": list(wide.shape),
            "subset_shapes": {k: [int(len(wide)), int(len(v))] for k, v in feature_subsets.items()},
        })
        config = {
            "experiment": {
                "experiment_id": experiment_id,
                "experiment_name": "H3_S1_keep_feature_subset_VB_prediction",
                "analysis_type": "Exploratory / Predictive validation",
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            "source_h3_s0_dir": str(h3s0),
            "case_scope": sorted(wide["case_id"].unique().tolist()),
            "domain_cases": DOMAIN_CASES,
            "shift_scenarios": [f"{s}_to_{t}" for s, t in SHIFT_SCENARIOS],
            "feature_subsets": {k: len(v) for k, v in feature_subsets.items()},
            "seeds": run_config.seeds,
            "models": run_config.models,
        }
        write_yaml_like(output / "configs" / f"{PREFIX}_input_config.yaml", config)
        write_yaml_like(output / "configs" / f"{PREFIX}_resolved_config.yaml", config)

        # Required staged execution: initial seeds first, then extras when all selected.
        seeds = run_config.seeds
        initial = [s for s in seeds if s in {0, 1, 2}]
        extra = [s for s in seeds if s not in {0, 1, 2}]
        results = []
        predictions = []
        for stage_name, stage_seeds in [("initial", initial), ("extra", extra)]:
            if not stage_seeds:
                continue
            log(f"Running {stage_name} seeds: {stage_seeds}")
            stage_results, stage_predictions = run_grid(wide, feature_subsets, stage_seeds, run_config.models)
            results.extend(stage_results)
            predictions.extend(stage_predictions)
            log(f"Finished {stage_name} seeds: atomic_evaluations={len(stage_results)}")

        shift_metrics = pd.DataFrame(results)
        pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
        if not shift_metrics.empty:
            shift_metrics["experiment_id"] = experiment_id
        seed_metrics = aggregate_seed_metrics(shift_metrics)
        seed_metrics.insert(0, "experiment_id", experiment_id)
        subset_metrics = aggregate_subset_metrics(seed_metrics)
        subset_metrics.insert(0, "experiment_id", experiment_id)
        comparison = baseline_comparison(seed_metrics)
        if not comparison.empty:
            comparison["experiment_id"] = experiment_id
        stability = model_stability(seed_metrics)
        if not stability.empty:
            stability["experiment_id"] = experiment_id
        robustness = shift_robustness(shift_metrics)
        if not robustness.empty:
            robustness["experiment_id"] = experiment_id
        r2pos = r2_positive_count(subset_metrics, seed_metrics)
        if not r2pos.empty:
            r2pos["experiment_id"] = experiment_id
        analysis_summary = {
            "experiment_id": experiment_id,
            "source_h3_s0": str(h3s0),
            "atomic_evaluations": int(len(shift_metrics)),
            "feature_subsets": {k: len(v) for k, v in feature_subsets.items()},
            "best_by_r2": subset_metrics.sort_values("mean_r2", ascending=False).head(10).to_dict(orient="records"),
            "validation": validation,
        }

        shift_metrics.to_csv(output / "metrics" / f"{PREFIX}_shift_metrics.csv", index=False)
        seed_metrics.to_csv(output / "metrics" / f"{PREFIX}_seed_metrics.csv", index=False)
        subset_metrics.to_csv(output / "metrics" / f"{PREFIX}_feature_subset_metrics.csv", index=False)
        subset_metrics.loc[subset_metrics["feature_subset"] == "keep_only"].to_csv(output / "metrics" / f"{PREFIX}_keep_feature_metrics.csv", index=False)
        comparison.to_csv(output / "metrics" / f"{PREFIX}_comparison_metrics.csv", index=False)
        pred_df.to_csv(output / "predictions" / f"{PREFIX}_predictions.csv", index=False)
        prediction_errors = pred_df.sort_values("absolute_error", ascending=False).head(200) if not pred_df.empty else pred_df
        prediction_errors.to_csv(output / "predictions" / f"{PREFIX}_prediction_errors.csv", index=False)
        comparison.to_csv(output / "analysis" / f"{PREFIX}_baseline_comparison.csv", index=False)
        stability.to_csv(output / "analysis" / f"{PREFIX}_model_stability_summary.csv", index=False)
        robustness.to_csv(output / "analysis" / f"{PREFIX}_shift_robustness_summary.csv", index=False)
        r2pos.to_csv(output / "analysis" / f"{PREFIX}_r2_positive_count_summary.csv", index=False)
        write_json(output / "analysis" / f"{PREFIX}_analysis_summary.json", analysis_summary)
        make_figures(output, subset_metrics, seed_metrics, shift_metrics, composition)
        write_report(output, experiment_id, h3s0, recommendation, feature_subsets, subset_metrics, comparison, stability, robustness, r2pos, composition, validation)
        log(f"H3.S1 execution finished: {experiment_id}")
        print(json.dumps({"experiment_id": experiment_id, "execution_dir": str(output), "atomic_evaluations": len(shift_metrics)}, indent=2, ensure_ascii=False))
    except Exception:
        error_log.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def find_latest_h3_s0(root: Path) -> Path:
    candidates = sorted((root / "experiments" / "executions").glob("*H3_S0_feature_quality_analysis_for_VB_prediction*"))
    if not candidates:
        raise FileNotFoundError("No H3_S0 feature quality execution directory found.")
    return candidates[-1]


def read_recommendation(path: Path) -> pd.DataFrame:
    recommendation = pd.read_csv(path)
    if "final_recommendation" not in recommendation.columns:
        recommendation = pd.read_csv(path, skiprows=1)
    if "final_recommendation" not in recommendation.columns:
        raise ValueError(f"final_recommendation column not found in {path}")
    return recommendation


def resolve_run_config(config_path: Path, seed_mode: str) -> S1RunConfig:
    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    seed_list = config.get("seeds", {}).get("seed_list", list(range(20)))
    if seed_mode == "initial":
        seeds = config.get("seeds", {}).get("initial_seeds", [0, 1, 2])
    elif seed_mode == "extra":
        seeds = config.get("seeds", {}).get("extra_seeds", [s for s in seed_list if s not in {0, 1, 2}])
    elif seed_mode == "all":
        seeds = seed_list
    else:
        seeds = [int(s) for s in seed_mode.split(",") if s.strip()]
    dataset_cfg = config["dataset"]
    return S1RunConfig(
        process_info_path=Path(dataset_cfg["process_info_path"]),
        signal_data_path=Path(dataset_cfg["signal_data_path"]),
        heuristic_sequence_path=Path(dataset_cfg["heuristic_sequence_path"]),
        seeds=[int(seed) for seed in seeds],
        models=config.get("models", ["random_forest", "mlp"]),
        random_forest_params=dict(config.get("model_defaults", {}).get("random_forest", {})),
        mlp_params=dict(config.get("model_defaults", {}).get("mlp", {})),
    )


def build_wide_feature_table(dataset: pd.DataFrame, feature_table: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    sample_cols = ["sample_id", "dataset_run_id", "case", "run", "domain_id", "VB"]
    process_cols = ["DOC", "feed", "material_name", "time"]
    base = dataset[sample_cols + process_cols].drop_duplicates("sample_id").copy()
    base = base.rename(columns={"case": "case_id", "DOC": "DoC", "feed": "Feed", "material_name": "Material", "time": "Time"})
    base["pair_id"] = base["case_id"].map(PAIR_BY_CASE)
    wide = base[["sample_id", "dataset_run_id", "case_id", "run", "pair_id", "domain_id", "VB", "DoC", "Feed", "Material", "Time"]].copy()
    for segment in sorted(feature_table["segment_setting"].unique()):
        segment_rows = feature_table.loc[feature_table["segment_setting"] == segment]
        for sensor in sensors:
            rows = segment_rows.loc[segment_rows["sensor"] == sensor, ["sample_id"] + FEATURE_NAMES].copy()
            rows = rows.rename(columns={feature: f"{sensor}__{segment}__{feature}" for feature in FEATURE_NAMES})
            wide = wide.merge(rows, on="sample_id", how="inner", validate="one_to_one")
    return wide


def build_feature_subsets(recommendation: pd.DataFrame, columns: list[str], requested: list[str]) -> dict[str, list[str]]:
    requested = [r.strip() for r in requested if r.strip()]
    available = set(columns)
    keep = recommendation.loc[recommendation["final_recommendation"] == "Keep", "feature_name"].tolist()
    transform = recommendation.loc[recommendation["final_recommendation"] == "Transform", "feature_name"].tolist()
    validation = recommendation.loc[recommendation["final_recommendation"] == "Need domain validation", "feature_name"].tolist()
    all_features = recommendation["feature_name"].tolist()
    subsets = {
        "keep_only": keep,
        "keep_plus_transform": keep + transform,
        "keep_plus_domain_validation": keep + validation,
        "all_features_reference": all_features,
    }
    filtered = {}
    for name in requested:
        values = [f for f in subsets[name] if f in available]
        filtered[name] = list(dict.fromkeys(values))
    return filtered


def validate_inputs(wide: pd.DataFrame, feature_subsets: dict[str, list[str]]) -> dict[str, Any]:
    errors = []
    warnings = []
    leakage_cols = {"VB", "case_id", "domain_id", "pair_id", "source_domain", "target_domain", "split", "dataset_run_id", "sample_id", "run"}
    if sorted(wide["case_id"].unique().tolist()) != [1, 2, 8, 9, 12, 14]:
        errors.append(f"case filtering failed: {sorted(wide['case_id'].unique().tolist())}")
    for name, features in feature_subsets.items():
        if not features:
            errors.append(f"empty feature subset: {name}")
        leak = sorted(set(features) & leakage_cols)
        if leak:
            errors.append(f"leakage columns in {name}: {leak}")
    if "keep_only" in feature_subsets:
        forbidden = sorted(set(feature_subsets["keep_only"]) & {"Material", "Time"})
        if forbidden:
            errors.append(f"Material/Time unexpectedly included in keep_only: {forbidden}")
    split_checks = []
    for source, target in SHIFT_SCENARIOS:
        split = split_frame(wide, source, target)
        train_cases = set(split.loc[split["split"].isin(["train", "validation"]), "case_id"])
        test_cases = set(split.loc[split["split"] == "test", "case_id"])
        overlap = sorted(train_cases & test_cases)
        duplicate_samples = int(split.groupby("sample_id")["split"].nunique().gt(1).sum())
        if overlap:
            errors.append(f"source/target overlap for {source}_to_{target}: {overlap}")
        if duplicate_samples:
            errors.append(f"duplicate sample split assignment for {source}_to_{target}: {duplicate_samples}")
        split_checks.append({"scenario": f"{source}_to_{target}", "source_cases": DOMAIN_CASES[source], "target_cases": DOMAIN_CASES[target], "overlap": overlap, "duplicate_samples": duplicate_samples})
    return {"ok": not errors, "errors": errors, "warnings": warnings, "split_checks": split_checks}


def split_frame(data: pd.DataFrame, source_domain: str, target_domain: str) -> pd.DataFrame:
    source_cases = DOMAIN_CASES[source_domain]
    target_cases = DOMAIN_CASES[target_domain]
    validation_case = min(source_cases)
    split = data.loc[data["case_id"].isin(source_cases + target_cases)].copy()
    split["split"] = "excluded"
    split.loc[split["case_id"].isin(target_cases), "split"] = "test"
    split.loc[split["case_id"].isin(source_cases) & (split["case_id"] != validation_case), "split"] = "train"
    split.loc[split["case_id"] == validation_case, "split"] = "validation"
    return split


def run_grid(data: pd.DataFrame, feature_subsets: dict[str, list[str]], seeds: list[int], models: list[str]) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    results = []
    predictions = []
    for subset_name, features in feature_subsets.items():
        for model_name in models:
            for seed in seeds:
                for source, target in SHIFT_SCENARIOS:
                    result, preds = run_one(data, features, subset_name, model_name, seed, source, target)
                    results.append(result)
                    predictions.append(preds)
    return results, predictions


def run_one(data: pd.DataFrame, features: list[str], subset_name: str, model_name: str, seed: int, source: str, target: str) -> tuple[dict[str, Any], pd.DataFrame]:
    split = split_frame(data, source, target)
    train = split.loc[split["split"] == "train"].copy()
    test = split.loc[split["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError(f"empty train/test for {source}_to_{target}")
    model = make_model(model_name, seed, train[features])
    model.fit(train[features], train["VB"].astype(float))
    y_true = test["VB"].astype(float).to_numpy()
    y_pred = model.predict(test[features])
    metrics = compute_metrics(y_true, y_pred)
    child_key = f"{model_name}_{subset_name}_seed{seed}_{source}_to_{target}"
    result = {
        "experiment_id": "",
        "child_execution_key": child_key,
        "model": model_name,
        "feature_subset": subset_name,
        "seed": seed,
        "source_domain": source,
        "target_domain": target,
        "source_cases": ",".join(map(str, DOMAIN_CASES[source])),
        "target_cases": ",".join(map(str, DOMAIN_CASES[target])),
        "num_features": len(features),
        **metrics,
    }
    preds = test[["sample_id", "dataset_run_id", "case_id", "run", "pair_id", "domain_id", "VB"]].copy()
    preds["child_execution_key"] = child_key
    preds["model"] = model_name
    preds["feature_subset"] = subset_name
    preds["seed"] = seed
    preds["source_domain"] = source
    preds["target_domain"] = target
    preds["split"] = "test"
    preds["y_true"] = y_true
    preds["y_pred"] = y_pred
    preds["residual"] = preds["y_true"] - preds["y_pred"]
    preds["absolute_error"] = preds["residual"].abs()
    return result, preds


def make_model(model_name: str, seed: int, frame: pd.DataFrame) -> Pipeline:
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in frame.columns if c not in numeric]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )
    if model_name == "random_forest":
        model = RandomForestRegressor(random_state=seed, n_jobs=-1)
    elif model_name == "mlp":
        model = MLPRegressor(hidden_layer_sizes=(64, 32, 16), random_state=seed, max_iter=300, learning_rate_init=0.001)
    else:
        raise ValueError(f"unsupported model: {model_name}")
    return Pipeline([("preprocess", pre), ("model", model)])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        pearson = float(pearsonr(y_true, y_pred).statistic)
        spearman = float(spearmanr(y_true, y_pred).statistic)
    else:
        pearson = math.nan
        spearman = math.nan
    return {
        "metric_mae": float(mean_absolute_error(y_true, y_pred)),
        "metric_rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "metric_r2": float(r2_score(y_true, y_pred)),
        "metric_pearson": pearson,
        "metric_spearman": spearman,
    }


def aggregate_seed_metrics(shift_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = shift_metrics.groupby(["model", "feature_subset", "seed", "num_features"], dropna=False).agg(
        mean_mae_over_6_shifts=("metric_mae", "mean"),
        mean_rmse_over_6_shifts=("metric_rmse", "mean"),
        mean_r2_over_6_shifts=("metric_r2", "mean"),
        mean_pearson_over_6_shifts=("metric_pearson", "mean"),
        mean_spearman_over_6_shifts=("metric_spearman", "mean"),
    )
    return grouped.reset_index()


def aggregate_subset_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = seed_metrics.groupby(["model", "feature_subset", "num_features"], dropna=False).agg(
        mean_mae=("mean_mae_over_6_shifts", "mean"),
        std_mae=("mean_mae_over_6_shifts", "std"),
        mean_rmse=("mean_rmse_over_6_shifts", "mean"),
        std_rmse=("mean_rmse_over_6_shifts", "std"),
        mean_r2=("mean_r2_over_6_shifts", "mean"),
        std_r2=("mean_r2_over_6_shifts", "std"),
        mean_pearson=("mean_pearson_over_6_shifts", "mean"),
        std_pearson=("mean_pearson_over_6_shifts", "std"),
        mean_spearman=("mean_spearman_over_6_shifts", "mean"),
        std_spearman=("mean_spearman_over_6_shifts", "std"),
    )
    return grouped.reset_index()


def baseline_comparison(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = seed_metrics.loc[seed_metrics["feature_subset"] == "all_features_reference"].copy()
    rows = []
    for _, row in seed_metrics.loc[seed_metrics["feature_subset"] != "all_features_reference"].iterrows():
        base = baseline.loc[(baseline["model"] == row["model"]) & (baseline["seed"] == row["seed"])]
        if base.empty:
            continue
        base = base.iloc[0]
        rows.append({
            "experiment_id": "",
            "model": row["model"],
            "feature_subset": row["feature_subset"],
            "baseline_feature_subset": "all_features_reference",
            "seed": row["seed"],
            "delta_mae": row["mean_mae_over_6_shifts"] - base["mean_mae_over_6_shifts"],
            "delta_rmse": row["mean_rmse_over_6_shifts"] - base["mean_rmse_over_6_shifts"],
            "delta_r2": row["mean_r2_over_6_shifts"] - base["mean_r2_over_6_shifts"],
            "delta_pearson": row["mean_pearson_over_6_shifts"] - base["mean_pearson_over_6_shifts"],
            "delta_spearman": row["mean_spearman_over_6_shifts"] - base["mean_spearman_over_6_shifts"],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["improved_mae"] = out["delta_mae"] < 0
    out["improved_rmse"] = out["delta_rmse"] < 0
    out["improved_r2"] = out["delta_r2"] > 0
    out["improved_pearson"] = out["delta_pearson"] > 0
    out["improved_spearman"] = out["delta_spearman"] > 0
    return out


def model_stability(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in seed_metrics.groupby(["model", "feature_subset"], dropna=False):
        model, subset = keys
        r2 = group["mean_r2_over_6_shifts"]
        rows.append({
            "experiment_id": "",
            "model": model,
            "feature_subset": subset,
            "mean_r2": r2.mean(),
            "std_r2": r2.std(),
            "min_r2": r2.min(),
            "max_r2": r2.max(),
            "num_r2_positive": int((r2 > 0).sum()),
            "ratio_r2_positive": float((r2 > 0).mean()),
            "num_r2_negative": int((r2 < 0).sum()),
            "ratio_r2_negative": float((r2 < 0).mean()),
            "mean_rmse": group["mean_rmse_over_6_shifts"].mean(),
            "std_rmse": group["mean_rmse_over_6_shifts"].std(),
            "stability_comment": "stable_positive" if (r2 > 0).mean() >= 0.8 else "mostly_negative_or_unstable",
        })
    return pd.DataFrame(rows)


def shift_robustness(shift_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in shift_metrics.groupby(["model", "feature_subset", "source_domain", "target_domain"], dropna=False):
        model, subset, source, target = keys
        rows.append({
            "experiment_id": "",
            "model": model,
            "feature_subset": subset,
            "source_domain": source,
            "target_domain": target,
            "mean_mae": group["metric_mae"].mean(),
            "mean_rmse": group["metric_rmse"].mean(),
            "mean_r2": group["metric_r2"].mean(),
            "std_r2": group["metric_r2"].std(),
            "num_r2_positive": int((group["metric_r2"] > 0).sum()),
            "robustness_comment": "some_positive_r2" if (group["metric_r2"] > 0).any() else "all_negative_r2",
        })
    return pd.DataFrame(rows)


def r2_positive_count(subset_metrics: pd.DataFrame, seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in seed_metrics.groupby(["model", "feature_subset"], dropna=False):
        model, subset = keys
        rows.append({
            "experiment_id": "",
            "model": model,
            "feature_subset": subset,
            "num_total_conditions": int(len(group)),
            "num_r2_positive": int((group["mean_r2_over_6_shifts"] > 0).sum()),
            "ratio_r2_positive": float((group["mean_r2_over_6_shifts"] > 0).mean()),
            "mean_r2": float(group["mean_r2_over_6_shifts"].mean()),
            "min_r2": float(group["mean_r2_over_6_shifts"].min()),
            "max_r2": float(group["mean_r2_over_6_shifts"].max()),
        })
    return pd.DataFrame(rows)


def write_feature_lists(output: Path, recommendation: pd.DataFrame, feature_subsets: dict[str, list[str]]) -> None:
    keep = recommendation.loc[recommendation["feature_name"].isin(feature_subsets["keep_only"])].copy()
    keep.to_csv(output / "data" / f"{PREFIX}_keep_feature_list.csv", index=False)
    rows = []
    for subset, features in feature_subsets.items():
        for feature in features:
            rows.append({"feature_subset": subset, "feature_name": feature})
    pd.DataFrame(rows).to_csv(output / "data" / f"{PREFIX}_feature_subset_lists.csv", index=False)


def feature_subset_composition(recommendation: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    sub = recommendation.loc[recommendation["feature_name"].isin(features)].copy()
    rows = []
    for axis in ["sensor_name", "segment_setting", "base_feature_name", "feature_group"]:
        for value, count in sub[axis].fillna("").value_counts().items():
            rows.append({"subset": "keep_only", "composition_axis": axis, "value": value if value else "process_or_unknown", "count": int(count), "ratio": float(count / len(sub))})
    acoustic_entry_exit = sub["sensor_name"].astype(str).str.contains("AE|acoustic", case=False, regex=True) & sub["segment_setting"].astype(str).str.contains("entry|exit", case=False, regex=True)
    rows.append({"subset": "keep_only", "composition_axis": "special", "value": "acoustic_entry_exit_family", "count": int(acoustic_entry_exit.sum()), "ratio": float(acoustic_entry_exit.mean())})
    return pd.DataFrame(rows)


def split_artifacts(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, target in SHIFT_SCENARIOS:
        split = split_frame(wide, source, target)
        for _, row in split.iterrows():
            rows.append({
                "sample_id": row["sample_id"],
                "dataset_run_id": row["dataset_run_id"],
                "case_id": row["case_id"],
                "pair_id": row["pair_id"],
                "domain_id": row["domain_id"],
                "label": row["VB"],
                "source_domain": source,
                "target_domain": target,
                "split": row["split"],
            })
    return pd.DataFrame(rows)


def split_summary(wide: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for source, target in SHIFT_SCENARIOS:
        split = split_frame(wide, source, target)
        rows.append({"scenario": f"{source}_to_{target}", "source_cases": DOMAIN_CASES[source], "target_cases": DOMAIN_CASES[target], "split_counts": split["split"].value_counts().to_dict()})
    return {"shift_scenarios": rows}


def make_figures(output: Path, subset_metrics: pd.DataFrame, seed_metrics: pd.DataFrame, shift_metrics: pd.DataFrame, composition: pd.DataFrame) -> None:
    figdir = output / "figures"
    pivot = subset_metrics.pivot(index="feature_subset", columns="model", values="mean_r2")
    pivot.plot(kind="bar", figsize=(8, 4.8))
    plt.title("H3_S1 Keep vs All Feature Performance (mean R2)")
    plt.ylabel("mean R2 over seeds")
    plt.tight_layout()
    plt.savefig(figdir / f"{PREFIX}_keep_vs_all_feature_performance.png", dpi=180)
    plt.close()

    keep = subset_metrics.loc[subset_metrics["feature_subset"] == "keep_only"]
    x = np.arange(len(keep["model"]))
    fig, ax1 = plt.subplots(figsize=(7, 4.8))
    ax1.bar(x - 0.2, keep["mean_rmse"], 0.4, yerr=keep["std_rmse"], label="RMSE", color="#4E79A7")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, keep["mean_r2"], 0.4, yerr=keep["std_r2"], label="R2", color="#F28E2B")
    ax1.set_xticks(x)
    ax1.set_xticklabels(keep["model"])
    ax1.set_ylabel("RMSE")
    ax2.set_ylabel("R2")
    plt.title("H3_S1 Model Comparison on Keep Features")
    fig.tight_layout()
    plt.savefig(figdir / f"{PREFIX}_model_comparison_keep_features.png", dpi=180)
    plt.close()

    shift_plot = shift_metrics.groupby(["model", "feature_subset", "source_domain", "target_domain"])["metric_r2"].mean().reset_index()
    shift_plot["scenario"] = shift_plot["source_domain"] + "_to_" + shift_plot["target_domain"]
    for subset in shift_plot["feature_subset"].unique():
        sub = shift_plot.loc[shift_plot["feature_subset"] == subset]
        for model in sub["model"].unique():
            vals = sub.loc[sub["model"] == model].set_index("scenario")["metric_r2"]
            plt.plot(vals.index, vals.values, marker="o", label=f"{model}/{subset}")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("mean R2")
    plt.title("H3_S1 Shift-wise Performance")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figdir / f"{PREFIX}_shift_wise_performance.png", dpi=180)
    plt.close()

    r2_counts = seed_metrics.groupby(["model", "feature_subset"])["mean_r2_over_6_shifts"].apply(lambda s: (s > 0).sum()).reset_index(name="num_r2_positive")
    labels = r2_counts["model"] + "\n" + r2_counts["feature_subset"]
    plt.figure(figsize=(10, 4.8))
    plt.bar(labels, r2_counts["num_r2_positive"], color="#59A14F")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("R2 > 0 seed count")
    plt.title("H3_S1 R2 Positive Count Comparison")
    plt.tight_layout()
    plt.savefig(figdir / f"{PREFIX}_r2_positive_count_comparison.png", dpi=180)
    plt.close()

    comp = composition.loc[composition["composition_axis"].isin(["sensor_name", "segment_setting", "feature_group"])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, axis in zip(axes, ["sensor_name", "segment_setting", "feature_group"]):
        sub = comp.loc[comp["composition_axis"] == axis].sort_values("count", ascending=False)
        ax.bar(sub["value"], sub["count"])
        ax.set_title(axis)
        ax.tick_params(axis="x", rotation=60)
    fig.suptitle("H3_S1 Keep Feature Subset Composition")
    fig.tight_layout()
    plt.savefig(figdir / f"{PREFIX}_feature_subset_composition.png", dpi=180)
    plt.close()


def write_report(output: Path, experiment_id: str, h3s0: Path, recommendation: pd.DataFrame, feature_subsets: dict[str, list[str]], subset_metrics: pd.DataFrame, comparison: pd.DataFrame, stability: pd.DataFrame, robustness: pd.DataFrame, r2pos: pd.DataFrame, composition: pd.DataFrame, validation: dict[str, Any]) -> None:
    def table(df: pd.DataFrame, n: int = 20) -> str:
        return df.head(n).to_string(index=False) if not df.empty else "none"

    keep = subset_metrics.loc[subset_metrics["feature_subset"] == "keep_only"].sort_values("mean_r2", ascending=False)
    comp_keep = comparison.loc[comparison["feature_subset"] == "keep_only"]
    improvement_summary = comp_keep.groupby("model").agg(
        improved_mae=("improved_mae", "sum"),
        improved_rmse=("improved_rmse", "sum"),
        improved_r2=("improved_r2", "sum"),
        total=("seed", "count"),
        mean_delta_r2=("delta_r2", "mean"),
    ).reset_index() if not comp_keep.empty else pd.DataFrame()
    special = composition.loc[(composition["composition_axis"] == "special") & (composition["value"] == "acoustic_entry_exit_family")]
    acoustic_count = int(special["count"].iloc[0]) if not special.empty else 0
    acoustic_ratio = float(special["ratio"].iloc[0]) if not special.empty else 0.0
    report = f"""# H3_S1 Keep Feature Subset VB Prediction Report

## 1. Objective

This experiment validates whether the H3_S0 `Keep` feature subset improves feature-based VB prediction under the fixed A/B/C domain-shift protocol.

## 2. Source H3_S0

- Source directory: `{h3s0}`
- Recommendation file: `{h3s0 / 'analysis' / 'H3_S0_feature_recommendation.csv'}`
- Keep features: {len(feature_subsets.get('keep_only', []))}
- Feature subsets: {json.dumps({k: len(v) for k, v in feature_subsets.items()}, ensure_ascii=False)}

Feature selection was based on prior EDA over the available data. This is useful for validation, but it is not a train-only feature-selection protocol.

## 3. Protocol

- Cases: 1, 2, 8, 9, 12, 14
- Domain pairs: A={{1,9}}, B={{2,12}}, C={{8,14}}
- Shifts: A→B, A→C, B→A, B→C, C→A, C→B
- Models: Random Forest, MLP
- Seeds: 0..19
- Scaling/imputation/encoding: fit on source train split only.

## 4. Feature Subset Performance

```text
{table(subset_metrics.sort_values('mean_r2', ascending=False), 30)}
```

## 5. Keep Only vs All Features Reference

```text
{table(improvement_summary)}
```

Negative delta MAE/RMSE and positive delta R2 mean the feature subset improved over `all_features_reference`.

## 6. RF vs MLP Stability

```text
{table(stability.sort_values(['feature_subset','model']))}
```

## 7. Shift Robustness

```text
{table(robustness.sort_values(['feature_subset','model','mean_r2'], ascending=[True, True, False]), 40)}
```

## 8. R2 Positive Count

```text
{table(r2pos.sort_values(['feature_subset','model']))}
```

## 9. Keep Feature Composition

The keep_only subset contains {acoustic_count} acoustic entry/exit-family features ({acoustic_ratio:.1%} of Keep features). This is consistent with the earlier H2 acoustic entry/exit signal, but prediction robustness remains model- and shift-dependent.

```text
{table(composition, 80)}
```

## 10. Leakage Checks

```json
{json.dumps(validation, indent=2, ensure_ascii=False)}
```

## 11. Interpretation

If `keep_only` improves over `all_features_reference`, H3_S0 filtering is useful as a practical dimensionality reduction step. If improvement is model-specific, the feature quality signal should be treated as a candidate-prior rather than a guaranteed prediction improvement. MLP stability should be interpreted cautiously because the dataset has only 58 samples and the domain-shift splits are small.

## 12. Limitations and Next Steps

- H3_S0 feature selection used all available data, so future confirmatory runs should perform feature selection inside source-train folds only.
- Material and Time remain excluded from `keep_only`; `keep_plus_domain_validation` is sensitivity analysis only.
- Follow-up: RF-centered reduced candidate search, train-only feature selection, and acoustic entry/exit confirmatory validation.
"""
    (output / "reports" / f"{PREFIX}_report.md").write_text(report, encoding="utf-8")
    html_report = "<!doctype html><html><head><meta charset='utf-8'><title>H3_S1 Report</title></head><body><pre>" + html.escape(report) + "</pre></body></html>"
    (output / "reports" / f"{PREFIX}_report.html").write_text(html_report, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_yaml_like(path: Path, payload: Any) -> None:
    write_json(path, payload)


if __name__ == "__main__":
    main()

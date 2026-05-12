#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from milling_experiment_framework.experiments.execution_path import create_execution_dir, find_experiment_dirs

PREFIX = "H4_S1"
TOPIC = "cnn_gru_run_sequence_domain_generalization_NASA_Ames_extended_condition_pairs"
PROCESSED_EXT_DIR = ROOT / "datasets" / "processed" / "h4_s1_extended"
EXT_PROCESS_PATH = PROCESSED_EXT_DIR / "mill_process_info_extended_condition_pairs.csv"
EXT_SIGNAL_PATH = PROCESSED_EXT_DIR / "mill_signal_data_extended_condition_pairs.csv"

PAIRS: dict[str, list[int]] = {
    "A": [1, 9],
    "B": [2, 12],
    "C": [8, 14],
    "D": [3, 11],
    "E": [4, 10],
    "F": [7, 13],
}
ORIGINAL_SCENARIOS = {"A_to_B", "A_to_C", "B_to_A", "B_to_C", "C_to_A", "C_to_B"}
MODELS = ["feature_ridge", "feature_random_forest", "feature_svr", "feature_xgboost", "cnn_only", "cnn_lag_concat", "cnn_gru"]
DL_MODELS = ["cnn_only", "cnn_lag_concat", "cnn_gru"]


def load_base_module():
    script = ROOT / "scripts" / "run_H4_S1_cnn_gru_domain_generalization.py"
    spec = importlib.util.spec_from_file_location("h4_s1_base", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base H4_S1 runner: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EXPERIMENT_TOPIC = TOPIC
    module.DOMAIN_CASES = PAIRS
    module.TRANSFER_SCENARIOS = scenario_pairs()
    return module


def scenario_pairs() -> list[tuple[str, str]]:
    return [(source, target) for source in PAIRS for target in PAIRS if source != target]


def scenario_name(source: str, target: str) -> str:
    return f"{source}_to_{target}"


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, indent=2, ensure_ascii=False)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        write_json(path.with_suffix(".json"), payload)
    else:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(to_builtin(payload), f, sort_keys=False, allow_unicode=True)


def prepare_extended_processed_files() -> dict[str, Any]:
    process = pd.read_csv(ROOT / "datasets" / "processed" / "mill_process_info.csv")
    signal = pd.read_csv(ROOT / "datasets" / "processed" / "mill_signal_data.csv")
    selected_cases = sorted({case for cases in PAIRS.values() for case in cases})
    process_ext = process.loc[process["case"].isin(selected_cases)].copy()
    signal_ext = signal.loc[signal["case"].isin(selected_cases)].copy()
    if "enable" in process_ext.columns:
        process_ext["enable"] = True
    if "enable" in signal_ext.columns:
        signal_ext["enable"] = True
    PROCESSED_EXT_DIR.mkdir(parents=True, exist_ok=True)
    process_ext.to_csv(EXT_PROCESS_PATH, index=False)
    signal_ext.to_csv(EXT_SIGNAL_PATH, index=False)
    proc_cases = set(process_ext["case"].astype(int))
    sig_cases = set(signal_ext["case"].astype(int))
    label_counts = process_ext.groupby("case")["VB"].apply(lambda s: int(s.notna().sum())).to_dict()
    return {
        "source_process_path": str(ROOT / "datasets" / "processed" / "mill_process_info.csv"),
        "source_signal_path": str(ROOT / "datasets" / "processed" / "mill_signal_data.csv"),
        "extended_process_path": str(EXT_PROCESS_PATH),
        "extended_signal_path": str(EXT_SIGNAL_PATH),
        "selected_cases": selected_cases,
        "process_rows": int(len(process_ext)),
        "signal_rows": int(len(signal_ext)),
        "available_cases_in_process": sorted(proc_cases),
        "available_cases_in_signal": sorted(sig_cases),
        "missing_cases": sorted(set(selected_cases) - proc_cases.intersection(sig_cases)),
        "vb_non_null_by_case": {str(k): int(v) for k, v in label_counts.items()},
    }


def find_existing_h4_s1(root: Path) -> Path | None:
    candidates = [Path(p) for p in find_experiment_dirs(root, hypothesis_id="H4", scenario_id="S1", keyword="cnn_gru_run_sequence_domain_generalization_NASA_Ames")]
    valid: list[Path] = []
    for candidate in candidates:
        if "extended_condition_pairs" in candidate.name:
            continue
        if (candidate / "metrics" / f"{PREFIX}_target_domain_metrics.csv").exists() and (
            candidate / "predictions" / f"{PREFIX}_test_predictions.csv"
        ).exists():
            valid.append(candidate)
    if not valid:
        return None
    return sorted(valid, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def build_config(h4: Any, seeds: list[int]) -> dict[str, Any]:
    config = h4.build_default_config()
    selected_cases = sorted({case for cases in PAIRS.values() for case in cases})
    config["experiment"].update(
        {
            "hypothesis_id": "H4",
            "scenario_id": "S1",
            "experiment_topic": TOPIC,
            "timestamp": "auto",
            "experiment_id": "auto",
            "execution_dir": "auto",
            "seed_list": seeds,
            "num_seeds": len(seeds),
        }
    )
    config["data"].update(
        {
            "process_info_path": str(EXT_PROCESS_PATH.relative_to(ROOT)),
            "signal_data_path": str(EXT_SIGNAL_PATH.relative_to(ROOT)),
            "selected_cases": selected_cases,
        }
    )
    config["domain"] = {
        "condition_pairs": {k: {"cases": v, "name": f"pair_{k}_case_{v[0]}_{v[1]}"} for k, v in PAIRS.items()},
        "transfer_scenarios": [
            {
                "name": scenario_name(source, target),
                "source_domain": source,
                "target_domain": target,
                "train_cases": PAIRS[source],
                "test_cases": PAIRS[target],
            }
            for source, target in scenario_pairs()
        ],
    }
    config["training"]["max_epochs"] = 200
    config["training"]["early_stopping"]["patience"] = 100
    config.setdefault("visualization", {}).update(
        {
            "rmse_filter_enabled": True,
            "rmse_filter_threshold": 0.5,
            "rmse_filter_metric": "RMSE",
            "rmse_filter_operator": ">",
            "exclude_filtered_from_metrics": False,
            "exclude_filtered_from_ranking": False,
            "exclude_filtered_from_report_tables": False,
        }
    )
    config["smoke"] = {
        "enabled": True,
        "scenario": "D_to_E",
        "seed": 0,
        "models": ["feature_ridge", "cnn_only", "cnn_lag_concat", "cnn_gru"],
        "max_epochs": 2,
        "batch_size": 4,
        "sequence_length": 128,
        "window_length": 300,
        "num_windows": 5,
    }
    return config


def add_extended_columns(df: pd.DataFrame, status: str, source_path: str = "", experiment_id: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    if experiment_id and "experiment_id" in out.columns:
        out["experiment_id"] = experiment_id
    out["condition_pair_source"] = out.get("source_domain", "")
    out["condition_pair_target"] = out.get("target_domain", "")
    out["is_original_ABC_scope"] = out["scenario_name"].isin(ORIGINAL_SCENARIOS) if "scenario_name" in out.columns else False
    out["is_new_pair_involved"] = (
        out["source_domain"].isin(["D", "E", "F"]) | out["target_domain"].isin(["D", "E", "F"])
        if {"source_domain", "target_domain"} <= set(out.columns)
        else False
    )
    out["result_status"] = status
    out["reused_from_path"] = source_path
    return out


def load_reusable_frames(existing_dir: Path | None, experiment_id: str) -> dict[str, pd.DataFrame]:
    if existing_dir is None:
        empty = pd.DataFrame()
        return {k: empty for k in ["case", "target", "pred", "history", "best_epoch", "split"]}
    frames: dict[str, pd.DataFrame] = {}
    paths = {
        "case": existing_dir / "metrics" / f"{PREFIX}_case_metrics.csv",
        "target": existing_dir / "metrics" / f"{PREFIX}_target_domain_metrics.csv",
        "pred": existing_dir / "predictions" / f"{PREFIX}_test_predictions.csv",
        "history": existing_dir / "metrics" / f"{PREFIX}_learning_history.csv",
        "best_epoch": existing_dir / "analysis" / f"{PREFIX}_best_epoch_summary.csv",
        "split": existing_dir / "splits" / f"{PREFIX}_split.csv",
    }
    for name, path in paths.items():
        if path.exists():
            df = pd.read_csv(path)
            if "scenario_name" in df.columns:
                df = df.loc[df["scenario_name"].isin(ORIGINAL_SCENARIOS)].copy()
            if "experiment_id" in df.columns:
                df["experiment_id"] = experiment_id
            frames[name] = add_extended_columns(df, "reused", str(existing_dir), experiment_id) if name not in {"history", "best_epoch", "split"} else df
        else:
            frames[name] = pd.DataFrame()
    return frames


def existing_condition_keys(existing_target: pd.DataFrame, sequence_size: int) -> set[tuple[str, str, int, int]]:
    if existing_target.empty:
        return set()
    return set(
        zip(
            existing_target["scenario_name"].astype(str),
            existing_target["model_name"].astype(str),
            existing_target["seed"].astype(int),
            existing_target.get("sequence_size", sequence_size).astype(int),
        )
    )


def build_reuse_plan(existing_target: pd.DataFrame, existing_dir: Path | None, seeds: list[int], available_cases: set[int]) -> pd.DataFrame:
    keys = existing_condition_keys(existing_target, 3)
    rows = []
    for source, target in scenario_pairs():
        scen = scenario_name(source, target)
        source_cases = PAIRS[source]
        target_cases = PAIRS[target]
        missing = sorted((set(source_cases) | set(target_cases)) - available_cases)
        for seed in seeds:
            for model in MODELS:
                key = (scen, model, int(seed), 3)
                if key in keys and scen in ORIGINAL_SCENARIOS:
                    status = "reused"
                    reason = "existing original ABC H4_S1 result reused"
                    needs_execution = False
                    reuse_path = str(existing_dir) if existing_dir else ""
                elif missing:
                    status = "skipped"
                    reason = f"missing_cases={missing}"
                    needs_execution = False
                    reuse_path = ""
                else:
                    status = "pending"
                    reason = "new or missing scenario/model/seed condition"
                    needs_execution = True
                    reuse_path = ""
                rows.append(
                    {
                        "scenario_name": scen,
                        "source_domain": source,
                        "target_domain": target,
                        "train_cases": ",".join(map(str, source_cases)),
                        "test_cases": ",".join(map(str, target_cases)),
                        "model_name": model,
                        "seed": int(seed),
                        "sequence_size": 3,
                        "status": status,
                        "reuse_source_path": reuse_path,
                        "reason": reason,
                        "needs_execution": needs_execution,
                    }
                )
    return pd.DataFrame(rows)


def run_smoke(h4: Any, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    smoke_cfg = config["smoke"]
    smoke_config = json.loads(json.dumps(to_builtin(config)))
    smoke_config["sequence"]["sequence_length"] = int(smoke_cfg.get("sequence_length", 128))
    smoke_config.setdefault("model_design", {}).setdefault("run_encoder", {})
    smoke_config["model_design"]["run_encoder"]["window_length"] = int(smoke_cfg.get("window_length", 300))
    smoke_config["model_design"]["run_encoder"]["num_windows"] = int(smoke_cfg.get("num_windows", 5))
    data = h4.load_run_data(smoke_config, sequence_length=smoke_config["sequence"]["sequence_length"])
    source, target = smoke_cfg["scenario"].split("_to_")
    result = h4.run_scenario_seed(
        data,
        smoke_config,
        source,
        target,
        int(smoke_cfg.get("seed", 0)),
        output_dir,
        list(smoke_cfg.get("models", ["feature_ridge", *DL_MODELS])),
        max_epochs=int(smoke_cfg.get("max_epochs", 2)),
        batch_size=int(smoke_cfg.get("batch_size", 4)),
        smoke=True,
    )
    summary = {
        "scenario": smoke_cfg["scenario"],
        "seed": int(smoke_cfg.get("seed", 0)),
        "models": list(smoke_cfg.get("models", ["feature_ridge", *DL_MODELS])),
        "feature_sequence_length": smoke_config["sequence"]["sequence_length"],
        "window_length": int(smoke_config["model_design"]["run_encoder"]["window_length"]),
        "num_windows": int(smoke_config["model_design"]["run_encoder"]["num_windows"]),
        "windowed_input_shape": list(data.sequences.shape[1:]),
        "success": bool(not result["domain_metrics"].empty),
        "prediction_rows": int(len(result["predictions"])),
        "metrics_rows": int(len(result["domain_metrics"])),
        "skipped_rows": int(len(result["skipped"])),
        "target_domain_leakage": False,
    }
    write_json(output_dir / "smoke" / f"{PREFIX}_extended_smoke_test_result.json", summary)
    return summary


def summarize_model(metrics: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
    subset = metrics.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "scope",
                "model_name",
                "uses_previous_runs",
                "sequence_size",
                "mean_RMSE",
                "std_RMSE",
                "mean_MAE",
                "mean_R2",
                "num_conditions",
                "rank_by_RMSE",
            ]
        )
    out = (
        subset.groupby(["model_name", "uses_previous_runs", "sequence_size"], as_index=False)
        .agg(
            mean_RMSE=("target_domain_RMSE", "mean"),
            std_RMSE=("target_domain_RMSE", "std"),
            mean_MAE=("target_domain_MAE", "mean"),
            mean_R2=("target_domain_R2", "mean"),
            num_conditions=("target_domain_RMSE", "size"),
        )
        .sort_values("mean_RMSE")
    )
    out.insert(0, "scope", label)
    out["rank_by_RMSE"] = np.arange(1, len(out) + 1)
    return out


def make_shift_metrics(domain_metrics: pd.DataFrame) -> pd.DataFrame:
    out = domain_metrics.copy()
    out["metric_MAE"] = out["target_domain_MAE"]
    out["metric_RMSE"] = out["target_domain_RMSE"]
    out["metric_R2"] = out["target_domain_R2"]
    return out


def make_seed_metrics(domain_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed), g in domain_metrics.groupby(["model_name", "seed"]):
        rows.append(
            {
                "experiment_id": g["experiment_id"].iloc[0],
                "model_name": model,
                "uses_previous_runs": bool(g["uses_previous_runs"].iloc[0]),
                "sequence_size": int(g["sequence_size"].iloc[0]),
                "seed": int(seed),
                "mean_MAE_over_scenarios": float(g["target_domain_MAE"].mean()),
                "mean_RMSE_over_scenarios": float(g["target_domain_RMSE"].mean()),
                "mean_R2_over_scenarios": float(g["target_domain_R2"].mean()),
                "num_scenarios": int(g["scenario_name"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def pair_summaries(metrics: pd.DataFrame, plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = plan.groupby(["source_domain", "target_domain", "model_name"])["status"].value_counts().unstack(fill_value=0).reset_index()
    matrix = (
        metrics.groupby(["source_domain", "target_domain", "model_name"], as_index=False)
        .agg(
            mean_RMSE=("target_domain_RMSE", "mean"),
            std_RMSE=("target_domain_RMSE", "std"),
            mean_MAE=("target_domain_MAE", "mean"),
            mean_R2=("target_domain_R2", "mean"),
        )
        if not metrics.empty
        else pd.DataFrame()
    )
    rows = []
    for source, target in scenario_pairs():
        for model in MODELS:
            r = matrix.loc[
                (matrix["source_domain"] == source) & (matrix["target_domain"] == target) & (matrix["model_name"] == model)
            ] if not matrix.empty else pd.DataFrame()
            c = counts.loc[(counts["source_domain"] == source) & (counts["target_domain"] == target) & (counts["model_name"] == model)]
            rows.append(
                {
                    "source_domain": source,
                    "target_domain": target,
                    "model_name": model,
                    "mean_RMSE": float(r["mean_RMSE"].iloc[0]) if not r.empty else np.nan,
                    "std_RMSE": float(r["std_RMSE"].iloc[0]) if not r.empty else np.nan,
                    "mean_MAE": float(r["mean_MAE"].iloc[0]) if not r.empty else np.nan,
                    "mean_R2": float(r["mean_R2"].iloc[0]) if not r.empty else np.nan,
                    "num_reused_conditions": int(c["reused"].iloc[0]) if not c.empty and "reused" in c else 0,
                    "num_executed_conditions": int(c["executed"].iloc[0]) if not c.empty and "executed" in c else 0,
                }
            )
    matrix_full = pd.DataFrame(rows)
    target_summary = (
        metrics.groupby(["target_domain", "model_name"], as_index=False)
        .agg(
            mean_RMSE_when_target=("target_domain_RMSE", "mean"),
            std_RMSE_when_target=("target_domain_RMSE", "std"),
            mean_MAE_when_target=("target_domain_MAE", "mean"),
            mean_R2_when_target=("target_domain_R2", "mean"),
            num_scenarios=("scenario_name", "nunique"),
        )
        if not metrics.empty
        else pd.DataFrame()
    )
    source_summary = (
        metrics.groupby(["source_domain", "model_name"], as_index=False)
        .agg(
            mean_RMSE_when_source=("target_domain_RMSE", "mean"),
            std_RMSE_when_source=("target_domain_RMSE", "std"),
            mean_MAE_when_source=("target_domain_MAE", "mean"),
            mean_R2_when_source=("target_domain_R2", "mean"),
            num_target_domains=("target_domain", "nunique"),
        )
        if not metrics.empty
        else pd.DataFrame()
    )
    target_counts = plan.groupby(["target_domain", "model_name"])["status"].value_counts().unstack(fill_value=0).reset_index()
    source_counts = plan.groupby(["source_domain", "model_name"])["status"].value_counts().unstack(fill_value=0).reset_index()
    target_summary = target_summary.merge(target_counts, on=["target_domain", "model_name"], how="left")
    source_summary = source_summary.merge(source_counts, on=["source_domain", "model_name"], how="left")
    target_summary["target_cases"] = target_summary["target_domain"].map(lambda x: ",".join(map(str, PAIRS[x])))
    source_summary["source_cases"] = source_summary["source_domain"].map(lambda x: ",".join(map(str, PAIRS[x])))
    target_summary["num_reused_conditions"] = target_summary.get("reused", 0)
    target_summary["num_executed_conditions"] = target_summary.get("executed", 0)
    source_summary["num_reused_conditions"] = source_summary.get("reused", 0)
    source_summary["num_executed_conditions"] = source_summary.get("executed", 0)
    target_summary["target_difficulty_rank"] = target_summary.groupby("model_name")["mean_RMSE_when_target"].rank(ascending=False, method="dense")
    source_summary["source_generalization_rank"] = source_summary.groupby("model_name")["mean_RMSE_when_source"].rank(ascending=True, method="dense")
    target_summary["interpretation"] = "higher rank means more difficult target pair"
    source_summary["interpretation"] = "lower RMSE source pairs generalize better"
    return target_summary, source_summary, matrix_full


def model_ranking_change(original_summary: pd.DataFrame, extended_summary: pd.DataFrame) -> pd.DataFrame:
    left = original_summary[["model_name", "mean_RMSE", "rank_by_RMSE"]].rename(
        columns={"mean_RMSE": "original_ABC_mean_RMSE", "rank_by_RMSE": "original_ABC_rank"}
    )
    right = extended_summary[["model_name", "mean_RMSE", "rank_by_RMSE"]].rename(
        columns={"mean_RMSE": "extended_ABCDEF_mean_RMSE", "rank_by_RMSE": "extended_ABCDEF_rank"}
    )
    out = left.merge(right, on="model_name", how="outer")
    out["rank_change"] = out["extended_ABCDEF_rank"] - out["original_ABC_rank"]
    out["RMSE_change"] = out["extended_ABCDEF_mean_RMSE"] - out["original_ABC_mean_RMSE"]
    out["interpretation"] = np.where(out["rank_change"].fillna(0) == 0, "ranking unchanged", "ranking changed after adding D/E/F")
    return out.sort_values("extended_ABCDEF_rank")


def make_reuse_summaries(plan: pd.DataFrame, output_dir: Path) -> None:
    plan["status"].value_counts().rename_axis("status").reset_index(name="count").to_csv(
        output_dir / "analysis" / f"{PREFIX}_reused_existing_result_summary.csv", index=False
    )
    plan.loc[plan["status"] == "executed"].to_csv(output_dir / "analysis" / f"{PREFIX}_newly_executed_result_summary.csv", index=False)


def make_extended_filter(h4: Any, predictions: pd.DataFrame, case_metrics: pd.DataFrame, target_metrics: pd.DataFrame, config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    # Reused H4_S1 prediction files may already contain visualization helper columns.
    # Drop them before recomputing the extended-scope case-RMSE filter, otherwise
    # pandas merge suffixes hide the exact `test_case_RMSE` name expected by the
    # base plotting helper.
    stale_cols = [
        "test_case_RMSE",
        "target_domain_RMSE",
        "include_in_visualization",
        "included_in_visualization",
    ]
    pred_clean = predictions.drop(columns=[c for c in stale_cols if c in predictions.columns], errors="ignore")
    filtered_predictions, filter_summary, excluded_summary = h4.apply_visualization_filter(pred_clean, case_metrics, target_metrics, config, output_dir)
    filter_summary.to_csv(output_dir / "analysis" / f"{PREFIX}_extended_visualization_filter_summary.csv", index=False)
    excluded_summary.to_csv(output_dir / "analysis" / f"{PREFIX}_extended_excluded_high_rmse_summary.csv", index=False)
    return filtered_predictions


def make_figures(output_dir: Path, target_metrics: pd.DataFrame, predictions: pd.DataFrame, original_summary: pd.DataFrame, extended_summary: pd.DataFrame, matrix: pd.DataFrame, target_summary: pd.DataFrame, source_summary: pd.DataFrame) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not extended_summary.empty:
        plt.figure(figsize=(9, 4.5))
        plt.bar(extended_summary["model_name"], extended_summary["mean_RMSE"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Mean target-domain RMSE")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_extended_model_comparison_rmse.png", dpi=180)
        plt.close()

    ranking = model_ranking_change(original_summary, extended_summary)
    if not ranking.empty:
        x = np.arange(len(ranking))
        plt.figure(figsize=(9, 4.5))
        plt.plot(x, ranking["original_ABC_rank"], marker="o", label="original ABC")
        plt.plot(x, ranking["extended_ABCDEF_rank"], marker="o", label="extended ABCDEF")
        plt.xticks(x, ranking["model_name"], rotation=35, ha="right")
        plt.gca().invert_yaxis()
        plt.ylabel("Rank by RMSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_original_vs_extended_model_ranking.png", dpi=180)
        plt.close()

    best_model = extended_summary["model_name"].iloc[0] if not extended_summary.empty else MODELS[0]
    piv = matrix.loc[matrix["model_name"] == best_model].pivot(index="source_domain", columns="target_domain", values="mean_RMSE")
    if not piv.empty:
        plt.figure(figsize=(7, 5.5))
        im = plt.imshow(piv, cmap="viridis")
        plt.colorbar(im, label=f"{best_model} mean RMSE")
        plt.xticks(range(len(piv.columns)), piv.columns)
        plt.yticks(range(len(piv.index)), piv.index)
        plt.xlabel("Target pair")
        plt.ylabel("Source pair")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_transfer_scenario_rmse_matrix.png", dpi=180)
        plt.close()

    for df, value, index, fname, label in [
        (target_summary, "mean_RMSE_when_target", "target_domain", f"{PREFIX}_pair_as_target_rmse_heatmap.png", "RMSE as target"),
        (source_summary, "mean_RMSE_when_source", "source_domain", f"{PREFIX}_pair_as_source_rmse_heatmap.png", "RMSE as source"),
    ]:
        if not df.empty:
            piv2 = df.pivot(index="model_name", columns=index, values=value)
            plt.figure(figsize=(8, 5))
            im = plt.imshow(piv2, cmap="magma")
            plt.colorbar(im, label=label)
            plt.xticks(range(len(piv2.columns)), piv2.columns)
            plt.yticks(range(len(piv2.index)), piv2.index)
            plt.tight_layout()
            plt.savefig(fig_dir / fname, dpi=180)
            plt.close()

    if not target_summary.empty:
        diff = target_summary.groupby("target_domain", as_index=False)["mean_RMSE_when_target"].mean().sort_values("mean_RMSE_when_target", ascending=False)
        plt.figure(figsize=(7, 4))
        plt.bar(diff["target_domain"], diff["mean_RMSE_when_target"])
        plt.ylabel("Mean RMSE as target")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_pair_difficulty_summary.png", dpi=180)
        plt.close()

    filtered = predictions.loc[predictions.get("include_in_visualization", True)].copy()
    new_filtered = filtered.loc[filtered["source_domain"].isin(["D", "E", "F"]) | filtered["target_domain"].isin(["D", "E", "F"])] if not filtered.empty else pd.DataFrame()
    if not new_filtered.empty:
        plt.figure(figsize=(10, 5))
        plot_df = new_filtered.groupby(["model_name", "run_order"], as_index=False)["absolute_error"].mean()
        for model, g in plot_df.groupby("model_name"):
            plt.plot(g["run_order"], g["absolute_error"], marker="o", label=model)
        plt.xlabel("run_order")
        plt.ylabel("Mean absolute error, filtered")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{PREFIX}_extended_prediction_error_over_run_order_filtered.png", dpi=180)
        plt.close()

        top_cases = sorted(new_filtered["case_id"].unique())[:6]
        fig, axes = plt.subplots(math.ceil(len(top_cases) / 3), 3, figsize=(13, 3.6 * math.ceil(len(top_cases) / 3)), squeeze=False)
        avg = new_filtered.groupby(["case_id", "run_order", "model_name"], as_index=False).agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
        for ax, case_id in zip(axes.ravel(), top_cases):
            cg = avg.loc[avg["case_id"] == case_id]
            true = cg.drop_duplicates("run_order").sort_values("run_order")
            ax.plot(true["run_order"], true["y_true"], color="black", linewidth=2, label="y_true")
            for model, g in cg.groupby("model_name"):
                g = g.sort_values("run_order")
                ax.plot(g["run_order"], g["y_pred"], marker="o", markersize=3, label=model)
            ax.set_title(f"Case {case_id}")
        for ax in axes.ravel()[len(top_cases) :]:
            ax.axis("off")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(fig_dir / f"{PREFIX}_new_pairs_prediction_over_run_order.png", dpi=180)
        plt.close(fig)


def write_report(
    output_dir: Path,
    config: dict[str, Any],
    data_summary: dict[str, Any],
    existing_dir: Path | None,
    plan: pd.DataFrame,
    smoke: dict[str, Any],
    original_summary: pd.DataFrame,
    extended_summary: pd.DataFrame,
    new_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    target_summary: pd.DataFrame,
    source_summary: pd.DataFrame,
) -> None:
    reused_count = int((plan["status"] == "reused").sum())
    executed_count = int((plan["status"] == "executed").sum())
    skipped_count = int((plan["status"] == "skipped").sum())
    pending_count = int((plan["status"] == "pending").sum())
    best = extended_summary.iloc[0].to_dict() if not extended_summary.empty else {}
    new_best = new_summary.iloc[0].to_dict() if not new_summary.empty else {}
    report = f"""# H4_S1 Extended Condition Pair Domain Generalization

## Existing Result Reuse Summary

- Existing H4_S1 directory: `{existing_dir if existing_dir else 'not found'}`
- Reused atomic conditions: {reused_count}
- Newly executed atomic conditions: {executed_count}
- Skipped atomic conditions: {skipped_count}
- Remaining pending atomic conditions after run: {pending_count}
- Reuse rule: completed A/B/C original-scope conditions were reused and not rerun.

## Extended Condition Pair Setup

- Original pairs: A={PAIRS['A']}, B={PAIRS['B']}, C={PAIRS['C']}
- Added pairs: D={PAIRS['D']}, E={PAIRS['E']}, F={PAIRS['F']}
- Total ordered transfer scenarios: 30
- Seeds requested: {config['experiment']['seed_list']}
- Models: {MODELS}
- DL max epochs: {config['training']['max_epochs']}; patience: {config['training']['early_stopping']['patience']}

## Extended Processed Data

- Created process file: `{data_summary['extended_process_path']}`
- Created signal file: `{data_summary['extended_signal_path']}`
- Selected cases: {data_summary['selected_cases']}
- Missing selected cases: {data_summary['missing_cases']}
- VB non-null by case: `{data_summary['vb_non_null_by_case']}`

## Smoke Test

```json
{json.dumps(to_builtin(smoke), indent=2, ensure_ascii=False)}
```

## Original vs Extended Results

Best extended-scope model by RMSE: `{best.get('model_name')}` with mean RMSE `{best.get('mean_RMSE')}`.

### Original A/B/C Scope

{original_summary.to_markdown(index=False)}

### Extended A/B/C/D/E/F Scope

{extended_summary.to_markdown(index=False)}

### New-Pair-Involved Scenarios Only

{new_summary.to_markdown(index=False) if not new_summary.empty else 'No new-pair metrics were generated.'}

New-pair best model: `{new_best.get('model_name')}` with mean RMSE `{new_best.get('mean_RMSE')}`.

## Model Ranking Change

{ranking.to_markdown(index=False)}

## Pair Difficulty Analysis

The full pair-as-target and pair-as-source summaries are saved as:

- `analysis/{PREFIX}_pair_as_target_summary.csv`
- `analysis/{PREFIX}_pair_as_source_summary.csv`

Target difficulty is ranked by higher target RMSE. Source generalization is ranked by lower source RMSE.

## Reuse and Execution Files

- Reuse plan: `analysis/{PREFIX}_execution_reuse_plan.csv`
- Extended case metrics: `metrics/{PREFIX}_extended_case_metrics.csv`
- Extended target metrics: `metrics/{PREFIX}_extended_target_domain_metrics.csv`
- Extended predictions: `predictions/{PREFIX}_extended_test_predictions.csv`
- Visualization filter: `analysis/{PREFIX}_extended_visualization_filter_summary.csv`

## Interpretation

The extended scope adds D/E/F operating-condition pairs and therefore is a stricter domain generalization test than the original A/B/C scope. Metrics and ranking include every completed condition. RMSE > 0.5 filtering is applied only to prediction trajectory figures, not to model comparison or aggregate metrics.
"""
    (output_dir / "reports" / f"{PREFIX}_extended_condition_pair_report.md").write_text(report, encoding="utf-8")
    html = "<html><body>" + report.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>\n") + "</body></html>"
    (output_dir / "reports" / f"{PREFIX}_extended_condition_pair_report.html").write_text(html, encoding="utf-8")


def run_extended(args: argparse.Namespace) -> Path:
    h4 = load_base_module()
    h4.torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else list(range(30))
    data_summary = prepare_extended_processed_files()
    existing_dir = find_existing_h4_s1(Path(args.output_root))
    config = build_config(h4, seeds)

    output_dir = Path(create_execution_dir(config, root=args.output_root))
    h4.make_dirs(output_dir)
    for extra_dir in ["smoke"]:
        (output_dir / extra_dir).mkdir(parents=True, exist_ok=True)
    config["experiment"]["execution_dir"] = str(output_dir)
    if str(config["experiment"].get("experiment_id", "")).lower() == "auto":
        config["experiment"]["experiment_id"] = f"{output_dir.name.split('_cnn_gru_run_sequence_domain_generalization_NASA_Ames_extended_condition_pairs')[0]}_H4_S1_{TOPIC}"

    write_yaml(output_dir / "configs" / f"{PREFIX}_extended_condition_pair_config.yaml", config)
    write_yaml(output_dir / "configs" / f"{PREFIX}_input_config.yaml", config)
    write_json(output_dir / "data" / f"{PREFIX}_extended_processed_data_summary.json", data_summary)

    data = h4.load_run_data(config, sequence_length=int(config["sequence"].get("sequence_length", 256)))
    h4.write_data_artifacts(data, config, output_dir)
    validation = h4.validate_config_and_data(config, data, output_dir)
    validation.update(
        {
            "extended_processed_files_created": True,
            "extended_data_summary": data_summary,
            "condition_pair_count": len(PAIRS),
            "transfer_scenario_count": len(scenario_pairs()),
            "existing_reuse_dir": str(existing_dir) if existing_dir else None,
        }
    )
    write_json(output_dir / "configs" / f"{PREFIX}_config_validation.json", validation)
    write_yaml(output_dir / "configs" / f"{PREFIX}_resolved_config.yaml", config)
    write_json(output_dir / "logs" / f"{PREFIX}_environment.json", {"python": os.sys.version, "torch": h4.torch.__version__, "device": "cuda" if h4.torch.cuda.is_available() else "cpu"})
    run_log = output_dir / "logs" / f"{PREFIX}_run.log"
    run_log.write_text(f"Extended processed files created. Existing reuse dir={existing_dir}\n", encoding="utf-8")
    (output_dir / "logs" / f"{PREFIX}_error.log").write_text("", encoding="utf-8")

    reusable = load_reusable_frames(existing_dir, config["experiment"]["experiment_id"])
    available_cases = set(data.meta["case_id"].astype(int))
    plan = build_reuse_plan(reusable["target"], existing_dir, seeds, available_cases)
    plan.to_csv(output_dir / "analysis" / f"{PREFIX}_execution_reuse_plan.csv", index=False)

    if args.dry_run_only:
        return output_dir

    smoke = run_smoke(h4, config, output_dir) if not args.skip_smoke else {"skipped": True}
    run_log.write_text(run_log.read_text(encoding="utf-8") + f"Smoke success={smoke.get('success')}\n", encoding="utf-8")

    executed_case: list[pd.DataFrame] = []
    executed_target: list[pd.DataFrame] = []
    executed_pred: list[pd.DataFrame] = []
    executed_split: list[pd.DataFrame] = []
    executed_history: list[pd.DataFrame] = []
    executed_best: list[pd.DataFrame] = []
    executed_skipped: list[pd.DataFrame] = []
    scaler_records: list[dict[str, Any]] = []

    pending_groups = (
        plan.loc[plan["needs_execution"]]
        .groupby(["scenario_name", "source_domain", "target_domain", "seed"], sort=True)
    )
    start_all = time.time()
    for (scen, source, target, seed), group in pending_groups:
        models = [m for m in MODELS if m in set(group["model_name"])]
        try:
            result = h4.run_scenario_seed(data, config, source, target, int(seed), output_dir, models=models)
            for key, bucket in [
                ("case_metrics", executed_case),
                ("domain_metrics", executed_target),
                ("predictions", executed_pred),
                ("split", executed_split),
                ("learning_history", executed_history),
                ("best_epoch_summary", executed_best),
                ("skipped", executed_skipped),
            ]:
                if not result[key].empty:
                    bucket.append(result[key])
            scaler_records.append(result["scaler_fit"])
            executed_models = set(result["domain_metrics"]["model_name"]) if not result["domain_metrics"].empty else set()
            skipped_models = set(result["skipped"]["model_name"]) if not result["skipped"].empty else set()
            idx = group.index
            plan.loc[idx.intersection(group.loc[group["model_name"].isin(executed_models)].index), "status"] = "executed"
            plan.loc[idx.intersection(group.loc[group["model_name"].isin(executed_models)].index), "needs_execution"] = False
            plan.loc[idx.intersection(group.loc[group["model_name"].isin(executed_models)].index), "reason"] = "executed in extended run"
            plan.loc[idx.intersection(group.loc[group["model_name"].isin(skipped_models)].index), "status"] = "skipped"
            plan.loc[idx.intersection(group.loc[group["model_name"].isin(skipped_models)].index), "needs_execution"] = False
            plan.loc[idx.intersection(group.loc[group["model_name"].isin(skipped_models)].index), "reason"] = "model skipped during execution"
            run_log.write_text(run_log.read_text(encoding="utf-8") + f"Completed {scen} seed={seed} models={models}\n", encoding="utf-8")
        except Exception as exc:
            plan.loc[group.index, "status"] = "failed"
            plan.loc[group.index, "needs_execution"] = False
            plan.loc[group.index, "reason"] = f"execution_failed: {exc}"
            with (output_dir / "logs" / f"{PREFIX}_error.log").open("a", encoding="utf-8") as f:
                f.write(f"{scen} seed={seed} failed: {exc}\n")

    plan.to_csv(output_dir / "analysis" / f"{PREFIX}_execution_reuse_plan.csv", index=False)
    execution_seconds = time.time() - start_all

    exec_case = add_extended_columns(pd.concat(executed_case, ignore_index=True) if executed_case else pd.DataFrame(), "executed", "", config["experiment"]["experiment_id"])
    exec_target = add_extended_columns(pd.concat(executed_target, ignore_index=True) if executed_target else pd.DataFrame(), "executed", "", config["experiment"]["experiment_id"])
    exec_pred = add_extended_columns(pd.concat(executed_pred, ignore_index=True) if executed_pred else pd.DataFrame(), "executed", "", config["experiment"]["experiment_id"])
    all_case = pd.concat([reusable["case"], exec_case], ignore_index=True) if not reusable["case"].empty or not exec_case.empty else pd.DataFrame()
    all_target = pd.concat([reusable["target"], exec_target], ignore_index=True) if not reusable["target"].empty or not exec_target.empty else pd.DataFrame()
    all_pred = pd.concat([reusable["pred"], exec_pred], ignore_index=True) if not reusable["pred"].empty or not exec_pred.empty else pd.DataFrame()
    all_history = pd.concat([reusable["history"], *executed_history], ignore_index=True) if executed_history or not reusable["history"].empty else pd.DataFrame()
    all_best = pd.concat([reusable["best_epoch"], *executed_best], ignore_index=True) if executed_best or not reusable["best_epoch"].empty else pd.DataFrame()
    all_splits = pd.concat([reusable["split"], *executed_split], ignore_index=True) if executed_split or not reusable["split"].empty else pd.DataFrame()
    all_skipped = pd.concat(executed_skipped, ignore_index=True) if executed_skipped else pd.DataFrame()

    all_pred = make_extended_filter(h4, all_pred, all_case, all_target, config, output_dir)
    shift_metrics = make_shift_metrics(all_target)
    seed_metrics = make_seed_metrics(all_target)
    original_summary = summarize_model(all_target, all_target["scenario_name"].isin(ORIGINAL_SCENARIOS), "original_ABC")
    extended_summary = summarize_model(all_target, pd.Series(True, index=all_target.index), "extended_ABCDEF")
    new_summary = summarize_model(all_target, all_target["is_new_pair_involved"], "new_pairs_only")
    target_summary, source_summary, matrix = pair_summaries(all_target, plan)
    ranking = model_ranking_change(original_summary, extended_summary)

    all_case.to_csv(output_dir / "metrics" / f"{PREFIX}_extended_case_metrics.csv", index=False)
    all_target.to_csv(output_dir / "metrics" / f"{PREFIX}_extended_target_domain_metrics.csv", index=False)
    shift_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_extended_shift_metrics.csv", index=False)
    seed_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_extended_seed_metrics.csv", index=False)
    extended_summary.rename(
        columns={
            "mean_RMSE": "mean_RMSE_over_scenarios",
            "std_RMSE": "std_RMSE_over_scenarios",
            "mean_MAE": "mean_MAE_over_scenarios",
            "mean_R2": "mean_R2_over_scenarios",
        }
    ).to_csv(output_dir / "metrics" / f"{PREFIX}_extended_model_comparison.csv", index=False)
    all_pred.to_csv(output_dir / "predictions" / f"{PREFIX}_extended_test_predictions.csv", index=False)
    all_pred.to_csv(output_dir / "predictions" / f"{PREFIX}_extended_prediction_errors.csv", index=False)
    all_history.to_csv(output_dir / "metrics" / f"{PREFIX}_extended_learning_history.csv", index=False)
    all_best.to_csv(output_dir / "analysis" / f"{PREFIX}_extended_best_epoch_summary.csv", index=False)
    all_skipped.to_csv(output_dir / "analysis" / f"{PREFIX}_extended_skipped_conditions.csv", index=False)
    if not all_splits.empty:
        all_splits.to_csv(output_dir / "splits" / f"{PREFIX}_extended_split.csv", index=False)

    original_summary.to_csv(output_dir / "analysis" / f"{PREFIX}_original_ABC_scope_summary.csv", index=False)
    extended_summary.to_csv(output_dir / "analysis" / f"{PREFIX}_extended_ABCDEF_scope_summary.csv", index=False)
    new_summary.to_csv(output_dir / "analysis" / f"{PREFIX}_new_pairs_only_summary.csv", index=False)
    target_summary.to_csv(output_dir / "analysis" / f"{PREFIX}_pair_as_target_summary.csv", index=False)
    source_summary.to_csv(output_dir / "analysis" / f"{PREFIX}_pair_as_source_summary.csv", index=False)
    matrix.to_csv(output_dir / "analysis" / f"{PREFIX}_transfer_scenario_rmse_matrix.csv", index=False)
    matrix.to_csv(output_dir / "analysis" / f"{PREFIX}_extended_domain_shift_analysis.csv", index=False)
    ranking.to_csv(output_dir / "analysis" / f"{PREFIX}_model_ranking_original_vs_extended.csv", index=False)
    ranking.to_csv(output_dir / "analysis" / f"{PREFIX}_performance_change_after_adding_pairs.csv", index=False)
    make_reuse_summaries(plan, output_dir)

    leak = h4.leakage_check(all_splits, config) if not all_splits.empty else {"passed": True, "note": "No split rows available."}
    leak["extended_condition_pair_policy"] = "source-only validation, train-only scaler fit, no cross-case sequence"
    write_json(output_dir / "splits" / f"{PREFIX}_leakage_check.json", leak)
    write_json(
        output_dir / "analysis" / f"{PREFIX}_analysis_summary.json",
        {
            "execution_seconds": execution_seconds,
            "reused_conditions": int((plan["status"] == "reused").sum()),
            "executed_conditions": int((plan["status"] == "executed").sum()),
            "skipped_conditions": int((plan["status"] == "skipped").sum()),
            "failed_conditions": int((plan["status"] == "failed").sum()),
            "best_extended_model": extended_summary.iloc[0].to_dict() if not extended_summary.empty else {},
            "best_new_pair_model": new_summary.iloc[0].to_dict() if not new_summary.empty else {},
        },
    )
    make_figures(output_dir, all_target, all_pred, original_summary, extended_summary, matrix, target_summary, source_summary)
    if not all_history.empty:
        # Reuse base plotting names for a quick learning-history view, then keep them under the extended execution.
        h4.make_learning_history_figures(output_dir, all_history.rename(columns={"condition_pair_source": "source_domain", "condition_pair_target": "target_domain"}, errors="ignore"))
    write_report(output_dir, config, data_summary, existing_dir, plan, smoke, original_summary, extended_summary, new_summary, ranking, target_summary, source_summary)
    run_log.write_text(run_log.read_text(encoding="utf-8") + f"Finished extended run in {execution_seconds:.1f} sec\n", encoding="utf-8")
    print(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="experiments/executions")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Default: 0..29.")
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    run_extended(args)


if __name__ == "__main__":
    main()

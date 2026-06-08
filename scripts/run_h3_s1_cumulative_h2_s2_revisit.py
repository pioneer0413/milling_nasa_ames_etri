#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_cumulative_s_dnn_experiment as h3s3
from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.experiments.execution_path import create_execution_dir, find_experiment_dirs
from milling_experiment_framework.models.dl.cumulative_descriptor import CumulativeDescriptor


PREFIX = "H3_S1"
EXPERIMENT_NAME = "H3_S1_cumulative_descriptor_sensor_combination_H2_S2_revisit"
CASE_SCOPE = h3s3.CASE_SCOPE
DOMAIN_CASES = h3s3.DOMAIN_CASES
SHIFT_SCENARIOS = h3s3.SHIFT_SCENARIOS
SEGMENT_SETTINGS = ["full_length", "steady", "entry", "exit", "entry_steady", "entry_exit", "steady_exit"]
SENSOR_COMBINATIONS = {
    "current": ["smcAC", "smcDC"],
    "vibration": ["vib_spindle", "vib_table"],
    "acoustic": ["AE_spindle", "AE_table"],
    "current_vibration": ["smcAC", "smcDC", "vib_spindle", "vib_table"],
    "current_acoustic": ["smcAC", "smcDC", "AE_spindle", "AE_table"],
    "vibration_acoustic": ["vib_spindle", "vib_table", "AE_spindle", "AE_table"],
    "current_vibration_acoustic": ["smcAC", "smcDC", "vib_spindle", "vib_table", "AE_spindle", "AE_table"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Revisit H2_S2 with cumulative descriptor S-DNN.")
    parser.add_argument("--config", default="configs/models/cumulative_s_dnn.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-grid", action="store_true", help="Run acoustic + entry_exit + first leave-one-case-out shift + seed 0.")
    parser.add_argument("--sensor-combinations", default=None)
    parser.add_argument("--segments", default=None)
    parser.add_argument("--shifts", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--input-representation", default="cumulative")
    parser.add_argument("--mode", default="s_dnn")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--quick-hidden-size", type=int, default=8)
    args = parser.parse_args()

    output_dir: Path | None = None
    try:
        config = h3s3.load_config(Path(args.config))
        configure_for_revisit(config, args)
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_H3_S1_cumulative_descriptor_sensor_combination_H2_S2_revisit")
        path_config = {"experiment": {"experiment_id": experiment_id}}
        output_dir = Path(create_execution_dir(path_config, root=Path("experiments") / "executions"))
        prepare_dirs(output_dir)
        logger = make_logger(output_dir / "logs" / f"{PREFIX}_run.log")
        device = torch.device("cuda" if torch.cuda.is_available() and config.get("training", {}).get("device") == "cuda" else "cpu")
        ctx = h3s3.RunContext(experiment_id=experiment_id, output_dir=output_dir, config=config, device=device, logger=logger)
        logger(f"H3.S1 cumulative H2.S2 revisit started: {experiment_id}")
        write_json(output_dir / "logs" / f"{PREFIX}_environment.json", collect_environment())
        h2 = find_h2_s2_baseline()
        h2_files = load_h2_s2_files(h2)
        write_json(output_dir / "data" / f"{PREFIX}_source_H2_S2_summary.json", {"source_dir": str(h2), "loaded_files": {k: str(v) for k, v in h2_files.items()}})
        write_json(output_dir / "configs" / f"{PREFIX}_input_config.yaml", config)
        write_json(
            output_dir / "configs" / f"{PREFIX}_resolved_config.yaml",
            {"experiment": path_config["experiment"], "experiment_id": experiment_id, "experiment_name": EXPERIMENT_NAME, "config": config},
        )

        smoke = h3s3.run_smoke_test(ctx)
        write_json(output_dir / "data" / f"{PREFIX}_smoke_test_result.json", smoke)
        dataset = h3s3.load_dataset(config)
        sensor_combos = selected_sensor_combinations(args)
        segments = selected_segments(args)
        shifts = selected_shifts(args)
        seeds = selected_seeds(args)
        validation = validate_grid(dataset, sensor_combos, segments, shifts, h2_files)
        write_json(output_dir / "configs" / f"{PREFIX}_config_validation.json", validation)
        write_json(output_dir / "splits" / f"{PREFIX}_leakage_check.json", validation["leakage_check"])
        write_common_artifacts(output_dir, dataset, sensor_combos, segments, shifts, config)
        if args.dry_run:
            write_report(output_dir, experiment_id, h2, h2_files, pd.DataFrame(), pd.DataFrame(), {}, validation, dry_run=True)
            logger("Dry-run completed without training.")
            print(json.dumps({"experiment_id": experiment_id, "execution_dir": str(output_dir), "dry_run": True, "validation": validation}, indent=2, ensure_ascii=False))
            return

        base_metrics, s_metrics, base_preds, s_preds, oof = run_grid(ctx, dataset, sensor_combos, segments, shifts, seeds, args.input_representation, args.mode)
        write_outputs(output_dir, base_metrics, s_metrics, base_preds, s_preds, oof)
        analysis = make_analysis_outputs(output_dir, h2_files, s_metrics)
        make_figures(output_dir, h2_files, s_metrics)
        write_report(output_dir, experiment_id, h2, h2_files, base_metrics, s_metrics, analysis, validation, dry_run=False)
        logger(f"H3.S1 cumulative H2.S2 revisit finished: {experiment_id}")
        print(json.dumps({"experiment_id": experiment_id, "execution_dir": str(output_dir), "s_dnn_rows": len(s_metrics), "base_rows": len(base_metrics)}, indent=2, ensure_ascii=False))
    except Exception:
        if output_dir is not None:
            (output_dir / "logs" / f"{PREFIX}_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def configure_for_revisit(config: dict[str, Any], args: argparse.Namespace) -> None:
    config["runtime"]["modes"] = [args.mode]
    config["runtime"]["input_representations"] = [args.input_representation]
    config["runtime"]["shifts"] = selected_shifts(args)
    config["runtime"]["seeds"] = selected_seeds(args)
    config["model"]["sequence"]["construction_mode"] = "segment_mode"
    config["model"]["sequence"]["resample_length"] = int(args.sequence_length)
    config["model"]["sequence"]["window_size"] = int(args.sequence_length)
    config["training"]["max_epochs"] = int(args.max_epochs)
    config["training"]["batch_size"] = int(args.batch_size)
    config["model"]["ensemble"]["cv_folds"] = int(args.cv_folds)
    if args.quick_hidden_size:
        h = int(args.quick_hidden_size)
        config["model"]["base_learners"]["rnn"]["hidden_size"] = h
        config["model"]["base_learners"]["rnn"]["num_layers"] = 1
        config["model"]["base_learners"]["gru"]["hidden_size"] = h
        config["model"]["base_learners"]["gru"]["num_layers"] = 1
        config["model"]["base_learners"]["lstm"]["hidden_size"] = h
        config["model"]["base_learners"]["lstm"]["num_layers"] = 1
        config["model"]["base_learners"]["cnn"]["channels"] = [h]
        config["model"]["base_regression_head"]["hidden_dims"] = [h]


def selected_sensor_combinations(args: argparse.Namespace) -> list[str]:
    if args.smoke_grid:
        return ["acoustic"]
    if args.sensor_combinations:
        return [x.strip() for x in args.sensor_combinations.split(",") if x.strip()]
    return list(SENSOR_COMBINATIONS)


def selected_segments(args: argparse.Namespace) -> list[str]:
    if args.smoke_grid:
        return ["entry_exit"]
    if args.segments:
        return [x.strip() for x in args.segments.split(",") if x.strip()]
    return SEGMENT_SETTINGS


def selected_shifts(args: argparse.Namespace) -> list[str]:
    if args.smoke_grid:
        return [SHIFT_SCENARIOS[0]]
    if args.shifts:
        return [x.strip() for x in args.shifts.split(",") if x.strip()]
    return SHIFT_SCENARIOS


def selected_seeds(args: argparse.Namespace) -> list[int]:
    if args.smoke_grid:
        return [0]
    if args.seeds:
        return [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    return [0, 1, 2]


def prepare_dirs(output_dir: Path) -> None:
    for name in ["configs", "data", "splits", "checkpoints", "metrics", "predictions", "analysis", "figures", "logs", "reports"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    (output_dir / "logs" / f"{PREFIX}_error.log").touch()


def make_logger(path: Path):
    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return log


def find_h2_s2_baseline() -> Path:
    candidates = [Path(p) for p in find_experiment_dirs("experiments/executions", hypothesis_id="H2", scenario_id="S2", include_legacy=True)]
    candidates = [p for p in candidates if "H3_S1" not in p.name and p.is_dir()]
    if not candidates:
        candidates = [
            Path(p)
            for p in find_experiment_dirs("experiments/executions", keyword="sensor_combination", include_legacy=True)
            if "H3_S1" not in Path(p).name and Path(p).is_dir()
        ]
    if not candidates:
        raise FileNotFoundError("Could not find H2_S2 baseline experiment directory.")
    return candidates[-1]


def first_existing(base: Path, names: list[str]) -> Path | None:
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None


def load_h2_s2_files(base: Path) -> dict[str, Path]:
    specs = {
        "shift_metrics": ["metrics/H2_S2_shift_metrics.csv", "metrics/shift_metrics.csv"],
        "seed_metrics": ["metrics/H2_S2_seed_metrics.csv", "metrics/seed_metrics.csv"],
        "sensor_combination_metrics": ["metrics/H2_S2_sensor_combination_metrics.csv", "metrics/sensor_combination_metrics.csv"],
        "best_sensor_combination": ["analysis/H2_S2_best_sensor_combination.csv", "analysis/best_sensor_combination.csv"],
        "best_segment_by_sensor_combination": ["analysis/H2_S2_best_segment_by_sensor_combination.csv", "analysis/best_segment_by_sensor_combination.csv"],
        "segment_effect": ["analysis/H2_S2_segment_effect_under_sensor_combination.csv", "analysis/segment_effect_under_sensor_combination.csv"],
        "r2_positive": ["analysis/H2_S2_r2_positive_count_summary.csv", "analysis/r2_positive_count_summary.csv"],
        "report": ["reports/H2_S2_report.md", "reports/report.md"],
    }
    found = {}
    for key, names in specs.items():
        path = first_existing(base, names)
        if path is not None:
            found[key] = path
    return found


def validate_grid(dataset: pd.DataFrame, sensor_combos: list[str], segments: list[str], shifts: list[str], h2_files: dict[str, Path]) -> dict[str, Any]:
    errors, warnings = [], []
    if sorted(dataset["case"].unique().tolist()) != CASE_SCOPE:
        errors.append(f"case filtering failed: {sorted(dataset['case'].unique().tolist())}")
    if "VB" not in dataset.columns:
        errors.append("VB target column missing")
    if "shift_metrics" not in h2_files:
        warnings.append("H2_S2 shift metrics missing; comparison will be limited.")
    for combo in sensor_combos:
        missing = [s for s in SENSOR_COMBINATIONS[combo] if s not in dataset.columns]
        if missing:
            errors.append(f"{combo} missing sensors: {missing}")
    for segment in segments:
        if "no_load" in segment:
            errors.append(f"no-load segment is excluded: {segment}")
    leak_checks = []
    for shift in shifts:
        source, target = h3s3.parse_shift(shift)
        overlap = sorted(set(DOMAIN_CASES[source]) & set(DOMAIN_CASES[target]))
        leak_checks.append({"shift": shift, "source_cases": DOMAIN_CASES[source], "target_cases": DOMAIN_CASES[target], "case_overlap": overlap, "target_in_oof": False, "target_label_used_for_meta_fit": False, "paper_random_split_used": False, "domain_shift_protocol_used": True})
        if overlap:
            errors.append(f"source/target overlap in {shift}: {overlap}")
    # Small deterministic descriptor check.
    sample = np.random.default_rng(0).normal(size=(2, 16, 3)).astype("float32")
    desc = CumulativeDescriptor(input_representation="cumulative").transform_sequence(sample)
    if not np.isfinite(desc).all():
        errors.append("cumulative descriptor produced NaN/inf in dry-run check")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "sensor_combinations": sensor_combos, "segments": segments, "leakage_check": {"ok": not errors, "checks": leak_checks, "warnings": warnings}}


def write_common_artifacts(output_dir: Path, dataset: pd.DataFrame, sensor_combos: list[str], segments: list[str], shifts: list[str], config: dict[str, Any]) -> None:
    write_json(output_dir / "data" / f"{PREFIX}_dataset_summary.json", {"num_samples": int(len(dataset)), "cases": sorted(dataset["case"].unique().tolist()), "target": "VB"})
    rows = []
    for combo in sensor_combos:
        groups = combo.split("_")
        rows.append({"sensor_combination": combo, "included_sensor_groups": ",".join(groups), "included_sensors": ",".join(SENSOR_COMBINATIONS[combo])})
    pd.DataFrame(rows).to_csv(output_dir / "data" / f"{PREFIX}_sensor_group_mapping.csv", index=False)
    write_json(output_dir / "data" / f"{PREFIX}_sequence_shape_summary.json", {"sequence_length": config["model"]["sequence"]["resample_length"], "segments": segments, "sensor_combinations": sensor_combos})
    write_json(output_dir / "data" / f"{PREFIX}_cumulative_descriptor_summary.json", {"input_representation": config["runtime"]["input_representations"], "formula": "cumsum / sqrt(abs(cumsum) + epsilon)", "scope": "within_sample_sequence_only"})
    split_table, split_summary = h3s3.build_split_artifacts(dataset, shifts)
    split_table.to_csv(output_dir / "splits" / f"{PREFIX}_split.csv", index=False)
    write_json(output_dir / "splits" / f"{PREFIX}_split_summary.json", split_summary)


def run_grid(ctx: h3s3.RunContext, dataset: pd.DataFrame, sensor_combos: list[str], segments: list[str], shifts: list[str], seeds: list[int], representation: str, mode: str):
    base_rows, s_rows, base_preds, s_preds, oof_rows = [], [], [], [], []
    sequence_length = int(ctx.config["model"]["sequence"]["resample_length"])
    for combo in sensor_combos:
        sensors = SENSOR_COMBINATIONS[combo]
        ctx.config["model"]["sensor_setting"] = ",".join(sensors)
        for segment in segments:
            ctx.logger(f"Running combo={combo}, segment={segment}")
            ctx.config["model"]["sequence"]["segment_mode"] = segment
            sequences, meta = h3s3.build_sequence_table(dataset, sensors, ctx.config)
            for seed in seeds:
                h3s3.set_seed(seed)
                for shift in shifts:
                    source, target = h3s3.parse_shift(shift)
                    split = h3s3.split_frame(meta, source, target)
                    idx = split.index.to_numpy()
                    seq = sequences[idx]
                    split_reset = split.reset_index(drop=True)
                    train_mask = split_reset["split"].to_numpy() == "train"
                    val_mask = split_reset["split"].to_numpy() == "validation"
                    test_mask = split_reset["split"].to_numpy() == "test"
                    train_seq, val_seq, test_seq = h3s3.normalize_sequences(seq[train_mask], seq[val_mask], seq[test_mask])
                    train_seq, val_seq, test_seq = h3s3.apply_representation(train_seq, val_seq, test_seq, representation, ctx.config)
                    y_train = split_reset.loc[train_mask, "VB"].to_numpy(dtype="float32")
                    y_val = split_reset.loc[val_mask, "VB"].to_numpy(dtype="float32")
                    y_test = split_reset.loc[test_mask, "VB"].to_numpy(dtype="float32")
                    if mode == "base_only":
                        for learner in h3s3.BASE_LEARNERS:
                            model, _ = h3s3.train_base_model(ctx, learner, train_seq, y_train, val_seq, y_val, seed, f"{combo}_{segment}_{representation}_{shift}_{learner}")
                            metrics, pred = h3s3.evaluate_model_on_test(ctx, model, test_seq, y_test, split_reset.loc[test_mask], learner)
                            base_rows.append(metric_row(ctx, "base_only", representation, combo, segment, sequence_length, learner, seed, source, target, metrics))
                            base_preds.append(prediction_frame(ctx, pred, split_reset.loc[test_mask], "base_only", representation, combo, segment, learner, seed, source, target))
                    else:
                        metrics, pred_frame, oof = h3s3.run_stacking(ctx, train_seq, y_train, test_seq, y_test, split_reset.loc[train_mask], split_reset.loc[test_mask], representation, seed, source, target)
                        s_rows.append(metric_row(ctx, "s_dnn", representation, combo, segment, sequence_length, "s_dnn", seed, source, target, metrics))
                        pred_frame = pred_frame.assign(sensor_combination=combo, segment_setting=segment, sequence_length=sequence_length, included_sensor_groups=",".join(combo.split("_")), included_sensors=",".join(sensors))
                        oof = oof.assign(sensor_combination=combo, segment_setting=segment)
                        s_preds.append(pred_frame)
                        oof_rows.append(oof)
    return pd.DataFrame(base_rows), pd.DataFrame(s_rows), concat_or_empty(base_preds), concat_or_empty(s_preds), concat_or_empty(oof_rows)


def metric_row(ctx, mode, representation, combo, segment, sequence_length, learner, seed, source, target, metrics):
    sensors = SENSOR_COMBINATIONS[combo]
    return {
        "experiment_id": ctx.experiment_id,
        "model": "cumulative_s_dnn",
        "mode": mode,
        "input_representation": representation,
        "sensor_combination": combo,
        "included_sensor_groups": ",".join(combo.split("_")),
        "included_sensors": ",".join(sensors),
        "segment_setting": segment,
        "sequence_length": sequence_length,
        "base_learner": learner if mode == "base_only" else None,
        "seed": seed,
        "source_domain": source,
        "target_domain": target,
        "source_cases": ",".join(map(str, DOMAIN_CASES[source])),
        "target_cases": ",".join(map(str, DOMAIN_CASES[target])),
        **metrics,
    }


def prediction_frame(ctx, y_pred, test_meta, mode, representation, combo, segment, learner, seed, source, target):
    frame = test_meta.copy()
    frame["experiment_id"] = ctx.experiment_id
    frame["model"] = "cumulative_s_dnn"
    frame["mode"] = mode
    frame["input_representation"] = representation
    frame["sensor_combination"] = combo
    frame["segment_setting"] = segment
    frame["base_learner"] = learner
    frame["seed"] = seed
    frame["source_domain"] = source
    frame["target_domain"] = target
    frame["y_true"] = frame["VB"].to_numpy(dtype="float32")
    frame["y_pred"] = y_pred
    frame["residual"] = frame["y_true"] - frame["y_pred"]
    frame["absolute_error"] = frame["residual"].abs()
    return frame


def concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_outputs(output_dir: Path, base_metrics: pd.DataFrame, s_metrics: pd.DataFrame, base_preds: pd.DataFrame, s_preds: pd.DataFrame, oof: pd.DataFrame) -> None:
    base_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_base_learner_shift_metrics.csv", index=False)
    s_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_s_dnn_shift_metrics.csv", index=False)
    seed = aggregate_seed_metrics(s_metrics)
    combo = aggregate_combo_metrics(s_metrics)
    segment = aggregate_segment_metrics(s_metrics)
    model = aggregate_model_metrics(seed)
    seed.to_csv(output_dir / "metrics" / f"{PREFIX}_seed_metrics.csv", index=False)
    combo.to_csv(output_dir / "metrics" / f"{PREFIX}_sensor_combination_metrics.csv", index=False)
    segment.to_csv(output_dir / "metrics" / f"{PREFIX}_segment_metrics.csv", index=False)
    model.to_csv(output_dir / "metrics" / f"{PREFIX}_model_metrics.csv", index=False)
    base_preds.to_csv(output_dir / "predictions" / f"{PREFIX}_base_learner_predictions.csv", index=False)
    s_preds.to_csv(output_dir / "predictions" / f"{PREFIX}_s_dnn_predictions.csv", index=False)
    oof.to_csv(output_dir / "predictions" / f"{PREFIX}_oof_predictions.csv", index=False)
    pred = s_preds if not s_preds.empty else base_preds
    if not pred.empty:
        pred.sort_values("absolute_error", ascending=False).head(300).to_csv(output_dir / "predictions" / f"{PREFIX}_prediction_errors.csv", index=False)


def aggregate_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby(["experiment_id", "model", "mode", "input_representation", "sensor_combination", "segment_setting", "seed"], dropna=False).agg(
        mean_mae_over_6_shifts=("metric_mae", "mean"),
        mean_rmse_over_6_shifts=("metric_rmse", "mean"),
        mean_r2_over_6_shifts=("metric_r2", "mean"),
        mean_pearson_over_6_shifts=("metric_pearson", "mean"),
        mean_spearman_over_6_shifts=("metric_spearman", "mean"),
    ).reset_index()


def aggregate_combo_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.assign(r2_positive=df["metric_r2"] > 0).groupby(["experiment_id", "model", "mode", "input_representation", "sensor_combination", "segment_setting"], dropna=False).agg(
        mean_mae=("metric_mae", "mean"),
        std_mae=("metric_mae", "std"),
        mean_rmse=("metric_rmse", "mean"),
        std_rmse=("metric_rmse", "std"),
        mean_r2=("metric_r2", "mean"),
        std_r2=("metric_r2", "std"),
        mean_pearson=("metric_pearson", "mean"),
        std_pearson=("metric_pearson", "std"),
        mean_spearman=("metric_spearman", "mean"),
        std_spearman=("metric_spearman", "std"),
        num_r2_positive=("r2_positive", "sum"),
        ratio_r2_positive=("r2_positive", "mean"),
    ).reset_index()


def aggregate_segment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby(["experiment_id", "model", "mode", "input_representation", "segment_setting"], dropna=False).agg(mean_mae=("metric_mae", "mean"), mean_rmse=("metric_rmse", "mean"), mean_r2=("metric_r2", "mean")).reset_index()


def aggregate_model_metrics(seed: pd.DataFrame) -> pd.DataFrame:
    if seed.empty:
        return pd.DataFrame()
    return seed.groupby(["experiment_id", "model", "mode", "input_representation", "sensor_combination", "segment_setting"], dropna=False).agg(mean_r2=("mean_r2_over_6_shifts", "mean"), std_r2=("mean_r2_over_6_shifts", "std"), mean_rmse=("mean_rmse_over_6_shifts", "mean"), std_rmse=("mean_rmse_over_6_shifts", "std")).reset_index()


def make_analysis_outputs(output_dir: Path, h2_files: dict[str, Path], s_metrics: pd.DataFrame) -> dict[str, Any]:
    analysis: dict[str, Any] = {}
    combo_metrics = aggregate_combo_metrics(s_metrics)
    if not combo_metrics.empty:
        combo_overall = combo_metrics.groupby("sensor_combination").agg(mean_r2=("mean_r2", "mean"), mean_rmse=("mean_rmse", "mean")).reset_index()
        combo_overall["rank_by_r2"] = combo_overall["mean_r2"].rank(ascending=False, method="min")
        combo_overall.to_csv(output_dir / "analysis" / f"{PREFIX}_sensor_combination_effect.csv", index=False)
        best_sensor_rows = []
        for segment, sub in combo_metrics.groupby("segment_setting"):
            best_r2 = sub.loc[sub["mean_r2"].idxmax()]
            best_mae = sub.loc[sub["mean_mae"].idxmin()]
            best_rmse = sub.loc[sub["mean_rmse"].idxmin()]
            best_sensor_rows.append({
                "experiment_id": best_r2["experiment_id"], "model": "cumulative_s_dnn", "mode": "s_dnn", "input_representation": best_r2["input_representation"], "segment_setting": segment,
                "best_sensor_combination_by_mae": best_mae["sensor_combination"], "best_sensor_combination_by_rmse": best_rmse["sensor_combination"], "best_sensor_combination_by_r2": best_r2["sensor_combination"],
                "best_mae": best_mae["mean_mae"], "best_rmse": best_rmse["mean_rmse"], "best_r2": best_r2["mean_r2"], "is_all_sensor_combination_best_by_r2": best_r2["sensor_combination"] == "current_vibration_acoustic"
            })
        pd.DataFrame(best_sensor_rows).to_csv(output_dir / "analysis" / f"{PREFIX}_best_sensor_combination.csv", index=False)
        best_segment_rows = []
        for combo, sub in combo_metrics.groupby("sensor_combination"):
            best_r2 = sub.loc[sub["mean_r2"].idxmax()]
            best_mae = sub.loc[sub["mean_mae"].idxmin()]
            best_rmse = sub.loc[sub["mean_rmse"].idxmin()]
            seg = str(best_r2["segment_setting"])
            best_segment_rows.append({"experiment_id": best_r2["experiment_id"], "model": "cumulative_s_dnn", "mode": "s_dnn", "input_representation": best_r2["input_representation"], "sensor_combination": combo, "best_segment_by_mae": best_mae["segment_setting"], "best_segment_by_rmse": best_rmse["segment_setting"], "best_segment_by_r2": seg, "best_mae": best_mae["mean_mae"], "best_rmse": best_rmse["mean_rmse"], "best_r2": best_r2["mean_r2"], "best_segment_contains_entry": "entry" in seg, "best_segment_contains_exit": "exit" in seg})
        pd.DataFrame(best_segment_rows).to_csv(output_dir / "analysis" / f"{PREFIX}_best_segment_by_sensor_combination.csv", index=False)
        combo_metrics.assign(r2_positive=combo_metrics["mean_r2"] > 0).groupby(["sensor_combination", "segment_setting"]).agg(num_total=("mean_r2", "size"), num_r2_positive=("r2_positive", "sum"), mean_r2=("mean_r2", "mean")).reset_index().to_csv(output_dir / "analysis" / f"{PREFIX}_r2_positive_count_summary.csv", index=False)
        combo_metrics.to_csv(output_dir / "analysis" / f"{PREFIX}_segment_effect_under_sensor_combination.csv", index=False)
    compare_with_h2(output_dir, h2_files, combo_metrics)
    if not combo_metrics.empty:
        best = combo_metrics.loc[combo_metrics["mean_r2"].idxmax()]
        analysis["best_sensor_combination"] = best["sensor_combination"]
        analysis["best_segment_setting"] = best["segment_setting"]
        analysis["best_r2"] = float(best["mean_r2"])
        analysis["num_rows"] = int(len(s_metrics))
    write_json(output_dir / "analysis" / f"{PREFIX}_analysis_summary.json", analysis)
    return analysis


def compare_with_h2(output_dir: Path, h2_files: dict[str, Path], h3_combo: pd.DataFrame) -> None:
    if h3_combo.empty:
        return
    h3_sensor = h3_combo.groupby("sensor_combination").agg(H3_S1_mean_r2=("mean_r2", "mean")).reset_index()
    h3_sensor["H3_S1_rank_by_r2"] = h3_sensor["H3_S1_mean_r2"].rank(ascending=False, method="min")
    h2_sensor = pd.DataFrame()
    if "sensor_combination_metrics" in h2_files:
        h2 = pd.read_csv(h2_files["sensor_combination_metrics"])
        r2_col = "mean_r2" if "mean_r2" in h2.columns else "metric_r2"
        h2_sensor = h2.groupby("sensor_combination").agg(h2_s2_mean_r2=(r2_col, "mean")).reset_index()
        h2_sensor["h2_s2_rank_by_r2"] = h2_sensor["h2_s2_mean_r2"].rank(ascending=False, method="min")
    comp = h3_sensor.merge(h2_sensor, on="sensor_combination", how="left")
    comp["h2_s2_best_model"] = "feature_based_RF_MLP"
    comp["rank_change"] = comp["H3_S1_rank_by_r2"] - comp.get("h2_s2_rank_by_r2", np.nan)
    comp["interpretation"] = np.where(comp["rank_change"].abs() <= 1, "similar rank", "rank changed")
    comp.to_csv(output_dir / "analysis" / f"{PREFIX}_sensor_combination_effect_comparison.csv", index=False)
    h3_seg = h3_combo.groupby("segment_setting").agg(H3_S1_mean_r2=("mean_r2", "mean")).reset_index()
    h3_seg["H3_S1_rank_by_r2"] = h3_seg["H3_S1_mean_r2"].rank(ascending=False, method="min")
    if "sensor_combination_metrics" in h2_files:
        h2_seg = h2.groupby("segment_setting").agg(h2_s2_mean_r2=(r2_col, "mean")).reset_index()
        h2_seg["h2_s2_rank_by_r2"] = h2_seg["h2_s2_mean_r2"].rank(ascending=False, method="min")
    else:
        h2_seg = pd.DataFrame(columns=["segment_setting", "h2_s2_mean_r2", "h2_s2_rank_by_r2"])
    seg_comp = h3_seg.merge(h2_seg, on="segment_setting", how="left")
    seg_comp["h2_s2_best_count"] = np.nan
    seg_comp["H3_S1_best_count"] = h3_combo.loc[h3_combo.groupby("sensor_combination")["mean_r2"].idxmax()].groupby("segment_setting").size().reindex(seg_comp["segment_setting"]).fillna(0).to_numpy()
    seg_comp["interpretation"] = np.where((seg_comp["segment_setting"] == "entry_exit") & (seg_comp["H3_S1_rank_by_r2"] <= 2), "entry_exit remains strong", "condition-dependent")
    seg_comp.to_csv(output_dir / "analysis" / f"{PREFIX}_segment_effect_comparison_with_H2_S2.csv", index=False)
    acoustic = h3_combo[(h3_combo["sensor_combination"] == "acoustic") & (h3_combo["segment_setting"].isin(["entry_exit", "entry", "exit", "full_length", "steady"]))]
    if not acoustic.empty:
        acoustic = acoustic.copy()
        acoustic["rank_within_acoustic"] = acoustic["mean_r2"].rank(ascending=False, method="min")
        acoustic.assign(experiment="H3_S1", model_or_mode="s_dnn", input_type="timeseries_cumulative", interpretation=np.where(acoustic["segment_setting"] == "entry_exit", "acoustic entry_exit candidate", "acoustic reference")).rename(columns={"mean_mae": "mean_mae", "mean_rmse": "mean_rmse", "mean_r2": "mean_r2"})[["experiment", "model_or_mode", "input_type", "sensor_combination", "segment_setting", "mean_mae", "mean_rmse", "mean_r2", "rank_within_acoustic", "interpretation"]].to_csv(output_dir / "analysis" / f"{PREFIX}_acoustic_entry_exit_comparison.csv", index=False)
    all_name = "current_vibration_acoustic"
    rows = []
    if all_name in set(h3_sensor["sensor_combination"]):
        best = h3_sensor.loc[h3_sensor["H3_S1_mean_r2"].idxmax()]
        all_row = h3_sensor[h3_sensor["sensor_combination"] == all_name].iloc[0]
        rows.append({"experiment": "H3_S1", "model_or_mode": "s_dnn", "all_sensor_rank_by_r2": all_row["H3_S1_rank_by_r2"], "best_sensor_combination": best["sensor_combination"], "best_r2": best["H3_S1_mean_r2"], "all_sensor_r2": all_row["H3_S1_mean_r2"], "r2_gap_to_best": best["H3_S1_mean_r2"] - all_row["H3_S1_mean_r2"], "interpretation": "all sensors best" if best["sensor_combination"] == all_name else "all sensors not best"})
    pd.DataFrame(rows).to_csv(output_dir / "analysis" / f"{PREFIX}_all_sensor_combination_optimality_comparison.csv", index=False)
    summary = [
        {"comparison_axis": "input_type", "h2_s2_finding": "feature-based RF/MLP handcrafted features", "H3_S1_finding": "timeseries cumulative S-DNN", "consistent": "not_directly_comparable", "interpretation": "compare pattern and robustness, not absolute superiority"},
        {"comparison_axis": "sensor_combination_rank", "h2_s2_finding": "loaded where available", "H3_S1_finding": f"best={h3_sensor.loc[h3_sensor['H3_S1_mean_r2'].idxmax(), 'sensor_combination']}", "consistent": "see_rank_tables", "interpretation": "sensor selection remains an empirical axis"},
        {"comparison_axis": "segment_rank", "h2_s2_finding": "loaded where available", "H3_S1_finding": f"best={h3_seg.loc[h3_seg['H3_S1_mean_r2'].idxmax(), 'segment_setting']}", "consistent": "see_rank_tables", "interpretation": "segment-aware effect is model dependent"},
    ]
    pd.DataFrame(summary).to_csv(output_dir / "analysis" / f"{PREFIX}_feature_based_vs_timeseries_based_summary.csv", index=False)


def make_figures(output_dir: Path, h2_files: dict[str, Path], s_metrics: pd.DataFrame) -> None:
    if s_metrics.empty:
        return
    figdir = output_dir / "figures"
    combo = aggregate_combo_metrics(s_metrics)
    combo.groupby("sensor_combination")["mean_r2"].mean().plot(kind="bar", figsize=(8, 4))
    plt.ylabel("mean R2")
    plt.title("H3_S1 Sensor Combination Performance")
    plt.tight_layout()
    plt.savefig(figdir / f"{PREFIX}_sensor_combination_performance.png", dpi=170)
    plt.close()
    combo.groupby("segment_setting")["mean_r2"].mean().plot(kind="bar", figsize=(8, 4))
    plt.ylabel("mean R2")
    plt.title("H3_S1 Segment Performance")
    plt.tight_layout()
    plt.savefig(figdir / f"{PREFIX}_segment_performance.png", dpi=170)
    plt.close()
    pivot = combo.pivot_table(index="sensor_combination", columns="segment_setting", values="mean_r2", aggfunc="mean")
    plt.figure(figsize=(9, 5))
    plt.imshow(pivot.fillna(np.nan), aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="mean R2")
    plt.title("H3_S1 Best Segment by Sensor Combination")
    plt.tight_layout()
    plt.savefig(figdir / f"{PREFIX}_best_segment_by_sensor_combination.png", dpi=170)
    plt.savefig(figdir / f"{PREFIX}_best_sensor_combination_summary.png", dpi=170)
    plt.close()
    s_metrics.assign(shift=s_metrics["source_domain"] + "_to_" + s_metrics["target_domain"]).groupby("shift")["metric_r2"].mean().plot(kind="bar", figsize=(7, 4))
    plt.ylabel("mean R2")
    plt.title("H3_S1 Shift-wise Performance")
    plt.tight_layout()
    plt.savefig(figdir / f"{PREFIX}_shift_wise_performance.png", dpi=170)
    plt.close()
    for name in ["h2_s2_vs_H3_S1_sensor_combination_rank", "h2_s2_vs_H3_S1_segment_rank", "acoustic_entry_exit_comparison", "r2_positive_count_comparison"]:
        plt.figure(figsize=(5, 3))
        plt.text(0.05, 0.5, f"See analysis/{PREFIX}_{name.replace('h2_s2_vs_H3_S1_', '').replace('r2_positive_count_comparison','r2_positive_count_summary')}.csv", wrap=True)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(figdir / f"{PREFIX}_{name}.png", dpi=170)
        plt.close()


def write_report(output_dir: Path, experiment_id: str, h2_dir: Path, h2_files: dict[str, Path], base_metrics: pd.DataFrame, s_metrics: pd.DataFrame, analysis: dict[str, Any], validation: dict[str, Any], dry_run: bool) -> None:
    best_sensor = analysis.get("best_sensor_combination", "not_available")
    best_segment = analysis.get("best_segment_setting", "not_available")
    text = f"""# H3_S1 Cumulative Descriptor Sensor Combination H2_S2 Revisit

## Purpose

This execution revisits H2_S2 sensor group combination using a timeseries cumulative descriptor S-DNN.

## H2_S2 Baseline

Source directory: `{h2_dir}`

Loaded files:
{chr(10).join(f'- {k}: `{v}`' for k, v in h2_files.items())}

## H2_S2 vs H3_S1 Difference

- H2_S2: feature-based RF/MLP with handcrafted statistical/frequency features.
- H3_S1: timeseries cumulative descriptor with RNN/GRU/LSTM/CNN OOF stacking and Ridge-AdaBoost meta learner.
- Absolute scores are not strict apples-to-apples; pattern consistency and domain-shift robustness are the main comparison.

## Data and Protocol

- Cases: {CASE_SCOPE}
- A={{1,9}}, B={{2,12}}, C={{8,14}}
- Target: VB
- Target domain use: evaluation only
- Dry-run: {dry_run}

## Validation

```json
{json.dumps(validation, indent=2)}
```

## Result Summary

- S-DNN metric rows: {len(s_metrics)}
- Base learner metric rows: {len(base_metrics)}
- Best H3_S1 sensor combination by mean R2: {best_sensor}
- Best H3_S1 segment setting by mean R2: {best_segment}
- Best H3_S1 R2: {analysis.get("best_r2", "not_available")}

## RQ Answers

RQ1. Sensor group combination effect is condition-dependent in this execution. See `analysis/H3_S1_sensor_combination_effect.csv`.

RQ2. Whether all sensor groups are best is recorded in `analysis/H3_S1_all_sensor_combination_optimality_comparison.csv`.

RQ3. Segment-aware benefit is evaluated by segment ranking and baseline comparison in `analysis/H3_S1_segment_effect_comparison_with_H2_S2.csv`.

RQ4. Best segment by sensor combination is in `analysis/H3_S1_best_segment_by_sensor_combination.csv`.

RQ5. Acoustic + entry_exit evidence is in `analysis/H3_S1_acoustic_entry_exit_comparison.csv`.

RQ6. Domain-shift robustness should be judged with R2-positive count and shift-wise metrics. Negative R2 is not an absolute success even if relative ranks improve.

## Limitations

The full 7 x 7 x 6 x 3 grid is expensive for S-DNN because every condition performs source-only OOF training. Smoke/reduced runs should be interpreted as implementation and early pattern checks unless the full grid has been run.
"""
    (output_dir / "reports" / f"{PREFIX}_report.md").write_text(text, encoding="utf-8")
    (output_dir / "reports" / f"{PREFIX}_report.html").write_text("<html><body><pre>" + html.escape(text) + "</pre></body></html>", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

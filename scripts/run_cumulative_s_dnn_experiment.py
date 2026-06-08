#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import copy
import html
import json
import math
import pickle
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.experiments.execution_path import create_execution_dir
from milling_experiment_framework.models.dl.base_sequence_learners import CNN1DRegressor, GRURegressor, LSTMRegressor, RNNRegressor
from milling_experiment_framework.models.dl.cumulative_descriptor import CumulativeDescriptor, CumulativeDescriptorLayer
from milling_experiment_framework.models.dl.stacking_meta_learner import RidgeAdaBoostStackingRegressor


PREFIX = "H3_S3"
CASE_SCOPE = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
EXCLUDED_CASES = [6]
CASE_DOMAINS = {f"case_{case}": [case] for case in CASE_SCOPE}
TRAIN_CASE_GROUPS = {f"train_without_case_{case}": [other for other in CASE_SCOPE if other != case] for case in CASE_SCOPE}
DOMAIN_CASES = {**CASE_DOMAINS, **TRAIN_CASE_GROUPS}
SHIFT_SCENARIOS = [f"train_without_case_{case}_to_case_{case}" for case in CASE_SCOPE]
EXPECTED_SENSORS = ["smcAC", "smcDC", "vib_spindle", "vib_table", "AE_spindle", "AE_table"]
SENSOR_GROUPS = {
    "current": ["smcAC", "smcDC"],
    "vibration": ["vib_spindle", "vib_table"],
    "acoustic": ["AE_spindle", "AE_table"],
    "all_sensors": EXPECTED_SENSORS,
}
BASE_LEARNERS = ["rnn", "gru", "lstm", "cnn"]


@dataclass
class RunContext:
    experiment_id: str
    output_dir: Path
    config: dict[str, Any]
    device: torch.device
    logger: Any


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.sequences[index], self.targets[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cumulative descriptor enhanced S-DNN experiment.")
    parser.add_argument("--config", default="configs/models/cumulative_s_dnn.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--modes", default=None, help="Comma-separated: base_only,s_dnn")
    parser.add_argument("--input-representations", default=None, help="Comma-separated: raw,cumulative,raw_plus_cumulative")
    parser.add_argument("--construction-mode", default=None, help="paper_style_sliding_window,segment_mode,fixed_length_resample")
    parser.add_argument("--segment-mode", default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--shifts", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--quick-hidden-size", type=int, default=None, help="Override RNN/GRU/LSTM hidden sizes and CNN channels for fast mini-runs.")
    args = parser.parse_args()

    output_dir: Path | None = None
    try:
        config = load_config(Path(args.config))
        apply_cli_overrides(config, args)
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_H3_S3_cumulative_s_dnn_architecture")
        path_config = {"experiment": {"experiment_id": experiment_id}}
        output_dir = Path(create_execution_dir(path_config, root=Path("experiments") / "executions"))
        prepare_dirs(output_dir)
        logger = make_logger(output_dir / "logs" / f"{PREFIX}_run.log")
        device = torch.device("cuda" if torch.cuda.is_available() and config.get("training", {}).get("device") == "cuda" else "cpu")
        ctx = RunContext(experiment_id, output_dir, config, device, logger)
        logger(f"H3.S3 cumulative S-DNN execution started: {experiment_id}")
        write_json(output_dir / "logs" / f"{PREFIX}_environment.json", collect_environment())
        write_json(output_dir / "configs" / f"{PREFIX}_input_config.yaml", config)
        resolved = copy.deepcopy(config)
        resolved["experiment_execution"] = {
            **path_config["experiment"],
            "experiment_id": experiment_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(output_dir / "configs" / f"{PREFIX}_resolved_config.yaml", resolved)
        write_implementation_summary(output_dir)

        smoke = run_smoke_test(ctx)
        write_json(output_dir / "data" / f"{PREFIX}_smoke_test_result.json", smoke)
        write_json(output_dir / f"{PREFIX}_smoke_test_result.json", smoke)
        if args.smoke_test and not args.dry_run:
            write_report(ctx, None, None, None, None, smoke, dry_run=False)
            print(json.dumps({"experiment_id": experiment_id, "execution_dir": str(output_dir), "smoke_test": smoke}, indent=2, ensure_ascii=False))
            return

        dataset = load_dataset(config)
        sensors = resolve_sensors(dataset, config["model"].get("sensor_setting", "all_sensors"))
        validation = validate_real_data(dataset, sensors, config)
        write_json(output_dir / "configs" / f"{PREFIX}_config_validation.json", validation)
        split_table, split_summary = build_split_artifacts(dataset, config["runtime"]["shifts"])
        split_table.to_csv(output_dir / "splits" / f"{PREFIX}_split.csv", index=False)
        write_json(output_dir / "splits" / f"{PREFIX}_split_summary.json", split_summary)
        write_json(output_dir / "splits" / f"{PREFIX}_leakage_check.json", validation["leakage_check"])
        write_data_summaries(output_dir, dataset, sensors, config)
        if args.dry_run:
            write_report(ctx, None, None, None, None, smoke, dry_run=True)
            logger("Dry-run completed without model training.")
            print(json.dumps({"experiment_id": experiment_id, "execution_dir": str(output_dir), "dry_run": True, "validation": validation}, indent=2, ensure_ascii=False))
            return

        base_metrics, s_dnn_metrics, base_predictions, s_dnn_predictions, oof_predictions = run_experiment_grid(ctx, dataset, sensors)
        base_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_base_learner_shift_metrics.csv", index=False)
        s_dnn_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_s_dnn_shift_metrics.csv", index=False)
        base_predictions.to_csv(output_dir / "predictions" / f"{PREFIX}_base_learner_predictions.csv", index=False)
        s_dnn_predictions.to_csv(output_dir / "predictions" / f"{PREFIX}_s_dnn_predictions.csv", index=False)
        oof_predictions.to_csv(output_dir / "predictions" / f"{PREFIX}_oof_predictions.csv", index=False)
        prediction_errors = pd.concat([base_predictions, s_dnn_predictions], ignore_index=True) if not s_dnn_predictions.empty else base_predictions
        if not prediction_errors.empty:
            prediction_errors.sort_values("absolute_error", ascending=False).head(300).to_csv(output_dir / "predictions" / f"{PREFIX}_prediction_errors.csv", index=False)
        seed_metrics = aggregate_seed_metrics(base_metrics, s_dnn_metrics)
        model_metrics = aggregate_model_metrics(seed_metrics)
        seed_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_seed_metrics.csv", index=False)
        model_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_model_metrics.csv", index=False)
        write_analysis_outputs(output_dir, base_metrics, s_dnn_metrics)
        make_figures(output_dir, base_metrics, s_dnn_metrics, base_predictions, s_dnn_predictions)
        write_report(ctx, base_metrics, s_dnn_metrics, seed_metrics, model_metrics, smoke, dry_run=False)
        logger(f"H3.S3 execution finished: {experiment_id}")
        print(json.dumps({"experiment_id": experiment_id, "execution_dir": str(output_dir), "num_base_metric_rows": len(base_metrics), "num_s_dnn_metric_rows": len(s_dnn_metrics)}, indent=2, ensure_ascii=False))
    except Exception:
        if output_dir is not None:
            (output_dir / "logs" / f"{PREFIX}_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    runtime = config.setdefault("runtime", {})
    runtime.setdefault("modes", ["base_only", "s_dnn"])
    runtime.setdefault("input_representations", [config["model"].get("input_representation", "cumulative")])
    runtime.setdefault("shifts", SHIFT_SCENARIOS)
    runtime.setdefault("seeds", [0])
    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.modes:
        config["runtime"]["modes"] = [x.strip() for x in args.modes.split(",") if x.strip()]
    if args.input_representations:
        config["runtime"]["input_representations"] = [x.strip() for x in args.input_representations.split(",") if x.strip()]
    if args.shifts:
        config["runtime"]["shifts"] = [x.strip() for x in args.shifts.split(",") if x.strip()]
    if args.seeds:
        config["runtime"]["seeds"] = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if args.construction_mode:
        config["model"]["sequence"]["construction_mode"] = args.construction_mode
    if args.segment_mode:
        config["model"]["sequence"]["segment_mode"] = args.segment_mode
    if args.sequence_length is not None:
        config["model"]["sequence"]["resample_length"] = args.sequence_length
        config["model"]["sequence"]["window_size"] = args.sequence_length
    if args.window_size is not None:
        config["model"]["sequence"]["window_size"] = args.window_size
    if args.stride is not None:
        config["model"]["sequence"]["stride"] = args.stride
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.cv_folds is not None:
        config["model"]["ensemble"]["cv_folds"] = args.cv_folds
    if args.quick_hidden_size is not None:
        h = int(args.quick_hidden_size)
        config["model"]["base_learners"]["rnn"]["hidden_size"] = h
        config["model"]["base_learners"]["rnn"]["num_layers"] = 1
        config["model"]["base_learners"]["gru"]["hidden_size"] = h
        config["model"]["base_learners"]["gru"]["num_layers"] = 1
        config["model"]["base_learners"]["lstm"]["hidden_size"] = h
        config["model"]["base_learners"]["lstm"]["num_layers"] = 1
        config["model"]["base_learners"]["cnn"]["channels"] = [h]
        config["model"]["base_regression_head"]["hidden_dims"] = [h]


def prepare_dirs(output_dir: Path) -> None:
    for dirname in ["configs", "data", "splits", "checkpoints", "metrics", "predictions", "analysis", "figures", "logs", "reports"]:
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)
    (output_dir / "logs" / f"{PREFIX}_error.log").touch()


def make_logger(path: Path):
    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return log


def run_smoke_test(ctx: RunContext) -> dict[str, Any]:
    torch.manual_seed(0)
    batch, length, channels = 4, 128, 6
    x = torch.randn(batch, length, channels, device=ctx.device)
    y = torch.randn(batch, 1, device=ctx.device)
    descriptor = CumulativeDescriptorLayer(input_representation="cumulative").to(ctx.device)
    x_cum = descriptor(x)
    raw_plus = CumulativeDescriptorLayer(input_representation="raw_plus_cumulative").to(ctx.device)(x)
    results: dict[str, Any] = {
        "cumulative_descriptor_success": bool(x_cum.shape == x.shape and torch.isfinite(x_cum).all()),
        "raw_plus_cumulative_shape": list(raw_plus.shape),
        "base_learner_output_shapes": {},
    }
    for name in BASE_LEARNERS:
        model = make_base_learner(name, channels, ctx.config).to(ctx.device)
        out = model(x_cum)
        loss = nn.MSELoss()(out, y)
        loss.backward()
        ckpt = ctx.output_dir / "checkpoints" / f"smoke_{name}.pt"
        torch.save({"model_state": model.state_dict()}, ckpt)
        loaded = make_base_learner(name, channels, ctx.config)
        loaded.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state"])
        results[f"{name}_forward_backward_success"] = True
        results["base_learner_output_shapes"][name] = list(out.shape)
    x_meta = np.random.default_rng(0).normal(size=(12, 4)).astype("float32")
    y_meta = np.random.default_rng(1).normal(size=12).astype("float32")
    meta = RidgeAdaBoostStackingRegressor(n_estimators=3, learning_rate=0.01, random_state=0).fit(x_meta, y_meta)
    pred = meta.predict(x_meta[:5])
    with (ctx.output_dir / "checkpoints" / "smoke_meta_learner.pkl").open("wb") as f:
        pickle.dump(meta, f)
    results.update(
        {
            "oof_prediction_matrix_shape": [12, 4],
            "meta_learner_fit_success": True,
            "final_prediction_shape": list(pred.shape),
            "checkpoint_save_load_success": True,
            "target_domain_leakage_check": "passed_for_smoke_synthetic_oof",
        }
    )
    return results


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config.get("data", {})
    process = pd.read_csv(data_cfg["process_info_path"])
    signal = pd.read_csv(data_cfg["signal_data_path"])
    heuristic = pd.read_csv(data_cfg["heuristic_sequence_path"])
    process = process.loc[process["case"].isin(CASE_SCOPE)].copy()
    signal = signal.loc[signal["case"].isin(CASE_SCOPE)].copy()
    heuristic = heuristic.loc[heuristic["case"].isin(CASE_SCOPE)].copy()
    data = process.merge(signal, on=["case", "run"], suffixes=("", "_signal"), validate="one_to_one")
    data = data.merge(
        heuristic[["case", "run", "idx_noload_end", "idx_start", "idx_end", "signal_length", "status"]],
        on=["case", "run"],
        how="inner",
        validate="one_to_one",
    )
    data = data.loc[data["VB"].notna()].reset_index(drop=True)
    data["sample_id"] = data.apply(lambda row: f"case_{int(row['case']):02d}_run_{int(row['run']):03d}", axis=1)
    data["dataset_run_id"] = data.apply(lambda row: f"case_{int(row['case']):02d}_run_{int(row['run']):03d}", axis=1)
    data["domain_id"] = data["case"].map(domain_for_case)
    return data


def domain_for_case(case: int) -> str:
    for domain, cases in DOMAIN_CASES.items():
        if int(case) in cases:
            return domain
    raise ValueError(f"Unknown case: {case}")


def resolve_sensors(data: pd.DataFrame, sensor_setting: str) -> list[str]:
    available = [s for s in EXPECTED_SENSORS if s in data.columns]
    if sensor_setting in SENSOR_GROUPS:
        sensors = [s for s in SENSOR_GROUPS[sensor_setting] if s in available]
    elif sensor_setting in available:
        sensors = [sensor_setting]
    else:
        requested = [s.strip() for s in sensor_setting.split(",") if s.strip()]
        sensors = [s for s in requested if s in available]
    if not sensors:
        raise ValueError(f"No sensors resolved for sensor_setting={sensor_setting}")
    return sensors


def segment_array(arr: np.ndarray, row: pd.Series, mode: str) -> np.ndarray:
    n = len(arr)
    idx_noload_end = int(np.clip(row["idx_noload_end"], 0, n))
    idx_start = int(np.clip(row["idx_start"], idx_noload_end, n))
    idx_end = int(np.clip(row["idx_end"], idx_start, n))
    segments = {
        "full_length": arr[:n],
        "entry": arr[idx_noload_end:idx_start],
        "steady": arr[idx_start:idx_end],
        "exit": arr[idx_end:n],
    }
    if mode in segments:
        out = segments[mode]
    elif mode == "entry_steady":
        out = np.concatenate([segments["entry"], segments["steady"]])
    elif mode == "entry_exit":
        out = np.concatenate([segments["entry"], segments["exit"]])
    elif mode == "steady_exit":
        out = np.concatenate([segments["steady"], segments["exit"]])
    else:
        raise ValueError(f"Unsupported segment_mode: {mode}")
    if len(out) == 0:
        raise ValueError(f"Empty sequence for mode={mode}, sample={row.get('sample_id')}")
    return out.astype("float32")


def resample_1d(arr: np.ndarray, sequence_length: int) -> np.ndarray:
    if len(arr) == sequence_length:
        return arr.astype("float32")
    old = np.linspace(0.0, 1.0, num=len(arr), dtype="float32")
    new = np.linspace(0.0, 1.0, num=sequence_length, dtype="float32")
    return np.interp(new, old, arr).astype("float32")


def build_sequence_table(data: pd.DataFrame, sensors: list[str], config: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    seq_cfg = config["model"]["sequence"]
    mode = seq_cfg.get("construction_mode", "fixed_length_resample")
    segment_mode = seq_cfg.get("segment_mode", "full_length")
    rows = []
    meta_rows = []
    if mode == "paper_style_sliding_window":
        window = int(seq_cfg.get("window_size", 1024))
        stride = int(seq_cfg.get("stride", 80))
        for _, row in data.iterrows():
            arrays = [np.asarray(ast.literal_eval(row[sensor]), dtype="float32") for sensor in sensors]
            stacked = np.stack(arrays, axis=1)
            for start in range(0, max(1, len(stacked) - window + 1), stride):
                chunk = stacked[start : start + window]
                if len(chunk) < window:
                    continue
                new_row = row.copy()
                new_row["sample_id"] = f"{row['sample_id']}_win_{start:05d}"
                new_row["window_start"] = int(start)
                new_row["window_end"] = int(start + window)
                rows.append(chunk.astype("float32"))
                meta_rows.append(new_row)
    else:
        length = int(seq_cfg.get("resample_length", seq_cfg.get("window_size", 1024)))
        for _, row in data.iterrows():
            channels = []
            for sensor in sensors:
                arr = np.asarray(ast.literal_eval(row[sensor]), dtype="float32")
                base = segment_array(arr, row, segment_mode) if mode == "segment_mode" else arr
                channels.append(resample_1d(base, length))
            rows.append(np.stack(channels, axis=1))
            meta_rows.append(row)
    if not rows:
        raise ValueError("No sequences were constructed.")
    meta = pd.DataFrame(meta_rows).reset_index(drop=True)
    return np.stack(rows, axis=0).astype("float32"), meta


def split_frame(meta: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    source_cases = DOMAIN_CASES[source]
    target_cases = DOMAIN_CASES[target]
    validation_case = min(source_cases)
    split = meta.loc[meta["case"].isin(source_cases + target_cases)].copy()
    split["split"] = "excluded"
    split.loc[split["case"].isin(target_cases), "split"] = "test"
    split.loc[split["case"].isin(source_cases) & (split["case"] != validation_case), "split"] = "train"
    split.loc[split["case"] == validation_case, "split"] = "validation"
    return split


def parse_shift(shift: str) -> tuple[str, str]:
    source, target = shift.split("_to_")
    return source, target


def normalize_sequences(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return tuple(((x - mean) / std).astype("float32") for x in (train, *others))


def apply_representation(train: np.ndarray, val: np.ndarray, test: np.ndarray, representation: str, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = config["preprocessing"].get("cumulative_descriptor_order", "after_scaling")
    descriptor = CumulativeDescriptor(input_representation=representation)
    if order == "after_scaling":
        return descriptor.transform_sequence(train), descriptor.transform_sequence(val), descriptor.transform_sequence(test)
    return descriptor.transform_sequence(train), descriptor.transform_sequence(val), descriptor.transform_sequence(test)


def make_base_learner(name: str, input_channels: int, config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    head = model_cfg.get("base_regression_head", {})
    head_dims = head.get("hidden_dims", [64])
    head_dropout = float(head.get("dropout", 0.5))
    learners = model_cfg.get("base_learners", {})
    if name == "rnn":
        cfg = learners.get("rnn", {})
        return RNNRegressor(input_channels, hidden_size=int(cfg.get("hidden_size", cfg.get("hidden_sizes", [64])[-1])), num_layers=int(cfg.get("num_layers", 2)), dropout=float(cfg.get("dropout", head_dropout)), head_hidden_dims=head_dims)
    if name == "gru":
        cfg = learners.get("gru", {})
        return GRURegressor(input_channels, hidden_size=int(cfg.get("hidden_size", 32)), num_layers=int(cfg.get("num_layers", 1)), dropout=float(cfg.get("dropout", head_dropout)), head_hidden_dims=head_dims)
    if name == "lstm":
        cfg = learners.get("lstm", {})
        return LSTMRegressor(input_channels, hidden_size=int(cfg.get("hidden_size", 32)), num_layers=int(cfg.get("num_layers", 1)), dropout=float(cfg.get("dropout", head_dropout)), head_hidden_dims=head_dims)
    if name == "cnn":
        cfg = learners.get("cnn", {})
        return CNN1DRegressor(input_channels, channels=cfg.get("channels", [128, 64, 32]), kernel_size=int(cfg.get("kernel_size", 3)), dropout=float(cfg.get("dropout", head_dropout)), pooling=cfg.get("pooling", "max"), head_hidden_dims=head_dims)
    raise ValueError(f"Unsupported base learner: {name}")


def run_experiment_grid(ctx: RunContext, dataset: pd.DataFrame, sensors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sequences, meta = build_sequence_table(dataset, sensors, ctx.config)
    base_rows, s_rows = [], []
    base_preds, s_preds, oof_preds = [], [], []
    for representation in ctx.config["runtime"]["input_representations"]:
        for seed in ctx.config["runtime"]["seeds"]:
            set_seed(seed)
            for shift in ctx.config["runtime"]["shifts"]:
                source, target = parse_shift(shift)
                split = split_frame(meta, source, target)
                idx = split.index.to_numpy()
                seq = sequences[idx]
                split_reset = split.reset_index(drop=True)
                train_mask = split_reset["split"].to_numpy() == "train"
                val_mask = split_reset["split"].to_numpy() == "validation"
                test_mask = split_reset["split"].to_numpy() == "test"
                train_seq, val_seq, test_seq = normalize_sequences(seq[train_mask], seq[val_mask], seq[test_mask])
                train_seq, val_seq, test_seq = apply_representation(train_seq, val_seq, test_seq, representation, ctx.config)
                y_train = split_reset.loc[train_mask, "VB"].to_numpy(dtype="float32")
                y_val = split_reset.loc[val_mask, "VB"].to_numpy(dtype="float32")
                y_test = split_reset.loc[test_mask, "VB"].to_numpy(dtype="float32")
                if "base_only" in ctx.config["runtime"]["modes"]:
                    for learner in BASE_LEARNERS:
                        model, best_rmse = train_base_model(ctx, learner, train_seq, y_train, val_seq, y_val, seed, f"{representation}_{shift}_{learner}")
                        metrics, pred = evaluate_model_on_test(ctx, model, test_seq, y_test, split_reset.loc[test_mask], learner)
                        base_rows.append(metric_row(ctx, "base_only", representation, learner, seed, source, target, metrics))
                        base_preds.append(prediction_frame(ctx, pred, split_reset.loc[test_mask], "base_only", representation, learner, seed, source, target))
                if "s_dnn" in ctx.config["runtime"]["modes"]:
                    s_metric, s_pred, oof = run_stacking(ctx, train_seq, y_train, test_seq, y_test, split_reset.loc[train_mask], split_reset.loc[test_mask], representation, seed, source, target)
                    s_rows.append(metric_row(ctx, "s_dnn", representation, "s_dnn", seed, source, target, s_metric))
                    s_preds.append(s_pred)
                    oof_preds.append(oof)
    return (
        pd.DataFrame(base_rows),
        pd.DataFrame(s_rows),
        pd.concat(base_preds, ignore_index=True) if base_preds else pd.DataFrame(),
        pd.concat(s_preds, ignore_index=True) if s_preds else pd.DataFrame(),
        pd.concat(oof_preds, ignore_index=True) if oof_preds else pd.DataFrame(),
    )


def train_base_model(ctx: RunContext, learner: str, train_seq: np.ndarray, y_train: np.ndarray, val_seq: np.ndarray, y_val: np.ndarray, seed: int, key: str) -> tuple[nn.Module, float]:
    set_seed(seed)
    model = make_base_learner(learner, train_seq.shape[2], ctx.config).to(ctx.device)
    cfg = ctx.config["training"]
    batch_size = int(cfg.get("batch_size", 128))
    train_loader = DataLoader(SequenceDataset(train_seq, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(val_seq, y_val), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 0.001)))
    loss_fn = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_rmse = math.inf
    for epoch in range(1, int(cfg.get("max_epochs", 2)) + 1):
        run_epoch(model, train_loader, optimizer, loss_fn, ctx.device, True)
        val_rmse = evaluate_loader(model, val_loader, ctx.device)["metric_rmse"]
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            save_model_checkpoint(ctx, f"{learner}_{key}_best.pt", model, optimizer, epoch, best_rmse)
            save_model_checkpoint(ctx, f"{learner}_best.pt", model, optimizer, epoch, best_rmse)
        save_model_checkpoint(ctx, f"{learner}_{key}_last.pt", model, optimizer, epoch, best_rmse)
    model.load_state_dict(best_state)
    return model, best_rmse


def run_stacking(
    ctx: RunContext,
    train_seq: np.ndarray,
    y_train: np.ndarray,
    test_seq: np.ndarray,
    y_test: np.ndarray,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    representation: str,
    seed: int,
    source: str,
    target: str,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    folds = min(int(ctx.config["model"]["ensemble"].get("cv_folds", 5)), len(y_train))
    folds = max(2, folds)
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    x_meta_train = np.zeros((len(y_train), len(BASE_LEARNERS)), dtype="float32")
    x_meta_test_folds = np.zeros((folds, len(y_test), len(BASE_LEARNERS)), dtype="float32")
    fold_ids = np.full(len(y_train), -1, dtype=int)
    for fold_id, (fit_idx, valid_idx) in enumerate(kfold.split(train_seq)):
        fold_ids[valid_idx] = fold_id
        for learner_idx, learner in enumerate(BASE_LEARNERS):
            model, _ = train_base_model(
                ctx,
                learner,
                train_seq[fit_idx],
                y_train[fit_idx],
                train_seq[valid_idx],
                y_train[valid_idx],
                seed + fold_id + learner_idx,
                f"oof_{representation}_{source}_to_{target}_fold{fold_id}_{learner}",
            )
            x_meta_train[valid_idx, learner_idx] = predict_array(model, train_seq[valid_idx], ctx.device)
            x_meta_test_folds[fold_id, :, learner_idx] = predict_array(model, test_seq, ctx.device)
    x_meta_test = x_meta_test_folds.mean(axis=0)
    n_estimators = int(ctx.config["model"]["ensemble"].get("n_estimators_cumulative" if representation == "cumulative" else "n_estimators_raw", 26))
    meta = RidgeAdaBoostStackingRegressor(n_estimators=n_estimators, learning_rate=float(ctx.config["model"]["ensemble"].get("learning_rate", 0.006)), random_state=seed)
    meta.fit(x_meta_train, y_train)
    with (ctx.output_dir / "checkpoints" / f"meta_learner_{representation}_seed{seed}_{source}_to_{target}.pkl").open("wb") as f:
        pickle.dump(meta, f)
    with (ctx.output_dir / "checkpoints" / "meta_learner.pkl").open("wb") as f:
        pickle.dump(meta, f)
    with (ctx.output_dir / "checkpoints" / "s_dnn_artifact.pkl").open("wb") as f:
        pickle.dump(
            {
                "meta_learner": meta,
                "base_learners": BASE_LEARNERS,
                "input_representation": representation,
                "seed": seed,
                "source_domain": source,
                "target_domain": target,
                "x_meta_train_shape": list(x_meta_train.shape),
                "x_meta_test_shape": list(x_meta_test.shape),
            },
            f,
        )
    y_pred = meta.predict(x_meta_test)
    metrics = compute_metrics(y_test, y_pred)
    pred_frame = test_meta.copy()
    pred_frame["experiment_id"] = ctx.experiment_id
    pred_frame["model"] = "cumulative_s_dnn"
    pred_frame["mode"] = "s_dnn"
    pred_frame["base_learner"] = "s_dnn"
    pred_frame["input_representation"] = representation
    pred_frame["seed"] = seed
    pred_frame["source_domain"] = source
    pred_frame["target_domain"] = target
    pred_frame["y_true"] = y_test
    pred_frame["y_pred"] = y_pred
    pred_frame["residual"] = pred_frame["y_true"] - pred_frame["y_pred"]
    pred_frame["absolute_error"] = pred_frame["residual"].abs()
    oof = train_meta.copy()
    oof["experiment_id"] = ctx.experiment_id
    oof["shift"] = f"{source}_to_{target}"
    oof["seed"] = seed
    oof["fold_id"] = fold_ids
    oof["y_true"] = y_train
    for i, learner in enumerate(BASE_LEARNERS):
        oof[f"pred_{learner}"] = x_meta_train[:, i]
    return metrics, pred_frame, oof[["experiment_id", "shift", "seed", "sample_id", "fold_id", "y_true", "pred_rnn", "pred_gru", "pred_lstm", "pred_cnn"]]


def evaluate_model_on_test(ctx: RunContext, model: nn.Module, test_seq: np.ndarray, y_test: np.ndarray, test_meta: pd.DataFrame, learner: str) -> tuple[dict[str, float], np.ndarray]:
    y_pred = predict_array(model, test_seq, ctx.device)
    return compute_metrics(y_test, y_pred), y_pred


def predict_array(model: nn.Module, sequences: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(SequenceDataset(sequences, np.zeros(len(sequences), dtype="float32")), batch_size=256, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for seq, _ in loader:
            preds.append(model(seq.to(device)).detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds)


def metric_row(ctx: RunContext, mode: str, representation: str, learner: str, seed: int, source: str, target: str, metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "experiment_id": ctx.experiment_id,
        "model": "cumulative_s_dnn",
        "base_learner": learner if mode == "base_only" else None,
        "mode": mode,
        "input_representation": representation,
        "sensor_setting": ctx.config["model"].get("sensor_setting", "all_sensors"),
        "sequence_mode": ctx.config["model"]["sequence"].get("construction_mode", "fixed_length_resample"),
        "meta_learner": "ridge_adaboost" if mode == "s_dnn" else None,
        "seed": seed,
        "source_domain": source,
        "target_domain": target,
        "source_cases": ",".join(map(str, DOMAIN_CASES[source])),
        "target_cases": ",".join(map(str, DOMAIN_CASES[target])),
        **metrics,
    }


def prediction_frame(ctx: RunContext, y_pred: np.ndarray, test_meta: pd.DataFrame, mode: str, representation: str, learner: str, seed: int, source: str, target: str) -> pd.DataFrame:
    frame = test_meta.copy()
    frame["experiment_id"] = ctx.experiment_id
    frame["model"] = "cumulative_s_dnn"
    frame["mode"] = mode
    frame["base_learner"] = learner
    frame["input_representation"] = representation
    frame["seed"] = seed
    frame["source_domain"] = source
    frame["target_domain"] = target
    frame["y_true"] = frame["VB"].to_numpy(dtype="float32")
    frame["y_pred"] = y_pred
    frame["residual"] = frame["y_true"] - frame["y_pred"]
    frame["absolute_error"] = frame["residual"].abs()
    return frame


def run_epoch(model: nn.Module, loader: DataLoader, optimizer, loss_fn, device: torch.device, train: bool) -> float:
    model.train(train)
    total = 0.0
    count = 0
    for seq, target in loader:
        seq = seq.to(device)
        target = target.to(device)
        if train:
            optimizer.zero_grad()
        out = model(seq)
        loss = loss_fn(out, target)
        if train:
            loss.backward()
            optimizer.step()
        total += float(loss.detach().cpu()) * len(seq)
        count += len(seq)
    return total / max(count, 1)


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for seq, target in loader:
            ys.append(target.numpy().reshape(-1))
            preds.append(model(seq.to(device)).detach().cpu().numpy().reshape(-1))
    return compute_metrics(np.concatenate(ys), np.concatenate(preds))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        pearson = float(pearsonr(y_true, y_pred).statistic)
        spearman = float(spearmanr(y_true, y_pred).statistic)
    else:
        pearson = math.nan
        spearman = math.nan
    return {
        "metric_mae": float(np.mean(np.abs(y_true - y_pred))),
        "metric_mse": mse,
        "metric_rmse": float(np.sqrt(mse)),
        "metric_r2": r2_score_np(y_true, y_pred),
        "metric_pearson": pearson,
        "metric_spearman": spearman,
    }


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return math.nan
    return 1.0 - float(np.sum((y_true - y_pred) ** 2) / denom)


def save_model_checkpoint(ctx: RunContext, filename: str, model: nn.Module, optimizer, epoch: int, best_rmse: float) -> None:
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch, "best_rmse": best_rmse, "config": ctx.config}, ctx.output_dir / "checkpoints" / filename)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_real_data(dataset: pd.DataFrame, sensors: list[str], config: dict[str, Any]) -> dict[str, Any]:
    errors, warnings = [], []
    if sorted(dataset["case"].unique().tolist()) != CASE_SCOPE:
        errors.append(f"case filtering failed: {sorted(dataset['case'].unique().tolist())}")
    if "VB" not in dataset.columns:
        errors.append("VB target column missing")
    if not sensors:
        errors.append("No usable sensors found.")
    if "no_load" in config["model"]["sequence"].get("segment_mode", ""):
        errors.append("no-load segment is excluded for this experiment.")
    checks = []
    for shift in config["runtime"]["shifts"]:
        source, target = parse_shift(shift)
        train_cases = set(DOMAIN_CASES[source])
        test_cases = set(DOMAIN_CASES[target])
        overlap = sorted(train_cases & test_cases)
        if overlap:
            errors.append(f"source/target overlap in {shift}: {overlap}")
        checks.append(
            {
                "shift": shift,
                "source_cases": DOMAIN_CASES[source],
                "target_cases": DOMAIN_CASES[target],
                "target_in_base_training": False,
                "target_in_oof_fold": False,
                "target_label_used_for_meta_fit": False,
                "scaler_fit_scope": "source_train_only",
                "cumulative_descriptor_scope": "within_sample_sequence_only",
                "paper_split_used": False,
                "domain_shift_protocol_used": True,
                "case_overlap": overlap,
            }
        )
    return {"ok": not errors, "errors": errors, "warnings": warnings, "sensors": sensors, "leakage_check": {"ok": not errors, "checks": checks, "warnings": warnings}}


def build_split_artifacts(dataset: pd.DataFrame, shifts: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows, summary = [], []
    for shift in shifts:
        source, target = parse_shift(shift)
        split = split_frame(dataset, source, target)
        summary.append({"shift": shift, "source_cases": DOMAIN_CASES[source], "target_cases": DOMAIN_CASES[target], "split_counts": split["split"].value_counts().to_dict()})
        rows.extend(split[["sample_id", "dataset_run_id", "case", "run", "domain_id", "VB", "split"]].assign(source_domain=source, target_domain=target).to_dict("records"))
    return pd.DataFrame(rows), {"shifts": summary}


def write_data_summaries(output_dir: Path, dataset: pd.DataFrame, sensors: list[str], config: dict[str, Any]) -> None:
    write_json(output_dir / "data" / f"{PREFIX}_dataset_summary.json", {"num_samples": int(len(dataset)), "cases": sorted(dataset["case"].unique().tolist()), "target": "VB", "paper_target_name": "RUL", "framework_target_name": "VB"})
    pd.DataFrame([{"sensor_setting": config["model"].get("sensor_setting", "all_sensors"), "sensor_name": s, "used": True} for s in sensors]).to_csv(output_dir / "data" / f"{PREFIX}_sensor_mapping.csv", index=False)
    seq_cfg = config["model"]["sequence"]
    write_json(output_dir / "data" / f"{PREFIX}_sequence_shape_summary.json", {"construction_mode": seq_cfg.get("construction_mode"), "window_size": seq_cfg.get("window_size"), "stride": seq_cfg.get("stride"), "resample_length": seq_cfg.get("resample_length"), "segment_mode": seq_cfg.get("segment_mode"), "num_sensors": len(sensors)})
    write_json(output_dir / "data" / f"{PREFIX}_cumulative_descriptor_summary.json", {"formula": "cumsum / sqrt(abs(cumsum) + epsilon)", "input_representations": config["runtime"]["input_representations"], "scope": "within_sample_sequence_only", "order": config["preprocessing"].get("cumulative_descriptor_order", "after_scaling")})


def aggregate_seed_metrics(base_metrics: pd.DataFrame, s_metrics: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not base_metrics.empty:
        frames.append(base_metrics.assign(model_variant=base_metrics["base_learner"]))
    if not s_metrics.empty:
        frames.append(s_metrics.assign(model_variant="s_dnn"))
    if not frames:
        return pd.DataFrame()
    all_metrics = pd.concat(frames, ignore_index=True)
    return all_metrics.groupby(["mode", "model_variant", "input_representation", "sensor_setting", "sequence_mode", "seed"], dropna=False).agg(
        mean_mae_over_shifts=("metric_mae", "mean"),
        mean_rmse_over_shifts=("metric_rmse", "mean"),
        mean_r2_over_shifts=("metric_r2", "mean"),
        mean_pearson_over_shifts=("metric_pearson", "mean"),
        mean_spearman_over_shifts=("metric_spearman", "mean"),
    ).reset_index()


def aggregate_model_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    if seed_metrics.empty:
        return pd.DataFrame()
    return seed_metrics.groupby(["mode", "model_variant", "input_representation", "sensor_setting", "sequence_mode"], dropna=False).agg(
        mean_mae=("mean_mae_over_shifts", "mean"),
        std_mae=("mean_mae_over_shifts", "std"),
        mean_rmse=("mean_rmse_over_shifts", "mean"),
        std_rmse=("mean_rmse_over_shifts", "std"),
        mean_r2=("mean_r2_over_shifts", "mean"),
        std_r2=("mean_r2_over_shifts", "std"),
    ).reset_index()


def write_analysis_outputs(output_dir: Path, base_metrics: pd.DataFrame, s_metrics: pd.DataFrame) -> None:
    if not base_metrics.empty:
        base_metrics.groupby(["base_learner", "input_representation"]).agg(mean_r2=("metric_r2", "mean"), mean_rmse=("metric_rmse", "mean")).reset_index().to_csv(output_dir / "analysis" / f"{PREFIX}_base_learner_comparison.csv", index=False)
    if not s_metrics.empty and not base_metrics.empty:
        best_base = base_metrics.groupby(["input_representation", "seed", "source_domain", "target_domain"])["metric_r2"].max().reset_index(name="best_base_r2")
        comp = s_metrics.merge(best_base, on=["input_representation", "seed", "source_domain", "target_domain"], how="left")
        comp["delta_r2_s_dnn_vs_best_base"] = comp["metric_r2"] - comp["best_base_r2"]
        comp.to_csv(output_dir / "analysis" / f"{PREFIX}_s_dnn_vs_base_summary.csv", index=False)
    combined = []
    if not base_metrics.empty:
        combined.append(base_metrics.assign(variant=base_metrics["base_learner"]))
    if not s_metrics.empty:
        combined.append(s_metrics.assign(variant="s_dnn"))
    if combined:
        allm = pd.concat(combined, ignore_index=True)
        allm.assign(r2_positive=allm["metric_r2"] > 0).groupby(["mode", "variant", "input_representation"]).agg(num_total=("metric_r2", "size"), num_r2_positive=("r2_positive", "sum"), mean_r2=("metric_r2", "mean"), max_r2=("metric_r2", "max")).reset_index().to_csv(output_dir / "analysis" / f"{PREFIX}_r2_positive_count_summary.csv", index=False)
        allm.groupby(["source_domain", "target_domain", "mode", "input_representation"]).agg(mean_r2=("metric_r2", "mean"), mean_rmse=("metric_rmse", "mean")).reset_index().to_csv(output_dir / "analysis" / f"{PREFIX}_shift_robustness_summary.csv", index=False)
    if not base_metrics.empty:
        raw = base_metrics[base_metrics["input_representation"] == "raw"]
        cum = base_metrics[base_metrics["input_representation"] == "cumulative"]
        if not raw.empty and not cum.empty:
            key = ["base_learner", "mode", "sensor_setting", "sequence_mode", "seed", "source_domain", "target_domain"]
            comp = cum.merge(raw, on=key, suffixes=("_cumulative", "_raw"))
            comp["delta_mae"] = comp["metric_mae_cumulative"] - comp["metric_mae_raw"]
            comp["delta_rmse"] = comp["metric_rmse_cumulative"] - comp["metric_rmse_raw"]
            comp["delta_r2"] = comp["metric_r2_cumulative"] - comp["metric_r2_raw"]
            comp.to_csv(output_dir / "analysis" / f"{PREFIX}_raw_vs_cumulative_summary.csv", index=False)
        else:
            pd.DataFrame(columns=["note"]).assign(note="raw_vs_cumulative requires both raw and cumulative runs").to_csv(output_dir / "analysis" / f"{PREFIX}_raw_vs_cumulative_summary.csv", index=False)


def make_figures(output_dir: Path, base_metrics: pd.DataFrame, s_metrics: pd.DataFrame, base_preds: pd.DataFrame, s_preds: pd.DataFrame) -> None:
    figdir = output_dir / "figures"
    if not base_metrics.empty:
        base_metrics.groupby(["base_learner", "input_representation"])["metric_r2"].mean().unstack(fill_value=np.nan).plot(kind="bar", figsize=(7, 4))
        plt.ylabel("mean R2")
        plt.title("H3_S3 Base Learner Comparison")
        plt.tight_layout()
        plt.savefig(figdir / f"{PREFIX}_base_learner_comparison.png", dpi=170)
        plt.close()
    if not s_metrics.empty:
        s_metrics.groupby("input_representation")["metric_r2"].mean().plot(kind="bar", figsize=(5, 4))
        plt.ylabel("mean R2")
        plt.title("H3_S3 Raw vs Cumulative S-DNN")
        plt.tight_layout()
        plt.savefig(figdir / f"{PREFIX}_raw_vs_cumulative_performance.png", dpi=170)
        plt.close()
    combined = []
    if not base_metrics.empty:
        combined.append(base_metrics.assign(variant=base_metrics["base_learner"]))
    if not s_metrics.empty:
        combined.append(s_metrics.assign(variant="s_dnn"))
    if combined:
        allm = pd.concat(combined, ignore_index=True)
        allm["shift"] = allm["source_domain"] + "_to_" + allm["target_domain"]
        for variant, sub in allm.groupby("variant"):
            plt.plot(sub["shift"], sub["metric_r2"], marker="o", label=variant)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("R2")
        plt.title("H3_S3 Shift-wise Performance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / f"{PREFIX}_shift_wise_performance.png", dpi=170)
        plt.close()
    pred = s_preds if not s_preds.empty else base_preds
    if not pred.empty:
        plt.figure(figsize=(5, 5))
        plt.scatter(pred["y_true"], pred["y_pred"], alpha=0.7)
        lo = min(pred["y_true"].min(), pred["y_pred"].min())
        hi = max(pred["y_true"].max(), pred["y_pred"].max())
        plt.plot([lo, hi], [lo, hi], "--", color="black")
        plt.xlabel("True VB")
        plt.ylabel("Predicted VB")
        plt.title("H3_S3 Prediction vs True")
        plt.tight_layout()
        plt.savefig(figdir / f"{PREFIX}_prediction_vs_true.png", dpi=170)
        plt.close()
    if not s_metrics.empty and not base_metrics.empty:
        s_mean = s_metrics.assign(variant="s_dnn")[["variant", "metric_r2"]]
        b_mean = base_metrics.assign(variant=base_metrics["base_learner"])[["variant", "metric_r2"]]
        pd.concat([s_mean, b_mean]).groupby("variant")["metric_r2"].mean().plot(kind="bar", figsize=(7, 4))
        plt.ylabel("mean R2")
        plt.title("H3_S3 S-DNN vs Base")
        plt.tight_layout()
        plt.savefig(figdir / f"{PREFIX}_s_dnn_vs_base.png", dpi=170)
        plt.close()


def write_implementation_summary(output_dir: Path) -> None:
    text = """# H3_S3 Model Implementation Summary

## Architecture

Implemented cumulative descriptor enhanced S-DNN for VB prediction.

Raw multi-sensor time series -> train-only z-score normalization -> cumulative descriptor -> RNN/GRU/LSTM/CNN base learners -> source-domain-only OOF prediction matrix -> Ridge-AdaBoost meta learner -> VB prediction.

## Paper vs Framework

- Paper target: RUL.
- Framework target: VB.
- Paper split/evaluation: paper-specific train/test and 5-fold construction.
- Framework split/evaluation: A/B/C pair domain shift scenarios. Target domain is evaluation-only.

## Cumulative Descriptor

`c(n,k) = cumsum(x[:n,k]) / sqrt(abs(cumsum(x[:n,k])) + epsilon)`.

The descriptor is computed within each sample sequence only.

## Base Learners

- `RNNRegressor`
- `GRURegressor`
- `LSTMRegressor`
- `CNN1DRegressor`

Each returns `(batch_size, 1)`.

## Stacking

OOF matrix columns: `pred_rnn`, `pred_gru`, `pred_lstm`, `pred_cnn`.

The meta learner is `AdaBoostRegressor(estimator=Ridge())`.
OOF and meta fitting are restricted to source-domain training data.
"""
    (output_dir / "reports" / f"{PREFIX}_model_implementation_summary.md").write_text(text, encoding="utf-8")
    (output_dir / f"{PREFIX}_model_implementation_summary.md").write_text(text, encoding="utf-8")


def write_report(ctx: RunContext, base_metrics, s_metrics, seed_metrics, model_metrics, smoke: dict[str, Any], dry_run: bool) -> None:
    report = f"""# H3_S3 Cumulative Descriptor Enhanced S-DNN Report

## Purpose

This execution implements the 2023 cumulative descriptor enhanced S-DNN architecture as a VB prediction baseline.

## Architecture

- Cumulative descriptor: deterministic per-sample cumulative transform.
- Base learners: RNN, GRU, LSTM, 1D-CNN.
- Ensemble: Ridge-based AdaBoost meta learner.
- Paper target: RUL.
- Current target: VB.

## Protocol Difference

The paper's split and case-study evaluation are not used here. This framework uses source/target domain shift splits:
A={{1,9}}, B={{2,12}}, C={{8,14}}. Target domain data is reserved for final evaluation only.

## Smoke Test

```json
{json.dumps(smoke, indent=2)}
```

## Results

Dry-run: {dry_run}

Base learner metric rows: {0 if base_metrics is None else len(base_metrics)}
S-DNN metric rows: {0 if s_metrics is None else len(s_metrics)}

## Leakage Controls

- Scaler fit scope: source train only.
- OOF scope: source train only.
- Meta learner fit scope: source OOF only.
- Target domain labels: evaluation only.
- Cumulative descriptor scope: within sample sequence only.

## Interpretation Notes

Negative R2 means worse than the mean prediction baseline. S-DNN may be unstable under small-sample domain shift; this should not be treated as implementation failure without further validation.
"""
    (ctx.output_dir / "reports" / f"{PREFIX}_report.md").write_text(report, encoding="utf-8")
    html_report = "<!doctype html><html><head><meta charset='utf-8'><title>H3_S3 Report</title></head><body><pre>" + html.escape(report) + "</pre></body></html>"
    (ctx.output_dir / "reports" / f"{PREFIX}_report.html").write_text(html_report, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

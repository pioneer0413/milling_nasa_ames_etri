#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import copy
import html
import json
import math
import shutil
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
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.experiment_logging.environment import collect_environment
from milling_experiment_framework.experiments.execution_path import create_execution_dir
from milling_experiment_framework.models.dl.hybrid_lstm_process import HybridLSTMProcessRegressor


PREFIX = "H3_S2"
CASE_SCOPE = [1, 2, 8, 9, 12, 14]
DOMAIN_CASES = {"A": [1, 9], "B": [2, 12], "C": [8, 14]}
SHIFT_SCENARIOS = [("A", "B"), ("A", "C"), ("B", "A"), ("B", "C"), ("C", "A"), ("C", "B")]
EXPECTED_SENSORS = ["smcAC", "smcDC", "vib_spindle", "vib_table", "AE_spindle", "AE_table"]
SENSOR_GROUPS = {
    "current": ["smcAC", "smcDC"],
    "vibration": ["vib_spindle", "vib_table"],
    "acoustic": ["AE_spindle", "AE_table"],
    "all_sensors": EXPECTED_SENSORS,
}
PROCESS_COLUMNS = {
    "doc": ["DOC"],
    "feed": ["feed"],
    "material": ["material_name"],
    "time": ["time"],
}
PROCESS_COMBINATIONS = {
    "none": [],
    "doc": ["doc"],
    "feed": ["feed"],
    "material": ["material"],
    "time": ["time"],
    "doc_feed": ["doc", "feed"],
    "doc_material": ["doc", "material"],
    "doc_time": ["doc", "time"],
    "feed_material": ["feed", "material"],
    "feed_time": ["feed", "time"],
    "material_time": ["material", "time"],
    "doc_feed_material": ["doc", "feed", "material"],
    "doc_feed_time": ["doc", "feed", "time"],
    "doc_material_time": ["doc", "material", "time"],
    "feed_material_time": ["feed", "material", "time"],
    "doc_feed_material_time": ["doc", "feed", "material", "time"],
    "without_time": ["doc", "feed", "material"],
    "without_material": ["doc", "feed", "time"],
    "without_time_material": ["doc", "feed"],
}


@dataclass
class RunContext:
    experiment_id: str
    output_dir: Path
    config: dict[str, Any]
    device: torch.device
    logger: Any
    global_best_val_rmse: float | None = None


class HybridSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, process_features: np.ndarray, targets: np.ndarray):
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.process_features = torch.as_tensor(process_features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.sequences[index], self.process_features[index], self.targets[index]


class ProcessPreprocessor:
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.numeric_cols = [c for c in columns if c != "material_name"]
        self.categorical_cols = [c for c in columns if c == "material_name"]
        self.numeric_mean: pd.Series | None = None
        self.numeric_std: pd.Series | None = None
        self.categories: dict[str, list[str]] = {}

    def fit(self, frame: pd.DataFrame) -> "ProcessPreprocessor":
        if self.numeric_cols:
            self.numeric_mean = frame[self.numeric_cols].astype(float).mean()
            self.numeric_std = frame[self.numeric_cols].astype(float).std().replace(0, 1.0).fillna(1.0)
        for col in self.categorical_cols:
            self.categories[col] = sorted(frame[col].fillna("__missing__").astype(str).unique().tolist())
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        parts = []
        names = []
        if self.numeric_cols:
            assert self.numeric_mean is not None and self.numeric_std is not None
            numeric = (frame[self.numeric_cols].astype(float) - self.numeric_mean) / self.numeric_std
            parts.append(numeric.to_numpy(dtype="float32"))
            names.extend(self.numeric_cols)
        for col in self.categorical_cols:
            cats = self.categories[col]
            values = frame[col].fillna("__missing__").astype(str)
            one_hot = np.zeros((len(frame), len(cats)), dtype="float32")
            cat_to_idx = {cat: idx for idx, cat in enumerate(cats)}
            for row_idx, value in enumerate(values):
                if value in cat_to_idx:
                    one_hot[row_idx, cat_to_idx[value]] = 1.0
            parts.append(one_hot)
            names.extend([f"{col}={cat}" for cat in cats])
        if not parts:
            return np.zeros((len(frame), 0), dtype="float32"), []
        return np.concatenate(parts, axis=1), names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid LSTM + process information VB prediction experiment.")
    parser.add_argument("--config", default="configs/models/hybrid_lstm_process.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--segment-modes", default=None, help="Comma-separated segment modes.")
    parser.add_argument("--process-combinations", default=None, help="Comma-separated process combinations.")
    parser.add_argument("--shifts", default=None, help="Comma-separated shifts such as A_to_B.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    args = parser.parse_args()

    output_dir: Path | None = None
    try:
        config = load_config(Path(args.config))
        apply_cli_overrides(config, args)
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_H3_S2_hybrid_lstm_process_architecture")
        path_config = {"experiment": {"experiment_id": experiment_id}}
        output_dir = Path(create_execution_dir(path_config, root=Path("experiments") / "executions"))
        prepare_dirs(output_dir)
        logger = make_logger(output_dir / "logs" / f"{PREFIX}_run.log")
        device = torch.device("cuda" if torch.cuda.is_available() and config.get("training", {}).get("device") == "cuda" else "cpu")
        ctx = RunContext(experiment_id=experiment_id, output_dir=output_dir, config=config, device=device, logger=logger)
        logger(f"H3.S2 hybrid LSTM process execution started: {experiment_id}")
        write_json(output_dir / "logs" / f"{PREFIX}_environment.json", collect_environment())
        write_json(output_dir / "configs" / f"{PREFIX}_input_config.yaml", config)
        resolved = copy.deepcopy(config)
        resolved["experiment"] = {**path_config["experiment"], "experiment_id": experiment_id, "created_at": datetime.now().isoformat(timespec="seconds")}
        write_json(output_dir / "configs" / f"{PREFIX}_resolved_config.yaml", resolved)
        write_implementation_summary(output_dir)

        smoke_result = run_smoke_test(ctx)
        write_json(output_dir / "data" / f"{PREFIX}_smoke_test_result.json", smoke_result)
        write_json(output_dir / f"{PREFIX}_smoke_test_result.json", smoke_result)
        if args.smoke_test and not args.dry_run:
            write_report(ctx, None, None, None, smoke_result, dry_run=False)
            print(json.dumps({"experiment_id": experiment_id, "execution_dir": str(output_dir), "smoke_test": smoke_result}, indent=2, ensure_ascii=False))
            return

        dataset = load_dataset(config)
        sensors = resolve_sensors(dataset, config["model"].get("sensor_setting", "all_sensors"))
        sequence_modes = config["runtime"]["segment_modes"]
        process_combos = config["runtime"]["process_combinations"]
        shifts = config["runtime"]["shifts"]
        seeds = config["runtime"]["seeds"]
        sequence_length = int(config["preprocessing"].get("sequence_length", 512))
        validation = validate_real_data(dataset, sensors, sequence_modes, process_combos, shifts)
        write_json(output_dir / "configs" / f"{PREFIX}_config_validation.json", validation)
        write_data_summaries(output_dir, dataset, sensors, sequence_modes, process_combos, sequence_length)
        split_table, split_summary = build_split_artifacts(dataset, shifts)
        split_table.to_csv(output_dir / "splits" / f"{PREFIX}_split.csv", index=False)
        write_json(output_dir / "splits" / f"{PREFIX}_split_summary.json", split_summary)
        write_json(output_dir / "splits" / f"{PREFIX}_leakage_check.json", validation["leakage_check"])
        if args.dry_run:
            write_report(ctx, None, None, None, smoke_result, dry_run=True)
            logger("Dry-run completed without training.")
            print(json.dumps({"experiment_id": experiment_id, "execution_dir": str(output_dir), "dry_run": True, "validation": validation}, indent=2, ensure_ascii=False))
            return

        shift_metrics, predictions, history_rows = run_training_grid(ctx, dataset, sensors, sequence_modes, process_combos, shifts, seeds, sequence_length)
        seed_metrics = aggregate_seed_metrics(shift_metrics)
        model_metrics = aggregate_model_metrics(seed_metrics)
        pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
        shift_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_shift_metrics.csv", index=False)
        seed_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_seed_metrics.csv", index=False)
        model_metrics.to_csv(output_dir / "metrics" / f"{PREFIX}_model_metrics.csv", index=False)
        pred_df.to_csv(output_dir / "predictions" / f"{PREFIX}_predictions.csv", index=False)
        if not pred_df.empty:
            pred_df.sort_values("absolute_error", ascending=False).head(200).to_csv(output_dir / "predictions" / f"{PREFIX}_prediction_errors.csv", index=False)
        pd.DataFrame(history_rows).to_csv(output_dir / "metrics" / f"{PREFIX}_training_history.csv", index=False)
        write_analysis_outputs(output_dir, shift_metrics, seed_metrics, model_metrics)
        make_figures(output_dir, shift_metrics, pred_df)
        write_report(ctx, shift_metrics, seed_metrics, model_metrics, smoke_result, dry_run=False)
        logger(f"H3.S2 execution finished: {experiment_id}")
        print(json.dumps({"experiment_id": experiment_id, "execution_dir": str(output_dir), "num_shift_rows": len(shift_metrics)}, indent=2, ensure_ascii=False))
    except Exception:
        if output_dir is not None:
            (output_dir / "logs" / f"{PREFIX}_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    runtime = config.setdefault("runtime", {})
    runtime.setdefault("segment_modes", [config["model"].get("segment_mode", "entry_exit")])
    runtime.setdefault("process_combinations", [config["model"].get("process_combination", "doc_feed_material_time")])
    runtime.setdefault("shifts", ["A_to_B"])
    runtime.setdefault("seeds", [0])
    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.segment_modes:
        config["runtime"]["segment_modes"] = [x.strip() for x in args.segment_modes.split(",") if x.strip()]
    if args.process_combinations:
        config["runtime"]["process_combinations"] = [x.strip() for x in args.process_combinations.split(",") if x.strip()]
    if args.shifts:
        config["runtime"]["shifts"] = [x.strip() for x in args.shifts.split(",") if x.strip()]
    if args.seeds:
        config["runtime"]["seeds"] = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.sequence_length is not None:
        config["preprocessing"]["sequence_length"] = args.sequence_length
    if args.hidden_size is not None:
        config["model"]["lstm"]["hidden_size"] = args.hidden_size
    if args.num_layers is not None:
        config["model"]["lstm"]["num_layers"] = args.num_layers


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
    batch_size = 4
    sequence_length = 128
    num_sensors = 6
    num_process_features = 4
    model = HybridLSTMProcessRegressor(
        num_sensors=num_sensors,
        process_feature_dim=num_process_features,
        hidden_size=32,
        num_layers=2,
        regression_hidden_dims=[16, 8],
        use_process_info=True,
    ).to(ctx.device)
    sensor_sequence = torch.randn(batch_size, sequence_length, num_sensors, device=ctx.device)
    process_features = torch.randn(batch_size, num_process_features, device=ctx.device)
    target = torch.randn(batch_size, 1, device=ctx.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    out = model(sensor_sequence, process_features)
    loss = nn.MSELoss()(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ckpt_path = ctx.output_dir / "checkpoints" / "smoke_hybrid_lstm_process.pt"
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": 1}, ckpt_path)
    loaded = HybridLSTMProcessRegressor(
        num_sensors=num_sensors,
        process_feature_dim=num_process_features,
        hidden_size=32,
        num_layers=2,
        regression_hidden_dims=[16, 8],
        use_process_info=True,
    )
    loaded.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state"])
    sensor_only = HybridLSTMProcessRegressor(
        num_sensors=num_sensors,
        process_feature_dim=0,
        hidden_size=16,
        num_layers=1,
        regression_hidden_dims=[8],
        use_process_info=False,
        input_mode="sensor_only",
    )
    process_only = HybridLSTMProcessRegressor(
        num_sensors=1,
        process_feature_dim=num_process_features,
        hidden_size=16,
        num_layers=1,
        regression_hidden_dims=[8],
        input_mode="process_only",
    )
    sensor_only(torch.randn(2, 32, num_sensors), None)
    process_only(None, torch.randn(2, num_process_features))
    return {
        "model_forward_success": True,
        "batch_input_shape": [batch_size, sequence_length, num_sensors],
        "process_input_shape": [batch_size, num_process_features],
        "output_shape": list(out.shape),
        "loss": float(loss.detach().cpu()),
        "loss_backward_success": True,
        "checkpoint_save_load_success": True,
        "checkpoint_path": str(ckpt_path),
        "sensor_only_mode_success": True,
        "process_only_mode_success": True,
    }


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config.get("data", {})
    process = pd.read_csv(data_cfg["process_info_path"])
    signal = pd.read_csv(data_cfg["signal_data_path"])
    heuristic = pd.read_csv(data_cfg["heuristic_sequence_path"])
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
    data = data.loc[data["VB"].notna()].reset_index(drop=True)
    data["sample_id"] = data.apply(lambda row: f"case_{int(row['case']):02d}_run_{int(row['run']):03d}", axis=1)
    data["dataset_run_id"] = data.apply(lambda row: f"case_{int(row['case']):02d}", axis=1)
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
        raise ValueError(f"Empty sequence for segment_mode={mode}, sample={row.get('sample_id')}")
    return out.astype("float32")


def resample_1d(arr: np.ndarray, sequence_length: int) -> np.ndarray:
    if len(arr) == sequence_length:
        return arr.astype("float32")
    old = np.linspace(0.0, 1.0, num=len(arr), dtype="float32")
    new = np.linspace(0.0, 1.0, num=sequence_length, dtype="float32")
    return np.interp(new, old, arr).astype("float32")


def build_sequences(data: pd.DataFrame, sensors: list[str], segment_mode: str, sequence_length: int) -> np.ndarray:
    rows = []
    for _, row in data.iterrows():
        channels = []
        for sensor in sensors:
            arr = np.asarray(ast.literal_eval(row[sensor]), dtype="float32")
            channels.append(resample_1d(segment_array(arr, row, segment_mode), sequence_length))
        rows.append(np.stack(channels, axis=1))
    return np.stack(rows, axis=0).astype("float32")


def process_columns(process_combination: str) -> list[str]:
    factors = PROCESS_COMBINATIONS[process_combination]
    cols = []
    for factor in factors:
        cols.extend(PROCESS_COLUMNS[factor])
    return cols


def split_frame(data: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    source_cases = DOMAIN_CASES[source]
    target_cases = DOMAIN_CASES[target]
    validation_case = min(source_cases)
    split = data.loc[data["case"].isin(source_cases + target_cases)].copy()
    split["split"] = "excluded"
    split.loc[split["case"].isin(target_cases), "split"] = "test"
    split.loc[split["case"].isin(source_cases) & (split["case"] != validation_case), "split"] = "train"
    split.loc[split["case"] == validation_case, "split"] = "validation"
    return split


def normalize_sequences(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return tuple(((x - mean) / std).astype("float32") for x in (train, *others))


def make_model(config: dict[str, Any], num_sensors: int, process_dim: int, input_mode: str) -> HybridLSTMProcessRegressor:
    model_cfg = config["model"]
    lstm = model_cfg["lstm"]
    head = model_cfg["regression_head"]
    return HybridLSTMProcessRegressor(
        num_sensors=num_sensors,
        process_feature_dim=process_dim,
        hidden_size=int(lstm.get("hidden_size", 256)),
        num_layers=int(lstm.get("num_layers", 3)),
        lstm_dropout=float(lstm.get("dropout", 0.2)),
        bidirectional=bool(lstm.get("bidirectional", False)),
        temporal_output_mode=lstm.get("temporal_output_mode", "last_hidden"),
        regression_hidden_dims=head.get("hidden_dims", [32, 8]),
        regression_dropout=float(head.get("dropout", 0.2)),
        use_process_info=process_dim > 0 and input_mode != "sensor_only",
        input_mode=input_mode,
    )


def run_training_grid(
    ctx: RunContext,
    dataset: pd.DataFrame,
    sensors: list[str],
    segment_modes: list[str],
    process_combos: list[str],
    shifts: list[str],
    seeds: list[int],
    sequence_length: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[dict[str, Any]]]:
    shift_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    history_rows: list[dict[str, Any]] = []
    for segment_mode in segment_modes:
        ctx.logger(f"Building sequences for segment_mode={segment_mode}")
        all_sequences = build_sequences(dataset, sensors, segment_mode, sequence_length)
        for process_combo in process_combos:
            if process_combo not in PROCESS_COMBINATIONS:
                raise ValueError(f"Unsupported process_combination: {process_combo}")
            if "time" in PROCESS_COMBINATIONS[process_combo] or "material" in PROCESS_COMBINATIONS[process_combo]:
                ctx.logger(f"Leakage warning: process_combination={process_combo} includes Time and/or Material.")
            for seed in seeds:
                set_seed(seed)
                for shift in shifts:
                    source, target = parse_shift(shift)
                    result, preds, history = run_one_training(ctx, dataset, all_sequences, sensors, segment_mode, process_combo, seed, source, target)
                    shift_rows.append(result)
                    predictions.append(preds)
                    history_rows.extend(history)
    return pd.DataFrame(shift_rows), predictions, history_rows


def run_one_training(
    ctx: RunContext,
    dataset: pd.DataFrame,
    all_sequences: np.ndarray,
    sensors: list[str],
    segment_mode: str,
    process_combo: str,
    seed: int,
    source: str,
    target: str,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    split = split_frame(dataset, source, target)
    indices = split.index.to_numpy()
    train_mask = split["split"].to_numpy() == "train"
    val_mask = split["split"].to_numpy() == "validation"
    test_mask = split["split"].to_numpy() == "test"
    seq = all_sequences[indices]
    train_seq, val_seq, test_seq = normalize_sequences(seq[train_mask], seq[val_mask], seq[test_mask])
    cols = process_columns(process_combo)
    pre = ProcessPreprocessor(cols).fit(split.loc[train_mask])
    train_proc, process_feature_names = pre.transform(split.loc[train_mask])
    val_proc, _ = pre.transform(split.loc[val_mask])
    test_proc, _ = pre.transform(split.loc[test_mask])
    input_mode = "sensor_only" if len(cols) == 0 else "sensor_plus_process"
    model = make_model(ctx.config, len(sensors), train_proc.shape[1], input_mode).to(ctx.device)
    training_cfg = ctx.config["training"]
    batch_size = int(training_cfg.get("batch_size", 16))
    train_loader = DataLoader(HybridSequenceDataset(train_seq, train_proc, split.loc[train_mask, "VB"].to_numpy(dtype="float32")), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(HybridSequenceDataset(val_seq, val_proc, split.loc[val_mask, "VB"].to_numpy(dtype="float32")), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(HybridSequenceDataset(test_seq, test_proc, split.loc[test_mask, "VB"].to_numpy(dtype="float32")), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg.get("learning_rate", 0.001)))
    loss_name = training_cfg.get("loss", "mse")
    loss_fn = RMSELoss() if loss_name == "rmse" else nn.MSELoss()
    max_epochs = int(training_cfg.get("max_epochs", training_cfg.get("epochs", 2)))
    child_key = f"hybrid_lstm_process_{segment_mode}_{process_combo}_seed{seed}_{source}_to_{target}"
    best_val = math.inf
    best_state: dict[str, Any] | None = None
    history = []
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, ctx.device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, loss_fn, ctx.device, train=False)
        val_rmse = evaluate_loader(model, val_loader, ctx.device)["metric_rmse"]
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(ctx, f"{child_key}_best.pt", model, optimizer, epoch, best_val, child_key)
            if ctx.global_best_val_rmse is None or best_val < ctx.global_best_val_rmse:
                ctx.global_best_val_rmse = best_val
                save_checkpoint(ctx, "best.pt", model, optimizer, epoch, best_val, child_key)
        save_checkpoint(ctx, f"{child_key}_last.pt", model, optimizer, epoch, best_val, child_key)
        save_checkpoint(ctx, "last.pt", model, optimizer, epoch, best_val, child_key)
        history.append(
            {
                "child_execution_key": child_key,
                "segment_mode": segment_mode,
                "process_combination": process_combo,
                "seed": seed,
                "source_domain": source,
                "target_domain": target,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
            }
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate_loader(model, test_loader, ctx.device)
    y_true, y_pred = predict_loader(model, test_loader, ctx.device)
    result = {
        "experiment_id": ctx.experiment_id,
        "child_execution_key": child_key,
        "model": "hybrid_lstm_process",
        "sensor_setting": ctx.config["model"].get("sensor_setting", "all_sensors"),
        "segment_mode": segment_mode,
        "process_combination": process_combo,
        "seed": seed,
        "source_domain": source,
        "target_domain": target,
        "source_cases": ",".join(map(str, DOMAIN_CASES[source])),
        "target_cases": ",".join(map(str, DOMAIN_CASES[target])),
        **metrics,
    }
    pred_meta = split.loc[test_mask, ["sample_id", "dataset_run_id", "case", "run", "domain_id", "VB"]].copy()
    pred_meta["child_execution_key"] = child_key
    pred_meta["model"] = "hybrid_lstm_process"
    pred_meta["sensor_setting"] = ctx.config["model"].get("sensor_setting", "all_sensors")
    pred_meta["segment_mode"] = segment_mode
    pred_meta["process_combination"] = process_combo
    pred_meta["seed"] = seed
    pred_meta["source_domain"] = source
    pred_meta["target_domain"] = target
    pred_meta["split"] = "test"
    pred_meta["y_true"] = y_true
    pred_meta["y_pred"] = y_pred
    pred_meta["residual"] = pred_meta["y_true"] - pred_meta["y_pred"]
    pred_meta["absolute_error"] = pred_meta["residual"].abs()
    return result, pred_meta, history


class RMSELoss(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2) + 1e-8)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, loss_fn, device, train: bool) -> float:
    model.train(train)
    total = 0.0
    count = 0
    for seq, proc, target in loader:
        seq = seq.to(device)
        proc = proc.to(device)
        target = target.to(device)
        if train:
            optimizer.zero_grad()
        out = model(seq, proc if proc.shape[1] > 0 else None)
        loss = loss_fn(out, target)
        if train:
            loss.backward()
            optimizer.step()
        total += float(loss.detach().cpu()) * len(seq)
        count += len(seq)
    return total / max(count, 1)


def predict_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    preds = []
    with torch.no_grad():
        for seq, proc, target in loader:
            seq = seq.to(device)
            proc = proc.to(device)
            out = model(seq, proc if proc.shape[1] > 0 else None)
            ys.append(target.numpy().reshape(-1))
            preds.append(out.detach().cpu().numpy().reshape(-1))
    return np.concatenate(ys), np.concatenate(preds)


def evaluate_loader(model, loader, device) -> dict[str, float]:
    y_true, y_pred = predict_loader(model, loader, device)
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        pearson = float(pearsonr(y_true, y_pred).statistic)
        spearman = float(spearmanr(y_true, y_pred).statistic)
    else:
        pearson = math.nan
        spearman = math.nan
    return {
        "metric_mae": float(np.mean(np.abs(y_true - y_pred))),
        "metric_rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "metric_r2": float(r2_score_np(y_true, y_pred)),
        "metric_pearson": pearson,
        "metric_spearman": spearman,
    }


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return math.nan
    return 1.0 - float(np.sum((y_true - y_pred) ** 2) / denom)


def save_checkpoint(ctx: RunContext, filename: str, model, optimizer, epoch: int, best_metric: float, child_key: str) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "child_execution_key": child_key,
            "config": ctx.config,
        },
        ctx.output_dir / "checkpoints" / filename,
    )


def parse_shift(shift: str) -> tuple[str, str]:
    source, target = shift.split("_to_")
    return source, target


def validate_real_data(dataset: pd.DataFrame, sensors: list[str], segment_modes: list[str], process_combos: list[str], shifts: list[str]) -> dict[str, Any]:
    errors = []
    warnings = []
    if sorted(dataset["case"].unique().tolist()) != CASE_SCOPE:
        errors.append(f"case filtering failed: {sorted(dataset['case'].unique().tolist())}")
    if "VB" not in dataset.columns:
        errors.append("VB target column missing")
    for mode in segment_modes:
        if "no_load" in mode:
            errors.append(f"no-load segment is not allowed: {mode}")
    for combo in process_combos:
        if combo not in PROCESS_COMBINATIONS:
            errors.append(f"unsupported process combination: {combo}")
        factors = PROCESS_COMBINATIONS.get(combo, [])
        if "time" in factors:
            warnings.append(f"{combo} includes Time; leakage/run-order validation is required.")
        if "material" in factors:
            warnings.append(f"{combo} includes Material; domain shortcut validation is required.")
    leakage_checks = []
    for shift in shifts:
        source, target = parse_shift(shift)
        split = split_frame(dataset, source, target)
        train_cases = set(split.loc[split["split"].isin(["train", "validation"]), "case"])
        test_cases = set(split.loc[split["split"] == "test", "case"])
        overlap = sorted(train_cases & test_cases)
        if overlap:
            errors.append(f"source/target overlap in {shift}: {overlap}")
        leakage_checks.append({"shift": shift, "source_cases": DOMAIN_CASES[source], "target_cases": DOMAIN_CASES[target], "case_overlap": overlap})
    return {"ok": not errors, "errors": errors, "warnings": warnings, "sensors": sensors, "leakage_check": {"ok": not errors, "checks": leakage_checks, "warnings": warnings}}


def write_data_summaries(output_dir: Path, dataset: pd.DataFrame, sensors: list[str], segment_modes: list[str], process_combos: list[str], sequence_length: int) -> None:
    write_json(
        output_dir / "data" / f"{PREFIX}_dataset_summary.json",
        {
            "num_samples": int(len(dataset)),
            "cases": sorted(dataset["case"].unique().tolist()),
            "rows_by_case": dataset.groupby("case").size().to_dict(),
            "target": "VB",
        },
    )
    pd.DataFrame([{"sensor_setting": "resolved", "sensor_name": s, "used": True} for s in sensors]).to_csv(output_dir / "data" / f"{PREFIX}_sensor_mapping.csv", index=False)
    rows = []
    for combo in process_combos:
        cols = process_columns(combo)
        rows.append({"process_combination": combo, "columns": ",".join(cols), "num_raw_columns": len(cols), "leakage_warning": "time/material requires validation" if any(c in {"time", "material_name"} for c in cols) else ""})
    process_summary = pd.DataFrame(rows)
    process_summary.to_csv(output_dir / "data" / f"{PREFIX}_process_feature_summary.csv", index=False)
    write_json(output_dir / "data" / f"{PREFIX}_process_feature_summary.json", rows)
    pd.DataFrame([{"segment_mode": mode, "sequence_length_after_resample": sequence_length, "num_sensors": len(sensors)} for mode in segment_modes]).to_csv(output_dir / "data" / f"{PREFIX}_sequence_shape_summary.csv", index=False)


def build_split_artifacts(dataset: pd.DataFrame, shifts: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    summary = []
    for shift in shifts:
        source, target = parse_shift(shift)
        split = split_frame(dataset, source, target)
        summary.append({"shift": shift, "source_cases": DOMAIN_CASES[source], "target_cases": DOMAIN_CASES[target], "split_counts": split["split"].value_counts().to_dict()})
        for _, row in split.iterrows():
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "dataset_run_id": row["dataset_run_id"],
                    "case": row["case"],
                    "domain_id": row["domain_id"],
                    "label": row["VB"],
                    "source_domain": source,
                    "target_domain": target,
                    "split": row["split"],
                }
            )
    return pd.DataFrame(rows), {"shifts": summary}


def aggregate_seed_metrics(shift_metrics: pd.DataFrame) -> pd.DataFrame:
    if shift_metrics.empty:
        return pd.DataFrame()
    return (
        shift_metrics.groupby(["model", "sensor_setting", "segment_mode", "process_combination", "seed"], dropna=False)
        .agg(
            mean_mae_over_6_shifts=("metric_mae", "mean"),
            mean_rmse_over_6_shifts=("metric_rmse", "mean"),
            mean_r2_over_6_shifts=("metric_r2", "mean"),
            mean_pearson_over_6_shifts=("metric_pearson", "mean"),
            mean_spearman_over_6_shifts=("metric_spearman", "mean"),
        )
        .reset_index()
    )


def aggregate_model_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    if seed_metrics.empty:
        return pd.DataFrame()
    return (
        seed_metrics.groupby(["model", "sensor_setting", "segment_mode", "process_combination"], dropna=False)
        .agg(
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
        .reset_index()
    )


def write_analysis_outputs(output_dir: Path, shift_metrics: pd.DataFrame, seed_metrics: pd.DataFrame, model_metrics: pd.DataFrame) -> None:
    if model_metrics.empty:
        return
    proc = model_metrics.groupby("process_combination").agg(mean_r2=("mean_r2", "mean"), mean_rmse=("mean_rmse", "mean")).reset_index()
    seg = model_metrics.groupby("segment_mode").agg(mean_r2=("mean_r2", "mean"), mean_rmse=("mean_rmse", "mean")).reset_index()
    r2 = model_metrics.assign(r2_positive=model_metrics["mean_r2"] > 0).groupby(["segment_mode", "process_combination"]).agg(num_conditions=("mean_r2", "size"), num_r2_positive=("r2_positive", "sum"), max_r2=("mean_r2", "max")).reset_index()
    proc.to_csv(output_dir / "analysis" / f"{PREFIX}_process_ablation_summary.csv", index=False)
    seg.to_csv(output_dir / "analysis" / f"{PREFIX}_segment_ablation_summary.csv", index=False)
    r2.to_csv(output_dir / "analysis" / f"{PREFIX}_r2_positive_count_summary.csv", index=False)


def make_figures(output_dir: Path, shift_metrics: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    figdir = output_dir / "figures"
    if not pred_df.empty:
        plt.figure(figsize=(5.5, 5))
        plt.scatter(pred_df["y_true"], pred_df["y_pred"], alpha=0.7)
        lo = min(pred_df["y_true"].min(), pred_df["y_pred"].min())
        hi = max(pred_df["y_true"].max(), pred_df["y_pred"].max())
        plt.plot([lo, hi], [lo, hi], "--", color="black")
        plt.xlabel("True VB")
        plt.ylabel("Predicted VB")
        plt.title("H3_S2 Prediction vs True")
        plt.tight_layout()
        plt.savefig(figdir / f"{PREFIX}_prediction_vs_true.png", dpi=180)
        plt.close()
    if not shift_metrics.empty:
        plot = shift_metrics.copy()
        plot["shift"] = plot["source_domain"] + "_to_" + plot["target_domain"]
        grouped = plot.groupby(["shift", "segment_mode"])["metric_r2"].mean().reset_index()
        for seg, sub in grouped.groupby("segment_mode"):
            plt.plot(sub["shift"], sub["metric_r2"], marker="o", label=seg)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("R2")
        plt.title("H3_S2 Shift-wise Performance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / f"{PREFIX}_shift_wise_performance.png", dpi=180)
        plt.close()
        shift_metrics.groupby("segment_mode")["metric_r2"].mean().plot(kind="bar", figsize=(7, 4))
        plt.ylabel("mean R2")
        plt.title("H3_S2 Segment Comparison")
        plt.tight_layout()
        plt.savefig(figdir / f"{PREFIX}_segment_comparison.png", dpi=180)
        plt.close()
        shift_metrics.groupby("process_combination")["metric_r2"].mean().plot(kind="bar", figsize=(7, 4))
        plt.ylabel("mean R2")
        plt.title("H3_S2 Process Ablation")
        plt.tight_layout()
        plt.savefig(figdir / f"{PREFIX}_process_ablation.png", dpi=180)
        plt.close()


def write_implementation_summary(output_dir: Path) -> None:
    text = """# H3_S2 Model Implementation Summary

## Architecture

Implemented class: `HybridLSTMProcessRegressor`.

Raw multi-sensor sequence `(batch, sequence_length, num_sensors)` is z-score normalized with train split statistics only, encoded by a stacked `nn.LSTM`, optionally concatenated with process information, and passed to a nonlinear regression head.

## Paper Correspondence

- Stacked LSTM temporal encoder: supported, default `num_layers=3`, `hidden_size=256`.
- Dropout after LSTM layers: PyTorch LSTM inter-layer dropout plus output dropout.
- Process information concatenation: supported through `process_features`.
- Nonlinear regression model: `Linear(input_dim, 32) -> ReLU -> Dropout -> Linear(32, 8) -> ReLU -> Dropout -> Linear(8, 1)`.
- Loss/optimizer: MSE or RMSE loss option, Adam optimizer.

## Framework Adaptations

NASA Ames milling data uses configurable `segment_mode`, `sensor_setting`, and process combinations. Time and Material are supported but warnings are emitted because they can leak run order or domain identity.

## Input Shapes

- `sensor_sequence`: `(batch_size, sequence_length, num_sensors)`
- `process_features`: `(batch_size, num_process_features)` or `None`
- output: `(batch_size, 1)`

## Config

Use `configs/models/hybrid_lstm_process.yaml`.
"""
    (output_dir / "reports" / f"{PREFIX}_model_implementation_summary.md").write_text(text, encoding="utf-8")
    (output_dir / f"{PREFIX}_model_implementation_summary.md").write_text(text, encoding="utf-8")


def write_report(ctx: RunContext, shift_metrics: pd.DataFrame | None, seed_metrics: pd.DataFrame | None, model_metrics: pd.DataFrame | None, smoke_result: dict[str, Any], dry_run: bool) -> None:
    best_table = "not available"
    if model_metrics is not None and not model_metrics.empty:
        best_table = model_metrics.sort_values("mean_r2", ascending=False).head(20).to_string(index=False)
    report = f"""# H3_S2 Hybrid LSTM Process Architecture Report

## Purpose

This execution implements and validates a hybrid LSTM + process-information architecture inspired by the paper `A hybrid information model based on long short-term memory network for tool condition monitoring`.

## Architecture

Raw multi-sensor time series are normalized with train-only z-score statistics, encoded by a stacked LSTM, concatenated with process information, and fed into a nonlinear regression head for VB prediction.

## Difference from H2 Feature Experiments

H2 used engineered statistical/frequency features with RF/MLP. H3_S2 uses raw or low-processed time sequences and learns temporal features with an LSTM, so direct comparison should account for different input representations.

## Smoke Test

```json
{json.dumps(smoke_result, indent=2, ensure_ascii=False)}
```

## Data and Training Setup

- sensor_setting: {ctx.config['model'].get('sensor_setting', 'all_sensors')}
- segment modes: {ctx.config['runtime']['segment_modes']}
- process combinations: {ctx.config['runtime']['process_combinations']}
- shifts: {ctx.config['runtime']['shifts']}
- seeds: {ctx.config['runtime']['seeds']}
- max_epochs: {ctx.config['training'].get('max_epochs')}
- batch_size: {ctx.config['training'].get('batch_size')}
- dry_run: {dry_run}

## Performance Summary

```text
{best_table}
```

## Interpretation Notes

R2 < 0 means worse than predicting the target mean. If process information improves metrics, Time/Material shortcut risks must still be checked. LSTM performance may be limited by the small number of runs and strong domain shift; this is a DL baseline, not a replacement for engineered-feature RF/MLP experiments.

## Next Steps

Run the reduced grid first, then expand only if the A→B mini-run is stable. Consider `without_time_material` as the leakage-safer process setting.
"""
    (ctx.output_dir / "reports" / f"{PREFIX}_report.md").write_text(report, encoding="utf-8")
    html_report = "<!doctype html><html><head><meta charset='utf-8'><title>H3_S2 Report</title></head><body><pre>" + html.escape(report) + "</pre></body></html>"
    (ctx.output_dir / "reports" / f"{PREFIX}_report.html").write_text(html_report, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

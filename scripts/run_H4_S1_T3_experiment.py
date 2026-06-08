#!/usr/bin/env python3
"""H4_S1_T3: Input Signal Length Effect — LOCV Target Cases Restricted

Context spec: docs/agent_context/task_context/H4_S1_T3_context.md
Change vs T1: LOCV test cases limited to {1,2,5,8,9,12,14,16}.
              Training still uses all CASE_SCOPE cases (minus test case).

Output:
  experiments/executions/H4/S1/T3/{timestamp}_H4_S1_T3_locv_target_restricted/
    configs/H4_S1_T3_input_config.json
    metrics/H4_S1_T3_shift_metrics.csv
    analysis/H4_S1_T3_locv_aggregated.csv
    analysis/H4_S1_T3_100pct_per_case.csv
    logs/H4_S1_T3_run_log.txt
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

# ─── Experiment constants ──────────────────────────────────────────────────────

CASE_SCOPE = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
LOCV_TARGET_CASES = [1, 2, 5, 8, 9, 12, 14, 16]  # T3: restricted test cases
SENSORS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
PREFIX_PERCENTS = list(range(10, 101, 10))
SEED = 0
EXCLUDED_RUNS = {(2, 1), (12, 1)}
TOPIC = "locv_target_restricted"
PREFIX_LABEL = "H4_S1_T3"
THRESH = 1e6
TABULAR_MODELS = ["ridge", "random_forest", "xgboost", "svr", "mlp"]
META_FEATURES = ["DOC", "feed", "material"]

# ─── Feature extraction ────────────────────────────────────────────────────────

def parse_signal(value: object) -> np.ndarray:
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros(1)
    return {
        "mean": float(np.mean(arr)),
        "rms": float(np.sqrt(np.mean(arr ** 2))),
        "std": float(np.std(arr)),
        "peak": float(np.abs(arr).max()),
    }


def feature_names_for_sensors(sensors: list[str]) -> list[str]:
    return [f"{s}__{f}" for s in sensors for f in ["mean", "rms", "std", "peak"]]


# ─── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(process: pd.DataFrame) -> pd.DataFrame:
    df = process.copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run_per_case = df.groupby("case")["run"].transform("min")
    is_run1 = df["run"] == min_run_per_case
    df.loc[is_run1 & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside")
    )
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda row: (int(row["case"]), int(row["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)


# ─── Feature matrix builder ───────────────────────────────────────────────────

def build_feature_matrix(
    signal_df: pd.DataFrame,
    process_df: pd.DataFrame,
    prefix_percents: list[int],
    sensors: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []

    for row in process_df.itertuples(index=False):
        case_id = int(row.case)
        run_id = int(row.run)
        vb = float(row.VB) if not pd.isna(row.VB) else 0.0
        doc = float(getattr(row, "DOC", np.nan))
        feed = float(getattr(row, "feed", np.nan))
        material = int(getattr(row, "material", 0))

        sig_row = signal_df[(signal_df["case"] == case_id) & (signal_df["run"] == run_id)]
        if sig_row.empty:
            continue
        sig_row = sig_row.iloc[0]

        arrays = {s: parse_signal(sig_row[s]) for s in sensors}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        base_length = min(len(a) for a in arrays.values())

        for pct in prefix_percents:
            end_idx = max(1, int(np.ceil(base_length * pct / 100.0)))
            record: dict = {
                "case_id": case_id, "run": run_id,
                "signal_prefix_percent": float(pct),
                "VB": vb, "DOC": doc, "feed": feed, "material": material,
            }
            for s in sensors:
                feats = extract_features(arrays[s][:end_idx])
                for fname, fval in feats.items():
                    record[f"{s}__{fname}"] = fval
            rows.append(record)

    return pd.DataFrame(rows).sort_values(["signal_prefix_percent", "case_id", "run"]).reset_index(drop=True)


# ─── Models ───────────────────────────────────────────────────────────────────

def make_model(name: str) -> object:
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "xgboost":
        return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42, verbosity=0)
    if name == "svr":
        return SVR(kernel="rbf", C=10, epsilon=0.05)
    if name == "mlp":
        return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=SEED, learning_rate_init=0.001)
    raise ValueError(f"Unknown model: {name}")


# ─── GRU-DeltaMeta ────────────────────────────────────────────────────────────

class DeltaMetaGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out).squeeze(-1)


def build_delta_meta_sequences(
    feature_df: pd.DataFrame,
    signal_feat_cols: list[str],
    pct: float,
) -> dict[int, dict]:
    subset = feature_df[feature_df["signal_prefix_percent"] == float(pct)].copy()
    cases: dict[int, dict] = {}
    for case_id in subset["case_id"].unique():
        c_data = subset[subset["case_id"] == case_id].sort_values("run").reset_index(drop=True)
        raw = c_data[signal_feat_cols].to_numpy(dtype=np.float32)
        meta = c_data[META_FEATURES].to_numpy(dtype=np.float32)
        delta = raw - raw[0:1]
        seq = np.concatenate([delta, meta], axis=1)
        vb = c_data["VB"].to_numpy(dtype=np.float32)
        cases[int(case_id)] = {"seq": seq, "vb": vb, "n_runs": len(c_data)}
    return cases


def fit_predict_gru(
    train_cases: dict[int, dict],
    test_cases: dict[int, dict],
    input_dim: int,
    device: torch.device,
    epochs: int = 200,
    lr: float = 0.001,
) -> dict[int, np.ndarray]:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    model = DeltaMetaGRU(input_dim=input_dim, hidden_size=256, num_layers=3, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_seqs = [info["seq"] for info in train_cases.values()]
    train_vbs = [info["vb"] for info in train_cases.values()]
    all_vb = np.concatenate(train_vbs)
    y_mean = float(all_vb.mean())
    y_std = float(all_vb.std()) if all_vb.std() > 1e-8 else 1.0

    tensors = [torch.tensor(s, dtype=torch.float32) for s in train_seqs]
    lengths = torch.tensor([len(s) for s in train_seqs])
    x_train = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0).to(device)
    tgt_tensors = [torch.tensor((t - y_mean) / y_std, dtype=torch.float32) for t in train_vbs]
    y_train = nn.utils.rnn.pad_sequence(tgt_tensors, batch_first=True, padding_value=float("nan")).to(device)
    mask_train = ~torch.isnan(y_train)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train, lengths)
        loss = ((pred[mask_train] - y_train[mask_train]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    predictions: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for case_id, info in test_cases.items():
            seq_t = torch.tensor(info["seq"], dtype=torch.float32).unsqueeze(0).to(device)
            length_t = torch.tensor([info["n_runs"]])
            pred_norm = model(seq_t, length_t).squeeze(0).cpu().numpy()
            predictions[case_id] = pred_norm * y_std + y_mean
    return predictions


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# ─── LOCV evaluation (T3: only LOCV_TARGET_CASES as test) ────────────────────

def run_locv(
    feature_df: pd.DataFrame,
    signal_feat_cols: list[str],
    pct: float,
    model_name: str,
    log: list[str],
    device: torch.device,
) -> list[dict]:
    subset = feature_df[feature_df["signal_prefix_percent"] == float(pct)].copy()
    feature_cols = signal_feat_cols + META_FEATURES
    results: list[dict] = []

    if model_name == "feature_gru":
        input_dim = len(signal_feat_cols) + len(META_FEATURES)
        all_case_data = build_delta_meta_sequences(feature_df, signal_feat_cols, pct)

        for test_case in LOCV_TARGET_CASES:
            if test_case not in all_case_data:
                continue
            # Train on all CASE_SCOPE cases (minus test_case)
            train_case_ids = [c for c in CASE_SCOPE if c != test_case and c in all_case_data]
            train_data = {c: all_case_data[c] for c in train_case_ids}
            test_data = {test_case: all_case_data[test_case]}

            preds = fit_predict_gru(train_data, test_data, input_dim, device)
            y_pred = preds[test_case]
            y_true = all_case_data[test_case]["vb"]
            m = compute_metrics(y_true, y_pred)
            results.append({
                "signal_prefix_percent": pct, "model": model_name,
                "seed": SEED, "target_case": test_case, "n_runs": len(y_true), **m,
            })
            log.append(f"  pct={pct}% case={test_case} RMSE={m['rmse']:.4f} R2={m['r2']:.4f}")
        return results

    # Tabular models: test only on LOCV_TARGET_CASES, train on all CASE_SCOPE
    for test_case in LOCV_TARGET_CASES:
        train_df = subset[subset["case_id"] != test_case]
        test_df = subset[subset["case_id"] == test_case]
        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
        y_train = train_df["VB"].to_numpy(dtype=np.float64)
        X_test = test_df[feature_cols].to_numpy(dtype=np.float64)
        y_true = test_df["VB"].to_numpy(dtype=np.float64)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        mdl = make_model(model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl.fit(X_train_s, y_train)
            y_pred = mdl.predict(X_test_s)

        m = compute_metrics(y_true, y_pred)
        results.append({
            "signal_prefix_percent": pct, "model": model_name,
            "seed": SEED, "target_case": test_case, "n_runs": len(y_true), **m,
        })
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = ROOT / "experiments" / "executions" / "H4" / "S1" / "T3" / f"{timestamp}_H4_S1_T3_{TOPIC}"
    for subdir in ["configs", "metrics", "analysis", "logs"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H4_S1_T3 Experiment start ===")
    log(f"LOCV target cases: {LOCV_TARGET_CASES}")
    log(f"Training pool: {CASE_SCOPE} (all, minus test case)")
    log(f"Output dir: {output_dir}")

    log("Loading data...")
    signal_df = pd.read_csv(ROOT / "datasets/processed/mill_signal_data.csv",
                            usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/processed/mill_process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    log(f"Loaded {len(process_df)} process rows, {len(signal_df)} signal rows")

    log("Preprocessing...")
    process_clean = preprocess(process_df)
    log(f"After preprocessing: {len(process_clean)} rows")

    log("Building feature matrix...")
    feat_df = build_feature_matrix(signal_df, process_clean, PREFIX_PERCENTS, SENSORS)
    signal_feat_cols = feature_names_for_sensors(SENSORS)
    log(f"Feature matrix: {len(feat_df)} rows, {len(signal_feat_cols)} signal features")

    config = {
        "experiment": {"id": f"{timestamp}_{PREFIX_LABEL}", "context": "docs/agent_context/task_context/H4_S1_T3_context.md"},
        "data": {"case_scope": CASE_SCOPE, "locv_target_cases": LOCV_TARGET_CASES, "sensors": SENSORS, "excluded_runs": list(EXCLUDED_RUNS)},
        "features": {"names": ["mean", "rms", "std", "peak"], "n_signal_features": len(signal_feat_cols), "meta_features": META_FEATURES},
        "prefix_percents": PREFIX_PERCENTS,
        "models": TABULAR_MODELS + ["feature_gru"],
        "seed": SEED,
        "split": "leave_one_case_out (test cases restricted to LOCV_TARGET_CASES)",
    }
    (output_dir / "configs" / f"{PREFIX_LABEL}_input_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    all_models = TABULAR_MODELS + ["feature_gru"]
    all_results: list[dict] = []

    for model_name in all_models:
        log(f"\nRunning model: {model_name}")
        for pct in PREFIX_PERCENTS:
            try:
                results = run_locv(feat_df, signal_feat_cols, float(pct), model_name, log_lines, device)
                all_results.extend(results)
            except Exception as exc:
                log(f"  ERROR pct={pct}% model={model_name}: {exc}")

    metrics_df = pd.DataFrame(all_results)
    metrics_df.to_csv(output_dir / "metrics" / f"{PREFIX_LABEL}_shift_metrics.csv", index=False)
    log(f"\nSaved {len(metrics_df)} metric rows")

    agg = (
        metrics_df
        .groupby(["signal_prefix_percent", "model"])
        .agg(mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"), mean_mae=("mae", "mean"), n_folds=("target_case", "count"))
        .reset_index()
        .sort_values(["model", "signal_prefix_percent"])
    )
    agg.to_csv(output_dir / "analysis" / f"{PREFIX_LABEL}_locv_aggregated.csv", index=False)

    pct100 = metrics_df[metrics_df["signal_prefix_percent"] == 100.0].copy()
    pct100.to_csv(output_dir / "analysis" / f"{PREFIX_LABEL}_100pct_per_case.csv", index=False)

    log("\n=== SUMMARY: Mean RMSE by model at each prefix % ===")
    pivot = agg.pivot(index="signal_prefix_percent", columns="model", values="mean_rmse")
    log("\n" + pivot.to_string())

    log("\n=== SUMMARY: Mean RMSE by model at 100% prefix ===")
    summary100 = agg[agg["signal_prefix_percent"] == 100.0][["model", "mean_rmse", "mean_r2", "mean_mae"]]
    log("\n" + summary100.sort_values("mean_rmse").to_string(index=False))

    log("\n=== SUMMARY: Per-case RMSE at 100% prefix ===")
    per_case = pct100.sort_values(["model", "target_case"])[["model", "target_case", "rmse", "r2", "mae"]]
    log("\n" + per_case.to_string(index=False))

    (output_dir / "logs" / f"{PREFIX_LABEL}_run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nDone. Results in: {output_dir}")


if __name__ == "__main__":
    main()

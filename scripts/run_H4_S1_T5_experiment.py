#!/usr/bin/env python3
"""H4_S1_T5: Input Signal Length Effect — Raw+Delta+Meta, All Cases incl. Case 6

Context spec: docs/agent_context/task_context/H4_S1_T5_context.md
Change vs T2: Case 6 added to CASE_SCOPE (T2 excluded Case 6).
              Case 6 has only Run 1 (VB=0.0); delta features are all-zero for it.

Feature vector per sample (51):
  Raw   (24): mean/rms/std/peak × 6 sensors
  Delta (24): Δ from first available run per case
  Meta   (3): DOC, feed, material

Note: R² is undefined for Case 6 (single prediction point) → stored as NaN.

Output:
  experiments/executions/H4/S1/T5/{timestamp}_H4_S1_T5_with_case6/
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

# ─── Constants ────────────────────────────────────────────────────────────────

CASE_SCOPE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # T5: incl. Case 6
SENSORS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
PREFIX_PERCENTS = list(range(10, 101, 10))
SEED = 0
EXCLUDED_RUNS = {(2, 1), (12, 1)}  # Case 6 Run 1 is NOT excluded (it's the only run)
TOPIC = "with_case6"
PREFIX_LABEL = "H4_S1_T5"
META_FEATURES = ["DOC", "feed", "material"]
TABULAR_MODELS = ["ridge", "random_forest", "xgboost", "svr", "mlp"]
THRESH = 1e6

# ─── Feature extraction ───────────────────────────────────────────────────────

def parse_signal(value: object) -> np.ndarray:
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros(1)
    return np.array([
        np.mean(arr),
        np.sqrt(np.mean(arr ** 2)),
        np.std(arr),
        np.abs(arr).max(),
    ], dtype=np.float64)


def raw_col_names() -> list[str]:
    return [f"{s}__{f}" for s in SENSORS for f in ["mean", "rms", "std", "peak"]]


def delta_col_names() -> list[str]:
    return [f"{s}__delta_{f}" for s in SENSORS for f in ["mean", "rms", "std", "peak"]]


# ─── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(process: pd.DataFrame) -> pd.DataFrame:
    df = process.copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    is_run1 = df["run"] == min_run
    df.loc[is_run1 & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside")
    )
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)


# ─── Feature matrix (Raw + Delta + Meta) ─────────────────────────────────────

def build_feature_matrix(
    signal_df: pd.DataFrame,
    process_df: pd.DataFrame,
    prefix_percents: list[int],
    sensors: list[str],
) -> pd.DataFrame:
    raw_cols = raw_col_names()
    delta_cols = delta_col_names()

    cache: dict[tuple[int, int, int], np.ndarray] = {}
    for row in process_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        sig_row = signal_df[(signal_df["case"] == case_id) & (signal_df["run"] == run_id)]
        if sig_row.empty:
            continue
        sig_row = sig_row.iloc[0]
        arrays = {s: parse_signal(sig_row[s]) for s in sensors}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        base_len = min(len(a) for a in arrays.values())
        for pct in prefix_percents:
            end = max(1, int(np.ceil(base_len * pct / 100.0)))
            feats = np.concatenate([extract_features(arrays[s][:end]) for s in sensors])
            cache[(case_id, run_id, pct)] = feats

    first_run: dict[int, int] = {}
    for case_id in process_df["case"].unique():
        runs = sorted(r for (c, r, _) in cache if c == case_id and (c, r) not in EXCLUDED_RUNS)
        if runs:
            first_run[int(case_id)] = runs[0]

    rows: list[dict] = []
    for row in process_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        if (case_id, run_id) in EXCLUDED_RUNS:
            continue
        ref_run = first_run.get(case_id, run_id)
        vb = float(row.VB) if not pd.isna(row.VB) else 0.0
        doc = float(getattr(row, "DOC", 0.0))
        feed = float(getattr(row, "feed", 0.0))
        material = int(getattr(row, "material", 0))

        for pct in prefix_percents:
            raw_feats = cache.get((case_id, run_id, pct))
            ref_feats = cache.get((case_id, ref_run, pct))
            if raw_feats is None or ref_feats is None:
                continue
            delta_feats = raw_feats - ref_feats
            record: dict = {
                "case_id": case_id, "run": run_id,
                "signal_prefix_percent": float(pct),
                "VB": vb, "DOC": doc, "feed": feed, "material": material,
            }
            for col, val in zip(raw_cols, raw_feats):
                record[col] = float(val)
            for col, val in zip(delta_cols, delta_feats):
                record[col] = float(val)
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


# ─── GRU ──────────────────────────────────────────────────────────────────────

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


def build_gru_sequences(
    feature_df: pd.DataFrame,
    raw_cols: list[str],
    delta_cols: list[str],
    pct: float,
) -> dict[int, dict]:
    subset = feature_df[feature_df["signal_prefix_percent"] == float(pct)].copy()
    cases: dict[int, dict] = {}
    all_feat_cols = raw_cols + delta_cols + META_FEATURES
    for case_id in subset["case_id"].unique():
        c_data = subset[subset["case_id"] == case_id].sort_values("run").reset_index(drop=True)
        seq = c_data[all_feat_cols].to_numpy(dtype=np.float32)
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
    y_mean, y_std = float(all_vb.mean()), float(all_vb.std())
    if y_std < 1e-8:
        y_std = 1.0

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
            len_t = torch.tensor([info["n_runs"]])
            pred_norm = model(seq_t, len_t).squeeze(0).cpu().numpy()
            predictions[case_id] = pred_norm * y_std + y_mean
    return predictions


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except ValueError:
        r2 = float("nan")  # undefined for single-point cases (e.g. Case 6)
    return {"rmse": rmse, "r2": r2, "mae": mae}


# ─── LOCV ─────────────────────────────────────────────────────────────────────

def run_locv(
    feature_df: pd.DataFrame,
    raw_cols: list[str],
    delta_cols: list[str],
    pct: float,
    model_name: str,
    device: torch.device,
) -> list[dict]:
    all_feat_cols = raw_cols + delta_cols + META_FEATURES
    subset = feature_df[feature_df["signal_prefix_percent"] == float(pct)].copy()
    results: list[dict] = []

    if model_name == "feature_gru":
        input_dim = len(all_feat_cols)
        case_data = build_gru_sequences(feature_df, raw_cols, delta_cols, pct)
        for test_case in CASE_SCOPE:
            if test_case not in case_data:
                continue
            train_data = {c: case_data[c] for c in CASE_SCOPE if c != test_case and c in case_data}
            preds = fit_predict_gru(train_data, {test_case: case_data[test_case]}, input_dim, device)
            y_pred = preds[test_case]
            y_true = case_data[test_case]["vb"]
            m = compute_metrics(y_true, y_pred)
            results.append({"signal_prefix_percent": pct, "model": model_name,
                            "seed": SEED, "target_case": test_case, "n_runs": len(y_true), **m})
        return results

    for test_case in CASE_SCOPE:
        train_df = subset[subset["case_id"] != test_case]
        test_df = subset[subset["case_id"] == test_case]
        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[all_feat_cols].to_numpy(dtype=np.float64)
        y_train = train_df["VB"].to_numpy(dtype=np.float64)
        X_test = test_df[all_feat_cols].to_numpy(dtype=np.float64)
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
        results.append({"signal_prefix_percent": pct, "model": model_name,
                        "seed": SEED, "target_case": test_case, "n_runs": len(y_true), **m})
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = ROOT / "experiments" / "executions" / "H4" / "S1" / "T5" / f"{timestamp}_H4_S1_T5_{TOPIC}"
    for subdir in ["configs", "metrics", "analysis", "logs"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H4_S1_T5 Experiment start (Raw+Delta+Meta, Case 6 included) ===")
    log(f"CASE_SCOPE: {CASE_SCOPE}  ({len(CASE_SCOPE)} cases)")
    log(f"Note: Case 6 has only Run 1 (VB=0.0); R² for Case 6 will be NaN.")
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

    log("Building Raw+Delta+Meta feature matrix...")
    raw_cols = raw_col_names()
    delta_cols = delta_col_names()
    feat_df = build_feature_matrix(signal_df, process_clean, PREFIX_PERCENTS, SENSORS)
    n_case6 = len(feat_df[feat_df["case_id"] == 6])
    log(f"Feature matrix: {len(feat_df)} rows total, Case 6 rows: {n_case6}")

    config = {
        "experiment": {"id": f"{timestamp}_{PREFIX_LABEL}", "context": "docs/agent_context/task_context/H4_S1_T5_context.md"},
        "change_vs_T2": "Case 6 included in CASE_SCOPE (T2 excluded it). Case 6: 1 run, VB=0.0.",
        "data": {"case_scope": CASE_SCOPE, "sensors": SENSORS, "excluded_runs": list(EXCLUDED_RUNS)},
        "features": {"raw": 24, "delta": 24, "meta": 3, "total": 51},
        "prefix_percents": PREFIX_PERCENTS,
        "models": TABULAR_MODELS + ["feature_gru"],
        "seed": SEED,
        "split": "leave_one_case_out",
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
                results = run_locv(feat_df, raw_cols, delta_cols, float(pct), model_name, device)
                all_results.extend(results)
            except Exception as exc:
                log(f"  ERROR pct={pct}% model={model_name}: {exc}")

    metrics_df = pd.DataFrame(all_results)
    metrics_df.to_csv(output_dir / "metrics" / f"{PREFIX_LABEL}_shift_metrics.csv", index=False)
    log(f"\nSaved {len(metrics_df)} metric rows")

    # Aggregate excluding Case 6 from mean (single-point R² is NaN)
    agg_all = (
        metrics_df
        .groupby(["signal_prefix_percent", "model"])
        .agg(mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"), mean_mae=("mae", "mean"), n_folds=("target_case", "count"))
        .reset_index()
        .sort_values(["model", "signal_prefix_percent"])
    )
    agg_all.to_csv(output_dir / "analysis" / f"{PREFIX_LABEL}_locv_aggregated.csv", index=False)

    # Aggregate excluding Case 6 (for fair comparison with T2)
    agg_no6 = (
        metrics_df[metrics_df["target_case"] != 6]
        .groupby(["signal_prefix_percent", "model"])
        .agg(mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"), mean_mae=("mae", "mean"), n_folds=("target_case", "count"))
        .reset_index()
        .sort_values(["model", "signal_prefix_percent"])
    )
    agg_no6.to_csv(output_dir / "analysis" / f"{PREFIX_LABEL}_locv_aggregated_excl_case6.csv", index=False)

    pct100 = metrics_df[metrics_df["signal_prefix_percent"] == 100.0].copy()
    pct100.to_csv(output_dir / "analysis" / f"{PREFIX_LABEL}_100pct_per_case.csv", index=False)

    log("\n=== SUMMARY: Mean RMSE at 100% (all 16 cases) ===")
    s100_all = agg_all[agg_all["signal_prefix_percent"] == 100.0][["model", "mean_rmse", "mean_r2", "mean_mae"]]
    log("\n" + s100_all.sort_values("mean_rmse").to_string(index=False))

    log("\n=== SUMMARY: Mean RMSE at 100% (excl. Case 6, for T2 comparison) ===")
    s100_no6 = agg_no6[agg_no6["signal_prefix_percent"] == 100.0][["model", "mean_rmse", "mean_r2", "mean_mae"]]
    log("\n" + s100_no6.sort_values("mean_rmse").to_string(index=False))

    log("\n=== Case 6 predictions at 100% ===")
    c6_100 = pct100[pct100["target_case"] == 6][["model", "rmse", "r2", "mae", "n_runs"]]
    log("\n" + c6_100.sort_values("rmse").to_string(index=False))

    log("\n=== SUMMARY: Mean RMSE by model at each prefix % (all 16 cases) ===")
    pivot = agg_all.pivot(index="signal_prefix_percent", columns="model", values="mean_rmse")
    log("\n" + pivot.to_string())

    (output_dir / "logs" / f"{PREFIX_LABEL}_run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nDone. Results in: {output_dir}")


if __name__ == "__main__":
    main()

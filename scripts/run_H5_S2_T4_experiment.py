#!/usr/bin/env python3
"""H5_S2_T4: Multi-seed statistical significance for per-case Top-1 models

For each case, take the Top-1 model+configuration from the H4 per-case
leaderboard (`leaderboards/h4/per_case/top5.csv`) and run with 5 random
seeds (0–4).
Deterministic models (SVR, Ridge) are included with their H4 seed=0 result.

Sources:
  Case 1  → H4_S3_T4: XGBoost, Raw+Delta+Meta(51), noload_excl_prefix, 80%
  Case 2  → H4_S1_T6: XGBoost, Raw(24),             prefix,             90%
  Case 3  → H4_S1_T4: feature_gru, Delta+Meta(27),  prefix,             50%
  Case 4  → H4_S1_T1: XGBoost, Raw+Meta(27),        prefix,             90%
  Case 5  → H4_S1_T4: feature_gru, Delta+Meta(27),  prefix,             30%
  Case 6  → H4_S1_T5: XGBoost, Raw+Delta+Meta(51),  prefix_case6,      100%
  Case 7  → H4_S1_T4: feature_gru, Delta+Meta(27),  prefix,             20%
  Case 8  → H4_S1_T4: RandomForest, Delta+Meta(27), prefix,             10%
  Case 9  → H4_S1_T4: MLP, Delta+Meta(27),          prefix,            100%
  Case 10 → H4_S1_T4: feature_gru, Delta+Meta(27),  prefix,             80%
  Case 11 → H4_S1_T4: feature_gru, Delta+Meta(27),  prefix,             60%
  Case 12 → H4_S1_T1: SVR (DETERMINISTIC), Raw+Meta(27), prefix,        70%
  Case 13 → H4_S1_T4: feature_gru, Delta+Meta(27),  prefix,             70%
  Case 14 → H4_S1_T4: feature_gru, Delta+Meta(27),  prefix,             80%
  Case 15 → H4_S1_T4: feature_gru, Delta+Meta(27),  prefix,             60%
  Case 16 → H4_S1_T7: Ridge (DETERMINISTIC), Delta(24), prefix,          80%

Output:
  experiments/executions/H5/S2/T4/{timestamp}_H5_S2_T4_multi_seed_top1/
"""
from __future__ import annotations

import json
import math
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

# ─── Constants ───────────────────────────────────────────────────────────────

SENSORS       = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
CASE_SCOPE_C6 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # includes Case 6
EXCLUDED_RUNS = {(2, 1), (12, 1)}
META_COLS     = ["DOC", "feed", "material"]
THRESH        = 1e6
SEEDS         = [0, 1, 2, 3, 4]

# Per-case Top-1 configuration (from leaderboards/h4/per_case/top5.csv, case_rank=1)
TOP1 = {
    1:  {"model": "xgboost",      "feature": "raw+delta+meta", "pct": 80.0,  "scenario": "s3_noload",    "deterministic": False,
         "h4": {"rmse": 0.044626, "r2": 0.919187, "mae": 0.039144}},
    2:  {"model": "xgboost",      "feature": "raw",            "pct": 90.0,  "scenario": "s1_prefix",    "deterministic": False,
         "h4": {"rmse": 0.032673, "r2": 0.944160, "mae": 0.028680}},
    3:  {"model": "feature_gru",  "feature": "delta+meta",     "pct": 50.0,  "scenario": "s1_prefix",    "deterministic": False,
         "h4": {"rmse": 0.039802, "r2": 0.911357, "mae": 0.033097}},
    4:  {"model": "xgboost",      "feature": "raw+meta",       "pct": 90.0,  "scenario": "s1_prefix",    "deterministic": False,
         "h4": {"rmse": 0.040236, "r2": 0.914922, "mae": 0.029942}},
    5:  {"model": "feature_gru",  "feature": "delta+meta",     "pct": 30.0,  "scenario": "s1_prefix",    "deterministic": False,
         "h4": {"rmse": 0.025662, "r2": 0.988845, "mae": 0.021852}},
    6:  {"model": "xgboost",      "feature": "raw+delta+meta", "pct": 100.0, "scenario": "s1_prefix_c6", "deterministic": False,
         "h4": {"rmse": 0.006156, "r2": None,      "mae": 0.006156}},
    7:  {"model": "feature_gru",  "feature": "delta+meta",     "pct": 20.0,  "scenario": "s1_prefix",    "deterministic": False,
         "h4": {"rmse": 0.027818, "r2": 0.962717, "mae": 0.023415}},
    8:  {"model": "random_forest","feature": "delta+meta",     "pct": 10.0,  "scenario": "s1_prefix",    "deterministic": False,
         "h4": {"rmse": 0.051962, "r2": 0.929343, "mae": 0.047151}},
    9:  {"model": "mlp",          "feature": "delta+meta",     "pct": 100.0, "scenario": "s1_prefix",    "deterministic": False,
         "h4": {"rmse": 0.062562, "r2": 0.938156, "mae": 0.053876}},
    10: {"model": "feature_gru",  "feature": "delta+meta",     "pct": 80.0,  "scenario": "s1_prefix",    "deterministic": False,
         "h4": {"rmse": 0.047152, "r2": 0.954660, "mae": 0.041780}},
    11: {"model": "feature_gru",  "feature": "delta+meta",     "pct": 60.0,  "scenario": "s1_prefix",    "deterministic": False,
         "h4": {"rmse": 0.031069, "r2": 0.979431, "mae": 0.024672}},
    12: {"model": "svr",          "feature": "raw+meta",       "pct": 70.0,  "scenario": "s1_prefix",    "deterministic": True,
         "h4": {"rmse": 0.046853, "r2": 0.936932, "mae": 0.039931}},
    13: {"model": "feature_gru",  "feature": "delta+meta",     "pct": 70.0,  "scenario": "s1_prefix",    "deterministic": False,
         "h4": {"rmse": 0.096915, "r2": 0.955718, "mae": 0.062844}},
    14: {"model": "feature_gru",  "feature": "delta+meta",     "pct": 80.0,  "scenario": "s1_prefix",    "deterministic": False,
         "h4": {"rmse": 0.093156, "r2": 0.929180, "mae": 0.071691}},
    15: {"model": "feature_gru",  "feature": "delta+meta",     "pct": 60.0,  "scenario": "s1_prefix",    "deterministic": False,
         "h4": {"rmse": 0.017991, "r2": 0.993529, "mae": 0.014207}},
    16: {"model": "ridge",        "feature": "delta",          "pct": 80.0,  "scenario": "s1_prefix",    "deterministic": True,
         "h4": {"rmse": 0.032064, "r2": 0.973987, "mae": 0.025736}},
}


# ─── Logging ─────────────────────────────────────────────────────────────────

_log_lines: list[str] = []

def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _log_lines.append(line)


# ─── Feature extraction ───────────────────────────────────────────────────────

def parse_signal(v: object) -> np.ndarray:
    a = np.fromstring(str(v).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def extract_4stats(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        a = np.zeros(1)
    return np.array([np.mean(a), np.sqrt(np.mean(a**2)), np.std(a), np.abs(a).max()], dtype=np.float64)


def s1_raw_col_names() -> list[str]:
    return [f"{s}__{f}" for s in SENSORS for f in ["mean", "rms", "std", "peak"]]


def s1_delta_col_names() -> list[str]:
    return [f"{s}__delta_{f}" for s in SENSORS for f in ["mean", "rms", "std", "peak"]]


def s3_raw_col_names() -> list[str]:
    return [f"{s}__{f}" for s in SENSORS for f in ["mean", "rms", "std", "peak"]]


def s3_delta_col_names() -> list[str]:
    return [f"delta_{s}__{f}" for s in SENSORS for f in ["mean", "rms", "std", "peak"]]


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(proc: pd.DataFrame, case_scope: list[int]) -> pd.DataFrame:
    df = proc[proc["case"].isin(case_scope)].copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"] == min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside")
    )
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)


# ─── Feature matrix builders ─────────────────────────────────────────────────

def build_s1_feature_matrix(
    signal_df: pd.DataFrame,
    proc_df: pd.DataFrame,
    pcts: list[float],
    feature_type: str,
) -> pd.DataFrame:
    """S1-prefix style: prefix of full signal, various feature types."""
    raw_cols   = s1_raw_col_names()
    delta_cols = s1_delta_col_names()

    cache: dict[tuple[int, int, float], np.ndarray] = {}
    for row in proc_df.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        sig = signal_df[(signal_df["case"] == cid) & (signal_df["run"] == rid)]
        if sig.empty:
            continue
        sig = sig.iloc[0]
        arrays = {s: parse_signal(sig[s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        base_len = min(len(a) for a in arrays.values())
        for pct in pcts:
            end = max(1, math.ceil(base_len * pct / 100.0))
            feats = np.concatenate([extract_4stats(arrays[s][:end]) for s in SENSORS])
            cache[(cid, rid, pct)] = feats

    first_run: dict[tuple[int, float], np.ndarray] = {}
    for cid in proc_df["case"].unique():
        for pct in pcts:
            runs_sorted = sorted(
                r for (c, r, p) in cache if c == int(cid) and p == pct and (int(cid), r) not in EXCLUDED_RUNS
            )
            if runs_sorted:
                first_run[(int(cid), pct)] = cache[(int(cid), runs_sorted[0], pct)]

    rows: list[dict] = []
    for row in proc_df.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        if (cid, rid) in EXCLUDED_RUNS:
            continue
        vb  = float(row.VB) if not pd.isna(row.VB) else 0.0
        doc = float(getattr(row, "DOC", 0.0))
        feed = float(getattr(row, "feed", 0.0))
        mat  = int(getattr(row, "material", 0))

        for pct in pcts:
            rf  = cache.get((cid, rid, pct))
            ref = first_run.get((cid, pct))
            if rf is None or ref is None:
                continue
            delta = rf - ref

            rec: dict = {"case_id": cid, "run": rid, "pct": pct,
                         "VB": vb, "DOC": doc, "feed": feed, "material": mat}
            for col, val in zip(raw_cols, rf):
                rec[col] = float(val)
            for col, val in zip(delta_cols, delta):
                rec[col] = float(val)
            rows.append(rec)

    return pd.DataFrame(rows).sort_values(["pct", "case_id", "run"]).reset_index(drop=True)


def build_s3_feature_matrix(
    signal_df: pd.DataFrame,
    proc_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    ratios: list[float],
) -> pd.DataFrame:
    """S3_T4-style: Entry+Steady+Exit (no-load excluded) then prefix ratio."""
    raw_cols   = s3_raw_col_names()
    delta_cols = s3_delta_col_names()
    META_COLS_LOCAL = ["DOC", "feed", "material"]

    merged = proc_df.merge(
        seg_df[["case", "run", "idx_noload_end", "signal_length"]],
        on=["case", "run"], how="inner"
    )

    cache: dict[tuple[int, int, float], np.ndarray] = {}
    for row in merged.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        cut_start = int(row.idx_noload_end)
        sig = signal_df[(signal_df["case"] == cid) & (signal_df["run"] == rid)]
        if sig.empty:
            continue
        sig = sig.iloc[0]
        arrays = {s: parse_signal(sig[s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        cut_len = min(len(a) - cut_start for a in arrays.values())
        if cut_len <= 0:
            continue
        for ratio in ratios:
            end_offset = max(1, math.ceil(cut_len * ratio / 100.0))
            sliced = {s: arrays[s][cut_start: cut_start + end_offset] for s in SENSORS}
            feats = np.concatenate([extract_4stats(sliced[s]) for s in SENSORS])
            cache[(cid, rid, ratio)] = feats

    first_cache: dict[tuple[int, float], np.ndarray] = {}
    for ratio in ratios:
        for cid in merged["case"].unique():
            cd = merged[merged["case"] == int(cid)].sort_values("run")
            for r in cd.itertuples(index=False):
                rid = int(r.run)
                if (int(cid), rid) in EXCLUDED_RUNS:
                    continue
                key = (int(cid), rid, ratio)
                if key in cache:
                    first_cache[(int(cid), ratio)] = cache[key]
                    break

    rows: list[dict] = []
    for row in merged.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        vb  = float(row.VB)
        doc = float(row.DOC)
        feed = float(row.feed)
        mat  = int(row.material)
        for ratio in ratios:
            rf  = cache.get((cid, rid, ratio))
            ref = first_cache.get((cid, ratio))
            if rf is None or ref is None:
                continue
            delta = rf - ref
            rec: dict = {"case_id": cid, "run": rid, "pct": ratio,
                         "VB": vb, "DOC": doc, "feed": feed, "material": mat}
            for col, val in zip(raw_cols, rf):
                rec[col] = float(val)
            for col, val in zip(delta_cols, delta):
                rec[col] = float(val)
            rows.append(rec)

    return pd.DataFrame(rows).sort_values(["pct", "case_id", "run"]).reset_index(drop=True)


# ─── Feature column selector ─────────────────────────────────────────────────

def get_feature_cols(feature_type: str, scenario: str) -> list[str]:
    if scenario == "s3_noload":
        raw   = s3_raw_col_names()
        delta = s3_delta_col_names()
    else:
        raw   = s1_raw_col_names()
        delta = s1_delta_col_names()
    if feature_type == "raw":
        return raw
    if feature_type == "delta":
        return delta
    if feature_type == "raw+meta":
        return raw + META_COLS
    if feature_type == "delta+meta":
        return delta + META_COLS
    if feature_type == "raw+delta+meta":
        return raw + delta + META_COLS
    raise ValueError(f"Unknown feature_type: {feature_type}")


# ─── GRU model ───────────────────────────────────────────────────────────────

class DeltaMetaGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 256,
                 num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out).squeeze(-1)


def build_gru_sequences(
    feat_df: pd.DataFrame,
    feat_cols: list[str],
    pct: float,
) -> dict[int, dict]:
    subset = feat_df[feat_df["pct"] == float(pct)].copy()
    cases: dict[int, dict] = {}
    for cid in subset["case_id"].unique():
        cd = subset[subset["case_id"] == int(cid)].sort_values("run").reset_index(drop=True)
        seq = cd[feat_cols].to_numpy(dtype=np.float32)
        vb  = cd["VB"].to_numpy(dtype=np.float32)
        cases[int(cid)] = {"seq": seq, "vb": vb, "n_runs": len(cd)}
    return cases


def fit_predict_gru(
    train_cases: dict[int, dict],
    test_cases: dict[int, dict],
    input_dim: int,
    device: torch.device,
    seed: int,
    epochs: int = 200,
    lr: float = 0.001,
) -> dict[int, np.ndarray]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = DeltaMetaGRU(input_dim=input_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    seqs  = [info["seq"] for info in train_cases.values()]
    vbs   = [info["vb"]  for info in train_cases.values()]
    all_vb = np.concatenate(vbs)
    ym = float(all_vb.mean())
    ys = float(all_vb.std()) if all_vb.std() > 1e-8 else 1.0

    tensors = [torch.tensor(s, dtype=torch.float32) for s in seqs]
    lengths = torch.tensor([len(s) for s in seqs])
    x_tr = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0).to(device)
    tgts  = [torch.tensor((t - ym) / ys, dtype=torch.float32) for t in vbs]
    y_tr  = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=float("nan")).to(device)
    mask  = ~torch.isnan(y_tr)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_tr, lengths)
        loss = ((pred[mask] - y_tr[mask]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    model.eval()
    preds: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for cid, info in test_cases.items():
            xt = torch.tensor(info["seq"], dtype=torch.float32).unsqueeze(0).to(device)
            lt = torch.tensor([info["n_runs"]])
            pred_norm = model(xt, lt).squeeze(0).cpu().numpy()
            preds[int(cid)] = pred_norm * ys + ym
    return preds


# ─── Tabular model factory ────────────────────────────────────────────────────

def make_tabular_model(model_name: str, seed: int) -> object:
    if model_name == "ridge":
        return Ridge(alpha=1.0)
    if model_name == "random_forest":
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    if model_name == "xgboost":
        return XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
            random_state=seed, verbosity=0
        )
    if model_name == "svr":
        return SVR(kernel="rbf", C=10, epsilon=0.05)
    if model_name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=300,
            random_state=seed, learning_rate_init=0.001
        )
    raise ValueError(f"Unknown model: {model_name}")


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) < 2:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float("nan"),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
    }


# ─── Single LOCV fold ─────────────────────────────────────────────────────────

def run_locv_fold(
    feat_df: pd.DataFrame,
    case_scope: list[int],
    target_case: int,
    model_name: str,
    feature_type: str,
    scenario: str,
    pct: float,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    feat_cols = get_feature_cols(feature_type, scenario)
    subset = feat_df[feat_df["pct"] == float(pct)].copy()

    if model_name == "feature_gru":
        gru_feat_cols = feat_cols  # delta+meta only for GRU
        case_data = build_gru_sequences(feat_df, gru_feat_cols, pct)
        if target_case not in case_data:
            raise ValueError(f"Target case {target_case} not in feature data")
        train_data = {c: case_data[c] for c in case_scope if c != target_case and c in case_data}
        test_data  = {target_case: case_data[target_case]}
        input_dim  = len(gru_feat_cols)
        preds = fit_predict_gru(train_data, test_data, input_dim, device, seed)
        y_true = case_data[target_case]["vb"]
        y_pred = preds[target_case]
        return compute_metrics(y_true, y_pred)

    # Tabular model
    train_df = subset[subset["case_id"] != target_case]
    test_df  = subset[subset["case_id"] == target_case]
    if train_df.empty or test_df.empty:
        raise ValueError(f"Empty split for case {target_case}")
    X_train = train_df[feat_cols].to_numpy(dtype=np.float64)
    y_train = train_df["VB"].to_numpy(dtype=np.float64)
    X_test  = test_df[feat_cols].to_numpy(dtype=np.float64)
    y_true  = test_df["VB"].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    mdl = make_tabular_model(model_name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl.fit(X_train_s, y_train)
        y_pred = mdl.predict(X_test_s)

    return compute_metrics(y_true, y_pred)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ts  = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT / "experiments" / "executions" / "H5" / "S2" / "T4" / f"{ts}_H5_S2_T4_multi_seed_top1"
    for d in ["configs", "metrics", "analysis", "logs"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    log(f"=== H5_S2_T4: Multi-seed Top-1 Model Validation ===")
    log(f"Seeds: {SEEDS}")
    log(f"Cases: {sorted(TOP1.keys())}")

    # ── Load data ──────────────────────────────────────────────────────────────
    proc_path = ROOT / "datasets" / "processed" / "mill_process_info.csv"
    sig_path  = ROOT / "datasets" / "processed" / "mill_signal_data.csv"
    seg_path  = ROOT / "datasets" / "cutting_segment" / "seg_heuristic.csv"

    log("Loading datasets...")
    proc_raw = pd.read_csv(proc_path)
    sig_df   = pd.read_csv(sig_path)
    seg_df   = pd.read_csv(seg_path, usecols=["case", "run", "idx_noload_end", "idx_start", "idx_end", "signal_length"])

    # ── Preprocess ─────────────────────────────────────────────────────────────
    proc_s1   = preprocess(proc_raw, CASE_SCOPE)
    proc_s1c6 = preprocess(proc_raw, CASE_SCOPE_C6)  # includes case 6
    log(f"Preprocessed rows: S1={len(proc_s1)}, S1+C6={len(proc_s1c6)}")

    # ── Collect required pcts per scenario ────────────────────────────────────
    pcts_s1   = sorted({cfg["pct"] for cid, cfg in TOP1.items() if cfg["scenario"] == "s1_prefix"})
    pcts_s1c6 = sorted({cfg["pct"] for cid, cfg in TOP1.items() if cfg["scenario"] == "s1_prefix_c6"})
    pcts_s3   = sorted({cfg["pct"] for cid, cfg in TOP1.items() if cfg["scenario"] == "s3_noload"})

    # ── Build feature matrices ────────────────────────────────────────────────
    log(f"Building S1-prefix feature matrix (pcts={pcts_s1})...")
    feat_s1 = build_s1_feature_matrix(sig_df, proc_s1, pcts_s1, "both")

    feat_s1c6 = None
    if pcts_s1c6:
        log(f"Building S1-prefix+Case6 feature matrix (pcts={pcts_s1c6})...")
        feat_s1c6 = build_s1_feature_matrix(sig_df, proc_s1c6, pcts_s1c6, "both")

    feat_s3 = None
    if pcts_s3:
        log(f"Building S3-noload feature matrix (pcts={pcts_s3})...")
        proc_s3 = preprocess(proc_raw, CASE_SCOPE)
        feat_s3 = build_s3_feature_matrix(sig_df, proc_s3, seg_df, pcts_s3)

    log(f"Feature rows: S1={len(feat_s1)}, "
        f"S1+C6={len(feat_s1c6) if feat_s1c6 is not None else 0}, "
        f"S3={len(feat_s3) if feat_s3 is not None else 0}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ── Gather scope per scenario ─────────────────────────────────────────────
    scope_map = {
        "s1_prefix":    CASE_SCOPE,
        "s1_prefix_c6": CASE_SCOPE_C6,
        "s3_noload":    CASE_SCOPE,
    }
    feat_map = {
        "s1_prefix":    feat_s1,
        "s1_prefix_c6": feat_s1c6,
        "s3_noload":    feat_s3,
    }

    # ── Run experiments ───────────────────────────────────────────────────────
    seed_results: list[dict] = []

    # Deterministic cases: include once (seed=0 equivalent = H4 result)
    for cid, cfg in sorted(TOP1.items()):
        if not cfg["deterministic"]:
            continue
        h4 = cfg["h4"]
        log(f"  Case {cid:2d} [{cfg['model']:15s}] DETERMINISTIC  "
            f"RMSE={h4['rmse']:.4f}  R2={h4['r2']:.4f}  MAE={h4['mae']:.4f}")
        seed_results.append({
            "target_case": cid,
            "model": cfg["model"],
            "feature": cfg["feature"],
            "pct": cfg["pct"],
            "scenario": cfg["scenario"],
            "seed": "deterministic",
            "rmse": h4["rmse"],
            "r2": h4["r2"],
            "mae": h4["mae"],
            "n_seeds": 1,
            "is_deterministic": True,
        })

    # Non-deterministic cases: run 5 seeds
    stochastic_cases = {cid: cfg for cid, cfg in TOP1.items() if not cfg["deterministic"]}
    n_stochastic = len(stochastic_cases)
    total_runs = n_stochastic * len(SEEDS)
    log(f"\nRunning {n_stochastic} stochastic cases × {len(SEEDS)} seeds = {total_runs} experiments")

    per_case_per_seed: dict[int, list[dict]] = {cid: [] for cid in stochastic_cases}

    for seed_idx, seed in enumerate(SEEDS):
        log(f"\n--- Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ---")
        for cid, cfg in sorted(stochastic_cases.items()):
            scenario = cfg["scenario"]
            feat_df_cur = feat_map.get(scenario)
            scope_cur   = scope_map.get(scenario, CASE_SCOPE)
            if feat_df_cur is None:
                log(f"  Case {cid}: no feature data for scenario={scenario}, skipping")
                continue

            try:
                m = run_locv_fold(
                    feat_df=feat_df_cur,
                    case_scope=scope_cur,
                    target_case=cid,
                    model_name=cfg["model"],
                    feature_type=cfg["feature"],
                    scenario=scenario,
                    pct=cfg["pct"],
                    seed=seed,
                    device=device,
                )
                per_case_per_seed[cid].append({"seed": seed, **m})
                log(f"  Case {cid:2d} [{cfg['model']:15s}] pct={cfg['pct']:5.1f}%  "
                    f"seed={seed}  RMSE={m['rmse']:.4f}  R2={m.get('r2', float('nan')):.4f}")
            except Exception as exc:
                log(f"  Case {cid:2d} ERROR seed={seed}: {exc}")
                per_case_per_seed[cid].append({"seed": seed, "rmse": float("nan"),
                                                "r2": float("nan"), "mae": float("nan")})

    # ── Aggregate stochastic results ──────────────────────────────────────────
    log("\n=== Final Results (5-seed average) ===")
    for cid, cfg in sorted(stochastic_cases.items()):
        runs = per_case_per_seed[cid]
        rmse_vals = [r["rmse"] for r in runs if not np.isnan(r["rmse"])]
        r2_vals   = [r["r2"]   for r in runs if not np.isnan(r.get("r2", float("nan")))]
        mae_vals  = [r["mae"]  for r in runs if not np.isnan(r["mae"])]
        if not rmse_vals:
            log(f"  Case {cid:2d}: no valid results")
            continue
        avg_rmse = float(np.mean(rmse_vals))
        avg_r2   = float(np.mean(r2_vals)) if r2_vals else float("nan")
        avg_mae  = float(np.mean(mae_vals))
        std_rmse = float(np.std(rmse_vals)) if len(rmse_vals) > 1 else 0.0
        h4_rmse  = cfg["h4"]["rmse"]
        delta    = avg_rmse - h4_rmse
        log(f"  Case {cid:2d} [{cfg['model']:15s}] pct={cfg['pct']:5.1f}%  "
            f"avg_RMSE={avg_rmse:.4f}±{std_rmse:.4f}  H4={h4_rmse:.4f}  Δ={delta:+.4f}")
        seed_results.append({
            "target_case": cid,
            "model": cfg["model"],
            "feature": cfg["feature"],
            "pct": cfg["pct"],
            "scenario": cfg["scenario"],
            "seed": "avg_5",
            "rmse": avg_rmse,
            "rmse_std": std_rmse,
            "r2": avg_r2,
            "mae": avg_mae,
            "n_seeds": len(rmse_vals),
            "is_deterministic": False,
            "h4_rmse": h4_rmse,
            "rmse_delta": delta,
        })

    # ── Save per-seed raw results ─────────────────────────────────────────────
    raw_rows: list[dict] = []
    for cid, cfg in sorted(stochastic_cases.items()):
        for r in per_case_per_seed[cid]:
            raw_rows.append({
                "target_case": cid, "model": cfg["model"],
                "feature": cfg["feature"], "pct": cfg["pct"],
                "scenario": cfg["scenario"], **r,
            })
    pd.DataFrame(raw_rows).to_csv(out / "metrics" / "H5_S2_T4_per_seed_raw.csv", index=False)

    # ── Save final summary ────────────────────────────────────────────────────
    summary_df = pd.DataFrame(seed_results).sort_values("target_case")
    summary_df.to_csv(out / "metrics" / "H5_S2_T4_results.csv", index=False)

    # ── Save config ───────────────────────────────────────────────────────────
    config = {
        "experiment": {"id": f"{ts}_H5_S2_T4", "context": "docs/agent_context/task_context/H5_S2_T4_context.md"},
        "seeds": SEEDS,
        "case_scope": CASE_SCOPE,
        "n_stochastic_cases": n_stochastic,
        "n_deterministic_cases": len(TOP1) - n_stochastic,
        "top1_configs": {str(k): {kk: vv for kk, vv in v.items() if kk != "h4"} for k, v in TOP1.items()},
        "gru_params": {"hidden_size": 256, "num_layers": 3, "dropout": 0.1, "epochs": 200},
        "ml_params": {
            "xgboost": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8},
            "random_forest": {"n_estimators": 200},
            "mlp": {"hidden_layer_sizes": [64, 32], "max_iter": 300, "learning_rate_init": 0.001},
            "svr": {"kernel": "rbf", "C": 10, "epsilon": 0.05},
            "ridge": {"alpha": 1.0},
        },
    }
    (out / "configs" / "H5_S2_T4_input_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    # ── Save log ──────────────────────────────────────────────────────────────
    (out / "logs" / "H5_S2_T4_run_log.txt").write_text("\n".join(_log_lines), encoding="utf-8")

    log(f"\nDone. Results at: {out}/metrics/H5_S2_T4_results.csv")


if __name__ == "__main__":
    main()

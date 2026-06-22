#!/usr/bin/env python3
"""B6_S1: ID-SWE — Inverse-Disagreement Sequence-Weighted Ensemble

[B6] Phase 1. Sequence model(FeatGRU/FeatLSTM)은 run-sequence 기반이라 전체 wear
trajectory를 매끄럽게 따라가지만 일부 case 후반에서 과대예측할 수 있다. 반대로
run-independent ML 모델(XGBoost/RF/SVR)은 덜 매끄럽지만 sequence 모델의 과상승을
완화할 수 있다. 따라서 **모델 간 disagreement가 낮을 때는 sequence prior를 더 신뢰,
높을 때는 run-independent 모델 쪽 비중을 높이는 adaptive ensemble**을 실험한다.

핵심 수식 (각 run t):
  y_seq(t)   = mean(FeatGRU(t), FeatLSTM(t))
  y_other(t) = mean / median(XGBoost(t), RF(t), SVR(t))
  D(t)       = std(FeatGRU, FeatLSTM, XGBoost, RF, SVR)
  D_norm(t)  = clip((D(t) - q_low) / (q_high - q_low + eps), 0, 1)
  w_seq(t)   = w_max - (w_max - w_min) * D_norm(t)        (w_max=0.75, w_min=0.40)
  y_final(t) = w_seq(t) * y_seq(t) + (1 - w_seq(t)) * y_other(t)

Leakage 방지(핵심): q_low/q_high는 test case 예측이 아니라 **train OOF disagreement
분포**로 calibration한다. 매 outer LOCV fold마다:
  1) held-out case tc 고정, train_pool = 14 cases
  2) inner LOOCV(train_pool 14-fold)로 5개 base model의 OOF 예측 생성
  3) train OOF 예측으로 D_train 분포 계산 → q_low,q_high = quantile(D_train,[0.2,0.8])
  4) train_pool 전체로 base 재학습 → tc 예측 (메타 입력)
  5) tc의 D_test를 train 기준 q_low/q_high로 normalize → w_seq → y_final

비교 baseline:
  E0  Mean Ensemble        : mean(GRU,LSTM,XGB,RF,SVR)
  E1  Static Seq-Heavy     : 0.70*mean(GRU,LSTM) + 0.30*mean(XGB,RF,SVR)
  E2  ID-SWE (Mean-Other)  : y_other = mean(XGB,RF,SVR)
  E3  ID-SWE (Median-Other): y_other = median(XGB,RF,SVR)

Base 모델 구성:
  FeatGRU/FeatLSTM : all sensors(mask=63) Delta+Meta, dim=27 (I1_S2 표준 config)
  XGBoost          : AC+vS      (B5_S1 own-best)
  RandomForest     : AC+vT+aS   (B5_S1 own-best)
  SVR              : AC+vS+aS   (B5_S1 own-best)

평가: LOCV 15 cases, observed-VB-only RMSE, 5-seed(mean±std). loss=epsilon-RMSE.

Output: experiments/executions/B6/S1/{timestamp}_id_swe/
"""
from __future__ import annotations

import json
import sys
import time as time_mod
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS       = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
SENSOR_ABBR   = {"smcAC": "AC", "smcDC": "DC", "vib_table": "vT",
                 "vib_spindle": "vS", "AE_table": "aT", "AE_spindle": "aS"}
ABBR_TO_IDX   = {v: i for i, v in enumerate(SENSOR_ABBR.values())}
EXCLUDED_RUNS = {(2, 1), (12, 1)}
NON_OBSERVED_RUNS = {
    (1,2),(1,3),(1,5),(1,16),(2,6),(8,4),(11,7),(11,14),(11,17),
    (12,4),(12,10),(13,1),(13,2),(14,1),(14,5),(15,1),(16,1),(16,2),(16,4),
}
META_FEATURES = ["DOC", "feed", "material"]
SEEDS         = [0, 1, 2, 3, 4]
THRESH        = 1e6
N_SENSORS     = len(SENSORS)
MASK_ALL      = 63   # all 6 sensors for sequence models

SEQ_MODELS    = ["FeatGRU", "FeatLSTM"]
ML_MODELS     = ["XGBoost", "RandomForest", "SVR"]
ALL_MODELS    = SEQ_MODELS + ML_MODELS
ML_OWN_BEST   = {"XGBoost": "AC+vS", "RandomForest": "AC+vT+aS", "SVR": "AC+vS+aS"}

# ID-SWE gate hyper-parameters
W_MAX     = 0.75   # disagreement 낮을 때 sequence model 비중
W_MIN     = 0.40   # disagreement 높을 때도 유지할 sequence prior 하한
Q_LOW     = 0.20
Q_HIGH    = 0.80
EPS       = 1e-8

# External references
B5_S2_AVG_ENSEMBLE = 0.103189   # B5_S2 own-best 4-ML simple average

RNN_CFG = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0,
)
XGB_CFG = dict(
    learning_rate=0.03, n_estimators=600, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.9,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, n_jobs=4,
)
RF_CFG  = dict(n_estimators=200, max_depth=None, n_jobs=4)
RMSE_EPS = 1e-8   # epsilon-stabilized RMSE loss


def label_to_mask(label: str) -> int:
    mask = 0
    for abbr in label.split("+"):
        mask |= (1 << ABBR_TO_IDX[abbr])
    return mask


# ─── Utils ────────────────────────────────────────────────────────────────────
def parse_signal(value: object) -> np.ndarray:
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros(1)
    return np.array(
        [np.mean(arr), np.sqrt(np.mean(arr ** 2)), np.std(arr), np.abs(arr).max()],
        dtype=np.float64,
    )


def preprocess(process: pd.DataFrame) -> pd.DataFrame:
    df = process.copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"] == min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside")
    )
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)


def mask_sensor_indices(mask: int) -> list[int]:
    return [i * 4 + j for i in range(N_SENSORS) if (mask >> i) & 1 for j in range(4)]


def build_full_raw_cache(signal_df, proc_clean):
    """Full 24-dim raw 4-stat per (case,run), plus first-run lookup for delta."""
    cache: dict[tuple[int, int], np.ndarray] = {}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        sig_row = signal_df[(signal_df["case"] == case_id) & (signal_df["run"] == run_id)]
        if sig_row.empty:
            continue
        sr = sig_row.iloc[0]
        arrays = {s: parse_signal(sr[s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        base_len = min(len(a) for a in arrays.values())
        cache[(case_id, run_id)] = np.concatenate(
            [extract_features(arrays[s][:base_len]) for s in SENSORS])
    first_run: dict[int, int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c, r) in cache if c == case_id)
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


# ─── ML feature frames (Delta+Meta per sensor subset) ─────────────────────────
def build_feat_df_for_mask(full_cache, first_run, proc_clean, mask):
    sensor_indices = mask_sensor_indices(mask)
    rows = []
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id)
        if key not in full_cache:
            continue
        raw_vec = full_cache[key][sensor_indices]
        ref_vec = full_cache.get((case_id, first_run.get(case_id, run_id)), np.zeros(24))[sensor_indices]
        delta = raw_vec - ref_vec
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        meta = np.array([float(getattr(row, mf, 0.0)) for mf in META_FEATURES])
        feat = np.concatenate([delta, meta])
        rec = {"case_id": case_id, "run": run_id, "VB": float(row.VB) if not pd.isna(row.VB) else 0.0}
        for k, v in enumerate(feat):
            rec[f"f{k}"] = float(v)
        rows.append(rec)
    df = pd.DataFrame(rows)
    feature_cols = [f"f{k}" for k in range(bin(mask).count("1") * 4 + len(META_FEATURES))]
    return df, feature_cols


def fit_ml(model_type: str, seed: int):
    if model_type == "XGBoost":
        return XGBRegressor(**XGB_CFG, random_state=seed, verbosity=0)
    if model_type == "RandomForest":
        return RandomForestRegressor(**RF_CFG, random_state=seed)
    return SVR(kernel="rbf", C=10.0, gamma="scale")


def predict_ml(model_type, train_df, test_df, feature_cols, seed) -> dict[int, float]:
    use_scaler = model_type == "SVR"
    X_tr_raw = train_df[feature_cols].values
    X_te_raw = test_df[feature_cols].values
    if use_scaler:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)
    else:
        X_tr, X_te = X_tr_raw, X_te_raw
    y_tr = train_df["VB"].values
    model = fit_ml(model_type, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
    if use_scaler:
        y_pred = np.clip(y_pred, 0.0, None)
    return {int(r): float(p) for r, p in zip(test_df["run"].to_numpy(int), y_pred)}


# ─── Sequence model (FeatGRU / FeatLSTM, Delta+Meta all sensors) ──────────────
def build_sequences(full_cache, first_run, proc_clean) -> dict[int, dict]:
    """Delta(all sensors)+Meta sequences per case (dim=27)."""
    sensor_indices = mask_sensor_indices(MASK_ALL)
    n_sensor_feats = len(sensor_indices)
    case_rows: dict[int, list[dict]] = {c: [] for c in CASE_SCOPE}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id)
        if key not in full_cache:
            continue
        raw_vec = full_cache[key][sensor_indices]
        ref_vec = full_cache.get((case_id, first_run.get(case_id, run_id)),
                                 np.zeros(24))[sensor_indices]
        delta = raw_vec - ref_vec
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        meta = np.array([float(getattr(row, mf, 0.0)) for mf in META_FEATURES])
        feat = np.concatenate([delta, meta]).astype(np.float32)
        vb = float(row.VB) if not pd.isna(row.VB) else 0.0
        case_rows[case_id].append({"feat": feat, "vb": vb, "run": run_id})
    cases: dict[int, dict] = {}
    for case_id, rows in case_rows.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["run"])
        cases[case_id] = {
            "seq": np.stack([r["feat"] for r in rows_sorted]),
            "vb": np.array([r["vb"] for r in rows_sorted], dtype=np.float32),
            "runs": np.array([r["run"] for r in rows_sorted], dtype=int),
            "n_runs": len(rows_sorted),
        }
    return cases, n_sensor_feats + len(META_FEATURES)


class DeltaMetaRNN(nn.Module):
    def __init__(self, input_dim: int, cell_type: str) -> None:
        super().__init__()
        cfg = RNN_CFG
        rnn_cls = nn.GRU if cell_type == "FeatGRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_dim, cfg["hidden_size"], cfg["num_layers"],
            batch_first=True, dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(cfg["hidden_size"], cfg["head_hidden"]), nn.ReLU(),
            nn.Linear(cfg["head_hidden"], 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out).squeeze(-1)


def fit_predict_seq(train_cases, test_cases, input_dim, cell_type, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = DeltaMetaRNN(input_dim, cell_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=RNN_CFG["lr"],
                                 weight_decay=RNN_CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RNN_CFG["epochs"])

    train_seqs = [info["seq"] for info in train_cases.values()]
    train_vbs  = [info["vb"]  for info in train_cases.values()]
    all_vb = np.concatenate(train_vbs)
    y_mean = float(all_vb.mean())
    y_std  = max(float(all_vb.std()), 1e-8)

    tensors = [torch.tensor(s, dtype=torch.float32) for s in train_seqs]
    lengths = torch.tensor([len(s) for s in train_seqs])
    x_train = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0).to(device)
    tgt     = [torch.tensor((t - y_mean) / y_std, dtype=torch.float32) for t in train_vbs]
    y_train = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=float("nan")).to(device)
    mask_tr = ~torch.isnan(y_train)

    model.train()
    for _ in range(RNN_CFG["epochs"]):
        optimizer.zero_grad()
        pred = model(x_train, lengths)
        # epsilon-stabilized RMSE loss
        loss = torch.sqrt(((pred[mask_tr] - y_train[mask_tr]) ** 2).mean() + RMSE_EPS)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), RNN_CFG["grad_clip"])
        optimizer.step()
        scheduler.step()

    model.eval()
    out: dict[int, dict[int, float]] = {}
    with torch.no_grad():
        for cid, info in test_cases.items():
            pred = model(
                torch.tensor(info["seq"], dtype=torch.float32).unsqueeze(0).to(device),
                torch.tensor([info["n_runs"]]),
            ).squeeze(0).cpu().numpy() * y_std + y_mean
            out[cid] = {int(r): float(p) for r, p in zip(info["runs"], pred)}
    return out


def obs_mask(case_id: int, runs: np.ndarray) -> np.ndarray:
    return np.array([(case_id, int(r)) not in NON_OBSERVED_RUNS for r in runs])


# ─── Base-prediction generation (one seed): OOF for train, direct for test ────
def base_predictions_for_fold(tc, seq_cases, seq_input_dim, ml_feat_dfs,
                              device, seed):
    """Return (oof_train, test_pred) prediction maps for outer fold tc.

    oof_train: {model: {(case,run): pred}}  via inner LOOCV over the 14 train cases
    test_pred: {model: {run: pred}}         base models trained on full 14-case pool
    """
    train_pool = [c for c in CASE_SCOPE if c != tc]

    oof_train: dict[str, dict[tuple[int, int], float]] = {m: {} for m in ALL_MODELS}
    test_pred: dict[str, dict[int, float]] = {m: {} for m in ALL_MODELS}

    # ---- Inner LOOCV for OOF disagreement calibration ----
    for ic in train_pool:
        inner_train = [c for c in train_pool if c != ic]
        # sequence models
        seq_train = {c: seq_cases[c] for c in inner_train if c in seq_cases}
        if ic in seq_cases:
            for cell in SEQ_MODELS:
                preds = fit_predict_seq(seq_train, {ic: seq_cases[ic]},
                                        seq_input_dim, cell, device, seed)
                for r, p in preds[ic].items():
                    oof_train[cell][(ic, r)] = p
        # ML models
        for m in ML_MODELS:
            df, cols = ml_feat_dfs[m]
            tr = df[df["case_id"].isin(inner_train)]
            te = df[df["case_id"] == ic].sort_values("run")
            if tr.empty or te.empty:
                continue
            for r, p in predict_ml(m, tr, te, cols, seed).items():
                oof_train[m][(ic, r)] = p

    # ---- Final base models on full train_pool -> predict tc ----
    seq_train_full = {c: seq_cases[c] for c in train_pool if c in seq_cases}
    if tc in seq_cases:
        for cell in SEQ_MODELS:
            preds = fit_predict_seq(seq_train_full, {tc: seq_cases[tc]},
                                    seq_input_dim, cell, device, seed)
            test_pred[cell] = preds[tc]
    for m in ML_MODELS:
        df, cols = ml_feat_dfs[m]
        tr = df[df["case_id"].isin(train_pool)]
        te = df[df["case_id"] == tc].sort_values("run")
        if not tr.empty and not te.empty:
            test_pred[m] = predict_ml(m, tr, te, cols, seed)

    return oof_train, test_pred


def disagreement_quantiles(oof_train) -> tuple[float, float]:
    """q_low, q_high from std-over-models on train OOF predictions."""
    # collect keys present in ALL models
    common = set.intersection(*[set(oof_train[m].keys()) for m in ALL_MODELS])
    d_vals = []
    for key in common:
        preds = np.array([oof_train[m][key] for m in ALL_MODELS], dtype=float)
        d_vals.append(float(np.std(preds)))
    d_arr = np.array(d_vals, dtype=float)
    return float(np.quantile(d_arr, Q_LOW)), float(np.quantile(d_arr, Q_HIGH))


# ─── Ensemble assembly for one outer fold ─────────────────────────────────────
def assemble_fold(tc, seq_cases, test_pred, q_low, q_high):
    """Build per-run ensemble predictions for outer fold tc.

    Returns list of per-run dicts (all columns) ordered by run.
    """
    runs = seq_cases[tc]["runs"]
    vb   = seq_cases[tc]["vb"]
    # restrict to runs predicted by every model
    common_runs = [int(r) for r in runs
                   if all(int(r) in test_pred[m] for m in ALL_MODELS)]
    rows = []
    for i, r in enumerate(runs):
        ri = int(r)
        if ri not in common_runs:
            continue
        base = {m: test_pred[m][ri] for m in ALL_MODELS}
        arr5 = np.array([base[m] for m in ALL_MODELS], dtype=float)
        y_seq = float(np.mean([base["FeatGRU"], base["FeatLSTM"]]))
        other_vals = np.array([base["XGBoost"], base["RandomForest"], base["SVR"]], dtype=float)
        y_other_mean = float(np.mean(other_vals))
        y_other_med  = float(np.median(other_vals))
        D = float(np.std(arr5))
        D_norm = float(np.clip((D - q_low) / (q_high - q_low + EPS), 0.0, 1.0))
        w_seq = W_MAX - (W_MAX - W_MIN) * D_norm

        E0 = float(np.mean(arr5))
        E1 = 0.70 * y_seq + 0.30 * y_other_mean
        E2 = w_seq * y_seq + (1.0 - w_seq) * y_other_mean
        E3 = w_seq * y_seq + (1.0 - w_seq) * y_other_med

        rows.append({
            "case": tc, "run": ri,
            "observed": (tc, ri) not in NON_OBSERVED_RUNS,
            "y_true": float(vb[i]),
            "FeatGRU_pred": base["FeatGRU"], "FeatLSTM_pred": base["FeatLSTM"],
            "XGBoost_pred": base["XGBoost"], "RandomForest_pred": base["RandomForest"],
            "SVR_pred": base["SVR"],
            "D": D, "D_norm": D_norm, "w_seq": w_seq,
            "E0_mean_pred": E0, "E1_static_seq_pred": E1,
            "E2_id_swe_mean_pred": E2, "E3_id_swe_median_pred": E3,
        })
    return rows


# ─── Metrics ──────────────────────────────────────────────────────────────────
ENSEMBLE_COLS = {
    "E0_mean": "E0_mean_pred",
    "E1_static_seq": "E1_static_seq_pred",
    "E2_id_swe_mean": "E2_id_swe_mean_pred",
    "E3_id_swe_median": "E3_id_swe_median_pred",
}
BASE_COLS = {m: f"{m}_pred" for m in ALL_MODELS}


def observed_rmse(df: pd.DataFrame, pred_col: str) -> float:
    """Macro RMSE = mean over cases of per-case observed RMSE."""
    rmses = []
    for c, g in df[df["observed"]].groupby("case"):
        if len(g) == 0:
            continue
        rmses.append(float(np.sqrt(mean_squared_error(g["y_true"], g[pred_col]))))
    return float(np.mean(rmses)) if rmses else float("nan")


def phase_of(g: pd.DataFrame) -> np.ndarray:
    """early/mid/late by within-case run-index thirds (observed runs)."""
    n = len(g)
    idx = np.arange(n)
    frac = (idx + 0.5) / n
    return np.where(frac < 1/3, "early", np.where(frac < 2/3, "mid", "late"))


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B6" / "S1" / f"{ts}_id_swe"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== B6_S1: ID-SWE — Inverse-Disagreement Sequence-Weighted Ensemble ===")
    log(f"Seq models: {SEQ_MODELS} (all-sensor Delta+Meta, dim=27)")
    log(f"ML models (own-best subset): {ML_OWN_BEST}")
    log(f"Gate: w_max={W_MAX}, w_min={W_MIN}, quantiles=[{Q_LOW},{Q_HIGH}], seeds={SEEDS}")
    log(f"Ensembles: E0 mean | E1 static-seq(.7/.3) | E2 ID-SWE mean | E3 ID-SWE median")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    full_cache, first_run = build_full_raw_cache(signal_df, proc_clean)
    log(f"Clean runs: {len(proc_clean)}, cache size: {len(full_cache)}")

    seq_cases, seq_input_dim = build_sequences(full_cache, first_run, proc_clean)
    ml_feat_dfs = {m: build_feat_df_for_mask(full_cache, first_run, proc_clean,
                                             label_to_mask(ML_OWN_BEST[m]))
                   for m in ML_MODELS}
    log(f"Sequence input_dim={seq_input_dim}; cases with sequences: {sorted(seq_cases)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    t_total = time_mod.time()

    # seed -> long predictions df (all runs, all columns)
    seed_pred_frames: dict[int, pd.DataFrame] = {}
    # also store per-seed q_low/q_high per fold for diagnostics / averaging
    fold_gate: dict[int, dict[int, tuple[float, float]]] = {s: {} for s in SEEDS}

    for seed in SEEDS:
        log(f"\n========== SEED {seed} ==========")
        t_seed = time_mod.time()
        all_rows = []
        for oi, tc in enumerate(CASE_SCOPE):
            if tc not in seq_cases:
                continue
            t0 = time_mod.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oof_train, test_pred = base_predictions_for_fold(
                    tc, seq_cases, seq_input_dim, ml_feat_dfs, device, seed)
            q_low, q_high = disagreement_quantiles(oof_train)
            fold_gate[seed][tc] = (q_low, q_high)
            all_rows.extend(assemble_fold(tc, seq_cases, test_pred, q_low, q_high))
            log(f"  [{oi+1}/{len(CASE_SCOPE)}] tc={tc:>2}  "
                f"q_low={q_low:.4f} q_high={q_high:.4f}  [{time_mod.time()-t0:.0f}s]")
        seed_pred_frames[seed] = pd.DataFrame(all_rows)
        log(f"  seed {seed} done [{time_mod.time()-t_seed:.0f}s]")

    # ── Per-seed ensemble RMSE (macro over cases) ───────────────────────────────
    seed_rmse: dict[str, list[float]] = {k: [] for k in ENSEMBLE_COLS}
    seed_base_rmse: dict[str, list[float]] = {m: [] for m in ALL_MODELS}
    for seed in SEEDS:
        df = seed_pred_frames[seed]
        for name, col in ENSEMBLE_COLS.items():
            seed_rmse[name].append(observed_rmse(df, col))
        for m in ALL_MODELS:
            seed_base_rmse[m].append(observed_rmse(df, BASE_COLS[m]))

    def agg(vals): return float(np.mean(vals)), float(np.std(vals))

    log("\n=== SUMMARY: Observed-VB RMSE (5-seed mean ± std) ===")
    log(f"{'Method':<22}{'Mean':>12}{'Std':>12}")
    log("-" * 46)
    for m in ALL_MODELS:
        mu, sd = agg(seed_base_rmse[m])
        log(f"{('[base] '+m):<22}{mu:>12.6f}{sd:>12.6f}")
    log("-" * 46)
    ens_summary = {}
    for name in ENSEMBLE_COLS:
        mu, sd = agg(seed_rmse[name])
        ens_summary[name] = {"mean": mu, "std": sd, "seed_rmses": seed_rmse[name]}
        log(f"{name:<22}{mu:>12.6f}{sd:>12.6f}")
    log("-" * 46)
    log(f"{'B5_S2 4ML-avg ref':<22}{B5_S2_AVG_ENSEMBLE:>12.6f}")

    best_ens = min(ens_summary, key=lambda k: ens_summary[k]["mean"])
    best_base = min(ALL_MODELS, key=lambda m: agg(seed_base_rmse[m])[0])
    log(f"\nBest ensemble: {best_ens} = {ens_summary[best_ens]['mean']:.6f}")
    log(f"Best single base: {best_base} = {agg(seed_base_rmse[best_base])[0]:.6f}")
    for name in ["E2_id_swe_mean", "E3_id_swe_median"]:
        d_vs_e0 = ens_summary[name]["mean"] - ens_summary["E0_mean"]["mean"]
        d_vs_e1 = ens_summary[name]["mean"] - ens_summary["E1_static_seq"]["mean"]
        d_vs_b  = ens_summary[name]["mean"] - agg(seed_base_rmse[best_base])[0]
        log(f"  {name}: vs E0 {d_vs_e0:+.6f} | vs E1 {d_vs_e1:+.6f} | "
            f"vs best base({best_base}) {d_vs_b:+.6f}")

    # ── Seed-averaged predictions for csv / trajectory / phase ────────────────
    base_long = pd.concat(seed_pred_frames.values(), ignore_index=True)
    num_cols = (["y_true", "D", "D_norm", "w_seq"]
                + list(ENSEMBLE_COLS.values()) + list(BASE_COLS.values()))
    avg_df = (base_long.groupby(["case", "run", "observed"], as_index=False)[num_cols]
              .mean()
              .sort_values(["case", "run"]).reset_index(drop=True))

    # ── metrics/predictions.csv ───────────────────────────────────────────────
    pred_cols = ["case", "run", "observed", "y_true",
                 "FeatGRU_pred", "FeatLSTM_pred", "XGBoost_pred",
                 "RandomForest_pred", "SVR_pred", "D", "D_norm", "w_seq",
                 "E0_mean_pred", "E1_static_seq_pred",
                 "E2_id_swe_mean_pred", "E3_id_swe_median_pred"]
    avg_df[pred_cols].to_csv(out_dir / "metrics" / "predictions.csv", index=False)

    # ── metrics/case_rmse.csv (per-seed mean per case for each ensemble) ──────
    case_rows = []
    for seed in SEEDS:
        df = seed_pred_frames[seed]
        for c, g in df[df["observed"]].groupby("case"):
            rec = {"seed": seed, "case": int(c), "n_obs": int(len(g))}
            for name, col in ENSEMBLE_COLS.items():
                rec[name] = float(np.sqrt(mean_squared_error(g["y_true"], g[col])))
            for m in ALL_MODELS:
                rec[f"base_{m}"] = float(np.sqrt(mean_squared_error(g["y_true"], g[BASE_COLS[m]])))
            case_rows.append(rec)
    case_rmse_df = pd.DataFrame(case_rows)
    case_rmse_df.to_csv(out_dir / "metrics" / "case_rmse.csv", index=False)
    case_mean = case_rmse_df.groupby("case")[list(ENSEMBLE_COLS)].mean()

    # ── metrics/phase_rmse.csv (early/mid/late pooled, per-seed then mean) ────
    phase_rows = []
    for seed in SEEDS:
        df = seed_pred_frames[seed]
        obs = df[df["observed"]].sort_values(["case", "run"]).copy()
        obs["phase"] = obs.groupby("case", group_keys=False).apply(
            lambda g: pd.Series(phase_of(g), index=g.index))
        for ph, g in obs.groupby("phase"):
            rec = {"seed": seed, "phase": ph, "n": int(len(g))}
            for name, col in ENSEMBLE_COLS.items():
                rec[name] = float(np.sqrt(mean_squared_error(g["y_true"], g[col])))
            phase_rows.append(rec)
    phase_df = pd.DataFrame(phase_rows)
    phase_df.to_csv(out_dir / "metrics" / "phase_rmse.csv", index=False)
    phase_mean = phase_df.groupby("phase")[list(ENSEMBLE_COLS)].mean()
    log("\nPhase RMSE (5-seed mean):")
    for ph in ["early", "mid", "late"]:
        if ph in phase_mean.index:
            vals = "  ".join(f"{k}={phase_mean.loc[ph, k]:.5f}" for k in ENSEMBLE_COLS)
            log(f"  {ph:<6} {vals}")

    # ── Figure: trajectory_id_swe (E2 vs base seq vs true) ────────────────────
    cases = sorted(avg_df["case"].unique())
    ncols = 5
    nrows = int(np.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for idx, c in enumerate(cases):
        ax = axes[idx // ncols][idx % ncols]
        d = avg_df[avg_df["case"] == c].sort_values("run")
        runs = d["run"].to_numpy(int)
        is_obs = d["observed"].to_numpy(bool)
        y_true = d["y_true"].to_numpy(float)
        y_seq = 0.5 * (d["FeatGRU_pred"].to_numpy() + d["FeatLSTM_pred"].to_numpy())
        ax.plot(runs, d["E2_id_swe_mean_pred"], "-", color="crimson", marker="o",
                markersize=4, linewidth=1.4, label="ID-SWE (E2)")
        ax.plot(runs, y_seq, "--", color="#4878d0", linewidth=1.1, label="seq mean (GRU+LSTM)")
        ax.plot(runs[is_obs], y_true[is_obs], "-", color="black", marker="o",
                markersize=4, linewidth=1.0, label="VB (observed)")
        if (~is_obs).any():
            ax.plot(runs[~is_obs], y_true[~is_obs], "o", mfc="none", mec="gray",
                    markersize=5, linewidth=0, label="VB (interpolated)")
        ax.set_title(f"Case {c}", fontsize=10)
        ax.set_xlabel("run", fontsize=8); ax.set_ylabel("VB", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6.5, loc="upper left")
    for idx in range(len(cases), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(f"B6_S1 ID-SWE (E2) — RMSE={ens_summary['E2_id_swe_mean']['mean']:.6f}", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(out_dir / "figures" / "trajectory_id_swe.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "figures" / "trajectory_id_swe.svg"), bbox_inches="tight")
    plt.close(fig)

    # ── Figure: ensemble_weight_trajectory (w_seq per case) ───────────────────
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for idx, c in enumerate(cases):
        ax = axes2[idx // ncols][idx % ncols]
        d = avg_df[avg_df["case"] == c].sort_values("run")
        runs = d["run"].to_numpy(int)
        ax.plot(runs, d["w_seq"], "-", color="darkgreen", marker="o", markersize=3, linewidth=1.2)
        ax.axhline(W_MAX, color="gray", ls=":", lw=0.8)
        ax.axhline(W_MIN, color="gray", ls=":", lw=0.8)
        ax.set_ylim(W_MIN - 0.05, W_MAX + 0.05)
        ax.set_title(f"Case {c}", fontsize=10)
        ax.set_xlabel("run", fontsize=8); ax.set_ylabel("w_seq", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)
    for idx in range(len(cases), nrows * ncols):
        axes2[idx // ncols][idx % ncols].axis("off")
    fig2.suptitle("B6_S1 ID-SWE: sequence weight w_seq trajectory (lower ⇒ more ML)", fontsize=13)
    plt.tight_layout()
    fig2.savefig(str(out_dir / "figures" / "ensemble_weight_trajectory.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── Figure: comparison bar ────────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    names = [f"base:{m}" for m in ALL_MODELS] + list(ENSEMBLE_COLS) + ["B5_S2 4ML-avg"]
    vals  = [agg(seed_base_rmse[m])[0] for m in ALL_MODELS] \
            + [ens_summary[n]["mean"] for n in ENSEMBLE_COLS] + [B5_S2_AVG_ENSEMBLE]
    colors = ["#4878d0"] * len(ALL_MODELS) + ["#6acc65", "#ee854a", "#d65f5f", "#956cb4"] + ["#999999"]
    bars = ax3.bar(range(len(names)), vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f"{v:.4f}",
                 ha="center", fontsize=7)
    ax3.set_xticks(range(len(names))); ax3.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax3.set_ylabel("Observed-VB RMSE")
    ax3.set_title("B6_S1: Base models vs Ensembles (ID-SWE)")
    ax3.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig3.savefig(str(out_dir / "figures" / "comparison_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # ── logs/summary.json ─────────────────────────────────────────────────────
    summary = {
        "experiment": "B6_S1_inverse_disagreement_sequence_weighted_ensemble",
        "seq_models": SEQ_MODELS, "ml_models": ML_OWN_BEST,
        "gate": {"w_max": W_MAX, "w_min": W_MIN, "q_low": Q_LOW, "q_high": Q_HIGH},
        "seeds": SEEDS,
        "base_rmse": {m: dict(zip(["mean", "std"], agg(seed_base_rmse[m]))) for m in ALL_MODELS},
        "ensemble_rmse": ens_summary,
        "best_ensemble": {"name": best_ens, "mean": ens_summary[best_ens]["mean"]},
        "best_single_base": {"name": best_base, "mean": agg(seed_base_rmse[best_base])[0]},
        "b5_s2_avg_ensemble_ref": B5_S2_AVG_ENSEMBLE,
        "phase_rmse_mean": {ph: {k: float(phase_mean.loc[ph, k]) for k in ENSEMBLE_COLS}
                            for ph in phase_mean.index},
        "case_rmse_mean": {int(c): {k: float(case_mean.loc[c, k]) for k in ENSEMBLE_COLS}
                           for c in case_mean.index},
        "fold_gate": {str(s): {str(c): v for c, v in fold_gate[s].items()} for s in SEEDS},
        "execution_dir": str(out_dir),
        "total_elapsed_s": time_mod.time() - t_total,
    }
    (out_dir / "logs" / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    log(f"\nDone in {time_mod.time()-t_total:.0f}s. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

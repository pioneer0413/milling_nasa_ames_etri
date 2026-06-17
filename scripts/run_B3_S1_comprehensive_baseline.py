#!/usr/bin/env python3
"""B3_S1: Comprehensive Baseline — H18 확장판.

B3 업데이트. Feature-GRU vs 13종 baseline 모델 종합 비교.

─── Per-run, handcrafted feature (Delta+Meta 15-dim) ───────────────
CF       : Carry-forward (no training)          — original H18
RL       : RunIndex Linear                      — original H18
ML       : Meta+RunIndex Linear                 — original H18
DL       : Delta+Meta Linear                    — original H18
Ridge    : RidgeCV                              — B3 NEW
RF       : RandomForest (seed-sensitive)        — B3 NEW
SVR      : SVR RBF kernel                       — B3 NEW
MLP_Feat : 2-layer MLP [15→128→64→1]           — B3 NEW

─── Signal-based per-run (raw AC+vT+vS, L timesteps) ──────────────
SignalCNN: 1D Conv → GlobalAvgPool → head       — B3 NEW
SignalGRU: GRU over timesteps → last hidden     — B3 NEW

─── Feature-based sequence (Delta+Meta 15-dim over run sequence) ───
FeatRNN  : RNN(256×3)  + MLP head(32)          — B3 NEW
FeatLSTM : LSTM(256×3) + MLP head(32)          — B3 NEW
FeatGRU  : GRU(256×3)  + MLP head(32)          — B3 NEW (re-run for consistency)

─── Reference (from B4) ───────────────────────────────────────────
XGBoost (per-run): mean=0.109239

Protocol: LOCV 15 cases, observed_vb eval, 5-seed for stochastic models.
Output: experiments/executions/B3/S1/{timestamp}_comprehensive_baseline/
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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS       = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
SIGNAL_SENSORS = ["smcAC", "vib_table", "vib_spindle"]   # AC+vT+vS — match GRU_MASK=13
EXCLUDED_RUNS = {(2, 1), (12, 1)}
NON_OBSERVED_RUNS = {
    (1,2),(1,3),(1,5),(1,16),(2,6),(8,4),(11,7),(11,14),(11,17),
    (12,4),(12,10),(13,1),(13,2),(14,1),(14,5),(15,1),(16,1),(16,2),(16,4),
}
META_FEATURES = ["DOC", "feed", "material"]
SEEDS         = [0, 1, 2, 3, 4]
PCT           = 100
THRESH        = 1e6
N_SENSORS     = len(SENSORS)
GRU_MASK      = 13   # AC+vT+vS

# References from B4/H18 (XGBoost only; GRU is now re-run for consistency)
REF = {"XGBoost (B4)": 0.109239}

# Signal model hyperparams
SIG_EPOCHS    = 150
SIG_LR        = 1e-3
SIG_BATCH     = 32
SIG_GRU_HID   = 64
SIG_GRU_LAY   = 2
SIG_CNN_CH    = [32, 64, 128]

# Feature MLP hyperparams
MLP_EPOCHS    = 200
MLP_LR        = 1e-3
MLP_HIDDEN    = [128, 64]

# Feature sequence model hyperparams (RNN/LSTM/GRU over run sequence)
SEQ_HIDDEN    = 256
SEQ_LAYERS    = 3
SEQ_DROPOUT   = 0.1
SEQ_HEAD_HID  = 32
SEQ_EPOCHS    = 200
SEQ_LR        = 1e-3
SEQ_WD        = 1e-4
SEQ_GRAD_CLIP = 1.0


# ─── Preprocessing ────────────────────────────────────────────────────────────
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


# ─── Feature cache (Delta+Meta) ───────────────────────────────────────────────
def build_feat_cache(
    signal_df: pd.DataFrame, proc_clean: pd.DataFrame
) -> tuple[dict[tuple[int,int], np.ndarray], dict[int,int]]:
    """Raw 4-stat features for GRU_MASK sensors."""
    sensor_idx = mask_sensor_indices(GRU_MASK)
    cache: dict[tuple[int,int], np.ndarray] = {}
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
        end = max(1, int(np.ceil(base_len * PCT / 100.0)))
        full = np.concatenate([extract_features(arrays[s][:end]) for s in SENSORS])
        cache[(case_id, run_id)] = full[sensor_idx]

    first_run: dict[int,int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c, r) in cache if c == case_id)
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


def build_feat_df(
    feat_cache: dict[tuple[int,int], np.ndarray],
    first_run: dict[int,int],
    proc_clean: pd.DataFrame,
) -> pd.DataFrame:
    n_feat = bin(GRU_MASK).count("1") * 4
    rows: list[dict] = []
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id)
        if key not in feat_cache:
            continue
        ref_vec = feat_cache.get((case_id, first_run.get(case_id, run_id)), np.zeros(n_feat))
        delta = feat_cache[key] - ref_vec
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        rec = {"case_id": case_id, "run": run_id, "VB": float(row.VB)}
        for mf in META_FEATURES:
            rec[mf] = float(getattr(row, mf, 0.0))
        for k, v in enumerate(delta):
            rec[f"d{k}"] = float(v)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)
    return df


# ─── Signal cache (raw time series) ──────────────────────────────────────────
def build_signal_cache(
    signal_df: pd.DataFrame,
    proc_clean: pd.DataFrame,
    sig_len: int,
) -> dict[tuple[int,int], np.ndarray]:
    """(case, run) → float32 array of shape (sig_len, 3)."""
    cache: dict[tuple[int,int], np.ndarray] = {}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        sig_row = signal_df[(signal_df["case"] == case_id) & (signal_df["run"] == run_id)]
        if sig_row.empty:
            continue
        sr = sig_row.iloc[0]
        arrays = [parse_signal(sr[s]) for s in SIGNAL_SENSORS]
        if any(np.abs(a).max() > THRESH for a in arrays):
            continue
        # truncate to sig_len, pad if shorter
        stacked = np.zeros((sig_len, len(SIGNAL_SENSORS)), dtype=np.float32)
        for ch, arr in enumerate(arrays):
            arr = arr[:sig_len].astype(np.float32)
            stacked[:len(arr), ch] = arr
        cache[(case_id, run_id)] = stacked
    return cache


def compute_sig_len(signal_df: pd.DataFrame, proc_clean: pd.DataFrame) -> int:
    """10th-percentile of per-run minimum signal lengths across SIGNAL_SENSORS."""
    lengths: list[int] = []
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        sig_row = signal_df[(signal_df["case"] == case_id) & (signal_df["run"] == run_id)]
        if sig_row.empty:
            continue
        sr = sig_row.iloc[0]
        arrays = [parse_signal(sr[s]) for s in SIGNAL_SENSORS]
        if any(np.abs(a).max() > THRESH for a in arrays):
            continue
        lengths.append(min(len(a) for a in arrays))
    return int(np.percentile(lengths, 10)) if lengths else 1000


# ─── Observed-VB mask helper ──────────────────────────────────────────────────
def obs_mask(case_id: int, runs: np.ndarray) -> np.ndarray:
    return np.array([(case_id, int(r)) not in NON_OBSERVED_RUNS for r in runs])


# ─────────────────────────────────────────────────────────────────────────────
# PART A: Handcrafted feature models
# ─────────────────────────────────────────────────────────────────────────────

DELTA_COLS = [f"d{k}" for k in range(bin(GRU_MASK).count("1") * 4)]
FEAT_COLS  = DELTA_COLS + META_FEATURES


def _feat_locv_single(feat_df: pd.DataFrame, model_fn) -> tuple[float, dict[int,float]]:
    """One LOCV pass with a sklearn-style model_fn() factory."""
    case_rmses: dict[int,float] = {}
    for tc in CASE_SCOPE:
        tr = feat_df[feat_df["case_id"] != tc]
        te = feat_df[feat_df["case_id"] == tc].sort_values("run")
        if tr.empty or te.empty:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr[FEAT_COLS].values)
        X_te = scaler.transform(te[FEAT_COLS].values)
        y_tr = tr["VB"].values
        y_te = te["VB"].values
        runs = te["run"].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = model_fn()
            m.fit(X_tr, y_tr)
            y_pred = np.clip(m.predict(X_te), 0.0, None)

        obs = obs_mask(tc, runs)
        if obs.sum() == 0:
            continue
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_te[obs], y_pred[obs])))
    return float(np.mean(list(case_rmses.values()))), case_rmses


def run_cf(proc_clean: pd.DataFrame) -> tuple[float, dict[int,float]]:
    case_rmses: dict[int,float] = {}
    for tc in CASE_SCOPE:
        df = proc_clean[proc_clean["case"] == tc].sort_values("run")
        if df.empty:
            continue
        vb_true = df["VB"].to_numpy(float)
        runs    = df["run"].to_numpy(int)
        vb_pred = np.concatenate([[0.0], vb_true[:-1]])
        obs = obs_mask(tc, runs)
        if obs.sum() == 0:
            continue
        case_rmses[tc] = float(np.sqrt(mean_squared_error(vb_true[obs], vb_pred[obs])))
    return float(np.mean(list(case_rmses.values()))), case_rmses


def run_rl(feat_df: pd.DataFrame) -> tuple[float, dict[int,float]]:
    return _feat_locv_single(feat_df, lambda: RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
                             ) if False else __run_rl(feat_df)


def __run_rl(feat_df: pd.DataFrame) -> tuple[float, dict[int,float]]:
    case_rmses: dict[int,float] = {}
    for tc in CASE_SCOPE:
        tr = feat_df[feat_df["case_id"] != tc]
        te = feat_df[feat_df["case_id"] == tc].sort_values("run")
        if tr.empty or te.empty:
            continue
        X_tr = tr[["run"]].values.astype(float)
        X_te = te[["run"]].values.astype(float)
        y_tr, y_te = tr["VB"].values, te["VB"].values
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(X_tr, y_tr)
        y_pred = np.clip(m.predict(X_te), 0.0, None)
        obs = obs_mask(tc, te["run"].values)
        if obs.sum() == 0:
            continue
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_te[obs], y_pred[obs])))
    return float(np.mean(list(case_rmses.values()))), case_rmses


def run_ml(feat_df: pd.DataFrame) -> tuple[float, dict[int,float]]:
    case_rmses: dict[int,float] = {}
    from sklearn.linear_model import LinearRegression
    for tc in CASE_SCOPE:
        tr = feat_df[feat_df["case_id"] != tc]
        te = feat_df[feat_df["case_id"] == tc].sort_values("run")
        if tr.empty or te.empty:
            continue
        cols = META_FEATURES + ["run"]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr[cols].values.astype(float))
        X_te = scaler.transform(te[cols].values.astype(float))
        y_tr, y_te = tr["VB"].values, te["VB"].values
        m = LinearRegression().fit(X_tr, y_tr)
        y_pred = np.clip(m.predict(X_te), 0.0, None)
        obs = obs_mask(tc, te["run"].values)
        if obs.sum() == 0:
            continue
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_te[obs], y_pred[obs])))
    return float(np.mean(list(case_rmses.values()))), case_rmses


def run_dl(feat_df: pd.DataFrame) -> tuple[float, dict[int,float]]:
    from sklearn.linear_model import LinearRegression
    return _feat_locv_single_inline(feat_df, lambda: LinearRegression())


def _feat_locv_single_inline(feat_df, model_fn) -> tuple[float, dict[int,float]]:
    case_rmses: dict[int,float] = {}
    for tc in CASE_SCOPE:
        tr = feat_df[feat_df["case_id"] != tc]
        te = feat_df[feat_df["case_id"] == tc].sort_values("run")
        if tr.empty or te.empty:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr[FEAT_COLS].values)
        X_te = scaler.transform(te[FEAT_COLS].values)
        y_tr, y_te = tr["VB"].values, te["VB"].values
        m = model_fn()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_tr, y_tr)
            y_pred = np.clip(m.predict(X_te), 0.0, None)
        obs = obs_mask(tc, te["run"].values)
        if obs.sum() == 0:
            continue
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_te[obs], y_pred[obs])))
    return float(np.mean(list(case_rmses.values()))), case_rmses


def run_ridge(feat_df: pd.DataFrame) -> tuple[float, dict[int,float]]:
    return _feat_locv_single_inline(
        feat_df, lambda: RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]))


def run_rf(feat_df: pd.DataFrame, seed: int) -> tuple[float, dict[int,float]]:
    return _feat_locv_single_inline(
        feat_df, lambda: RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=seed, n_jobs=4))


def run_svr(feat_df: pd.DataFrame) -> tuple[float, dict[int,float]]:
    return _feat_locv_single_inline(feat_df, lambda: SVR(kernel="rbf", C=10.0, gamma="scale"))


# ─── Feature MLP ──────────────────────────────────────────────────────────────
class FeatMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, MLP_HIDDEN[0]), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(MLP_HIDDEN[0], MLP_HIDDEN[1]), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(MLP_HIDDEN[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def run_mlp_feat(feat_df: pd.DataFrame, device: torch.device, seed: int) -> tuple[float, dict[int,float]]:
    input_dim = len(FEAT_COLS)
    case_rmses: dict[int,float] = {}
    for tc in CASE_SCOPE:
        tr = feat_df[feat_df["case_id"] != tc]
        te = feat_df[feat_df["case_id"] == tc].sort_values("run")
        if tr.empty or te.empty:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr[FEAT_COLS].values).astype(np.float32)
        X_te = scaler.transform(te[FEAT_COLS].values).astype(np.float32)
        y_tr = tr["VB"].values.astype(np.float32)
        y_te = te["VB"].values

        torch.manual_seed(seed)
        model = FeatMLP(input_dim).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=MLP_LR)
        sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)
        x_t   = torch.tensor(X_tr).to(device)
        y_t   = torch.tensor(y_tr).to(device)
        model.train()
        for _ in range(MLP_EPOCHS):
            opt.zero_grad()
            loss = ((model(x_t) - y_t) ** 2).mean()
            loss.backward()
            opt.step()
            sch.step()

        model.eval()
        with torch.no_grad():
            y_pred = np.clip(
                model(torch.tensor(X_te).to(device)).cpu().numpy(), 0.0, None)

        obs = obs_mask(tc, te["run"].values)
        if obs.sum() == 0:
            continue
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_te[obs], y_pred[obs])))
    return float(np.mean(list(case_rmses.values()))), case_rmses


# ─────────────────────────────────────────────────────────────────────────────
# PART B: Signal-based models
# ─────────────────────────────────────────────────────────────────────────────

class SignalCNN(nn.Module):
    def __init__(self, n_channels: int, sig_len: int) -> None:
        super().__init__()
        chs = SIG_CNN_CH
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, chs[0], kernel_size=7, padding=3), nn.BatchNorm1d(chs[0]), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(chs[0], chs[1], kernel_size=5, padding=2), nn.BatchNorm1d(chs[1]), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(chs[1], chs[2], kernel_size=3, padding=1), nn.BatchNorm1d(chs[2]), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chs[2], 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        return self.head(self.encoder(x)).squeeze(-1)


class SignalGRUModel(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            n_channels, SIG_GRU_HID, SIG_GRU_LAY,
            batch_first=True, dropout=0.1 if SIG_GRU_LAY > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(SIG_GRU_HID, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels)
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


def _signal_locv(
    sig_cache: dict[tuple[int,int], np.ndarray],
    proc_clean: pd.DataFrame,
    model_cls,
    model_kwargs: dict,
    device: torch.device,
    seed: int,
    is_cnn: bool,
) -> tuple[float, dict[int,float]]:
    n_ch = len(SIGNAL_SENSORS)
    case_rmses: dict[int,float] = {}

    for tc in CASE_SCOPE:
        # collect train runs
        tr_rows = proc_clean[proc_clean["case"] != tc]
        te_rows = proc_clean[proc_clean["case"] == tc].sort_values("run")
        if tr_rows.empty or te_rows.empty:
            continue

        def get_signals(rows):
            sigs, vbs, runs = [], [], []
            for row in rows.itertuples(index=False):
                key = (int(row.case), int(row.run))
                if key not in sig_cache:
                    continue
                sigs.append(sig_cache[key])  # (L, 3)
                vbs.append(float(row.VB))
                runs.append(int(row.run))
            return np.stack(sigs), np.array(vbs, np.float32), np.array(runs)

        X_tr, y_tr, _   = get_signals(tr_rows)
        X_te, y_te, runs_te = get_signals(te_rows)
        if len(X_tr) == 0 or len(X_te) == 0:
            continue

        # Normalize per channel (fit on train)
        scaler = StandardScaler()
        L = X_tr.shape[1]
        X_tr_2d = X_tr.reshape(-1, n_ch)
        scaler.fit(X_tr_2d)
        X_tr = scaler.transform(X_tr_2d).reshape(-1, L, n_ch).astype(np.float32)
        X_te = scaler.transform(X_te.reshape(-1, n_ch)).reshape(-1, L, n_ch).astype(np.float32)

        if is_cnn:
            X_tr_t = torch.tensor(X_tr.transpose(0, 2, 1)).to(device)  # (N, C, L)
            X_te_t = torch.tensor(X_te.transpose(0, 2, 1)).to(device)
        else:
            X_tr_t = torch.tensor(X_tr).to(device)   # (N, L, C)
            X_te_t = torch.tensor(X_te).to(device)

        y_mean, y_std = float(y_tr.mean()), max(float(y_tr.std()), 1e-8)
        y_tr_t = torch.tensor((y_tr - y_mean) / y_std).to(device)

        torch.manual_seed(seed)
        model = model_cls(**model_kwargs).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=SIG_LR)
        sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SIG_EPOCHS)

        n_train = len(X_tr_t)
        model.train()
        for _ in range(SIG_EPOCHS):
            perm = torch.randperm(n_train)
            for i in range(0, n_train, SIG_BATCH):
                idx = perm[i:i+SIG_BATCH]
                opt.zero_grad()
                loss = ((model(X_tr_t[idx]) - y_tr_t[idx]) ** 2).mean()
                loss.backward()
                opt.step()
            sch.step()

        model.eval()
        with torch.no_grad():
            y_pred = (model(X_te_t).cpu().numpy() * y_std + y_mean)
        y_pred = np.clip(y_pred, 0.0, None)

        obs = obs_mask(tc, runs_te)
        if obs.sum() == 0:
            continue
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_te[obs], y_pred[obs])))

    mean_rmse = float(np.mean(list(case_rmses.values()))) if case_rmses else float("nan")
    return mean_rmse, case_rmses


def run_signal_cnn(sig_cache, proc_clean, sig_len, device, seed):
    return _signal_locv(
        sig_cache, proc_clean,
        SignalCNN, {"n_channels": len(SIGNAL_SENSORS), "sig_len": sig_len},
        device, seed, is_cnn=True)


def run_signal_gru(sig_cache, proc_clean, sig_len, device, seed):
    return _signal_locv(
        sig_cache, proc_clean,
        SignalGRUModel, {"n_channels": len(SIGNAL_SENSORS)},
        device, seed, is_cnn=False)


# ─────────────────────────────────────────────────────────────────────────────
# PART C: Feature sequence models (RNN / LSTM / GRU over run sequence)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureSeqModel(nn.Module):
    """Generic recurrent sequence model over run-level Delta+Meta features."""
    def __init__(self, input_dim: int, cell: str = "gru") -> None:
        super().__init__()
        self.cell_type = cell
        drop = SEQ_DROPOUT if SEQ_LAYERS > 1 else 0.0
        kw = dict(batch_first=True, dropout=drop)
        if cell == "rnn":
            self.rnn = nn.RNN(input_dim, SEQ_HIDDEN, SEQ_LAYERS, nonlinearity="tanh", **kw)
        elif cell == "lstm":
            self.rnn = nn.LSTM(input_dim, SEQ_HIDDEN, SEQ_LAYERS, **kw)
        else:
            self.rnn = nn.GRU(input_dim, SEQ_HIDDEN, SEQ_LAYERS, **kw)
        self.head = nn.Sequential(
            nn.Linear(SEQ_HIDDEN, SEQ_HEAD_HID), nn.ReLU(),
            nn.Linear(SEQ_HEAD_HID, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out).squeeze(-1)


def _build_seq_cases(feat_df: pd.DataFrame) -> dict[int, dict]:
    """Group feat_df rows into per-case run sequences."""
    cases: dict[int, dict] = {}
    for case_id in CASE_SCOPE:
        rows = feat_df[feat_df["case_id"] == case_id].sort_values("run")
        if rows.empty:
            continue
        cases[case_id] = {
            "seq":    rows[FEAT_COLS].values.astype(np.float32),
            "vb":     rows["VB"].values.astype(np.float32),
            "runs":   rows["run"].values.astype(int),
            "n_runs": len(rows),
        }
    return cases


def run_feat_seq(
    feat_df: pd.DataFrame,
    device: torch.device,
    seed: int,
    cell: str,
) -> tuple[float, dict[int, float]]:
    """LOCV for RNN/LSTM/GRU sequence model over run-level features."""
    all_cases = _build_seq_cases(feat_df)
    input_dim  = len(FEAT_COLS)
    case_rmses: dict[int, float] = {}

    for tc in CASE_SCOPE:
        if tc not in all_cases:
            continue
        train = {c: all_cases[c] for c in CASE_SCOPE if c != tc and c in all_cases}

        # Build padded training tensors
        train_seqs = [info["seq"] for info in train.values()]
        train_vbs  = [info["vb"]  for info in train.values()]
        all_vb = np.concatenate(train_vbs)
        y_mean = float(all_vb.mean())
        y_std  = max(float(all_vb.std()), 1e-8)

        tensors = [torch.tensor(s) for s in train_seqs]
        lengths = torch.tensor([len(s) for s in train_seqs])
        x_train = nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=0.0).to(device)
        tgt     = [torch.tensor((t - y_mean) / y_std, dtype=torch.float32) for t in train_vbs]
        y_train = nn.utils.rnn.pad_sequence(
            tgt, batch_first=True, padding_value=float("nan")).to(device)
        mask_tr = ~torch.isnan(y_train)

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = FeatureSeqModel(input_dim, cell=cell).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=SEQ_LR, weight_decay=SEQ_WD)
        sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SEQ_EPOCHS)

        model.train()
        for _ in range(SEQ_EPOCHS):
            opt.zero_grad()
            pred = model(x_train, lengths)
            loss = ((pred[mask_tr] - y_train[mask_tr]) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), SEQ_GRAD_CLIP)
            opt.step()
            sch.step()

        model.eval()
        with torch.no_grad():
            info  = all_cases[tc]
            x_te  = torch.tensor(info["seq"]).unsqueeze(0).to(device)
            y_pred = (
                model(x_te, torch.tensor([info["n_runs"]])).squeeze(0).cpu().numpy()
                * y_std + y_mean
            )
        y_pred = np.clip(y_pred, 0.0, None)

        obs = obs_mask(tc, info["runs"])
        if obs.sum() == 0:
            continue
        case_rmses[tc] = float(
            np.sqrt(mean_squared_error(info["vb"][obs], y_pred[obs])))

    return float(np.mean(list(case_rmses.values()))), case_rmses


# ─── Aggregation helper ───────────────────────────────────────────────────────
def aggregate_seeds(
    results_per_seed: list[tuple[float, dict[int,float]]]
) -> tuple[float, float, dict[int,float]]:
    means = [r[0] for r in results_per_seed]
    all_case_rmses: dict[int, list[float]] = {}
    for _, case_r in results_per_seed:
        for c, v in case_r.items():
            all_case_rmses.setdefault(c, []).append(v)
    case_mean = {c: float(np.mean(vs)) for c, vs in all_case_rmses.items()}
    return float(np.mean(means)), float(np.std(means)), case_mean


# ─── Plot ─────────────────────────────────────────────────────────────────────
_COLOR_MAP = {
    # naive / no-sensor
    "CF":        "#b5cfea",
    "RL":        "#aec7e8",
    "ML":        "#9ab8d8",
    "DL":        "#86a8c8",
    # per-run handcrafted
    "Ridge":     "#4878d0",
    "SVR":       "#3a64bc",
    "RF":        "#2c50a8",
    "MLP_Feat":  "#1e3c94",
    # signal-based
    "SignalCNN": "#d62728",
    "SignalGRU": "#c5b0d5",
    # feature sequence
    "FeatRNN":   "#ff7f0e",
    "FeatLSTM":  "#e8640c",
    "FeatGRU":   "#2ca02c",
    # references
    "XGBoost (B4)": "#8c564b",
}
_DEFAULT_COLOR = "#999999"


def plot_results(model_names, means, stds, out_dir):
    all_names = model_names + list(REF.keys())
    all_means = means + list(REF.values())
    all_stds  = stds + [0.0] * len(REF)

    colors = [_COLOR_MAP.get(n, _DEFAULT_COLOR) for n in all_names]

    fig, ax = plt.subplots(figsize=(18, 6))
    bars = ax.bar(range(len(all_names)), all_means,
                  color=colors, alpha=0.85, yerr=all_stds, capsize=3)
    for bar, m, s in zip(bars, all_means, all_stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.003,
                f"{m:.4f}", ha="center", va="bottom", fontsize=7)

    # FeatGRU reference line (from this run)
    if "FeatGRU" in model_names:
        gru_val = means[model_names.index("FeatGRU")]
        ax.axhline(gru_val, color="#2ca02c", linestyle="--", linewidth=1.2,
                   label=f"FeatGRU={gru_val:.4f}")

    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Observed-VB RMSE (mean ± std, LOCV)")
    ax.set_title("B3_S1: Comprehensive Baseline — 13 models vs FeatGRU")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_dir / "comprehensive_baseline.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "comprehensive_baseline.svg"), bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(results: dict, out_dir: Path) -> None:
    ckpt = {n: {"mean": m, "std": s} for n, (m, s, _) in results.items()}
    (out_dir / "metrics" / "checkpoint.json").write_text(
        json.dumps(ckpt, indent=2), encoding="utf-8"
    )


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B3" / "S1" / f"{ts}_comprehensive_baseline"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== B3_S1: Comprehensive Baseline ===")
    log("Models: CF, RL, ML, DL, Ridge, RF, SVR, MLP_Feat, SignalCNN, SignalGRU, FeatRNN, FeatLSTM, FeatGRU")
    log(f"Seeds={SEEDS}, PCT={PCT}%, LOCV={len(CASE_SCOPE)} cases")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    log("Building handcrafted feature cache...")
    feat_cache, first_run = build_feat_cache(signal_df, proc_clean)
    feat_df = build_feat_df(feat_cache, first_run, proc_clean)

    log("Computing signal length...")
    sig_len = compute_sig_len(signal_df, proc_clean)
    log(f"  sig_len (10th pct) = {sig_len}")

    log("Building signal cache...")
    sig_cache = build_signal_cache(signal_df, proc_clean, sig_len)
    log(f"  signal cache: {len(sig_cache)} runs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    results: dict[str, tuple[float, float, dict[int,float]]] = {}
    t_total = time_mod.time()

    # ── Deterministic models (run once) ──────────────────────────────────────
    for name, fn in [
        ("CF",    lambda: run_cf(proc_clean)),
        ("RL",    lambda: __run_rl(feat_df)),
        ("ML",    lambda: run_ml(feat_df)),
        ("DL",    lambda: run_dl(feat_df)),
        ("Ridge", lambda: run_ridge(feat_df)),
        ("SVR",   lambda: run_svr(feat_df)),
    ]:
        log(f"\n--- {name} (deterministic) ---")
        t0 = time_mod.time()
        mean_, case_r = fn()
        results[name] = (mean_, 0.0, case_r)
        log(f"  RMSE={mean_:.6f}  [{time_mod.time()-t0:.1f}s]")
        save_checkpoint(results, out_dir)

    # ── Stochastic models (5 seeds) ───────────────────────────────────────────
    for name, fn in [
        ("RF",         lambda seed: run_rf(feat_df, seed)),
        ("MLP_Feat",   lambda seed: run_mlp_feat(feat_df, device, seed)),
        ("SignalCNN",  lambda seed: run_signal_cnn(sig_cache, proc_clean, sig_len, device, seed)),
        ("SignalGRU",  lambda seed: run_signal_gru(sig_cache, proc_clean, sig_len, device, seed)),
        ("FeatRNN",    lambda seed: run_feat_seq(feat_df, device, seed, "rnn")),
        ("FeatLSTM",   lambda seed: run_feat_seq(feat_df, device, seed, "lstm")),
        ("FeatGRU",    lambda seed: run_feat_seq(feat_df, device, seed, "gru")),
    ]:
        log(f"\n--- {name} (5-seed) ---")
        seed_results: list[tuple[float, dict[int,float]]] = []
        t0 = time_mod.time()
        for seed in SEEDS:
            mean_, case_r = fn(seed)
            seed_results.append((mean_, case_r))
            log(f"  seed={seed}  RMSE={mean_:.6f}  [{time_mod.time()-t0:.0f}s]")
        m, s, case_r = aggregate_seeds(seed_results)
        results[name] = (m, s, case_r)
        log(f"  {name}: mean={m:.6f}  std={s:.6f}")
        save_checkpoint(results, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    gru_ref = results["FeatGRU"][0] if "FeatGRU" in results else 0.095122
    log(f"{'Model':<14} {'Mean':>10} {'Std':>10} {'vs FeatGRU':>12}")
    log("-" * 50)
    ordered = sorted(results.items(), key=lambda x: x[1][0])
    model_names_plot, means_plot, stds_plot = [], [], []
    for name, (m, s, _) in ordered:
        log(f"{name:<14} {m:>10.6f} {s:>10.6f} {m-gru_ref:>+12.6f}")
        model_names_plot.append(name)
        means_plot.append(m)
        stds_plot.append(s)

    log("-" * 50)
    for k, v in REF.items():
        log(f"{k:<14} {v:>10.6f} {'':>10} {'(ref)':>12}")

    # ── Save ──────────────────────────────────────────────────────────────────
    agg_rows = [
        {"model": n, "mean": m, "std": s, "vs_gru": m - gru_ref}
        for n, (m, s, _) in results.items()
    ]
    pd.DataFrame(agg_rows).sort_values("mean").to_csv(
        out_dir / "metrics" / "aggregate.csv", index=False)

    case_ids = sorted({c for _, _, cr in results.values() for c in cr})
    per_case = {"case": case_ids}
    for name, (_, _, cr) in results.items():
        per_case[name] = [cr.get(c, float("nan")) for c in case_ids]
    pd.DataFrame(per_case).to_csv(out_dir / "metrics" / "per_case.csv", index=False)

    plot_results(model_names_plot, means_plot, stds_plot, out_dir / "figures")

    summary = {
        "experiment": "B3_S1_comprehensive_baseline",
        "sig_len": sig_len,
        "seeds": SEEDS,
        "results": {n: {"mean": m, "std": s} for n, (m, s, _) in results.items()},
        "references": REF,
        "feat_gru_rmse": gru_ref,
        "execution_dir": str(out_dir),
        "total_elapsed_s": time_mod.time() - t_total,
    }
    (out_dir / "logs" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text(
        "\n".join(log_lines), encoding="utf-8")

    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

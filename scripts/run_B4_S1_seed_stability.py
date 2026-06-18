#!/usr/bin/env python3
"""B4_S1: Seed stability experiment — 100% input, 5 seeds.

(Notion [B4] 5-Seed Stability — 100% 입력 기준; 구 파일명 run_H17_S1_seed_stability.py)

논문 보고 수치의 안정성 검증. Prefix/segmentation 없이 전체 신호(100%) 사용.

T1: Feature-GRU — AC+vT+vS (mask=13), Delta+Meta 15-dim
    Ref: H12_S1_T1 pct=100% 3-seed mean RMSE=0.095010
T2: XGBoost    — AC+vS   (mask= 9), Delta+Meta 11-dim
    Ref: H12_S1_T2 pct=100% 3-seed mean RMSE=0.109247
T3: Cai2020 hybrid LSTM — LSTM(256×3) over 1/2-downsampled raw signal +
    process[case#, DOC, feed, material, prev-VB] → FC(32,8).
    Reported with / without prev-VB to isolate the carry-forward term.

Seeds: [0, 1, 2, 3, 4]
Protocol: LOCV (15 cases), observed_vb eval
Output: experiments/executions/B4/S1/{timestamp}_seed_stability/
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
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS       = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
SENSOR_ABBR   = {"smcAC": "AC", "smcDC": "DC", "vib_table": "vT",
                 "vib_spindle": "vS", "AE_table": "aT", "AE_spindle": "aS"}
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

GRU_MASK = 13   # AC+vT+vS
XGB_MASK = 9    # AC+vS

REF_GRU_3SEED = 0.095010   # H12_S1_T1 pct=100%, 3-seed
REF_XGB_3SEED = 0.109247   # H12_S1_T2 pct=100%, 3-seed

GRU_CFG = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0,
)
XGB_CFG = dict(
    learning_rate=0.03, n_estimators=600, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.9,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, n_jobs=4,
)

# T3: Cai2020 hybrid LSTM (Notion Model DB: cai2020_hybridLSTM).
#   3-layer LSTM(256) over 1/2-downsampled raw signal, last hidden + process
#   info -> FC(32,8). Same model as integrated into B3; here measured for
#   5-seed stability. SIGNAL_SENSORS follows the B3 convention (AC+vT+vS).
SIGNAL_SENSORS   = ["smcAC", "vib_table", "vib_spindle"]
CAI_HIDDEN       = 256
CAI_LAYERS       = 3
CAI_LSTM_DROPOUT = 0.2
CAI_HEAD_HID     = (32, 8)
CAI_HEAD_DROPOUT = 0.2
CAI_LR           = 1e-3
CAI_BATCH        = 16
CAI_EPOCHS       = 200
CAI_DOWNSAMPLE   = 2     # paper: 1/2 downsampling of the raw run signal


# ─── Signal / feature utils ───────────────────────────────────────────────────
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
    return [
        i * 4 + j
        for i in range(N_SENSORS) if (mask >> i) & 1
        for j in range(4)
    ]


def mask_label(mask: int) -> str:
    return "+".join(SENSOR_ABBR[SENSORS[i]] for i in range(N_SENSORS) if (mask >> i) & 1)


# ─── Feature cache (single pct) ───────────────────────────────────────────────
def build_cache(
    signal_df: pd.DataFrame,
    proc_clean: pd.DataFrame,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[int, int]]:
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
        end = max(1, int(np.ceil(base_len * PCT / 100.0)))
        feats = np.concatenate([extract_features(arrays[s][:end]) for s in SENSORS])
        cache[(case_id, run_id)] = feats

    first_run: dict[int, int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c, r) in cache if c == case_id)
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


# ─── GRU data builder ─────────────────────────────────────────────────────────
def build_gru_sequences(
    cache: dict[tuple[int, int], np.ndarray],
    first_run: dict[int, int],
    proc_clean: pd.DataFrame,
    mask: int,
) -> dict[int, dict]:
    sensor_indices = mask_sensor_indices(mask)
    case_rows: dict[int, list[dict]] = {c: [] for c in CASE_SCOPE}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id)
        if key not in cache:
            continue
        ref_run = first_run.get(case_id, run_id)
        ref_vec = cache.get((case_id, ref_run), np.zeros(N_SENSORS * 4))
        delta = (cache[key] - ref_vec)[sensor_indices]
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        meta = np.array([float(getattr(row, mf, 0.0)) for mf in META_FEATURES], dtype=np.float64)
        feat = np.concatenate([delta, meta]).astype(np.float32)
        vb = float(row.VB) if not pd.isna(row.VB) else 0.0
        case_rows[case_id].append({"feat": feat, "vb": vb, "run": run_id})

    cases: dict[int, dict] = {}
    for case_id, rows in case_rows.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["run"])
        seq  = np.stack([r["feat"] for r in rows_sorted])
        vb   = np.array([r["vb"]  for r in rows_sorted], dtype=np.float32)
        runs = np.array([r["run"] for r in rows_sorted], dtype=int)
        cases[case_id] = {"seq": seq, "vb": vb, "runs": runs, "n_runs": len(rows_sorted)}
    return cases


# ─── GRU model ────────────────────────────────────────────────────────────────
class DeltaMetaGRU(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        cfg = GRU_CFG
        self.gru = nn.GRU(
            input_dim, cfg["hidden_size"], cfg["num_layers"],
            batch_first=True,
            dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg["hidden_size"], cfg["head_hidden"]),
            nn.ReLU(),
            nn.Linear(cfg["head_hidden"], 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out).squeeze(-1)


def fit_predict_gru(
    train_cases: dict[int, dict],
    test_cases: dict[int, dict],
    input_dim: int,
    device: torch.device,
    seed: int,
) -> dict[int, np.ndarray]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = DeltaMetaGRU(input_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=GRU_CFG["lr"], weight_decay=GRU_CFG["weight_decay"])
    epochs    = GRU_CFG["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train, lengths)
        loss = ((pred[mask_tr] - y_train[mask_tr]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRU_CFG["grad_clip"])
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        return {
            cid: (
                model(
                    torch.tensor(info["seq"], dtype=torch.float32).unsqueeze(0).to(device),
                    torch.tensor([info["n_runs"]]),
                ).squeeze(0).cpu().numpy() * y_std + y_mean
            )
            for cid, info in test_cases.items()
        }


def run_gru_locv_seed(
    all_cases: dict[int, dict], input_dim: int, device: torch.device, seed: int
) -> tuple[float, dict[int, float]]:
    """Returns (mean_rmse, {case: rmse})."""
    case_rmses: dict[int, float] = {}
    for tc in CASE_SCOPE:
        if tc not in all_cases:
            continue
        train = {c: all_cases[c] for c in CASE_SCOPE if c != tc and c in all_cases}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = fit_predict_gru(train, {tc: all_cases[tc]}, input_dim, device, seed)
        runs = all_cases[tc]["runs"]
        obs  = np.array([(tc, int(r)) not in NON_OBSERVED_RUNS for r in runs])
        y_pred = preds[tc][obs]
        y_true = all_cases[tc]["vb"][obs]
        if len(y_true) == 0:
            continue
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mean_rmse = float(np.mean(list(case_rmses.values()))) if case_rmses else float("nan")
    return mean_rmse, case_rmses


# ─── XGB data builder ─────────────────────────────────────────────────────────
def build_xgb_df(
    cache: dict[tuple[int, int], np.ndarray],
    first_run: dict[int, int],
    proc_clean: pd.DataFrame,
    mask: int,
) -> pd.DataFrame:
    sensor_indices = mask_sensor_indices(mask)
    rows: list[dict] = []
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id)
        if key not in cache:
            continue
        ref_run = first_run.get(case_id, run_id)
        ref_vec = cache.get((case_id, ref_run), np.zeros(N_SENSORS * 4))
        delta   = (cache[key] - ref_vec)[sensor_indices]
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        vb  = float(row.VB) if not pd.isna(row.VB) else 0.0
        rec: dict = {"case_id": case_id, "run": run_id, "VB": vb}
        for mf in META_FEATURES:
            rec[mf] = float(getattr(row, mf, 0.0))
        for k, val in enumerate(delta):
            rec[f"f{k}"] = float(val)
        rows.append(rec)
    return pd.DataFrame(rows)


def run_xgb_locv_seed(
    feat_df: pd.DataFrame, feature_cols: list[str], seed: int
) -> tuple[float, dict[int, float]]:
    """Returns (mean_rmse, {case: rmse})."""
    case_rmses: dict[int, float] = {}
    for tc in CASE_SCOPE:
        test_df  = feat_df[feat_df["case_id"] == tc]
        train_df = feat_df[feat_df["case_id"] != tc]
        if test_df.empty or train_df.empty:
            continue
        model = XGBRegressor(**XGB_CFG, random_state=seed, verbosity=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_df[feature_cols], train_df["VB"])
            y_pred = model.predict(test_df[feature_cols])
        obs = test_df.apply(
            lambda r: (int(r["case_id"]), int(r["run"])) not in NON_OBSERVED_RUNS, axis=1
        ).to_numpy()
        y_true = test_df["VB"].to_numpy(float)[obs]
        y_pred_obs = y_pred[obs]
        if len(y_true) == 0:
            continue
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_true, y_pred_obs)))
    mean_rmse = float(np.mean(list(case_rmses.values()))) if case_rmses else float("nan")
    return mean_rmse, case_rmses


# ─── T3: Cai2020 hybrid LSTM (signal + process) ───────────────────────────────
def obs_mask(case_id: int, runs: np.ndarray) -> np.ndarray:
    return np.array([(case_id, int(r)) not in NON_OBSERVED_RUNS for r in runs])


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


def build_signal_cache(
    signal_df: pd.DataFrame, proc_clean: pd.DataFrame, sig_len: int
) -> dict[tuple[int,int], np.ndarray]:
    """(case, run) → float32 array of shape (sig_len, len(SIGNAL_SENSORS))."""
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
        stacked = np.zeros((sig_len, len(SIGNAL_SENSORS)), dtype=np.float32)
        for ch, arr in enumerate(arrays):
            arr = arr[:sig_len].astype(np.float32)
            stacked[:len(arr), ch] = arr
        cache[(case_id, run_id)] = stacked
    return cache


def build_process_cache(
    proc_clean: pd.DataFrame, use_prev_vb: bool
) -> dict[tuple[int,int], np.ndarray]:
    """(case, run) → process vector [case#, DOC, feed, material, (prev-run VB)].

    ``use_prev_vb`` toggles the carry-forward prev-run VB term (first run = 0),
    reported as a separate with/without variant to isolate its contribution.
    """
    cache: dict[tuple[int,int], np.ndarray] = {}
    for case_id, grp in proc_clean.groupby("case"):
        grp = grp.sort_values("run")
        vbs = grp["VB"].to_numpy(float)
        prev = np.concatenate([[0.0], vbs[:-1]])
        for i, row in enumerate(grp.itertuples(index=False)):
            feats = [float(case_id), float(row.DOC), float(row.feed), float(row.material)]
            if use_prev_vb:
                feats.append(float(prev[i]))
            cache[(int(case_id), int(row.run))] = np.array(feats, dtype=np.float32)
    return cache


class HybridLSTMModel(nn.Module):
    """Cai2020 hybrid information model: stacked LSTM over the (downsampled) raw
    signal, last hidden state concatenated with a process vector, FC(32,8) head."""
    def __init__(self, n_channels: int, n_process: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_channels, CAI_HIDDEN, CAI_LAYERS, batch_first=True,
            dropout=CAI_LSTM_DROPOUT if CAI_LAYERS > 1 else 0.0)
        h1, h2 = CAI_HEAD_HID
        self.head = nn.Sequential(
            nn.Linear(CAI_HIDDEN + n_process, h1), nn.ReLU(), nn.Dropout(CAI_HEAD_DROPOUT),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(CAI_HEAD_DROPOUT),
            nn.Linear(h2, 1),
        )

    def forward(self, signal: torch.Tensor, process: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(signal)
        return self.head(torch.cat([h[-1], process], dim=1)).squeeze(-1)


def _hybrid_locv(
    sig_cache: dict[tuple[int,int], np.ndarray],
    proc_cache: dict[tuple[int,int], np.ndarray],
    proc_clean: pd.DataFrame,
    device: torch.device,
    seed: int,
    n_process: int,
) -> tuple[float, dict[int,float]]:
    from sklearn.preprocessing import StandardScaler
    n_ch = len(SIGNAL_SENSORS)
    case_rmses: dict[int,float] = {}

    for tc in CASE_SCOPE:
        tr_rows = proc_clean[proc_clean["case"] != tc]
        te_rows = proc_clean[proc_clean["case"] == tc].sort_values("run")
        if tr_rows.empty or te_rows.empty:
            continue

        def gather(rows):
            sigs, procs, vbs, runs = [], [], [], []
            for row in rows.itertuples(index=False):
                key = (int(row.case), int(row.run))
                if key not in sig_cache or key not in proc_cache:
                    continue
                sigs.append(sig_cache[key]); procs.append(proc_cache[key])
                vbs.append(float(row.VB)); runs.append(int(row.run))
            if not sigs:
                return None
            return (np.stack(sigs), np.stack(procs),
                    np.array(vbs, np.float32), np.array(runs))

        tr, te = gather(tr_rows), gather(te_rows)
        if tr is None or te is None:
            continue
        X_tr, P_tr, y_tr, _       = tr
        X_te, P_te, y_te, runs_te = te

        # Paper: 1/2 downsampling of the raw run signal
        X_tr = X_tr[:, ::CAI_DOWNSAMPLE, :]
        X_te = X_te[:, ::CAI_DOWNSAMPLE, :]

        L = X_tr.shape[1]
        sig_scaler = StandardScaler().fit(X_tr.reshape(-1, n_ch))
        X_tr = sig_scaler.transform(X_tr.reshape(-1, n_ch)).reshape(-1, L, n_ch).astype(np.float32)
        X_te = sig_scaler.transform(X_te.reshape(-1, n_ch)).reshape(-1, L, n_ch).astype(np.float32)

        proc_scaler = StandardScaler().fit(P_tr)
        P_tr = proc_scaler.transform(P_tr).astype(np.float32)
        P_te = proc_scaler.transform(P_te).astype(np.float32)

        y_mean, y_std = float(y_tr.mean()), max(float(y_tr.std()), 1e-8)

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = HybridLSTMModel(n_ch, n_process).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=CAI_LR)
        sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CAI_EPOCHS)

        Xtr_t = torch.tensor(X_tr).to(device)
        Ptr_t = torch.tensor(P_tr).to(device)
        ytr_t = torch.tensor((y_tr - y_mean) / y_std).to(device)

        n_train = len(Xtr_t)
        model.train()
        for _ in range(CAI_EPOCHS):
            perm = torch.randperm(n_train)
            for i in range(0, n_train, CAI_BATCH):
                idx = perm[i:i+CAI_BATCH]
                opt.zero_grad()
                loss = ((model(Xtr_t[idx], Ptr_t[idx]) - ytr_t[idx]) ** 2).mean()
                loss.backward()
                opt.step()
            sch.step()

        model.eval()
        with torch.no_grad():
            y_pred = (model(torch.tensor(X_te).to(device),
                            torch.tensor(P_te).to(device)).cpu().numpy()
                      * y_std + y_mean)
        y_pred = np.clip(y_pred, 0.0, None)

        obs = obs_mask(tc, runs_te)
        if obs.sum() == 0:
            continue
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_te[obs], y_pred[obs])))

    mean_rmse = float(np.mean(list(case_rmses.values()))) if case_rmses else float("nan")
    return mean_rmse, case_rmses


def run_hybrid_lstm(
    sig_cache: dict[tuple[int,int], np.ndarray],
    proc_clean: pd.DataFrame,
    device: torch.device,
    seed: int,
    use_prev_vb: bool,
) -> tuple[float, dict[int,float]]:
    proc_cache = build_process_cache(proc_clean, use_prev_vb)
    n_process  = 5 if use_prev_vb else 4
    return _hybrid_locv(sig_cache, proc_cache, proc_clean, device, seed, n_process)


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_stability(
    model_seed_rmses: dict[str, list[float]],
    gru_case_matrix: dict[int, list[float]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: seed-level aggregate RMSE (one grouped bar set per model)
    ax = axes[0]
    x = np.arange(len(SEEDS))
    names = list(model_seed_rmses.keys())
    n = max(len(names), 1)
    total_w = 0.8
    w = total_w / n
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, name in enumerate(names):
        offset = -total_w / 2 + w * (i + 0.5)
        rmses = model_seed_rmses[name]
        ax.bar(x + offset, rmses, w, label=name, color=colors[i % 10], alpha=0.85)
        ax.axhline(float(np.mean(rmses)), color=colors[i % 10], linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("Observed-VB RMSE")
    ax.set_title("Aggregate RMSE per Seed")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: per-case RMSE box plot (GRU)
    ax = axes[1]
    case_ids = sorted(gru_case_matrix.keys())
    data = [gru_case_matrix[c] for c in case_ids]
    bp = ax.boxplot(data, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.5),
                    medianprops=dict(color="navy"))
    ax.set_xticks(range(1, len(case_ids) + 1))
    ax.set_xticklabels([f"C{c}" for c in case_ids], fontsize=8)
    ax.set_ylabel("RMSE (5 seeds)")
    ax.set_title(f"Feature-GRU Per-Case RMSE Distribution ({len(SEEDS)} seeds)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"B4_S1: Seed Stability — 100% Input, {len(SEEDS)} Seeds", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(out_dir / "seed_stability.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "seed_stability.svg"), bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B4" / "S1" / f"{ts}_seed_stability"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== B4_S1: Seed Stability — 100% Input, 5 Seeds ===")
    log(f"T1: Feature-GRU  mask={GRU_MASK} ({mask_label(GRU_MASK)})  ref_3seed={REF_GRU_3SEED}")
    log(f"T2: XGBoost      mask={XGB_MASK} ({mask_label(XGB_MASK)})  ref_3seed={REF_XGB_3SEED}")
    log(f"PCT={PCT}%, Seeds={SEEDS}, LOCV={len(CASE_SCOPE)} cases")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    log("Building feature cache (pct=100%)...")
    cache, first_run = build_cache(signal_df, proc_clean)
    log(f"Cache size: {len(cache)} runs")

    log("Building signal cache (Cai2020 raw-signal input)...")
    sig_len   = compute_sig_len(signal_df, proc_clean)
    sig_cache = build_signal_cache(signal_df, proc_clean, sig_len)
    log(f"  sig_len (10th pct)={sig_len}  signal cache: {len(sig_cache)} runs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ── T1: Feature-GRU ──────────────────────────────────────────────────────
    gru_n_sensors = bin(GRU_MASK).count("1")
    gru_input_dim = gru_n_sensors * 4 + len(META_FEATURES)
    log(f"\n--- T1: Feature-GRU  input_dim={gru_input_dim}  label={mask_label(GRU_MASK)} ---")
    all_cases = build_gru_sequences(cache, first_run, proc_clean, GRU_MASK)

    gru_seed_rmses: list[float] = []
    gru_case_matrix: dict[int, list[float]] = {c: [] for c in CASE_SCOPE}
    t0 = time_mod.time()

    for seed in SEEDS:
        mean_rmse, case_rmses = run_gru_locv_seed(all_cases, gru_input_dim, device, seed)
        gru_seed_rmses.append(mean_rmse)
        for c, r in case_rmses.items():
            gru_case_matrix[c].append(r)
        elapsed = time_mod.time() - t0
        log(f"  seed={seed}  RMSE={mean_rmse:.6f}  [{elapsed:.0f}s]")

    gru_mean = float(np.mean(gru_seed_rmses))
    gru_std  = float(np.std(gru_seed_rmses))
    gru_cv   = gru_std / gru_mean if gru_mean > 0 else float("nan")
    log(f"  GRU 5-seed: mean={gru_mean:.6f}  std={gru_std:.6f}  CV={gru_cv:.4f}"
        f"  (3-seed ref={REF_GRU_3SEED:.6f})")

    # ── T2: XGBoost ──────────────────────────────────────────────────────────
    n_delta = bin(XGB_MASK).count("1") * 4
    delta_cols   = [f"f{k}" for k in range(n_delta)]
    feature_cols = delta_cols + META_FEATURES
    log(f"\n--- T2: XGBoost  n_feat={len(feature_cols)}  label={mask_label(XGB_MASK)} ---")
    feat_df = build_xgb_df(cache, first_run, proc_clean, XGB_MASK)
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_df.fillna(0.0, inplace=True)

    xgb_seed_rmses: list[float] = []
    xgb_case_matrix: dict[int, list[float]] = {c: [] for c in CASE_SCOPE}
    t0 = time_mod.time()

    for seed in SEEDS:
        mean_rmse, case_rmses = run_xgb_locv_seed(feat_df, feature_cols, seed)
        xgb_seed_rmses.append(mean_rmse)
        for c, r in case_rmses.items():
            xgb_case_matrix[c].append(r)
        elapsed = time_mod.time() - t0
        log(f"  seed={seed}  RMSE={mean_rmse:.6f}  [{elapsed:.0f}s]")

    xgb_mean = float(np.mean(xgb_seed_rmses))
    xgb_std  = float(np.std(xgb_seed_rmses))
    xgb_cv   = xgb_std / xgb_mean if xgb_mean > 0 else float("nan")
    log(f"  XGB 5-seed: mean={xgb_mean:.6f}  std={xgb_std:.6f}  CV={xgb_cv:.4f}"
        f"  (3-seed ref={REF_XGB_3SEED:.6f})")

    # ── T3/T4: Cai2020 hybrid LSTM (with / without prev-VB) ──────────────────
    cai_results: dict[str, dict] = {}
    for label, use_prev_vb in [("Cai2020", True), ("Cai2020_noPVB", False)]:
        log(f"\n--- {label}: Cai2020 hybrid LSTM  use_prev_vb={use_prev_vb} ---")
        seed_rmses: list[float] = []
        case_matrix: dict[int, list[float]] = {c: [] for c in CASE_SCOPE}
        t0 = time_mod.time()
        for seed in SEEDS:
            mean_rmse, case_rmses = run_hybrid_lstm(sig_cache, proc_clean, device, seed, use_prev_vb)
            seed_rmses.append(mean_rmse)
            for c, r in case_rmses.items():
                case_matrix[c].append(r)
            log(f"  seed={seed}  RMSE={mean_rmse:.6f}  [{time_mod.time()-t0:.0f}s]")
        c_mean = float(np.mean(seed_rmses))
        c_std  = float(np.std(seed_rmses))
        c_cv   = c_std / c_mean if c_mean > 0 else float("nan")
        log(f"  {label} 5-seed: mean={c_mean:.6f}  std={c_std:.6f}  CV={c_cv:.4f}")
        cai_results[label] = {
            "seed_rmses": seed_rmses, "case_matrix": case_matrix,
            "mean": c_mean, "std": c_std, "cv": c_cv,
        }

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    # Aggregate seed table
    agg_df = pd.DataFrame({
        "seed": SEEDS,
        "gru_rmse": gru_seed_rmses,
        "xgb_rmse": xgb_seed_rmses,
        "cai2020_rmse": cai_results["Cai2020"]["seed_rmses"],
        "cai2020_noPVB_rmse": cai_results["Cai2020_noPVB"]["seed_rmses"],
    })
    agg_df.to_csv(out_dir / "metrics" / "seed_aggregate.csv", index=False)

    # Per-case table (GRU)
    gru_rows = []
    for c in CASE_SCOPE:
        if not gru_case_matrix[c]:
            continue
        row = {"case": c}
        for i, s in enumerate(SEEDS):
            row[f"seed{s}"] = gru_case_matrix[c][i] if i < len(gru_case_matrix[c]) else float("nan")
        vals = [v for v in gru_case_matrix[c] if not np.isnan(v)]
        row["mean"] = float(np.mean(vals)) if vals else float("nan")
        row["std"]  = float(np.std(vals))  if vals else float("nan")
        gru_rows.append(row)
    gru_case_df = pd.DataFrame(gru_rows)
    gru_case_df.to_csv(out_dir / "metrics" / "gru_per_case.csv", index=False)

    # Per-case table (XGB)
    xgb_rows = []
    for c in CASE_SCOPE:
        if not xgb_case_matrix[c]:
            continue
        row = {"case": c}
        for i, s in enumerate(SEEDS):
            row[f"seed{s}"] = xgb_case_matrix[c][i] if i < len(xgb_case_matrix[c]) else float("nan")
        vals = [v for v in xgb_case_matrix[c] if not np.isnan(v)]
        row["mean"] = float(np.mean(vals)) if vals else float("nan")
        row["std"]  = float(np.std(vals))  if vals else float("nan")
        xgb_rows.append(row)
    xgb_case_df = pd.DataFrame(xgb_rows)
    xgb_case_df.to_csv(out_dir / "metrics" / "xgb_per_case.csv", index=False)

    # Per-case tables (Cai2020 variants)
    for label, info in cai_results.items():
        cmatrix = info["case_matrix"]
        rows = []
        for c in CASE_SCOPE:
            if not cmatrix[c]:
                continue
            row = {"case": c}
            for i, s in enumerate(SEEDS):
                row[f"seed{s}"] = cmatrix[c][i] if i < len(cmatrix[c]) else float("nan")
            vals = [v for v in cmatrix[c] if not np.isnan(v)]
            row["mean"] = float(np.mean(vals)) if vals else float("nan")
            row["std"]  = float(np.std(vals))  if vals else float("nan")
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            out_dir / "metrics" / f"{label.lower()}_per_case.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_stability(
        {
            f"GRU ({mask_label(GRU_MASK)})": gru_seed_rmses,
            f"XGB ({mask_label(XGB_MASK)})": xgb_seed_rmses,
            "Cai2020":       cai_results["Cai2020"]["seed_rmses"],
            "Cai2020_noPVB": cai_results["Cai2020_noPVB"]["seed_rmses"],
        },
        {c: v for c, v in gru_case_matrix.items() if v},
        out_dir / "figures",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    log(f"{'Model':<20} {'Mean':>10} {'Std':>10} {'CV':>8} {'3-seed ref':>12} {'Δ':>10}")
    log("-" * 74)
    log(f"{'GRU '+mask_label(GRU_MASK):<20} {gru_mean:>10.6f} {gru_std:>10.6f}"
        f" {gru_cv:>8.4f} {REF_GRU_3SEED:>12.6f} {gru_mean-REF_GRU_3SEED:>+10.6f}")
    log(f"{'XGB '+mask_label(XGB_MASK):<20} {xgb_mean:>10.6f} {xgb_std:>10.6f}"
        f" {xgb_cv:>8.4f} {REF_XGB_3SEED:>12.6f} {xgb_mean-REF_XGB_3SEED:>+10.6f}")
    for label, info in cai_results.items():
        log(f"{label:<20} {info['mean']:>10.6f} {info['std']:>10.6f}"
            f" {info['cv']:>8.4f} {'—':>12} {'—':>10}")

    log("\nGRU per-seed:")
    for s, r in zip(SEEDS, gru_seed_rmses):
        log(f"  seed={s}  {r:.6f}")
    log("\nXGB per-seed:")
    for s, r in zip(SEEDS, xgb_seed_rmses):
        log(f"  seed={s}  {r:.6f}")

    summary = {
        "experiment": "B4_S1_seed_stability",
        "pct": PCT,
        "seeds": SEEDS,
        "gru": {
            "mask": GRU_MASK, "sensor_subset": mask_label(GRU_MASK),
            "input_dim": gru_input_dim,
            "seed_rmses": gru_seed_rmses,
            "mean": gru_mean, "std": gru_std, "cv": gru_cv,
            "ref_3seed": REF_GRU_3SEED,
        },
        "xgb": {
            "mask": XGB_MASK, "sensor_subset": mask_label(XGB_MASK),
            "n_features": len(feature_cols),
            "seed_rmses": xgb_seed_rmses,
            "mean": xgb_mean, "std": xgb_std, "cv": xgb_cv,
            "ref_3seed": REF_XGB_3SEED,
        },
        "cai2020": {
            "signal_sensors": SIGNAL_SENSORS, "sig_len": sig_len,
            "downsample": CAI_DOWNSAMPLE,
            "with_prev_vb": {
                "seed_rmses": cai_results["Cai2020"]["seed_rmses"],
                "mean": cai_results["Cai2020"]["mean"],
                "std":  cai_results["Cai2020"]["std"],
                "cv":   cai_results["Cai2020"]["cv"],
            },
            "without_prev_vb": {
                "seed_rmses": cai_results["Cai2020_noPVB"]["seed_rmses"],
                "mean": cai_results["Cai2020_noPVB"]["mean"],
                "std":  cai_results["Cai2020_noPVB"]["std"],
                "cv":   cai_results["Cai2020_noPVB"]["cv"],
            },
        },
        "execution_dir": str(out_dir),
    }
    (out_dir / "logs" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text(
        "\n".join(log_lines), encoding="utf-8")

    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

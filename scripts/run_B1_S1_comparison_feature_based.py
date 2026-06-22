#!/usr/bin/env python3
"""B1_S1: Comparison: Feature-based Model — 8 models x Full segment

[B1] 명세(Notion) 실행. Segment=Full 고정, All sensors(mask=63)+Delta+Meta(27-dim)을
8개 모델에 동일 적용해 feature 기반 모델 전체를 한 leaderboard로 비교한다.

모델 (8종):
  Naive       : Carry-forward
  ML          : Ridge(RidgeCV), Random Forest, XGBoost, SVR
  DL-Feature  : Feature-MLP, Feature-GRU, Feature-LSTM

Feature-GRU/LSTM/XGBoost/RandomForest는 [B2]_S1(2026-06-22, all-sensors 재실행)과
동일 조건이므로 재계산 결과가 일치해야 함(sanity check). Ridge/SVR/Feature-MLP/
Carry-forward는 본 실험에서 신규 실행.

평가: LOCV 15 cases, observed-VB-only RMSE.
  - 확률적 모델(RandomForest/FeatureMLP/FeatGRU/FeatLSTM): 5-seed mean±std
  - 결정론적 모델(CarryForward/Ridge/SVR/XGBoost): 1회 실행

Output: experiments/executions/B1/S1/{timestamp}_comparison_feature_based/
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
THRESH        = 1e6
N_SENSORS     = len(SENSORS)
MASK          = 63   # All sensors
MODELS        = ["CarryForward", "Ridge", "RandomForest", "XGBoost", "SVR", "FeatureMLP", "FeatGRU", "FeatLSTM"]
STOCHASTIC    = {"RandomForest", "FeatureMLP", "FeatGRU", "FeatLSTM"}

# [B2]_S1 reference (all-sensors, Delta+Meta, Full, 2026-06-22) — expect this run to reproduce these
REF_B2_S1 = {"FeatGRU": 0.114839, "FeatLSTM": 0.117919, "XGBoost": 0.113095, "RandomForest": 0.135995}

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
MLP_HIDDEN = [128, 64]
MLP_EPOCHS = 200
MLP_LR     = 1e-3


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
    return [
        i * 4 + j
        for i in range(N_SENSORS) if (mask >> i) & 1
        for j in range(4)
    ]


def mask_label(mask: int) -> str:
    return "+".join(SENSOR_ABBR[SENSORS[i]] for i in range(N_SENSORS) if (mask >> i) & 1)


def obs_mask(case_id: int, runs: np.ndarray) -> np.ndarray:
    return np.array([(case_id, int(r)) not in NON_OBSERVED_RUNS for r in runs])


# ─── Feature cache (raw 4-stat per run, Full segment) ────────────────────────
def build_raw_cache(
    signal_df: pd.DataFrame, proc_clean: pd.DataFrame
) -> tuple[dict[tuple[int,int], np.ndarray], dict[int,int]]:
    sensor_indices = mask_sensor_indices(MASK)
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
        full = np.concatenate([extract_features(arrays[s][:base_len]) for s in SENSORS])
        cache[(case_id, run_id)] = full[sensor_indices]

    first_run: dict[int,int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c, r) in cache if c == case_id)
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


# ─── Delta+Meta builder ───────────────────────────────────────────────────────
def build_delta_meta(
    raw_cache: dict[tuple[int,int], np.ndarray],
    first_run: dict[int,int],
    proc_clean: pd.DataFrame,
) -> dict[int, dict]:
    n_sensor_feats = bin(MASK).count("1") * 4
    case_rows: dict[int, list[dict]] = {c: [] for c in CASE_SCOPE}

    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id)
        if key not in raw_cache:
            continue
        raw_vec = raw_cache[key]
        ref_key = (case_id, first_run.get(case_id, run_id))
        ref_vec = raw_cache.get(ref_key, np.zeros(n_sensor_feats))
        delta = raw_vec - ref_vec
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        meta = np.array([float(getattr(row, mf, 0.0)) for mf in META_FEATURES])
        feat = np.concatenate([delta, meta]).astype(np.float32)
        vb   = float(row.VB) if not pd.isna(row.VB) else 0.0
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


def cases_to_df(cases: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for case_id, info in cases.items():
        for i in range(info["n_runs"]):
            rec = {"case_id": case_id, "run": int(info["runs"][i]), "VB": float(info["vb"][i])}
            for k, v in enumerate(info["seq"][i]):
                rec[f"f{k}"] = float(v)
            rows.append(rec)
    return pd.DataFrame(rows)


# ─── Carry-forward ────────────────────────────────────────────────────────────
def run_carry_forward(proc_clean: pd.DataFrame) -> float:
    case_rmses: list[float] = []
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
        case_rmses.append(float(np.sqrt(mean_squared_error(vb_true[obs], vb_pred[obs]))))
    return float(np.mean(case_rmses)) if case_rmses else float("nan")


# ─── Tabular models (Ridge / SVR / XGBoost / RandomForest), StandardScaler where noted ──
def run_tabular_locv(
    feat_df: pd.DataFrame, feature_cols: list[str], model_type: str, seed: int,
) -> float:
    case_rmses: list[float] = []
    use_scaler = model_type in ("Ridge", "SVR")
    for tc in CASE_SCOPE:
        test_df  = feat_df[feat_df["case_id"] == tc].sort_values("run")
        train_df = feat_df[feat_df["case_id"] != tc]
        if test_df.empty or train_df.empty:
            continue
        X_tr_raw = train_df[feature_cols].values
        X_te_raw = test_df[feature_cols].values
        if use_scaler:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)
        else:
            X_tr, X_te = X_tr_raw, X_te_raw
        y_tr = train_df["VB"].values

        if model_type == "Ridge":
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        elif model_type == "SVR":
            model = SVR(kernel="rbf", C=10.0, gamma="scale")
        elif model_type == "XGBoost":
            model = XGBRegressor(**XGB_CFG, random_state=seed, verbosity=0)
        else:  # RandomForest
            model = RandomForestRegressor(**RF_CFG, random_state=seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
        if use_scaler:
            y_pred = np.clip(y_pred, 0.0, None)

        obs = obs_mask(tc, test_df["run"].to_numpy(int))
        y_true = test_df["VB"].to_numpy(float)[obs]
        y_pred_obs = y_pred[obs]
        if len(y_true) == 0:
            continue
        case_rmses.append(float(np.sqrt(mean_squared_error(y_true, y_pred_obs))))
    return float(np.mean(case_rmses)) if case_rmses else float("nan")


# ─── Feature-MLP ──────────────────────────────────────────────────────────────
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


def run_mlp_locv(feat_df: pd.DataFrame, feature_cols: list[str], device: torch.device, seed: int) -> float:
    input_dim = len(feature_cols)
    case_rmses: list[float] = []
    for tc in CASE_SCOPE:
        train_df = feat_df[feat_df["case_id"] != tc]
        test_df  = feat_df[feat_df["case_id"] == tc].sort_values("run")
        if train_df.empty or test_df.empty:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_df[feature_cols].values).astype(np.float32)
        X_te = scaler.transform(test_df[feature_cols].values).astype(np.float32)
        y_tr = train_df["VB"].values.astype(np.float32)
        y_te = test_df["VB"].values

        torch.manual_seed(seed)
        model = FeatMLP(input_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)
        x_t = torch.tensor(X_tr).to(device)
        y_t = torch.tensor(y_tr).to(device)
        model.train()
        for _ in range(MLP_EPOCHS):
            opt.zero_grad()
            loss = ((model(x_t) - y_t) ** 2).mean()
            loss.backward()
            opt.step()
            sch.step()

        model.eval()
        with torch.no_grad():
            y_pred = np.clip(model(torch.tensor(X_te).to(device)).cpu().numpy(), 0.0, None)

        obs = obs_mask(tc, test_df["run"].to_numpy(int))
        if obs.sum() == 0:
            continue
        case_rmses.append(float(np.sqrt(mean_squared_error(y_te[obs], y_pred[obs]))))
    return float(np.mean(case_rmses)) if case_rmses else float("nan")


# ─── RNN model (GRU/LSTM) ─────────────────────────────────────────────────────
class DeltaMetaRNN(nn.Module):
    def __init__(self, input_dim: int, cell_type: str) -> None:
        super().__init__()
        cfg = RNN_CFG
        rnn_cls = nn.GRU if cell_type == "FeatGRU" else nn.LSTM
        self.rnn = rnn_cls(
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
        out_packed, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out).squeeze(-1)


def fit_predict_rnn(
    train_cases: dict[int, dict], test_cases: dict[int, dict],
    input_dim: int, cell_type: str, device: torch.device, seed: int,
) -> dict[int, np.ndarray]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = DeltaMetaRNN(input_dim, cell_type).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=RNN_CFG["lr"], weight_decay=RNN_CFG["weight_decay"])
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
        loss = ((pred[mask_tr] - y_train[mask_tr]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), RNN_CFG["grad_clip"])
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


def run_rnn_locv_seed(
    all_cases: dict[int, dict], input_dim: int, cell_type: str, device: torch.device, seed: int,
) -> float:
    rmses: list[float] = []
    for tc in CASE_SCOPE:
        if tc not in all_cases:
            continue
        train = {c: all_cases[c] for c in CASE_SCOPE if c != tc and c in all_cases}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = fit_predict_rnn(train, {tc: all_cases[tc]}, input_dim, cell_type, device, seed)
        runs = all_cases[tc]["runs"]
        obs  = np.array([(tc, int(r)) not in NON_OBSERVED_RUNS for r in runs])
        y_pred = preds[tc][obs]
        y_true = all_cases[tc]["vb"][obs]
        if len(y_true) == 0:
            continue
        rmses.append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
    return float(np.mean(rmses)) if rmses else float("nan")


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_results(results: dict[str, dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    names = MODELS
    means = [results[m]["mean"] for m in names]
    stds  = [results[m]["std"]  for m in names]
    x = np.arange(len(names))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    best_idx = int(np.argmin(means))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.003,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Observed-VB RMSE")
    ax.set_title("B1_S1: Comparison: Feature-based Model — All sensors, Delta+Meta, Full")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_dir / "comparison_feature_based.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "comparison_feature_based.svg"), bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B1" / "S1" / f"{ts}_comparison_feature_based"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== B1_S1: Comparison: Feature-based Model ===")
    log(f"Sensors: {mask_label(MASK)} (mask={MASK}), Models={MODELS}, Seeds={SEEDS}, Segment=Full")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    log("Building raw feature cache (all sensors, Full)...")
    raw_cache, first_run = build_raw_cache(signal_df, proc_clean)
    all_cases = build_delta_meta(raw_cache, first_run, proc_clean)
    feat_df   = cases_to_df(all_cases)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    input_dim = bin(MASK).count("1") * 4 + len(META_FEATURES)
    feature_cols = [f"f{k}" for k in range(input_dim)]

    results: dict[str, dict] = {}
    t_total = time_mod.time()

    for model_type in MODELS:
        t0 = time_mod.time()
        seeds_to_run = SEEDS if model_type in STOCHASTIC else [0]
        seed_rmses: list[float] = []
        for seed in seeds_to_run:
            if model_type == "CarryForward":
                rmse = run_carry_forward(proc_clean)
            elif model_type in ("Ridge", "SVR", "XGBoost", "RandomForest"):
                rmse = run_tabular_locv(feat_df, feature_cols, model_type, seed)
            elif model_type == "FeatureMLP":
                rmse = run_mlp_locv(feat_df, feature_cols, device, seed)
            else:  # FeatGRU, FeatLSTM
                rmse = run_rnn_locv_seed(all_cases, input_dim, model_type, device, seed)
            seed_rmses.append(rmse)
        mean_ = float(np.mean(seed_rmses))
        std_  = float(np.std(seed_rmses))
        ref_note = f"  (ref B2_S1={REF_B2_S1[model_type]:.6f})" if model_type in REF_B2_S1 else ""
        log(f"  {model_type:<14} mean={mean_:.6f}  std={std_:.6f}  n_seed={len(seed_rmses)}  [{time_mod.time()-t0:.0f}s]{ref_note}")
        results[model_type] = {"mean": mean_, "std": std_, "seed_rmses": seed_rmses, "n_seed": len(seeds_to_run)}

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    agg_rows = [
        {"model": m, "mean": r["mean"], "std": r["std"], "n_seed": r["n_seed"],
         "ref_b2_s1": REF_B2_S1.get(m)}
        for m, r in results.items()
    ]
    pd.DataFrame(agg_rows).to_csv(out_dir / "metrics" / "aggregate.csv", index=False)

    seed_rows = []
    for m, r in results.items():
        for i, v in enumerate(r["seed_rmses"]):
            seed_rows.append({"model": m, "seed_idx": i, "rmse": v})
    pd.DataFrame(seed_rows).to_csv(out_dir / "metrics" / "seed_detail.csv", index=False)

    plot_results(results, out_dir / "figures")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    log(f"{'Model':<14} {'Mean':>10} {'Std':>10} {'B2_S1 ref':>12} {'Match?':>8}")
    log("-" * 60)
    for m in MODELS:
        r = results[m]
        ref = REF_B2_S1.get(m)
        match = "" if ref is None else ("OK" if abs(r["mean"] - ref) < 0.002 else "DIFF")
        log(f"{m:<14} {r['mean']:>10.6f} {r['std']:>10.6f} "
            f"{'' if ref is None else f'{ref:>12.6f}'} {match:>8}")

    best = min(results, key=lambda m: results[m]["mean"])
    log(f"\nBest: {best}  mean={results[best]['mean']:.6f}")

    summary = {
        "experiment": "B1_S1_comparison_feature_based",
        "mask": MASK, "sensor_subset": mask_label(MASK), "segment": "Full",
        "models": MODELS, "seeds": SEEDS,
        "results": results,
        "ref_b2_s1": REF_B2_S1,
        "best_model": best,
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

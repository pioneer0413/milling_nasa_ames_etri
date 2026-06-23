#!/usr/bin/env python3
"""B7_S1: Fixed sensor policy (AC+vT+vS+aS) — individual model performance.

센서 선택을 smcDC·AE_table 제외 → **AC+vT+vS+aS (4센서)** 로 고정하고, 6개 모델의
개별 LOCV observed-VB RMSE를 산출한다. 모든 모델 동일 입력(Delta+Meta, dim=19).

근거: B5_S1(ML 63-mask 전수)·H11(GRU/XGB/RF sweep)에서 smcDC는 best subset 0회·
top-10 빈도 최저·단일 성능 최약, AE_table은 best subset 0회(smcAC와 정보 중복).

모델: Ridge, SVR, XGBoost, RandomForest, FeatGRU, FeatLSTM.
seed: Ridge/SVR 결정론적(1), XGBoost/RandomForest/FeatGRU/FeatLSTM 5-seed.
loss(seq): epsilon-stabilized RMSE.

Output: experiments/executions/B7/S1/{ts}_fixed_sensor_individual/
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
THRESH        = 1e6
N_SENSORS     = len(SENSORS)

FIXED_SUBSET  = "AC+vT+vS+aS"        # smcDC, AE_table 제외
SEEDS_STOCH   = [0, 1, 2, 3, 4]
ML_MODELS     = ["Ridge", "SVR", "XGBoost", "RandomForest"]
SEQ_MODELS    = ["FeatGRU", "FeatLSTM"]
DET_MODELS    = {"Ridge", "SVR"}     # seed 무관

RNN_CFG = dict(hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
               lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0)
XGB_CFG = dict(learning_rate=0.03, n_estimators=600, max_depth=3, min_child_weight=8,
               subsample=0.8, colsample_bytree=0.9, gamma=0.0, reg_alpha=0.0,
               reg_lambda=1.0, n_jobs=4)
RF_CFG  = dict(n_estimators=200, max_depth=None, n_jobs=4)
RMSE_EPS = 1e-8


def label_to_mask(label):
    mask = 0
    for abbr in label.split("+"):
        mask |= (1 << ABBR_TO_IDX[abbr])
    return mask


def parse_signal(value):
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros(1)
    return np.array([np.mean(arr), np.sqrt(np.mean(arr ** 2)), np.std(arr), np.abs(arr).max()])


def preprocess(process):
    df = process.copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"] == min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside"))
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)


def mask_sensor_indices(mask):
    return [i * 4 + j for i in range(N_SENSORS) if (mask >> i) & 1 for j in range(4)]


def build_full_raw_cache(signal_df, proc_clean):
    cache = {}
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
    first_run = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c, r) in cache if c == case_id)
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


def build_feat_df(full_cache, first_run, proc_clean, mask):
    si = mask_sensor_indices(mask)
    rows = []
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        if (case_id, run_id) not in full_cache:
            continue
        raw_vec = full_cache[(case_id, run_id)][si]
        ref_vec = full_cache.get((case_id, first_run.get(case_id, run_id)), np.zeros(24))[si]
        delta = raw_vec - ref_vec
        delta = np.where(np.isfinite(delta), delta, 0.0)
        meta = np.array([float(getattr(row, mf, 0.0)) for mf in META_FEATURES])
        feat = np.concatenate([delta, meta])
        rec = {"case_id": case_id, "run": run_id, "VB": float(row.VB) if not pd.isna(row.VB) else 0.0}
        for k, v in enumerate(feat):
            rec[f"f{k}"] = float(v)
        rows.append(rec)
    df = pd.DataFrame(rows)
    cols = [f"f{k}" for k in range(bin(mask).count("1") * 4 + len(META_FEATURES))]
    return df, cols


def build_sequences(full_cache, first_run, proc_clean, mask):
    si = mask_sensor_indices(mask)
    n_sf = len(si)
    case_rows = {c: [] for c in CASE_SCOPE}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        if (case_id, run_id) not in full_cache:
            continue
        raw_vec = full_cache[(case_id, run_id)][si]
        ref_vec = full_cache.get((case_id, first_run.get(case_id, run_id)), np.zeros(24))[si]
        delta = raw_vec - ref_vec
        delta = np.where(np.isfinite(delta), delta, 0.0)
        meta = np.array([float(getattr(row, mf, 0.0)) for mf in META_FEATURES])
        feat = np.concatenate([delta, meta]).astype(np.float32)
        vb = float(row.VB) if not pd.isna(row.VB) else 0.0
        case_rows[case_id].append({"feat": feat, "vb": vb, "run": run_id})
    cases = {}
    for case_id, rows in case_rows.items():
        if not rows:
            continue
        rs = sorted(rows, key=lambda r: r["run"])
        cases[case_id] = {
            "seq": np.stack([r["feat"] for r in rs]),
            "vb": np.array([r["vb"] for r in rs], dtype=np.float32),
            "runs": np.array([r["run"] for r in rs], dtype=int),
            "n_runs": len(rs),
        }
    return cases, n_sf + len(META_FEATURES)


def fit_ml(model_type, seed):
    if model_type == "Ridge":
        return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    if model_type == "SVR":
        return SVR(kernel="rbf", C=10.0, gamma="scale")
    if model_type == "XGBoost":
        return XGBRegressor(**XGB_CFG, random_state=seed, verbosity=0)
    return RandomForestRegressor(**RF_CFG, random_state=seed)


def predict_ml(model_type, train_df, test_df, cols, seed):
    use_scaler = model_type in ("Ridge", "SVR")
    X_tr = train_df[cols].values
    X_te = test_df[cols].values
    if use_scaler:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
    m = fit_ml(model_type, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X_tr, train_df["VB"].values)
        yp = m.predict(X_te)
    if use_scaler:
        yp = np.clip(yp, 0.0, None)
    return yp


class DeltaMetaRNN(nn.Module):
    def __init__(self, input_dim, cell):
        super().__init__()
        cfg = RNN_CFG
        rnn = nn.GRU if cell == "FeatGRU" else nn.LSTM
        self.rnn = rnn(input_dim, cfg["hidden_size"], cfg["num_layers"], batch_first=True,
                       dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(cfg["hidden_size"], cfg["head_hidden"]),
                                  nn.ReLU(), nn.Linear(cfg["head_hidden"], 1))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.head(out).squeeze(-1)


def fit_predict_seq(train_cases, test_case, input_dim, cell, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    model = DeltaMetaRNN(input_dim, cell).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=RNN_CFG["lr"], weight_decay=RNN_CFG["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=RNN_CFG["epochs"])
    seqs = [i["seq"] for i in train_cases.values()]
    vbs = [i["vb"] for i in train_cases.values()]
    all_vb = np.concatenate(vbs)
    ym, ys = float(all_vb.mean()), max(float(all_vb.std()), 1e-8)
    tens = [torch.tensor(s, dtype=torch.float32) for s in seqs]
    lengths = torch.tensor([len(s) for s in tens])
    x = nn.utils.rnn.pad_sequence(tens, batch_first=True, padding_value=0.0).to(device)
    tgt = [torch.tensor((t - ym) / ys, dtype=torch.float32) for t in vbs]
    y = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=float("nan")).to(device)
    mk = ~torch.isnan(y)
    model.train()
    for _ in range(RNN_CFG["epochs"]):
        opt.zero_grad()
        pred = model(x, lengths)
        loss = torch.sqrt(((pred[mk] - y[mk]) ** 2).mean() + RMSE_EPS)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), RNN_CFG["grad_clip"])
        opt.step(); sch.step()
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(test_case["seq"], dtype=torch.float32).unsqueeze(0).to(device),
                    torch.tensor([test_case["n_runs"]])).squeeze(0).cpu().numpy() * ys + ym
    return out


def obs_mask(case_id, runs):
    return np.array([(case_id, int(r)) not in NON_OBSERVED_RUNS for r in runs])


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B7" / "S1" / f"{ts}_fixed_sensor_individual"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    log_lines = []
    def log(m):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); log_lines.append(line)

    mask = label_to_mask(FIXED_SUBSET)
    log(f"=== B7_S1: Fixed sensor policy {FIXED_SUBSET} (mask={mask}) — individual models ===")

    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    full_cache, first_run = build_full_raw_cache(signal_df, proc_clean)
    feat_df, cols = build_feat_df(full_cache, first_run, proc_clean, mask)
    seq_cases, seq_dim = build_sequences(full_cache, first_run, proc_clean, mask)
    log(f"Clean runs: {len(proc_clean)}, ML feat dim={len(cols)}, seq dim={seq_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    t0 = time_mod.time()

    results = {}        # model -> {"mean","std","seed_rmses","case_rmse"}
    case_rmse_all = {}  # model -> {case: mean rmse}

    # ── ML models ──
    for m in ML_MODELS:
        seeds = [0] if m in DET_MODELS else SEEDS_STOCH
        seed_means, case_acc = [], {c: [] for c in CASE_SCOPE}
        for seed in seeds:
            crmse = {}
            for tc in CASE_SCOPE:
                te = feat_df[feat_df["case_id"] == tc].sort_values("run")
                tr = feat_df[feat_df["case_id"] != tc]
                if te.empty or tr.empty:
                    continue
                yp = predict_ml(m, tr, te, cols, seed)
                runs = te["run"].to_numpy(int); yt = te["VB"].to_numpy(float)
                ob = obs_mask(tc, runs)
                if ob.sum():
                    crmse[tc] = float(np.sqrt(mean_squared_error(yt[ob], yp[ob])))
            seed_means.append(float(np.mean(list(crmse.values()))))
            for c, v in crmse.items():
                case_acc[c].append(v)
        results[m] = {"mean": float(np.mean(seed_means)), "std": float(np.std(seed_means)),
                      "seed_rmses": seed_means, "n_seed": len(seeds)}
        case_rmse_all[m] = {c: float(np.mean(v)) for c, v in case_acc.items() if v}
        log(f"  {m:<13} mean={results[m]['mean']:.6f} std={results[m]['std']:.6f} ({len(seeds)}-seed)")

    # ── Sequence models ──
    for m in SEQ_MODELS:
        seed_means, case_acc = [], {c: [] for c in CASE_SCOPE}
        for seed in SEEDS_STOCH:
            crmse = {}
            for tc in CASE_SCOPE:
                if tc not in seq_cases:
                    continue
                train = {c: seq_cases[c] for c in CASE_SCOPE if c != tc and c in seq_cases}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    yp = fit_predict_seq(train, seq_cases[tc], seq_dim, m, device, seed)
                runs = seq_cases[tc]["runs"]; yt = seq_cases[tc]["vb"]
                ob = obs_mask(tc, runs)
                if ob.sum():
                    crmse[tc] = float(np.sqrt(mean_squared_error(yt[ob], yp[ob])))
            seed_means.append(float(np.mean(list(crmse.values()))))
            for c, v in crmse.items():
                case_acc[c].append(v)
        results[m] = {"mean": float(np.mean(seed_means)), "std": float(np.std(seed_means)),
                      "seed_rmses": seed_means, "n_seed": len(SEEDS_STOCH)}
        case_rmse_all[m] = {c: float(np.mean(v)) for c, v in case_acc.items() if v}
        log(f"  {m:<13} mean={results[m]['mean']:.6f} std={results[m]['std']:.6f} (5-seed)")

    # ── Save + summary ──
    order = sorted(results, key=lambda k: results[k]["mean"])
    log("\n=== SUMMARY (fixed AC+vT+vS+aS, observed-VB RMSE) ===")
    log(f"{'Model':<14}{'Mean':>11}{'Std':>11}{'seeds':>7}")
    log("-" * 43)
    for m in order:
        log(f"{m:<14}{results[m]['mean']:>11.6f}{results[m]['std']:>11.6f}{results[m]['n_seed']:>7}")

    agg = pd.DataFrame([{"model": m, **{k: results[m][k] for k in ("mean", "std", "n_seed")}}
                        for m in order])
    agg.to_csv(out_dir / "metrics" / "aggregate.csv", index=False)
    crows = [{"model": m, "case": c, "rmse": r}
             for m in results for c, r in case_rmse_all[m].items()]
    pd.DataFrame(crows).to_csv(out_dir / "metrics" / "case_rmse.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    means = [results[m]["mean"] for m in order]
    stds = [results[m]["std"] for m in order]
    colors = ["#4878d0" if m in SEQ_MODELS else "#ee854a" for m in order]
    bars = ax.bar(range(len(order)), means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Observed-VB RMSE")
    ax.set_title(f"B7_S1: individual models @ fixed {FIXED_SUBSET}\n(blue=sequence, orange=run-independent)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_dir / "figures" / "individual_rmse.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {"experiment": "B7_S1_fixed_sensor_individual", "fixed_subset": FIXED_SUBSET,
               "mask": mask, "excluded": ["smcDC", "AE_table"],
               "ml_feat_dim": len(cols), "seq_dim": seq_dim,
               "results": results, "ranking": order,
               "execution_dir": str(out_dir), "elapsed_s": time_mod.time() - t0}
    (out_dir / "logs" / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nDone in {time_mod.time()-t0:.0f}s. {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

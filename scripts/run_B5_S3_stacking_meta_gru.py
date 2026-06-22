#!/usr/bin/env python3
"""B5_S3: Stacking Ensemble — FeatGRU Meta-learner over base-model predictions

[B5] Phase 1/2 후속. B5_S2의 단순평균 앙상블(0.103189) 대신, 4개 ML 베이스 모델
(Ridge/SVR/XGBoost/RandomForest, 각자 B5_S1 최적 sensor subset)의 예측을 **run
순서를 따라가는 FeatGRU 메타러너**로 stacking한다.

핵심 아이디어: 베이스 모델은 각 run을 독립적으로 추정하지만, 메타러너(FeatGRU)는
"이전 run들의 베이스 모델 예측 패턴"을 시퀀스로 학습해 보정한다 — feature
engineering(ML)과 run-sequence modeling(DL)의 결합.

Nested LOCV (leakage 방지):
  for outer_test_case tc (15):
    train_pool = 14 cases
    1) Inner LOOCV(train_pool 내부 14-fold)로 OOF 베이스 예측 생성 → 메타러너 학습용
    2) train_pool 전체로 베이스 모델 재학습 → tc에 대한 베이스 예측(메타러너 입력)
    3) 메타 시퀀스: [Ridge_pred, SVR_pred, XGBoost_pred, RandomForest_pred] + Meta(DOC/feed/material)
    4) FeatGRU를 OOF 시퀀스(14 case)로 학습 후 tc 시퀀스 예측
    5) observed-VB RMSE 계산

평가: 5-seed(FeatGRU), 베이스 모델은 고정 seed(결정론적 컨벤션 유지).

Output: experiments/executions/B5/S3/{timestamp}_stacking_meta_gru/
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
BASE_MODELS   = ["Ridge", "SVR", "XGBoost", "RandomForest"]
OWN_BEST = {"Ridge": "AC+vT", "SVR": "AC+vS+aS", "XGBoost": "AC+vS", "RandomForest": "AC+vT+aS"}
INDIVIDUAL_RMSE = {"Ridge": 0.122213, "SVR": 0.109065, "XGBoost": 0.108260, "RandomForest": 0.119037}
AVG_ENSEMBLE_RMSE = 0.103189   # B5_S2 own-best simple-average reference

META_RNN_CFG = dict(
    hidden_size=64, num_layers=2, dropout=0.1, head_hidden=16,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0,
)
XGB_CFG = dict(
    learning_rate=0.03, n_estimators=600, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.9,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, n_jobs=4,
)
RF_CFG  = dict(n_estimators=200, max_depth=None, n_jobs=4, random_state=0)


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
    return [
        i * 4 + j
        for i in range(N_SENSORS) if (mask >> i) & 1
        for j in range(4)
    ]


def build_full_raw_cache(signal_df, proc_clean):
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
        cache[(case_id, run_id)] = np.concatenate(
            [extract_features(arrays[s][:base_len]) for s in SENSORS])
    first_run: dict[int,int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c, r) in cache if c == case_id)
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


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


def fit_model(model_type: str):
    if model_type == "Ridge":
        return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    if model_type == "SVR":
        return SVR(kernel="rbf", C=10.0, gamma="scale")
    if model_type == "XGBoost":
        return XGBRegressor(**XGB_CFG, random_state=0, verbosity=0)
    return RandomForestRegressor(**RF_CFG)


def predict_model(model_type: str, train_df, test_df, feature_cols) -> np.ndarray:
    use_scaler = model_type in ("Ridge", "SVR")
    X_tr_raw = train_df[feature_cols].values
    X_te_raw = test_df[feature_cols].values
    if use_scaler:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)
    else:
        X_tr, X_te = X_tr_raw, X_te_raw
    y_tr = train_df["VB"].values
    model = fit_model(model_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
    if use_scaler:
        y_pred = np.clip(y_pred, 0.0, None)
    return y_pred


def obs_mask(case_id: int, runs: np.ndarray) -> np.ndarray:
    return np.array([(case_id, int(r)) not in NON_OBSERVED_RUNS for r in runs])


# ─── Meta-GRU ─────────────────────────────────────────────────────────────────
class MetaGRU(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        cfg = META_RNN_CFG
        self.gru = nn.GRU(
            input_dim, cfg["hidden_size"], cfg["num_layers"],
            batch_first=True, dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg["hidden_size"], cfg["head_hidden"]), nn.ReLU(),
            nn.Linear(cfg["head_hidden"], 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out).squeeze(-1)


def fit_predict_meta_gru(train_cases: dict, test_case: dict, input_dim: int, device, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = MetaGRU(input_dim).to(device)
    cfg = META_RNN_CFG
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

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
    for _ in range(cfg["epochs"]):
        optimizer.zero_grad()
        pred = model(x_train, lengths)
        loss = ((pred[mask_tr] - y_train[mask_tr]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        out = model(
            torch.tensor(test_case["seq"], dtype=torch.float32).unsqueeze(0).to(device),
            torch.tensor([test_case["n_runs"]]),
        ).squeeze(0).cpu().numpy()
    return out * y_std + y_mean


# ─── Main nested-LOCV stacking ────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B5" / "S3" / f"{ts}_stacking_meta_gru"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== B5_S3: Stacking Ensemble — FeatGRU Meta-learner ===")
    log(f"Base models (own-best subset): {OWN_BEST}")
    log(f"Meta-learner: FeatGRU(input_dim=4 base preds + 3 meta = 7), seeds={SEEDS}")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    full_cache, first_run = build_full_raw_cache(signal_df, proc_clean)
    log(f"Clean runs: {len(proc_clean)}, cache size: {len(full_cache)}")

    feat_dfs = {
        m: build_feat_df_for_mask(full_cache, first_run, proc_clean, label_to_mask(OWN_BEST[m]))
        for m in BASE_MODELS
    }
    meta_lookup = proc_clean.set_index(["case", "run"])[META_FEATURES].to_dict("index")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    input_dim = len(BASE_MODELS) + len(META_FEATURES)   # 4 + 3 = 7
    t_total = time_mod.time()

    # seed -> {tc: rmse}; also keep per-case trajectory for seed=0
    seed_case_rmses: dict[int, list[float]] = {s: [] for s in SEEDS}
    traj_by_case: dict[int, dict] = {}

    for oi, tc in enumerate(CASE_SCOPE):
        t0 = time_mod.time()
        train_pool = [c for c in CASE_SCOPE if c != tc]

        # 1) Inner LOOCV -> OOF base predictions for train_pool
        oof_preds: dict[tuple[int,int], dict[str, float]] = {}
        for ic in train_pool:
            inner_train = [c for c in train_pool if c != ic]
            for model_type in BASE_MODELS:
                df, cols = feat_dfs[model_type]
                train_df = df[df["case_id"].isin(inner_train)]
                test_df  = df[df["case_id"] == ic].sort_values("run")
                if train_df.empty or test_df.empty:
                    continue
                y_pred = predict_model(model_type, train_df, test_df, cols)
                for run_id, p in zip(test_df["run"].to_numpy(int), y_pred):
                    oof_preds.setdefault((ic, run_id), {})[model_type] = float(p)

        # 2) Final base models on full train_pool -> predictions for tc
        test_preds: dict[int, dict[str, float]] = {}
        for model_type in BASE_MODELS:
            df, cols = feat_dfs[model_type]
            train_df = df[df["case_id"].isin(train_pool)]
            test_df  = df[df["case_id"] == tc].sort_values("run")
            y_pred = predict_model(model_type, train_df, test_df, cols)
            for run_id, p in zip(test_df["run"].to_numpy(int), y_pred):
                test_preds.setdefault(run_id, {})[model_type] = float(p)

        # 3) Build meta sequences (train: OOF, test: tc)
        def make_meta_case(case_id: int, run_pred_map: dict[int, dict[str, float]]) -> dict:
            runs_sorted = sorted(run_pred_map.keys())
            feats, vbs = [], []
            for r in runs_sorted:
                preds = run_pred_map[r]
                base_vec = [preds.get(m, 0.0) for m in BASE_MODELS]
                meta_row = meta_lookup.get((case_id, r), {mf: 0.0 for mf in META_FEATURES})
                meta_vec = [float(meta_row.get(mf, 0.0)) for mf in META_FEATURES]
                feats.append(base_vec + meta_vec)
                vb_row = proc_clean[(proc_clean["case"] == case_id) & (proc_clean["run"] == r)]
                vbs.append(float(vb_row["VB"].iloc[0]) if not vb_row.empty else 0.0)
            return {
                "seq": np.array(feats, dtype=np.float32),
                "vb": np.array(vbs, dtype=np.float32),
                "runs": np.array(runs_sorted, dtype=int),
                "n_runs": len(runs_sorted),
            }

        meta_train_cases: dict[int, dict] = {}
        for ic in train_pool:
            run_map = {r: p for (c, r), p in oof_preds.items() if c == ic}
            if run_map:
                meta_train_cases[ic] = make_meta_case(ic, run_map)
        meta_test_case = make_meta_case(tc, test_preds)

        # 4) Train FeatGRU meta-learner (per seed) and predict on tc
        for seed in SEEDS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta_pred = fit_predict_meta_gru(meta_train_cases, meta_test_case, input_dim, device, seed)
            runs = meta_test_case["runs"]
            y_true = meta_test_case["vb"]
            obs = obs_mask(tc, runs)
            if obs.sum() == 0:
                continue
            rmse = float(np.sqrt(mean_squared_error(y_true[obs], meta_pred[obs])))
            seed_case_rmses[seed].append(rmse)
            if seed == 0:
                traj_by_case[tc] = {"run": runs, "y_true": y_true, "y_pred": meta_pred}

        log(f"  [{oi+1}/{len(CASE_SCOPE)}] tc={tc}  [{time_mod.time()-t0:.0f}s]")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    seed_means = [float(np.mean(v)) for v in seed_case_rmses.values()]
    overall_mean = float(np.mean(seed_means))
    overall_std  = float(np.std(seed_means))

    log("\n=== SUMMARY ===")
    log(f"Stacking+FeatGRU meta-learner: mean={overall_mean:.6f}  std={overall_std:.6f}  (5-seed)")
    log(f"vs Simple-average ensemble (B5_S2): {AVG_ENSEMBLE_RMSE:.6f}  "
        f"Δ={overall_mean - AVG_ENSEMBLE_RMSE:+.6f} ({'개선' if overall_mean < AVG_ENSEMBLE_RMSE else '악화'})")
    log(f"vs Best individual base model (XGBoost): {INDIVIDUAL_RMSE['XGBoost']:.6f}  "
        f"Δ={overall_mean - INDIVIDUAL_RMSE['XGBoost']:+.6f}")

    # ── Plot: trajectories (seed=0) ──────────────────────────────────────────
    cases = sorted(traj_by_case.keys())
    ncols = 5
    nrows = int(np.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for idx, case_id in enumerate(cases):
        ax = axes[idx // ncols][idx % ncols]
        d = traj_by_case[case_id]
        runs, y_true, y_pred = d["run"], d["y_true"], d["y_pred"]
        is_obs = np.array([(case_id, int(r)) not in NON_OBSERVED_RUNS for r in runs])
        ax.plot(runs, y_pred, "-", color="crimson", marker="o", markersize=4, linewidth=1.3, label="Meta-GRU pred")
        ax.plot(runs[is_obs], y_true[is_obs], "-", color="black", marker="o", markersize=4, linewidth=1.0, label="VB (observed)")
        if (~is_obs).any():
            ax.plot(runs[~is_obs], y_true[~is_obs], "o", mfc="none", mec="gray", markersize=5, linewidth=0, label="VB (interpolated)")
        ax.set_title(f"Case {case_id}", fontsize=10)
        ax.set_xlabel("run", fontsize=8); ax.set_ylabel("VB", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")
    for idx in range(len(cases), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(f"B5_S3: Stacking (FeatGRU meta-learner) — RMSE={overall_mean:.6f}", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(out_dir / "figures" / "trajectory_stacking_metagru.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "figures" / "trajectory_stacking_metagru.svg"), bbox_inches="tight")
    plt.close(fig)

    # ── Plot: comparison bar ─────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    names = list(INDIVIDUAL_RMSE.keys()) + ["Ensemble(avg)", "Stacking(FeatGRU)"]
    vals  = list(INDIVIDUAL_RMSE.values()) + [AVG_ENSEMBLE_RMSE, overall_mean]
    colors = ["#4878d0"]*4 + ["#ee854a", "#d65f5f"]
    bars = ax2.bar(range(len(names)), vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, f"{v:.4f}", ha="center", fontsize=8)
    ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylabel("Observed-VB RMSE")
    ax2.set_title("B5_S3: Individual / Average-ensemble / Stacking(FeatGRU)")
    ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig2.savefig(str(out_dir / "figures" / "comparison_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    summary = {
        "experiment": "B5_S3_stacking_meta_gru",
        "base_models_own_best": OWN_BEST,
        "individual_rmse": INDIVIDUAL_RMSE,
        "avg_ensemble_rmse": AVG_ENSEMBLE_RMSE,
        "stacking_meta_gru": {"mean": overall_mean, "std": overall_std, "seed_means": seed_means},
        "execution_dir": str(out_dir),
        "total_elapsed_s": time_mod.time() - t_total,
    }
    (out_dir / "logs" / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

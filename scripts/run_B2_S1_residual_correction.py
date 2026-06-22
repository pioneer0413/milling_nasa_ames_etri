#!/usr/bin/env python3
"""B2_S1 (variant): Residual Correction — FeatLSTM backbone + residual model

FeatLSTM(AC+vT+vS, Delta+Meta, Full)을 trajectory backbone으로 고정하고, 남은
오차(residual)만 별도의 모델(Ridge/RandomForest/2-layer MLP)이 보정한다.

  base_pred       = FeatLSTM(sequence)
  residual_model  = f(base_pred, delta features)   <- case_id, process info, previous true VB 절대 미포함
  final_pred      = base_pred + residual_model(...)

Leakage 방지를 위해 nested LOCV로 OOF residual을 생성:

  for outer_test_case tc (15):
      train_pool = 14 cases
      1) train_pool 전체로 FeatLSTM 학습 -> tc에 대한 base_pred (실제 평가용)
      2) train_pool 내부 14-fold inner LOOCV
         - case k 제외 학습 -> k에 대한 OOF base_pred 생성
         - residual_k = y_true_k - OOF_pred_k  (run 단위)
      3) OOF residual들을 모아 residual model(Ridge/RF/MLP) 학습
      4) tc에 대해 final_pred = base_pred_tc + residual_model.predict(base_pred_tc, delta+meta_tc)
      5) observed-VB RMSE 계산 (base-only / corrected 둘 다)

Compute 비용 때문에 backbone은 seed=0 단일 시드로 진행(공식 5-seed 결과와는
프로토콜이 다름 — seed=0 base-only가 이 실행의 fair baseline).

Output: experiments/executions/B2/S1/{timestamp}_residual_correction/
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
THRESH        = 1e6
N_SENSORS     = len(SENSORS)
MASK          = 13   # AC+vT+vS — top1 config
BACKBONE      = "FeatLSTM"
BACKBONE_SEED = 0
RESIDUAL_MODELS = ["Ridge", "RandomForest", "ResidualMLP"]

REF_OFFICIAL_5SEED = {"FeatGRU": 0.095122, "FeatLSTM": 0.092217}   # official, for context only

RNN_CFG = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0,
)
RF_RESID_CFG  = dict(n_estimators=100, max_depth=3, n_jobs=4, random_state=0)
MLP_RESID_CFG = dict(
    hidden_dims=(64, 32), lr=1e-3, weight_decay=1e-4, epochs=500,
    grad_clip=1.0, seed=0,
)


# ─── Utils (identical to run_B2_S1_top1_trajectory_plot.py) ──────────────────
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


def obs_mask(case_id: int, runs: np.ndarray) -> np.ndarray:
    return np.array([(case_id, int(r)) not in NON_OBSERVED_RUNS for r in runs])


def build_raw_cache(signal_df, proc_clean):
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


def build_delta_meta(raw_cache, first_run, proc_clean):
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


# ─── RNN backbone ─────────────────────────────────────────────────────────────
class DeltaMetaRNN(nn.Module):
    def __init__(self, input_dim: int, cell_type: str) -> None:
        super().__init__()
        cfg = RNN_CFG
        rnn_cls = nn.GRU if cell_type == "FeatGRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_dim, cfg["hidden_size"], cfg["num_layers"],
            batch_first=True, dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg["hidden_size"], cfg["head_hidden"]), nn.ReLU(),
            nn.Linear(cfg["head_hidden"], 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out).squeeze(-1)


def fit_predict_rnn(train_cases, test_cases, input_dim, cell_type, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = DeltaMetaRNN(input_dim, cell_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=RNN_CFG["lr"], weight_decay=RNN_CFG["weight_decay"])
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


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_predict_residual_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    cfg = MLP_RESID_CFG
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    np.random.seed(cfg["seed"])

    x_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train)
    x_test_s = x_scaler.transform(x_test)
    y_mean = float(np.mean(y_train))
    y_std = max(float(np.std(y_train)), 1e-8)
    y_train_s = (y_train - y_mean) / y_std

    x_t = torch.tensor(x_train_s, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train_s, dtype=torch.float32, device=device)
    model = ResidualMLP(x_train.shape[1], cfg["hidden_dims"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    model.train()
    for _ in range(cfg["epochs"]):
        optimizer.zero_grad()
        pred = model(x_t)
        loss = ((pred - y_t) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        x_te = torch.tensor(x_test_s, dtype=torch.float32, device=device)
        pred = model(x_te).cpu().numpy()
    return pred * y_std + y_mean


def make_residual_model(name: str):
    if name == "Ridge":
        return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    if name == "RandomForest":
        return RandomForestRegressor(**RF_RESID_CFG)
    raise ValueError(f"Unknown sklearn residual model: {name}")


# ─── Main nested-LOCV residual correction ────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B2" / "S1" / f"{ts}_residual_correction"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== B2_S1: Residual Correction — FeatLSTM backbone + residual model ===")
    log(f"Backbone: {BACKBONE} (seed={BACKBONE_SEED}), AC+vT+vS, Delta+Meta(15-dim), Full")
    log(f"Residual models: {RESIDUAL_MODELS}")
    log("Residual features: base_pred + delta-only(12-dim). NO process info, NO case_id, NO previous true VB.")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    raw_cache, first_run = build_raw_cache(signal_df, proc_clean)
    all_cases = build_delta_meta(raw_cache, first_run, proc_clean)
    log(f"Clean runs: {len(proc_clean)}, cache size: {len(raw_cache)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    input_dim = bin(MASK).count("1") * 4 + len(META_FEATURES)

    base_only_case_rmses: dict[int, float] = {}
    corrected_case_rmses: dict[str, dict[int, float]] = {m: {} for m in RESIDUAL_MODELS}
    n_inner_trainings = 0
    t_total = time_mod.time()

    for oi, tc in enumerate(CASE_SCOPE):
        t0 = time_mod.time()
        train_pool = [c for c in CASE_SCOPE if c != tc]

        # 1) base_pred for outer test case (train on full train_pool)
        train_pool_cases = {c: all_cases[c] for c in train_pool}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_pred_test = fit_predict_rnn(
                train_pool_cases, {tc: all_cases[tc]}, input_dim, BACKBONE, device, BACKBONE_SEED)[tc]

        runs_test = all_cases[tc]["runs"]
        y_true_test = all_cases[tc]["vb"]
        obs_test = obs_mask(tc, runs_test)
        base_only_case_rmses[tc] = float(np.sqrt(mean_squared_error(
            y_true_test[obs_test], base_pred_test[obs_test])))

        # 2) inner LOOCV -> OOF base predictions + residuals for train_pool
        oof_X, oof_y = [], []
        for ic in train_pool:
            inner_train = [c for c in train_pool if c != ic]
            inner_train_cases = {c: all_cases[c] for c in inner_train}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oof_pred_ic = fit_predict_rnn(
                    inner_train_cases, {ic: all_cases[ic]}, input_dim, BACKBONE, device, BACKBONE_SEED)[ic]
            n_inner_trainings += 1
            seq_ic = all_cases[ic]["seq"]          # (n_runs, 15) delta+meta backbone features
            resid_feat_ic = seq_ic[:, : -len(META_FEATURES)]  # remove DOC/feed/material for residual learner
            y_true_ic = all_cases[ic]["vb"]
            residual_ic = y_true_ic - oof_pred_ic
            for i in range(len(oof_pred_ic)):
                oof_X.append(np.concatenate([[oof_pred_ic[i]], resid_feat_ic[i]]))
                oof_y.append(residual_ic[i])
        oof_X = np.array(oof_X, dtype=np.float64)
        oof_y = np.array(oof_y, dtype=np.float64)

        # 3) test features (base_pred_test + delta+meta of tc)
        seq_test = all_cases[tc]["seq"][:, : -len(META_FEATURES)]
        X_test = np.concatenate(
            [base_pred_test.reshape(-1, 1), seq_test], axis=1)

        # 4) train each residual model on OOF, predict + correct on tc
        for resid_name in RESIDUAL_MODELS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if resid_name == "ResidualMLP":
                    resid_pred_test = fit_predict_residual_mlp(oof_X, oof_y, X_test, device)
                else:
                    resid_model = make_residual_model(resid_name)
                    resid_model.fit(oof_X, oof_y)
                    resid_pred_test = resid_model.predict(X_test)
            final_pred_test = base_pred_test + resid_pred_test
            corrected_case_rmses[resid_name][tc] = float(np.sqrt(mean_squared_error(
                y_true_test[obs_test], final_pred_test[obs_test])))

        log(f"  [{oi+1}/{len(CASE_SCOPE)}] tc={tc}  base={base_only_case_rmses[tc]:.4f}  "
            f"corrected={ {m: round(corrected_case_rmses[m][tc],4) for m in RESIDUAL_MODELS} }  "
            f"[{time_mod.time()-t0:.0f}s]")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    base_mean = float(np.mean(list(base_only_case_rmses.values())))
    log("\n=== SUMMARY (seed=0, nested-OOF) ===")
    log(f"Base-only (FeatLSTM, seed=0):        mean={base_mean:.6f}")
    log(f"  (official 5-seed reference: {REF_OFFICIAL_5SEED['FeatLSTM']:.6f} — different protocol, context only)")
    for resid_name in RESIDUAL_MODELS:
        corrected_mean = float(np.mean(list(corrected_case_rmses[resid_name].values())))
        delta = corrected_mean - base_mean
        log(f"Corrected ({resid_name:<12}): mean={corrected_mean:.6f}  Δ vs base={delta:+.6f} "
            f"({'개선' if delta < 0 else '악화'})")

    # ── Save per-case CSV ─────────────────────────────────────────────────────
    rows = []
    for tc in CASE_SCOPE:
        row = {"case": tc, "base_only": base_only_case_rmses[tc]}
        for resid_name in RESIDUAL_MODELS:
            row[f"corrected_{resid_name}"] = corrected_case_rmses[resid_name][tc]
        rows.append(row)
    case_df = pd.DataFrame(rows)
    case_df.to_csv(out_dir / "metrics" / "residual_correction_per_case.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["Base-only\n(FeatLSTM)"] + [f"Corrected\n({m})" for m in RESIDUAL_MODELS]
    vals = [base_mean] + [float(np.mean(list(corrected_case_rmses[m].values()))) for m in RESIDUAL_MODELS]
    colors = ["#4878d0", "#d65f5f", "#ee854a", "#6acc65"]
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel("Observed-VB RMSE (seed=0, nested-OOF)")
    ax.set_title("B2_S1: Residual Correction on FeatLSTM backbone")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_dir / "figures" / "residual_correction_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Per-case comparison plot (base vs best corrected) ───────────────────
    best_resid = min(RESIDUAL_MODELS, key=lambda m: np.mean(list(corrected_case_rmses[m].values())))
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    x = np.arange(len(CASE_SCOPE))
    w = 0.35
    ax2.bar(x - w/2, [base_only_case_rmses[c] for c in CASE_SCOPE], w, label="Base-only", color="#4878d0", alpha=0.85)
    ax2.bar(x + w/2, [corrected_case_rmses[best_resid][c] for c in CASE_SCOPE], w,
            label=f"Corrected ({best_resid})", color="#d65f5f", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels([str(c) for c in CASE_SCOPE])
    ax2.set_xlabel("Case"); ax2.set_ylabel("Observed-VB RMSE")
    ax2.set_title(f"Per-case: Base-only vs Corrected ({best_resid})")
    ax2.legend(); ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig2.savefig(str(out_dir / "figures" / "per_case_base_vs_corrected.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    summary = {
        "experiment": "B2_S1_residual_correction",
        "backbone": BACKBONE, "backbone_seed": BACKBONE_SEED,
        "residual_features": "base_pred + delta-only (process info excluded)",
        "residual_models": RESIDUAL_MODELS,
        "base_only_mean": base_mean,
        "base_only_case_rmses": base_only_case_rmses,
        "corrected_means": {m: float(np.mean(list(corrected_case_rmses[m].values()))) for m in RESIDUAL_MODELS},
        "corrected_case_rmses": corrected_case_rmses,
        "ref_official_5seed": REF_OFFICIAL_5SEED,
        "best_residual_model": best_resid,
        "n_inner_trainings": n_inner_trainings,
        "execution_dir": str(out_dir),
        "total_elapsed_s": time_mod.time() - t_total,
    }
    (out_dir / "logs" / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    log(f"\nBest residual model: {best_resid}")
    log(f"Done. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

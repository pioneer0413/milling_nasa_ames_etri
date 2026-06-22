#!/usr/bin/env python3
"""B2_S1 (addendum): Residual Correction — Prediction Trajectory plot

run_B2_S1_residual_correction.py와 동일한 nested-OOF 파이프라인을 재실행하되,
이번엔 case별 run-level 예측(base_pred, corrected_pred, y_true)을 저장해
trajectory를 시각화한다. Residual model은 best였던 XGBoost만 사용.

[[feedback_always_plot_vb_trajectory]] 컨벤션 적용 — 평가 시 항상 trajectory 산출.

Output: experiments/executions/B2/S1/{timestamp}_residual_correction_trajectory/
"""
from __future__ import annotations

import sys
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
EXCLUDED_RUNS = {(2, 1), (12, 1)}
NON_OBSERVED_RUNS = {
    (1,2),(1,3),(1,5),(1,16),(2,6),(8,4),(11,7),(11,14),(11,17),
    (12,4),(12,10),(13,1),(13,2),(14,1),(14,5),(15,1),(16,1),(16,2),(16,4),
}
META_FEATURES = ["DOC", "feed", "material"]
THRESH        = 1e6
N_SENSORS     = len(SENSORS)
MASK          = 13   # AC+vT+vS
BACKBONE      = "FeatLSTM"
BACKBONE_SEED = 0

RNN_CFG = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0,
)
XGB_RESID_CFG = dict(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=4)


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
        loss = torch.sqrt(((pred[mask_tr] - y_train[mask_tr]) ** 2).mean() + 1e-8)
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


def obs_mask(case_id: int, runs: np.ndarray) -> np.ndarray:
    return np.array([(case_id, int(r)) not in NON_OBSERVED_RUNS for r in runs])


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_trajectories(traj: dict[int, dict], out_dir: Path) -> None:
    cases = sorted(traj.keys())
    ncols = 5
    nrows = int(np.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, case_id in enumerate(cases):
        ax = axes[idx // ncols][idx % ncols]
        d = traj[case_id]
        runs, y_true = d["run"], d["y_true"]
        base_pred, corr_pred = d["base_pred"], d["corrected_pred"]
        is_obs = np.array([(case_id, int(r)) not in NON_OBSERVED_RUNS for r in runs])

        ax.plot(runs, base_pred, "-", color="steelblue", marker="o", markersize=4,
                linewidth=1.2, label="Base (FeatLSTM)")
        ax.plot(runs, corr_pred, "-", color="crimson", marker="s", markersize=4,
                linewidth=1.2, label="Corrected (XGBoost)")
        ax.plot(runs[is_obs], y_true[is_obs], "-", color="black", marker="o",
                markersize=4, linewidth=1.0, label="VB (observed)")
        if (~is_obs).any():
            ax.plot(runs[~is_obs], y_true[~is_obs], "o", mfc="none", mec="gray",
                    markersize=5, linewidth=0, label="VB (interpolated)")

        ax.set_title(f"Case {case_id}  (base={d['base_rmse']:.3f}, corr={d['corr_rmse']:.3f})", fontsize=9)
        ax.set_xlabel("run", fontsize=8)
        ax.set_ylabel("VB", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")

    for idx in range(len(cases), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("B2_S1: Residual Correction — Base vs Corrected(XGBoost) vs True (seed=0, nested-OOF)", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(out_dir / "trajectory_residual_correction.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "trajectory_residual_correction.svg"), bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B2" / "S1" / f"{ts}_residual_correction_trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    raw_cache, first_run = build_raw_cache(signal_df, proc_clean)
    all_cases = build_delta_meta(raw_cache, first_run, proc_clean)
    print(f"Clean runs: {len(proc_clean)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = bin(MASK).count("1") * 4 + len(META_FEATURES)

    traj: dict[int, dict] = {}

    for oi, tc in enumerate(CASE_SCOPE):
        train_pool = [c for c in CASE_SCOPE if c != tc]
        train_pool_cases = {c: all_cases[c] for c in train_pool}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_pred_test = fit_predict_rnn(
                train_pool_cases, {tc: all_cases[tc]}, input_dim, BACKBONE, device, BACKBONE_SEED)[tc]

        runs_test = all_cases[tc]["runs"]
        y_true_test = all_cases[tc]["vb"]
        obs_test = obs_mask(tc, runs_test)

        oof_X, oof_y = [], []
        for ic in train_pool:
            inner_train = [c for c in train_pool if c != ic]
            inner_train_cases = {c: all_cases[c] for c in inner_train}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oof_pred_ic = fit_predict_rnn(
                    inner_train_cases, {ic: all_cases[ic]}, input_dim, BACKBONE, device, BACKBONE_SEED)[ic]
            seq_ic = all_cases[ic]["seq"]
            y_true_ic = all_cases[ic]["vb"]
            residual_ic = y_true_ic - oof_pred_ic
            for i in range(len(oof_pred_ic)):
                oof_X.append(np.concatenate([[oof_pred_ic[i]], seq_ic[i]]))
                oof_y.append(residual_ic[i])
        oof_X = np.array(oof_X, dtype=np.float64)
        oof_y = np.array(oof_y, dtype=np.float64)

        seq_test = all_cases[tc]["seq"]
        X_test = np.concatenate([base_pred_test.reshape(-1, 1), seq_test], axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resid_model = XGBRegressor(**XGB_RESID_CFG, random_state=0, verbosity=0)
            resid_model.fit(oof_X, oof_y)
            resid_pred_test = resid_model.predict(X_test)
        corrected_pred_test = base_pred_test + resid_pred_test

        base_rmse = float(np.sqrt(mean_squared_error(y_true_test[obs_test], base_pred_test[obs_test])))
        corr_rmse = float(np.sqrt(mean_squared_error(y_true_test[obs_test], corrected_pred_test[obs_test])))

        traj[tc] = {
            "run": runs_test, "y_true": y_true_test,
            "base_pred": base_pred_test, "corrected_pred": corrected_pred_test,
            "base_rmse": base_rmse, "corr_rmse": corr_rmse,
        }
        print(f"[{oi+1}/{len(CASE_SCOPE)}] tc={tc}  base={base_rmse:.4f}  corrected={corr_rmse:.4f}")

    plot_trajectories(traj, out_dir)

    rows = []
    for tc, d in traj.items():
        for i in range(len(d["run"])):
            rows.append({
                "case": tc, "run": int(d["run"][i]), "y_true": float(d["y_true"][i]),
                "base_pred": float(d["base_pred"][i]), "corrected_pred": float(d["corrected_pred"][i]),
            })
    pd.DataFrame(rows).to_csv(out_dir / "residual_correction_trajectory_data.csv", index=False)

    print(f"\nDone. Figures in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}")


if __name__ == "__main__":
    main()

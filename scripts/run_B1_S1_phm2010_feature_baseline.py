#!/usr/bin/env python3
"""B1_S1: PHM2010 Feature Baseline

PHM2010 데이터셋 첫 번째 baseline 실험.
feature_vector.csv의 70개 features (7 sensors × 10 stats)로 LOCV-3 수행.

Target: VB_max = max(VB_flute_1, VB_flute_2, VB_flute_3) [µm]
Protocol: LOCV-3 (case 1, 4, 6), 모든 run 관측됨 (NON_OBSERVED_RUNS 없음)

Models:
  CF        : Carry-Forward (VB_pred[t] = VB_true[t-1])
  RL        : RunIndex Linear
  Ridge     : RidgeCV (70 features)
  RF        : RandomForestRegressor (70 features)
  SVR       : SVR RBF (70 features)
  MLP_Feat  : 2-layer MLP (70→128→64→1)
  FeatRNN   : RNN over run sequence (70 features/step, 315 steps)
  FeatLSTM  : LSTM over run sequence (70 features/step, 315 steps)
  FeatGRU   : GRU over run sequence (70 features/step, 315 steps)

Output: experiments/executions/B1/S1/{timestamp}_phm2010_feature_baseline/
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
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE = [1, 4, 6]
SEEDS      = [0, 1, 2, 3, 4]

FEAT_SENSORS = ["force_x", "force_y", "force_z", "vib_x", "vib_y", "vib_z", "ae_rms"]
FEAT_STATS   = ["mean", "std", "var", "max", "min", "rms", "peak2peak", "sum", "kurtosis", "skewness"]
FEAT_COLS    = [f"{s}__{st}" for s in FEAT_SENSORS for st in FEAT_STATS]  # 70 features

# Sequence model hyperparams (315 runs/case — longer sequences than NASA)
SEQ_HIDDEN    = 128
SEQ_LAYERS    = 2
SEQ_DROPOUT   = 0.1
SEQ_HEAD_HID  = 32
SEQ_EPOCHS    = 150
SEQ_LR        = 1e-3
SEQ_WD        = 1e-4
SEQ_GRAD_CLIP = 1.0

# Feature MLP hyperparams
MLP_EPOCHS = 200
MLP_LR     = 1e-3
MLP_HIDDEN = [128, 64]


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    process_df = pd.read_csv(ROOT / "datasets/phm2010/process_info.csv")
    feat_df    = pd.read_csv(ROOT / "datasets/phm2010/feature_vector.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    feat_df    = feat_df[feat_df["case"].isin(CASE_SCOPE)].copy()
    return process_df, feat_df


def build_merged(process_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    """Merge process info and features; add VB_max target."""
    df = pd.merge(process_df[["case", "run", "VB_flute_1", "VB_flute_2", "VB_flute_3"]],
                  feat_df[["case", "run"] + FEAT_COLS],
                  on=["case", "run"])
    df["VB_max"] = df[["VB_flute_1", "VB_flute_2", "VB_flute_3"]].max(axis=1)
    df = df.sort_values(["case", "run"]).reset_index(drop=True)
    return df


# ─── Model runners ────────────────────────────────────────────────────────────
def run_cf(df: pd.DataFrame) -> tuple[float, dict[int, float]]:
    case_rmses: dict[int, float] = {}
    for tc in CASE_SCOPE:
        sub = df[df["case"] == tc].sort_values("run")
        vb_true = sub["VB_max"].to_numpy(float)
        vb_pred = np.concatenate([[0.0], vb_true[:-1]])
        case_rmses[tc] = float(np.sqrt(mean_squared_error(vb_true, vb_pred)))
    return float(np.mean(list(case_rmses.values()))), case_rmses


def run_rl(df: pd.DataFrame) -> tuple[float, dict[int, float]]:
    case_rmses: dict[int, float] = {}
    for tc in CASE_SCOPE:
        tr = df[df["case"] != tc]
        te = df[df["case"] == tc].sort_values("run")
        m = LinearRegression().fit(tr[["run"]].values, tr["VB_max"].values)
        y_pred = np.clip(m.predict(te[["run"]].values), 0.0, None)
        case_rmses[tc] = float(np.sqrt(mean_squared_error(te["VB_max"].values, y_pred)))
    return float(np.mean(list(case_rmses.values()))), case_rmses


def _feat_locv(df: pd.DataFrame, model_fn) -> tuple[float, dict[int, float]]:
    case_rmses: dict[int, float] = {}
    for tc in CASE_SCOPE:
        tr = df[df["case"] != tc]
        te = df[df["case"] == tc].sort_values("run")
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr[FEAT_COLS].values)
        X_te = scaler.transform(te[FEAT_COLS].values)
        y_tr = tr["VB_max"].values
        y_te = te["VB_max"].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = model_fn()
            m.fit(X_tr, y_tr)
            y_pred = np.clip(m.predict(X_te), 0.0, None)
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    return float(np.mean(list(case_rmses.values()))), case_rmses


def run_ridge(df: pd.DataFrame) -> tuple[float, dict[int, float]]:
    return _feat_locv(df, lambda: RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]))


def run_rf(df: pd.DataFrame, seed: int) -> tuple[float, dict[int, float]]:
    return _feat_locv(df, lambda: RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=seed, n_jobs=4))


def run_svr(df: pd.DataFrame) -> tuple[float, dict[int, float]]:
    return _feat_locv(df, lambda: SVR(kernel="rbf", C=10.0, gamma="scale"))


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


def run_mlp_feat(df: pd.DataFrame, device: torch.device, seed: int) -> tuple[float, dict[int, float]]:
    input_dim = len(FEAT_COLS)
    case_rmses: dict[int, float] = {}
    for tc in CASE_SCOPE:
        tr = df[df["case"] != tc]
        te = df[df["case"] == tc].sort_values("run")
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr[FEAT_COLS].values).astype(np.float32)
        X_te = scaler.transform(te[FEAT_COLS].values).astype(np.float32)
        y_tr = tr["VB_max"].values.astype(np.float32)
        y_te = te["VB_max"].values

        torch.manual_seed(seed)
        model = FeatMLP(input_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)
        x_t = torch.tensor(X_tr).to(device)
        y_t = torch.tensor(y_tr).to(device)
        model.train()
        for _ in range(MLP_EPOCHS):
            opt.zero_grad()
            ((model(x_t) - y_t) ** 2).mean().backward()
            opt.step()
            sch.step()

        model.eval()
        with torch.no_grad():
            y_pred = np.clip(model(torch.tensor(X_te).to(device)).cpu().numpy(), 0.0, None)
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    return float(np.mean(list(case_rmses.values()))), case_rmses


# ─── Feature Sequence Models (RNN / LSTM / GRU) ───────────────────────────────
_CELL_CLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


class FeatureSeqModel(nn.Module):
    def __init__(self, input_dim: int, cell: str = "gru") -> None:
        super().__init__()
        drop = SEQ_DROPOUT if SEQ_LAYERS > 1 else 0.0
        self.is_lstm = cell == "lstm"
        self.rnn = _CELL_CLS[cell](input_dim, SEQ_HIDDEN, SEQ_LAYERS,
                                   batch_first=True, dropout=drop)
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


# keep alias for backward compatibility
FeatureSeqGRU = FeatureSeqModel


def _build_seq_cases(df: pd.DataFrame, scaler: StandardScaler) -> dict[int, dict]:
    cases: dict[int, dict] = {}
    for c in CASE_SCOPE:
        rows = df[df["case"] == c].sort_values("run")
        cases[c] = {
            "seq":    scaler.transform(rows[FEAT_COLS].values).astype(np.float32),
            "vb":     rows["VB_max"].values.astype(np.float32),
            "runs":   rows["run"].values.astype(int),
            "n_runs": len(rows),
        }
    return cases


def run_feat_seq(df: pd.DataFrame, device: torch.device, seed: int,
                 cell: str = "gru") -> tuple[float, dict[int, float]]:
    input_dim = len(FEAT_COLS)
    case_rmses: dict[int, float] = {}

    for tc in CASE_SCOPE:
        tr_df = df[df["case"] != tc]
        scaler = StandardScaler().fit(tr_df[FEAT_COLS].values)
        all_cases = _build_seq_cases(df, scaler)

        train = {c: all_cases[c] for c in CASE_SCOPE if c != tc}
        all_vb = np.concatenate([info["vb"] for info in train.values()])
        y_mean = float(all_vb.mean())
        y_std  = max(float(all_vb.std()), 1e-8)

        train_seqs = [torch.tensor(info["seq"]) for info in train.values()]
        train_vbs  = [
            torch.tensor((info["vb"] - y_mean) / y_std, dtype=torch.float32)
            for info in train.values()
        ]
        lengths = torch.tensor([len(s) for s in train_seqs])
        x_train = nn.utils.rnn.pad_sequence(train_seqs, batch_first=True).to(device)
        y_train = nn.utils.rnn.pad_sequence(
            train_vbs, batch_first=True, padding_value=float("nan")).to(device)
        mask_tr = ~torch.isnan(y_train)

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = FeatureSeqModel(input_dim, cell=cell).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=SEQ_LR, weight_decay=SEQ_WD)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SEQ_EPOCHS)

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
        case_rmses[tc] = float(np.sqrt(mean_squared_error(info["vb"], y_pred)))

    return float(np.mean(list(case_rmses.values()))), case_rmses


def run_feat_gru(df: pd.DataFrame, device: torch.device, seed: int) -> tuple[float, dict[int, float]]:
    return run_feat_seq(df, device, seed, cell="gru")


# ─── Aggregation ──────────────────────────────────────────────────────────────
def aggregate_seeds(results_per_seed: list[tuple[float, dict[int, float]]]) -> tuple[float, float, dict[int, float]]:
    means = [r[0] for r in results_per_seed]
    all_case_rmses: dict[int, list[float]] = {}
    for _, cr in results_per_seed:
        for c, v in cr.items():
            all_case_rmses.setdefault(c, []).append(v)
    case_mean = {c: float(np.mean(vs)) for c, vs in all_case_rmses.items()}
    return float(np.mean(means)), float(np.std(means)), case_mean


# ─── Plot ─────────────────────────────────────────────────────────────────────
_COLOR_MAP = {
    "CF":       "#b5cfea",
    "RL":       "#aec7e8",
    "Ridge":    "#4878d0",
    "SVR":      "#3a64bc",
    "RF":       "#2c50a8",
    "MLP_Feat": "#1e3c94",
    "FeatRNN":  "#98df8a",
    "FeatLSTM": "#17becf",
    "FeatGRU":  "#2ca02c",
}
_DEFAULT_COLOR = "#999999"


def plot_results(model_names: list[str], means: list[float], stds: list[float], out_dir: Path) -> None:
    colors = [_COLOR_MAP.get(n, _DEFAULT_COLOR) for n in model_names]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(model_names)), means,
                  color=colors, alpha=0.85, yerr=stds, capsize=3)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.5,
                f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    if "FeatGRU" in model_names:
        gru_val = means[model_names.index("FeatGRU")]
        ax.axhline(gru_val, color="#2ca02c", linestyle="--", linewidth=1.2,
                   label=f"FeatGRU={gru_val:.2f} µm")

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("RMSE [µm] (LOCV-3 mean ± std)")
    ax.set_title("B1_S1: PHM2010 Feature Baseline — 9 models, LOCV-3")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_dir / "b1_phm2010_baseline.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "b1_phm2010_baseline.svg"), bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B1" / "S1" / f"{ts}_phm2010_feature_baseline"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== B1_S1: PHM2010 Feature Baseline ===")
    log(f"Cases={CASE_SCOPE}, Seeds={SEEDS}, Features={len(FEAT_COLS)}")
    log("Target: VB_max = max(VB_flute_1, VB_flute_2, VB_flute_3) [µm]")

    log("\nLoading data...")
    process_df, feat_df = load_data()
    df = build_merged(process_df, feat_df)
    log(f"Total rows: {len(df)}  (cases: {sorted(df.case.unique())}, runs/case=315)")
    log(f"VB_max stats: mean={df.VB_max.mean():.1f}, std={df.VB_max.std():.1f}, "
        f"min={df.VB_max.min():.1f}, max={df.VB_max.max():.1f} µm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    results: dict[str, tuple[float, float, dict[int, float]]] = {}
    t_total = time_mod.time()

    # ── Deterministic models ─────────────────────────────────────────────────
    for name, fn in [
        ("CF",    lambda: run_cf(df)),
        ("RL",    lambda: run_rl(df)),
        ("Ridge", lambda: run_ridge(df)),
        ("SVR",   lambda: run_svr(df)),
    ]:
        log(f"\n--- {name} ---")
        t0 = time_mod.time()
        mean_, case_r = fn()
        results[name] = (mean_, 0.0, case_r)
        log(f"  RMSE={mean_:.2f} µm  per_case={case_r}  [{time_mod.time()-t0:.1f}s]")

    # ── Stochastic models (5 seeds) ──────────────────────────────────────────
    for name, fn in [
        ("RF",       lambda seed: run_rf(df, seed)),
        ("MLP_Feat", lambda seed: run_mlp_feat(df, device, seed)),
        ("FeatRNN",  lambda seed: run_feat_seq(df, device, seed, "rnn")),
        ("FeatLSTM", lambda seed: run_feat_seq(df, device, seed, "lstm")),
        ("FeatGRU",  lambda seed: run_feat_seq(df, device, seed, "gru")),
    ]:
        log(f"\n--- {name} (5-seed) ---")
        seed_results: list[tuple[float, dict[int, float]]] = []
        t0 = time_mod.time()
        for seed in SEEDS:
            mean_, case_r = fn(seed)
            seed_results.append((mean_, case_r))
            log(f"  seed={seed}  RMSE={mean_:.2f} µm  [{time_mod.time()-t0:.0f}s]")
        m, s, case_r = aggregate_seeds(seed_results)
        results[name] = (m, s, case_r)
        log(f"  {name}: mean={m:.2f}  std={s:.2f} µm")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    best_model = min(results, key=lambda n: results[n][0])
    best_rmse  = results[best_model][0]
    log(f"{'Model':<12} {'Mean (µm)':>10} {'Std':>8} {'vs CF':>10}")
    log("-" * 44)
    cf_rmse = results["CF"][0]
    ordered = sorted(results.items(), key=lambda x: x[1][0])
    model_names_plot, means_plot, stds_plot = [], [], []
    for name, (m, s, _) in ordered:
        log(f"{name:<12} {m:>10.2f} {s:>8.2f} {m - cf_rmse:>+10.2f}")
        model_names_plot.append(name)
        means_plot.append(m)
        stds_plot.append(s)

    log(f"\nBest model: {best_model}  RMSE={best_rmse:.2f} µm")

    # ── Save ──────────────────────────────────────────────────────────────────
    agg_rows = [
        {"model": n, "mean_um": m, "std_um": s, "vs_cf_um": m - cf_rmse}
        for n, (m, s, _) in results.items()
    ]
    pd.DataFrame(agg_rows).sort_values("mean_um").to_csv(
        out_dir / "metrics" / "aggregate.csv", index=False)

    case_ids = sorted({c for _, _, cr in results.values() for c in cr})
    per_case = {"case": case_ids}
    for name, (_, _, cr) in results.items():
        per_case[name] = [cr.get(c, float("nan")) for c in case_ids]
    pd.DataFrame(per_case).to_csv(out_dir / "metrics" / "per_case.csv", index=False)

    plot_results(model_names_plot, means_plot, stds_plot, out_dir / "figures")

    summary = {
        "experiment": "B1_S1_phm2010_feature_baseline",
        "dataset": "phm2010",
        "target": "VB_max = max(VB_flute_1, VB_flute_2, VB_flute_3)",
        "unit": "µm",
        "n_features": len(FEAT_COLS),
        "cases": CASE_SCOPE,
        "seeds": SEEDS,
        "results": {n: {"mean_um": m, "std_um": s} for n, (m, s, _) in results.items()},
        "best_model": best_model,
        "best_rmse_um": best_rmse,
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

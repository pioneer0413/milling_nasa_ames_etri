#!/usr/bin/env python3
"""H4_S7: Raw-signal temporal-representation GRU (learned vs handcrafted features).

Instead of the feature-based run-to-run sequence model ([[Concept_Feature_GRU]]
DeltaMetaGRU over handcrafted mean/rms/std/peak deltas), this learns a temporal
representation directly from the raw signal of a single run and regresses VB:

  raw run signal [C=6 sensors, L]  (truncated to prefix r%)
    -> non-overlapping windows [K, C, W]
    -> shared 1D-CNN per window -> per-window embedding [K, d]   (within-run time axis)
    -> GRU over the K windows -> last hidden
    -> regression head -> VB(run)              (per-run prediction; no run-to-run)

Input length sweep: prefix r in {30,40,...,100}% (wide step). LOCV 15 cases, 3 seeds.
Per-sensor standardization is fit on TRAIN runs per fold (no leakage).

Output: experiments/executions/H4/S7/{timestamp}_raw_temporal_gru/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_H4_S5_feature_gru_hpo import (  # noqa: E402
    CASE_SCOPE, EXCLUDED_RUNS, SENSORS, THRESH, compute_metrics, parse_signal, preprocess,
)

WINDOW = 300            # samples per window (non-overlapping)
CFG = dict(cnn_channels=32, kernel=7, emb=48, gru_hidden=64, gru_layers=2,
           dropout=0.1, epochs=80, lr=1e-3, wd=1e-4, batch=32, grad_clip=1.0)
FEATURE_GRU_ANCHOR = 0.090773   # head32 prefix-80 (broad prefix level ~0.10-0.11)


# ─── Model ─────────────────────────────────────────────────────────────────────
class RawTemporalGRU(nn.Module):
    def __init__(self, in_ch, cfg):
        super().__init__()
        c, k, e = cfg["cnn_channels"], cfg["kernel"], cfg["emb"]
        pad = k // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, c, k, padding=pad), nn.BatchNorm1d(c), nn.ReLU(),
            nn.Conv1d(c, c, k, padding=pad), nn.BatchNorm1d(c), nn.ReLU(),
            nn.Conv1d(c, e, k, padding=pad), nn.BatchNorm1d(e), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.gru = nn.GRU(e, cfg["gru_hidden"], cfg["gru_layers"], batch_first=True,
                          dropout=cfg["dropout"] if cfg["gru_layers"] > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(cfg["gru_hidden"], 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):                       # x: [B, K, C, W]
        b, k, c, w = x.shape
        h = self.cnn(x.reshape(b * k, c, w)).reshape(b, k, -1)   # [B, K, emb]
        out, _ = self.gru(h)
        return self.head(out[:, -1, :]).squeeze(-1)


# ─── Data ──────────────────────────────────────────────────────────────────────
def load_raw_cache(sig, proc):
    """raw[(case,run)] = float32 [C, 9000]; vb[(case,run)] = float."""
    raw, vb = {}, {}
    for row in proc.itertuples(index=False):
        c, r = int(row.case), int(row.run)
        if (c, r) in EXCLUDED_RUNS or np.isnan(row.VB):
            continue
        s = sig[(sig.case == c) & (sig.run == r)]
        if s.empty:
            continue
        s = s.iloc[0]
        arrs = [parse_signal(s[sn]) for sn in SENSORS]
        if any(np.abs(a).max() > THRESH for a in arrs):
            continue
        base = min(len(a) for a in arrs)
        raw[(c, r)] = np.stack([a[:base] for a in arrs]).astype(np.float32)   # [C, base]
        vb[(c, r)] = float(row.VB)
    return raw, vb


def case_runs(proc, raw, c):
    return sorted(int(r.run) for r in proc[proc.case == c].itertuples(index=False) if (c, int(r.run)) in raw)


def window_run(arr, length, K, mean, std):
    """arr [C, base] -> normalized windows [K, C, W]."""
    x = (arr[:, :length] - mean[:, None]) / std[:, None]          # [C, length]
    x = x[:, : K * WINDOW].reshape(arr.shape[0], K, WINDOW)        # [C, K, W]
    return np.transpose(x, (1, 0, 2)).astype(np.float32)          # [K, C, W]


def fit_predict_fold(train, test, device, cfg, seed, length, K):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # per-sensor stats from TRAIN samples
    cat = np.concatenate([a[:, :length] for a, _ in train], axis=1)   # [C, sum_len]
    mean = cat.mean(axis=1); std = cat.std(axis=1); std[std < 1e-6] = 1.0

    Xtr = np.stack([window_run(a, length, K, mean, std) for a, _ in train])   # [N,K,C,W]
    ytr = np.array([y for _, y in train], dtype=np.float32)
    ymu, ysd = float(ytr.mean()), float(ytr.std() if ytr.std() > 1e-8 else 1.0)
    Xtr = torch.tensor(Xtr, device=device)
    ytr_n = torch.tensor((ytr - ymu) / ysd, device=device)

    model = RawTemporalGRU(len(SENSORS), cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    n = len(train); bs = cfg["batch"]
    model.train()
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = model(Xtr[idx])
            loss = ((pred - ytr_n[idx]) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
        sched.step()

    model.eval()
    Xte = torch.tensor(np.stack([window_run(a, length, K, mean, std) for a, _ in test]), device=device)
    with torch.no_grad():
        pred = model(Xte).cpu().numpy() * ysd + ymu
    return pred


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefixes", type=int, nargs="+", default=[30, 40, 50, 60, 70, 80, 90, 100])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    prefixes = args.prefixes[:1] if args.smoke else args.prefixes
    seeds = args.seeds[:1] if args.smoke else args.seeds

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT / "experiments" / "executions" / "H4" / "S7" / f"{ts}_raw_temporal_gru"
    for sub in ["analysis", "metrics", "figures", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(m):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"=== H4_S7 raw-temporal GRU === prefixes={prefixes} seeds={seeds} window={WINDOW}")
    sig = pd.read_csv(ROOT / "datasets/processed/mill_signal_data.csv", usecols=["case", "run"] + SENSORS)
    proc = pd.read_csv(ROOT / "datasets/processed/mill_process_info.csv")
    proc = proc[proc.case.isin(CASE_SCOPE)].copy()
    sig = sig[sig.case.isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(proc)
    log("Caching raw signal arrays...")
    raw, vb = load_raw_cache(sig, proc_clean)
    base_len = min(a.shape[1] for a in raw.values())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"cached runs={len(raw)}  base_len={base_len}  device={device}")

    seed_rows, per_case_rows = [], []
    t0 = time.time()
    for pct in prefixes:
        length = max(WINDOW, int(np.ceil(base_len * pct / 100.0)))
        length = min(length, base_len)
        K = length // WINDOW
        for seed in seeds:
            for tc in CASE_SCOPE:
                tr = [(raw[(c, r)], vb[(c, r)]) for c in CASE_SCOPE if c != tc for r in case_runs(proc_clean, raw, c)]
                te_runs = case_runs(proc_clean, raw, tc)
                if not te_runs:
                    continue
                te = [(raw[(tc, r)], vb[(tc, r)]) for r in te_runs]
                preds = fit_predict_fold(tr, te, device, CFG, seed, length, K)
                y_true = np.array([y for _, y in te])
                m = compute_metrics(y_true, preds)
                per_case_rows.append({"prefix": pct, "seed": seed, "target_case": tc, "n_runs": len(te), **m})
            ssd = pd.DataFrame(per_case_rows).query("prefix == @pct and seed == @seed")[["rmse", "r2", "mae"]].mean()
            seed_rows.append({"prefix": pct, "seed": seed, "locv_rmse": float(ssd.rmse),
                              "locv_r2": float(ssd.r2), "locv_mae": float(ssd.mae)})
        psd = pd.DataFrame(seed_rows).query("prefix == @pct")
        log(f"prefix={pct}% (K={K})  mean_rmse={psd.locv_rmse.mean():.6f} (+/-{psd.locv_rmse.std():.6f}) "
            f"mean_r2={psd.locv_r2.mean():.4f}  [{time.time()-t0:.0f}s]")

    seed_df = pd.DataFrame(seed_rows)
    per_case_df = pd.DataFrame(per_case_rows)
    pref_df = (seed_df.groupby("prefix", as_index=False)
               .agg(mean_rmse=("locv_rmse", "mean"), std_rmse=("locv_rmse", "std"),
                    mean_r2=("locv_r2", "mean"), mean_mae=("locv_mae", "mean")))
    pref_df["std_rmse"] = pref_df["std_rmse"].fillna(0.0)
    seed_df.to_csv(out / "metrics" / "seed_metrics.csv", index=False)
    per_case_df.to_csv(out / "metrics" / "per_case_metrics.csv", index=False)
    pref_df.to_csv(out / "analysis" / "prefix_metrics.csv", index=False)

    best = pref_df.sort_values("mean_rmse").iloc[0]

    # Fig 1: prefix vs RMSE/R2
    fig, ax1 = plt.subplots(figsize=(8.4, 5))
    ax1.errorbar(pref_df.prefix, pref_df.mean_rmse, yerr=pref_df.std_rmse, marker="o", color="#d62728", capsize=3, label="raw-GRU RMSE")
    ax1.axhline(FEATURE_GRU_ANCHOR, color="#7f7f7f", ls=":", lw=1.3, label=f"feature_gru @80% ({FEATURE_GRU_ANCHOR:.4f})")
    ax1.set_xlabel("prefix input ratio (%)"); ax1.set_ylabel("LOCV mean RMSE", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728"); ax1.grid(True, color="#eee", lw=0.5); ax1.set_xticks(prefixes)
    ax2 = ax1.twinx(); ax2.plot(pref_df.prefix, pref_df.mean_r2, marker="s", color="#1f77b4", label="R2")
    ax2.set_ylabel("LOCV mean R2", color="#1f77b4"); ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_title("H4_S7 raw-temporal GRU — input ratio vs VB prediction (3-seed)")
    ax1.legend(frameon=False, fontsize=8, loc="upper center")
    fig.tight_layout(); fig.savefig(out / "figures" / "01_prefix_rmse_r2_curve.png", dpi=150); plt.close(fig)

    # Fig 2: per-case heatmap
    pc = per_case_df.groupby(["target_case", "prefix"]).rmse.mean().reset_index()
    heat = pc.pivot(index="target_case", columns="prefix", values="rmse")
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(heat.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels([int(c) for c in heat.columns])
    ax.set_yticks(range(len(heat))); ax.set_yticklabels([f"case {c}" for c in heat.index], fontsize=8)
    ax.set_xlabel("prefix (%)"); ax.set_ylabel("hold-out case"); ax.set_title("Per-case LOCV RMSE vs prefix (raw-GRU, 3-seed mean)")
    fig.colorbar(im, ax=ax, label="RMSE", fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out / "figures" / "02_per_case_rmse_heatmap.png", dpi=150); plt.close(fig)

    summary = {
        "experiment": "H4_S7_raw_temporal_gru", "prefixes": prefixes, "seeds": seeds,
        "window": WINDOW, "model_cfg": CFG, "feature_gru_anchor": FEATURE_GRU_ANCHOR,
        "best_prefix": int(best.prefix), "best_rmse": float(best.mean_rmse), "best_r2": float(best.mean_r2),
        "prefix_table": pref_df.to_dict("records"), "execution_dir": str(out),
    }
    (out / "analysis" / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    log("\n=== prefix vs LOCV RMSE (3-seed) ===")
    for r in pref_df.itertuples(index=False):
        mark = "  <- best" if int(r.prefix) == int(best.prefix) else ""
        log(f"  prefix={int(r.prefix)}%  rmse={r.mean_rmse:.6f} (+/-{r.std_rmse:.6f}) r2={r.mean_r2:.4f}{mark}")
    log(f"\nBest={int(best.prefix)}% rmse={best.mean_rmse:.6f}; feature_gru@80%={FEATURE_GRU_ANCHOR:.6f}")
    (out / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"EXECUTION_DIR={out}", flush=True)


if __name__ == "__main__":
    main()

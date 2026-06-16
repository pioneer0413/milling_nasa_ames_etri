#!/usr/bin/env python3
"""H4_S7_T2: Raw-signal GRU-only representation + MLP regression (no CNN).

Variant of H4_S7 that drops the learnable 1D-CNN window encoder. The raw signal is
reduced to a tractable sequence by fixed non-learned average-pooling, then a multi-layer
GRU learns the temporal representation and an MLP head regresses VB:

  raw run signal [C=6, L]  (prefix r%)
    -> per-sensor z-score (train stats)
    -> fixed avg-pool by POOL -> sequence [L', C]   (no learned conv)
    -> GRU (temporal representation) -> last hidden
    -> MLP head -> VB(run)            (per-run; no run-to-run)

Input sweep r in {30,...,100}%. LOCV 15 cases, 3 seeds. Compare to the CNN-GRU (H4_S7)
and the handcrafted feature_gru anchor.

Output: experiments/executions/H4/S7/{timestamp}_raw_gru_only/
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

from scripts.run_H4_S5_feature_gru_hpo import CASE_SCOPE, SENSORS, compute_metrics, preprocess  # noqa: E402
from scripts.run_H4_S7_raw_temporal_gru import (  # noqa: E402
    FEATURE_GRU_ANCHOR, case_runs, load_raw_cache,
)

POOL = 60               # fixed avg-pool factor (9000 -> 150 steps at 100%)
CFG = dict(gru_hidden=64, gru_layers=2, mlp=64, dropout=0.1,
           epochs=80, lr=1e-3, wd=1e-4, batch=32, grad_clip=1.0)
CNN_GRU_REF = 0.147368  # H4_S7 raw CNN-GRU best (50%) for reference


class RawGRUOnly(nn.Module):
    def __init__(self, in_ch, cfg):
        super().__init__()
        self.gru = nn.GRU(in_ch, cfg["gru_hidden"], cfg["gru_layers"], batch_first=True,
                          dropout=cfg["dropout"] if cfg["gru_layers"] > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(cfg["gru_hidden"], cfg["mlp"]), nn.ReLU(), nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["mlp"], 1),
        )

    def forward(self, x):                      # x: [B, L', C]
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def downsample(arr, length, Lp, mean, std):
    """arr [C, base] -> normalized avg-pooled sequence [L', C]."""
    x = (arr[:, :length] - mean[:, None]) / std[:, None]      # [C, length]
    x = x[:, : Lp * POOL].reshape(arr.shape[0], Lp, POOL).mean(axis=2)   # [C, L']
    return np.transpose(x, (1, 0)).astype(np.float32)        # [L', C]


def fit_predict_fold(train, test, device, cfg, seed, length, Lp):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    cat = np.concatenate([a[:, :length] for a, _ in train], axis=1)
    mean = cat.mean(axis=1); std = cat.std(axis=1); std[std < 1e-6] = 1.0

    Xtr = np.stack([downsample(a, length, Lp, mean, std) for a, _ in train])   # [N,L',C]
    ytr = np.array([y for _, y in train], dtype=np.float32)
    ymu, ysd = float(ytr.mean()), float(ytr.std() if ytr.std() > 1e-8 else 1.0)
    Xtr = torch.tensor(Xtr, device=device)
    ytr_n = torch.tensor((ytr - ymu) / ysd, device=device)

    model = RawGRUOnly(len(SENSORS), cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    n = len(train); bs = cfg["batch"]
    model.train()
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = ((model(Xtr[idx]) - ytr_n[idx]) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
        sched.step()

    model.eval()
    Xte = torch.tensor(np.stack([downsample(a, length, Lp, mean, std) for a, _ in test]), device=device)
    with torch.no_grad():
        pred = model(Xte).cpu().numpy() * ysd + ymu
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefixes", type=int, nargs="+", default=[30, 40, 50, 60, 70, 80, 90, 100])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    prefixes = args.prefixes[:1] if args.smoke else args.prefixes
    seeds = args.seeds[:1] if args.smoke else args.seeds

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT / "experiments" / "executions" / "H4" / "S7" / f"{ts}_raw_gru_only"
    for sub in ["analysis", "metrics", "figures", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(m):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); log_lines.append(line)

    log(f"=== H4_S7_T2 raw GRU-only + MLP === prefixes={prefixes} seeds={seeds} POOL={POOL}")
    sig = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    proc = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    proc = proc[proc.case.isin(CASE_SCOPE)].copy()
    sig = sig[sig.case.isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(proc)
    log("Caching raw signal arrays...")
    raw, vb = load_raw_cache(sig, proc_clean)
    base_len = min(a.shape[1] for a in raw.values())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"cached runs={len(raw)} base_len={base_len} device={device}")

    seed_rows, per_case_rows = [], []
    t0 = time.time()
    for pct in prefixes:
        length = min(base_len, max(POOL, int(np.ceil(base_len * pct / 100.0))))
        Lp = length // POOL
        for seed in seeds:
            for tc in CASE_SCOPE:
                tr = [(raw[(c, r)], vb[(c, r)]) for c in CASE_SCOPE if c != tc for r in case_runs(proc_clean, raw, c)]
                te_runs = case_runs(proc_clean, raw, tc)
                if not te_runs:
                    continue
                te = [(raw[(tc, r)], vb[(tc, r)]) for r in te_runs]
                preds = fit_predict_fold(tr, te, device, CFG, seed, length, Lp)
                m = compute_metrics(np.array([y for _, y in te]), preds)
                per_case_rows.append({"prefix": pct, "seed": seed, "target_case": tc, "n_runs": len(te), **m})
            ssd = pd.DataFrame(per_case_rows).query("prefix == @pct and seed == @seed")[["rmse", "r2", "mae"]].mean()
            seed_rows.append({"prefix": pct, "seed": seed, "locv_rmse": float(ssd.rmse),
                              "locv_r2": float(ssd.r2), "locv_mae": float(ssd.mae)})
        psd = pd.DataFrame(seed_rows).query("prefix == @pct")
        log(f"prefix={pct}% (L'={Lp})  mean_rmse={psd.locv_rmse.mean():.6f} (+/-{psd.locv_rmse.std():.6f}) "
            f"mean_r2={psd.locv_r2.mean():.4f}  [{time.time()-t0:.0f}s]")

    seed_df = pd.DataFrame(seed_rows); per_case_df = pd.DataFrame(per_case_rows)
    pref_df = (seed_df.groupby("prefix", as_index=False)
               .agg(mean_rmse=("locv_rmse", "mean"), std_rmse=("locv_rmse", "std"),
                    mean_r2=("locv_r2", "mean"), mean_mae=("locv_mae", "mean")))
    pref_df["std_rmse"] = pref_df["std_rmse"].fillna(0.0)
    seed_df.to_csv(out / "metrics" / "seed_metrics.csv", index=False)
    per_case_df.to_csv(out / "metrics" / "per_case_metrics.csv", index=False)
    pref_df.to_csv(out / "analysis" / "prefix_metrics.csv", index=False)
    best = pref_df.sort_values("mean_rmse").iloc[0]

    fig, ax1 = plt.subplots(figsize=(8.4, 5))
    ax1.errorbar(pref_df.prefix, pref_df.mean_rmse, yerr=pref_df.std_rmse, marker="o", color="#d62728", capsize=3, label="raw GRU-only RMSE")
    ax1.axhline(FEATURE_GRU_ANCHOR, color="#7f7f7f", ls=":", lw=1.3, label=f"feature_gru@80% ({FEATURE_GRU_ANCHOR:.4f})")
    ax1.axhline(CNN_GRU_REF, color="#2ca02c", ls="--", lw=1.2, label=f"raw CNN-GRU best ({CNN_GRU_REF:.4f})")
    ax1.set_xlabel("prefix input ratio (%)"); ax1.set_ylabel("LOCV mean RMSE", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728"); ax1.grid(True, color="#eee", lw=0.5); ax1.set_xticks(prefixes)
    ax2 = ax1.twinx(); ax2.plot(pref_df.prefix, pref_df.mean_r2, marker="s", color="#1f77b4", label="R2")
    ax2.set_ylabel("LOCV mean R2", color="#1f77b4"); ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_title("H4_S7_T2 raw GRU-only + MLP — input ratio vs VB prediction (3-seed)")
    ax1.legend(frameon=False, fontsize=8, loc="upper center")
    fig.tight_layout(); fig.savefig(out / "figures" / "01_prefix_rmse_r2_curve.png", dpi=150); plt.close(fig)

    summary = {"experiment": "H4_S7_T2_raw_gru_only", "prefixes": prefixes, "seeds": seeds, "POOL": POOL,
               "model_cfg": CFG, "feature_gru_anchor": FEATURE_GRU_ANCHOR, "cnn_gru_ref": CNN_GRU_REF,
               "best_prefix": int(best.prefix), "best_rmse": float(best.mean_rmse), "best_r2": float(best.mean_r2),
               "prefix_table": pref_df.to_dict("records"), "execution_dir": str(out)}
    (out / "analysis" / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    log("\n=== prefix vs LOCV RMSE (3-seed) ===")
    for r in pref_df.itertuples(index=False):
        mark = "  <- best" if int(r.prefix) == int(best.prefix) else ""
        log(f"  prefix={int(r.prefix)}%  rmse={r.mean_rmse:.6f} (+/-{r.std_rmse:.6f}) r2={r.mean_r2:.4f}{mark}")
    log(f"\nBest={int(best.prefix)}% {best.mean_rmse:.6f}; raw CNN-GRU best={CNN_GRU_REF}; feature_gru@80%={FEATURE_GRU_ANCHOR}")
    (out / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"EXECUTION_DIR={out}", flush=True)


if __name__ == "__main__":
    main()

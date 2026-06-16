#!/usr/bin/env python3
"""H4_S5 fine prefix sweep (70-90%, 2% step) in the V1 per-timestep frame.

Zooms into the best region of the input-ratio curve using the tuned best config
(`head32`: per-timestep DeltaMetaGRU, hidden=256/layers=3/dropout=0.1/head=32/
lr=1e-3/wd=1e-4/epochs=200). All runs truncated to the same prefix %, delta vs
run_1 @ same prefix, LOCV 15 cases, 3 seeds. Coarse sweep had its best at 80%
(0.0908); here we resolve 70-90% at 2% granularity.

Artifacts:
  figures/ prefix-vs-RMSE/R2 curve, per-case RMSE heatmap, per-case best-prefix bar,
           per-case RMSE small-multiples
  analysis/ prefix_metrics.csv, per_case_best_prefix.csv, summary.json
  metrics/  seed_metrics.csv, per_case_metrics.csv

Output: experiments/executions/H4/S5/{timestamp}_fine_prefix_sweep_70_90/
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_H4_S5_feature_gru_hpo import (  # noqa: E402
    BASE, CASE_SCOPE, META_FEATURES, SENSORS, build_feature_matrix, build_sequences,
    delta_col_names, preprocess, run_locv_seed,
)

PREFIXES = list(range(70, 91, 2))                # 70,72,...,90 (11 points)
SEEDS = [0, 1, 2]
BEST = {**BASE, "head_hidden": 32, "name": "head32"}
INPUT_DIM = 4 * len(SENSORS) + len(META_FEATURES)


def main():
    global PREFIXES
    p = argparse.ArgumentParser(description="H4_S5 fine prefix sweep (V1 per-timestep head32).")
    p.add_argument("--start", type=int, default=70)
    p.add_argument("--end", type=int, default=90)
    p.add_argument("--step", type=int, default=2)
    args = p.parse_args()
    PREFIXES = list(range(args.start, args.end + 1, args.step))

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT / "experiments" / "executions" / "H4" / "S5" / f"{ts}_fine_prefix_sweep_{args.start}_{args.end}"
    for sub in ["analysis", "metrics", "figures", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(m):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"=== H4_S5 fine prefix sweep {PREFIXES}% (V1 per-timestep head32) ===")
    sig = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    proc = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    proc = proc[proc.case.isin(CASE_SCOPE)].copy()
    sig = sig[sig.case.isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(proc)
    feat = build_feature_matrix(sig, proc_clean, PREFIXES)
    delta_cols = delta_col_names()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"feature matrix rows={len(feat)}  device={device}")

    seed_rows, per_case_rows = [], []
    t0 = time.time()
    for pct in PREFIXES:
        case_data = build_sequences(feat, delta_cols, float(pct))
        for seed in SEEDS:
            df = run_locv_seed(case_data, INPUT_DIM, device, BEST, seed)
            df.insert(0, "prefix", pct); df.insert(1, "seed", seed)
            per_case_rows.extend(df.to_dict("records"))
            seed_rows.append({"prefix": pct, "seed": seed, "locv_rmse": float(df.rmse.mean()),
                              "locv_r2": float(df.r2.mean()), "locv_mae": float(df.mae.mean())})
        ssd = pd.DataFrame(seed_rows).query("prefix == @pct")
        log(f"prefix={pct}%  mean_rmse={ssd.locv_rmse.mean():.6f} (+/-{ssd.locv_rmse.std():.6f}) "
            f"mean_r2={ssd.locv_r2.mean():.4f}  [{time.time()-t0:.0f}s]")

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

    # case x prefix RMSE (3-seed mean)
    pc = per_case_df.groupby(["target_case", "prefix"]).rmse.mean().reset_index()
    heat = pc.pivot(index="target_case", columns="prefix", values="rmse")
    best_pref = heat.idxmin(axis=1).rename("best_prefix").reset_index()
    best_pref["best_rmse"] = [heat.loc[r.target_case, r.best_prefix] for r in best_pref.itertuples(index=False)]
    best_pref.to_csv(out / "analysis" / "per_case_best_prefix.csv", index=False)

    # ── Fig 1: prefix vs RMSE/R2 curve ──
    fig, ax1 = plt.subplots(figsize=(8.4, 5))
    ax1.errorbar(pref_df.prefix, pref_df.mean_rmse, yerr=pref_df.std_rmse, marker="o",
                 color="#d62728", capsize=3, label="LOCV RMSE")
    ax1.scatter([best.prefix], [best.mean_rmse], s=140, facecolors="none", edgecolors="#d62728",
                linewidths=2, zorder=5, label=f"best={int(best.prefix)}% ({best.mean_rmse:.4f})")
    ax1.set_xlabel("prefix input ratio (%) — all runs truncated identically")
    ax1.set_ylabel("LOCV mean RMSE", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728"); ax1.grid(True, color="#eee", lw=0.5)
    ax1.set_xticks(PREFIXES)
    ax2 = ax1.twinx()
    ax2.plot(pref_df.prefix, pref_df.mean_r2, marker="s", color="#1f77b4", label="LOCV R2")
    ax2.set_ylabel("LOCV mean R2", color="#1f77b4"); ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_title(f"H4_S5 fine prefix sweep ({PREFIXES[0]}-{PREFIXES[-1]}%, step {PREFIXES[1]-PREFIXES[0]}) — head32, 3-seed")
    ax1.legend(frameon=False, fontsize=8, loc="upper center")
    fig.tight_layout(); fig.savefig(out / "figures" / "01_prefix_rmse_r2_curve.png", dpi=150); plt.close(fig)

    # ── Fig 2: per-case RMSE heatmap (case x prefix) ──
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(heat.values, aspect="auto", cmap="viridis",
                   extent=[PREFIXES[0] - 1, PREFIXES[-1] + 1, len(heat) - 0.5, -0.5])
    ax.set_yticks(range(len(heat))); ax.set_yticklabels([f"case {c}" for c in heat.index], fontsize=8)
    ax.set_xticks(PREFIXES); ax.set_xlabel("prefix (%)"); ax.set_ylabel("hold-out case")
    ax.set_title("Per-case LOCV RMSE vs prefix (3-seed mean)")
    # mark each case's best prefix
    for i, c in enumerate(heat.index):
        bp = heat.loc[c].idxmin()
        ax.scatter([bp], [i], marker="*", s=70, color="white", edgecolors="black", linewidths=0.4)
    fig.colorbar(im, ax=ax, label="RMSE", fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out / "figures" / "02_per_case_rmse_heatmap.png", dpi=150); plt.close(fig)

    # ── Fig 3: per-case best prefix (bar) ──
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bpv = best_pref.sort_values("target_case")
    ax.bar(bpv.target_case.astype(str), bpv.best_prefix, color="#4c72b0", alpha=0.85)
    ax.axhline(best.prefix, color="#d62728", ls="--", lw=1.2, label=f"global best {int(best.prefix)}%")
    ax.set_xlabel("hold-out case"); ax.set_ylabel("best prefix (%)")
    ax.set_title("Each case's individually-best prefix"); ax.set_ylim(PREFIXES[0] - 2, PREFIXES[-1] + 2)
    ax.legend(frameon=False); ax.grid(True, axis="y", color="#eee", lw=0.5)
    fig.tight_layout(); fig.savefig(out / "figures" / "03_per_case_best_prefix.png", dpi=150); plt.close(fig)

    # ── Fig 4: per-case RMSE small multiples ──
    cases = list(heat.index); n = len(cases); nc = 4; nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 2.4 * nr), squeeze=False)
    for a in axes.flat:
        a.axis("off")
    for i, c in enumerate(cases):
        a = axes[i // nc][i % nc]; a.axis("on")
        a.plot(PREFIXES, heat.loc[c].values, marker="o", ms=3, color="#d62728")
        a.axvline(best.prefix, color="#1f77b4", ls="--", lw=0.8)
        a.set_title(f"case {c}", fontsize=9); a.grid(True, color="#eee", lw=0.5)
    fig.suptitle("Per-case RMSE vs prefix (blue = global best)", fontsize=12)
    fig.supxlabel("prefix (%)"); fig.supylabel("RMSE")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / "figures" / "04_per_case_rmse_small_multiples.png", dpi=150); plt.close(fig)

    summary = {
        "experiment": "H4_S5_fine_prefix_sweep_70_90", "prefixes": PREFIXES, "seeds": SEEDS,
        "gru_config": BEST, "frame": "V1 per-timestep DeltaMetaGRU, all runs same prefix",
        "best_prefix": int(best.prefix), "best_rmse": float(best.mean_rmse),
        "best_r2": float(best.mean_r2),
        "prefix_table": pref_df.to_dict("records"),
        "per_case_best_prefix": best_pref.to_dict("records"),
        "execution_dir": str(out),
    }
    (out / "analysis" / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    log("\n=== prefix vs LOCV RMSE (3-seed) ===")
    for r in pref_df.itertuples(index=False):
        mark = "  <- best" if int(r.prefix) == int(best.prefix) else ""
        log(f"  prefix={int(r.prefix)}%  rmse={r.mean_rmse:.6f} (+/-{r.std_rmse:.6f}) r2={r.mean_r2:.4f}{mark}")
    log(f"\nBest prefix = {int(best.prefix)}%  RMSE {best.mean_rmse:.6f}  R2 {best.mean_r2:.4f}")
    (out / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"EXECUTION_DIR={out}", flush=True)


if __name__ == "__main__":
    main()

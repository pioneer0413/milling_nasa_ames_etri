#!/usr/bin/env python3
"""H4_S5_T1 diagnostics: learning dynamics & prediction trends of the best config.

Best config from H4_S5 HPO (`head32`): feature_gru (DeltaMetaGRU) with
  hidden=256, layers=3, dropout=0.1, head_hidden=32, lr=1e-3, wd=1e-4,
  epochs=200, CosineAnnealingLR, grad_clip=1.0, prefix 80%, Delta+Meta, LOCV.

For each LOCV fold (15 cases) x seed (0,1,2) this records the per-epoch training
loss and the per-run predictions, then renders diagnostic figures so we can see
*how the model learns* and *how its predictions trend* against the true VB wear.

Output:
  experiments/executions/H4/S5/{timestamp}_feature_gru_best_diagnostics/
    figures/   loss + prediction + residual + per-case diagnostics
    analysis/  loss_history.csv, predictions.csv, per_case_metrics.csv, summary.json
"""
from __future__ import annotations

import json
import sys
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
    BASE, CASE_SCOPE, META_FEATURES, SENSORS, DeltaMetaGRU, build_feature_matrix,
    build_sequences, compute_metrics, delta_col_names, preprocess,
)

SEEDS = [0, 1, 2]
PREFIX = 80
BEST = {**BASE, "head_hidden": 32, "name": "head32"}


# ─── Training with loss history ────────────────────────────────────────────────
def fit_with_history(train_cases, test_case_data, input_dim, device, cfg, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = DeltaMetaGRU(
        input_dim=input_dim, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
        dropout=cfg["dropout"], head_hidden=cfg["head_hidden"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    epochs = cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_seqs = [info["seq"] for info in train_cases.values()]
    train_vbs = [info["vb"] for info in train_cases.values()]
    all_vb = np.concatenate(train_vbs)
    y_mean = float(all_vb.mean())
    y_std = float(all_vb.std()) if all_vb.std() > 1e-8 else 1.0

    tensors = [torch.tensor(s, dtype=torch.float32) for s in train_seqs]
    lengths = torch.tensor([len(s) for s in train_seqs])
    x_train = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0).to(device)
    tgt = [torch.tensor((t - y_mean) / y_std, dtype=torch.float32) for t in train_vbs]
    y_train = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=float("nan")).to(device)
    mask = ~torch.isnan(y_train)

    loss_hist: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train, lengths)
        loss = ((pred[mask] - y_train[mask]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        scheduler.step()
        # report training loss in *original VB units* (RMSE) for interpretability
        loss_hist.append(float(np.sqrt(loss.item()) * y_std))

    model.eval()
    with torch.no_grad():
        seq_t = torch.tensor(test_case_data["seq"], dtype=torch.float32).unsqueeze(0).to(device)
        length_t = torch.tensor([test_case_data["n_runs"]])
        pred_norm = model(seq_t, length_t).squeeze(0).cpu().numpy()
        y_pred = pred_norm * y_std + y_mean
    return loss_hist, y_pred


# ─── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT / "experiments" / "executions" / "H4" / "S5" / f"{ts}_feature_gru_best_diagnostics"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "analysis").mkdir(parents=True, exist_ok=True)

    print("Loading data + building Delta+Meta feature matrix (prefix 80%)...", flush=True)
    sig = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    proc = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    proc = proc[proc["case"].isin(CASE_SCOPE)].copy()
    sig = sig[sig["case"].isin(CASE_SCOPE)].copy()
    feat = build_feature_matrix(sig, preprocess(proc), [PREFIX])
    delta_cols = delta_col_names()
    input_dim = len(delta_cols) + len(META_FEATURES)
    case_data = build_sequences(feat, delta_cols, float(PREFIX))
    # run numbers per case (x-axis for trajectory)
    runs_by_case = {
        int(c): feat[(feat.signal_prefix_percent == float(PREFIX)) & (feat.case_id == c)]
        .sort_values("run")["run"].to_numpy()
        for c in case_data
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"input_dim={input_dim}  cases={len(case_data)}  device={device}", flush=True)

    loss_rows, pred_rows = [], []
    cases = [c for c in CASE_SCOPE if c in case_data]
    for seed in SEEDS:
        for tc in cases:
            train = {c: case_data[c] for c in cases if c != tc}
            loss_hist, y_pred = fit_with_history(train, case_data[tc], input_dim, device, BEST, seed)
            for ep, lv in enumerate(loss_hist):
                loss_rows.append({"seed": seed, "case": tc, "epoch": ep, "train_rmse": lv})
            y_true = case_data[tc]["vb"]
            for i, rn in enumerate(runs_by_case[tc]):
                pred_rows.append({"seed": seed, "case": tc, "run": int(rn),
                                  "run_idx": i, "y_true": float(y_true[i]), "y_pred": float(y_pred[i])})
        print(f"  seed {seed} done", flush=True)

    loss_df = pd.DataFrame(loss_rows)
    pred_df = pd.DataFrame(pred_rows)
    loss_df.to_csv(out / "analysis" / "loss_history.csv", index=False)
    pred_df.to_csv(out / "analysis" / "predictions.csv", index=False)

    # seed-averaged predictions per (case, run)
    agg = (pred_df.groupby(["case", "run", "run_idx"], as_index=False)
           .agg(y_true=("y_true", "first"), y_pred_mean=("y_pred", "mean"), y_pred_std=("y_pred", "std")))
    agg["y_pred_std"] = agg["y_pred_std"].fillna(0.0)
    agg["residual"] = agg["y_pred_mean"] - agg["y_true"]

    per_case = []
    for c in cases:
        g = agg[agg.case == c]
        per_case.append({"case": c, "n_runs": len(g), **compute_metrics(g.y_true.to_numpy(), g.y_pred_mean.to_numpy())})
    per_case_df = pd.DataFrame(per_case).sort_values("rmse", ascending=False)
    per_case_df.to_csv(out / "analysis" / "per_case_metrics.csv", index=False)

    overall = compute_metrics(agg.y_true.to_numpy(), agg.y_pred_mean.to_numpy())
    locv_mean_rmse = float(per_case_df["rmse"].mean())

    n = len(cases)
    nc = 4
    nr = int(np.ceil(n / nc))

    # ── 1. Loss trend per fold (3 seeds each) ──────────────────────────────────
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 2.6 * nr), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, c in enumerate(cases):
        ax = axes[i // nc][i % nc]; ax.axis("on")
        for seed in SEEDS:
            d = loss_df[(loss_df.case == c) & (loss_df.seed == seed)]
            ax.plot(d.epoch, d.train_rmse, lw=1.0, alpha=0.85, label=f"seed{seed}")
        ax.set_yscale("log")
        ax.set_title(f"hold-out case {c}", fontsize=9)
        ax.grid(True, color="#eee", lw=0.5)
        if i == 0:
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Per-fold training loss (RMSE, VB units) vs epoch — log scale", fontsize=12)
    fig.supxlabel("epoch"); fig.supylabel("train RMSE")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / "figures" / "01_loss_trend_per_fold.png", dpi=150); plt.close(fig)

    # ── 2. Loss overlay (seed 0) — convergence speed comparison ────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, n))
    for col, c in zip(cmap, cases):
        d = loss_df[(loss_df.case == c) & (loss_df.seed == 0)]
        ax.plot(d.epoch, d.train_rmse, lw=1.1, color=col, label=f"c{c}")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("train RMSE (VB units)")
    ax.set_title("Training loss across folds (seed 0) — convergence comparison")
    ax.grid(True, color="#eee", lw=0.5)
    ax.legend(ncol=5, fontsize=7, frameon=False, title="hold-out case")
    fig.tight_layout(); fig.savefig(out / "figures" / "02_loss_overlay_seed0.png", dpi=150); plt.close(fig)

    # ── 3. Prediction trajectory per case (true vs 3-seed mean ± std) ──────────
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 2.6 * nr), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, c in enumerate(cases):
        ax = axes[i // nc][i % nc]; ax.axis("on")
        g = agg[agg.case == c].sort_values("run_idx")
        x = g.run_idx.to_numpy()
        ax.plot(x, g.y_true, "o-", color="#111", ms=3, lw=1.3, label="true")
        ax.plot(x, g.y_pred_mean, "s--", color="#d62728", ms=3, lw=1.2, label="pred")
        ax.fill_between(x, g.y_pred_mean - g.y_pred_std, g.y_pred_mean + g.y_pred_std,
                        color="#d62728", alpha=0.18)
        rmse = float(per_case_df[per_case_df.case == c]["rmse"].iloc[0])
        ax.set_title(f"case {c}  RMSE={rmse:.3f}", fontsize=9)
        ax.grid(True, color="#eee", lw=0.5)
        if i == 0:
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("VB prediction trajectory per hold-out case (3-seed mean ± std)", fontsize=12)
    fig.supxlabel("run index (chronological)"); fig.supylabel("VB")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / "figures" / "03_prediction_trajectory_per_case.png", dpi=150); plt.close(fig)

    # ── 4. Parity plot (pred vs actual), colored by case ───────────────────────
    fig, ax = plt.subplots(figsize=(6.4, 6))
    sc = ax.scatter(agg.y_true, agg.y_pred_mean, c=agg.case, cmap="tab20", s=28, alpha=0.85, edgecolors="none")
    lim = [min(agg.y_true.min(), agg.y_pred_mean.min()), max(agg.y_true.max(), agg.y_pred_mean.max())]
    ax.plot(lim, lim, "k--", lw=1, label="ideal")
    ax.set_xlabel("true VB"); ax.set_ylabel("predicted VB (3-seed mean)")
    ax.set_title(f"Parity plot — RMSE={overall['rmse']:.4f}  R2={overall['r2']:.3f}  MAE={overall['mae']:.4f}")
    ax.grid(True, color="#eee", lw=0.5); ax.legend(frameon=False)
    fig.colorbar(sc, ax=ax, label="case", fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out / "figures" / "04_parity_plot.png", dpi=150); plt.close(fig)

    # ── 5. Residual vs true VB & vs run index ──────────────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    a1.scatter(agg.y_true, agg.residual, c=agg.case, cmap="tab20", s=24, alpha=0.85, edgecolors="none")
    a1.axhline(0, color="k", lw=1)
    a1.set_xlabel("true VB"); a1.set_ylabel("residual (pred - true)"); a1.set_title("Residual vs true VB")
    a1.grid(True, color="#eee", lw=0.5)
    a2.scatter(agg.run_idx, agg.residual, c=agg.case, cmap="tab20", s=24, alpha=0.85, edgecolors="none")
    a2.axhline(0, color="k", lw=1)
    a2.set_xlabel("run index (chronological)"); a2.set_ylabel("residual"); a2.set_title("Residual vs run index")
    a2.grid(True, color="#eee", lw=0.5)
    fig.tight_layout(); fig.savefig(out / "figures" / "05_residual_diagnostics.png", dpi=150); plt.close(fig)

    # ── 6. Per-case RMSE bar + residual histogram ──────────────────────────────
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(12, 4.8))
    b1.bar(per_case_df.case.astype(str), per_case_df.rmse, color="#4c72b0", alpha=0.9)
    b1.axhline(locv_mean_rmse, color="#d62728", ls="--", lw=1.2, label=f"LOCV mean={locv_mean_rmse:.4f}")
    b1.set_xlabel("hold-out case"); b1.set_ylabel("RMSE"); b1.set_title("Per-case RMSE (3-seed mean)")
    b1.legend(frameon=False); b1.grid(True, axis="y", color="#eee", lw=0.5)
    b2.hist(agg.residual, bins=25, color="#55a868", alpha=0.9)
    b2.axvline(0, color="k", lw=1)
    b2.set_xlabel("residual (pred - true)"); b2.set_ylabel("count"); b2.set_title("Residual distribution")
    b2.grid(True, axis="y", color="#eee", lw=0.5)
    fig.tight_layout(); fig.savefig(out / "figures" / "06_per_case_rmse_and_error_hist.png", dpi=150); plt.close(fig)

    summary = {
        "config": BEST, "prefix_percent": PREFIX, "seeds": SEEDS,
        "overall_pooled": overall, "locv_mean_rmse": locv_mean_rmse,
        "hardest_cases": per_case_df.head(3)[["case", "rmse"]].to_dict("records"),
        "easiest_cases": per_case_df.tail(3)[["case", "rmse"]].to_dict("records"),
        "figures": sorted(p.name for p in (out / "figures").glob("*.png")),
        "execution_dir": str(out),
    }
    (out / "analysis" / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"\nLOCV mean RMSE={locv_mean_rmse:.6f}  pooled R2={overall['r2']:.4f}")
    print(f"Hardest cases: {summary['hardest_cases']}")
    print(f"Done. {out}")


if __name__ == "__main__":
    main()

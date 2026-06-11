#!/usr/bin/env python3
"""H4_S8_T2: Stacking — base XGBoost + Feature-GRU (Delta+Meta), meta MLP.

Base learners (identical inputs, Delta(24)+Meta(3)=27 feature vector, per-run):
  - XGBoost: tabular regressor on the 27-dim vector
  - Feature-GRU (head32): per-timestep DeltaMetaGRU over the run sequence of 27-dim vectors

Meta-learner: small MLP on [xgb_pred, gru_pred] -> VB.

Leak-handling (leave-one-group-out stacking, non-nested):
  For each (prefix, seed) the LOCV prediction of every case doubles as its OOF base
  prediction. For outer test case tc the MLP is trained on the other 14 cases' OOF
  rows and applied to tc. Caveat: OOF preds for training cases come from base models
  whose training sets included tc (standard non-nested LOGO stacking; same convention
  as the earlier prj05/prj06 stacking experiments).

Sweep prefix 10-100% (10% step), LOCV 15 cases, 3 seeds.
Reports: xgb / gru / avg-ensemble / stacking on identical base predictions.

Output: experiments/executions/H4/S8/{timestamp}_stacking_mlp/
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_H4_S5_feature_gru_hpo import (  # noqa: E402
    BASE, CASE_SCOPE, META_FEATURES, SENSORS, build_feature_matrix, build_sequences,
    compute_metrics, delta_col_names, fit_predict_gru, preprocess,
)

PREFIXES = list(range(10, 101, 10))
SEEDS = [0, 1, 2]
BEST_GRU = {**BASE, "head_hidden": 32, "name": "head32"}
INPUT_DIM = 4 * len(SENSORS) + len(META_FEATURES)
MODELS = ["xgb", "gru", "avg_ensemble", "stacking_mlp"]


def make_xgb(seed):
    return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
                        colsample_bytree=0.9, random_state=seed, verbosity=0, n_jobs=4)


def make_meta_mlp(seed):
    return MLPRegressor(hidden_layer_sizes=(16,), activation="relu", solver="adam",
                        max_iter=4000, random_state=seed, learning_rate_init=1e-3,
                        alpha=1e-3, early_stopping=False)


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT / "experiments" / "executions" / "H4" / "S8" / f"{ts}_stacking_mlp"
    for sub in ["analysis", "metrics", "figures", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(m):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); log_lines.append(line)

    log(f"=== H4_S8_T2 stacking (base XGB+GRU on Delta+Meta, meta MLP) === prefixes={PREFIXES} seeds={SEEDS}")
    sig = pd.read_csv(ROOT / "datasets/processed/mill_signal_data.csv", usecols=["case", "run"] + SENSORS)
    proc = pd.read_csv(ROOT / "datasets/processed/mill_process_info.csv")
    proc = proc[proc.case.isin(CASE_SCOPE)].copy()
    sig = sig[sig.case.isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(proc)
    feat = build_feature_matrix(sig, proc_clean, PREFIXES)
    delta_cols = delta_col_names()
    feature_cols = delta_cols + META_FEATURES        # Delta(24)+Meta(3)=27 for BOTH bases
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"feature matrix rows={len(feat)}  base feature dim={len(feature_cols)}  device={device}")

    seed_rows, per_case_rows = [], []
    t0 = time.time()
    for pct in PREFIXES:
        subset = feat[feat.signal_prefix_percent == float(pct)].copy()
        case_data = build_sequences(feat, delta_cols, float(pct))
        for seed in SEEDS:
            # 1) OOF base predictions for every case (= its LOCV prediction)
            oof = []   # rows: case, run, y_true, xgb, gru
            for tc in CASE_SCOPE:
                test_rows = subset[subset.case_id == tc].sort_values("run")
                if test_rows.empty or tc not in case_data:
                    continue
                train_rows = subset[subset.case_id != tc]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    xgb = make_xgb(seed)
                    xgb.fit(train_rows[feature_cols], train_rows.VB)
                    y_xgb = np.asarray(xgb.predict(test_rows[feature_cols]), dtype=float)
                train_cases = {c: case_data[c] for c in CASE_SCOPE if c != tc and c in case_data}
                y_gru = np.asarray(
                    fit_predict_gru(train_cases, {tc: case_data[tc]}, INPUT_DIM, device, BEST_GRU, seed)[tc],
                    dtype=float)
                for run, yt, px, pg in zip(test_rows.run.to_numpy(), test_rows.VB.to_numpy(), y_xgb, y_gru):
                    oof.append({"case": int(tc), "run": int(run), "y_true": float(yt),
                                "xgb": float(px), "gru": float(pg)})
            oof_df = pd.DataFrame(oof)

            # 2) per outer test case: meta MLP on other cases' OOF rows
            for tc in sorted(oof_df.case.unique()):
                te = oof_df[oof_df.case == tc]
                tr = oof_df[oof_df.case != tc]
                scaler = StandardScaler().fit(tr[["xgb", "gru"]])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mlp = make_meta_mlp(seed)
                    mlp.fit(scaler.transform(tr[["xgb", "gru"]]), tr.y_true)
                    y_stack = mlp.predict(scaler.transform(te[["xgb", "gru"]]))
                y_true = te.y_true.to_numpy()
                preds = {"xgb": te.xgb.to_numpy(), "gru": te.gru.to_numpy(),
                         "avg_ensemble": 0.5 * te.xgb.to_numpy() + 0.5 * te.gru.to_numpy(),
                         "stacking_mlp": np.asarray(y_stack, dtype=float)}
                for name, p in preds.items():
                    m = compute_metrics(y_true, p)
                    per_case_rows.append({"prefix": pct, "seed": seed, "model": name, "target_case": int(tc), **m})
            for name in MODELS:
                ssd = pd.DataFrame(per_case_rows).query("prefix == @pct and seed == @seed and model == @name")[["rmse", "r2", "mae"]].mean()
                seed_rows.append({"prefix": pct, "seed": seed, "model": name, "locv_rmse": float(ssd.rmse),
                                  "locv_r2": float(ssd.r2), "locv_mae": float(ssd.mae)})
        row = pd.DataFrame(seed_rows).query("prefix == @pct").groupby("model").locv_rmse.mean()
        log(f"prefix={pct}%  xgb={row['xgb']:.5f}  gru={row['gru']:.5f}  "
            f"avg={row['avg_ensemble']:.5f}  stack={row['stacking_mlp']:.5f}  [{time.time()-t0:.0f}s]")

    seed_df = pd.DataFrame(seed_rows); per_case_df = pd.DataFrame(per_case_rows)
    agg = (seed_df.groupby(["model", "prefix"], as_index=False)
           .agg(mean_rmse=("locv_rmse", "mean"), std_rmse=("locv_rmse", "std"),
                mean_r2=("locv_r2", "mean"), mean_mae=("locv_mae", "mean")))
    agg["std_rmse"] = agg["std_rmse"].fillna(0.0)
    seed_df.to_csv(out / "metrics" / "seed_metrics.csv", index=False)
    per_case_df.to_csv(out / "metrics" / "per_case_metrics.csv", index=False)
    agg.to_csv(out / "analysis" / "model_prefix_metrics.csv", index=False)

    best = {m: agg[agg.model == m].sort_values("mean_rmse").iloc[0] for m in MODELS}

    fig, ax = plt.subplots(figsize=(8.6, 5))
    colors = {"xgb": "#1f77b4", "gru": "#ff7f0e", "avg_ensemble": "#2ca02c", "stacking_mlp": "#d62728"}
    for m in MODELS:
        d = agg[agg.model == m].sort_values("prefix")
        ax.errorbar(d.prefix, d.mean_rmse, yerr=d.std_rmse, marker="o", capsize=3,
                    color=colors[m], label=f"{m} (best {best[m].mean_rmse:.4f}@{int(best[m].prefix)}%)")
    ax.set_xlabel("prefix input ratio (%)"); ax.set_ylabel("LOCV mean RMSE (3-seed)")
    ax.set_xticks(PREFIXES); ax.grid(True, color="#eee", lw=0.5)
    ax.set_title("H4_S8_T2 stacking (meta MLP) vs avg ensemble vs singles")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figures" / "01_prefix_rmse_4models.png", dpi=150); plt.close(fig)

    summary = {
        "experiment": "H4_S8_T2_stacking_mlp", "prefixes": PREFIXES, "seeds": SEEDS,
        "base_features": "Delta(24)+Meta(3)=27 for both bases",
        "meta": "MLP(16) on [xgb_pred, gru_pred], standardized; non-nested LOGO stacking",
        "best": {m: {"prefix": int(best[m].prefix), "rmse": float(best[m].mean_rmse), "r2": float(best[m].mean_r2)} for m in MODELS},
        "model_prefix_table": agg.to_dict("records"),
        "execution_dir": str(out),
    }
    (out / "analysis" / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    log("\n=== best per model (3-seed LOCV) ===")
    for m in MODELS:
        log(f"  {m:<13} best RMSE {best[m].mean_rmse:.6f} @ {int(best[m].prefix)}%  (R2 {best[m].mean_r2:.4f})")
    piv = agg.pivot(index="prefix", columns="model", values="mean_rmse")
    piv["stack_minus_avg"] = piv["stacking_mlp"] - piv["avg_ensemble"]
    log("\n=== stacking - avg_ensemble per prefix (음수면 stacking 우수) ===")
    for p, v in piv["stack_minus_avg"].items():
        log(f"  prefix={int(p)}%  stack-avg={v:+.6f}")
    (out / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"EXECUTION_DIR={out}", flush=True)


if __name__ == "__main__":
    main()

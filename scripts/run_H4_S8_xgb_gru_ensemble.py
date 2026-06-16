#!/usr/bin/env python3
"""H4_S8: XGBoost + Feature-GRU ensemble, input-ratio sweep 10-100%.

The two strongest H4 models are complementary (per [[14_H4_Leaderboard]] analysis):
feature_gru (run-to-run sequence) wins high-wear cases (e.g. 13), XGBoost (per-run
tabular) wins short/irregular cases (1,4,8). This averages their predictions.

Both use the SAME Delta+Meta (27) features at the SAME prefix r%, per-run nowcasting:
  - XGBoost: tabular regressor on delta+meta -> VB(run)
  - Feature-GRU (head32): per-timestep DeltaMetaGRU over the run sequence -> VB(run)
  - Ensemble: 0.5*XGB + 0.5*GRU (per run)

Sweep r in {10,20,...,100}%. LOCV 15 cases, 3 seeds. Reports XGB / GRU / Ensemble.

Output: experiments/executions/H4/S8/{timestamp}_xgb_gru_ensemble/
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


def make_xgb(seed):
    return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
                        colsample_bytree=0.9, random_state=seed, verbosity=0, n_jobs=4)


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT / "experiments" / "executions" / "H4" / "S8" / f"{ts}_xgb_gru_ensemble"
    for sub in ["analysis", "metrics", "figures", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(m):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); log_lines.append(line)

    log(f"=== H4_S8 XGBoost+Feature-GRU ensemble === prefixes={PREFIXES} seeds={SEEDS}")
    sig = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    proc = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    proc = proc[proc.case.isin(CASE_SCOPE)].copy()
    sig = sig[sig.case.isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(proc)
    feat = build_feature_matrix(sig, proc_clean, PREFIXES)
    delta_cols = delta_col_names()
    feature_cols = delta_cols + META_FEATURES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"feature matrix rows={len(feat)}  device={device}")

    seed_rows, per_case_rows = [], []
    MODELS = ["xgb", "gru", "ensemble"]
    t0 = time.time()
    for pct in PREFIXES:
        subset = feat[feat.signal_prefix_percent == float(pct)].copy()
        case_data = build_sequences(feat, delta_cols, float(pct))
        for seed in SEEDS:
            for tc in CASE_SCOPE:
                test_rows = subset[subset.case_id == tc].sort_values("run")
                if test_rows.empty or tc not in case_data:
                    continue
                train_rows = subset[subset.case_id != tc]
                y_true = test_rows.VB.to_numpy(dtype=float)
                # XGBoost (per-run tabular)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    xgb = make_xgb(seed)
                    xgb.fit(train_rows[feature_cols], train_rows.VB)
                    y_xgb = xgb.predict(test_rows[feature_cols])
                # Feature-GRU (per-timestep over run sequence)
                train_cases = {c: case_data[c] for c in CASE_SCOPE if c != tc and c in case_data}
                y_gru = fit_predict_gru(train_cases, {tc: case_data[tc]}, INPUT_DIM, device, BEST_GRU, seed)[tc]
                y_ens = 0.5 * np.asarray(y_xgb) + 0.5 * np.asarray(y_gru)
                for name, pred in [("xgb", y_xgb), ("gru", y_gru), ("ensemble", y_ens)]:
                    m = compute_metrics(y_true, np.asarray(pred))
                    per_case_rows.append({"prefix": pct, "seed": seed, "model": name, "target_case": tc, **m})
            for name in MODELS:
                ssd = pd.DataFrame(per_case_rows).query("prefix == @pct and seed == @seed and model == @name")[["rmse", "r2", "mae"]].mean()
                seed_rows.append({"prefix": pct, "seed": seed, "model": name, "locv_rmse": float(ssd.rmse),
                                  "locv_r2": float(ssd.r2), "locv_mae": float(ssd.mae)})
        row = pd.DataFrame(seed_rows).query("prefix == @pct").groupby("model").locv_rmse.mean()
        log(f"prefix={pct}%  xgb={row['xgb']:.5f}  gru={row['gru']:.5f}  ensemble={row['ensemble']:.5f}  [{time.time()-t0:.0f}s]")

    seed_df = pd.DataFrame(seed_rows); per_case_df = pd.DataFrame(per_case_rows)
    agg = (seed_df.groupby(["model", "prefix"], as_index=False)
           .agg(mean_rmse=("locv_rmse", "mean"), std_rmse=("locv_rmse", "std"),
                mean_r2=("locv_r2", "mean"), mean_mae=("locv_mae", "mean")))
    agg["std_rmse"] = agg["std_rmse"].fillna(0.0)
    seed_df.to_csv(out / "metrics" / "seed_metrics.csv", index=False)
    per_case_df.to_csv(out / "metrics" / "per_case_metrics.csv", index=False)
    agg.to_csv(out / "analysis" / "model_prefix_metrics.csv", index=False)

    best = {m: agg[agg.model == m].sort_values("mean_rmse").iloc[0] for m in MODELS}

    # Fig: prefix vs RMSE for the 3 models
    fig, ax = plt.subplots(figsize=(8.6, 5))
    colors = {"xgb": "#1f77b4", "gru": "#ff7f0e", "ensemble": "#d62728"}
    for m in MODELS:
        d = agg[agg.model == m].sort_values("prefix")
        ax.errorbar(d.prefix, d.mean_rmse, yerr=d.std_rmse, marker="o", capsize=3,
                    color=colors[m], label=f"{m} (best {best[m].mean_rmse:.4f}@{int(best[m].prefix)}%)")
    ax.set_xlabel("prefix input ratio (%)"); ax.set_ylabel("LOCV mean RMSE (3-seed)")
    ax.set_xticks(PREFIXES); ax.grid(True, color="#eee", lw=0.5)
    ax.set_title("H4_S8 XGBoost / Feature-GRU / Ensemble — input ratio sweep")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "figures" / "01_prefix_rmse_3models.png", dpi=150); plt.close(fig)

    # per-case: ensemble vs best single at each case's own best prefix
    pc = per_case_df.groupby(["model", "target_case", "prefix"]).rmse.mean().reset_index()
    rows = []
    for c in CASE_SCOPE:
        d = pc[pc.target_case == c]
        if d.empty:
            continue
        rec = {"case": c}
        for m in MODELS:
            dm = d[d.model == m]
            rec[m] = float(dm.rmse.min()) if not dm.empty else np.nan
        rows.append(rec)
    bycase = pd.DataFrame(rows)
    bycase["ens_best_of_single"] = bycase["ensemble"] - bycase[["xgb", "gru"]].min(axis=1)
    bycase.to_csv(out / "analysis" / "per_case_best_by_model.csv", index=False)

    summary = {
        "experiment": "H4_S8_xgb_gru_ensemble", "prefixes": PREFIXES, "seeds": SEEDS,
        "best": {m: {"prefix": int(best[m].prefix), "rmse": float(best[m].mean_rmse), "r2": float(best[m].mean_r2)} for m in MODELS},
        "model_prefix_table": agg.to_dict("records"),
        "execution_dir": str(out),
    }
    (out / "analysis" / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    log("\n=== best per model (3-seed LOCV) ===")
    for m in MODELS:
        log(f"  {m:<9} best RMSE {best[m].mean_rmse:.6f} @ {int(best[m].prefix)}%  (R2 {best[m].mean_r2:.4f})")
    # ensemble vs best single overall (at each prefix)
    piv = agg.pivot(index="prefix", columns="model", values="mean_rmse")
    piv["ens_minus_bestsingle"] = piv["ensemble"] - piv[["xgb", "gru"]].min(axis=1)
    log("\n=== ensemble - min(xgb,gru) per prefix (음수면 앙상블이 단일보다 우수) ===")
    for p, v in piv["ens_minus_bestsingle"].items():
        log(f"  prefix={int(p)}%  ens-bestsingle={v:+.6f}")
    (out / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"EXECUTION_DIR={out}", flush=True)


if __name__ == "__main__":
    main()

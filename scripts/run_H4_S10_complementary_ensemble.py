#!/usr/bin/env python3
"""H4_S10: Complementary ensemble strategies for XGBoost + Feature-GRU.

Motivation
----------
The two strongest H4 models fail on *different* cases:
  - Feature-GRU: Case 1 (non-monotone VB), Cases 4/8 (short sequences)
  - XGBoost   : Cases 13/14 (extreme VB > 1.0mm), Case 11 (long monotone wear)
Yet the naive 0.5/0.5 average (H4_S8) is WORSE than GRU alone (0.0985 vs 0.0908)
because XGBoost (0.1136) drags GRU down, and on shared-failure cases averaging
two wrong answers stays wrong.

This experiment evaluates smarter complementary schemes, all built on a single
standard LOCV pass that yields out-of-fold (OOF) predictions for every case
(reused across strategies; non-nested stacking, matching the H4_S8_T2 precedent).

Strategies (prefix=80%, GRU's best, the number to beat = 0.0908)
  Baselines : gru, xgb, avg(0.5)
  T1 wavg   : w*gru + (1-w)*xgb,  w in {0.5..1.0}
  T2 resid  : gru + alpha * XGB_resid(features [+gru_pred]),  alpha in {0.25..1.0}
              -> residual-correcting boost; XGB learns where GRU is wrong
  T3 oracle : per-case / per-run best-of-two (complementarity CEILING, not deployable)

Usage:
  venv312/bin/python3 scripts/run_H4_S10_complementary_ensemble.py --smoke
  venv312/bin/python3 scripts/run_H4_S10_complementary_ensemble.py
  venv312/bin/python3 scripts/run_H4_S10_complementary_ensemble.py --prefixes 80 100

Output: experiments/executions/H4/S10/{timestamp}_complementary_ensemble/
"""
from __future__ import annotations

import argparse
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
    BASE, CASE_SCOPE, META_FEATURES, SENSORS,
    build_feature_matrix, build_sequences, compute_metrics,
    delta_col_names, fit_predict_gru, preprocess,
)

SEEDS = [0, 1, 2]
BEST_GRU = {**BASE, "head_hidden": 32, "name": "head32"}
INPUT_DIM = 4 * len(SENSORS) + len(META_FEATURES)

W_GRU_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ALPHA_GRID = [0.25, 0.5, 0.75, 1.0]


def make_xgb(seed: int) -> XGBRegressor:
    """Base XGBoost (matches H4_S8 baseline)."""
    return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.9,
                        random_state=seed, verbosity=0, n_jobs=4)


def make_resid_xgb(seed: int) -> XGBRegressor:
    """Shallower XGBoost for residual correction (less overfit to GRU noise)."""
    return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                        subsample=0.8, colsample_bytree=0.9,
                        random_state=seed, verbosity=0, n_jobs=4)


# ─── OOF collection ──────────────────────────────────────────────────────────
def collect_oof(feat, feature_cols, delta_cols, pct, seed, device, log):
    """One standard LOCV pass: OOF per-run predictions for GRU and XGB.

    Returns a per-run DataFrame with columns:
      case, run, y_true, gru_pred, xgb_pred + feature columns
    """
    subset = feat[feat.signal_prefix_percent == float(pct)].copy()
    case_data = build_sequences(feat, delta_cols, float(pct))
    rows: list[dict] = []
    for tc in CASE_SCOPE:
        test_rows = subset[subset.case_id == tc].sort_values("run")
        if test_rows.empty or tc not in case_data:
            continue
        train_rows = subset[subset.case_id != tc]
        # XGBoost OOF
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xgb = make_xgb(seed)
            xgb.fit(train_rows[feature_cols], train_rows.VB)
            y_xgb = xgb.predict(test_rows[feature_cols])
        # GRU OOF
        train_cases = {c: case_data[c] for c in CASE_SCOPE if c != tc and c in case_data}
        y_gru = fit_predict_gru(train_cases, {tc: case_data[tc]}, INPUT_DIM, device, BEST_GRU, seed)[tc]

        tr = test_rows.reset_index(drop=True)
        for i in range(len(tr)):
            rec = {"case": tc, "run": int(tr.loc[i, "run"]),
                   "y_true": float(tr.loc[i, "VB"]),
                   "gru_pred": float(y_gru[i]), "xgb_pred": float(y_xgb[i])}
            for fc in feature_cols:
                rec[fc] = float(tr.loc[i, fc])
            rows.append(rec)
    return pd.DataFrame(rows)


# ─── Strategy evaluation (per seed) ──────────────────────────────────────────
def per_case_rmse(df: pd.DataFrame, pred_col: str) -> float:
    """LOCV RMSE = mean over cases of per-case RMSE (matches H4_S8 convention)."""
    per_case = []
    for c, g in df.groupby("case"):
        per_case.append(np.sqrt(np.mean((g[pred_col] - g["y_true"]) ** 2)))
    return float(np.mean(per_case))


def eval_strategies(oof: pd.DataFrame, feature_cols, seed) -> dict[str, np.ndarray]:
    """Add prediction columns for every strategy. Returns dict name->preds (per run)."""
    df = oof.copy()
    preds: dict[str, np.ndarray] = {}

    # Baselines
    preds["gru"] = df["gru_pred"].to_numpy()
    preds["xgb"] = df["xgb_pred"].to_numpy()
    preds["avg_0.5"] = 0.5 * df["gru_pred"].to_numpy() + 0.5 * df["xgb_pred"].to_numpy()

    # T1 weighted average
    for w in W_GRU_GRID:
        preds[f"wavg_g{w:.1f}"] = w * df["gru_pred"].to_numpy() + (1 - w) * df["xgb_pred"].to_numpy()

    # T2 residual boosting (LOCV over cases; residual-XGB trained on OOF residuals)
    df = df.copy()
    df["gru_resid"] = df["y_true"] - df["gru_pred"]
    for use_gpred in (False, True):
        cols = list(feature_cols) + (["gru_pred"] if use_gpred else [])
        tag = "gf" if use_gpred else "nf"
        resid_pred = np.zeros(len(df))
        for tc in df["case"].unique():
            tr = df[df["case"] != tc]
            te_idx = df.index[df["case"] == tc]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rx = make_resid_xgb(seed)
                rx.fit(tr[cols], tr["gru_resid"])
                resid_pred[df.index.get_indexer(te_idx)] = rx.predict(df.loc[te_idx, cols])
        for a in ALPHA_GRID:
            preds[f"resid_{tag}_a{a:.2f}"] = df["gru_pred"].to_numpy() + a * resid_pred

    # T3 oracle (ceiling, not deployable)
    # per-case: choose, for each case, the single model with lower case-RMSE
    gp, xp, yt = df["gru_pred"].to_numpy(), df["xgb_pred"].to_numpy(), df["y_true"].to_numpy()
    oracle_case = gp.copy()
    for c in df["case"].unique():
        m = (df["case"] == c).to_numpy()
        rg = np.sqrt(np.mean((gp[m] - yt[m]) ** 2))
        rx = np.sqrt(np.mean((xp[m] - yt[m]) ** 2))
        if rx < rg:
            oracle_case[m] = xp[m]
    preds["oracle_case"] = oracle_case
    # per-run: choose whichever prediction is closer (extreme ceiling)
    oracle_run = np.where(np.abs(gp - yt) <= np.abs(xp - yt), gp, xp)
    preds["oracle_run"] = oracle_run

    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", type=int, nargs="+", default=[80])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--smoke", action="store_true", help="1 seed, 5 cases.")
    args = ap.parse_args()

    seeds = [args.seeds[0]] if args.smoke else args.seeds
    scope = CASE_SCOPE[:5] if args.smoke else CASE_SCOPE

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    tag = "smoke" if args.smoke else "complementary_ensemble"
    out = ROOT / "experiments" / "executions" / "H4" / "S10" / f"{ts}_{tag}"
    for sub in ["analysis", "metrics", "figures", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(m):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); log_lines.append(line)

    log(f"=== H4_S10 complementary ensemble === prefixes={args.prefixes} seeds={seeds}")
    sig = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    proc = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    proc = proc[proc.case.isin(scope)].copy()
    sig = sig[sig.case.isin(scope)].copy()
    proc_clean = preprocess(proc)
    feat = build_feature_matrix(sig, proc_clean, args.prefixes)
    delta_cols = delta_col_names()
    feature_cols = delta_cols + META_FEATURES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"feature rows={len(feat)}  device={device}  features={len(feature_cols)}")

    all_seed_rows = []     # (prefix, seed, strategy, locv_rmse)
    all_oof = []
    t0 = time.time()
    for pct in args.prefixes:
        for seed in seeds:
            oof = collect_oof(feat, feature_cols, delta_cols, pct, seed, device, log)
            oof.insert(0, "seed", seed); oof.insert(0, "prefix", pct)
            all_oof.append(oof)
            preds = eval_strategies(oof.drop(columns=["prefix", "seed"]), feature_cols, seed)
            tmp = oof.copy()
            for name, p in preds.items():
                tmp["_p"] = p
                r = per_case_rmse(tmp.rename(columns={"_p": name}), name)
                all_seed_rows.append({"prefix": pct, "seed": seed, "strategy": name, "locv_rmse": r})
            gru_r = [x for x in all_seed_rows if x["prefix"] == pct and x["seed"] == seed and x["strategy"] == "gru"][0]["locv_rmse"]
            log(f"prefix={pct}% seed={seed}  gru={gru_r:.5f}  [{time.time()-t0:.0f}s]")

    seed_df = pd.DataFrame(all_seed_rows)
    oof_df = pd.concat(all_oof, ignore_index=True)
    agg = (seed_df.groupby(["prefix", "strategy"], as_index=False)
           .agg(mean_rmse=("locv_rmse", "mean"), std_rmse=("locv_rmse", "std")))
    agg["std_rmse"] = agg["std_rmse"].fillna(0.0)
    agg = agg.sort_values(["prefix", "mean_rmse"]).reset_index(drop=True)

    seed_df.to_csv(out / "metrics" / "seed_metrics.csv", index=False)
    oof_df.to_csv(out / "metrics" / "oof_predictions.csv", index=False)
    agg.to_csv(out / "analysis" / "strategy_ranking.csv", index=False)

    # ── Per-case breakdown at primary prefix (best deployable strategy vs gru/xgb) ──
    primary = args.prefixes[0]
    deployable = agg[(agg.prefix == primary) & (~agg.strategy.str.startswith("oracle"))]
    best_dep = deployable.sort_values("mean_rmse").iloc[0]["strategy"]
    # recompute per-case for gru / xgb / best_dep at primary, averaged over seeds
    pc_rows = []
    for seed in seeds:
        sub = oof_df[(oof_df.prefix == primary) & (oof_df.seed == seed)]
        preds = eval_strategies(sub.drop(columns=["prefix", "seed"]), feature_cols, seed)
        for name in ["gru", "xgb", best_dep]:
            tmp = sub.copy(); tmp["_p"] = preds[name]
            for c, g in tmp.groupby("case"):
                pc_rows.append({"seed": seed, "strategy": name, "case": int(c),
                                "rmse": float(np.sqrt(np.mean((g["_p"] - g["y_true"]) ** 2)))})
    pc_df = (pd.DataFrame(pc_rows).groupby(["strategy", "case"], as_index=False)
             .rmse.mean())
    pc_df.to_csv(out / "analysis" / "per_case_primary.csv", index=False)

    # ── Figures ──
    _fig_strategy_ranking(agg[agg.prefix == primary], out / "figures", primary)
    _fig_per_case(pc_df, best_dep, out / "figures", primary)
    if len(args.prefixes) > 1:
        _fig_prefix(agg, out / "figures")

    # ── Summary ──
    gru_primary = float(agg[(agg.prefix == primary) & (agg.strategy == "gru")].mean_rmse.iloc[0])
    best_dep_rmse = float(deployable.sort_values("mean_rmse").iloc[0]["mean_rmse"])
    oracle_case = float(agg[(agg.prefix == primary) & (agg.strategy == "oracle_case")].mean_rmse.iloc[0])
    oracle_run = float(agg[(agg.prefix == primary) & (agg.strategy == "oracle_run")].mean_rmse.iloc[0])
    summary = {
        "experiment": "H4_S10_complementary_ensemble",
        "prefixes": args.prefixes, "seeds": seeds, "primary_prefix": primary,
        "gru_baseline": gru_primary,
        "best_deployable": {"strategy": best_dep, "rmse": best_dep_rmse,
                            "delta_vs_gru": best_dep_rmse - gru_primary},
        "oracle_case_ceiling": oracle_case,
        "oracle_run_ceiling": oracle_run,
        "ranking_primary": agg[agg.prefix == primary].to_dict("records"),
        "execution_dir": str(out),
    }
    (out / "analysis" / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    log("\n=== strategy ranking @ primary prefix {}% (3-seed LOCV) ===".format(primary))
    for r in agg[agg.prefix == primary].itertuples():
        mark = ""
        if r.strategy == "gru": mark = "  <- baseline to beat"
        elif r.strategy == best_dep: mark = "  <- best deployable"
        elif r.strategy.startswith("oracle"): mark = "  (ceiling, not deployable)"
        log(f"  {r.strategy:<16} RMSE={r.mean_rmse:.6f} ±{r.std_rmse:.6f}{mark}")
    log(f"\nGRU baseline      : {gru_primary:.6f}")
    log(f"Best deployable   : {best_dep_rmse:.6f}  ({best_dep})   Δ={best_dep_rmse-gru_primary:+.6f}")
    log(f"Oracle (per-case) : {oracle_case:.6f}   (complementarity ceiling)")
    log(f"Oracle (per-run)  : {oracle_run:.6f}   (extreme ceiling)")

    (out / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"EXECUTION_DIR={out}", flush=True)


# ─── Figures ─────────────────────────────────────────────────────────────────
def _fig_strategy_ranking(agg_p, out: Path, prefix):
    d = agg_p.sort_values("mean_rmse")
    colors = []
    for s in d.strategy:
        if s.startswith("oracle"): colors.append("#2ca02c")
        elif s == "gru": colors.append("#ff7f0e")
        elif s == "xgb": colors.append("#1f77b4")
        else: colors.append("#7f7f7f")
    fig, ax = plt.subplots(figsize=(9, max(4, len(d) * 0.32)))
    ax.barh(range(len(d)), d.mean_rmse, xerr=d.std_rmse, color=colors, alpha=0.85, capsize=2)
    gru_r = float(agg_p[agg_p.strategy == "gru"].mean_rmse.iloc[0])
    ax.axvline(gru_r, color="#ff7f0e", ls="--", lw=1.2, label=f"GRU={gru_r:.4f}")
    ax.axvline(0.09, color="#888", ls=":", lw=1, label="target 0.09")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.strategy, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("LOCV mean RMSE (3-seed)")
    ax.set_title(f"H4_S10 complementary strategies @ prefix={prefix}%")
    ax.legend(frameon=False, fontsize=8)
    for i, (r, e) in enumerate(zip(d.mean_rmse, d.std_rmse)):
        ax.text(r + e + 3e-4, i, f"{r:.4f}", va="center", fontsize=6.5)
    fig.tight_layout(); fig.savefig(out / "01_strategy_ranking.png", dpi=150); plt.close(fig)


def _fig_per_case(pc_df, best_dep, out: Path, prefix):
    cases = sorted(pc_df.case.unique())
    piv = pc_df.pivot(index="case", columns="strategy", values="rmse").reindex(cases)
    x = np.arange(len(cases)); w = 0.26
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, piv["gru"], width=w, color="#ff7f0e", alpha=0.85, label="gru")
    ax.bar(x, piv["xgb"], width=w, color="#1f77b4", alpha=0.85, label="xgb")
    ax.bar(x + w, piv[best_dep], width=w, color="#d62728", alpha=0.85, label=f"best ({best_dep})")
    ax.set_xticks(x); ax.set_xticklabels([f"C{c}" for c in cases], fontsize=8)
    ax.axhline(0.09, color="#888", ls=":", lw=1, label="target 0.09")
    ax.set_ylabel("per-case RMSE (3-seed mean)")
    ax.set_title(f"Per-case RMSE @ prefix={prefix}% — gru vs xgb vs best deployable")
    ax.legend(frameon=False, fontsize=8); ax.grid(True, axis="y", color="#eee", lw=0.5)
    fig.tight_layout(); fig.savefig(out / "02_per_case_rmse.png", dpi=150); plt.close(fig)


def _fig_prefix(agg, out: Path):
    fig, ax = plt.subplots(figsize=(8.6, 5))
    keep = ["gru", "xgb", "avg_0.5", "oracle_case"]
    # also include best wavg and best resid per prefix
    for strat in keep:
        d = agg[agg.strategy == strat].sort_values("prefix")
        if not d.empty:
            ax.plot(d.prefix, d.mean_rmse, marker="o", label=strat)
    ax.set_xlabel("prefix (%)"); ax.set_ylabel("LOCV mean RMSE")
    ax.set_title("H4_S10 strategies across prefixes"); ax.grid(True, color="#eee", lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "03_prefix_curve.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()

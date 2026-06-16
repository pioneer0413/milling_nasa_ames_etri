#!/usr/bin/env python3
"""H4_S9: XGBoost hyperparameter optimization.

Base setting (H4_S8 baseline XGBoost):
  Delta+Meta features (24 delta + 3 meta = 27), prefix=100%, LOCV 15 cases, 3 seeds
  XGBRegressor(n_estimators=200, lr=0.05, max_depth=4, subsample=0.8,
               colsample_bytree=0.9)  ->  mean RMSE = 0.1136

Two-stage grid search:
  Stage 1 (~5 min,  ~80 configs): single-factor sweeps from BASE
  Stage 2 (~25 min, ~300 configs): systematic grid (lr×n×depth×mcw + reg + stochastic)
  --all: runs both stages back-to-back (~30 min)

Usage:
  venv312/bin/python3 scripts/run_H4_S9_xgb_hpo.py --smoke          # quick check
  venv312/bin/python3 scripts/run_H4_S9_xgb_hpo.py --stage 1        # ~5 min
  venv312/bin/python3 scripts/run_H4_S9_xgb_hpo.py --stage 2        # ~25 min
  venv312/bin/python3 scripts/run_H4_S9_xgb_hpo.py --all            # ~30 min total
  venv312/bin/python3 scripts/run_H4_S9_xgb_hpo.py --stage 2 --prefix-sweep

Output: experiments/executions/H4/S9/{timestamp}_xgb_hpo_{tag}/
  analysis/  config_ranking.csv  summary.json  prefix_sweep.csv(opt)
  metrics/   per_case_metrics.csv  seed_metrics.csv
  figures/   01_config_ranking.png  02_factor_sensitivity.png(s1)
             03_per_case_rmse.png  04_pred_vs_true.png  05_prefix_sweep.png(opt)
  logs/      run_log.txt
"""
from __future__ import annotations

import argparse
import itertools
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_H4_S5_feature_gru_hpo import (  # noqa: E402
    CASE_SCOPE, META_FEATURES, SENSORS,
    build_feature_matrix, compute_metrics, delta_col_names, preprocess,
)
from xgboost import XGBRegressor  # noqa: E402

# ─── BASE config ─────────────────────────────────────────────────────────────
BASE = dict(
    n_estimators=200, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.9,
    min_child_weight=1, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
    n_jobs=4,
)
SEEDS = [0, 1, 2]
DEFAULT_PREFIX = 100


# ─── Config builders ─────────────────────────────────────────────────────────
def _dedup(configs: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for c in configs:
        key = tuple(sorted((k, v) for k, v in c.items() if k != "name"))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def build_stage1_configs() -> list[dict]:
    """Single-factor sweeps — each param varied while others stay at BASE."""
    cfgs: list[dict] = [{**BASE, "name": "baseline"}]

    def add(name, **kw):
        cfgs.append({**BASE, **kw, "name": name})

    # n_estimators alone
    for n in [50, 100, 150, 300, 500, 800, 1000, 1500]:
        add(f"n{n}", n_estimators=n)

    # learning_rate paired with compensating n_estimators
    for lr, n in [(0.005, 1000), (0.01, 800), (0.01, 500),
                  (0.02, 400), (0.03, 300), (0.03, 500),
                  (0.08, 200), (0.1, 150), (0.15, 120), (0.2, 100)]:
        add(f"lr{lr}_n{n}", learning_rate=lr, n_estimators=n)

    # max_depth
    for d in [2, 3, 5, 6, 8, 10]:
        add(f"depth{d}", max_depth=d)

    # min_child_weight
    for mcw in [2, 3, 5, 8, 12, 20]:
        add(f"mcw{mcw}", min_child_weight=mcw)

    # subsample
    for ss in [0.4, 0.5, 0.6, 0.7, 0.9, 1.0]:
        add(f"ss{ss}", subsample=ss)

    # colsample_bytree
    for cs in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        add(f"cs{cs}", colsample_bytree=cs)

    # colsample_bylevel
    for cl in [0.5, 0.6, 0.7, 0.8]:
        add(f"cl{cl}", colsample_bylevel=cl)

    # reg_alpha (L1)
    for a in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0]:
        add(f"alpha{a}", reg_alpha=a)

    # reg_lambda (L2)
    for l in [0.1, 0.3, 0.5, 2.0, 3.0, 5.0, 10.0]:
        add(f"lambda{l}", reg_lambda=l)

    # gamma (min loss reduction for split)
    for g in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]:
        add(f"gamma{g}", gamma=g)

    # max_delta_step (useful for imbalanced regression)
    for mds in [0.5, 1.0, 2.0, 5.0]:
        add(f"mds{mds}", max_delta_step=mds)

    return _dedup(cfgs)


def build_stage2_configs() -> list[dict]:
    """Systematic multi-factor grid:
      Sub-grid A: lr × n_estimators × max_depth × min_child_weight  (~120 configs, heavy)
      Sub-grid B: reg_alpha × reg_lambda × gamma                     (~ 60 configs)
      Sub-grid C: subsample × colsample_bytree                       (~ 25 configs)
      Sub-grid D: selected 3-way combos (lr+n, depth, reg)           (~ 90 configs)
    """
    cfgs: list[dict] = [{**BASE, "name": "baseline"}]

    # Sub-grid A: core learning params (most impactful for XGBoost)
    # lr paired with n_estimators to keep total budget similar
    lr_n_pairs = [
        (0.005, 1000), (0.005, 1500),
        (0.01,  800),  (0.01, 1000),
        (0.02,  500),  (0.02,  800),
        (0.03,  400),  (0.03,  600),
        (0.05,  300),  (0.05,  500),
        (0.08,  250),  (0.1,   200),
    ]
    depths  = [3, 4, 5, 6]
    mcws    = [1, 3, 5, 8]
    for (lr, n), d, mcw in itertools.product(lr_n_pairs, depths, mcws):
        name = f"g_lr{lr}_n{n}_d{d}_mcw{mcw}"
        cfgs.append({**BASE, "learning_rate": lr, "n_estimators": n,
                     "max_depth": d, "min_child_weight": mcw, "name": name})

    # Sub-grid B: regularization (at a good base: lr=0.03, n=500, depth=4)
    reg_base = {**BASE, "learning_rate": 0.03, "n_estimators": 500, "max_depth": 4}
    for a, l, g in itertools.product(
        [0.0, 0.01, 0.1, 0.5, 2.0],
        [0.5, 1.0, 2.0, 5.0],
        [0.0, 0.1, 0.5],
    ):
        name = f"reg_a{a}_l{l}_g{g}"
        cfgs.append({**reg_base, "reg_alpha": a, "reg_lambda": l, "gamma": g, "name": name})

    # Sub-grid C: stochastic subsampling (at lr=0.03, n=500, depth=4, mcw=3)
    stoch_base = {**BASE, "learning_rate": 0.03, "n_estimators": 500,
                  "max_depth": 4, "min_child_weight": 3}
    for ss, cs in itertools.product([0.5, 0.6, 0.7, 0.8, 0.9], [0.5, 0.6, 0.7, 0.8, 0.9]):
        name = f"sc_ss{ss}_cs{cs}"
        cfgs.append({**stoch_base, "subsample": ss, "colsample_bytree": cs, "name": name})

    # Sub-grid D: depth × mcw × subsample for slow-lr large-n configs (3 anchor points)
    anchors = [(0.01, 1000), (0.02, 600), (0.03, 500)]
    d3_mcws = [3, 5]
    subs    = [0.7, 0.9]
    for (lr, n), d, mcw, ss in itertools.product(anchors, [3, 4, 5], d3_mcws, subs):
        name = f"d_lr{lr}_n{n}_d{d}_mcw{mcw}_ss{ss}"
        cfgs.append({**BASE, "learning_rate": lr, "n_estimators": n,
                     "max_depth": d, "min_child_weight": mcw, "subsample": ss, "name": name})

    return _dedup(cfgs)


# ─── LOCV runner ─────────────────────────────────────────────────────────────
def _make_xgb(cfg: dict, seed: int) -> XGBRegressor:
    kw = {k: v for k, v in cfg.items() if k != "name"}
    return XGBRegressor(**kw, random_state=seed, verbosity=0)


def run_locv(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    pct: int,
    cfg: dict,
    seed: int,
    case_scope: list[int] | None = None,
) -> pd.DataFrame:
    scope = case_scope or CASE_SCOPE
    subset = feat_df[feat_df["signal_prefix_percent"] == float(pct)].copy()
    rows: list[dict] = []
    for tc in scope:
        test_df = subset[subset["case_id"] == tc]
        train_df = subset[subset["case_id"] != tc]
        if test_df.empty or train_df.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = _make_xgb(cfg, seed)
            m.fit(train_df[feature_cols], train_df["VB"])
            y_pred = m.predict(test_df[feature_cols])
        y_true = test_df["VB"].to_numpy(dtype=float)
        rows.append({"target_case": tc, "n_runs": len(y_true),
                     **compute_metrics(y_true, y_pred)})
    return pd.DataFrame(rows)


def collect_predictions(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    pct: int,
    cfg: dict,
    seed: int,
) -> pd.DataFrame:
    """Full per-run predictions for scatter / trajectory plots."""
    subset = feat_df[feat_df["signal_prefix_percent"] == float(pct)].copy()
    rows: list[dict] = []
    for tc in CASE_SCOPE:
        test_df = subset[subset["case_id"] == tc]
        train_df = subset[subset["case_id"] != tc]
        if test_df.empty or train_df.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = _make_xgb(cfg, seed)
            m.fit(train_df[feature_cols], train_df["VB"])
            y_pred = m.predict(test_df[feature_cols])
        for r, yp, yt in zip(test_df["run"].values, y_pred, test_df["VB"].values):
            rows.append({"case": tc, "run": r, "y_true": yt, "y_pred": yp})
    return pd.DataFrame(rows)


# ─── Visualisation ───────────────────────────────────────────────────────────
def _fig_config_ranking(cfg_df: pd.DataFrame, baseline_rmse: float, out: Path,
                         top_n: int = 30) -> None:
    top = cfg_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.28)))
    colors = ["#d62728" if r == 1 else "#1f77b4" for r in top["rank"]]
    bars = ax.barh(range(len(top)), top["mean_rmse"], color=colors, alpha=0.85)
    ax.axvline(baseline_rmse, color="#ff7f0e", ls="--", lw=1.2, label=f"baseline={baseline_rmse:.4f}")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["config"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("LOCV mean RMSE (3-seed)")
    ax.set_title(f"XGBoost HPO — Top-{top_n} configs")
    ax.legend(frameon=False, fontsize=8)
    for i, (bar, row) in enumerate(zip(bars, top.itertuples())):
        ax.text(bar.get_width() + 5e-4, i, f"{row.mean_rmse:.5f}", va="center", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "01_config_ranking.png", dpi=150)
    plt.close(fig)


def _fig_factor_sensitivity(cfg_df: pd.DataFrame, baseline_rmse: float, out: Path) -> None:
    """For Stage 1: group configs by factor prefix and plot sensitivity."""
    factor_map = {
        "n_estimators": "n",
        "learning_rate": "lr",
        "max_depth": "depth",
        "min_child_weight": "mcw",
        "subsample": "ss",
        "colsample_bytree": "cs",
        "colsample_bylevel": "cl",
        "reg_alpha": "alpha",
        "reg_lambda": "lambda",
        "gamma": "gamma",
        "max_delta_step": "mds",
    }
    prefix_groups: dict[str, pd.DataFrame] = {}
    for full_name, prefix in factor_map.items():
        mask = cfg_df["config"].str.startswith(prefix) & ~cfg_df["config"].str.contains("_")
        if mask.sum() < 2:
            # allow compound prefix names like lr0.01_n500
            mask = cfg_df["config"].str.startswith(prefix)
        sub = cfg_df[mask].copy()
        if not sub.empty:
            prefix_groups[full_name] = sub

    n = len(prefix_groups)
    if n == 0:
        return
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (fname, sub) in enumerate(prefix_groups.items()):
        ax = axes[i // ncols][i % ncols]
        ax.axis("on")
        sub_sorted = sub.sort_values("mean_rmse")
        ax.bar(range(len(sub_sorted)), sub_sorted["mean_rmse"], color="#4c72b0", alpha=0.8)
        ax.axhline(baseline_rmse, color="#d62728", ls="--", lw=1, label=f"base={baseline_rmse:.4f}")
        ax.set_xticks(range(len(sub_sorted)))
        ax.set_xticklabels(sub_sorted["config"], rotation=45, ha="right", fontsize=6.5)
        ax.set_title(fname, fontsize=9)
        ax.set_ylabel("RMSE", fontsize=8)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(True, axis="y", color="#eee", lw=0.5)
    fig.suptitle("XGBoost HPO Stage 1 — per-factor sensitivity", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "02_factor_sensitivity.png", dpi=150)
    plt.close(fig)


def _fig_per_case_rmse(per_case_df: pd.DataFrame, best_cfg: str,
                        baseline_rmse_per_case: pd.Series, out: Path) -> None:
    """Grouped bar: best config vs baseline per case (3-seed mean)."""
    best_pc = (per_case_df[per_case_df["config"] == best_cfg]
               .groupby("target_case")["rmse"].mean()
               .reindex(CASE_SCOPE))
    x = np.arange(len(CASE_SCOPE))
    w = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, baseline_rmse_per_case.reindex(CASE_SCOPE).values,
           width=w, color="#1f77b4", alpha=0.8, label="baseline")
    ax.bar(x + w / 2, best_pc.values, width=w,
           color="#d62728", alpha=0.8, label=f"best ({best_cfg})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in CASE_SCOPE], fontsize=8)
    ax.set_ylabel("LOCV RMSE (3-seed mean)")
    ax.set_title("Per-case RMSE: best config vs baseline")
    ax.axhline(0.09, color="#888", ls=":", lw=1, label="target 0.09")
    ax.axhline(0.08, color="#555", ls=":", lw=1, label="target 0.08")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", color="#eee", lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "03_per_case_rmse.png", dpi=150)
    plt.close(fig)


def _fig_pred_vs_true(pred_df: pd.DataFrame, best_cfg: str, out: Path) -> None:
    """Scatter + trajectory plots for the best config (seed=0)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter
    ax = axes[0]
    cases = sorted(pred_df["case"].unique())
    cmap = matplotlib.colormaps.get_cmap("tab20")
    for i, c in enumerate(cases):
        sub = pred_df[pred_df["case"] == c]
        ax.scatter(sub["y_true"], sub["y_pred"], s=20, alpha=0.75,
                   color=cmap(i), label=f"C{c}")
    lims = [pred_df[["y_true", "y_pred"]].min().min() - 0.02,
            pred_df[["y_true", "y_pred"]].max().max() + 0.02]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True VB (mm)"); ax.set_ylabel("Predicted VB (mm)")
    ax.set_title(f"Predicted vs True — {best_cfg} (seed=0)")
    ax.legend(fontsize=5, ncol=3, frameon=False)
    ax.grid(True, color="#eee", lw=0.5)

    # Right: per-case RMSE dot plot
    ax2 = axes[1]
    rmse_by_case = (pred_df.groupby("case")
                    .apply(lambda g: np.sqrt(np.mean((g["y_pred"] - g["y_true"]) ** 2)))
                    .reindex(CASE_SCOPE).dropna())
    ax2.barh([f"C{c}" for c in rmse_by_case.index], rmse_by_case.values,
             color="#4c72b0", alpha=0.85)
    ax2.axvline(rmse_by_case.mean(), color="#d62728", ls="--", lw=1.2,
                label=f"mean={rmse_by_case.mean():.4f}")
    ax2.axvline(0.09, color="#888", ls=":", lw=1, label="target 0.09")
    ax2.set_xlabel("RMSE"); ax2.set_title("Per-case RMSE (seed=0)")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(True, axis="x", color="#eee", lw=0.5)

    fig.suptitle(f"Best config: {best_cfg}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "04_pred_vs_true.png", dpi=150)
    plt.close(fig)


def _fig_pred_trajectory(pred_df: pd.DataFrame, best_cfg: str, out: Path) -> None:
    """Per-case VB trajectory: true vs predicted over run progression (seed=0)."""
    cases = sorted(pred_df["case"].unique())
    ncols = 5
    nrows = int(np.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for i, case in enumerate(cases):
        ax = axes[i // ncols][i % ncols]
        ax.axis("on")
        sub = pred_df[pred_df["case"] == case].sort_values("run")
        runs = sub["run"].values
        rmse = float(np.sqrt(np.mean((sub["y_pred"] - sub["y_true"]) ** 2)))

        ax.plot(runs, sub["y_true"].values, "o-", color="#1f77b4", lw=1.5,
                ms=4, label="True")
        ax.plot(runs, sub["y_pred"].values, "s--", color="#d62728", lw=1.5,
                ms=4, label="Pred")
        ax.fill_between(runs, sub["y_true"].values, sub["y_pred"].values,
                        alpha=0.12, color="#888")
        ax.axhline(0.3, color="#aaa", ls=":", lw=0.8)   # worn threshold
        ax.set_title(f"Case {case}  RMSE={rmse:.4f}", fontsize=8)
        ax.set_xlabel("Run", fontsize=7)
        ax.set_ylabel("VB (mm)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, color="#eee", lw=0.5)
        if i == 0:
            ax.legend(fontsize=6, frameon=False)

    fig.suptitle(f"VB Prediction Trajectory — {best_cfg} (seed=0)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "06_pred_trajectory.png", dpi=150)
    plt.close(fig)


def _fig_prefix_sweep(ps_df: pd.DataFrame, best_cfg: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(ps_df["prefix"], ps_df["mean_rmse"], yerr=ps_df["std_rmse"],
                marker="o", capsize=3, color="#d62728", label="RMSE")
    best_row = ps_df.sort_values("mean_rmse").iloc[0]
    ax.scatter([best_row["prefix"]], [best_row["mean_rmse"]], s=120,
               facecolors="none", edgecolors="#d62728", linewidths=2, zorder=5,
               label=f"best={int(best_row['prefix'])}% ({best_row['mean_rmse']:.4f})")
    ax.axhline(0.09, color="#888", ls=":", lw=1, label="target 0.09")
    ax.axhline(0.08, color="#555", ls=":", lw=1, label="target 0.08")
    ax.set_xlabel("Prefix input ratio (%)"); ax.set_ylabel("LOCV mean RMSE (3-seed)")
    ax.set_xticks(ps_df["prefix"].tolist())
    ax.set_title(f"Prefix sweep — {best_cfg}")
    ax.legend(frameon=False, fontsize=8); ax.grid(True, color="#eee", lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "05_prefix_sweep.png", dpi=150)
    plt.close(fig)


# ─── Core sweep runner ────────────────────────────────────────────────────────
def run_sweep(
    configs: list[dict],
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    pct: int,
    seeds: list[int],
    log_fn,
    smoke_scope: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run HPO sweep; return (per_case_df, seed_df, config_ranking_df)."""
    per_case_rows: list[dict] = []
    seed_rows: list[dict] = []
    config_rows: list[dict] = []
    n_total = len(configs)
    t0 = time.time()

    for ci, cfg in enumerate(configs, 1):
        name = cfg["name"]
        c_rows: list[dict] = []
        for seed in seeds:
            tmp = run_locv(feat_df, feature_cols, pct, cfg, seed, smoke_scope)
            tmp.insert(0, "config", name); tmp.insert(1, "seed", seed)
            tmp.insert(2, "prefix", pct)
            per_case_rows.extend(tmp.to_dict("records"))
            c_rows.extend(tmp.to_dict("records"))
            seed_rows.append({
                "config": name, "seed": seed, "prefix": pct,
                "locv_rmse": float(tmp["rmse"].mean()),
                "locv_r2": float(tmp["r2"].mean()),
                "locv_mae": float(tmp["mae"].mean()),
            })
        c_df = pd.DataFrame(c_rows)
        by_seed = c_df.groupby("seed")["rmse"].mean()
        mean_rmse = float(by_seed.mean())
        std_rmse = float(by_seed.std()) if len(seeds) > 1 else 0.0
        config_rows.append({
            "config": name, "prefix": pct,
            "mean_rmse": mean_rmse, "std_rmse": std_rmse,
            "mean_r2": float(c_df["r2"].mean()),
            "mean_mae": float(c_df["mae"].mean()),
        })
        w = len(str(n_total))
        log_fn(f"[{ci:>{w}}/{n_total}] {name:<40}"
               f"  RMSE={mean_rmse:.6f} ±{std_rmse:.6f}"
               f"  R²={float(c_df['r2'].mean()):+.4f}"
               f"  [{time.time()-t0:>5.0f}s]")

    cfg_df = (pd.DataFrame(config_rows)
              .sort_values("mean_rmse")
              .reset_index(drop=True))
    cfg_df["rank"] = cfg_df.index + 1
    return pd.DataFrame(per_case_rows), pd.DataFrame(seed_rows), cfg_df


# ─── Main ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H4_S9 XGBoost HPO.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--stage", type=int, choices=[1, 2],
                     help="Stage 1: single-factor sweep. Stage 2: grid sweep.")
    grp.add_argument("--all", action="store_true",
                     help="Run stage 1 then stage 2 back-to-back (~30 min).")
    grp.add_argument("--smoke", action="store_true",
                     help="Baseline only, 1 seed, 5 cases (timing check).")
    p.add_argument("--prefix", type=int, default=DEFAULT_PREFIX)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--prefix-sweep", action="store_true",
                   help="After HPO, run best config over prefixes 10-100%%.")
    p.add_argument("--best-config", type=str, default=None,
                   help="Override auto-selected config for prefix sweep.")
    p.add_argument("--only", nargs="+", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [args.seeds[0]] if args.smoke else args.seeds

    if args.smoke:
        tag = "smoke"
    elif args.all:
        tag = "all"
    elif args.stage == 1:
        tag = "stage1"
    else:
        tag = "stage2"

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT / "experiments" / "executions" / "H4" / "S9" / f"{ts}_xgb_hpo_{tag}"
    for sub in ["metrics", "analysis", "figures", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    # ── Load data (once) ──────────────────────────────────────────────────
    log("Loading data...")
    sig_df = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                         usecols=["case", "run"] + SENSORS)
    proc_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    proc_df = proc_df[proc_df["case"].isin(CASE_SCOPE)].copy()
    sig_df = sig_df[sig_df["case"].isin(CASE_SCOPE)].copy()

    prefixes_needed = list(range(10, 101, 10)) if args.prefix_sweep else [args.prefix]
    log(f"Building feature matrix for prefixes={prefixes_needed}...")
    proc_clean = preprocess(proc_df)
    feat_df = build_feature_matrix(sig_df, proc_clean, prefixes_needed)
    delta_cols = delta_col_names()
    feature_cols = delta_cols + META_FEATURES
    log(f"Feature matrix: {len(feat_df)} rows  features={len(feature_cols)}")

    # ── Build config lists ────────────────────────────────────────────────
    if args.smoke:
        all_configs = [c for c in build_stage1_configs() if c["name"] == "baseline"]
        smoke_scope = CASE_SCOPE[:5]
    elif args.all:
        s1 = build_stage1_configs()
        s2 = build_stage2_configs()
        # combine; dedup by removing s2 entries already in s1
        s1_keys = {tuple(sorted((k, v) for k, v in c.items() if k != "name")) for c in s1}
        s2_new = [c for c in s2
                  if tuple(sorted((k, v) for k, v in c.items() if k != "name")) not in s1_keys]
        all_configs = s1 + s2_new
        smoke_scope = None
    elif args.stage == 1:
        all_configs = build_stage1_configs()
        smoke_scope = None
    else:
        all_configs = build_stage2_configs()
        smoke_scope = None

    if args.only:
        all_configs = [c for c in all_configs if c["name"] in set(args.only)]

    log(f"=== H4_S9 XGBoost HPO === tag={tag}  prefix={args.prefix}%"
        f"  seeds={seeds}  configs={len(all_configs)}")

    # ── Sweep ─────────────────────────────────────────────────────────────
    per_case_df, seed_df, cfg_df = run_sweep(
        all_configs, feat_df, feature_cols, args.prefix, seeds, log,
        smoke_scope=smoke_scope,
    )

    # ── Prefix sweep ──────────────────────────────────────────────────────
    prefix_sweep_rows: list[dict] = []
    if args.prefix_sweep:
        sweep_name = args.best_config or cfg_df.iloc[0]["config"]
        sweep_cfg = next((c for c in all_configs if c["name"] == sweep_name), None)
        if sweep_cfg is None:
            sweep_cfg = {**BASE, "name": sweep_name}
        log(f"\n=== Prefix sweep: config={sweep_name} ===")
        for pct in range(10, 101, 10):
            pct_rows: list[dict] = []
            for seed in seeds:
                tmp = run_locv(feat_df, feature_cols, pct, sweep_cfg, seed)
                pct_rows.extend(tmp.to_dict("records"))
                prefix_sweep_rows.append({
                    "config": sweep_name, "prefix": pct, "seed": seed,
                    "locv_rmse": float(tmp["rmse"].mean()),
                })
            log(f"  prefix={pct:>3}%  RMSE={pd.DataFrame(pct_rows)['rmse'].mean():.6f}")

    # ── Save CSVs ─────────────────────────────────────────────────────────
    per_case_df.to_csv(out / "metrics" / "per_case_metrics.csv", index=False)
    seed_df.to_csv(out / "metrics" / "seed_metrics.csv", index=False)
    cfg_df.to_csv(out / "analysis" / "config_ranking.csv", index=False)
    (out / "analysis" / "configs.json").write_text(
        json.dumps({c["name"]: {k: v for k, v in c.items() if k != "name"}
                    for c in all_configs}, indent=2), encoding="utf-8"
    )

    ps_df: pd.DataFrame | None = None
    if prefix_sweep_rows:
        ps_raw = pd.DataFrame(prefix_sweep_rows)
        ps_df = (ps_raw.groupby("prefix", as_index=False)
                 .agg(mean_rmse=("locv_rmse", "mean"), std_rmse=("locv_rmse", "std")))
        ps_df["std_rmse"] = ps_df["std_rmse"].fillna(0.0)
        ps_df.to_csv(out / "analysis" / "prefix_sweep.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────
    log("Generating figures...")
    baseline_row = cfg_df[cfg_df["config"] == "baseline"]
    baseline_rmse = float(baseline_row["mean_rmse"].values[0]) if not baseline_row.empty else float("nan")
    best_cfg_name = cfg_df.iloc[0]["config"]

    _fig_config_ranking(cfg_df, baseline_rmse, out / "figures")

    if args.stage == 1 or args.smoke:
        _fig_factor_sensitivity(cfg_df, baseline_rmse, out / "figures")

    # Per-case RMSE: best vs baseline
    base_pc = (per_case_df[per_case_df["config"] == "baseline"]
               .groupby("target_case")["rmse"].mean())
    if not base_pc.empty:
        _fig_per_case_rmse(per_case_df, best_cfg_name, base_pc, out / "figures")

    # Predicted vs true (seed=0, best prefix, best config)
    best_cfg = next((c for c in all_configs if c["name"] == best_cfg_name), {**BASE, "name": best_cfg_name})
    pred_df = collect_predictions(feat_df, feature_cols, args.prefix, best_cfg, seed=0)
    if not pred_df.empty:
        _fig_pred_vs_true(pred_df, best_cfg_name, out / "figures")
        _fig_pred_trajectory(pred_df, best_cfg_name, out / "figures")

    if ps_df is not None:
        _fig_prefix_sweep(ps_df, best_cfg_name, out / "figures")

    # ── Summary ───────────────────────────────────────────────────────────
    best = cfg_df.iloc[0]
    improvement = baseline_rmse - float(best["mean_rmse"])
    summary = {
        "experiment": f"H4_S9_xgb_hpo_{tag}",
        "prefix": args.prefix, "seeds": seeds,
        "n_configs": len(all_configs),
        "base_config": BASE,
        "baseline_rmse": baseline_rmse,
        "best_config": best["config"],
        "best_rmse": float(best["mean_rmse"]),
        "best_std": float(best["std_rmse"]),
        "best_r2": float(best["mean_r2"]),
        "improvement_vs_baseline": improvement,
        "config_ranking": cfg_df.head(20).to_dict("records"),
        "execution_dir": str(out),
    }
    (out / "analysis" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )

    log("\n=== Top-10 configs ===")
    for r in cfg_df.head(10).itertuples(index=False):
        mark = " <- best" if r.rank == 1 else ""
        log(f"  rank={r.rank:>3}  {r.config:<40}  RMSE={r.mean_rmse:.6f} ±{r.std_rmse:.6f}{mark}")
    log(f"\nBaseline : {baseline_rmse:.6f}")
    log(f"Best     : {float(best.mean_rmse):.6f}  ({best.config})")
    log(f"Δ improve: {improvement:+.6f}")

    (out / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"EXECUTION_DIR={out}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""H16_S1: Custom feature engineering + Input ratio sweep for XGB / SVR / RF.

Feature design:
  smcAC  → std, peak_to_peak (max-min), robust_p2p (p95-p5)
  smcDC  → mean, rms
  Delta: (current_run - first_run) × 5 features + meta (DOC, feed, material) = 8-dim

Models:
  T1: XGBoost  (lr=0.03, n_estimators=600, max_depth=3, min_child_weight=8)
  T2: SVR      (RBF, C=10, epsilon=0.05, StandardScaler)
  T3: Random Forest (n_estimators=300, min_samples_leaf=3)

Prefix ratio sweep: {10, 20, ..., 100}% of total signal length.
LOCV: 15 cases, seeds=[0,1,2], observed-VB RMSE.
"""
from __future__ import annotations

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
EXCLUDED_RUNS = {(2, 1), (12, 1)}
NON_OBSERVED_RUNS = {
    (1,2),(1,3),(1,5),(1,16),(2,6),(8,4),(11,7),(11,14),(11,17),
    (12,4),(12,10),(13,1),(13,2),(14,1),(14,5),(15,1),(16,1),(16,2),(16,4),
}
META_FEATURES = ["DOC", "feed", "material"]
THRESH        = 1e6
RATIOS        = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SEEDS         = [0, 1, 2]

XGB_CFG = dict(
    learning_rate=0.03, n_estimators=600, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.9, n_jobs=4, verbosity=0,
)
RF_CFG = dict(n_estimators=300, min_samples_leaf=3, n_jobs=-1)
SVR_CFG = dict(kernel="rbf", C=10, epsilon=0.05)

FEATURE_NAMES = ["AC_std", "AC_p2p", "AC_rp2p", "DC_mean", "DC_rms"]
FEATURE_COLS  = [f"d_{n}" for n in FEATURE_NAMES] + META_FEATURES  # 8 total


# ─── Data utils ───────────────────────────────────────────────────────────────
def parse_signal(value: object) -> np.ndarray:
    return np.nan_to_num(
        np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"] == min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside")
    )
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    excl = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~excl].reset_index(drop=True)


def extract_custom_features(ac: np.ndarray, dc: np.ndarray) -> np.ndarray:
    """Extract 5 custom features from smcAC and smcDC arrays."""
    if len(ac) == 0:
        return np.zeros(5, dtype=np.float32)
    ac_std  = float(np.std(ac))
    ac_p2p  = float(np.max(ac) - np.min(ac))
    ac_rp2p = float(np.percentile(ac, 95) - np.percentile(ac, 5))
    dc_mean = float(np.mean(dc)) if len(dc) > 0 else 0.0
    dc_rms  = float(np.sqrt(np.mean(dc ** 2))) if len(dc) > 0 else 0.0
    return np.array([ac_std, ac_p2p, ac_rp2p, dc_mean, dc_rms], dtype=np.float32)


# ─── Feature cache ────────────────────────────────────────────────────────────
def build_feature_cache(
    signal_df: pd.DataFrame,
    proc_clean: pd.DataFrame,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Pre-compute custom features for all (case, run, ratio) combinations."""
    cache: dict[tuple[int, int, int], np.ndarray] = {}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        sig_row = signal_df[
            (signal_df["case"] == case_id) & (signal_df["run"] == run_id)
        ]
        if sig_row.empty:
            continue
        sr  = sig_row.iloc[0]
        ac  = parse_signal(sr["smcAC"])
        dc  = parse_signal(sr["smcDC"])
        if np.abs(ac).max() > THRESH or np.abs(dc).max() > THRESH:
            continue
        base_len = min(len(ac), len(dc))
        for r in RATIOS:
            cutoff = max(1, int(np.floor(base_len * r / 100)))
            feat   = extract_custom_features(ac[:cutoff], dc[:cutoff])
            cache[(case_id, run_id, r)] = feat
    return cache


def build_feat_df(
    cache: dict[tuple[int, int, int], np.ndarray],
    first_run: dict[int, int],
    proc_clean: pd.DataFrame,
    ratio: int,
) -> pd.DataFrame:
    """Build feature DataFrame with delta features for a given ratio."""
    rows = []
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id, ratio)
        if key not in cache:
            continue
        raw   = cache[key].astype(np.float64)
        ref_r = first_run.get(case_id, run_id)
        ref   = cache.get((case_id, ref_r, ratio), raw)
        delta = raw - ref
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        rec = {
            "case_id": case_id, "run": run_id,
            "VB": float(row.VB) if not pd.isna(row.VB) else 0.0,
        }
        for mf in META_FEATURES:
            rec[mf] = float(getattr(row, mf, 0.0))
        for k, n in enumerate(FEATURE_NAMES):
            rec[f"d_{n}"] = float(delta[k])
        rows.append(rec)
    return pd.DataFrame(rows)


# ─── LOCV ─────────────────────────────────────────────────────────────────────
def locv(feat_df: pd.DataFrame, make_model, seed: int) -> float:
    """LOCV on 15 cases; returns mean observed-VB RMSE."""
    rmses: list[float] = []
    for tc in CASE_SCOPE:
        test_df  = feat_df[feat_df["case_id"] == tc]
        train_df = feat_df[feat_df["case_id"] != tc]
        if test_df.empty or train_df.empty:
            continue
        model = make_model(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_df[FEATURE_COLS], train_df["VB"])
            y_pred = model.predict(test_df[FEATURE_COLS])
        obs = test_df.apply(
            lambda r: (int(r["case_id"]), int(r["run"])) not in NON_OBSERVED_RUNS,
            axis=1,
        ).to_numpy()
        y_true = test_df["VB"].to_numpy(float)[obs]
        if len(y_true) > 0:
            rmses.append(float(np.sqrt(mean_squared_error(y_true, y_pred[obs]))))
    return float(np.mean(rmses)) if rmses else float("nan")


def run_ratio_sweep(
    cache: dict,
    first_run: dict,
    proc_clean: pd.DataFrame,
    model_name: str,
    make_model,
    t0: float,
) -> dict[int, dict]:
    """Sweep over all ratios × seeds for one model; return per-ratio stats."""
    results: dict[int, dict] = {}
    for r in RATIOS:
        feat_df = build_feat_df(cache, first_run, proc_clean, r)
        feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat_df.fillna(0.0, inplace=True)
        seed_rmses = []
        for s in SEEDS:
            rm = locv(feat_df, make_model, s)
            seed_rmses.append(rm)
        mean_r = float(np.mean(seed_rmses))
        std_r  = float(np.std(seed_rmses))
        results[r] = {"mean": mean_r, "std": std_r, "seeds": seed_rmses}
        elapsed = time_mod.time() - t0
        print(
            f"  {model_name} ratio={r:3d}%  RMSE={mean_r:.6f}  std={std_r:.6f}  [{elapsed:.0f}s]",
            flush=True,
        )
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_dir = (
        ROOT / "experiments" / "executions" / "H16" / "S1"
        / f"{ts}_custom_feat_ratio_sweep"
    )
    for sub in ["data", "figures"]:
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)

    t0 = time_mod.time()
    def log(msg: str) -> None:
        print(f"[{time_mod.time()-t0:6.0f}s] {msg}", flush=True)

    log("=== H16_S1: Custom Feature Ratio Sweep ===")
    log("Feature: AC(std,p2p,rp2p) + DC(mean,rms)  →  delta+meta = 8-dim")
    log(f"Models: XGBoost / SVR / RandomForest")
    log(f"Ratios={RATIOS}, Seeds={SEEDS}, LOCV=15 cases")
    log("")

    log("Loading data...")
    signal_df  = pd.read_csv(
        ROOT / "datasets/nasa/raw_signal.csv",
        usecols=["case", "run", "smcAC", "smcDC"],
    )
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    first_run: dict[int, int] = {
        int(g): int(df.sort_values("run")["run"].iloc[0])
        for g, df in proc_clean.groupby("case")
    }

    log("Building feature cache...")
    cache = build_feature_cache(signal_df, proc_clean)
    log(f"Cache size: {len(cache)} entries ({len(proc_clean)} runs × {len(RATIOS)} ratios)")
    log("")

    XGB_BEST  = 0.103660  # H12_S1_T2 prefix-90%

    def make_xgb(seed: int):
        return XGBRegressor(**XGB_CFG, random_state=seed)

    def make_svr(_seed: int):
        return Pipeline([("scaler", StandardScaler()), ("svr", SVR(**SVR_CFG))])

    def make_rf(seed: int):
        return RandomForestRegressor(**RF_CFG, random_state=seed)

    all_results: dict[str, dict] = {}

    for model_name, make_fn in [("XGBoost", make_xgb), ("SVR", make_svr), ("RF", make_rf)]:
        log(f"--- {model_name} ---")
        res = run_ratio_sweep(cache, first_run, proc_clean, model_name, make_fn, t0)
        all_results[model_name] = res
        best_r    = min(res, key=lambda r: res[r]["mean"])
        best_rmse = res[best_r]["mean"]
        log(f"  {model_name} best: ratio={best_r}%  RMSE={best_rmse:.6f}  (XGB_baseline={XGB_BEST:.6f})")
        log("")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    rows = []
    for model_name, res in all_results.items():
        for r, v in res.items():
            rows.append({
                "model": model_name,
                "ratio": r,
                "rmse_mean": v["mean"],
                "rmse_std":  v["std"],
                **{f"seed{s}": v["seeds"][i] for i, s in enumerate(SEEDS)},
            })
    result_df = pd.DataFrame(rows)
    result_df.to_csv(exp_dir / "data" / "results.csv", index=False)
    log(f"Saved: {exp_dir / 'data' / 'results.csv'}")

    # ── Summary table ─────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    header = f"{'Ratio':>6}  {'XGB_RMSE':>10}  {'SVR_RMSE':>10}  {'RF_RMSE':>10}"
    log(header)
    log("-" * len(header))
    for r in RATIOS:
        xgb_r = all_results["XGBoost"][r]["mean"]
        svr_r = all_results["SVR"][r]["mean"]
        rf_r  = all_results["RF"][r]["mean"]
        stars = {
            "XGBoost": "★" if r == min(all_results["XGBoost"], key=lambda x: all_results["XGBoost"][x]["mean"]) else " ",
            "SVR":     "★" if r == min(all_results["SVR"],     key=lambda x: all_results["SVR"][x]["mean"])     else " ",
            "RF":      "★" if r == min(all_results["RF"],      key=lambda x: all_results["RF"][x]["mean"])      else " ",
        }
        log(f"{r:>5}%  {stars['XGBoost']}{xgb_r:.6f}  {stars['SVR']}{svr_r:.6f}  {stars['RF']}{rf_r:.6f}")
    log("")
    for model_name, res in all_results.items():
        best_r    = min(res, key=lambda r: res[r]["mean"])
        best_rmse = res[best_r]["mean"]
        log(f"{model_name:>10} best: ratio={best_r:3d}%  RMSE={best_rmse:.6f}"
            f"  Δ(XGB_baseline)={best_rmse - XGB_BEST:+.6f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = {"XGBoost": "darkorange", "SVR": "steelblue", "RF": "seagreen"}
    markers = {"XGBoost": "o", "SVR": "s", "RF": "^"}
    for model_name, res in all_results.items():
        xs  = RATIOS
        ys  = [res[r]["mean"] for r in RATIOS]
        errs = [res[r]["std"]  for r in RATIOS]
        ax.plot(xs, ys, color=colors[model_name], marker=markers[model_name],
                linewidth=1.5, label=model_name)
        ax.fill_between(xs,
                        [y - e for y, e in zip(ys, errs)],
                        [y + e for y, e in zip(ys, errs)],
                        color=colors[model_name], alpha=0.12)
    ax.axhline(XGB_BEST, color="black", linestyle="--", linewidth=1.2,
               label=f"XGB prefix-90% best ({XGB_BEST:.4f})")
    ax.set_xlabel("Prefix ratio (%)")
    ax.set_ylabel("Observed-VB RMSE (LOCV)")
    ax.set_title(
        "H16.S1: Custom Feature (AC std/p2p/rp2p + DC mean/rms) + Prefix Ratio Sweep\n"
        "XGBoost / SVR / Random Forest"
    )
    ax.set_xticks(RATIOS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(exp_dir / "figures" / "ratio_sweep.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(exp_dir / "figures" / "ratio_sweep.svg"), bbox_inches="tight")
    plt.close(fig)

    log(f"\nTotal time: {time_mod.time()-t0:.0f}s")
    log(f"Results: {exp_dir}")
    print(f"EXECUTION_DIR={exp_dir}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""prj02 current-only: Input-Ratio Experiment (all cases, LOCV).

Constraints (per request):
  - Models  : SVR, XGBoost  (only these two)
  - Sensors : Current only  -> smcAC, smcDC
  - Features: Delta-base only -> delta(mean,rms,std,peak) of the 2 current
              sensors = 8 features  (no Raw, no Meta)

Experiment: signal-prefix input ratio 10%..100% (10% step). For every case in
CASE_SCOPE, Leave-One-Case-Out: train on the other cases at a fixed ratio,
predict the held-out case's runs. Report RMSE per (case, ratio, model) and the
15-fold mean RMSE per (ratio, model).

Fixed hyperparameters (so the only varying experimental factor is input ratio):
  SVR     : rbf, C=100, gamma='scale', epsilon=0.05      (+ StandardScaler)
  XGBoost : n_estimators=200, max_depth=3, lr=0.1, subsample=1.0, seed=42

Outputs: cache/results.csv (per case/ratio/model), cache/summary.csv (mean per
ratio/model), console summary.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)

CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
CURRENT       = ["smcAC", "smcDC"]
EXCLUDED_RUNS = {(2, 1), (12, 1)}
THRESH        = 1e6
PREFIXES      = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
STATS         = ["mean", "rms", "std", "peak"]
SEED          = 0

DELTA_COLS = [f"{s}__delta_{f}" for s in CURRENT for f in STATS]   # 8 features

# ── feature extraction (current sensors only) ────────────────────────────────

def parse_signal(v):
    a = np.fromstring(str(v).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

def extract_stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        a = np.zeros(1)
    return np.array([np.mean(a), np.sqrt(np.mean(a**2)), np.std(a), np.abs(a).max()])

def preprocess(df):
    df = df.copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"] == min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside"))
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)

def build_feature_matrix():
    cache_pkl = CACHE / "feat_current_delta.pkl"
    if cache_pkl.exists():
        return pd.read_pickle(cache_pkl)
    print("Building current-delta feature matrix...", flush=True)
    sig  = pd.read_csv(ROOT/"datasets/processed/mill_signal_data.csv",
                       usecols=["case", "run"]+CURRENT)
    proc = pd.read_csv(ROOT/"datasets/processed/mill_process_info.csv")
    proc = proc[proc["case"].isin(CASE_SCOPE)].copy()
    sig  = sig[sig["case"].isin(CASE_SCOPE)].copy()
    proc = preprocess(proc)

    raw_cache = {}
    for row in proc.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        sr = sig[(sig["case"] == cid) & (sig["run"] == rid)]
        if sr.empty:
            continue
        arrays = {s: parse_signal(sr.iloc[0][s]) for s in CURRENT}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        blen = min(len(a) for a in arrays.values())
        for pct in PREFIXES:
            end = max(1, int(np.ceil(blen * pct / 100.0)))
            raw_cache[(cid, rid, pct)] = np.concatenate(
                [extract_stats(arrays[s][:end]) for s in CURRENT])

    first_run = {}
    for cid in proc["case"].unique():
        runs = sorted(r for (c, r, _) in raw_cache if c == cid)
        if runs:
            first_run[int(cid)] = runs[0]

    rows = []
    for row in proc.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        if (cid, rid) in EXCLUDED_RUNS:
            continue
        vb = float(row.VB) if not pd.isna(row.VB) else 0.0
        ref = first_run.get(cid, rid)
        for pct in PREFIXES:
            rf, rr = raw_cache.get((cid, rid, pct)), raw_cache.get((cid, ref, pct))
            if rf is None or rr is None:
                continue
            delta = rf - rr
            rec = {"case_id": cid, "run": rid, "signal_prefix_percent": float(pct), "VB": vb}
            for col, val in zip(DELTA_COLS, delta):
                rec[col] = float(val)
            rows.append(rec)
    feat = pd.DataFrame(rows).sort_values(
        ["signal_prefix_percent", "case_id", "run"]).reset_index(drop=True)
    feat.to_pickle(cache_pkl)
    print(f"  cached {cache_pkl}  shape={feat.shape}", flush=True)
    return feat

# ── models ────────────────────────────────────────────────────────────────────

def make_svr():
    return SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.05)

def make_xgb():
    return XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                        subsample=1.0, random_state=42, n_jobs=1, verbosity=0)

def fit_predict(model_name, Xtr, ytr, Xte):
    if model_name == "SVR":
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        m = make_svr()
    else:
        m = make_xgb()
    m.fit(Xtr, ytr)
    return m.predict(Xte)

# ── experiment ────────────────────────────────────────────────────────────────

def main():
    feat = build_feature_matrix()
    print(f"Feature matrix: {feat.shape}  features={DELTA_COLS}\n", flush=True)
    rows = []
    for model_name in ["SVR", "XGBoost"]:
        for pct in PREFIXES:
            sub = feat[feat["signal_prefix_percent"] == float(pct)]
            for tc in CASE_SCOPE:
                tr = sub[sub["case_id"] != tc]
                te = sub[sub["case_id"] == tc].sort_values("run")
                if tr.empty or te.empty:
                    continue
                Xtr, ytr = tr[DELTA_COLS].to_numpy(float), tr["VB"].to_numpy(float)
                Xte, yte = te[DELTA_COLS].to_numpy(float), te["VB"].to_numpy(float)
                pred = fit_predict(model_name, Xtr, ytr, Xte)
                rmse = float(np.sqrt(mean_squared_error(yte, pred)))
                rows.append({"model": model_name, "input_ratio_pct": pct, "case": tc,
                             "n_runs": len(te), "rmse": rmse,
                             "mae": float(mean_absolute_error(yte, pred)),
                             "r2": float(r2_score(yte, pred)) if len(yte) > 1 else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(CACHE/"results.csv", index=False)

    # 15-fold mean RMSE per (model, ratio)
    summ = (df.groupby(["model", "input_ratio_pct"])
              .agg(mean_rmse=("rmse", "mean"), median_rmse=("rmse", "median"),
                   mean_r2=("r2", "mean"))
              .reset_index())
    summ.to_csv(CACHE/"summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("="*70)
    print("15-fold mean LOCV RMSE per input ratio")
    print("="*70)
    piv = summ.pivot(index="input_ratio_pct", columns="model", values="mean_rmse")
    print(piv.round(4).to_string())
    print("\nBest ratio per model (by 15-fold mean RMSE):")
    for m in ["SVR", "XGBoost"]:
        s = summ[summ.model == m].sort_values("mean_rmse").iloc[0]
        print(f"  {m:8s}: ratio={int(s.input_ratio_pct):3d}%  mean_rmse={s.mean_rmse:.4f}  mean_r2={s.mean_r2:.3f}")

    print("\nPer-case best (min RMSE over ratios), SVR | XGBoost:")
    for tc in CASE_SCOPE:
        d = df[df.case == tc]
        best = {}
        for m in ["SVR", "XGBoost"]:
            dm = d[d.model == m].sort_values("rmse")
            if not dm.empty:
                b = dm.iloc[0]; best[m] = (b.rmse, int(b.input_ratio_pct))
        s_svr = f"{best['SVR'][0]:.4f}@{best['SVR'][1]:3d}%" if "SVR" in best else "  -  "
        s_xgb = f"{best['XGBoost'][0]:.4f}@{best['XGBoost'][1]:3d}%" if "XGBoost" in best else "  -  "
        print(f"  Case {tc:2d}: SVR {s_svr}   XGB {s_xgb}")
    print(f"\nSaved: cache/results.csv ({len(df)} rows), cache/summary.csv")


if __name__ == "__main__":
    main()

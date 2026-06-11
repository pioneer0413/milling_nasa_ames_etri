#!/usr/bin/env python3
"""Broad ML sweep for Case 13/14 LOCV optimization (target RMSE < 0.08).

Sweeps feature-set x model x hyperparameter x prefix combinations under
Leave-One-Case-Out: for target case T, train on all other CASE_SCOPE cases
at a given prefix and predict T's runs. Reports per-case best configs.

Usage: python optimize_ml.py [--target 13] [--top 30]
Outputs: cache/ml_results.csv  (all combos)  +  console top-K summary.
"""
from __future__ import annotations
import argparse, itertools, warnings, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

HERE   = Path(__file__).resolve().parent
FEAT   = pd.read_pickle(HERE / "cache" / "feat_matrix.pkl")
CASE_SCOPE = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
META   = ["DOC", "feed", "material"]
SENSORS_ALL  = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
SENSORS_NOVIB = ["smcAC", "smcDC", "AE_table", "AE_spindle"]
SENSORS_CUR  = ["smcAC", "smcDC"]
STATS = ["mean", "rms", "std", "peak"]

def raw_cols(sensors):   return [f"{s}__{f}"        for s in sensors for f in STATS]
def delta_cols(sensors): return [f"{s}__delta_{f}"  for s in sensors for f in STATS]

# ── feature sets ──────────────────────────────────────────────────────────────
FEATURE_SETS = {
    "delta_meta_all":        delta_cols(SENSORS_ALL) + META,
    "raw_meta_all":          raw_cols(SENSORS_ALL) + META,
    "rawdelta_meta_all":     raw_cols(SENSORS_ALL) + delta_cols(SENSORS_ALL) + META,
    "delta_only_all":        delta_cols(SENSORS_ALL),
    "delta_meta_novib":      delta_cols(SENSORS_NOVIB) + META,
    "rawdelta_meta_novib":   raw_cols(SENSORS_NOVIB) + delta_cols(SENSORS_NOVIB) + META,
    "delta_meta_cur":        delta_cols(SENSORS_CUR) + META,
    "rawdelta_meta_cur":     raw_cols(SENSORS_CUR) + delta_cols(SENSORS_CUR) + META,
    "delta_cur_ae":          delta_cols(SENSORS_NOVIB) + META,   # alias clarity
    "delta_only_novib":      delta_cols(SENSORS_NOVIB),
    "delta_only_cur":        delta_cols(SENSORS_CUR),
}

# ── model grids ───────────────────────────────────────────────────────────────
def model_grid():
    grid = []
    for a in [0.01, 0.1, 1.0, 10.0, 100.0]:
        grid.append(("Ridge", {"alpha": a}))
    for a in [0.001, 0.01, 0.1]:
        grid.append(("Lasso", {"alpha": a, "max_iter": 5000}))
    for a, l1 in itertools.product([0.01, 0.1, 1.0], [0.2, 0.5, 0.8]):
        grid.append(("ElasticNet", {"alpha": a, "l1_ratio": l1, "max_iter": 5000}))
    grid.append(("BayesianRidge", {}))
    for e in [0.01, 0.1]:
        grid.append(("Huber", {"epsilon": 1.35, "alpha": e}))
    for C, g, eps in itertools.product([1, 10, 100, 300], ["scale", 0.01, 0.05], [0.01, 0.05, 0.1]):
        grid.append(("SVR", {"C": C, "gamma": g, "epsilon": eps, "kernel": "rbf"}))
    for a, g in itertools.product([0.01, 0.1, 1.0], [None, 0.01, 0.1]):
        grid.append(("KernelRidge", {"alpha": a, "kernel": "rbf", "gamma": g}))
    for k, w in itertools.product([3, 5, 7], ["uniform", "distance"]):
        grid.append(("KNN", {"n_neighbors": k, "weights": w}))
    for n, d, lr, ss in itertools.product([200, 400], [3, 4, 6], [0.03, 0.05, 0.1], [0.8, 1.0]):
        grid.append(("XGBoost", {"n_estimators": n, "max_depth": d, "learning_rate": lr,
                                 "subsample": ss, "random_state": 42, "n_jobs": 1}))
    for n, d in itertools.product([300, 600], [None, 6, 10]):
        grid.append(("RandomForest", {"n_estimators": n, "max_depth": d, "random_state": 42, "n_jobs": 1}))
        grid.append(("ExtraTrees",  {"n_estimators": n, "max_depth": d, "random_state": 42, "n_jobs": 1}))
    for n, d, lr in itertools.product([200, 400], [2, 3], [0.05, 0.1]):
        grid.append(("GradBoost", {"n_estimators": n, "max_depth": d, "learning_rate": lr,
                                   "random_state": 42}))
    return grid

def make_model(name, p):
    return {
        "Ridge": Ridge, "Lasso": Lasso, "ElasticNet": ElasticNet,
        "BayesianRidge": BayesianRidge, "Huber": HuberRegressor,
        "SVR": SVR, "KernelRidge": KernelRidge, "KNN": KNeighborsRegressor,
        "XGBoost": XGBRegressor, "RandomForest": RandomForestRegressor,
        "ExtraTrees": ExtraTreesRegressor, "GradBoost": GradientBoostingRegressor,
    }[name](**p)

# tree models don't need scaling; flag which ones do
NEED_SCALE = {"Ridge","Lasso","ElasticNet","BayesianRidge","Huber","SVR","KernelRidge","KNN"}


def evaluate(target, fs_name, feat_cols, pct, name, params):
    sub = FEAT[FEAT["signal_prefix_percent"] == float(pct)]
    tr  = sub[sub["case_id"] != target]
    te  = sub[sub["case_id"] == target]
    if tr.empty or te.empty:
        return None
    Xtr = tr[feat_cols].to_numpy(np.float64); ytr = tr["VB"].to_numpy(np.float64)
    Xte = te[feat_cols].to_numpy(np.float64); yte = te["VB"].to_numpy(np.float64)
    if name in NEED_SCALE:
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    try:
        m = make_model(name, params); m.fit(Xtr, ytr); pred = m.predict(Xte)
    except Exception:
        return None
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    return {"target_case": target, "feature_set": fs_name, "n_feat": len(feat_cols),
            "prefix_pct": pct, "model": name, "params": str(params),
            "rmse": rmse, "mae": float(mean_absolute_error(yte, pred)),
            "r2": float(r2_score(yte, pred)) if len(yte) > 1 else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[13, 14])
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    grid     = model_grid()
    prefixes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    jobs = [(target, fs_name, feat_cols, pct, name, params)
            for target in args.targets
            for fs_name, feat_cols in FEATURE_SETS.items()
            for pct in prefixes
            for name, params in grid]
    print(f"Total combos: {len(jobs)}  (parallel)", flush=True)
    raw = Parallel(n_jobs=-1, verbose=5, batch_size=64)(
        delayed(evaluate)(*j) for j in jobs)
    results = [r for r in raw if r]

    df = pd.DataFrame(results)
    df.to_csv(HERE/"cache"/"ml_results.csv", index=False)
    print(f"\nSaved {len(df)} results to cache/ml_results.csv\n")
    for target in args.targets:
        d = df[df["target_case"] == target].sort_values("rmse").head(args.top)
        print(f"\n===== Case {target}  TOP {args.top} (target RMSE < 0.08) =====")
        nbelow = (df[df["target_case"]==target]["rmse"] < 0.08).sum()
        print(f"  configs below 0.08: {nbelow}")
        for _, r in d.iterrows():
            flag = " <<<" if r["rmse"] < 0.08 else ""
            print(f"  RMSE={r['rmse']:.4f} r2={r['r2']:.3f} | {r['model']:13s} "
                  f"{r['feature_set']:20s} {int(r['prefix_pct']):3d}% | {r['params']}{flag}")


if __name__ == "__main__":
    main()

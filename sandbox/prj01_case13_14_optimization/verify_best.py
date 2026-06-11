#!/usr/bin/env python3
"""Reproduce & verify the two winning configs (Case 13 & 14, RMSE < 0.08).

Deterministic LOCV: train on all other CASE_SCOPE cases at the given prefix,
predict the target case. Prints exact config, RMSE, R2, MAE, and per-run
(true VB, predicted VB) so the result in context.md is fully reproducible.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
FEAT = pd.read_pickle(HERE / "cache" / "feat_matrix.pkl")
META  = ["DOC", "feed", "material"]
CUR   = ["smcAC", "smcDC"]
STATS = ["mean", "rms", "std", "peak"]
def dlt(s): return [f"{x}__delta_{f}" for x in s for f in STATS]

CONFIGS = [
    {"name": "Case 13", "target": 13, "prefix": 50,
     "features": dlt(CUR) + META, "feat_label": "current-delta(8)+meta(3)",
     "model": "SVR", "scale": True,
     "params": {"C": 200, "gamma": 0.005, "epsilon": 0.005, "kernel": "rbf"}},
    {"name": "Case 14", "target": 14, "prefix": 20,
     "features": dlt(CUR), "feat_label": "current-delta(8)",
     "model": "XGBoost", "scale": False,
     "params": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1,
                "subsample": 1.0, "random_state": 42, "n_jobs": 1}},
]

for cfg in CONFIGS:
    sub = FEAT[FEAT.signal_prefix_percent == float(cfg["prefix"])]
    tr  = sub[sub.case_id != cfg["target"]]
    te  = sub[sub.case_id == cfg["target"]].sort_values("run")
    cols = cfg["features"]
    Xtr, ytr = tr[cols].to_numpy(float), tr.VB.to_numpy(float)
    Xte, yte = te[cols].to_numpy(float), te.VB.to_numpy(float)
    if cfg["scale"]:
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    m = SVR(**cfg["params"]) if cfg["model"] == "SVR" else XGBRegressor(**cfg["params"])
    m.fit(Xtr, ytr); pred = m.predict(Xte)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    r2   = r2_score(yte, pred); mae = mean_absolute_error(yte, pred)
    print(f"\n===== {cfg['name']} =====")
    print(f"  model={cfg['model']}  features={cfg['feat_label']}  prefix={cfg['prefix']}%")
    print(f"  params={cfg['params']}")
    print(f"  train rows={len(tr)} (cases {sorted(tr.case_id.unique())})  test runs={len(te)}")
    print(f"  >>> RMSE={rmse:.4f}  R2={r2:.4f}  MAE={mae:.4f}  (target < 0.08)")
    print(f"  per-run [run: true -> pred]:")
    for run, t, p in zip(te.run.to_numpy(), yte, pred):
        print(f"      run {int(run):2d}: {t:.3f} -> {p:.3f}")

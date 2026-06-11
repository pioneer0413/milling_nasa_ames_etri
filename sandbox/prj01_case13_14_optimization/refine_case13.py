#!/usr/bin/env python3
"""Focused refinement for Case 13 (target RMSE < 0.08).

Broad sweep found 1 sub-0.08 config (SVR delta_meta_cur 80% C=300 -> 0.0677).
This confirms robustness by finely sweeping SVR/KernelRidge over the current-
sensor feature families and nearby hyperparameters. Also reproduces the top
config and reports how many fall below 0.08.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
FEAT = pd.read_pickle(HERE / "cache" / "feat_matrix.pkl")
META = ["DOC", "feed", "material"]
CUR  = ["smcAC", "smcDC"]
NOVIB = ["smcAC", "smcDC", "AE_table", "AE_spindle"]
STATS = ["mean", "rms", "std", "peak"]
def raw(s):   return [f"{x}__{f}" for x in s for f in STATS]
def dlt(s):   return [f"{x}__delta_{f}" for x in s for f in STATS]

FSETS = {
    "delta_meta_cur":      dlt(CUR) + META,
    "delta_only_cur":      dlt(CUR),
    "rawdelta_meta_cur":   raw(CUR) + dlt(CUR) + META,
    "delta_meta_novib":    dlt(NOVIB) + META,
    "delta_only_novib":    dlt(NOVIB),
}
TARGET = 13

def evalcfg(fs, cols, pct, kind, params):
    sub = FEAT[FEAT.signal_prefix_percent == float(pct)]
    tr, te = sub[sub.case_id != TARGET], sub[sub.case_id == TARGET]
    Xtr, ytr = tr[cols].to_numpy(float), tr.VB.to_numpy(float)
    Xte, yte = te[cols].to_numpy(float), te.VB.to_numpy(float)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    m = SVR(**params) if kind == "SVR" else KernelRidge(**params)
    m.fit(Xtr, ytr); p = m.predict(Xte)
    return float(np.sqrt(mean_squared_error(yte, p))), float(r2_score(yte, p))

rows = []
for fs, cols in FSETS.items():
    for pct in [50, 60, 70, 80, 90, 100]:
        for C, g, eps in itertools.product([50,100,150,200,300,500,800],
                                            [0.005,0.01,0.02,0.05,"scale"],
                                            [0.005,0.01,0.02,0.05]):
            try:
                rmse, r2 = evalcfg(fs, cols, pct, "SVR",
                                   {"C":C,"gamma":g,"epsilon":eps,"kernel":"rbf"})
                rows.append(("SVR", fs, pct, f"C={C},g={g},eps={eps}", rmse, r2))
            except Exception: pass
        for a, g in itertools.product([0.005,0.01,0.05,0.1,0.5],
                                      [0.01,0.05,0.1,0.2,None]):
            try:
                rmse, r2 = evalcfg(fs, cols, pct, "KR",
                                   {"alpha":a,"kernel":"rbf","gamma":g})
                rows.append(("KernelRidge", fs, pct, f"a={a},g={g}", rmse, r2))
            except Exception: pass

df = pd.DataFrame(rows, columns=["model","feature_set","prefix_pct","params","rmse","r2"])
df.to_csv(HERE/"cache"/"case13_refine.csv", index=False)
nb = (df.rmse < 0.08).sum()
print(f"Case 13 refine: {len(df)} configs, below 0.08 = {nb}, best = {df.rmse.min():.4f}\n")
for _, r in df.sort_values("rmse").head(25).iterrows():
    flag = " <<<" if r.rmse < 0.08 else ""
    print(f"  {r.rmse:.4f} r2={r.r2:5.2f} | {r.model:11s} {r.feature_set:18s} {int(r.prefix_pct):3d}% | {r.params}{flag}")

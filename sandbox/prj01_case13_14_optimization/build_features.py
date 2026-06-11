#!/usr/bin/env python3
"""Build & cache the feature matrix for Case 13/14 optimization.

Extracts Raw + Delta statistics (mean/rms/std/peak) for all 6 sensors at
prefix levels 10..100%, plus VB target and meta features. Caches to parquet
so the expensive signal parsing runs only once.

Output: sandbox/prj01_case13_14_optimization/cache/feat_matrix.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).resolve().parent / "cache"
OUT.mkdir(exist_ok=True)

CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS       = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
META_FEATURES = ["DOC", "feed", "material"]
EXCLUDED_RUNS = {(2, 1), (12, 1)}
THRESH        = 1e6
PREFIXES      = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def raw_col_names():
    return [f"{s}__{f}" for s in SENSORS for f in ["mean", "rms", "std", "peak"]]

def delta_col_names():
    return [f"{s}__delta_{f}" for s in SENSORS for f in ["mean", "rms", "std", "peak"]]

def parse_signal(v):
    a = np.fromstring(str(v).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

def extract_stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        a = np.zeros(1)
    return np.array([np.mean(a), np.sqrt(np.mean(a**2)), np.std(a), np.abs(a).max()],
                    dtype=np.float64)

def preprocess(df):
    df = df.copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"] == min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside"))
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)

def build_feature_matrix(sig_df, proc_df, prefix_percents):
    all_pcts   = sorted(set(prefix_percents))
    raw_cols   = raw_col_names()
    delta_cols = delta_col_names()
    raw_cache  = {}

    for row in proc_df.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        sr = sig_df[(sig_df["case"] == cid) & (sig_df["run"] == rid)]
        if sr.empty:
            continue
        arrays = {s: parse_signal(sr.iloc[0][s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        blen = min(len(a) for a in arrays.values())
        for pct in all_pcts:
            end = max(1, int(np.ceil(blen * pct / 100.0)))
            raw_cache[(cid, rid, pct)] = np.concatenate(
                [extract_stats(arrays[s][:end]) for s in SENSORS])

    first_run = {}
    for cid in proc_df["case"].unique():
        runs = sorted(r for (c, r, _) in raw_cache if c == cid)
        if runs:
            first_run[int(cid)] = runs[0]

    rows = []
    for row in proc_df.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        if (cid, rid) in EXCLUDED_RUNS:
            continue
        vb       = float(row.VB) if not pd.isna(row.VB) else 0.0
        ref_run  = first_run.get(cid, rid)
        for pct in all_pcts:
            rf  = raw_cache.get((cid, rid, pct))
            ref = raw_cache.get((cid, ref_run, pct))
            if rf is None or ref is None:
                continue
            delta = rf - ref
            rec   = {"case_id": cid, "run": rid, "signal_prefix_percent": float(pct),
                     "VB": vb,
                     "DOC":      float(getattr(row, "DOC", 0.0)),
                     "feed":     float(getattr(row, "feed", 0.0)),
                     "material": int(getattr(row, "material", 0))}
            for col, val in zip(raw_cols, rf):
                rec[col] = float(val)
            for col, val in zip(delta_cols, delta):
                rec[col] = float(val)
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(
        ["signal_prefix_percent", "case_id", "run"]).reset_index(drop=True)


def main():
    print("Loading signal + process data...", flush=True)
    sig  = pd.read_csv(ROOT/"datasets/processed/mill_signal_data.csv",
                       usecols=["case", "run"]+SENSORS)
    proc = pd.read_csv(ROOT/"datasets/processed/mill_process_info.csv")
    proc = proc[proc["case"].isin(CASE_SCOPE)].copy()
    sig  = sig[sig["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(proc)
    print(f"Building feature matrix for prefixes {PREFIXES}...", flush=True)
    feat = build_feature_matrix(sig, proc_clean, PREFIXES)
    # add VB_ratio per case (VB / case VB_max) for target-transform experiments
    feat["VB_max_case"] = feat.groupby("case_id")["VB"].transform("max")
    feat["VB_ratio"]    = feat["VB"] / feat["VB_max_case"].replace(0, np.nan)
    p = OUT / "feat_matrix.pkl"
    feat.to_pickle(p)
    print(f"Wrote {p}  shape={feat.shape}", flush=True)
    print(f"Cases: {sorted(feat['case_id'].unique())}", flush=True)
    print(feat.groupby("case_id")["run"].nunique())


if __name__ == "__main__":
    main()

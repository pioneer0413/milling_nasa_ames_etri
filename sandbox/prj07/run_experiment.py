#!/usr/bin/env python3
"""prj07 — Faithful 5x reproduction & aggregation of every H4_case_model_top5 row.

For each of the 80 leaderboard rows we reproduce the EXACT original experiment
by importing the original H4 script module (scripts/run_H4_*.py) and calling its
own build_feature_matrix + run_locv. Nothing is re-derived from the (sometimes
inaccurate) leaderboard labels — the script's real logic is the source of truth.

Repetition: each configuration is run with 5 seeds [0..4]. To make repetition
meaningful, the rep seed is propagated to every stochastic component
(RandomForest / XGBoost / MLP random_state, and the GRU torch/numpy seeds via
the module-level SEED). SVR and Ridge are deterministic -> std = 0 by design.
(Note: the original scripts hard-code random_state=42 for RF/XGB regardless of
SEED; here we tie it to the rep seed so the 5 reps are genuine repeats.)

Each distinct (task, model, input_length_pct) is computed once per seed via the
module's run_locv (which returns all cases), then filtered to the leaderboard's
target cases. Results are aggregated to mean±std over 5 seeds.

Outputs: cache/per_cell.csv (80 rows × seeds), cache/aggregated.csv, console.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
SCRIPTS = ROOT / "scripts"
SEEDS = [0, 1, 2, 3, 4]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LB2SCRIPT = {"XGBoost": "xgboost", "SVR": "svr", "Ridge": "ridge",
             "RandomForest": "random_forest", "MLP": "mlp", "feature_gru": "feature_gru"}

# ── module loading ────────────────────────────────────────────────────────────

def load_mod(stem):
    path = SCRIPTS / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod
    spec.loader.exec_module(mod)
    return mod

TASK_MODULE = {
    "S1_T1": "run_H4_S1_T1_experiment", "S1_T2": "run_H4_S1_T2_experiment",
    "S1_T3": "run_H4_S1_T3_experiment", "S1_T4": "run_H4_S1_T4_experiment",
    "S1_T5": "run_H4_S1_T5_experiment", "S1_T6": "run_H4_S1_T6_experiment",
    "S1_T7": "run_H4_S1_T7_experiment", "S1_T8": "run_H4_S1_T8_experiment",
    "S3_T1": "run_H4_S3_T1_experiment", "S3_T2": "run_H4_S3_T2_experiment",
    "S3_T4": "run_H4_S3_T4_experiment",
}

# ── seed patching ─────────────────────────────────────────────────────────────

def make_patched_make_model(seed):
    def mm(name):
        if name == "ridge":         return Ridge(alpha=1.0)
        if name == "random_forest": return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
        if name == "xgboost":       return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                                                        subsample=0.8, random_state=seed, n_jobs=1, verbosity=0)
        if name == "svr":           return SVR(kernel="rbf", C=10, epsilon=0.05)
        if name == "mlp":           return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                                                        random_state=seed, learning_rate_init=0.001)
        raise ValueError(name)
    return mm

def patch_seed(mod, seed):
    mod.SEED = seed
    mod.make_model = make_patched_make_model(seed)

# ── feature matrix builders (mirror each module's main()) ────────────────────

SIGNAL = ROOT / "datasets/processed/mill_signal_data.csv"
PROCESS = ROOT / "datasets/processed/mill_process_info.csv"
HMETA = ROOT / "datasets/metadata/heuristic_sequence_peng2026.csv"

def build_feat_s1(mod):
    sig = pd.read_csv(SIGNAL, usecols=["case", "run"] + mod.SENSORS)
    proc = pd.read_csv(PROCESS)
    proc = proc[proc["case"].isin(mod.CASE_SCOPE)].copy()
    sig = sig[sig["case"].isin(mod.CASE_SCOPE)].copy()
    proc_clean = mod.preprocess(proc)
    return mod.build_feature_matrix(sig, proc_clean, mod.PREFIX_PERCENTS, mod.SENSORS)

def build_feat_s3(mod, with_meta_arg):
    sig = pd.read_csv(SIGNAL, usecols=["case", "run"] + mod.SENSORS)
    proc = pd.read_csv(PROCESS)
    hmeta = pd.read_csv(HMETA, usecols=["case", "run", "idx_noload_end",
                                        "idx_start", "idx_end", "signal_length"])
    proc = proc[proc["case"].isin(mod.CASE_SCOPE)].copy()
    sig = sig[sig["case"].isin(mod.CASE_SCOPE)].copy()
    proc_clean = mod.preprocess_proc(proc)
    proc_merged = proc_clean.merge(hmeta, on=["case", "run"], how="inner")
    if with_meta_arg:
        return mod.build_feature_matrix(sig, proc_merged, hmeta)   # S3_T1/T2
    return mod.build_feature_matrix(sig, proc_merged)              # S3_T4

def build_feat(task, mod):
    if task.startswith("S1"):
        return build_feat_s1(mod)
    if task in ("S3_T1", "S3_T2"):
        return build_feat_s3(mod, with_meta_arg=True)
    if task == "S3_T4":
        return build_feat_s3(mod, with_meta_arg=False)
    raise ValueError(task)

# ── run_locv dispatch (per-task signature) -> list of all-case result dicts ──

def run_cell(task, mod, feat, mscript, pct):
    if task in ("S1_T1", "S1_T3"):
        cols = mod.feature_names_for_sensors(mod.SENSORS)
        return mod.run_locv(feat, cols, float(pct), mscript, [], DEVICE)
    if task in ("S1_T2", "S1_T5"):
        return mod.run_locv(feat, mod.raw_col_names(), mod.delta_col_names(),
                            float(pct), mscript, DEVICE)
    if task == "S1_T4":
        return mod.run_locv(feat, mod.delta_col_names(), float(pct), mscript, [], DEVICE)
    if task == "S1_T6":
        return mod.run_locv(feat, mod.raw_col_names(), float(pct), mscript, DEVICE)
    if task == "S1_T7":
        return mod.run_locv(feat, mod.delta_col_names(), float(pct), mscript, DEVICE)
    if task == "S1_T8":
        return mod.run_locv(feat, mod.all_feat_cols(), float(pct), mscript, DEVICE)
    if task in ("S3_T1", "S3_T2"):
        all_feat = mod.raw_col_names() + mod.delta_col_names() + mod.META_COLS
        return mod.run_locv(feat, all_feat, mod.delta_col_names(), mscript, DEVICE)
    if task == "S3_T4":
        all_feat = mod.raw_col_names() + mod.delta_col_names() + mod.META_COLS
        return mod.run_locv(feat, all_feat, mod.delta_col_names(), float(pct), mscript, DEVICE)
    raise ValueError(task)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    lb = pd.read_csv(ROOT / "leader_board" / "H4_case_model_top5.csv")
    lb["task_key"] = lb["scenario"] + "_" + lb["task"]
    print(f"Leaderboard rows: {len(lb)}  | device={DEVICE}  | seeds={SEEDS}\n", flush=True)

    # group needed cells: (task, model_lb, pct) -> set(cases)
    need = {}
    for r in lb.itertuples(index=False):
        key = (r.task_key, r.model, float(r.input_length_pct))
        need.setdefault(key, set()).add(int(r.target_case))

    # build each task's feature matrix once (seed-independent)
    tasks = sorted({k[0] for k in need})
    mods, feats = {}, {}
    for task in tasks:
        print(f"[load+features] {task} ...", flush=True)
        mods[task] = load_mod(TASK_MODULE[task])
        feats[task] = build_feat(task, mods[task])

    rows = []
    total = len(need) * len(SEEDS)
    done = 0
    for (task, model_lb, pct), cases in sorted(need.items()):
        mscript = LB2SCRIPT[model_lb]
        for seed in SEEDS:
            patch_seed(mods[task], seed)
            res = run_cell(task, mods[task], feats[task], mscript, pct)
            by_case = {int(d["target_case"]): d for d in res}
            for c in cases:
                d = by_case.get(c)
                if d is None:
                    continue
                rows.append({"task": task, "model": model_lb, "input_length_pct": pct,
                             "target_case": c, "seed": seed,
                             "rmse": d["rmse"], "r2": d.get("r2"), "mae": d.get("mae")})
            done += 1
            if done % 20 == 0 or done == total:
                print(f"  ...{done}/{total} cells (last: {task} {model_lb} {pct}% seed{seed})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "per_cell.csv", index=False)

    # merge leaderboard reference rmse
    lbref = lb[["task_key", "model", "input_length_pct", "target_case", "features", "rmse"]].rename(
        columns={"task_key": "task", "rmse": "lb_rmse"})
    lbref["input_length_pct"] = lbref["input_length_pct"].astype(float)

    agg = (df.groupby(["task", "model", "input_length_pct", "target_case"])
             .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                  r2_mean=("r2", "mean"), n_seeds=("seed", "nunique")).reset_index())
    agg = agg.merge(lbref, on=["task", "model", "input_length_pct", "target_case"], how="left")
    agg["abs_diff_vs_lb"] = (agg["rmse_mean"] - agg["lb_rmse"]).abs()
    agg = agg.sort_values(["task", "target_case", "model", "input_length_pct"])
    agg.to_csv(CACHE / "aggregated.csv", index=False)

    print("\n" + "=" * 96)
    print(f"AGGREGATED  ({len(agg)} unique leaderboard configs, 5 seeds each)")
    print("=" * 96)
    print(f"{'task':6} {'model':12} {'feat':16} {'pct':>5} {'case':>4} | "
          f"{'rmse_mean':>9} {'±std':>7} | {'lb_rmse':>8} {'Δ':>8}")
    print("-" * 96)
    for r in agg.itertuples(index=False):
        feat = (str(r.features)[:15]) if pd.notna(r.features) else "-"
        print(f"{r.task:6} {r.model:12} {feat:16} {int(r.input_length_pct):5d} {int(r.target_case):4d} | "
              f"{r.rmse_mean:9.4f} {r.rmse_std:7.4f} | {r.lb_rmse:8.4f} {r.rmse_mean-r.lb_rmse:+8.4f}")

    # aggregate stats
    print("\n" + "=" * 60)
    print("STABILITY SUMMARY")
    print("=" * 60)
    by_model = (df.groupby(["model", "task", "input_length_pct", "target_case"])["rmse"].std()
                  .reset_index(name="std"))
    ms = by_model.groupby("model")["std"].mean()
    print("Mean per-config seed-std by model (0 = deterministic):")
    for m, v in ms.sort_values().items():
        print(f"  {m:13s}: {v:.5f}")
    print(f"\nMean |reproduced_mean - leaderboard| over {len(agg)} configs: {agg['abs_diff_vs_lb'].mean():.4f}")
    print(f"Median |diff|: {agg['abs_diff_vs_lb'].median():.4f}")
    det = agg[agg["model"].isin(["SVR", "Ridge"])]
    print(f"Deterministic (SVR/Ridge) configs: {len(det)}, mean |diff| vs lb = {det['abs_diff_vs_lb'].mean():.5f}")
    print(f"\nSaved: cache/per_cell.csv ({len(df)} rows), cache/aggregated.csv ({len(agg)} configs)")


if __name__ == "__main__":
    main()

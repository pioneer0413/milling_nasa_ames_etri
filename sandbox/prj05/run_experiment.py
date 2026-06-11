#!/usr/bin/env python3
"""prj05 stacking — base = SVR/RandomForest/XGBoost, meta-learner = MLP.

Same validation protocol as prj04: input 100%, leakage-free NESTED Leave-One-
Case-Out, repeated over 5 seeds; report mean±std. Feature = Raw+Delta+Meta(51).

  Outer fold (predict held-out case T):
    1) Inner LOCV over train cases -> OOF base predictions
    2) Train MLP meta-learner on [OOF base preds] -> VB  (meta inputs scaled)
    3) Refit base learners on all train cases, predict T
    4) MLP meta-learner maps T's base preds -> final VB

Outputs: cache/per_case.csv, cache/summary.csv
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)

CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS       = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
META          = ["DOC", "feed", "material"]
STATS         = ["mean", "rms", "std", "peak"]
EXCLUDED_RUNS = {(2, 1), (12, 1)}
THRESH        = 1e6
PREFIX        = 100
SEEDS         = [0, 1, 2, 3, 4]
BASE_NAMES    = ["SVR", "RandomForest", "XGBoost"]      # meta-learner = MLP
SCALE_MODELS  = {"SVR"}

RAW_COLS   = [f"{s}__{f}" for s in SENSORS for f in STATS]
DELTA_COLS = [f"{s}__delta_{f}" for s in SENSORS for f in STATS]
FEAT_COLS  = RAW_COLS + DELTA_COLS + META          # 51

# ── feature extraction (100% prefix) ─────────────────────────────────────────

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
    pkl = CACHE / "feat_rawdeltameta_100.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    print("Building Raw+Delta+Meta @100% feature matrix...", flush=True)
    sig  = pd.read_csv(ROOT/"datasets/processed/mill_signal_data.csv",
                       usecols=["case", "run"]+SENSORS)
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
        arrays = {s: parse_signal(sr.iloc[0][s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        blen = min(len(a) for a in arrays.values())
        end  = max(1, int(np.ceil(blen * PREFIX / 100.0)))
        raw_cache[(cid, rid)] = np.concatenate([extract_stats(arrays[s][:end]) for s in SENSORS])
    first_run = {}
    for cid in proc["case"].unique():
        runs = sorted(r for (c, r) in raw_cache if c == cid)
        if runs:
            first_run[int(cid)] = runs[0]
    rows = []
    for row in proc.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        rf = raw_cache.get((cid, rid)); rr = raw_cache.get((cid, first_run.get(cid, rid)))
        if rf is None or rr is None:
            continue
        delta = rf - rr
        rec = {"case_id": cid, "run": rid, "VB": float(row.VB) if not pd.isna(row.VB) else 0.0,
               "DOC": float(getattr(row, "DOC", 0.0)), "feed": float(getattr(row, "feed", 0.0)),
               "material": int(getattr(row, "material", 0))}
        for c, v in zip(RAW_COLS, rf):    rec[c] = float(v)
        for c, v in zip(DELTA_COLS, delta): rec[c] = float(v)
        rows.append(rec)
    feat = pd.DataFrame(rows).sort_values(["case_id", "run"]).reset_index(drop=True)
    feat.to_pickle(pkl)
    print(f"  cached {pkl}  shape={feat.shape}", flush=True)
    return feat

# ── base learners & meta-learner ─────────────────────────────────────────────

def make_base(seed):
    return {
        "SVR":          SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.05),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1),
        "XGBoost":      XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                                     subsample=1.0, random_state=seed, n_jobs=1, verbosity=0),
    }

def make_meta(seed):
    # MLP meta-learner (input = 3 base preds); scaled inputs for stability
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(16, 8), activation="relu", max_iter=3000,
                     random_state=seed, early_stopping=False))

def fit_base(Xtr, ytr, seed):
    sc = StandardScaler().fit(Xtr)
    Xs = sc.transform(Xtr)
    fitted = {n: m.fit(Xs if n in SCALE_MODELS else Xtr, ytr)
              for n, m in make_base(seed).items()}
    return fitted, sc

def predict_base(fitted, sc, X):
    Xs = sc.transform(X)
    return np.column_stack([
        fitted[n].predict(Xs if n in SCALE_MODELS else X) for n in BASE_NAMES])

def stack_predict(seqs, target, seed):
    train_ids = [c for c in CASE_SCOPE if c != target and c in seqs]
    oof_P, oof_y = [], []
    for c in train_ids:
        inner = [c2 for c2 in train_ids if c2 != c]
        Xtr = np.concatenate([seqs[i]["X"] for i in inner])
        ytr = np.concatenate([seqs[i]["vb"] for i in inner])
        fit, sc = fit_base(Xtr, ytr, seed)
        oof_P.append(predict_base(fit, sc, seqs[c]["X"]))
        oof_y.append(seqs[c]["vb"])
    oof_P = np.vstack(oof_P); oof_y = np.concatenate(oof_y)
    meta = make_meta(seed).fit(oof_P, oof_y)
    Xtr = np.concatenate([seqs[i]["X"] for i in train_ids])
    ytr = np.concatenate([seqs[i]["vb"] for i in train_ids])
    fit_full, sc_full = fit_base(Xtr, ytr, seed)
    base_test = predict_base(fit_full, sc_full, seqs[target]["X"])
    return meta.predict(base_test), base_test.mean(axis=1), base_test

# ── main ──────────────────────────────────────────────────────────────────────

def case_sequences(feat):
    seqs = {}
    for cid in feat["case_id"].unique():
        g = feat[feat["case_id"] == cid].sort_values("run")
        seqs[int(cid)] = {"X": g[FEAT_COLS].to_numpy(np.float64),
                          "vb": g["VB"].to_numpy(np.float64)}
    return seqs

def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

def main():
    feat = build_feature_matrix()
    seqs = case_sequences(feat)
    print(f"features={len(FEAT_COLS)}  cases={len(seqs)}  base={BASE_NAMES}  "
          f"meta=MLP(16,8)  seeds={SEEDS}\n", flush=True)

    def one_fold(seed, tc):
        sp, ap, base_test = stack_predict(seqs, tc, seed)
        yt = seqs[tc]["vb"]
        rec = {"seed": seed, "case": tc, "n_runs": len(yt),
               "stack_rmse": rmse(yt, sp), "avg_rmse": rmse(yt, ap),
               "stack_r2": r2_score(yt, sp) if len(yt) > 1 else np.nan,
               "stack_mae": mean_absolute_error(yt, sp)}
        for i, n in enumerate(BASE_NAMES):
            rec[f"{n}_rmse"] = rmse(yt, base_test[:, i])
        return rec

    jobs = [(s, t) for s in SEEDS for t in CASE_SCOPE if t in seqs]
    print(f"Nested-stacking folds: {len(jobs)} (parallel)\n", flush=True)
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(one_fold)(s, t) for s, t in jobs)
    for seed in SEEDS:
        v = np.mean([r["stack_rmse"] for r in rows if r["seed"] == seed])
        print(f"  seed={seed}: 15-fold mean STACK RMSE = {v:.4f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(CACHE/"per_case.csv", index=False)
    rmse_cols = ["stack_rmse", "avg_rmse"] + [f"{n}_rmse" for n in BASE_NAMES]
    fold_means = df.groupby("seed")[rmse_cols].mean()
    overall = fold_means.mean(); overall_std = fold_means.std()

    print("\n" + "="*64)
    print("OVERALL 15-fold LOCV RMSE @100%  (mean ± std over 5 seeds)")
    print("="*64)
    for c in sorted(rmse_cols, key=lambda c: overall[c]):
        label = c.replace("_rmse", "")
        tag = "  <-- STACK (MLP meta)" if c == "stack_rmse" else (
              "  (simple average)" if c == "avg_rmse" else "")
        print(f"  {label:14s}: {overall[c]:.4f} ± {overall_std[c]:.4f}{tag}")

    print("\nPer-seed STACK 15-fold mean RMSE:")
    for s, v in fold_means["stack_rmse"].items():
        print(f"  seed {int(s)}: {v:.4f}")

    summ = (df.groupby("case").agg(mean=("stack_rmse","mean"), std=("stack_rmse","std"),
                                   r2=("stack_r2","mean")).reset_index())
    summ.to_csv(CACHE/"summary.csv", index=False)
    print("\nPer-case STACK RMSE (mean ± std over seeds):")
    for _, r in summ.sort_values("mean").iterrows():
        print(f"  Case {int(r['case']):2d}: {r['mean']:.4f} ± {r['std']:.4f}  (r2={r['r2']:.3f})")
    print(f"\nSaved: cache/per_case.csv, cache/summary.csv")


if __name__ == "__main__":
    main()

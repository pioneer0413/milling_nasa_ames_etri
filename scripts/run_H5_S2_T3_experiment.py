#!/usr/bin/env python3
"""H5_S2_T3: VB Normalization — Predict wear ratio (VB / VB_max) then rescale

Context spec: docs/agent_context/task_context/H5_S2_T3_context.md

Finding from H5_S1_T1: Cases 13/14 have same meta conditions as 7/8 but
VB_max 1.8-3.3x higher. Predicting absolute VB is hard because the model
learns the wrong VB_max prior from Cases 7/8.

Approach:
  Training:  VB_norm = VB / actual_VB_max[case]  → target in [0, 1]
  Test:      VB_max_pred = Ridge(DOC, feed, material) trained on fold
             VB_pred = model_output(VB_norm) × VB_max_pred
  Evaluate:  original VB scale (RMSE in mm)

Sensors/features: all 6 sensors, Delta(24)+Meta(3)=27 (same as H4_S1_T4)

Output:
  experiments/executions/H5/S2/T3/{timestamp}_H5_S2_T3_vb_normalized/
"""
from __future__ import annotations

import json, sys, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

CASE_SCOPE     = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS        = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
PREFIX_PERCENTS = list(range(10, 101, 10))
SEED           = 0
EXCLUDED_RUNS  = {(2, 1), (12, 1)}
META_FEATURES  = ["DOC", "feed", "material"]
TABULAR_MODELS = ["ridge", "random_forest", "xgboost", "svr", "mlp"]
PREFIX_LABEL   = "H5_S2_T3"
TOPIC          = "vb_normalized"
THRESH         = 1e6


def parse_signal(v):
    a = np.fromstring(str(v).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

def extract_features(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0: a = np.zeros(1)
    return np.array([np.mean(a), np.sqrt(np.mean(a**2)), np.std(a), np.abs(a).max()], dtype=np.float64)

def delta_col_names():
    return [f"{s}__delta_{f}" for s in SENSORS for f in ["mean","rms","std","peak"]]

def preprocess(df):
    df = df.copy().sort_values(["case","run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"]==min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside"))
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]),int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)

def build_feature_matrix(sig_df, proc_df, prefix_percents, sensors):
    delta_cols = delta_col_names()
    cache = {}
    for row in proc_df.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        sr = sig_df[(sig_df["case"]==cid)&(sig_df["run"]==rid)]
        if sr.empty: continue
        sr = sr.iloc[0]
        arrays = {s: parse_signal(sr[s]) for s in sensors}
        if any(np.abs(a).max() > THRESH for a in arrays.values()): continue
        blen = min(len(a) for a in arrays.values())
        for pct in prefix_percents:
            end = max(1, int(np.ceil(blen * pct / 100.0)))
            feats = np.concatenate([extract_features(arrays[s][:end]) for s in sensors])
            cache[(cid, rid, pct)] = feats

    first_run = {}
    for cid in proc_df["case"].unique():
        runs = sorted(r for (c, r, _) in cache if c == cid and (c, r) not in EXCLUDED_RUNS)
        if runs: first_run[int(cid)] = runs[0]

    rows = []
    for row in proc_df.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        if (cid, rid) in EXCLUDED_RUNS: continue
        ref_run = first_run.get(cid, rid)
        vb = float(row.VB) if not pd.isna(row.VB) else 0.0
        doc = float(getattr(row,"DOC",0.0))
        feed = float(getattr(row,"feed",0.0))
        material = int(getattr(row,"material",0))
        for pct in prefix_percents:
            rf  = cache.get((cid, rid, pct))
            ref = cache.get((cid, ref_run, pct))
            if rf is None or ref is None: continue
            delta = rf - ref
            rec = {"case_id":cid,"run":rid,"signal_prefix_percent":float(pct),
                   "VB":vb,"DOC":doc,"feed":feed,"material":material}
            for col, val in zip(delta_cols, delta):
                rec[col] = float(val)
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["signal_prefix_percent","case_id","run"]).reset_index(drop=True)


def predict_vb_max(train_df: pd.DataFrame, test_meta: np.ndarray) -> float:
    """Ridge regression on (DOC, feed, material) → VB_max, per LOCV fold."""
    vb_max_per_case = train_df.groupby("case_id")["VB"].max().reset_index()
    vb_max_per_case.columns = ["case_id","VB_max"]
    meta_df = train_df[["case_id"]+META_FEATURES].drop_duplicates("case_id")
    train_meta = meta_df.merge(vb_max_per_case, on="case_id")
    X = train_meta[META_FEATURES].to_numpy(dtype=np.float64)
    y = train_meta["VB_max"].to_numpy(dtype=np.float64)
    reg = Ridge(alpha=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(X, y)
    pred = float(reg.predict(test_meta.reshape(1,-1))[0])
    return max(pred, 0.1)   # floor at 0.1 mm


def make_model(name):
    if name == "ridge":         return Ridge(alpha=1.0)
    if name == "random_forest": return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "xgboost":       return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42, verbosity=0)
    if name == "svr":           return SVR(kernel="rbf", C=10, epsilon=0.05)
    if name == "mlp":           return MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=SEED, learning_rate_init=0.001)
    raise ValueError(name)


class DeltaMetaGRU(nn.Module):
    def __init__(self, input_dim, hidden_size=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = nn.utils.rnn.pad_packed_sequence(self.gru(packed)[0], batch_first=True)
        return self.head(out).squeeze(-1)


def build_sequences_normalized(feat_df, delta_cols, pct, vb_max_per_case):
    """Build sequences with VB normalized by case VB_max."""
    subset = feat_df[feat_df["signal_prefix_percent"]==float(pct)].copy()
    cases = {}
    for cid in subset["case_id"].unique():
        g = subset[subset["case_id"]==cid].sort_values("run").reset_index(drop=True)
        delta = g[delta_cols].to_numpy(dtype=np.float32)
        meta  = g[META_FEATURES].to_numpy(dtype=np.float32)
        seq   = np.concatenate([delta, meta], axis=1)
        vb    = g["VB"].to_numpy(dtype=np.float32)
        vb_max = float(vb_max_per_case.get(int(cid), max(vb.max(), 0.1)))
        cases[int(cid)] = {
            "seq":    seq,
            "vb":     vb,
            "vb_norm": vb / vb_max,
            "vb_max": vb_max,
            "n_runs": len(g),
            "meta":   g[META_FEATURES].iloc[0].to_numpy(dtype=np.float64),
        }
    return cases


def fit_predict_gru_normalized(train_cases, test_cases, input_dim, device,
                                vb_max_pred: float, epochs=200, lr=0.001):
    """GRU trained on VB_norm, predictions rescaled by vb_max_pred."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    if device.type == "cuda": torch.cuda.manual_seed_all(SEED)
    model = DeltaMetaGRU(input_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    seqs     = [v["seq"]     for v in train_cases.values()]
    vb_norms = [v["vb_norm"] for v in train_cases.values()]
    all_norm = np.concatenate(vb_norms)
    ym, ys   = float(all_norm.mean()), float(all_norm.std()) or 1.0

    tensors = [torch.tensor(s, dtype=torch.float32) for s in seqs]
    lengths = torch.tensor([len(s) for s in seqs])
    x_tr    = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0).to(device)
    tgts    = [torch.tensor((t-ym)/ys, dtype=torch.float32) for t in vb_norms]
    y_tr    = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=float("nan")).to(device)
    mask    = ~torch.isnan(y_tr)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_tr, lengths)
        ((pred[mask]-y_tr[mask])**2).mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    model.eval()
    preds = {}
    with torch.no_grad():
        for cid, info in test_cases.items():
            xt = torch.tensor(info["seq"], dtype=torch.float32).unsqueeze(0).to(device)
            lt = torch.tensor([info["n_runs"]])
            vb_norm_pred = (model(xt, lt).squeeze(0).cpu().numpy() * ys + ym)
            preds[cid] = vb_norm_pred * vb_max_pred   # denormalize
    return preds


def compute_metrics(y_true, y_pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2":   float(r2_score(y_true, y_pred)),
            "mae":  float(mean_absolute_error(y_true, y_pred))}


def run_locv(feat_df, delta_cols, pct, model_name, device):
    feature_cols = delta_cols + META_FEATURES
    subset = feat_df[feat_df["signal_prefix_percent"]==float(pct)].copy()
    results = []
    vb_max_records = []

    # Compute per-case VB_max from full dataset
    vb_max_per_case = feat_df.groupby("case_id")["VB"].max().to_dict()

    if model_name == "feature_gru":
        input_dim = len(delta_cols) + len(META_FEATURES)
        all_cases = build_sequences_normalized(feat_df, delta_cols, pct, vb_max_per_case)
        for tc in CASE_SCOPE:
            if tc not in all_cases: continue
            tr = {c: all_cases[c] for c in CASE_SCOPE if c!=tc and c in all_cases}
            # Build VB_max predictor for this fold
            tr_df_fold = subset[subset["case_id"]!=tc]
            test_meta  = all_cases[tc]["meta"]
            vb_max_pred = predict_vb_max(tr_df_fold, test_meta)
            actual_vb_max = all_cases[tc]["vb_max"]
            vb_max_records.append({"model":model_name,"target_case":tc,
                                    "predicted_vb_max":vb_max_pred,"actual_vb_max":actual_vb_max})
            ps = fit_predict_gru_normalized(tr, {tc: all_cases[tc]}, input_dim, device, vb_max_pred)
            y_true = all_cases[tc]["vb"]
            m = compute_metrics(y_true, ps[tc])
            results.append({"signal_prefix_percent":pct,"model":model_name,
                             "seed":SEED,"target_case":tc,"n_runs":len(y_true),
                             "predicted_vb_max":vb_max_pred,"actual_vb_max":actual_vb_max,**m})
        return results, vb_max_records

    for tc in CASE_SCOPE:
        tr_df = subset[subset["case_id"]!=tc]
        te_df = subset[subset["case_id"]==tc]
        if tr_df.empty or te_df.empty: continue

        # Predict VB_max for test case
        test_meta   = te_df[META_FEATURES].iloc[0].to_numpy(dtype=np.float64)
        vb_max_pred = predict_vb_max(tr_df, test_meta)
        actual_vb_max = float(vb_max_per_case.get(tc, 1.0))
        vb_max_records.append({"model":model_name,"target_case":tc,
                                "predicted_vb_max":vb_max_pred,"actual_vb_max":actual_vb_max})

        # Normalize training VB by per-case VB_max
        tr_df = tr_df.copy()
        for cid in tr_df["case_id"].unique():
            cmax = float(vb_max_per_case.get(cid, 1.0))
            tr_df.loc[tr_df["case_id"]==cid, "VB_norm"] = tr_df.loc[tr_df["case_id"]==cid,"VB"] / cmax

        Xtr = tr_df[feature_cols].to_numpy(dtype=np.float64)
        ytr = tr_df["VB_norm"].to_numpy(dtype=np.float64)   # normalized target
        Xte = te_df[feature_cols].to_numpy(dtype=np.float64)
        yte = te_df["VB"].to_numpy(dtype=np.float64)         # evaluate in original scale

        sc  = StandardScaler()
        Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        mdl = make_model(model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl.fit(Xtr, ytr)
            ypred_norm = mdl.predict(Xte)

        ypred = ypred_norm * vb_max_pred   # denormalize
        m = compute_metrics(yte, ypred)
        results.append({"signal_prefix_percent":pct,"model":model_name,
                         "seed":SEED,"target_case":tc,"n_runs":len(yte),
                         "predicted_vb_max":vb_max_pred,"actual_vb_max":actual_vb_max,**m})
    return results, vb_max_records


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT/"experiments"/"executions"/"H5"/"S2"/"T3"/f"{ts}_H5_S2_T3_{TOPIC}"
    for d in ["configs","metrics","analysis","logs"]: (out/d).mkdir(parents=True, exist_ok=True)
    lines = []
    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); lines.append(line)

    log("=== H5_S2_T3: VB Normalization (predict VB_ratio, rescale by VB_max) ===")

    sig  = pd.read_csv(ROOT/"datasets/processed/mill_signal_data.csv", usecols=["case","run"]+SENSORS)
    proc = pd.read_csv(ROOT/"datasets/processed/mill_process_info.csv")
    proc = proc[proc["case"].isin(CASE_SCOPE)].copy()
    sig  = sig[sig["case"].isin(CASE_SCOPE)].copy()

    proc_clean = preprocess(proc)
    feat_df    = build_feature_matrix(sig, proc_clean, PREFIX_PERCENTS, SENSORS)
    delta_cols = delta_col_names()
    log(f"Feature matrix: {len(feat_df)} rows, {len(delta_cols)+len(META_FEATURES)} features")

    (out/"configs"/f"{PREFIX_LABEL}_input_config.json").write_text(
        json.dumps({"experiment":{"id":f"{ts}_{PREFIX_LABEL}","context":"docs/agent_context/task_context/H5_S2_T3_context.md"},
                    "sensors":SENSORS,"features":f"Delta(24)+Meta(3)=27",
                    "normalization":"VB/VB_max per case (train=actual, test=predicted from meta)",
                    "case_scope":CASE_SCOPE,"prefix_percents":PREFIX_PERCENTS,"seed":SEED},indent=2),
        encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    all_results  = []
    all_vb_max_r = []
    for mn in TABULAR_MODELS + ["feature_gru"]:
        log(f"\nModel: {mn}")
        for pct in PREFIX_PERCENTS:
            try:
                res, vmr = run_locv(feat_df, delta_cols, float(pct), mn, device)
                all_results.extend(res)
                all_vb_max_r.extend(vmr)
            except Exception as e:
                log(f"  ERROR pct={pct} {e}")

    mdf = pd.DataFrame(all_results)
    mdf.to_csv(out/"metrics"/f"{PREFIX_LABEL}_shift_metrics.csv", index=False)

    agg = (mdf.groupby(["signal_prefix_percent","model"])
           .agg(mean_rmse=("rmse","mean"), mean_r2=("r2","mean"), mean_mae=("mae","mean"), n_folds=("target_case","count"))
           .reset_index().sort_values(["model","signal_prefix_percent"]))
    agg.to_csv(out/"analysis"/f"{PREFIX_LABEL}_locv_aggregated.csv", index=False)
    mdf[mdf["signal_prefix_percent"]==100.0].to_csv(out/"analysis"/f"{PREFIX_LABEL}_100pct_per_case.csv", index=False)

    # Hard case analysis
    hard_mdf  = mdf[mdf["target_case"].isin([13,14])]
    hard_best = hard_mdf.groupby(["model","target_case"])["rmse"].min().reset_index()
    hard_best.to_csv(out/"analysis"/f"{PREFIX_LABEL}_hard_case_best.csv", index=False)

    # VB_max prediction accuracy
    vmdf = pd.DataFrame(all_vb_max_r).drop_duplicates(["model","target_case"])
    vmdf["vb_max_error"] = vmdf["predicted_vb_max"] - vmdf["actual_vb_max"]
    vmdf.to_csv(out/"analysis"/f"{PREFIX_LABEL}_vb_max_prediction.csv", index=False)

    log("\n=== VB_max prediction accuracy (Case 13, 14) ===")
    hard_vm = vmdf[vmdf["target_case"].isin([13,14])][["model","target_case","predicted_vb_max","actual_vb_max","vb_max_error"]]
    log("\n" + hard_vm.drop_duplicates(["target_case"]).sort_values("target_case").to_string(index=False))

    log("\n=== SUMMARY @80% prefix ===")
    s80 = agg[agg["signal_prefix_percent"]==80.0][["model","mean_rmse","mean_r2"]].sort_values("mean_rmse")
    log("\n" + s80.to_string(index=False))
    log("\n=== Hard Cases (13, 14) Best RMSE ===")
    log("\n" + hard_best.pivot(index="model",columns="target_case",values="rmse").to_string())
    log("\n=== GRU RMSE by prefix % ===")
    gru_agg = agg[agg["model"]=="feature_gru"][["signal_prefix_percent","mean_rmse"]]
    log("\n" + gru_agg.to_string(index=False))

    (out/"logs"/f"{PREFIX_LABEL}_run_log.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nDone: {out}")

if __name__ == "__main__":
    main()

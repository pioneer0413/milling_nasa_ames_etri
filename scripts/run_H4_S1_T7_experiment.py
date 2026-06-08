#!/usr/bin/env python3
"""H4_S1_T7: Input Signal Length Effect — Delta Features Only (No Raw, No Meta)

Context spec: docs/agent_context/task_context/H4_S1_T7_context.md

Feature vectors:
  Tabular: Delta(24) = current_run_stats − first_run_stats per case
  GRU:     Delta(24) = delta-stats sequence per run step

Output:
  experiments/executions/H4/S1/T7/{timestamp}_H4_S1_T7_delta_only/
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

CASE_SCOPE      = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS         = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
PREFIX_PERCENTS = list(range(10, 101, 10))
SEED            = 0
EXCLUDED_RUNS   = {(2, 1), (12, 1)}
PREFIX_LABEL    = "H4_S1_T7"
TOPIC           = "delta_only"
THRESH          = 1e6
TABULAR_MODELS  = ["ridge", "random_forest", "xgboost", "svr", "mlp"]

def parse_signal(v):
    a = np.fromstring(str(v).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

def extract_features(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0: a = np.zeros(1)
    return np.array([np.mean(a), np.sqrt(np.mean(a**2)), np.std(a), np.abs(a).max()], dtype=np.float64)

def raw_col_names():
    return [f"{s}__{f}" for s in SENSORS for f in ["mean","rms","std","peak"]]

def delta_col_names():
    return [f"delta_{s}__{f}" for s in SENSORS for f in ["mean","rms","std","peak"]]

def preprocess(df):
    df = df.copy().sort_values(["case","run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"]==min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside"))
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]),int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)

def build_feature_matrix(signal_df, process_df, prefix_percents, sensors):
    raw_cols   = raw_col_names()
    delta_cols = delta_col_names()
    raw_cache = {}
    for row in process_df.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        sr = signal_df[(signal_df["case"]==cid)&(signal_df["run"]==rid)]
        if sr.empty: continue
        sr = sr.iloc[0]
        arrays = {s: parse_signal(sr[s]) for s in sensors}
        if any(np.abs(a).max() > THRESH for a in arrays.values()): continue
        blen = min(len(a) for a in arrays.values())
        for pct in prefix_percents:
            end = max(1, int(np.ceil(blen * pct / 100.0)))
            feats = np.concatenate([extract_features(arrays[s][:end]) for s in sensors])
            raw_cache[(cid, rid, pct)] = feats
    # compute first-available-run raw per case per pct
    first_run_cache = {}
    for pct in prefix_percents:
        for cid in process_df["case"].unique():
            cd = process_df[process_df["case"]==cid].sort_values("run")
            for rrow in cd.itertuples(index=False):
                rid = int(rrow.run)
                if (cid, rid) in EXCLUDED_RUNS: continue
                if (cid, rid, pct) in raw_cache:
                    first_run_cache[(cid, pct)] = raw_cache[(cid, rid, pct)]
                    break
    rows = []
    for row in process_df.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        if (cid, rid) in EXCLUDED_RUNS: continue
        vb = float(row.VB) if not pd.isna(row.VB) else 0.0
        for pct in prefix_percents:
            rf = raw_cache.get((cid, rid, pct))
            if rf is None: continue
            ref = first_run_cache.get((cid, pct))
            if ref is None: continue
            delta_feats = rf - ref
            rec = {"case_id": cid, "run": rid, "signal_prefix_percent": float(pct), "VB": vb}
            for col, val in zip(delta_cols, delta_feats):
                rec[col] = float(val)
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["signal_prefix_percent","case_id","run"]).reset_index(drop=True)

def make_model(name):
    if name == "ridge":         return Ridge(alpha=1.0)
    if name == "random_forest": return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "xgboost":       return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42, verbosity=0)
    if name == "svr":           return SVR(kernel="rbf", C=10, epsilon=0.05)
    if name == "mlp":           return MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=SEED, learning_rate_init=0.001)
    raise ValueError(name)

class DeltaGRU(nn.Module):
    def __init__(self, input_dim, hidden_size=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = nn.utils.rnn.pad_packed_sequence(self.gru(packed)[0], batch_first=True)
        return self.head(out).squeeze(-1)

def build_delta_sequences(feature_df, delta_cols, pct):
    subset = feature_df[feature_df["signal_prefix_percent"]==float(pct)].copy()
    cases = {}
    for cid in subset["case_id"].unique():
        cd = subset[subset["case_id"]==cid].sort_values("run").reset_index(drop=True)
        seq = cd[delta_cols].to_numpy(dtype=np.float32)
        vb  = cd["VB"].to_numpy(dtype=np.float32)
        cases[int(cid)] = {"seq": seq, "vb": vb, "n_runs": len(cd)}
    return cases

def fit_predict_gru(train_cases, test_cases, input_dim, device, epochs=200, lr=0.001):
    torch.manual_seed(SEED); np.random.seed(SEED)
    if device.type == "cuda": torch.cuda.manual_seed_all(SEED)
    model = DeltaGRU(input_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    seqs = [info["seq"] for info in train_cases.values()]
    vbs  = [info["vb"]  for info in train_cases.values()]
    all_vb = np.concatenate(vbs)
    ym, ys = float(all_vb.mean()), float(all_vb.std()) or 1.0

    tensors = [torch.tensor(s, dtype=torch.float32) for s in seqs]
    lengths = torch.tensor([len(s) for s in seqs])
    x_tr = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0).to(device)
    tgts  = [torch.tensor((t-ym)/ys, dtype=torch.float32) for t in vbs]
    y_tr  = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=float("nan")).to(device)
    mask  = ~torch.isnan(y_tr)

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
            preds[cid] = model(xt, lt).squeeze(0).cpu().numpy() * ys + ym
    return preds

def compute_metrics(y_true, y_pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2":   float(r2_score(y_true, y_pred)),
            "mae":  float(mean_absolute_error(y_true, y_pred))}

def run_locv(feature_df, feat_cols, pct, model_name, device):
    subset = feature_df[feature_df["signal_prefix_percent"]==float(pct)].copy()
    results = []

    if model_name == "feature_gru":
        case_data = build_delta_sequences(feature_df, feat_cols, pct)
        input_dim = len(feat_cols)
        for tc in CASE_SCOPE:
            if tc not in case_data: continue
            tr = {c: case_data[c] for c in CASE_SCOPE if c!=tc and c in case_data}
            ps = fit_predict_gru(tr, {tc: case_data[tc]}, input_dim, device)
            y_true = case_data[tc]["vb"]
            m = compute_metrics(y_true, ps[tc])
            results.append({"signal_prefix_percent": pct, "model": model_name,
                            "seed": SEED, "target_case": tc, "n_runs": len(y_true), **m})
        return results

    for tc in CASE_SCOPE:
        tr_df = subset[subset["case_id"]!=tc]
        te_df = subset[subset["case_id"]==tc]
        if tr_df.empty or te_df.empty: continue
        Xtr = tr_df[feat_cols].to_numpy(dtype=np.float64)
        ytr = tr_df["VB"].to_numpy(dtype=np.float64)
        Xte = te_df[feat_cols].to_numpy(dtype=np.float64)
        yte = te_df["VB"].to_numpy(dtype=np.float64)
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        mdl = make_model(model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl.fit(Xtr, ytr); ypred = mdl.predict(Xte)
        m = compute_metrics(yte, ypred)
        results.append({"signal_prefix_percent": pct, "model": model_name,
                        "seed": SEED, "target_case": tc, "n_runs": len(yte), **m})
    return results

def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT/"experiments"/"executions"/"H4"/"S1"/"T7"/f"{ts}_H4_S1_T7_{TOPIC}"
    for d in ["configs","metrics","analysis","logs"]: (out/d).mkdir(parents=True, exist_ok=True)
    lines = []
    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); lines.append(line)

    log("=== H4_S1_T7: Delta only ===")
    sig = pd.read_csv(ROOT/"datasets/processed/mill_signal_data.csv", usecols=["case","run"]+SENSORS)
    proc= pd.read_csv(ROOT/"datasets/processed/mill_process_info.csv")
    proc= proc[proc["case"].isin(CASE_SCOPE)].copy()
    sig = sig[sig["case"].isin(CASE_SCOPE)].copy()
    log(f"Loaded {len(proc)} process rows")
    proc_clean = preprocess(proc)
    log(f"After preprocess: {len(proc_clean)} rows")
    feat_df = build_feature_matrix(sig, proc_clean, PREFIX_PERCENTS, SENSORS)
    delta_cols = delta_col_names()
    log(f"Feature matrix: {len(feat_df)} rows, {len(delta_cols)} features")

    (out/"configs"/f"{PREFIX_LABEL}_input_config.json").write_text(
        json.dumps({"experiment": {"id": f"{ts}_{PREFIX_LABEL}",
                                   "context": "docs/agent_context/task_context/H4_S1_T7_context.md"},
                    "features": {"tabular": "Delta(24)", "gru": "Delta(24)"},
                    "case_scope": CASE_SCOPE, "prefix_percents": PREFIX_PERCENTS, "seed": SEED}, indent=2),
        encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    all_results = []
    for mn in TABULAR_MODELS + ["feature_gru"]:
        log(f"\nModel: {mn}")
        for pct in PREFIX_PERCENTS:
            try:
                all_results.extend(run_locv(feat_df, delta_cols, float(pct), mn, device))
            except Exception as e:
                log(f"  ERROR pct={pct} {e}")

    mdf = pd.DataFrame(all_results)
    mdf.to_csv(out/"metrics"/f"{PREFIX_LABEL}_shift_metrics.csv", index=False)
    agg = (mdf.groupby(["signal_prefix_percent","model"])
           .agg(mean_rmse=("rmse","mean"), mean_r2=("r2","mean"), mean_mae=("mae","mean"), n_folds=("target_case","count"))
           .reset_index().sort_values(["model","signal_prefix_percent"]))
    agg.to_csv(out/"analysis"/f"{PREFIX_LABEL}_locv_aggregated.csv", index=False)
    mdf[mdf["signal_prefix_percent"]==100.0].to_csv(out/"analysis"/f"{PREFIX_LABEL}_100pct_per_case.csv", index=False)

    log("\n=== SUMMARY at 100% ===")
    s100 = agg[agg["signal_prefix_percent"]==100.0][["model","mean_rmse","mean_r2","mean_mae"]]
    log("\n" + s100.sort_values("mean_rmse").to_string(index=False))
    log("\n=== RMSE pivot by prefix % ===")
    log("\n" + agg.pivot(index="signal_prefix_percent", columns="model", values="mean_rmse").to_string())
    (out/"logs"/f"{PREFIX_LABEL}_run_log.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nDone: {out}")

if __name__ == "__main__":
    main()

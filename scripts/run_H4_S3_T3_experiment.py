#!/usr/bin/env python3
"""H4_S3_T3: Segment-Based Feature Extraction — Steady only

Context spec: docs/agent_context/task_context/H4_S3_T3_context.md

Segment: signal[idx_start : idx_end]  (Steady region only)
Features: Raw(24) + Delta(24) + Meta(3) = 51 tabular; Delta+Meta(27) GRU

Output:
  experiments/executions/H4/S3/T3/{timestamp}_H4_S3_T3_steady/
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

CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS       = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
SEED          = 0
EXCLUDED_RUNS = {(2, 1), (12, 1)}
PREFIX_LABEL  = "H4_S3_T3"
TOPIC         = "steady"
THRESH        = 1e6
META_COLS     = ["DOC", "feed", "material"]
TABULAR_MODELS = ["ridge", "random_forest", "xgboost", "svr", "mlp"]

# segment slice: Steady only
def get_slice(row):
    return int(row.idx_start), int(row.idx_end)

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

def preprocess_proc(df):
    df = df.copy().sort_values(["case","run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"]==min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside"))
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]),int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)

def build_feature_matrix(signal_df, proc_df, meta_df):
    raw_cols   = raw_col_names()
    delta_cols = delta_col_names()
    raw_cache = {}
    for row in proc_df.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        sl_start, sl_end = get_slice(row)
        sr = signal_df[(signal_df["case"]==cid)&(signal_df["run"]==rid)]
        if sr.empty: continue
        sr = sr.iloc[0]
        arrays = {s: parse_signal(sr[s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()): continue
        sliced = {s: arrays[s][sl_start:sl_end] for s in SENSORS}
        feats = np.concatenate([extract_features(sliced[s]) for s in SENSORS])
        raw_cache[(cid, rid)] = feats
    first_cache = {}
    for cid in proc_df["case"].unique():
        cd = proc_df[proc_df["case"]==cid].sort_values("run")
        for r in cd.itertuples(index=False):
            rid = int(r.run)
            if (cid, rid) in EXCLUDED_RUNS: continue
            if (cid, rid) in raw_cache:
                first_cache[cid] = raw_cache[(cid, rid)]
                break
    rows = []
    for row in proc_df.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        rf = raw_cache.get((cid, rid))
        if rf is None: continue
        ref = first_cache.get(cid)
        if ref is None: continue
        vb = float(row.VB)
        delta_feats = rf - ref
        meta_vals = [float(row.DOC), float(row.feed), float(row.material)]
        rec = {"case_id": cid, "run": rid, "VB": vb}
        for col, val in zip(raw_cols, rf):            rec[col] = float(val)
        for col, val in zip(delta_cols, delta_feats): rec[col] = float(val)
        for col, val in zip(META_COLS, meta_vals):    rec[col] = float(val)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["case_id","run"]).reset_index(drop=True)

def make_model(name):
    if name == "ridge":         return Ridge(alpha=1.0)
    if name == "random_forest": return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "xgboost":       return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42, verbosity=0)
    if name == "svr":           return SVR(kernel="rbf", C=10, epsilon=0.05)
    if name == "mlp":           return MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=SEED, learning_rate_init=0.001)
    raise ValueError(name)

class DeltaMetaGRU(nn.Module):
    def __init__(self, input_dim, hidden_size=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = nn.utils.rnn.pad_packed_sequence(self.gru(packed)[0], batch_first=True)
        return self.head(out).squeeze(-1)

def build_gru_sequences(feature_df, delta_cols):
    cases = {}
    for cid in feature_df["case_id"].unique():
        cd = feature_df[feature_df["case_id"]==cid].sort_values("run").reset_index(drop=True)
        seq = cd[delta_cols + META_COLS].to_numpy(dtype=np.float32)
        vb  = cd["VB"].to_numpy(dtype=np.float32)
        cases[int(cid)] = {"seq": seq, "vb": vb, "n_runs": len(cd)}
    return cases

def fit_predict_gru(train_cases, test_cases, input_dim, device, epochs=200, lr=0.001):
    torch.manual_seed(SEED); np.random.seed(SEED)
    if device.type == "cuda": torch.cuda.manual_seed_all(SEED)
    model = DeltaMetaGRU(input_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    seqs   = [info["seq"] for info in train_cases.values()]
    vbs    = [info["vb"]  for info in train_cases.values()]
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

def run_locv(feature_df, all_cols, delta_cols, model_name, device):
    results = []
    active_cases = sorted(feature_df["case_id"].unique())
    if model_name == "feature_gru":
        case_data = build_gru_sequences(feature_df, delta_cols)
        input_dim = len(delta_cols) + len(META_COLS)
        for tc in active_cases:
            if tc not in case_data: continue
            tr = {c: case_data[c] for c in active_cases if c!=tc and c in case_data}
            ps = fit_predict_gru(tr, {tc: case_data[tc]}, input_dim, device)
            y_true = case_data[tc]["vb"]
            m = compute_metrics(y_true, ps[tc])
            results.append({"model": model_name, "seed": SEED, "target_case": tc,
                            "n_runs": len(y_true), **m})
        return results
    for tc in active_cases:
        tr_df = feature_df[feature_df["case_id"]!=tc]
        te_df = feature_df[feature_df["case_id"]==tc]
        if tr_df.empty or te_df.empty: continue
        Xtr = tr_df[all_cols].to_numpy(dtype=np.float64)
        ytr = tr_df["VB"].to_numpy(dtype=np.float64)
        Xte = te_df[all_cols].to_numpy(dtype=np.float64)
        yte = te_df["VB"].to_numpy(dtype=np.float64)
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        mdl = make_model(model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl.fit(Xtr, ytr); ypred = mdl.predict(Xte)
        m = compute_metrics(yte, ypred)
        results.append({"model": model_name, "seed": SEED, "target_case": tc,
                        "n_runs": len(yte), **m})
    return results

def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT/"experiments"/"executions"/"H4"/"S3"/"T3"/f"{ts}_H4_S3_T3_{TOPIC}"
    for d in ["configs","metrics","analysis","logs"]: (out/d).mkdir(parents=True, exist_ok=True)
    lines = []
    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); lines.append(line)

    log(f"=== H4_S3_T3: {TOPIC} ===")
    sig  = pd.read_csv(ROOT/"datasets/processed/mill_signal_data.csv", usecols=["case","run"]+SENSORS)
    proc = pd.read_csv(ROOT/"datasets/processed/mill_process_info.csv")
    hmeta= pd.read_csv(ROOT/"datasets/metadata/heuristic_sequence_peng2026.csv",
                       usecols=["case","run","idx_noload_end","idx_start","idx_end","signal_length"])
    proc = proc[proc["case"].isin(CASE_SCOPE)].copy()
    sig  = sig[sig["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess_proc(proc)
    proc_merged = proc_clean.merge(hmeta, on=["case","run"], how="inner")
    log(f"Usable runs (with metadata): {len(proc_merged)} / {len(proc_clean)}")
    feat_df = build_feature_matrix(sig, proc_merged, hmeta)
    raw_cols   = raw_col_names()
    delta_cols = delta_col_names()
    all_feat   = raw_cols + delta_cols + META_COLS
    log(f"Feature matrix: {len(feat_df)} rows, {len(all_feat)} features")
    log(f"Cases covered: {sorted(feat_df['case_id'].unique())}")

    (out/"configs"/f"{PREFIX_LABEL}_input_config.json").write_text(
        json.dumps({"experiment": {"id": f"{ts}_{PREFIX_LABEL}",
                                   "context": "docs/agent_context/task_context/H4_S3_T3_context.md"},
                    "segment": TOPIC, "features": "Raw(24)+Delta(24)+Meta(3)",
                    "case_scope": CASE_SCOPE, "seed": SEED}, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    all_results = []
    for mn in TABULAR_MODELS + ["feature_gru"]:
        log(f"\nModel: {mn}")
        try:
            all_results.extend(run_locv(feat_df, all_feat, delta_cols, mn, device))
        except Exception as e:
            log(f"  ERROR: {e}")

    mdf = pd.DataFrame(all_results)
    mdf.to_csv(out/"metrics"/f"{PREFIX_LABEL}_locv_per_case.csv", index=False)
    agg = (mdf.groupby("model")
           .agg(mean_rmse=("rmse","mean"), mean_r2=("r2","mean"), mean_mae=("mae","mean"), n_folds=("target_case","count"))
           .reset_index().sort_values("mean_rmse"))
    agg.to_csv(out/"analysis"/f"{PREFIX_LABEL}_locv_aggregated.csv", index=False)

    log("\n=== SUMMARY ===")
    log("\n" + agg.to_string(index=False))
    (out/"logs"/f"{PREFIX_LABEL}_run_log.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nDone: {out}")

if __name__ == "__main__":
    main()

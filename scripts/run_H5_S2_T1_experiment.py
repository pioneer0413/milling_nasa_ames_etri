#!/usr/bin/env python3
"""H5_S2_T1: Vibration Sensor Removal — Cases 13/14 Top-5 Evaluation

Context spec: docs/agent_context/task_context/H5_S2_T1_context.md

EDA finding (H5_S1_T1): vib_table/vib_spindle lose VB correlation in the
severe wear regime (Cases 13/14). This experiment removes those two sensors
and evaluates only the Case 13/14 Top-5 (model, features, input_pct) combinations
from H4_case_model_top5.csv, keeping all other settings identical to H4_S1_T4.

Sensors used: smcAC, smcDC, AE_table, AE_spindle  (vib_table / vib_spindle removed)
Feature dims:  Delta(16)+Meta(3)=19  /  Raw(16)  /  Raw+Meta(19)  /  Raw+Delta+Meta(35)

Output:
  experiments/executions/H5/S2/T1/{timestamp}_H5_S2_T1_no_vib_hard_cases/
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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
ALL_SENSORS   = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
SENSORS       = ["smcAC", "smcDC", "AE_table", "AE_spindle"]   # vib removed
META_FEATURES = ["DOC", "feed", "material"]
SEED          = 0
EXCLUDED_RUNS = {(2, 1), (12, 1)}
THRESH        = 1e6
PREFIX_LABEL  = "H5_S2_T1"

# Case 13/14 Top-5 from H4_case_model_top5.csv, with vib-removed feature dims
EXPERIMENTS = [
    # Case 13: all feature_gru, Delta(16)+Meta(3)=19, prefix
    {"target_case": 13, "model": "feature_gru", "features": "Delta+Meta", "input_pct": 70, "h4_rmse": 0.096915},
    {"target_case": 13, "model": "feature_gru", "features": "Delta+Meta", "input_pct": 60, "h4_rmse": 0.119013},
    {"target_case": 13, "model": "feature_gru", "features": "Delta+Meta", "input_pct": 50, "h4_rmse": 0.120829},
    {"target_case": 13, "model": "feature_gru", "features": "Delta+Meta", "input_pct": 40, "h4_rmse": 0.125645},
    {"target_case": 13, "model": "feature_gru", "features": "Delta+Meta", "input_pct": 30, "h4_rmse": 0.132593},
    # Case 14: mixed models, prefix
    {"target_case": 14, "model": "feature_gru", "features": "Delta+Meta",     "input_pct": 80,  "h4_rmse": 0.093156},
    {"target_case": 14, "model": "feature_gru", "features": "Delta+Meta",     "input_pct": 100, "h4_rmse": 0.099753},
    {"target_case": 14, "model": "xgboost",     "features": "Raw+Meta",       "input_pct": 30,  "h4_rmse": 0.099804},
    {"target_case": 14, "model": "xgboost",     "features": "Raw+Delta+Meta", "input_pct": 30,  "h4_rmse": 0.099804},
    {"target_case": 14, "model": "xgboost",     "features": "Raw",            "input_pct": 30,  "h4_rmse": 0.100608},
]


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_signal(v):
    a = np.fromstring(str(v).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

def extract_stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        a = np.zeros(1)
    return np.array([np.mean(a), np.sqrt(np.mean(a**2)), np.std(a), np.abs(a).max()], dtype=np.float64)

def raw_col_names():
    return [f"{s}__{f}" for s in SENSORS for f in ["mean", "rms", "std", "peak"]]

def delta_col_names():
    return [f"{s}__delta_{f}" for s in SENSORS for f in ["mean", "rms", "std", "peak"]]

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
    """Builds raw + delta + meta feature matrix using 4-sensor set (no vib)."""
    all_pcts   = sorted(set(prefix_percents))
    raw_cols   = raw_col_names()
    delta_cols = delta_col_names()

    raw_cache = {}
    for row in proc_df.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        sr = sig_df[(sig_df["case"] == cid) & (sig_df["run"] == rid)]
        if sr.empty:
            continue
        sr = sr.iloc[0]
        arrays = {s: parse_signal(sr[s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        blen = min(len(a) for a in arrays.values())
        for pct in all_pcts:
            end = max(1, int(np.ceil(blen * pct / 100.0)))
            feats = np.concatenate([extract_stats(arrays[s][:end]) for s in SENSORS])
            raw_cache[(cid, rid, pct)] = feats

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
        doc      = float(getattr(row, "DOC", 0.0))
        feed     = float(getattr(row, "feed", 0.0))
        material = int(getattr(row, "material", 0))
        ref_run  = first_run.get(cid, rid)
        for pct in all_pcts:
            rf  = raw_cache.get((cid, rid, pct))
            ref = raw_cache.get((cid, ref_run, pct))
            if rf is None or ref is None:
                continue
            delta = rf - ref
            rec = {"case_id": cid, "run": rid, "signal_prefix_percent": float(pct),
                   "VB": vb, "DOC": doc, "feed": feed, "material": material}
            for col, val in zip(raw_cols, rf):
                rec[col] = float(val)
            for col, val in zip(delta_cols, delta):
                rec[col] = float(val)
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["signal_prefix_percent", "case_id", "run"]).reset_index(drop=True)


def get_feature_cols(features_label):
    if features_label == "Delta+Meta":
        return delta_col_names() + META_FEATURES
    if features_label == "Raw+Meta":
        return raw_col_names() + META_FEATURES
    if features_label == "Raw+Delta+Meta":
        return raw_col_names() + delta_col_names() + META_FEATURES
    if features_label == "Raw":
        return raw_col_names()
    raise ValueError(f"Unknown feature set: {features_label}")


# ── GRU ──────────────────────────────────────────────────────────────────────

class DeltaMetaGRU(nn.Module):
    def __init__(self, input_dim, hidden_size=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.gru  = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = nn.utils.rnn.pad_packed_sequence(self.gru(packed)[0], batch_first=True)
        return self.head(out).squeeze(-1)


def build_sequences_gru(feat_df, pct, feature_cols):
    subset = feat_df[feat_df["signal_prefix_percent"] == float(pct)].copy()
    cases = {}
    for cid in subset["case_id"].unique():
        g   = subset[subset["case_id"] == cid].sort_values("run").reset_index(drop=True)
        seq = g[feature_cols].to_numpy(dtype=np.float32)
        cases[int(cid)] = {"seq": seq, "vb": g["VB"].to_numpy(dtype=np.float32), "n_runs": len(g)}
    return cases


def fit_predict_gru(train_cases, test_cases, input_dim, device, epochs=200, lr=0.001):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    model = DeltaMetaGRU(input_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    seqs   = [v["seq"] for v in train_cases.values()]
    vbs    = [v["vb"]  for v in train_cases.values()]
    all_vb = np.concatenate(vbs)
    ym, ys = float(all_vb.mean()), float(all_vb.std()) or 1.0

    tensors = [torch.tensor(s, dtype=torch.float32) for s in seqs]
    lengths = torch.tensor([len(s) for s in seqs])
    x_tr    = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0).to(device)
    tgts    = [torch.tensor((t - ym) / ys, dtype=torch.float32) for t in vbs]
    y_tr    = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=float("nan")).to(device)
    mask    = ~torch.isnan(y_tr)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_tr, lengths)
        ((pred[mask] - y_tr[mask]) ** 2).mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    model.eval()
    preds = {}
    with torch.no_grad():
        for cid, info in test_cases.items():
            xt = torch.tensor(info["seq"], dtype=torch.float32).unsqueeze(0).to(device)
            lt = torch.tensor([info["n_runs"]])
            preds[cid] = model(xt, lt).squeeze(0).cpu().numpy() * ys + ym
    return preds


# ── metrics & runner ─────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2":   float(r2_score(y_true, y_pred)),
            "mae":  float(mean_absolute_error(y_true, y_pred))}


def run_experiment(exp, feat_df, device, log):
    tc        = exp["target_case"]
    model_nm  = exp["model"]
    features  = exp["features"]
    pct       = exp["input_pct"]
    h4_rmse   = exp["h4_rmse"]
    feat_cols = get_feature_cols(features)

    n_sensors = len(SENSORS)
    log(f"  case={tc}  model={model_nm}  features={features}({len(feat_cols)}d)  "
        f"pct={pct}%  sensors={n_sensors}  h4_baseline={h4_rmse:.4f}")

    if model_nm == "feature_gru":
        all_cases = build_sequences_gru(feat_df, pct, feat_cols)
        if tc not in all_cases:
            log(f"  SKIP: case {tc} not in data at pct={pct}")
            return None
        train_cases = {c: all_cases[c] for c in CASE_SCOPE if c != tc and c in all_cases}
        preds   = fit_predict_gru(train_cases, {tc: all_cases[tc]}, len(feat_cols), device)
        y_true  = all_cases[tc]["vb"]
        metrics = compute_metrics(y_true, preds[tc])
    else:
        subset = feat_df[feat_df["signal_prefix_percent"] == float(pct)].copy()
        tr_df  = subset[subset["case_id"] != tc]
        te_df  = subset[subset["case_id"] == tc]
        if tr_df.empty or te_df.empty:
            log(f"  SKIP: empty split at pct={pct}")
            return None
        Xtr = tr_df[feat_cols].to_numpy(dtype=np.float64)
        ytr = tr_df["VB"].to_numpy(dtype=np.float64)
        Xte = te_df[feat_cols].to_numpy(dtype=np.float64)
        yte = te_df["VB"].to_numpy(dtype=np.float64)
        sc  = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        mdl = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                           subsample=0.8, random_state=42, verbosity=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl.fit(Xtr, ytr)
        y_pred  = mdl.predict(Xte)
        y_true  = yte
        metrics = compute_metrics(y_true, y_pred)

    delta = metrics["rmse"] - h4_rmse
    sign  = "+" if delta >= 0 else ""
    log(f"  → RMSE={metrics['rmse']:.4f}  Δ={sign}{delta:.4f}  R2={metrics['r2']:.4f}")
    return {**exp, **metrics, "rmse_delta": delta}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ts  = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT / "experiments" / "executions" / "H5" / "S2" / "T1" / f"{ts}_H5_S2_T1_no_vib_hard_cases"
    for d in ["configs", "metrics", "analysis", "logs"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    lines = []

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        lines.append(line)

    log("=== H5_S2_T1: Vibration Sensor Removal — Cases 13/14 Top-5 ===")
    log(f"Sensors used (vib removed): {SENSORS}")
    log(f"Feature dims: Delta({len(SENSORS)*4})+Meta(3)={len(SENSORS)*4+3}")
    log(f"Experiments: {len(EXPERIMENTS)} (Case 13: 5, Case 14: 5)")
    log("Baseline (H4 with vib): Case 13 best=0.0969, Case 14 best=0.0932\n")

    sig  = pd.read_csv(ROOT / "datasets/processed/mill_signal_data.csv",
                       usecols=["case", "run"] + SENSORS)
    proc = pd.read_csv(ROOT / "datasets/processed/mill_process_info.csv")
    proc = proc[proc["case"].isin(CASE_SCOPE)].copy()
    sig  = sig[sig["case"].isin(CASE_SCOPE)].copy()

    proc_clean  = preprocess(proc)
    needed_pcts = sorted(set(e["input_pct"] for e in EXPERIMENTS))
    log(f"Prefix levels: {needed_pcts}")

    feat_df = build_feature_matrix(sig, proc_clean, needed_pcts)
    log(f"Feature matrix: {len(feat_df)} rows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}\n")

    (out / "configs" / f"{PREFIX_LABEL}_input_config.json").write_text(
        json.dumps({"experiment": {"id": f"{ts}_{PREFIX_LABEL}",
                                   "context": "docs/agent_context/task_context/H5_S2_T1_context.md"},
                    "sensors": SENSORS,
                    "removed_sensors": ["vib_table", "vib_spindle"],
                    "target_cases": [13, 14],
                    "n_experiments": len(EXPERIMENTS),
                    "needed_pcts": needed_pcts,
                    "seed": SEED,
                    "gru": {"hidden_size": 256, "num_layers": 3, "dropout": 0.1, "epochs": 200}},
                   indent=2),
        encoding="utf-8")

    results = []
    for i, exp in enumerate(EXPERIMENTS):
        log(f"[{i+1}/{len(EXPERIMENTS)}] Running experiment...")
        r = run_experiment(exp, feat_df, device, log)
        if r is not None:
            results.append(r)
        log("")

    rdf = pd.DataFrame(results)
    rdf.to_csv(out / "metrics" / f"{PREFIX_LABEL}_results.csv", index=False)

    log("=" * 72)
    log("FINAL RESULTS (vib removed vs H4 with vib)")
    log("=" * 72)
    for _, row in rdf.iterrows():
        sign   = "+" if row["rmse_delta"] >= 0 else ""
        marker = " ✓" if row["rmse_delta"] < -0.002 else (" ✗" if row["rmse_delta"] > 0.002 else "  ")
        log(f"  Case {int(row['target_case']):2d} | {row['model']:11s} | {row['features']:14s} | "
            f"{int(row['input_pct']):3d}% | RMSE={row['rmse']:.4f} | "
            f"H4={row['h4_rmse']:.4f} | Δ={sign}{row['rmse_delta']:.4f}{marker}")

    log("\n=== Case 13 Summary ===")
    c13 = rdf[rdf["target_case"] == 13].sort_values("rmse")
    if not c13.empty:
        best = c13.iloc[0]
        sign = "+" if (best["rmse"] - 0.0969) >= 0 else ""
        log(f"  Best RMSE: {best['rmse']:.4f} @ {int(best['input_pct'])}%  "
            f"vs H4 0.0969  Δ={sign}{best['rmse']-0.0969:.4f}")
        improved = (c13["rmse_delta"] < 0).sum()
        log(f"  Improved configs: {improved}/{len(c13)}")

    log("\n=== Case 14 Summary ===")
    c14 = rdf[rdf["target_case"] == 14].sort_values("rmse")
    if not c14.empty:
        best = c14.iloc[0]
        sign = "+" if (best["rmse"] - 0.0932) >= 0 else ""
        log(f"  Best RMSE: {best['rmse']:.4f} @ {int(best['input_pct'])}% "
            f"{best['model']} {best['features']}  "
            f"vs H4 0.0932  Δ={sign}{best['rmse']-0.0932:.4f}")
        improved = (c14["rmse_delta"] < 0).sum()
        log(f"  Improved configs: {improved}/{len(c14)}")

    rdf.to_csv(out / "analysis" / f"{PREFIX_LABEL}_summary.csv", index=False)
    (out / "logs" / f"{PREFIX_LABEL}_run_log.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nDone: {out}")


if __name__ == "__main__":
    main()

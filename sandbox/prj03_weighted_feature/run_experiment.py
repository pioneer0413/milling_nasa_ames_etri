#!/usr/bin/env python3
"""prj03 weighted-feature GRU — LOCV @100% input, 5 seeds.

Model (per request):
  Feature-based GRU where the Raw+Delta+Meta feature vector (51-dim) is first
  scaled by a LEARNABLE per-feature weight, then encoded, then fed to a GRU
  that models the per-case run sequence and finally predicts VB.

    x (B,T,51)
      -> x * w           (w: learnable per-feature weight, init 1)   [feature gating]
      -> Linear(51->enc) + ReLU                                       [encoder]
      -> GRU(enc, hidden, layers)                                     [sequence model]
      -> Linear head -> VB per run

Evaluation: input length 100% only. Leave-One-Case-Out over all 15 cases
(train on other cases' run-sequences, predict held-out case sequence).
Repeat the full LOCV with 5 different seeds; report mean ± std.

Features are standardized per-fold (fit on training cases). VB target is
z-scored on the training batch. The learned feature weights |w| are saved
to interpret which features the model up/down-weights.

Outputs: cache/per_case.csv, cache/summary.csv, cache/feature_weights.csv
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

RAW_COLS   = [f"{s}__{f}" for s in SENSORS for f in STATS]
DELTA_COLS = [f"{s}__delta_{f}" for s in SENSORS for f in STATS]
FEAT_COLS  = RAW_COLS + DELTA_COLS + META          # 24 + 24 + 3 = 51

# GRU hyperparameters (H4_S1_T4 lineage)
ENC_DIM, HIDDEN, LAYERS, DROPOUT = 64, 256, 3, 0.1
EPOCHS, LR, WD = 200, 1e-3, 1e-4

# ── feature extraction (100% prefix, all sensors, raw+delta+meta) ─────────────

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
    cache_pkl = CACHE / "feat_rawdeltameta_100.pkl"
    if cache_pkl.exists():
        return pd.read_pickle(cache_pkl)
    print("Building Raw+Delta+Meta feature matrix @100%...", flush=True)
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
    feat.to_pickle(cache_pkl)
    print(f"  cached {cache_pkl}  shape={feat.shape}", flush=True)
    return feat

# ── model ─────────────────────────────────────────────────────────────────────

class WeightedFeatureGRU(nn.Module):
    def __init__(self, input_dim, enc_dim=ENC_DIM, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.feat_weight = nn.Parameter(torch.ones(input_dim))   # learnable per-feature weight
        self.encoder = nn.Sequential(nn.Linear(input_dim, enc_dim), nn.ReLU())
        self.gru = nn.GRU(enc_dim, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x, lengths):
        x = x * self.feat_weight                                 # feature gating
        x = self.encoder(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, _ = nn.utils.rnn.pad_packed_sequence(self.gru(packed)[0], batch_first=True)
        return self.head(out).squeeze(-1)

# ── data prep ─────────────────────────────────────────────────────────────────

def case_sequences(feat):
    seqs = {}
    for cid in feat["case_id"].unique():
        g = feat[feat["case_id"] == cid].sort_values("run")
        seqs[int(cid)] = {"X": g[FEAT_COLS].to_numpy(np.float64),
                          "vb": g["VB"].to_numpy(np.float64),
                          "runs": g["run"].to_numpy(int)}
    return seqs

def train_predict(seqs, target, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    train_ids = [c for c in CASE_SCOPE if c != target and c in seqs]
    # per-fold feature standardization (fit on training cases)
    Xtr_all = np.concatenate([seqs[c]["X"] for c in train_ids])
    sc = StandardScaler().fit(Xtr_all)
    vb_all = np.concatenate([seqs[c]["vb"] for c in train_ids])
    ym, ys = float(vb_all.mean()), float(vb_all.std()) or 1.0

    seqs_tr = [sc.transform(seqs[c]["X"]).astype(np.float32) for c in train_ids]
    vbs_tr  = [seqs[c]["vb"].astype(np.float32) for c in train_ids]
    tensors = [torch.tensor(s) for s in seqs_tr]
    lengths = torch.tensor([len(s) for s in seqs_tr])
    x_tr = nn.utils.rnn.pad_sequence(tensors, batch_first=True).to(device)
    tgts = [torch.tensor((t - ym) / ys) for t in vbs_tr]
    y_tr = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=float("nan")).to(device)
    mask = ~torch.isnan(y_tr)

    model = WeightedFeatureGRU(len(FEAT_COLS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred = model(x_tr, lengths)
        ((pred[mask] - y_tr[mask])**2).mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(sc.transform(seqs[target]["X"]).astype(np.float32)).unsqueeze(0).to(device)
        lt = torch.tensor([len(seqs[target]["X"])])
        pred = model(xt, lt).squeeze(0).cpu().numpy() * ys + ym
        w = model.feat_weight.detach().cpu().numpy()
    return pred, w

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat = build_feature_matrix()
    seqs = case_sequences(feat)
    print(f"Device={device}  features={len(FEAT_COLS)} (Raw24+Delta24+Meta3)  "
          f"cases={len(seqs)}  seeds={SEEDS}\n", flush=True)

    rows, wrows = [], []
    for seed in SEEDS:
        per_case = []
        for tc in CASE_SCOPE:
            if tc not in seqs:
                continue
            pred, w = train_predict(seqs, tc, seed, device)
            yt = seqs[tc]["vb"]
            rmse = float(np.sqrt(mean_squared_error(yt, pred)))
            r2   = float(r2_score(yt, pred)) if len(yt) > 1 else np.nan
            mae  = float(mean_absolute_error(yt, pred))
            rows.append({"seed": seed, "case": tc, "n_runs": len(yt),
                         "rmse": rmse, "mae": mae, "r2": r2})
            per_case.append(rmse)
            for fc, wv in zip(FEAT_COLS, w):
                wrows.append({"seed": seed, "target_case": tc, "feature": fc, "weight": float(wv)})
        m = float(np.mean(per_case))
        print(f"  seed={seed}: 15-fold mean RMSE = {m:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(CACHE/"per_case.csv", index=False)
    wdf = pd.DataFrame(wrows)
    wdf.to_csv(CACHE/"feature_weights.csv", index=False)

    # overall: mean RMSE across cases per seed, then mean+-std across seeds
    fold_mean = df.groupby("seed")["rmse"].mean()
    overall_mean, overall_std = float(fold_mean.mean()), float(fold_mean.std())
    summ = (df.groupby("case")
              .agg(mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"),
                   mean_r2=("r2", "mean")).reset_index())
    summ.to_csv(CACHE/"summary.csv", index=False)

    print("\n" + "="*60)
    print(f"OVERALL 15-fold LOCV RMSE @100%  =  {overall_mean:.4f} ± {overall_std:.4f}  "
          f"(mean±std over {len(SEEDS)} seeds)")
    print("="*60)
    print("\nPer-seed 15-fold mean RMSE:")
    for s, v in fold_mean.items():
        print(f"  seed {int(s)}: {v:.4f}")
    print("\nPer-case RMSE (mean ± std over seeds):")
    for _, r in summ.sort_values("mean_rmse").iterrows():
        print(f"  Case {int(r.case):2d}: {r.mean_rmse:.4f} ± {r.std_rmse:.4f}  (r2={r.mean_r2:.3f})")

    print("\nTop-12 features by mean |learned weight| (avg over seeds & folds):")
    wt = (wdf.assign(absw=wdf.weight.abs()).groupby("feature")["absw"].mean()
            .sort_values(ascending=False))
    for f, v in wt.head(12).items():
        print(f"  {f:28s} {v:.3f}")
    print("Bottom-6 (most down-weighted):")
    for f, v in wt.tail(6).items():
        print(f"  {f:28s} {v:.3f}")
    print(f"\nSaved: cache/per_case.csv, cache/summary.csv, cache/feature_weights.csv")


if __name__ == "__main__":
    main()

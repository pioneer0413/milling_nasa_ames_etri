#!/usr/bin/env python3
"""GRU sweep for Case 13/14 LOCV optimization (target RMSE < 0.08).

Feature-sequence GRU (per-case run sequence -> per-run VB). Sweeps feature
set, prefix, hidden size, layers, dropout, lr, epochs. LOCV: train on all
other cases, predict target case sequence.

Usage: python optimize_gru.py [--targets 13 14] [--quick]
"""
from __future__ import annotations
import argparse, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

HERE = Path(__file__).resolve().parent
FEAT = pd.read_pickle(HERE / "cache" / "feat_matrix.pkl")
CASE_SCOPE = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
META = ["DOC", "feed", "material"]
SENSORS_ALL  = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
SENSORS_NOVIB = ["smcAC", "smcDC", "AE_table", "AE_spindle"]
SENSORS_CUR  = ["smcAC", "smcDC"]
STATS = ["mean", "rms", "std", "peak"]
SEED = 0

def raw_cols(s):   return [f"{x}__{f}"       for x in s for f in STATS]
def delta_cols(s): return [f"{x}__delta_{f}" for x in s for f in STATS]

FEATURE_SETS = {
    "delta_meta_all":      delta_cols(SENSORS_ALL) + META,
    "delta_meta_novib":    delta_cols(SENSORS_NOVIB) + META,
    "delta_meta_cur":      delta_cols(SENSORS_CUR) + META,
    "rawdelta_meta_all":   raw_cols(SENSORS_ALL) + delta_cols(SENSORS_ALL) + META,
    "rawdelta_meta_novib": raw_cols(SENSORS_NOVIB) + delta_cols(SENSORS_NOVIB) + META,
}


class DeltaMetaGRU(nn.Module):
    def __init__(self, input_dim, hidden=256, layers=3, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, _ = nn.utils.rnn.pad_packed_sequence(self.gru(packed)[0], batch_first=True)
        return self.head(out).squeeze(-1)


def build_sequences(pct, feat_cols):
    sub = FEAT[FEAT["signal_prefix_percent"] == float(pct)]
    cases = {}
    for cid in sub["case_id"].unique():
        g = sub[sub["case_id"] == cid].sort_values("run")
        cases[int(cid)] = {"seq": g[feat_cols].to_numpy(np.float32),
                           "vb": g["VB"].to_numpy(np.float32), "n": len(g)}
    return cases


def fit_predict(train, test, input_dim, device, hidden, layers, dropout, lr, epochs):
    torch.manual_seed(SEED); np.random.seed(SEED)
    if device.type == "cuda": torch.cuda.manual_seed_all(SEED)
    model = DeltaMetaGRU(input_dim, hidden, layers, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    seqs = [v["seq"] for v in train.values()]; vbs = [v["vb"] for v in train.values()]
    allvb = np.concatenate(vbs); ym, ys = float(allvb.mean()), float(allvb.std()) or 1.0
    tensors = [torch.tensor(s) for s in seqs]
    lengths = torch.tensor([len(s) for s in seqs])
    x_tr = nn.utils.rnn.pad_sequence(tensors, batch_first=True).to(device)
    tgts = [torch.tensor((t - ym) / ys) for t in vbs]
    y_tr = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=float("nan")).to(device)
    mask = ~torch.isnan(y_tr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_tr, lengths)
        ((pred[mask] - y_tr[mask])**2).mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval()
    out = {}
    with torch.no_grad():
        for cid, info in test.items():
            xt = torch.tensor(info["seq"]).unsqueeze(0).to(device)
            lt = torch.tensor([info["n"]])
            out[cid] = model(xt, lt).squeeze(0).cpu().numpy() * ys + ym
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[13, 14])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prefixes = [40, 50, 60, 70, 80, 90, 100]
    if args.quick:
        hp = list(itertools.product([128, 256], [2, 3], [0.1], [1e-3], [200]))
    else:
        hp = list(itertools.product([64, 128, 256], [2, 3], [0.0, 0.1, 0.2],
                                    [5e-4, 1e-3, 2e-3], [200, 400]))
    fsets = list(FEATURE_SETS.items())
    total = len(args.targets)*len(fsets)*len(prefixes)*len(hp)
    print(f"Device={device}  total combos={total}", flush=True)

    results = []
    done = 0
    for target in args.targets:
        for fs_name, cols in fsets:
            for pct in prefixes:
                cases = build_sequences(pct, cols)
                if target not in cases:
                    continue
                tr = {c: cases[c] for c in CASE_SCOPE if c != target and c in cases}
                for hidden, layers, dropout, lr, epochs in hp:
                    pred = fit_predict(tr, {target: cases[target]}, len(cols), device,
                                       hidden, layers, dropout, lr, epochs)
                    yt = cases[target]["vb"]; pr = pred[target]
                    rmse = float(np.sqrt(mean_squared_error(yt, pr)))
                    results.append({"target_case": target, "feature_set": fs_name,
                        "n_feat": len(cols), "prefix_pct": pct, "hidden": hidden,
                        "layers": layers, "dropout": dropout, "lr": lr, "epochs": epochs,
                        "rmse": rmse, "mae": float(mean_absolute_error(yt, pr)),
                        "r2": float(r2_score(yt, pr)) if len(yt) > 1 else np.nan})
                    done += 1
                    if done % 50 == 0:
                        print(f"  ...{done}/{total}  last RMSE={rmse:.4f}", flush=True)

    df = pd.DataFrame(results)
    out = HERE/"cache"/("gru_results_quick.csv" if args.quick else "gru_results.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} -> {out}\n")
    for target in args.targets:
        d = df[df["target_case"] == target].sort_values("rmse").head(20)
        nb = (df[df["target_case"]==target]["rmse"] < 0.08).sum()
        print(f"\n===== Case {target} TOP 20  (below 0.08: {nb}) =====")
        for _, r in d.iterrows():
            flag = " <<<" if r["rmse"] < 0.08 else ""
            print(f"  RMSE={r['rmse']:.4f} r2={r['r2']:.3f} | {r['feature_set']:20s} "
                  f"{int(r['prefix_pct'])}% h{int(r['hidden'])} L{int(r['layers'])} "
                  f"d{r['dropout']} lr{r['lr']} e{int(r['epochs'])}{flag}")


if __name__ == "__main__":
    main()

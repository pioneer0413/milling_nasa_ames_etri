#!/usr/bin/env python3
"""H15_S1: Entry segment + surrounding ratio sweep.

Entry 구간 [noload_end : idx_start]을 항상 전부 사용하고,
ratio r%만큼 앞(no-load 구간)과 뒤(Steady 구간)를 추가 포함.

  window_start = max(0, noload_end - ceil(noload_len * r/100))
  window_end   = min(base_len, idx_start + ceil(steady_len * r/100))

r=0:   Entry only = [noload_end : idx_start]
r=100: [0 : idx_end] = full no-load + Entry + Steady

T1: Feature-GRU — AC+vT+vS (mask=13), 15-dim
    Baselines: ES-100%=0.087225, prefix-80%=0.081977
T2: XGBoost    — AC+vS   (mask= 9), 11-dim
    Baselines: ES-100%=0.105912, prefix-90%=0.103660

Sweep: r ∈ {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
Protocol: LOCV (15 cases), seeds=[0,1,2], observed_vb eval

Output: experiments/executions/H15/S1/{timestamp}_entry_surround_ratio_sweep/
"""
from __future__ import annotations

import json
import math
import sys
import time as time_mod
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS       = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
SENSOR_ABBR   = {"smcAC": "AC", "smcDC": "DC", "vib_table": "vT",
                 "vib_spindle": "vS", "AE_table": "aT", "AE_spindle": "aS"}
EXCLUDED_RUNS = {(2, 1), (12, 1)}
NON_OBSERVED_RUNS = {
    (1,2),(1,3),(1,5),(1,16),(2,6),(8,4),(11,7),(11,14),(11,17),
    (12,4),(12,10),(13,1),(13,2),(14,1),(14,5),(15,1),(16,1),(16,2),(16,4),
}
META_FEATURES = ["DOC", "feed", "material"]
SEEDS         = [0, 1, 2]
RATIOS        = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
THRESH        = 1e6
N_SENSORS     = len(SENSORS)

GRU_MASK = 13   # AC+vT+vS
XGB_MASK = 9    # AC+vS

BASELINES = {
    "gru_es100":    {"label": "H13_S1_T1 ES-100%",    "rmse": 0.087225},
    "gru_prefix80": {"label": "H11_S3_T1 prefix-80%", "rmse": 0.081977},
    "xgb_es100":    {"label": "H13_S1_T2 ES-100%",    "rmse": 0.105912},
    "xgb_prefix90": {"label": "H12_S1_T2 prefix-90%", "rmse": 0.103660},
}

SEG_CSV = ROOT / "datasets/nasa/cutting_segment_v2/seg_peng2026_steady5_exitfix_reverse_kurtosis.csv"

GRU_CFG = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0,
)
XGB_CFG = dict(
    learning_rate=0.03, n_estimators=600, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.9,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, n_jobs=4,
)


# ─── Signal utils ─────────────────────────────────────────────────────────────
def parse_signal(value: object) -> np.ndarray:
    return np.nan_to_num(
        np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


def extract_features(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros(1)
    return np.array(
        [np.mean(arr), np.sqrt(np.mean(arr**2)), np.std(arr), np.abs(arr).max()],
        dtype=np.float64,
    )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"] == min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside")
    )
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    excl = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~excl].reset_index(drop=True)


def mask_sensor_indices(mask: int) -> list[int]:
    return [i * 4 + j for i in range(N_SENSORS) if (mask >> i) & 1 for j in range(4)]


def mask_label(mask: int) -> str:
    return "+".join(SENSOR_ABBR[SENSORS[i]] for i in range(N_SENSORS) if (mask >> i) & 1)


# ─── Entry-surround cache ─────────────────────────────────────────────────────
def build_surround_cache(
    signal_df: pd.DataFrame,
    proc_clean: pd.DataFrame,
    seg_idx: dict[tuple[int, int], dict],
    ratios: list[int],
) -> dict[tuple[int, int, int], np.ndarray]:
    """cache[(case, run, r)] = 24-dim feature for Entry + r% surround."""
    cache: dict[tuple[int, int, int], np.ndarray] = {}

    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        sig_row = signal_df[
            (signal_df["case"] == case_id) & (signal_df["run"] == run_id)
        ]
        if sig_row.empty:
            continue
        sr = sig_row.iloc[0]
        arrays = {s: parse_signal(sr[s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        base_len = min(len(a) for a in arrays.values())

        seg = seg_idx.get((case_id, run_id))
        if seg is None:
            # Fallback: use prefix for all ratios
            for r in ratios:
                end = max(1, math.ceil(base_len * max(r, 50) / 100.0))
                feats = np.concatenate([extract_features(arrays[s][:end]) for s in SENSORS])
                cache[(case_id, run_id, r)] = feats
        else:
            nl_end   = min(int(seg["idx_noload_end"]), base_len)
            st_start = min(int(seg["idx_start"]),      base_len)
            st_end   = min(int(seg["idx_end"]),        base_len)
            noload_len = nl_end                        # length of no-load zone
            steady_len = max(0, st_end - st_start)    # length of Steady zone

            for r in ratios:
                # Extend left: r% of no-load zone before Entry
                left_ext  = math.ceil(noload_len * r / 100.0) if noload_len > 0 else 0
                # Extend right: r% of Steady zone after Entry
                right_ext = math.ceil(steady_len * r / 100.0) if steady_len > 0 else 0

                w_start = max(0,         nl_end - left_ext)
                w_end   = min(base_len,  st_start + right_ext)
                # Always guarantee Entry is included
                w_start = min(w_start, nl_end)
                w_end   = max(w_end,   st_start)

                feats = np.concatenate(
                    [extract_features(arrays[s][w_start:w_end]) for s in SENSORS]
                )
                cache[(case_id, run_id, r)] = feats

    return cache


def get_first_run(
    cache: dict[tuple[int, int, int], np.ndarray],
    proc_clean: pd.DataFrame,
    r: int,
) -> dict[int, int]:
    first: dict[int, int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(run for (c, run, rr) in cache if c == case_id and rr == r)
        if runs:
            first[int(case_id)] = runs[0]
    return first


# ─── GRU ──────────────────────────────────────────────────────────────────────
def build_gru_sequences(
    cache: dict, first_run: dict, proc_clean: pd.DataFrame, mask: int, r: int
) -> dict[int, dict]:
    si = mask_sensor_indices(mask)
    case_rows: dict[int, list[dict]] = {c: [] for c in CASE_SCOPE}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id, r)
        if key not in cache:
            continue
        ref_run = first_run.get(case_id, run_id)
        ref_vec = cache.get((case_id, ref_run, r), np.zeros(N_SENSORS * 4))
        delta   = (cache[key] - ref_vec)[si]
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        meta = np.array([float(getattr(row, mf, 0.0)) for mf in META_FEATURES], dtype=np.float64)
        feat = np.concatenate([delta, meta]).astype(np.float32)
        vb   = float(row.VB) if not pd.isna(row.VB) else 0.0
        case_rows[case_id].append({"feat": feat, "vb": vb, "run": run_id})

    return {
        c: {
            "seq":    np.stack([x["feat"] for x in sorted(rows, key=lambda x: x["run"])]),
            "vb":     np.array([x["vb"]   for x in sorted(rows, key=lambda x: x["run"])], dtype=np.float32),
            "runs":   np.array([x["run"]  for x in sorted(rows, key=lambda x: x["run"])], dtype=int),
            "n_runs": len(rows),
        }
        for c, rows in case_rows.items() if rows
    }


def fit_predict_gru(train_cases, test_cases, input_dim, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    class _GRU(nn.Module):
        def __init__(self):
            super().__init__()
            cfg = GRU_CFG
            self.gru = nn.GRU(input_dim, cfg["hidden_size"], cfg["num_layers"],
                               batch_first=True,
                               dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0)
            self.head = nn.Sequential(
                nn.Linear(cfg["hidden_size"], cfg["head_hidden"]), nn.ReLU(),
                nn.Linear(cfg["head_hidden"], 1),
            )
        def forward(self, x, lengths):
            pk = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(pk)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            return self.head(out).squeeze(-1)

    model     = _GRU().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=GRU_CFG["lr"], weight_decay=GRU_CFG["weight_decay"])
    epochs    = GRU_CFG["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_seqs = [info["seq"] for info in train_cases.values()]
    train_vbs  = [info["vb"]  for info in train_cases.values()]
    all_vb = np.concatenate(train_vbs)
    y_mean = float(all_vb.mean())
    y_std  = max(float(all_vb.std()), 1e-8)

    tensors = [torch.tensor(s, dtype=torch.float32) for s in train_seqs]
    lengths = torch.tensor([len(s) for s in train_seqs])
    x_train = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0).to(device)
    tgt     = [torch.tensor((t - y_mean) / y_std, dtype=torch.float32) for t in train_vbs]
    y_train = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=float("nan")).to(device)
    mask_tr = ~torch.isnan(y_train)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train, lengths)
        loss = ((pred[mask_tr] - y_train[mask_tr]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRU_CFG["grad_clip"])
        optimizer.step()
        scheduler.step()

    model.eval()
    preds = {}
    with torch.no_grad():
        for cid, info in test_cases.items():
            x = torch.tensor(info["seq"], dtype=torch.float32).unsqueeze(0).to(device)
            l = torch.tensor([info["n_runs"]])
            preds[cid] = model(x, l).squeeze(0).cpu().numpy() * y_std + y_mean
    return preds


def run_gru_locv(all_cases, input_dim, device, seed) -> float:
    rmses: list[float] = []
    for tc in CASE_SCOPE:
        if tc not in all_cases:
            continue
        train = {c: all_cases[c] for c in CASE_SCOPE if c != tc and c in all_cases}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = fit_predict_gru(train, {tc: all_cases[tc]}, input_dim, device, seed)
        runs = all_cases[tc]["runs"]
        obs  = np.array([(tc, int(run)) not in NON_OBSERVED_RUNS for run in runs])
        y_pred = preds[tc][obs]
        y_true = all_cases[tc]["vb"][obs]
        if len(y_true) == 0:
            continue
        rmses.append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
    return float(np.mean(rmses)) if rmses else float("nan")


# ─── XGB ──────────────────────────────────────────────────────────────────────
def build_xgb_df(cache, first_run, proc_clean, mask, r) -> pd.DataFrame:
    si   = mask_sensor_indices(mask)
    rows = []
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id, r)
        if key not in cache:
            continue
        ref_run = first_run.get(case_id, run_id)
        ref_vec = cache.get((case_id, ref_run, r), np.zeros(N_SENSORS * 4))
        delta   = (cache[key] - ref_vec)[si]
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        vb  = float(row.VB) if not pd.isna(row.VB) else 0.0
        rec = {"case_id": case_id, "run": run_id, "VB": vb}
        for mf in META_FEATURES:
            rec[mf] = float(getattr(row, mf, 0.0))
        for k, v in enumerate(delta):
            rec[f"f{k}"] = float(v)
        rows.append(rec)
    return pd.DataFrame(rows)


def run_xgb_locv(feat_df, feature_cols, seed) -> float:
    rmses: list[float] = []
    for tc in CASE_SCOPE:
        test_df  = feat_df[feat_df["case_id"] == tc]
        train_df = feat_df[feat_df["case_id"] != tc]
        if test_df.empty or train_df.empty:
            continue
        model = XGBRegressor(**XGB_CFG, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_df[feature_cols], train_df["VB"])
            y_pred = model.predict(test_df[feature_cols])
        obs    = test_df.apply(
            lambda r: (int(r["case_id"]), int(r["run"])) not in NON_OBSERVED_RUNS, axis=1
        ).to_numpy()
        y_true = test_df["VB"].to_numpy(float)[obs]
        if len(y_true) == 0:
            continue
        rmses.append(float(np.sqrt(mean_squared_error(y_true, y_pred[obs]))))
    return float(np.mean(rmses)) if rmses else float("nan")


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_results(gru_results, xgb_results, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    configs = [
        (axes[0], gru_results, "Feature-GRU (AC+vT+vS)", "steelblue",
         BASELINES["gru_es100"], BASELINES["gru_prefix80"]),
        (axes[1], xgb_results, "XGBoost (AC+vS)", "darkorange",
         BASELINES["xgb_es100"], BASELINES["xgb_prefix90"]),
    ]
    for ax, results, label, color, bl_es, bl_pfx in configs:
        rs    = [r["r"]         for r in results]
        means = [r["mean_rmse"] for r in results]
        stds  = [r["std_rmse"]  for r in results]
        ax.errorbar(rs, means, yerr=stds, marker="o", capsize=4,
                    color=color, label=label, linewidth=1.8)
        ax.axhline(bl_es["rmse"], color="gray", linestyle="--", linewidth=1.2,
                   label=f"ES-100% baseline: {bl_es['rmse']:.4f}")
        ax.axhline(bl_pfx["rmse"], color="black", linestyle=":", linewidth=1.2,
                   label=f"Prefix best baseline: {bl_pfx['rmse']:.4f}")
        best = min(results, key=lambda r: r["mean_rmse"])
        ax.axvline(best["r"], color="crimson", linestyle=":", linewidth=1.2,
                   label=f"Best r={best['r']}%  RMSE={best['mean_rmse']:.4f}")
        ax.set_xlabel("Surrounding ratio r (%)")
        ax.set_ylabel("Observed-VB RMSE (3-seed mean)")
        ax.set_title(label)
        ax.set_xticks(rs)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("H15_S1: Entry + Surrounding Ratio Sweep\n"
                 "(r=0: Entry only; r=100: no-load+Entry+Steady)", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(out_path.with_suffix(".png")), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".svg")), bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "H15" / "S1" / \
              f"{ts}_entry_surround_ratio_sweep"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H15_S1: Entry + Surrounding Ratio Sweep ===")
    log("Window: [noload_end - r%*noload_len : idx_start + r%*steady_len]")
    log(f"r=0: Entry only | r=100: full [0:idx_end]")
    log(f"T1: GRU  mask={GRU_MASK} ({mask_label(GRU_MASK)})  "
        f"baselines: ES-100%={BASELINES['gru_es100']['rmse']}, "
        f"prefix-80%={BASELINES['gru_prefix80']['rmse']}")
    log(f"T2: XGB  mask={XGB_MASK} ({mask_label(XGB_MASK)})  "
        f"baselines: ES-100%={BASELINES['xgb_es100']['rmse']}, "
        f"prefix-90%={BASELINES['xgb_prefix90']['rmse']}")
    log(f"Ratios={RATIOS}, Seeds={SEEDS}, LOCV={len(CASE_SCOPE)} cases")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    seg_df = pd.read_csv(SEG_CSV)
    seg_df = seg_df[seg_df["case"].isin(CASE_SCOPE) & (seg_df["status"] == "labeled")]
    seg_idx: dict[tuple[int, int], dict] = {
        (int(r.case), int(r.run)): {
            "idx_noload_end": int(r.idx_noload_end),
            "idx_start":      int(r.idx_start),
            "idx_end":        int(r.idx_end),
        }
        for r in seg_df.itertuples(index=False)
    }
    log(f"Segment-labeled runs: {len(seg_idx)}/{len(proc_clean)}")

    log(f"Building Entry-surround cache (ratios={RATIOS})...")
    cache = build_surround_cache(signal_df, proc_clean, seg_idx, RATIOS)
    log(f"Cache size: {len(cache)} entries")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ── T1: GRU ───────────────────────────────────────────────────────────────
    gru_n = bin(GRU_MASK).count("1")
    gru_input_dim = gru_n * 4 + len(META_FEATURES)
    log(f"\n--- T1: Feature-GRU  input_dim={gru_input_dim} ---")
    gru_results: list[dict] = []
    t0 = time_mod.time()

    for r in RATIOS:
        first_run = get_first_run(cache, proc_clean, r)
        all_cases = build_gru_sequences(cache, first_run, proc_clean, GRU_MASK, r)
        seed_rmses = [run_gru_locv(all_cases, gru_input_dim, device, s) for s in SEEDS]
        mean_rmse  = float(np.mean(seed_rmses))
        std_rmse   = float(np.std(seed_rmses))
        gru_results.append({"r": r, "mean_rmse": mean_rmse, "std_rmse": std_rmse,
                            "seed_rmses": seed_rmses})
        log(f"  GRU r={r:3d}%  RMSE={mean_rmse:.6f}  std={std_rmse:.6f}"
            f"  Δ(ES100)={mean_rmse-BASELINES['gru_es100']['rmse']:+.6f}"
            f"  Δ(pfx80)={mean_rmse-BASELINES['gru_prefix80']['rmse']:+.6f}"
            f"  [{time_mod.time()-t0:.0f}s]")

    gru_best = min(gru_results, key=lambda x: x["mean_rmse"])
    log(f"  GRU best: r={gru_best['r']}%  RMSE={gru_best['mean_rmse']:.6f}")

    # ── T2: XGB ───────────────────────────────────────────────────────────────
    n_delta      = bin(XGB_MASK).count("1") * 4
    feature_cols = [f"f{k}" for k in range(n_delta)] + META_FEATURES
    log(f"\n--- T2: XGBoost  n_feat={len(feature_cols)} ---")
    xgb_results: list[dict] = []
    t0 = time_mod.time()

    for r in RATIOS:
        first_run = get_first_run(cache, proc_clean, r)
        feat_df   = build_xgb_df(cache, first_run, proc_clean, XGB_MASK, r)
        feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat_df.fillna(0.0, inplace=True)
        seed_rmses = [run_xgb_locv(feat_df, feature_cols, s) for s in SEEDS]
        mean_rmse  = float(np.mean(seed_rmses))
        std_rmse   = float(np.std(seed_rmses))
        xgb_results.append({"r": r, "mean_rmse": mean_rmse, "std_rmse": std_rmse,
                            "seed_rmses": seed_rmses})
        log(f"  XGB r={r:3d}%  RMSE={mean_rmse:.6f}  std={std_rmse:.6f}"
            f"  Δ(ES100)={mean_rmse-BASELINES['xgb_es100']['rmse']:+.6f}"
            f"  Δ(pfx90)={mean_rmse-BASELINES['xgb_prefix90']['rmse']:+.6f}"
            f"  [{time_mod.time()-t0:.0f}s]")

    xgb_best = min(xgb_results, key=lambda x: x["mean_rmse"])
    log(f"  XGB best: r={xgb_best['r']}%  RMSE={xgb_best['mean_rmse']:.6f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    all_rows = []
    for res in gru_results:
        all_rows.append({"model": "feature_gru", "mask": GRU_MASK,
                         "sensor_subset": mask_label(GRU_MASK),
                         "strategy": "entry_surround", "surround_ratio": res["r"],
                         "mean_rmse": res["mean_rmse"], "std_rmse": res["std_rmse"],
                         **{f"seed{i}_rmse": res["seed_rmses"][i] for i in range(len(SEEDS))}})
    for res in xgb_results:
        all_rows.append({"model": "xgboost", "mask": XGB_MASK,
                         "sensor_subset": mask_label(XGB_MASK),
                         "strategy": "entry_surround", "surround_ratio": res["r"],
                         "mean_rmse": res["mean_rmse"], "std_rmse": res["std_rmse"],
                         **{f"seed{i}_rmse": res["seed_rmses"][i] for i in range(len(SEEDS))}})
    pd.DataFrame(all_rows).to_csv(
        out_dir / "metrics" / "entry_surround_results.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(gru_results, xgb_results,
                 out_dir / "figures" / "entry_surround_curve")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    log(f"{'r':>5}  {'GRU_RMSE':>10}  {'XGB_RMSE':>10}")
    log("-" * 32)
    for g, x in zip(gru_results, xgb_results):
        gm = "★" if g["r"] == gru_best["r"] else " "
        xm = "★" if x["r"] == xgb_best["r"] else " "
        log(f"{g['r']:>4}%  {gm}{g['mean_rmse']:>9.6f}  {xm}{x['mean_rmse']:>9.6f}")

    log(f"\nGRU  best: r={gru_best['r']}%  RMSE={gru_best['mean_rmse']:.6f}"
        f"  (ES-100%={BASELINES['gru_es100']['rmse']:.6f}"
        f", prefix-80%={BASELINES['gru_prefix80']['rmse']:.6f})")
    log(f"XGB  best: r={xgb_best['r']}%  RMSE={xgb_best['mean_rmse']:.6f}"
        f"  (ES-100%={BASELINES['xgb_es100']['rmse']:.6f}"
        f", prefix-90%={BASELINES['xgb_prefix90']['rmse']:.6f})")

    summary = {
        "experiment": "H15_S1_entry_surround_ratio_sweep",
        "strategy": "Entry [noload_end:idx_start] + r% no-load (before) + r% Steady (after)",
        "gru": {"mask": GRU_MASK, "sensor_subset": mask_label(GRU_MASK),
                "best_r": gru_best["r"], "best_rmse": gru_best["mean_rmse"],
                "results": gru_results},
        "xgb": {"mask": XGB_MASK, "sensor_subset": mask_label(XGB_MASK),
                "best_r": xgb_best["r"], "best_rmse": xgb_best["mean_rmse"],
                "results": xgb_results},
        "seeds": SEEDS, "ratios": RATIOS, "execution_dir": str(out_dir),
    }
    (out_dir / "logs" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text(
        "\n".join(log_lines), encoding="utf-8")

    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

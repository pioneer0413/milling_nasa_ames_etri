#!/usr/bin/env python3
"""H4_S11: Segment-wise + full-length feature vector (within input ratio).

Idea
----
Current Delta features are extracted over the WHOLE prefix [0:end] (one block).
Here we ALSO split the prefix by cutting segment and extract the same feature set
per segment that exists within the input ratio, then concatenate everything:

  feature_vector = [ FULL-prefix block | ENTRY block | STEADY block | EXIT block | meta ]

Each block = extract_features (mean/rms/std/peak) over its slice of the signal,
per sensor (6) -> 24 delta features per block. Delta = run features minus the
first-run (reference) features of the SAME block. Segments absent within the
prefix (e.g. EXIT at low input ratio) -> zero-filled block.

Segment boundaries (v2): seg_peng2026_steady5 with idx_end(=exit_start) overwritten
by reverse-kurtosis smcDC exit detection ->
  datasets/nasa/cutting_segment_v2/seg_peng2026_steady5_exitfix_reverse_kurtosis.csv
  No-load[0:idx_noload_end] / Entry[..:idx_start] / Steady[idx_start:idx_end] / Exit[idx_end:]

Compares, for XGBoost and Feature-GRU, two feature sets across an input-ratio sweep:
  - full_only  : current 24 delta + 3 meta = 27   (baseline)
  - full+seg   : 4 blocks x 24 + 3 meta = 99       (this experiment)

Usage:
  venv312/bin/python3 scripts/run_H4_S11_segment_feature_vector.py --smoke
  venv312/bin/python3 scripts/run_H4_S11_segment_feature_vector.py
  venv312/bin/python3 scripts/run_H4_S11_segment_feature_vector.py --prefixes 50 60 70 80 90 100

Output: experiments/executions/H4/S11/{timestamp}_segment_feature_vector/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_H4_S5_feature_gru_hpo import (  # noqa: E402
    BASE, CASE_SCOPE, EXCLUDED_RUNS, META_FEATURES, SENSORS, THRESH,
    compute_metrics, extract_features, fit_predict_gru, parse_signal, preprocess,
)

SEG_FILE = ROOT / "datasets/nasa/cutting_segment_v2/seg_peng2026_steady5_exitfix_reverse_kurtosis.csv"
SEG_BLOCKS = ["full", "entry", "steady", "exit"]   # No-load excluded (air-cutting noise)
STATS = ["mean", "rms", "std", "peak"]
SEEDS = [0, 1, 2]
BEST_GRU = {**BASE, "head_hidden": 32, "name": "head32"}
ALL_PREFIXES = list(range(10, 101, 10))


# ─── Feature columns ─────────────────────────────────────────────────────────
def seg_delta_cols(blocks: list[str]) -> list[str]:
    return [f"{b}__{s}__delta_{st}" for b in blocks for s in SENSORS for st in STATS]


def block_slices(noload_end, start, exit_start, end):
    """Index ranges of each block within prefix window [0:end]."""
    e = end
    return {
        "full":   (0, e),
        "noload": (0, min(noload_end, e)),
        "entry":  (min(noload_end, e), min(start, e)),
        "steady": (min(start, e), min(exit_start, e)),
        "exit":   (min(exit_start, e), e),
    }


def extract_blocks(arrays: dict[str, np.ndarray], bounds: dict, blocks: list[str]) -> np.ndarray:
    """Concatenated per-block, per-sensor features (raw, not yet delta)."""
    feats = []
    for b in blocks:
        lo, hi = bounds[b]
        lo, hi = int(max(0, lo)), int(max(0, hi))
        for s in SENSORS:
            seg = arrays[s][lo:hi] if hi > lo else np.empty(0)
            feats.append(extract_features(seg))
    return np.concatenate(feats)


# ─── Segment feature matrix ──────────────────────────────────────────────────
def build_segment_matrix(signal_df, process_df, seg_df, prefix_percents, blocks):
    seg_idx = {(int(r.case), int(r.run)):
               (int(r.idx_noload_end), int(r.idx_start), int(r.idx_end))
               for r in seg_df.itertuples(index=False)}
    cols = seg_delta_cols(blocks)
    cache: dict[tuple, np.ndarray] = {}
    for row in process_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        sig_row = signal_df[(signal_df["case"] == case_id) & (signal_df["run"] == run_id)]
        if sig_row.empty or (case_id, run_id) not in seg_idx:
            continue
        sig_row = sig_row.iloc[0]
        arrays = {s: parse_signal(sig_row[s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        base_len = min(len(a) for a in arrays.values())
        noload_end, start, exit_start = seg_idx[(case_id, run_id)]
        for pct in prefix_percents:
            end = max(1, int(np.ceil(base_len * pct / 100.0)))
            bounds = block_slices(noload_end, start, exit_start, end)
            cache[(case_id, run_id, pct)] = extract_blocks(arrays, bounds, blocks)

    first_run: dict[int, int] = {}
    for case_id in process_df["case"].unique():
        runs = sorted(r for (c, r, _) in cache if c == case_id and (c, r) not in EXCLUDED_RUNS)
        if runs:
            first_run[int(case_id)] = runs[0]

    rows = []
    for row in process_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        if (case_id, run_id) in EXCLUDED_RUNS:
            continue
        ref_run = first_run.get(case_id, run_id)
        vb = float(row.VB) if not pd.isna(row.VB) else 0.0
        for pct in prefix_percents:
            raw = cache.get((case_id, run_id, pct))
            ref = cache.get((case_id, ref_run, pct))
            if raw is None or ref is None:
                continue
            delta = raw - ref
            rec = {"case_id": case_id, "run": run_id, "signal_prefix_percent": float(pct),
                   "VB": vb, "DOC": float(getattr(row, "DOC", 0.0)),
                   "feed": float(getattr(row, "feed", 0.0)),
                   "material": int(getattr(row, "material", 0))}
            for col, val in zip(cols, delta):
                rec[col] = float(val)
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(
        ["signal_prefix_percent", "case_id", "run"]).reset_index(drop=True)


# ─── Sequences for GRU ───────────────────────────────────────────────────────
def build_seg_sequences(feat_df, feat_cols, pct):
    subset = feat_df[feat_df["signal_prefix_percent"] == float(pct)].copy()
    cases = {}
    for cid in subset["case_id"].unique():
        c = subset[subset["case_id"] == cid].sort_values("run").reset_index(drop=True)
        seq = c[feat_cols].to_numpy(dtype=np.float32)
        cases[int(cid)] = {"seq": seq, "vb": c["VB"].to_numpy(dtype=np.float32), "n_runs": len(c)}
    return cases


# ─── Model runners ───────────────────────────────────────────────────────────
def make_xgb(seed):
    return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
                        colsample_bytree=0.9, random_state=seed, verbosity=0, n_jobs=4)


def locv_xgb(feat_df, feat_cols, pct, seed):
    subset = feat_df[feat_df["signal_prefix_percent"] == float(pct)]
    per_case = []
    for tc in CASE_SCOPE:
        te = subset[subset.case_id == tc]; tr = subset[subset.case_id != tc]
        if te.empty or tr.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = make_xgb(seed); m.fit(tr[feat_cols], tr.VB)
            yp = m.predict(te[feat_cols])
        per_case.append(np.sqrt(np.mean((yp - te.VB.to_numpy()) ** 2)))
    return float(np.mean(per_case))


def locv_gru(feat_df, feat_cols, pct, seed, device):
    case_data = build_seg_sequences(feat_df, feat_cols, pct)
    input_dim = len(feat_cols)
    per_case = []
    for tc in CASE_SCOPE:
        if tc not in case_data:
            continue
        train = {c: case_data[c] for c in CASE_SCOPE if c != tc and c in case_data}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yp = fit_predict_gru(train, {tc: case_data[tc]}, input_dim, device, BEST_GRU, seed)[tc]
        yt = case_data[tc]["vb"]
        per_case.append(np.sqrt(np.mean((yp - yt) ** 2)))
    return float(np.mean(per_case))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", type=int, nargs="+", default=ALL_PREFIXES)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--models", nargs="+", default=["xgb", "gru"], choices=["xgb", "gru"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    seeds = [args.seeds[0]] if args.smoke else args.seeds
    prefixes = [80, 100] if args.smoke else args.prefixes

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    tag = "smoke" if args.smoke else "segment_feature_vector"
    out = ROOT / "experiments" / "executions" / "H4" / "S11" / f"{ts}_{tag}"
    for sub in ["analysis", "metrics", "figures", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(m):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); log_lines.append(line)

    log(f"=== H4_S11 segment feature vector === prefixes={prefixes} seeds={seeds} models={args.models}")
    log(f"segment file: {SEG_FILE.name}  blocks={SEG_BLOCKS}")

    sig = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    proc = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    seg = pd.read_csv(SEG_FILE)
    proc = proc[proc.case.isin(CASE_SCOPE)].copy()
    sig = sig[sig.case.isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(proc)

    t0 = time.time()
    feat = build_segment_matrix(sig, proc_clean, seg, prefixes, SEG_BLOCKS)
    log(f"feature matrix rows={len(feat)}  built in {time.time()-t0:.0f}s")

    seg_cols = seg_delta_cols(SEG_BLOCKS)
    full_cols = [c for c in seg_cols if c.startswith("full__")]
    # feature sets: full_only(27) vs full+seg(99)
    FEATSETS = {
        "full_only": full_cols + META_FEATURES,
        "full+seg":  seg_cols + META_FEATURES,
    }
    log(f"feature dims: full_only={len(FEATSETS['full_only'])}  full+seg={len(FEATSETS['full+seg'])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for model in args.models:
        for fs_name, fcols in FEATSETS.items():
            for pct in prefixes:
                for seed in seeds:
                    if model == "xgb":
                        r = locv_xgb(feat, fcols, pct, seed)
                    else:
                        r = locv_gru(feat, fcols, pct, seed, device)
                    rows.append({"model": model, "featset": fs_name, "prefix": pct,
                                 "seed": seed, "locv_rmse": r})
                sub = [x for x in rows if x["model"] == model and x["featset"] == fs_name
                       and x["prefix"] == pct]
                mr = np.mean([x["locv_rmse"] for x in sub])
                log(f"  {model:<3} {fs_name:<10} prefix={pct:>3}%  RMSE={mr:.5f}  [{time.time()-t0:.0f}s]")

    seed_df = pd.DataFrame(rows)
    agg = (seed_df.groupby(["model", "featset", "prefix"], as_index=False)
           .agg(mean_rmse=("locv_rmse", "mean"), std_rmse=("locv_rmse", "std")))
    agg["std_rmse"] = agg["std_rmse"].fillna(0.0)
    seed_df.to_csv(out / "metrics" / "seed_metrics.csv", index=False)
    agg.to_csv(out / "analysis" / "model_featset_prefix.csv", index=False)

    # best per (model, featset)
    best = {}
    for model in args.models:
        for fs in FEATSETS:
            d = agg[(agg.model == model) & (agg.featset == fs)]
            if not d.empty:
                b = d.sort_values("mean_rmse").iloc[0]
                best[(model, fs)] = {"prefix": int(b.prefix), "rmse": float(b.mean_rmse), "std": float(b.std_rmse)}

    # figure: prefix curves, one panel per model
    _fig_curves(agg, args.models, out / "figures")

    summary = {
        "experiment": "H4_S11_segment_feature_vector",
        "segment_file": SEG_FILE.name, "blocks": SEG_BLOCKS,
        "prefixes": prefixes, "seeds": seeds, "models": args.models,
        "feature_dims": {k: len(v) for k, v in FEATSETS.items()},
        "best": {f"{m}|{fs}": v for (m, fs), v in best.items()},
        "execution_dir": str(out),
    }
    (out / "analysis" / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    log("\n=== best per (model, featset) — 3-seed LOCV ===")
    for model in args.models:
        for fs in FEATSETS:
            if (model, fs) in best:
                b = best[(model, fs)]
                log(f"  {model:<3} {fs:<10}  RMSE={b['rmse']:.6f} ±{b['std']:.6f} @ {b['prefix']}%")
        if (model, "full_only") in best and (model, "full+seg") in best:
            d = best[(model, "full+seg")]["rmse"] - best[(model, "full_only")]["rmse"]
            log(f"  -> {model} full+seg vs full_only: Δ={d:+.6f}  ({'개선' if d < 0 else '악화'})")

    (out / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"EXECUTION_DIR={out}", flush=True)


def _fig_curves(agg, models, out: Path):
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 5), squeeze=False)
    colors = {"full_only": "#1f77b4", "full+seg": "#d62728"}
    for i, model in enumerate(models):
        ax = axes[0][i]
        for fs in ["full_only", "full+seg"]:
            d = agg[(agg.model == model) & (agg.featset == fs)].sort_values("prefix")
            if d.empty:
                continue
            b = d.sort_values("mean_rmse").iloc[0]
            ax.errorbar(d.prefix, d.mean_rmse, yerr=d.std_rmse, marker="o", capsize=3,
                        color=colors[fs], label=f"{fs} (best {b.mean_rmse:.4f}@{int(b.prefix)}%)")
        ax.axhline(0.09, color="#888", ls=":", lw=1, label="target 0.09")
        ax.axhline(0.08, color="#555", ls=":", lw=1, label="target 0.08")
        ax.set_xlabel("prefix input ratio (%)"); ax.set_ylabel("LOCV mean RMSE (3-seed)")
        ax.set_title(f"H4_S11 {model} — full_only vs full+seg")
        ax.set_xticks(sorted(agg.prefix.unique())); ax.grid(True, color="#eee", lw=0.5)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "01_prefix_curves.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""H6.S3 T5-T8: segment truncation in the V1 frame (per-timestep, homogeneous).

Corrected comparison for H6 (vs T1-T4 which mixed full history + segment target +
seq2one). Here EVERY run (history and target alike) is truncated to the SAME segment,
delta is computed against run_1 at the SAME segment, and the per-timestep DeltaMetaGRU
predicts every run — exactly the V1 (H4_S5 best `head32`) setup with prefix% replaced
by a cutting-segment window. This makes it apples-to-apples with V1 prefix-80 (0.0908):
the only changed axis is the truncation rule (prefix-80% -> segment).

Tasks (variable = the segment all runs are clipped to), cutting_segment steady5 idx_end:
  T5 no_exit       : [0 : idx_end]            (No-load+Entry+Steady, Exit removed)
  T6 entry_steady  : [idx_noload_end:idx_end] (Entry+Steady)
  T7 steady        : [idx_start:idx_end]      (Steady only)
  T8 full_with_exit: [0 : base_len]           (full, Exit included — control)

Output: experiments/executions/H6/S3/{timestamp}_segment_pertimestep_T5_T8/
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_H4_S5_feature_gru_hpo import (  # noqa: E402
    BASE, CASE_SCOPE, EXCLUDED_RUNS, META_FEATURES, SENSORS, THRESH,
    compute_metrics, delta_col_names, extract_features, parse_signal, preprocess,
    run_locv_seed,
)

SEEDS = [0, 1, 2]
BEST = {**BASE, "head_hidden": 32, "name": "head32"}          # V1 best (per-timestep)
INPUT_DIM = 4 * len(SENSORS) + len(META_FEATURES)             # 27
HEUR = ROOT / "datasets/cutting_segment/seg_peng2026_steady5.csv"
PREFIX80_ANCHOR = 0.090773                                    # V1 prefix-80 (head32, 3-seed)

TASKS = {
    "T5_no_exit": "All runs [:idx_end] (No-load+Entry+Steady, Exit removed)",
    "T6_entry_steady": "All runs [idx_noload_end:idx_end] (Entry+Steady)",
    "T7_steady": "All runs [idx_start:idx_end] (Steady only)",
    "T8_full_with_exit": "All runs full [:base_len] (Exit included) — control",
}


def segment_bounds(task: str, seg: dict, base_len: int) -> tuple[int, int]:
    end = min(int(seg["idx_end"]), base_len)
    if task == "T5_no_exit":
        return 0, end
    if task == "T6_entry_steady":
        return min(int(seg["idx_noload_end"]), end), end
    if task == "T7_steady":
        return min(int(seg["idx_start"]), end), end
    if task == "T8_full_with_exit":
        return 0, base_len
    raise ValueError(task)


def feats(arrays, start, end):
    start = max(0, start)
    end = max(start + 1, end)
    return np.concatenate([extract_features(arrays[s][start:end]) for s in SENSORS])


def build_seg_feats(sig, proc, seg_idx):
    """seg_feat[task][(c,r)] for runs that have BOTH VB (proc) and segment metadata."""
    seg_feat = {t: {} for t in TASKS}
    for row in proc.itertuples(index=False):
        c, r = int(row.case), int(row.run)
        if (c, r) not in seg_idx:
            continue
        s = sig[(sig.case == c) & (sig.run == r)]
        if s.empty:
            continue
        s = s.iloc[0]
        arrays = {sn: parse_signal(s[sn]) for sn in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        base_len = min(len(a) for a in arrays.values())
        for t in TASKS:
            a, b = segment_bounds(t, seg_idx[(c, r)], base_len)
            seg_feat[t][(c, r)] = feats(arrays, a, b)
    return seg_feat


def case_meta(proc):
    meta = {}
    for c in proc.case.unique():
        r = proc[proc.case == c].iloc[0]
        meta[int(c)] = np.array([float(r.DOC), float(r.feed), float(r.material)], dtype=np.float32)
    return meta


def build_case_data(proc, run_feat, meta):
    """all_case_data[case] = {seq[T,27] (delta vs run_1@segment + meta), vb[T], n_runs}."""
    out = {}
    for c in CASE_SCOPE:
        runs = sorted(int(r.run) for r in proc[proc.case == c].itertuples(index=False)
                      if (c, int(r.run)) not in EXCLUDED_RUNS and (c, int(r.run)) in run_feat)
        if not runs:
            continue
        ref = run_feat[(c, runs[0])]
        m = meta[c]
        vbs = {int(r.run): float(r.VB) for r in proc[proc.case == c].itertuples(index=False)}
        steps, vb = [], []
        for rn in runs:
            if rn not in vbs or np.isnan(vbs[rn]):
                continue
            steps.append(np.concatenate([run_feat[(c, rn)] - ref, m]))
            vb.append(vbs[rn])
        if steps:
            out[c] = {"seq": np.asarray(steps, dtype=np.float32),
                      "vb": np.asarray(vb, dtype=np.float32), "n_runs": len(steps)}
    return out


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT / "experiments" / "executions" / "H6" / "S3" / f"{ts}_segment_pertimestep_T5_T8"
    for sub in ["analysis", "metrics", "figures", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(m):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H6.S3 T5-T8: segment + per-timestep DeltaMetaGRU (V1 frame) ===")
    sig = pd.read_csv(ROOT / "datasets/processed/mill_signal_data.csv", usecols=["case", "run"] + SENSORS)
    proc = pd.read_csv(ROOT / "datasets/processed/mill_process_info.csv")
    proc = proc[proc.case.isin(CASE_SCOPE)].copy()
    sig = sig[sig.case.isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(proc)
    heur = pd.read_csv(HEUR)
    heur = heur[heur.case.isin(CASE_SCOPE) & (heur.status == "labeled")]
    seg_idx = {(int(r.case), int(r.run)): {"idx_noload_end": r.idx_noload_end,
               "idx_start": r.idx_start, "idx_end": r.idx_end} for r in heur.itertuples(index=False)}
    log(f"segment-labeled runs in scope: {len(seg_idx)}")

    seg_feat = build_seg_feats(sig, proc_clean, seg_idx)
    meta = case_meta(proc_clean)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"usable runs (VB & segment): {len(seg_feat['T5_no_exit'])}  device={device}")

    seed_rows, per_case_rows = [], []
    t0 = time.time()
    for task in TASKS:
        case_data = build_case_data(proc_clean, seg_feat[task], meta)
        n_runs = sum(d["n_runs"] for d in case_data.values())
        for seed in SEEDS:
            df = run_locv_seed(case_data, INPUT_DIM, device, BEST, seed)
            df.insert(0, "task", task); df.insert(1, "seed", seed)
            per_case_rows.extend(df.to_dict("records"))
            seed_rows.append({"task": task, "seed": seed, "locv_rmse": float(df.rmse.mean()),
                              "locv_r2": float(df.r2.mean()), "locv_mae": float(df.mae.mean())})
        ssd = pd.DataFrame(seed_rows).query("task == @task")
        log(f"{task:<18} ({n_runs} runs, {len(case_data)} cases)  "
            f"mean_rmse={ssd.locv_rmse.mean():.6f} (+/-{ssd.locv_rmse.std():.6f})  "
            f"mean_r2={ssd.locv_r2.mean():.4f}  [{time.time()-t0:.0f}s]")

    seed_df = pd.DataFrame(seed_rows)
    task_df = (seed_df.groupby("task", as_index=False)
               .agg(mean_rmse=("locv_rmse", "mean"), std_rmse=("locv_rmse", "std"),
                    mean_r2=("locv_r2", "mean"), mean_mae=("locv_mae", "mean")))
    task_df["std_rmse"] = task_df["std_rmse"].fillna(0.0)
    order = list(TASKS)
    task_df = task_df.sort_values(by="task", key=lambda s: s.map(order.index))
    task_df["desc"] = task_df.task.map(TASKS)
    task_df["delta_vs_prefix80"] = task_df.mean_rmse - PREFIX80_ANCHOR

    seed_df.to_csv(out / "metrics" / "H6_S3_T5_T8_seed_metrics.csv", index=False)
    pd.DataFrame(per_case_rows).to_csv(out / "metrics" / "H6_S3_T5_T8_per_case_metrics.csv", index=False)
    task_df.to_csv(out / "analysis" / "H6_S3_T5_T8_task_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.4, 5))
    colors = ["#2ca02c", "#2ca02c", "#2ca02c", "#d62728"]
    ax.bar(range(len(task_df)), task_df.mean_rmse, yerr=task_df.std_rmse, capsize=4, color=colors, alpha=0.85)
    ax.axhline(PREFIX80_ANCHOR, color="#1f77b4", ls="--", lw=1.4, label=f"V1 prefix80 = {PREFIX80_ANCHOR:.4f}")
    ax.set_xticks(range(len(task_df)))
    ax.set_xticklabels([t.replace("_", "\n") for t in task_df.task], fontsize=8)
    ax.set_ylabel("LOCV mean RMSE (3-seed)")
    ax.set_title("H6.S3 T5-T8 — segment + per-timestep (V1 frame) vs prefix-80")
    ax.legend(frameon=False); ax.grid(True, axis="y", color="#eee", lw=0.5)
    fig.tight_layout(); fig.savefig(out / "figures" / "H6_S3_T5_T8_task_rmse.png", dpi=150); plt.close(fig)

    exit_removed = task_df[task_df.task != "T8_full_with_exit"]
    best = exit_removed.sort_values("mean_rmse").iloc[0]
    t8 = task_df[task_df.task == "T8_full_with_exit"].iloc[0]
    summary = {
        "experiment": "H6_S3_T5_T8_segment_pertimestep", "seeds": SEEDS, "gru_config": BEST,
        "frame": "V1 per-timestep DeltaMetaGRU, all runs same segment, delta vs run_1@segment",
        "prefix80_anchor": PREFIX80_ANCHOR,
        "tasks": task_df.drop(columns="desc").to_dict("records"),
        "best_exit_removed": {"task": best.task, "mean_rmse": float(best.mean_rmse), "mean_r2": float(best.mean_r2)},
        "control_T8_full": {"mean_rmse": float(t8.mean_rmse), "mean_r2": float(t8.mean_r2)},
        "beats_prefix80": bool(best.mean_rmse < PREFIX80_ANCHOR),
        "exit_removed_beats_T8": bool(best.mean_rmse <= t8.mean_rmse),
        "execution_dir": str(out),
    }
    (out / "analysis" / "H6_S3_T5_T8_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    log("\n=== H6.S3 T5-T8 RESULTS (3-seed LOCV) ===")
    for r in task_df.itertuples(index=False):
        log(f"  {r.task:<18} rmse={r.mean_rmse:.6f} (+/-{r.std_rmse:.6f}) r2={r.mean_r2:.4f} "
            f"d(vs prefix80)={r.delta_vs_prefix80:+.6f}  | {r.desc}")
    log(f"\nBest Exit-removed: {best.task} {best.mean_rmse:.6f}; control T8(full+Exit) {t8.mean_rmse:.6f}; "
        f"V1 prefix80 {PREFIX80_ANCHOR:.6f}")
    log(f"beats prefix80: {summary['beats_prefix80']}; exit-removed <= T8: {summary['exit_removed_beats_T8']}")
    (out / "logs" / "H6_S3_T5_T8_run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"EXECUTION_DIR={out}", flush=True)


if __name__ == "__main__":
    main()

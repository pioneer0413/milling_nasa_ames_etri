#!/usr/bin/env python3
"""H6.S3: Per-run Exit-removal (segment clipping) test — causal check for H6.

Hypothesis (H6): prefix를 늘려 Exit 구간이 포함되면 노이즈가 유입되어 성능이 정체/하락.
This experiment isolates Exit as the noise source by truncating the *current* run at
heuristic segment boundaries (instead of a fixed prefix %), with full history.

Shared design (= H4_S6 history-aware seq2one):
  - history runs (run_1..run_{t-1}) @ full length
  - target run_t @ a segment variant (the single variable across tasks)
  - delta vs run_1@full, meta appended, sequence-to-one Feature-GRU (best `head32`)
  - LOCV (15 cases), 3 seeds; compared against H6.S1 prefix-80% (RMSE 0.1109)

Tasks (variable = target-run input segment), using heuristic indices
No-load[0:idx_noload_end] / Entry[..:idx_start] / Steady[idx_start:idx_end] / Exit[idx_end:]:
  T1 no_exit       : [0 : idx_end]            (No-load+Entry+Steady, Exit removed)
  T2 entry_steady  : [idx_noload_end:idx_end] (Entry+Steady)
  T3 steady        : [idx_start:idx_end]      (Steady only)
  T4 full_with_exit: [0 : signal_end]         (full, Exit included — control)

Output: experiments/executions/H6/S3/{timestamp}_exit_removal/
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

from scripts.run_H4_S6_history_aware_input_ratio import (  # noqa: E402
    BEST, CASE_SCOPE, EXCLUDED_RUNS, INPUT_DIM, META_FEATURES, SENSORS, THRESH,
    case_meta, compute_metrics, extract_features, fit_predict, parse_signal, preprocess,
)

SEEDS = [0, 1, 2]
# Cutting-segment source for the Exit boundary (idx_end). steady5 = Steady target_length 5000.
HEUR = ROOT / "datasets/nasa/cutting_segment/seg_peng2026_steady5.csv"
S1_PREFIX80_RMSE = 0.110854  # H6.S1 comparison anchor

TASKS = {
    "T1_no_exit": "No-load+Entry+Steady ([:idx_end], Exit removed)",
    "T2_entry_steady": "Entry+Steady ([idx_noload_end:idx_end])",
    "T3_steady": "Steady only ([idx_start:idx_end])",
    "T4_full_with_exit": "Full incl. Exit ([:signal_end]) — control",
}


def segment_bounds(task: str, seg: dict, base_len: int) -> tuple[int, int]:
    end = min(int(seg["idx_end"]), base_len)
    if task == "T1_no_exit":
        return 0, end
    if task == "T2_entry_steady":
        return min(int(seg["idx_noload_end"]), end), end
    if task == "T3_steady":
        return min(int(seg["idx_start"]), end), end
    if task == "T4_full_with_exit":
        return 0, base_len
    raise ValueError(task)


def feats(arrays: dict[str, np.ndarray], start: int, end: int) -> np.ndarray:
    start = max(0, start)
    end = max(start + 1, end)
    return np.concatenate([extract_features(arrays[s][start:end]) for s in SENSORS])


def build_caches(sig: pd.DataFrame, proc: pd.DataFrame, seg_idx: dict):
    """full_feat[(c,r)] and seg_feat[task][(c,r)] (target needs metadata; T4 = full)."""
    full_feat: dict = {}
    seg_feat: dict = {t: {} for t in TASKS}
    for row in proc.itertuples(index=False):
        c, r = int(row.case), int(row.run)
        s = sig[(sig.case == c) & (sig.run == r)]
        if s.empty:
            continue
        s = s.iloc[0]
        arrays = {sn: parse_signal(s[sn]) for sn in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        base_len = min(len(a) for a in arrays.values())
        full_feat[(c, r)] = feats(arrays, 0, base_len)
        for t in TASKS:
            if t == "T4_full_with_exit":
                seg_feat[t][(c, r)] = full_feat[(c, r)]
            elif (c, r) in seg_idx:
                a, b = segment_bounds(t, seg_idx[(c, r)], base_len)
                seg_feat[t][(c, r)] = feats(arrays, a, b)
    return full_feat, seg_feat


def ordered_runs(proc: pd.DataFrame, full_feat: dict, case_id: int) -> list[int]:
    return sorted(
        int(r.run) for r in proc[proc.case == case_id].itertuples(index=False)
        if (case_id, int(r.run)) not in EXCLUDED_RUNS and (case_id, int(r.run)) in full_feat
    )


def vb_map(proc: pd.DataFrame, case_id: int) -> dict[int, float]:
    return {int(r.run): float(r.VB) for r in proc[proc.case == case_id].itertuples(index=False)}


def build_samples(proc, full_feat, target_feat, meta, case_id):
    runs = ordered_runs(proc, full_feat, case_id)
    if not runs:
        return []
    ref = full_feat[(case_id, runs[0])]
    m = meta[case_id]
    vbs = vb_map(proc, case_id)
    out = []
    for pos, t_run in enumerate(runs):
        if (case_id, t_run) not in target_feat or t_run not in vbs or np.isnan(vbs[t_run]):
            continue
        steps = [np.concatenate([full_feat[(case_id, runs[k])] - ref, m]) for k in range(pos)]
        steps.append(np.concatenate([target_feat[(case_id, t_run)] - ref, m]))
        out.append((np.asarray(steps, dtype=np.float32), float(vbs[t_run]), int(t_run)))
    return out


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT / "experiments" / "executions" / "H6" / "S3" / f"{ts}_exit_removal"
    for sub in ["analysis", "metrics", "predictions", "figures", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(m):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H6.S3 Exit-removal (per-run segment clipping) ===")
    sig = pd.read_csv(ROOT / "datasets/processed/mill_signal_data.csv", usecols=["case", "run"] + SENSORS)
    proc = pd.read_csv(ROOT / "datasets/processed/mill_process_info.csv")
    proc = proc[proc.case.isin(CASE_SCOPE)].copy()
    sig = sig[sig.case.isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(proc)

    heur = pd.read_csv(HEUR)
    heur = heur[heur.case.isin(CASE_SCOPE) & (heur.status == "labeled")]
    seg_idx = {(int(r.case), int(r.run)): {"idx_noload_end": r.idx_noload_end,
               "idx_start": r.idx_start, "idx_end": r.idx_end} for r in heur.itertuples(index=False)}
    log(f"heuristic-labeled runs in scope: {len(seg_idx)}")

    full_feat, seg_feat = build_caches(sig, proc_clean, seg_idx)
    meta = case_meta(proc_clean)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"full_feat runs={len(full_feat)}  device={device}")

    seed_rows, per_case_rows, pred_rows = [], [], []
    t0 = time.time()
    for task, desc in TASKS.items():
        samples = {c: build_samples(proc_clean, full_feat, seg_feat[task], meta, c) for c in CASE_SCOPE}
        samples = {c: s for c, s in samples.items() if s}
        cases = sorted(samples)
        n_targets = sum(len(s) for s in samples.values())
        for seed in SEEDS:
            for tc in cases:
                train = [s for c in cases if c != tc for s in samples[c]]
                test = samples[tc]
                preds = fit_predict(train, test, device, BEST, seed)
                y_true = np.array([y for _, y, _ in test])
                m = compute_metrics(y_true, preds)
                per_case_rows.append({"task": task, "seed": seed, "case": tc, "n_runs": len(test), **m})
                for (_, yt, rn), yp in zip(test, preds):
                    pred_rows.append({"task": task, "seed": seed, "case": tc, "run": rn,
                                      "y_true": float(yt), "y_pred": float(yp)})
            sd = pd.DataFrame(per_case_rows).query("task == @task and seed == @seed")[["rmse", "r2", "mae"]].mean()
            seed_rows.append({"task": task, "seed": seed, "locv_rmse": float(sd.rmse),
                              "locv_r2": float(sd.r2), "locv_mae": float(sd.mae)})
        ssd = pd.DataFrame(seed_rows).query("task == @task")
        log(f"{task:<18} ({n_targets} targets, {len(cases)} cases)  "
            f"mean_rmse={ssd.locv_rmse.mean():.6f} (+/-{ssd.locv_rmse.std():.6f})  "
            f"mean_r2={ssd.locv_r2.mean():.4f}  [{time.time()-t0:.0f}s]")

    seed_df = pd.DataFrame(seed_rows)
    task_df = (seed_df.groupby("task", as_index=False)
               .agg(mean_rmse=("locv_rmse", "mean"), std_rmse=("locv_rmse", "std"),
                    mean_r2=("locv_r2", "mean"), mean_mae=("locv_mae", "mean")))
    task_df["std_rmse"] = task_df["std_rmse"].fillna(0.0)
    order = list(TASKS)
    task_df["__o"] = task_df.task.map(order.index)
    task_df = task_df.sort_values("__o").drop(columns="__o")
    task_df["desc"] = task_df.task.map(TASKS)
    task_df["delta_vs_S1_prefix80"] = task_df.mean_rmse - S1_PREFIX80_RMSE

    seed_df.to_csv(out / "metrics" / "H6_S3_seed_metrics.csv", index=False)
    pd.DataFrame(per_case_rows).to_csv(out / "metrics" / "H6_S3_per_case_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "predictions" / "H6_S3_predictions.csv", index=False)
    task_df.to_csv(out / "analysis" / "H6_S3_task_metrics.csv", index=False)

    # Figure: task vs RMSE bar with S1 prefix-80 reference
    fig, ax = plt.subplots(figsize=(8.4, 5))
    colors = ["#2ca02c", "#2ca02c", "#2ca02c", "#d62728"]
    ax.bar(task_df.task, task_df.mean_rmse, yerr=task_df.std_rmse, capsize=4, color=colors, alpha=0.85)
    ax.axhline(S1_PREFIX80_RMSE, color="#1f77b4", ls="--", lw=1.4, label=f"H6.S1 prefix80 = {S1_PREFIX80_RMSE:.4f}")
    ax.set_ylabel("LOCV mean RMSE (3-seed)")
    ax.set_title("H6.S3 — current-run segment (Exit removed vs included) vs prediction error")
    ax.set_xticklabels([t.replace("_", "\n") for t in task_df.task], fontsize=8)
    ax.legend(frameon=False); ax.grid(True, axis="y", color="#eee", lw=0.5)
    fig.tight_layout(); fig.savefig(out / "figures" / "H6_S3_task_rmse.png", dpi=150); plt.close(fig)

    exit_removed = task_df[task_df.task != "T4_full_with_exit"]
    best = exit_removed.sort_values("mean_rmse").iloc[0]
    t4 = task_df[task_df.task == "T4_full_with_exit"].iloc[0]
    verdict = bool(best.mean_rmse <= t4.mean_rmse)
    summary = {
        "experiment": "H6_S3_exit_removal", "seeds": SEEDS, "gru_config": BEST,
        "s1_prefix80_rmse": S1_PREFIX80_RMSE,
        "tasks": task_df.drop(columns="desc").to_dict("records"),
        "best_exit_removed": {"task": best.task, "mean_rmse": float(best.mean_rmse), "mean_r2": float(best.mean_r2)},
        "control_T4_full": {"mean_rmse": float(t4.mean_rmse), "mean_r2": float(t4.mean_r2)},
        "exit_is_noise_supported": verdict,
        "execution_dir": str(out),
    }
    (out / "analysis" / "H6_S3_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    log("\n=== H6.S3 RESULTS (3-seed LOCV) ===")
    for r in task_df.itertuples(index=False):
        log(f"  {r.task:<18} rmse={r.mean_rmse:.6f} (+/-{r.std_rmse:.6f}) r2={r.mean_r2:.4f} "
            f"d(vs S1-80)={r.delta_vs_S1_prefix80:+.6f}  | {r.desc}")
    log(f"\nBest Exit-removed: {best.task} rmse={best.mean_rmse:.6f}; control T4(full+Exit) rmse={t4.mean_rmse:.6f}")
    log(f"Exit=noise hypothesis supported (best Exit-removed <= T4 full): {verdict}")
    (out / "logs" / "H6_S3_run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"EXECUTION_DIR={out}", flush=True)


if __name__ == "__main__":
    main()

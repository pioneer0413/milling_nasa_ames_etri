#!/usr/bin/env python3
"""I3_S1: FeatGRU/FeatLSTM hyperparameter tuning + segment selection (NASA).

목표: 입력 길이 100% 고정 + 특정 segment 사용 허용 조건에서, HPO로
FeatGRU 또는 FeatLSTM의 LOCV-15 mean RMSE를 ≤ 0.080 mm 로 끌어내린다.

레버:
  A) segment  : feature 4-stat 추출 시간창 (입력비율은 100% 유지 — prefix 절단 아님)
                {Full, Excl_Exit, Entry_Steady, Steady(v1), Steady_v2(exitfix)}
  B) HPO      : hidden / layers / dropout / head / lr / epochs (FeatGRU & FeatLSTM)

전략 (coarse→fine):
  Stage A: 5 segment × {gru,lstm}, 기본 HPO, 3-seed → cell별 best segment 선택
  Stage B: best segment에서 HPO 그리드, 3-seed
  Stage C: cell별 상위 2 config 를 5-seed 확정 → mean±std 보고

기준선(B3, Full, 5-seed): FeatLSTM 0.092217 / FeatGRU 0.095121.
모델/LOCV/평가 로직은 run_B3_S1_comprehensive_baseline.py 재사용 (SEQ_* 상수 override).

Output: experiments/executions/I3/S1/{timestamp}_seq_hpo/ + leaderboards/i3/s1/
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import sys
import time as time_mod
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)   # each worker single-threaded; parallelism via processes

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# GPU: a single shared device, so one worker avoids cross-process contention.
# CPU: parallelize LOCV evaluations across processes.
N_WORKERS = 1 if DEVICE.type == "cuda" else (os.cpu_count() or 2)

# ─── Import B3 module (models, LOCV, helpers) ────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "b3", ROOT / "scripts" / "run_B3_S1_comprehensive_baseline.py")
b3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b3)

# Populated in main() before the pool is forked; workers inherit it (fork).
_FEAT_BY_SEG: dict[str, pd.DataFrame] = {}

TARGET   = 0.080
SEGMENTS = ["Full", "Excl_Exit", "Entry_Steady", "Steady", "Steady_v2"]
SEG_V1   = ROOT / "datasets/nasa/cutting_segment/seg_peng2026_steady3.csv"
SEG_V2   = ROOT / "datasets/nasa/cutting_segment_v2/seg_peng2026_steady5_exitfix_reverse_kurtosis.csv"

OUT_DIR  = ROOT / "experiments" / "executions" / "I3" / "S1" / (
    datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_seq_hpo")
LB_DIR   = ROOT / "leaderboards" / "i3" / "s1"

_LOG_LINES: list[str] = []
def log(msg: str = "") -> None:
    print(msg, flush=True)
    _LOG_LINES.append(msg)


# ─── Pre-parse raw signals once ──────────────────────────────────────────────
def load_raw_arrays(signal_df, proc_clean):
    """(case,run) → {sensor: np.ndarray}; skip THRESH-exceeding runs."""
    raw: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for row in proc_clean.itertuples(index=False):
        c, r = int(row.case), int(row.run)
        sig = signal_df[(signal_df["case"] == c) & (signal_df["run"] == r)]
        if sig.empty:
            continue
        sr = sig.iloc[0]
        arrays = {s: b3.parse_signal(sr[s]) for s in b3.SENSORS}
        if any(np.abs(a).max() > b3.THRESH for a in arrays.values()):
            continue
        raw[(c, r)] = arrays
    return raw


def load_segments():
    """(case,run) → dict of segment indices, from v1 + v2 files."""
    seg: dict[tuple[int, int], dict] = {}
    d1 = pd.read_csv(SEG_V1)
    for row in d1.itertuples(index=False):
        seg[(int(row.case), int(row.run))] = {
            "idx_noload_end": int(row.idx_noload_end),
            "idx_start": int(row.idx_start),
            "idx_end": int(row.idx_end),
        }
    d2 = pd.read_csv(SEG_V2)
    for row in d2.itertuples(index=False):
        k = (int(row.case), int(row.run))
        seg.setdefault(k, {})
        seg[k]["v2_start"] = int(row.idx_start)
        seg[k]["v2_exit_start"] = int(row.idx_exit_start)
    return seg


def seg_window(seg_name, info, base_len):
    """Return (a, b) slice for a segment; fall back to full on missing idx."""
    if seg_name == "Full" or info is None:
        return 0, base_len
    try:
        if seg_name == "Excl_Exit":
            a, b = 0, info["idx_end"]
        elif seg_name == "Entry_Steady":
            a, b = info["idx_noload_end"], info["idx_end"]
        elif seg_name == "Steady":
            a, b = info["idx_start"], info["idx_end"]
        elif seg_name == "Steady_v2":
            a, b = info["v2_start"], info["v2_exit_start"]
        else:
            return 0, base_len
    except KeyError:
        return 0, base_len
    a = max(0, min(int(a), base_len))
    b = max(0, min(int(b), base_len))
    if b - a < 10:                      # degenerate window → full
        return 0, base_len
    return a, b


def build_feat_df_seg(raw, segs, proc_clean, seg_name):
    """Mirror b3.build_feat_cache/build_feat_df but with a segment window."""
    sensor_idx = b3.mask_sensor_indices(b3.GRU_MASK)
    cache: dict[tuple[int, int], np.ndarray] = {}
    for (c, r), arrays in raw.items():
        base_len = min(len(a) for a in arrays.values())
        a, b = seg_window(seg_name, segs.get((c, r)), base_len)
        full = np.concatenate(
            [b3.extract_features(arrays[s][a:b]) for s in b3.SENSORS])
        cache[(c, r)] = full[sensor_idx]
    first_run: dict[int, int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(rr for (cc, rr) in cache if cc == int(case_id))
        if runs:
            first_run[int(case_id)] = runs[0]
    return b3.build_feat_df(cache, first_run, proc_clean)


# ─── HPO config application ──────────────────────────────────────────────────
def set_hp(cfg: dict) -> None:
    b3.SEQ_HIDDEN   = cfg["hidden"]
    b3.SEQ_LAYERS   = cfg["layers"]
    b3.SEQ_DROPOUT  = cfg["dropout"]
    b3.SEQ_HEAD_HID = cfg["head"]
    b3.SEQ_LR       = cfg["lr"]
    b3.SEQ_EPOCHS   = cfg["epochs"]
    b3.SEQ_WD       = cfg["wd"]


DEFAULT_HP = dict(hidden=256, layers=3, dropout=0.1, head=32,
                  lr=1e-3, epochs=120, wd=1e-4)


def evaluate(feat_df, cell, cfg, seeds, device):
    """Sequential single-config evaluator (used by smoke tests)."""
    set_hp(cfg)
    means = []
    for s in seeds:
        m, _ = b3.run_feat_seq(feat_df, device, s, cell)
        means.append(m)
    return float(np.mean(means)), float(np.std(means)), means


# ─── Parallel task: one (cell, segment, cfg, seed) → single-seed LOCV mean ────
def _eval_task(task: tuple) -> tuple:
    cell, seg_name, cfg_items, seed = task
    torch.set_num_threads(1)
    cfg = dict(cfg_items)
    set_hp(cfg)
    m, _ = b3.run_feat_seq(_FEAT_BY_SEG[seg_name], DEVICE, seed, cell)
    return (cell, seg_name, cfg_items, seed, float(m))


def run_parallel(tasks: list[tuple], pool: ProcessPoolExecutor | None) -> dict:
    """Map tasks → {(cell,seg,cfg_items): [per-seed means]}.

    pool=None runs sequentially in-process — required for CUDA, since forking
    a process after CUDA has been touched in the parent breaks the child's
    CUDA context ("Cannot re-initialize CUDA in forked subprocess").
    """
    agg: dict[tuple, list[float]] = {}
    results = map(_eval_task, tasks) if pool is None else pool.map(_eval_task, tasks)
    for cell, seg_name, cfg_items, seed, m in results:
        agg.setdefault((cell, seg_name, cfg_items), []).append(m)
    return agg


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    global _FEAT_BY_SEG
    t0 = time_mod.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LB_DIR.mkdir(parents=True, exist_ok=True)
    log(f"=== I3_S1: FeatGRU/LSTM HPO + segment (target ≤ {TARGET}) ===")
    log(f"device={DEVICE}, workers={N_WORKERS}")

    log("Loading data...")
    signal_df = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                            usecols=["case", "run"] + b3.SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(b3.CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(b3.CASE_SCOPE)].copy()
    proc_clean = b3.preprocess(process_df)
    raw  = load_raw_arrays(signal_df, proc_clean)
    segs = load_segments()
    log(f"Clean runs: {len(proc_clean)}, parsed signals: {len(raw)}, seg rows: {len(segs)}")

    # Pre-build feat_df per segment; expose as module global so forked workers inherit it.
    _FEAT_BY_SEG = {sg: build_feat_df_seg(raw, segs, proc_clean, sg) for sg in SEGMENTS}

    rows: list[dict] = []
    pool = None if DEVICE.type == "cuda" else ProcessPoolExecutor(max_workers=N_WORKERS)

    def agg_to_rows(agg, stage):
        out = {}
        for (cell, sg, cfg_items), means in agg.items():
            out[(cell, sg, cfg_items)] = (float(np.mean(means)), float(np.std(means)), means)
            rows.append({"stage": stage, "cell": cell, "segment": sg,
                         **dict(cfg_items), "seeds": len(means),
                         "rmse_mean": float(np.mean(means)), "rmse_std": float(np.std(means))})
        return out

    # ── Stage A: segment selection (default HP, 2-seed) ──────────────────────
    log("\n── Stage A: segment sweep (default HP, seeds 0-1) ──")
    seedsA = [0, 1]
    dflt_items = tuple(sorted(DEFAULT_HP.items()))
    tasksA = [(cell, sg, dflt_items, s)
              for cell in ["gru", "lstm"] for sg in SEGMENTS for s in seedsA]
    resA = agg_to_rows(run_parallel(tasksA, pool), "A")
    best_seg, best_seg_rmse = {}, {}
    for cell in ["gru", "lstm"]:
        cand = {sg: resA[(cell, sg, dflt_items)][0] for sg in SEGMENTS}
        for sg in SEGMENTS:
            log(f"  [A] {cell:4s} {sg:13s} rmse={cand[sg]:.6f}")
        best_seg[cell] = min(cand, key=cand.get)
        best_seg_rmse[cell] = cand[best_seg[cell]]
    log(f"  best segment: gru={best_seg['gru']} ({best_seg_rmse['gru']:.6f}), "
        f"lstm={best_seg['lstm']} ({best_seg_rmse['lstm']:.6f})")

    # ── Stage B: HPO grid on best segment per cell (2-seed) ──────────────────
    log("\n── Stage B: HPO grid (seeds 0-1) ──")
    grid = dict(
        hidden=[128, 256],
        layers=[2, 3],
        dropout=[0.0, 0.1],
        head=[32, 64],
        lr=[1e-3, 3e-4],
        epochs=[120],
        wd=[1e-4],
    )
    keys = list(grid)
    all_cfgs = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    log(f"  grid size per cell: {len(all_cfgs)} configs")
    tasksB = []
    for cell in ["gru", "lstm"]:
        sg = best_seg[cell]
        for cfg in all_cfgs:
            ci = tuple(sorted(cfg.items()))
            tasksB += [(cell, sg, ci, s) for s in seedsA]
    resB = agg_to_rows(run_parallel(tasksB, pool), "B")
    stageB_best = {"gru": [], "lstm": []}
    for cell in ["gru", "lstm"]:
        sg = best_seg[cell]
        ranked = sorted(
            ((resB[(cell, sg, tuple(sorted(c.items())))][0], c) for c in all_cfgs),
            key=lambda x: x[0])
        stageB_best[cell] = ranked[:2]
        log(f"  {cell} top3: " + ", ".join(f"{m:.6f}" for m, _ in ranked[:3]))

    # ── Stage C: 5-seed confirmation of top configs ──────────────────────────
    log("\n── Stage C: 5-seed confirmation ──")
    seedsC = [0, 1, 2, 3, 4]
    tasksC = []
    for cell in ["gru", "lstm"]:
        sg = best_seg[cell]
        for _, cfg in stageB_best[cell]:
            ci = tuple(sorted(cfg.items()))
            tasksC += [(cell, sg, ci, s) for s in seedsC]
    resC = agg_to_rows(run_parallel(tasksC, pool), "C")
    if pool is not None:
        pool.shutdown()

    overall_best = {"rmse": 1e9}
    for cell in ["gru", "lstm"]:
        sg = best_seg[cell]
        for rank, (m2, cfg) in enumerate(stageB_best[cell]):
            ci = tuple(sorted(cfg.items()))
            mean, std, per = resC[(cell, sg, ci)]
            log(f"  [C] {cell:4s} {sg:13s} rank{rank} {cfg} → {mean:.6f} ±{std:.6f}  "
                f"(2-seed was {m2:.6f})")
            if mean < overall_best["rmse"]:
                overall_best = {"rmse": mean, "std": std, "cell": cell,
                                "segment": sg, "cfg": cfg, "per_seed": per}

    # ── Report ───────────────────────────────────────────────────────────────
    elapsed = time_mod.time() - t0
    hit = overall_best["rmse"] <= TARGET
    log("\n=== RESULT ===")
    log(f"Best: {overall_best['cell']} / seg={overall_best['segment']} / "
        f"{overall_best['cfg']}")
    log(f"      RMSE = {overall_best['rmse']:.6f} ± {overall_best['std']:.6f}  "
        f"(5-seed)  target ≤ {TARGET} → {'MET ✅' if hit else 'NOT MET'}")
    log(f"Baseline (B3 Full): FeatLSTM 0.092217 / FeatGRU 0.095121")
    log(f"Elapsed: {elapsed/60:.1f} min")

    df = pd.DataFrame(rows)
    df.to_csv(LB_DIR / "hpo_results.csv", index=False)
    df.to_csv(OUT_DIR / "hpo_results.csv", index=False)
    summary = {"target": TARGET, "hit": hit, "best": overall_best,
               "best_seg": best_seg, "elapsed_min": elapsed / 60}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (OUT_DIR / "log.txt").write_text("\n".join(_LOG_LINES))
    log(f"\nSaved: {LB_DIR/'hpo_results.csv'}")
    print(f"EXECUTION_DIR={OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

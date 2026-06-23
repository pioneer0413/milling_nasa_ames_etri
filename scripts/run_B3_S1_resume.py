#!/usr/bin/env python3
"""B3_S1_resume: SignalGRU + FeatRNN/LSTM/GRU 재실행.

이전 B3 부분 실행에서 완료된 결과(CF/RL/ML/DL/Ridge/SVR/RF/MLP_Feat/SignalCNN)를
pre-load하고, 미완료 모델(SignalGRU, FeatRNN, FeatLSTM, FeatGRU)만 실행.
각 모델 완료 직후 checkpoint.json 저장.

Output: experiments/executions/B3/S1/{timestamp}_comprehensive_baseline_resume/
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time as time_mod
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

# ─── Import B3 module ────────────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "b3", ROOT / "scripts" / "run_B3_S1_comprehensive_baseline.py"
)
_b3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b3)

# ─── Known results (CF/RL/ML/DL from H18_S1; Ridge~SignalCNN from partial B3) ──
# (mean, std, case_rmses)  — case_rmses empty: not saved from interrupted run
KNOWN_RESULTS: dict[str, tuple[float, float, dict]] = {
    "CF":        (0.096361, 0.0,       {}),
    "RL":        (0.195592, 0.0,       {}),
    "ML":        (0.169600, 0.0,       {}),
    "DL":        (0.158675, 0.0,       {}),
    "Ridge":     (0.130990, 0.0,       {}),
    "SVR":       (0.131507, 0.0,       {}),
    "RF":        (0.129166, 0.000827,  {}),
    "MLP_Feat":  (0.123542, 0.004714,  {}),
    "SignalCNN": (0.177733, 0.001313,  {}),
}

SEEDS = _b3.SEEDS
REF   = _b3.REF


def save_checkpoint(results: dict, out_dir: Path) -> None:
    ckpt = {n: {"mean": m, "std": s} for n, (m, s, _) in results.items()}
    (out_dir / "metrics" / "checkpoint.json").write_text(
        json.dumps(ckpt, indent=2), encoding="utf-8"
    )


def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = (
        ROOT / "experiments" / "executions" / "B3" / "S1"
        / f"{ts}_comprehensive_baseline_resume"
    )
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== B3_S1 Resume: FeatRNN/LSTM/GRU ===")
    log("Pre-loaded: CF, RL, ML, DL, Ridge, SVR, RF, MLP_Feat, SignalCNN")
    log(f"Running: FeatRNN, FeatLSTM, FeatGRU  (seeds={SEEDS})")

    log("\nLoading data...")
    signal_df  = pd.read_csv(
        ROOT / "datasets/nasa/raw_signal.csv",
        usecols=["case", "run"] + _b3.SENSORS,
    )
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(_b3.CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(_b3.CASE_SCOPE)].copy()
    proc_clean = _b3.preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    log("Building handcrafted feature cache...")
    feat_cache, first_run = _b3.build_feat_cache(signal_df, proc_clean)
    feat_df = _b3.build_feat_df(feat_cache, first_run, proc_clean)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    results: dict[str, tuple[float, float, dict]] = dict(KNOWN_RESULTS)
    t_total = time_mod.time()
    save_checkpoint(results, out_dir)

    # ── Missing stochastic models ─────────────────────────────────────────────
    for name, fn in [
        ("FeatRNN",   lambda seed: _b3.run_feat_seq(feat_df, device, seed, "rnn")),
        ("FeatLSTM",  lambda seed: _b3.run_feat_seq(feat_df, device, seed, "lstm")),
        ("FeatGRU",   lambda seed: _b3.run_feat_seq(feat_df, device, seed, "gru")),
    ]:
        log(f"\n--- {name} (5-seed) ---")
        seed_results: list[tuple[float, dict[int, float]]] = []
        t0 = time_mod.time()
        for seed in SEEDS:
            mean_, case_r = fn(seed)
            seed_results.append((mean_, case_r))
            log(f"  seed={seed}  RMSE={mean_:.6f}  [{time_mod.time()-t0:.0f}s]")
        m, s, case_r = _b3.aggregate_seeds(seed_results)
        results[name] = (m, s, case_r)
        log(f"  {name}: mean={m:.6f}  std={s:.6f}")
        save_checkpoint(results, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    gru_ref = results["FeatGRU"][0]
    log(f"{'Model':<14} {'Mean':>10} {'Std':>10} {'vs FeatGRU':>12}")
    log("-" * 50)
    ordered = sorted(results.items(), key=lambda x: x[1][0])
    model_names_plot, means_plot, stds_plot = [], [], []
    for name, (m, s, _) in ordered:
        log(f"{name:<14} {m:>10.6f} {s:>10.6f} {m-gru_ref:>+12.6f}")
        model_names_plot.append(name)
        means_plot.append(m)
        stds_plot.append(s)
    log("-" * 50)
    for k, v in REF.items():
        log(f"{k:<14} {v:>10.6f} {'':>10} {'(ref)':>12}")

    # ── Save ──────────────────────────────────────────────────────────────────
    agg_rows = [
        {"model": n, "mean": m, "std": s, "vs_gru": m - gru_ref}
        for n, (m, s, _) in results.items()
    ]
    pd.DataFrame(agg_rows).sort_values("mean").to_csv(
        out_dir / "metrics" / "aggregate.csv", index=False
    )

    case_ids = sorted({c for _, _, cr in results.values() for c in cr})
    if case_ids:
        per_case: dict = {"case": case_ids}
        for name, (_, _, cr) in results.items():
            per_case[name] = [cr.get(c, float("nan")) for c in case_ids]
        pd.DataFrame(per_case).to_csv(out_dir / "metrics" / "per_case.csv", index=False)

    _b3.plot_results(model_names_plot, means_plot, stds_plot, out_dir / "figures")

    summary = {
        "experiment": "B3_S1_comprehensive_baseline_resume",
        "seeds": SEEDS,
        "results": {n: {"mean": m, "std": s} for n, (m, s, _) in results.items()},
        "references": REF,
        "feat_gru_rmse": gru_ref,
        "execution_dir": str(out_dir),
        "total_elapsed_s": time_mod.time() - t_total,
    }
    (out_dir / "logs" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "logs" / "run_log.txt").write_text(
        "\n".join(log_lines), encoding="utf-8"
    )

    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

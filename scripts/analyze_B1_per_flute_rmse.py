#!/usr/bin/env python3
"""B1 per-flute RMSE analysis.

VB_flute_1 / VB_flute_2 / VB_flute_3 / VB_max 각각을 target으로 삼아
B1의 전체 모델 라인업을 LOCV 실행, flute별 RMSE 비교.

각 target에서 독립적으로 학습·평가 (VB_max 모델을 flute에 재적용하는 것이 아님).
stochastic 모델: 3-seed (속도 최적화).

Output: experiments/executions/B1/S1/{ts}_per_flute_rmse/
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time as time_mod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Import B1 module ────────────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "b1", ROOT / "scripts" / "run_B1_S1_phm2010_feature_baseline.py"
)
_b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b1)

TARGETS  = ["VB_flute_1", "VB_flute_2", "VB_flute_3", "VB_max"]
SEEDS    = [0, 1, 2]   # 3-seed (speed)
MODELS_DET = [
    ("CF",    lambda df: _b1.run_cf(df)),
    ("RL",    lambda df: _b1.run_rl(df)),
    ("Ridge", lambda df: _b1.run_ridge(df)),
    ("SVR",   lambda df: _b1.run_svr(df)),
]
MODELS_STO = [
    ("RF",       lambda df, dev, s: _b1.run_rf(df, s)),
    ("MLP_Feat", lambda df, dev, s: _b1.run_mlp_feat(df, dev, s)),
    ("FeatRNN",  lambda df, dev, s: _b1.run_feat_seq(df, dev, s, "rnn")),
    ("FeatLSTM", lambda df, dev, s: _b1.run_feat_seq(df, dev, s, "lstm")),
    ("FeatGRU",  lambda df, dev, s: _b1.run_feat_seq(df, dev, s, "gru")),
]
MODEL_ORDER = ["CF", "RL", "Ridge", "SVR", "RF", "MLP_Feat",
               "FeatRNN", "FeatLSTM", "FeatGRU"]


def run_target(
    df_orig: pd.DataFrame,
    target: str,
    device: torch.device,
    log,
) -> dict[str, tuple[float, float, dict]]:
    # alias target column as VB_max so B1 functions work unchanged
    df = df_orig.copy()
    df["VB_max"] = df[target]

    results: dict[str, tuple[float, float, dict]] = {}

    for name, fn in MODELS_DET:
        t0 = time_mod.time()
        mean_, case_r = fn(df)
        results[name] = (mean_, 0.0, case_r)
        log(f"    {name:<12} {mean_:>8.2f} µm  [{time_mod.time()-t0:.1f}s]")

    for name, fn in MODELS_STO:
        t0 = time_mod.time()
        seed_results = [fn(df, device, s) for s in SEEDS]
        m, s_, case_r = _b1.aggregate_seeds(seed_results)
        results[name] = (m, s_, case_r)
        log(f"    {name:<12} {m:>8.2f} µm  ±{s_:.2f}  [{time_mod.time()-t0:.0f}s]")

    return results


def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = (
        ROOT / "experiments" / "executions" / "B1" / "S1"
        / f"{ts}_per_flute_rmse"
    )
    for sub in ["metrics", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== B1 Per-Flute RMSE Analysis ===")
    log(f"Targets: {TARGETS}")
    log(f"Seeds (stochastic): {SEEDS}")

    process_df, feat_df = _b1.load_data()
    df = _b1.build_merged(process_df, feat_df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    all_results: dict[str, dict[str, tuple[float, float, dict]]] = {}
    t_total = time_mod.time()

    for target in TARGETS:
        log(f"\n  --- Target: {target} ---")
        all_results[target] = run_target(df, target, device, log)

    # ── Summary matrix ────────────────────────────────────────────────────────
    log("\n=== SUMMARY MATRIX (mean RMSE µm) ===")
    col_w = 13
    header = f"{'Model':<12} " + "".join(f"{t:>{col_w}}" for t in TARGETS)
    log(header)
    log("-" * len(header))
    for m_name in MODEL_ORDER:
        row = f"{m_name:<12} "
        row += "".join(
            f"{all_results[t][m_name][0]:>{col_w}.2f}" for t in TARGETS
        )
        log(row)

    # Best per target
    log("\n  Best (excl. CF):")
    for t in TARGETS:
        non_cf = {n: v for n, v in all_results[t].items() if n != "CF"}
        best_n = min(non_cf, key=lambda n: non_cf[n][0])
        log(f"    {t}: {best_n} = {non_cf[best_n][0]:.2f} µm")

    # ── Save ──────────────────────────────────────────────────────────────────
    rows = []
    for target in TARGETS:
        for m_name in MODEL_ORDER:
            mean_, std_, _ = all_results[target][m_name]
            rows.append({
                "target":  target,
                "model":   m_name,
                "mean_um": mean_,
                "std_um":  std_,
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "metrics" / "per_flute_rmse.csv", index=False)

    # Pivot: model × target
    pivot = df_out.pivot(index="model", columns="target", values="mean_um")
    pivot = pivot.reindex(MODEL_ORDER)[TARGETS]
    pivot.to_csv(out_dir / "metrics" / "per_flute_pivot.csv")

    summary = {
        "targets": TARGETS,
        "seeds": SEEDS,
        "results": {
            t: {n: {"mean_um": m, "std_um": s}
                for n, (m, s, _) in all_results[t].items()}
            for t in TARGETS
        },
        "total_elapsed_s": time_mod.time() - t_total,
        "execution_dir": str(out_dir),
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

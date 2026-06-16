#!/usr/bin/env python3
"""E5_S1: PHM2010 Delta-feature 비대칭 해소 실험

배경(직전 Analysis): NASA와 PHM2010 sequence 모델 구현은 동일하나 feature 구성이
비대칭이다.
  - NASA  : first-run 기준 **Delta + Meta** (case baseline 제거 → cross-case 일반화 유리)
  - PHM2010: **raw 절대 통계 70-dim** (case별 센서 offset 잔존 → 일반화 불리)

본 실험은 PHM2010에 NASA식 first-run 기준 Delta feature를 적용해 비대칭을 해소하고,
sequence vs run-independent(RF) 격차가 좁혀지는지 검증한다.
Meta(DOC/feed/material)는 PHM2010에서 전부 NaN(동일 작동조건)이라 제외.

Feature 모드:
  raw        : 70 raw 통계 (= B1_S1 baseline)
  delta      : 70 delta = feat[run] − feat[first_run_of_case]  (leakage-free; 라벨 미사용)
  raw+delta  : 140 (raw 70 + delta 70)

Target: VB_max.  Protocol: LOCV-3 (case 1/4/6), 5-seed (확률 모델).
모델 러너는 run_B1_S1 모듈을 재사용 (_b1.FEAT_COLS 교체).

Output: experiments/executions/E5/S1/{timestamp}_delta_feature/
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

_spec = importlib.util.spec_from_file_location(
    "b1", ROOT / "scripts" / "run_B1_S1_phm2010_feature_baseline.py")
_b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b1)

CASE_SCOPE = _b1.CASE_SCOPE
SEEDS      = _b1.SEEDS
RAW_COLS   = list(_b1.FEAT_COLS)                       # 70
DELTA_COLS = [f"delta__{c}" for c in RAW_COLS]         # 70
MODES      = ["raw", "delta", "raw+delta"]
MODE_COLS  = {"raw": RAW_COLS, "delta": DELTA_COLS, "raw+delta": RAW_COLS + DELTA_COLS}

# B1_S1 reference (raw, VB_max 직접 학습, LOCV-3 5-seed mean RMSE µm)
BASELINE_RAW = {
    "Ridge": 40.81, "SVR": 27.20, "RF": 20.35, "MLP_Feat": 70.20,
    "FeatRNN": 23.64, "FeatLSTM": 21.14, "FeatGRU": 22.46,
}


def add_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Append first-run-referenced delta features per case (leakage-free)."""
    out = df.copy()
    parts = []
    for c in CASE_SCOPE:
        sub = df[df["case"] == c].sort_values("run")
        ref = sub.iloc[0][RAW_COLS].values.astype(float)    # first observed run
        d = sub[RAW_COLS].values.astype(float) - ref
        parts.append(pd.DataFrame(d, columns=DELTA_COLS, index=sub.index))
    out[DELTA_COLS] = pd.concat(parts).sort_index()
    return out


def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "E5" / "S1" / f"{ts}_delta_feature"
    for sub in ["metrics", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== E5_S1: PHM2010 Delta-feature 비대칭 해소 ===")
    log(f"Cases={CASE_SCOPE}, Seeds={SEEDS}, Modes={MODES}")
    log("Delta = feat[run] − feat[first_run_of_case] (NASA식, Meta 제외)")

    proc, feat = _b1.load_data()
    base = _b1.build_merged(proc, feat)
    df = add_delta(base)
    log(f"Total rows: {len(df)}  | raw=70, delta=70, raw+delta=140")
    # sanity: first run delta == 0
    fr = df[df.run == df.groupby('case')['run'].transform('min')]
    log(f"  first-run delta max abs = {df.loc[fr.index, DELTA_COLS].abs().values.max():.2e} (≈0 기대)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    DET = [
        ("Ridge", lambda: _b1.run_ridge(df)),
        ("SVR",   lambda: _b1.run_svr(df)),
    ]
    STO = [
        ("RF",       lambda s: _b1.run_rf(df, s)),
        ("MLP_Feat", lambda s: _b1.run_mlp_feat(df, device, s)),
        ("FeatRNN",  lambda s: _b1.run_feat_seq(df, device, s, "rnn")),
        ("FeatLSTM", lambda s: _b1.run_feat_seq(df, device, s, "lstm")),
        ("FeatGRU",  lambda s: _b1.run_feat_seq(df, device, s, "gru")),
    ]
    MODEL_ORDER = ["Ridge", "SVR", "RF", "MLP_Feat", "FeatRNN", "FeatLSTM", "FeatGRU"]

    # CF / RL reference (feature-independent → mode 불변, 1회만)
    cf_m, _  = _b1.run_cf(df)
    rl_m, _  = _b1.run_rl(df)
    log(f"\n[reference] CF={cf_m:.2f}  RL={rl_m:.2f} µm (mode-invariant)")

    results: dict[str, dict[str, tuple[float, float, dict]]] = {m: {} for m in MODES}
    t_total = time_mod.time()

    for mode in MODES:
        _b1.FEAT_COLS = MODE_COLS[mode]            # monkeypatch feature columns
        log(f"\n########## MODE = {mode}  ({len(MODE_COLS[mode])} features) ##########")
        for name, fn in DET:
            t0 = time_mod.time()
            m, cr = fn()
            results[mode][name] = (m, 0.0, cr)
            log(f"  {name:<10} {m:>7.2f} µm  [{time_mod.time()-t0:.1f}s]")
        for name, fn in STO:
            t0 = time_mod.time()
            seed_res = [fn(s) for s in SEEDS]
            m, s_, cr = _b1.aggregate_seeds(seed_res)
            results[mode][name] = (m, s_, cr)
            log(f"  {name:<10} {m:>7.2f} µm  ±{s_:.2f}  [{time_mod.time()-t0:.0f}s]")

    _b1.FEAT_COLS = RAW_COLS                        # restore

    # ── Summary ──
    log("\n=== SUMMARY (LOCV-3 mean RMSE µm) ===")
    log(f"{'Model':<10} {'raw':>8} {'delta':>8} {'raw+delta':>10} {'Δ(delta)':>9}")
    log("-" * 50)
    for name in MODEL_ORDER:
        r  = results["raw"][name][0]
        d  = results["delta"][name][0]
        rd = results["raw+delta"][name][0]
        log(f"{name:<10} {r:>8.2f} {d:>8.2f} {rd:>10.2f} {d-r:>+9.2f}")
    log(f"\nCF={cf_m:.2f} (oracle reference, 불변)")
    # best non-CF per mode
    for mode in MODES:
        best = min(MODEL_ORDER, key=lambda n: results[mode][n][0])
        log(f"  best non-CF [{mode}]: {best} = {results[mode][best][0]:.2f} µm")

    # ── Save ──
    rows = []
    for mode in MODES:
        for name in MODEL_ORDER:
            m, s_, cr = results[mode][name]
            rows.append({"mode": mode, "model": name, "mean_um": m, "std_um": s_,
                         **{f"case_{c}": cr.get(c, float('nan')) for c in CASE_SCOPE},
                         "baseline_raw": BASELINE_RAW.get(name, float('nan'))})
    pd.DataFrame(rows).to_csv(out_dir / "metrics" / "delta_results.csv", index=False)
    summary = {
        "modes": MODES, "seeds": SEEDS, "model_order": MODEL_ORDER,
        "cf_reference": cf_m, "rl_reference": rl_m,
        "results": {mode: {n: {"mean_um": results[mode][n][0],
                               "std_um": results[mode][n][1],
                               "per_case": results[mode][n][2]} for n in MODEL_ORDER}
                    for mode in MODES},
        "total_elapsed_s": time_mod.time() - t_total,
        "execution_dir": str(out_dir),
    }
    (out_dir / "logs" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

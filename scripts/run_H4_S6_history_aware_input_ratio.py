#!/usr/bin/env python3
"""H4_S6: History-aware input-ratio experiment for Feature-GRU (seq2one).

Intent (corrected vs H4_S1/H4_S5):
  To predict run_t's VB, the *completed* prior runs (run_1..run_{t-1}) are already
  available at FULL length; only the *current* run_t is observed partially (r%).
  This reflects the real temporal dependency: predict the current run's wear early,
  given the full history.

Per target run t at ratio r, one sample is built:
    sequence = [ run_1@100%, run_2@100%, ..., run_{t-1}@100%, run_t@r% ]   (length t)
    target   = VB[run_t]
Delta features are relative to run_1@100% (the always-available baseline); meta
(DOC/feed/material) is appended per timestep. A sequence-to-one GRU predicts VB
from the last (current-run) hidden state.

Design decisions:
  1. run_1 (t=1, no history) -> single-step sequence [run_1@r%].
  2. ratio applies ONLY to the target run; history is always full length.
  3. per-ratio models: one model trained/evaluated per r in {10,...,90}.
  4. New scenario H4_S6, separate from the leader board (different task).

Validation: LOCV (15 cases), 3 random seeds averaged.
GRU = best H4_S5 config (head32): hidden=256, layers=3, dropout=0.1, head=32,
      lr=1e-3, wd=1e-4, epochs=200, CosineAnnealingLR, grad_clip=1.0.

Output: experiments/executions/H4/S6/{timestamp}_history_aware_input_ratio/
"""
from __future__ import annotations

import argparse
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
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_H4_S5_feature_gru_hpo import (  # noqa: E402
    BASE, CASE_SCOPE, EXCLUDED_RUNS, META_FEATURES, SENSORS, THRESH,
    compute_metrics, extract_features, parse_signal, preprocess,
)

RATIOS = list(range(10, 91, 10))          # 10..90 (target run partial; history full)
SEEDS = [0, 1, 2]
FULL = 100
BEST = {**BASE, "head_hidden": 32, "name": "head32"}
N_STATS = 4 * len(SENSORS)                 # 24 delta stats
INPUT_DIM = N_STATS + len(META_FEATURES)   # 27


# ─── Raw per-(case,run,pct) statistics cache ───────────────────────────────────
def build_feature_cache(signal_df: pd.DataFrame, process_df: pd.DataFrame, pcts: list[int]):
    cache: dict[tuple[int, int, int], np.ndarray] = {}
    for row in process_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        sig = signal_df[(signal_df["case"] == case_id) & (signal_df["run"] == run_id)]
        if sig.empty:
            continue
        sig = sig.iloc[0]
        arrays = {s: parse_signal(sig[s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        base_len = min(len(a) for a in arrays.values())
        for pct in pcts:
            end = max(1, int(np.ceil(base_len * pct / 100.0)))
            cache[(case_id, run_id, pct)] = np.concatenate(
                [extract_features(arrays[s][:end]) for s in SENSORS]
            )
    return cache


def case_meta(process_df: pd.DataFrame) -> dict[int, np.ndarray]:
    meta: dict[int, np.ndarray] = {}
    for case_id in process_df["case"].unique():
        r = process_df[process_df["case"] == case_id].iloc[0]
        meta[int(case_id)] = np.array(
            [float(r["DOC"]), float(r["feed"]), float(r["material"])], dtype=np.float32
        )
    return meta


def ordered_runs(process_df: pd.DataFrame, cache, case_id: int) -> list[int]:
    runs = sorted(
        int(r.run) for r in process_df[process_df["case"] == case_id].itertuples(index=False)
        if (case_id, int(r.run)) not in EXCLUDED_RUNS and (case_id, int(r.run), FULL) in cache
    )
    return runs


def vb_map(process_df: pd.DataFrame, case_id: int) -> dict[int, float]:
    sub = process_df[process_df["case"] == case_id]
    return {int(r.run): float(r.VB) for r in sub.itertuples(index=False)}


def build_samples_for_ratio(process_df, cache, meta, ratio: int):
    """Return {case_id: [(seq[L,D] float32, target_vb float, run_id int), ...]}."""
    samples: dict[int, list[tuple[np.ndarray, float, int]]] = {}
    for case_id in CASE_SCOPE:
        runs = ordered_runs(process_df, cache, case_id)
        if not runs:
            continue
        ref = cache[(case_id, runs[0], FULL)]            # run_1@100% baseline
        m = meta[case_id]
        vbs = vb_map(process_df, case_id)
        case_samples = []
        for pos, t_run in enumerate(runs):
            steps = []
            for k in range(pos):                          # history runs @100%
                delta = cache[(case_id, runs[k], FULL)] - ref
                steps.append(np.concatenate([delta, m]))
            tgt_feat = cache[(case_id, t_run, ratio)]      # current run @r%
            steps.append(np.concatenate([tgt_feat - ref, m]))
            seq = np.asarray(steps, dtype=np.float32)      # [pos+1, D]
            if t_run not in vbs or np.isnan(vbs[t_run]):
                continue
            case_samples.append((seq, float(vbs[t_run]), int(t_run)))
        if case_samples:
            samples[case_id] = case_samples
    return samples


# ─── Sequence-to-one GRU ───────────────────────────────────────────────────────
class Seq2OneGRU(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout, head_hidden):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1)
        )

    def forward(self, x_padded, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)            # h_n: [num_layers, B, H]; last valid step per seq
        return self.head(h_n[-1]).squeeze(-1)


def _pad(seqs: list[np.ndarray], device):
    tensors = [torch.tensor(s, dtype=torch.float32) for s in seqs]
    lengths = torch.tensor([len(s) for s in seqs])
    x = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0).to(device)
    return x, lengths


def fit_predict(train_samples, test_samples, device, cfg, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    tr_seqs = [s for s, _, _ in train_samples]
    tr_y = np.array([y for _, y, _ in train_samples], dtype=np.float32)
    y_mean = float(tr_y.mean())
    y_std = float(tr_y.std()) if tr_y.std() > 1e-8 else 1.0

    x_tr, len_tr = _pad(tr_seqs, device)
    y_tr = torch.tensor((tr_y - y_mean) / y_std, dtype=torch.float32).to(device)

    model = Seq2OneGRU(INPUT_DIM, cfg["hidden_size"], cfg["num_layers"],
                       cfg["dropout"], cfg["head_hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    model.train()
    for _ in range(cfg["epochs"]):
        opt.zero_grad()
        pred = model(x_tr, len_tr)
        loss = ((pred - y_tr) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()

    model.eval()
    x_te, len_te = _pad([s for s, _, _ in test_samples], device)
    with torch.no_grad():
        pred = model(x_te, len_te).cpu().numpy() * y_std + y_mean
    return pred


# ─── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="H4_S6 history-aware input-ratio experiment.")
    p.add_argument("--ratios", type=int, nargs="+", default=RATIOS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ratios = args.ratios[:1] if args.smoke else args.ratios
    seeds = args.seeds[:1] if args.smoke else args.seeds

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT / "experiments" / "executions" / "H4" / "S6" / f"{ts}_history_aware_input_ratio"
    for sub in ["configs", "metrics", "analysis", "figures", "predictions", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(m):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"=== H4_S6 history-aware input ratio === ratios={ratios} seeds={seeds}")
    sig = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    proc = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    proc = proc[proc["case"].isin(CASE_SCOPE)].copy()
    sig = sig[sig["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(proc)
    cache = build_feature_cache(sig, proc_clean, sorted(set(ratios) | {FULL}))
    meta = case_meta(proc_clean)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"feature cache entries={len(cache)}  device={device}")

    seed_rows, pred_rows, per_case_rows = [], [], []
    t0 = time.time()
    for ratio in ratios:
        samples = build_samples_for_ratio(proc_clean, cache, meta, ratio)
        cases = [c for c in CASE_SCOPE if c in samples]
        for seed in seeds:
            case_rmse = []
            for tc in cases:
                train = [s for c in cases if c != tc for s in samples[c]]
                test = samples[tc]
                preds = fit_predict(train, test, device, BEST, seed)
                y_true = np.array([y for _, y, _ in test])
                runs = [r for _, _, r in test]
                m = compute_metrics(y_true, preds)
                case_rmse.append(m["rmse"])
                per_case_rows.append({"ratio": ratio, "seed": seed, "case": tc, "n_runs": len(test), **m})
                for rn, yt, yp in zip(runs, y_true, preds):
                    pred_rows.append({"ratio": ratio, "seed": seed, "case": tc, "run": rn,
                                      "y_true": float(yt), "y_pred": float(yp)})
            locv = (pd.DataFrame(per_case_rows)
                    .query("ratio == @ratio and seed == @seed")[["rmse", "r2", "mae"]].mean())
            seed_rows.append({"ratio": ratio, "seed": seed, "locv_rmse": float(locv["rmse"]),
                              "locv_r2": float(locv["r2"]), "locv_mae": float(locv["mae"])})
        sd = pd.DataFrame(seed_rows).query("ratio == @ratio")
        log(f"ratio={ratio:>2}%  mean_rmse={sd.locv_rmse.mean():.6f} "
            f"(+/-{sd.locv_rmse.std():.6f})  mean_r2={sd.locv_r2.mean():.4f}  [{time.time()-t0:.0f}s]")

    seed_df = pd.DataFrame(seed_rows)
    pred_df = pd.DataFrame(pred_rows)
    per_case_df = pd.DataFrame(per_case_rows)
    ratio_df = (seed_df.groupby("ratio", as_index=False)
                .agg(mean_rmse=("locv_rmse", "mean"), std_rmse=("locv_rmse", "std"),
                     mean_r2=("locv_r2", "mean"), mean_mae=("locv_mae", "mean"))
                .sort_values("ratio"))
    ratio_df["std_rmse"] = ratio_df["std_rmse"].fillna(0.0)

    seed_df.to_csv(out / "metrics" / "H4_S6_seed_metrics.csv", index=False)
    per_case_df.to_csv(out / "metrics" / "H4_S6_per_case_metrics.csv", index=False)
    pred_df.to_csv(out / "predictions" / "H4_S6_predictions.csv", index=False)
    ratio_df.to_csv(out / "analysis" / "H4_S6_ratio_metrics.csv", index=False)

    # ── Figure: ratio vs LOCV RMSE (+R2) ───────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.errorbar(ratio_df.ratio, ratio_df.mean_rmse, yerr=ratio_df.std_rmse,
                 marker="o", color="#d62728", capsize=3, label="LOCV RMSE")
    ax1.set_xlabel("current-run input ratio (%)  [history runs always 100%]")
    ax1.set_ylabel("LOCV mean RMSE", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(True, color="#eee", lw=0.5)
    ax2 = ax1.twinx()
    ax2.plot(ratio_df.ratio, ratio_df.mean_r2, marker="s", color="#1f77b4", label="LOCV R2")
    ax2.set_ylabel("LOCV mean R2", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_title("H4_S6 history-aware: current-run ratio vs prediction error (3-seed)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "H4_S6_ratio_rmse_curve.png", dpi=150)
    plt.close(fig)

    best = ratio_df.sort_values("mean_rmse").iloc[0]
    cfg_meta = {
        "experiment": "H4_S6_history_aware_input_ratio",
        "design": "history runs @100%, target run @r%, seq2one GRU, per-ratio model, LOCV, 3 seeds",
        "ratios": ratios, "seeds": seeds, "gru_config": BEST,
        "best_ratio": {"ratio": int(best.ratio), "mean_rmse": float(best.mean_rmse),
                       "mean_r2": float(best.mean_r2)},
        "execution_dir": str(out),
    }
    (out / "configs" / "H4_S6_input_config.json").write_text(
        json.dumps(cfg_meta, indent=2, default=float), encoding="utf-8")

    lines = [
        "# H4_S6 History-Aware Input-Ratio Experiment", "",
        "- Design: 이력 run은 full-length(100%), 예측 대상 run만 r%. seq2one GRU, per-ratio, LOCV, 3 seeds.",
        f"- GRU config: head32 (hidden=256, layers=3, dropout=0.1, head=32, lr=1e-3, wd=1e-4, epochs=200)",
        f"- Best ratio: **{int(best.ratio)}%** -> mean RMSE {best.mean_rmse:.6f}, mean R2 {best.mean_r2:.4f}", "",
        "## Ratio vs LOCV metrics (3-seed mean)", "",
        "| ratio % | mean_rmse | std_rmse | mean_r2 | mean_mae |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in ratio_df.itertuples(index=False):
        lines.append(f"| {int(r.ratio)} | {r.mean_rmse:.6f} | {r.std_rmse:.6f} | {r.mean_r2:.4f} | {r.mean_mae:.6f} |")
    (out / "analysis" / "H4_S6_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "logs" / "H4_S6_run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"Best ratio={int(best.ratio)}% rmse={best.mean_rmse:.6f}. Done: {out}")
    print(f"EXECUTION_DIR={out}", flush=True)


if __name__ == "__main__":
    main()

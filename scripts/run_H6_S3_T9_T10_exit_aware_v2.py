#!/usr/bin/env python3
"""H6.S3 T9-T10: Exit-aware truncation with the corrected v2 Exit detector.

The v2 cutting-segment table (seg_reverse_kurtosis_smcDC.csv) flags `has_exit` per run
and gives `idx_exit_start`. Crucially its has_exit matches the domain ground truth
("Exit cut visible"): exit cases = {1,2,5,8,9,12,14,16}, no-exit = {3,4,7,10,11,13,15}.
Unlike steady5 (which forced a Steady window even where no Exit exists), here runs
without Exit keep the FULL signal.

Two conditions (same V1 frame: per-timestep DeltaMetaGRU head32, delta vs run_1 @ same
rule, LOCV 15 cases, 3 seeds):
  T9  exit_aware : has_exit -> [:idx_exit_start] (Exit removed); else full signal
  T10 full       : full signal for every run (Exit included) — control

Reports overall + breakdown by Exit presence (exit-cases vs no-exit-cases); for no-exit
cases the two conditions are identical by construction, so the difference isolates the
Exit-removal effect on the exit-cases.

Output: experiments/executions/H6/S3/{timestamp}_exit_aware_v2_T9_T10/
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
    compute_metrics, extract_features, parse_signal, preprocess, run_locv_seed,
)

SEEDS = [0, 1, 2]
BEST = {**BASE, "head_hidden": 32, "name": "head32"}
INPUT_DIM = 4 * len(SENSORS) + len(META_FEATURES)            # 27
SEG_V2 = ROOT / "datasets/nasa/cutting_segment_v2/seg_reverse_kurtosis_smcDC.csv"
PREFIX80_ANCHOR = 0.090773                                   # V1 prefix-80 (head32, 3-seed)
T5_STEADY5 = 0.101950                                        # steady5 exit-removed (T5) for reference

CONDS = {
    "T9_exit_aware": "has_exit -> [:idx_exit_start] (Exit removed); no-exit -> full",
    "T10_full": "full signal for every run (Exit included) — control",
}


def feats(arrays, end):
    end = max(1, int(end))
    return np.concatenate([extract_features(arrays[s][:end]) for s in SENSORS])


def build_cond_feats(sig, proc, ex):
    """cond_feat[cond][(c,r)] for runs with VB (proc) and a v2 entry."""
    cond_feat = {c: {} for c in CONDS}
    for row in proc.itertuples(index=False):
        c, r = int(row.case), int(row.run)
        if (c, r) not in ex:
            continue
        s = sig[(sig.case == c) & (sig.run == r)]
        if s.empty:
            continue
        s = s.iloc[0]
        arrays = {sn: parse_signal(s[sn]) for sn in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        base_len = min(len(a) for a in arrays.values())
        has_exit, idx_exit = ex[(c, r)]
        exit_end = min(int(idx_exit), base_len) if has_exit else base_len
        cond_feat["T9_exit_aware"][(c, r)] = feats(arrays, exit_end)
        cond_feat["T10_full"][(c, r)] = feats(arrays, base_len)
    return cond_feat


def case_meta(proc):
    meta = {}
    for c in proc.case.unique():
        r = proc[proc.case == c].iloc[0]
        meta[int(c)] = np.array([float(r.DOC), float(r.feed), float(r.material)], dtype=np.float32)
    return meta


def build_case_data(proc, run_feat, meta):
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
    out = ROOT / "experiments" / "executions" / "H6" / "S3" / f"{ts}_exit_aware_v2_T9_T10"
    for sub in ["analysis", "metrics", "figures", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(m):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H6.S3 T9-T10: exit-aware truncation (v2 detector) ===")
    sig = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    proc = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    proc = proc[proc.case.isin(CASE_SCOPE)].copy()
    sig = sig[sig.case.isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(proc)

    v2 = pd.read_csv(SEG_V2)
    v2 = v2[v2.case.isin(CASE_SCOPE)]
    ex = {(int(r.case), int(r.run)): (bool(r.has_exit), int(r.idx_exit_start))
          for r in v2.itertuples(index=False)}

    cond_feat = build_cond_feats(sig, proc_clean, ex)
    meta = case_meta(proc_clean)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usable = sorted(cond_feat["T9_exit_aware"])
    # per-case Exit label = majority has_exit among usable runs
    case_has_exit = {}
    for c in CASE_SCOPE:
        flags = [ex[(c, r)][0] for (cc, r) in usable if cc == c]
        if flags:
            case_has_exit[c] = (np.mean(flags) > 0.5)
    exit_cases = sorted(c for c, v in case_has_exit.items() if v)
    noexit_cases = sorted(c for c, v in case_has_exit.items() if not v)
    log(f"usable runs={len(usable)}  exit-cases={exit_cases}  no-exit-cases={noexit_cases}  device={device}")

    seed_rows, per_case_rows = [], []
    t0 = time.time()
    for cond in CONDS:
        case_data = build_case_data(proc_clean, cond_feat[cond], meta)
        for seed in SEEDS:
            df = run_locv_seed(case_data, INPUT_DIM, device, BEST, seed)
            df.insert(0, "cond", cond); df.insert(1, "seed", seed)
            per_case_rows.extend(df.to_dict("records"))
            seed_rows.append({"cond": cond, "seed": seed, "locv_rmse": float(df.rmse.mean()),
                              "locv_r2": float(df.r2.mean()), "locv_mae": float(df.mae.mean())})
        ssd = pd.DataFrame(seed_rows).query("cond == @cond")
        log(f"{cond:<16} mean_rmse={ssd.locv_rmse.mean():.6f} (+/-{ssd.locv_rmse.std():.6f}) "
            f"mean_r2={ssd.locv_r2.mean():.4f}  [{time.time()-t0:.0f}s]")

    seed_df = pd.DataFrame(seed_rows)
    per_case_df = pd.DataFrame(per_case_rows)
    per_case_df["exit_case"] = per_case_df.target_case.map(case_has_exit)
    # per (cond, case) 3-seed mean
    pc = per_case_df.groupby(["cond", "target_case"], as_index=False).agg(
        rmse=("rmse", "mean"), r2=("r2", "mean"), mae=("mae", "mean"))
    pc["exit_case"] = pc.target_case.map(case_has_exit)

    cond_df = (seed_df.groupby("cond", as_index=False)
               .agg(mean_rmse=("locv_rmse", "mean"), std_rmse=("locv_rmse", "std"),
                    mean_r2=("locv_r2", "mean"), mean_mae=("locv_mae", "mean")))
    cond_df["std_rmse"] = cond_df["std_rmse"].fillna(0.0)

    # group breakdown: mean per-case RMSE within exit / no-exit groups, per condition
    grp = (pc.groupby(["cond", "exit_case"], as_index=False)
           .agg(mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"), n_cases=("target_case", "nunique")))

    seed_df.to_csv(out / "metrics" / "H6_S3_T9_T10_seed_metrics.csv", index=False)
    per_case_df.to_csv(out / "metrics" / "H6_S3_T9_T10_per_case_metrics.csv", index=False)
    cond_df.to_csv(out / "analysis" / "H6_S3_T9_T10_cond_metrics.csv", index=False)
    grp.to_csv(out / "analysis" / "H6_S3_T9_T10_group_metrics.csv", index=False)

    # figure: grouped bars [All, Exit-cases, No-exit] x [exit_aware, full]
    def cond_mean(cond, mask=None):
        d = pc[pc.cond == cond]
        if mask is not None:
            d = d[d.exit_case == mask]
        return d.rmse.mean()
    groups = ["All cases", "Exit-cases", "No-exit-cases"]
    ea = [cond_mean("T9_exit_aware"), cond_mean("T9_exit_aware", True), cond_mean("T9_exit_aware", False)]
    fu = [cond_mean("T10_full"), cond_mean("T10_full", True), cond_mean("T10_full", False)]
    x = np.arange(len(groups)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 5))
    ax.bar(x - w / 2, ea, w, label="T9 exit-aware (Exit removed)", color="#2ca02c", alpha=0.85)
    ax.bar(x + w / 2, fu, w, label="T10 full (Exit included)", color="#d62728", alpha=0.85)
    ax.axhline(PREFIX80_ANCHOR, color="#1f77b4", ls="--", lw=1.3, label=f"V1 prefix80={PREFIX80_ANCHOR:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("per-case mean RMSE (3-seed)")
    ax.set_title("H6.S3 T9-T10 — exit-aware (v2) vs full, by Exit presence")
    ax.legend(frameon=False, fontsize=8); ax.grid(True, axis="y", color="#eee", lw=0.5)
    fig.tight_layout(); fig.savefig(out / "figures" / "H6_S3_T9_T10_exit_presence.png", dpi=150); plt.close(fig)

    ea_rmse = float(cond_df[cond_df.cond == "T9_exit_aware"].mean_rmse.iloc[0])
    fu_rmse = float(cond_df[cond_df.cond == "T10_full"].mean_rmse.iloc[0])
    summary = {
        "experiment": "H6_S3_T9_T10_exit_aware_v2", "seeds": SEEDS, "gru_config": BEST,
        "exit_cases": exit_cases, "noexit_cases": noexit_cases,
        "prefix80_anchor": PREFIX80_ANCHOR, "steady5_T5": T5_STEADY5,
        "cond_overall": cond_df.to_dict("records"),
        "group_breakdown": grp.to_dict("records"),
        "exit_aware_minus_full": ea_rmse - fu_rmse,
        "exit_removal_helps": bool(ea_rmse < fu_rmse),
        "execution_dir": str(out),
    }
    (out / "analysis" / "H6_S3_T9_T10_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    log("\n=== RESULTS (3-seed LOCV) ===")
    for r in cond_df.itertuples(index=False):
        log(f"  {r.cond:<16} rmse={r.mean_rmse:.6f} (+/-{r.std_rmse:.6f}) r2={r.mean_r2:.4f} mae={r.mean_mae:.6f}")
    log("\n=== by Exit presence (per-case mean RMSE) ===")
    for r in grp.itertuples(index=False):
        grp_label = "exit-cases" if r.exit_case else "no-exit-cases"
        log(f"  {r.cond:<16} [{grp_label:<14} n={r.n_cases}]  rmse={r.mean_rmse:.6f} r2={r.mean_r2:.4f}")
    log(f"\nexit_aware - full = {ea_rmse - fu_rmse:+.6f}  (음수면 Exit 제거가 이득)")
    log(f"refs: V1 prefix80={PREFIX80_ANCHOR:.6f}  steady5-T5={T5_STEADY5:.6f}")
    (out / "logs" / "H6_S3_T9_T10_run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"EXECUTION_DIR={out}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prefix ratio → signal zone mapping analysis.

각 prefix ratio r%가 실제 신호 상에서 어느 구간(zone)에 해당하는지 분석.
  Zone: no-load | Entry [noload_end:idx_start] | Steady [idx_start:idx_end] | Exit [idx_end:]

Output: 각 ratio에서 cutoff가 어느 zone에 위치하는지 분포 + 통계
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS       = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
EXCLUDED_RUNS = {(2, 1), (12, 1)}
SEG_CSV       = ROOT / "datasets/cutting_segment_v2/seg_peng2026_steady5_exitfix_reverse_kurtosis.csv"
RATIOS        = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def parse_signal(value: object) -> np.ndarray:
    return np.nan_to_num(
        np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


def main() -> None:
    print("Loading data...", flush=True)
    signal_df = pd.read_csv(
        ROOT / "datasets/processed/mill_signal_data.csv",
        usecols=["case", "run", "smcAC"],
    )
    signal_df = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()

    seg_df = pd.read_csv(SEG_CSV)
    seg_df = seg_df[seg_df["case"].isin(CASE_SCOPE) & (seg_df["status"] == "labeled")]
    seg_idx = {
        (int(r.case), int(r.run)): {
            "idx_noload_end": int(r.idx_noload_end),
            "idx_start":      int(r.idx_start),
            "idx_end":        int(r.idx_end),
        }
        for r in seg_df.itertuples(index=False)
    }

    records: list[dict] = []
    for row in signal_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        if (case_id, run_id) in EXCLUDED_RUNS:
            continue
        seg = seg_idx.get((case_id, run_id))
        if seg is None:
            continue
        arr = parse_signal(row.smcAC)
        base_len = len(arr)
        if base_len == 0:
            continue

        nl_end = min(seg["idx_noload_end"], base_len)
        st_start = min(seg["idx_start"],    base_len)
        st_end   = min(seg["idx_end"],      base_len)

        records.append({
            "case": case_id, "run": run_id,
            "base_len": base_len,
            "nl_end":   nl_end,
            "st_start": st_start,
            "st_end":   st_end,
            "nl_frac":      nl_end   / base_len,
            "entry_frac":   st_start / base_len,
            "steady_frac":  st_end   / base_len,
            "noload_len":   nl_end,
            "entry_len":    max(0, st_start - nl_end),
            "steady_len":   max(0, st_end - st_start),
            "exit_len":     max(0, base_len - st_end),
        })

    df = pd.DataFrame(records)
    n = len(df)
    print(f"Runs with segment labels: {n}\n")

    # ── Segment length stats ──────────────────────────────────────────────────
    print("=== Segment Fraction Statistics (% of total signal) ===")
    print(f"{'Zone':>10}  {'mean%':>7}  {'std%':>7}  {'min%':>7}  {'max%':>7}")
    print("-" * 45)
    for zone, col in [("No-load",  "nl_frac"),
                      ("Entry end","entry_frac"),
                      ("Steady end","steady_frac")]:
        vals = df[col].values * 100
        print(f"{zone:>10}  {vals.mean():>6.1f}%  {vals.std():>6.1f}%  "
              f"{vals.min():>6.1f}%  {vals.max():>6.1f}%")

    # Derived zone fractions
    print()
    print(f"{'Zone':>10}  {'mean%':>7}  {'std%':>7}")
    print("-" * 30)
    for zone, col in [("no-load",  "noload_len"),
                      ("Entry",    "entry_len"),
                      ("Steady",   "steady_len"),
                      ("Exit",     "exit_len")]:
        frac = df[col].values / df["base_len"].values * 100
        print(f"{zone:>10}  {frac.mean():>6.1f}%  {frac.std():>6.1f}%")

    # ── Per-ratio zone placement ──────────────────────────────────────────────
    print("\n=== Per-ratio: Where Does the Prefix Cutoff Land? ===")
    print(f"{'Ratio':>6}  {'in NoLoad':>9}  {'in Entry':>8}  "
          f"{'in Steady':>9}  {'in Exit':>7}  "
          f"{'mean cutoff%':>12}  {'% into Entry':>12}  {'% into Steady':>13}")
    print("-" * 90)

    ratio_stats: list[dict] = []
    for r in RATIOS:
        cutoff_frac = r / 100.0  # cutoff as fraction of total signal
        in_noload = int(((cutoff_frac <= df["nl_frac"])).sum())
        in_entry  = int(((cutoff_frac > df["nl_frac"]) & (cutoff_frac <= df["entry_frac"])).sum())
        in_steady = int(((cutoff_frac > df["entry_frac"]) & (cutoff_frac <= df["steady_frac"])).sum())
        in_exit   = int((cutoff_frac > df["steady_frac"]).sum())

        # For runs where cutoff is inside Entry: how far into Entry?
        mask_entry = (cutoff_frac > df["nl_frac"]) & (cutoff_frac <= df["entry_frac"])
        if mask_entry.sum() > 0:
            sub = df[mask_entry]
            cutoff_abs  = (cutoff_frac * sub["base_len"]).values
            entry_pct   = np.clip(
                (cutoff_abs - sub["nl_end"].values) / sub["entry_len"].replace(0, 1).values * 100,
                0, 100
            )
            entry_into  = f"{entry_pct.mean():5.1f}% (n={mask_entry.sum()})"
        else:
            entry_into  = "   —  "

        # For runs where cutoff is inside Steady: how far into Steady?
        mask_steady = (cutoff_frac > df["entry_frac"]) & (cutoff_frac <= df["steady_frac"])
        if mask_steady.sum() > 0:
            sub = df[mask_steady]
            cutoff_abs  = (cutoff_frac * sub["base_len"]).values
            steady_pct  = np.clip(
                (cutoff_abs - sub["st_start"].values) / sub["steady_len"].replace(0, 1).values * 100,
                0, 100
            )
            steady_into = f"{steady_pct.mean():5.1f}% (n={mask_steady.sum()})"
        else:
            steady_into = "   —  "

        print(f"{r:>5}%  {in_noload:>5}/{n}    {in_entry:>4}/{n}    "
              f"{in_steady:>5}/{n}    {in_exit:>3}/{n}    "
              f"{r:>10.0f}%    {entry_into:>12}    {steady_into:>13}")

        ratio_stats.append({
            "ratio": r,
            "in_noload": in_noload, "in_entry": in_entry,
            "in_steady": in_steady, "in_exit":  in_exit,
        })

    # ── Per-ratio: meaningful content consumed ────────────────────────────────
    print("\n=== Meaningful Content Consumed at Each Ratio ===")
    print("(No-load 제외 후, Entry+Steady 대비 얼마나 커버하는가)")
    print(f"{'Ratio':>6}  {'Cutting covered%':>16}  {'Entry covered%':>14}  {'Steady covered%':>15}")
    print("-" * 60)
    for r in RATIOS:
        cutoff_frac = r / 100.0
        # Fraction of Entry+Steady covered by this cutoff (clipped)
        cutting_start_frac = df["nl_frac"].values   # noload_end / base_len
        cutting_end_frac   = df["steady_frac"].values  # idx_end / base_len
        cutting_len_frac   = cutting_end_frac - cutting_start_frac

        covered = np.clip(cutoff_frac - cutting_start_frac, 0, cutting_len_frac)
        cut_pct = np.where(cutting_len_frac > 0,
                           covered / cutting_len_frac * 100, 0.0)

        # Entry portion
        entry_start_frac = df["nl_frac"].values
        entry_end_frac   = df["entry_frac"].values
        entry_len_frac   = entry_end_frac - entry_start_frac
        e_covered = np.clip(cutoff_frac - entry_start_frac, 0, entry_len_frac)
        e_pct = np.where(entry_len_frac > 0, e_covered / entry_len_frac * 100, 0.0)

        # Steady portion
        steady_start_frac = df["entry_frac"].values
        steady_end_frac   = df["steady_frac"].values
        steady_len_frac   = steady_end_frac - steady_start_frac
        s_covered = np.clip(cutoff_frac - steady_start_frac, 0, steady_len_frac)
        s_pct = np.where(steady_len_frac > 0, s_covered / steady_len_frac * 100, 0.0)

        print(f"{r:>5}%  {cut_pct.mean():>13.1f}%    {e_pct.mean():>11.1f}%    {s_pct.mean():>12.1f}%")

    # ── Plot ─────────────────────────────────────────────────────────────────
    out_dir = ROOT / "experiments" / "analysis" / "prefix_timing"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: segment fraction distribution (box)
    ax = axes[0]
    frac_data = [
        df["nl_frac"].values * 100,
        (df["entry_frac"] - df["nl_frac"]).values * 100,  # entry zone width
        (df["steady_frac"] - df["entry_frac"]).values * 100,  # steady zone width
        (1 - df["steady_frac"]).values * 100,  # exit zone width
    ]
    ax.boxplot(frac_data, labels=["No-load", "Entry", "Steady", "Exit"])
    ax.set_ylabel("Zone fraction (% of total signal)")
    ax.set_title("Signal zone width distribution")
    ax.grid(True, alpha=0.3)
    # Overlay ratio lines
    for r in [50, 60, 70, 80, 90]:
        ax.axhline(r, color="crimson", linestyle=":", alpha=0.4, linewidth=0.8)

    # Right: stacked bar showing where each ratio cutoff lands
    ax = axes[1]
    rs = [str(r) for r in RATIOS]
    nl_counts  = [d["in_noload"] / n * 100 for d in ratio_stats]
    en_counts  = [d["in_entry"]  / n * 100 for d in ratio_stats]
    st_counts  = [d["in_steady"] / n * 100 for d in ratio_stats]
    ex_counts  = [d["in_exit"]   / n * 100 for d in ratio_stats]
    x = np.arange(len(RATIOS))
    w = 0.7
    b1 = ax.bar(x, nl_counts, w, label="No-load", color="lightgray")
    b2 = ax.bar(x, en_counts, w, bottom=nl_counts, label="Entry", color="steelblue")
    b3 = ax.bar(x, st_counts, w,
                bottom=[a+b for a,b in zip(nl_counts, en_counts)],
                label="Steady", color="darkorange")
    b4 = ax.bar(x, ex_counts, w,
                bottom=[a+b+c for a,b,c in zip(nl_counts, en_counts, st_counts)],
                label="Exit", color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}%" for r in RATIOS])
    ax.set_xlabel("Prefix ratio")
    ax.set_ylabel("% of runs with cutoff in zone")
    ax.set_title("Where does prefix cutoff land?")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Prefix Ratio → Signal Zone Mapping", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(out_dir / "prefix_timing.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "prefix_timing.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_dir}/prefix_timing.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""H9_S1: Input Ratio별 Feature Trajectory vs VB 분석.

각 prefix(10~100%)에서 추출한 Raw/Delta feature가 VB와 어떤 상관관계를 보이는지,
그리고 prefix 증가에 따라 어떻게 변화하는지 분석한다.

출력:
  - metrics/raw_corr_per_case.csv     : (case, sensor, stat, prefix, spearman_r)
  - metrics/delta_corr_per_case.csv   : (case, sensor, stat, prefix, spearman_r)
  - analysis/aggregate_raw_corr.csv   : (sensor, stat, prefix, mean_r, std_r, n_cases)
  - analysis/aggregate_delta_corr.csv : 동일 구조
  - plots/heatmap_raw_corr.png        : feature × prefix 상관 heatmap (raw)
  - plots/heatmap_delta_corr.png      : feature × prefix 상관 heatmap (delta)
  - plots/trajectory_*.png            : 케이스별 feature trajectory vs VB

Output:
  experiments/executions/H9/S1/{timestamp}_prefix_feature_trajectory/
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS       = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
STATS         = ["mean", "rms", "std", "peak"]
EXCLUDED_RUNS = {(2, 1), (12, 1)}
PREFIX_LIST   = [10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 100]
THRESH        = 1e6

# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_signal(value: object) -> np.ndarray:
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_stats(arr: np.ndarray) -> np.ndarray:
    """[mean, rms, std, peak]"""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros(1)
    return np.array([
        np.mean(arr),
        np.sqrt(np.mean(arr ** 2)),
        np.std(arr),
        np.abs(arr).max(),
    ], dtype=np.float64)


def preprocess_vb(process_df: pd.DataFrame) -> pd.DataFrame:
    df = process_df.copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"] == min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside")
    )
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)


def build_feature_cache(
    signal_df: pd.DataFrame, process_df: pd.DataFrame
) -> dict[tuple[int, int, int], np.ndarray]:
    """cache[(case, run, prefix_pct)] = feature_vec (24-dim raw)"""
    cache: dict[tuple[int, int, int], np.ndarray] = {}
    for row in process_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        sig_row = signal_df[
            (signal_df["case"] == case_id) & (signal_df["run"] == run_id)
        ]
        if sig_row.empty:
            continue
        sig_row = sig_row.iloc[0]
        arrays = {s: parse_signal(sig_row[s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        base_len = min(len(a) for a in arrays.values())
        for pct in PREFIX_LIST:
            end = max(1, int(np.ceil(base_len * pct / 100.0)))
            feats = np.concatenate([extract_stats(arrays[s][:end]) for s in SENSORS])
            cache[(case_id, run_id, pct)] = feats
    return cache


# ─── Correlation analysis ─────────────────────────────────────────────────────
def compute_correlations(
    cache: dict[tuple[int, int, int], np.ndarray],
    process_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (raw_corr_df, delta_corr_df)
    Each row: case, sensor, stat, prefix, spearman_r
    """
    feat_names = [f"{s}__{stat}" for s in SENSORS for stat in STATS]

    # Find first non-excluded run per case
    first_run: dict[int, int] = {}
    for case_id in CASE_SCOPE:
        runs = sorted(
            r for (c, r, _) in cache
            if c == case_id and (c, r) not in EXCLUDED_RUNS
        )
        if runs:
            first_run[case_id] = runs[0]

    raw_rows:   list[dict] = []
    delta_rows: list[dict] = []

    for case_id in CASE_SCOPE:
        case_df = process_df[process_df["case"] == case_id].sort_values("run")
        vb_series = {int(r.run): float(r.VB) for r in case_df.itertuples()}
        ref_run = first_run.get(case_id)

        for pct in PREFIX_LIST:
            # Collect runs with valid features for this prefix
            run_ids = sorted(
                r for (c, r, p) in cache
                if c == case_id and p == pct and (c, r) not in EXCLUDED_RUNS
            )
            if len(run_ids) < 3:
                continue

            raw_mat   = np.stack([cache[(case_id, r, pct)] for r in run_ids])
            vb_vec    = np.array([vb_series[r] for r in run_ids if r in vb_series])

            # Align: only runs that have both cache entry and VB
            valid_runs = [r for r in run_ids if r in vb_series]
            raw_mat    = np.stack([cache[(case_id, r, pct)] for r in valid_runs])
            vb_vec     = np.array([vb_series[r] for r in valid_runs])

            # Delta features
            ref_vec = cache.get((case_id, ref_run, pct)) if ref_run else None
            delta_mat = raw_mat - ref_vec if ref_vec is not None else None

            for feat_idx, feat_name in enumerate(feat_names):
                sensor, stat = feat_name.split("__")
                raw_col = raw_mat[:, feat_idx]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r_raw, _ = spearmanr(raw_col, vb_vec)
                raw_rows.append({
                    "case": case_id, "sensor": sensor, "stat": stat,
                    "prefix": pct, "spearman_r": float(r_raw) if np.isfinite(r_raw) else 0.0,
                })

                if delta_mat is not None:
                    delta_col = delta_mat[:, feat_idx]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r_delta, _ = spearmanr(delta_col, vb_vec)
                    delta_rows.append({
                        "case": case_id, "sensor": sensor, "stat": stat,
                        "prefix": pct, "spearman_r": float(r_delta) if np.isfinite(r_delta) else 0.0,
                    })

    return pd.DataFrame(raw_rows), pd.DataFrame(delta_rows)


# ─── Plots ────────────────────────────────────────────────────────────────────
SENSOR_ABBR = {
    "smcAC": "AC",
    "smcDC": "DC",
    "vib_table": "vib_T",
    "vib_spindle": "vib_S",
    "AE_table": "AE_T",
    "AE_spindle": "AE_S",
}


def plot_corr_heatmap(
    corr_df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    agg = (
        corr_df.groupby(["sensor", "stat", "prefix"])["spearman_r"]
        .mean()
        .reset_index()
    )
    feat_labels = [f"{SENSOR_ABBR[s]}_{t}" for s in SENSORS for t in STATS]
    feat_keys   = [(s, t) for s in SENSORS for t in STATS]
    prefix_vals = sorted(agg["prefix"].unique())

    mat = np.full((len(feat_keys), len(prefix_vals)), np.nan)
    for i, (sensor, stat) in enumerate(feat_keys):
        for j, pct in enumerate(prefix_vals):
            row = agg[(agg["sensor"] == sensor) & (agg["stat"] == stat) & (agg["prefix"] == pct)]
            if not row.empty:
                mat[i, j] = row["spearman_r"].iloc[0]

    fig, ax = plt.subplots(figsize=(len(prefix_vals) * 0.7 + 2, len(feat_keys) * 0.38 + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Spearman r")
    ax.set_xticks(range(len(prefix_vals)))
    ax.set_xticklabels([f"{p}%" for p in prefix_vals], fontsize=8)
    ax.set_yticks(range(len(feat_labels)))
    ax.set_yticklabels(feat_labels, fontsize=7)
    ax.set_xlabel("Input Ratio (prefix %)")
    ax.set_ylabel("Feature")
    ax.set_title(title)

    # Annotate values
    for i in range(len(feat_keys)):
        for j in range(len(prefix_vals)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color="black" if abs(v) < 0.7 else "white")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_prefix_effect_lines(
    corr_df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    """Line plot: x=prefix, y=mean Spearman r, one line per sensor (aggregated over stats)."""
    agg = (
        corr_df.groupby(["sensor", "prefix"])["spearman_r"]
        .mean()
        .reset_index()
    )
    prefix_vals = sorted(agg["prefix"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(SENSORS)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for sensor, color in zip(SENSORS, colors):
        sub = agg[agg["sensor"] == sensor].sort_values("prefix")
        ax.plot(sub["prefix"], sub["spearman_r"], marker="o", label=SENSOR_ABBR[sensor],
                color=color, linewidth=1.8, markersize=5)

    ax.axvline(80, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="80% (best prefix)")
    ax.set_xlabel("Input Ratio (prefix %)")
    ax.set_ylabel("Mean Spearman r (averaged over stat × case)")
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stat_prefix_lines(
    corr_df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    """Line plot: x=prefix, y=mean Spearman r, grid of (sensor × stat)."""
    agg = (
        corr_df.groupby(["sensor", "stat", "prefix"])["spearman_r"]
        .mean()
        .reset_index()
    )
    prefix_vals = sorted(agg["prefix"].unique())
    stat_colors = {"mean": "tab:blue", "rms": "tab:orange", "std": "tab:green", "peak": "tab:red"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, sensor in zip(axes.flat, SENSORS):
        for stat in STATS:
            sub = agg[(agg["sensor"] == sensor) & (agg["stat"] == stat)].sort_values("prefix")
            ax.plot(sub["prefix"], sub["spearman_r"], marker="o",
                    label=stat, color=stat_colors[stat], linewidth=1.5, markersize=4)
        ax.axvline(80, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(SENSOR_ABBR[sensor], fontsize=10, fontweight="bold")
        ax.set_xlabel("prefix %", fontsize=8)
        ax.set_ylabel("Mean Spearman r", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trajectory_grid(
    cache: dict[tuple[int, int, int], np.ndarray],
    process_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """For key sensors × stats, show run trajectory at selected prefixes vs VB."""
    selected_prefixes = [20, 50, 80, 100]
    colors = {20: "tab:blue", 50: "tab:orange", 80: "tab:green", 100: "tab:red"}
    key_pairs = [("smcAC", "std"), ("smcDC", "mean"), ("AE_table", "mean"), ("AE_spindle", "std")]

    for case_id in [1, 2, 9, 12]:
        case_df = process_df[process_df["case"] == case_id].sort_values("run").reset_index(drop=True)
        run_ids = [int(r.run) for r in case_df.itertuples() if (case_id, int(r.run)) not in EXCLUDED_RUNS]
        vb_map  = {int(r.run): float(r.VB) for r in case_df.itertuples()}

        fig, axes = plt.subplots(len(key_pairs), 1, figsize=(10, 3 * len(key_pairs)),
                                  sharex=True)
        for ax, (sensor, stat) in zip(axes, key_pairs):
            feat_idx = SENSORS.index(sensor) * 4 + STATS.index(stat)
            vb_vals  = [vb_map[r] for r in run_ids if r in vb_map]
            ax2 = ax.twinx()
            ax2.plot(range(len(run_ids)), vb_vals, "k--", linewidth=2, alpha=0.5, label="VB (mm)")
            ax2.set_ylabel("VB (mm)", color="gray", fontsize=8)
            ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)

            for pct in selected_prefixes:
                feat_vals = [
                    cache.get((case_id, r, pct), np.full(24, np.nan))[feat_idx]
                    for r in run_ids if r in vb_map
                ]
                ax.plot(range(len(run_ids)), feat_vals, marker="o", markersize=3,
                        linewidth=1.5, color=colors[pct], label=f"{pct}%")

            ax.set_ylabel(f"{SENSOR_ABBR[sensor]}_{stat}", fontsize=8)
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Run index")
        fig.suptitle(f"Case {case_id}: Feature Trajectory vs VB @ different prefix ratios", fontsize=11)
        plt.tight_layout()
        fig.savefig(out_dir / f"trajectory_case{case_id}.png", dpi=130)
        plt.close(fig)


def plot_topk_features(corr_df: pd.DataFrame, title: str, out_path: Path, topk: int = 6) -> None:
    """Top-K features by mean |Spearman r| at 80%, show their prefix curves."""
    at80 = corr_df[corr_df["prefix"] == 80].copy()
    agg80 = (
        at80.groupby(["sensor", "stat"])["spearman_r"]
        .agg(mean_r="mean", abs_mean=lambda x: x.abs().mean())
        .reset_index()
        .sort_values("abs_mean", ascending=False)
    )
    top_feats = [(row["sensor"], row["stat"]) for _, row in agg80.head(topk).iterrows()]

    agg_all = corr_df.groupby(["sensor", "stat", "prefix"])["spearman_r"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    for sensor, stat in top_feats:
        sub = agg_all[(agg_all["sensor"] == sensor) & (agg_all["stat"] == stat)].sort_values("prefix")
        label = f"{SENSOR_ABBR[sensor]}_{stat} (r80={agg80[(agg80['sensor']==sensor)&(agg80['stat']==stat)]['mean_r'].iloc[0]:.2f})"
        ax.plot(sub["prefix"], sub["spearman_r"], marker="o", linewidth=1.8,
                markersize=5, label=label)

    ax.axvline(80, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="80%")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Input Ratio (prefix %)")
    ax.set_ylabel("Mean Spearman r (across cases)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "H9" / "S1" / f"{ts}_prefix_feature_trajectory"
    for sub in ["metrics", "analysis", "plots", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H9_S1: Input Ratio별 Feature Trajectory vs VB 분석 ===")
    log(f"PREFIX_LIST = {PREFIX_LIST}")
    log(f"SENSORS     = {SENSORS}")
    log(f"STATS       = {STATS}")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess_vb(process_df)

    log(f"Cases: {CASE_SCOPE} (n={len(CASE_SCOPE)})")
    log(f"Total clean runs: {len(proc_clean)}")

    log("\nBuilding feature cache (this may take a minute)...")
    cache = build_feature_cache(signal_df, proc_clean)
    log(f"Cache entries: {len(cache)}")

    log("\nComputing Spearman correlations...")
    raw_corr_df, delta_corr_df = compute_correlations(cache, proc_clean)
    log(f"Raw corr rows: {len(raw_corr_df)}")
    log(f"Delta corr rows: {len(delta_corr_df)}")

    # Save per-case correlations
    raw_corr_df.to_csv(out_dir / "metrics" / "raw_corr_per_case.csv", index=False)
    delta_corr_df.to_csv(out_dir / "metrics" / "delta_corr_per_case.csv", index=False)

    # Aggregate
    agg_raw = (
        raw_corr_df.groupby(["sensor", "stat", "prefix"])["spearman_r"]
        .agg(mean_r="mean", std_r="std", n_cases="count")
        .reset_index()
    )
    agg_delta = (
        delta_corr_df.groupby(["sensor", "stat", "prefix"])["spearman_r"]
        .agg(mean_r="mean", std_r="std", n_cases="count")
        .reset_index()
    )
    agg_raw.to_csv(out_dir / "analysis" / "aggregate_raw_corr.csv", index=False)
    agg_delta.to_csv(out_dir / "analysis" / "aggregate_delta_corr.csv", index=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    log("\n=== TOP FEATURES BY |SPEARMAN r| AT 80% PREFIX (RAW) ===")
    at80_raw = agg_raw[agg_raw["prefix"] == 80].copy()
    at80_raw["abs_r"] = at80_raw["mean_r"].abs()
    top10_raw = at80_raw.sort_values("abs_r", ascending=False).head(10)
    log(f"{'Sensor':<14} {'Stat':<6} {'mean_r':>8} {'std_r':>7}")
    log("-" * 40)
    for _, row in top10_raw.iterrows():
        log(f"{row['sensor']:<14} {row['stat']:<6} {row['mean_r']:>8.4f} {row['std_r']:>7.4f}")

    log("\n=== TOP FEATURES BY |SPEARMAN r| AT 80% PREFIX (DELTA) ===")
    at80_delta = agg_delta[agg_delta["prefix"] == 80].copy()
    at80_delta["abs_r"] = at80_delta["mean_r"].abs()
    top10_delta = at80_delta.sort_values("abs_r", ascending=False).head(10)
    log(f"{'Sensor':<14} {'Stat':<6} {'mean_r':>8} {'std_r':>7}")
    log("-" * 40)
    for _, row in top10_delta.iterrows():
        log(f"{row['sensor']:<14} {row['stat']:<6} {row['mean_r']:>8.4f} {row['std_r']:>7.4f}")

    log("\n=== MEAN |SPEARMAN r| BY PREFIX (averaged over all features & cases) [RAW] ===")
    per_prefix_raw = raw_corr_df.groupby("prefix")["spearman_r"].agg(
        mean_abs=lambda x: x.abs().mean(), mean_signed="mean"
    ).reset_index().sort_values("prefix")
    log(f"{'prefix':>8} {'|r| mean':>10} {'r signed':>10}")
    log("-" * 32)
    for _, row in per_prefix_raw.iterrows():
        log(f"{row['prefix']:>7}% {row['mean_abs']:>10.4f} {row['mean_signed']:>10.4f}")

    log("\n=== MEAN |SPEARMAN r| BY PREFIX (DELTA) ===")
    per_prefix_delta = delta_corr_df.groupby("prefix")["spearman_r"].agg(
        mean_abs=lambda x: x.abs().mean(), mean_signed="mean"
    ).reset_index().sort_values("prefix")
    log(f"{'prefix':>8} {'|r| mean':>10} {'r signed':>10}")
    log("-" * 32)
    for _, row in per_prefix_delta.iterrows():
        log(f"{row['prefix']:>7}% {row['mean_abs']:>10.4f} {row['mean_signed']:>10.4f}")

    # ── Per-sensor breakdown at key prefix (80%) ──────────────────────────────
    log("\n=== PER-SENSOR MEAN r AT KEY PREFIXES [RAW] ===")
    sensor_prefix = raw_corr_df.groupby(["sensor", "prefix"])["spearman_r"].mean().unstack("prefix")
    key_cols = [p for p in [50, 70, 80, 90, 100] if p in sensor_prefix.columns]
    log(f"{'Sensor':<14}" + "".join(f"{p:>8}%" for p in key_cols))
    log("-" * (14 + len(key_cols) * 9))
    for sensor in SENSORS:
        if sensor not in sensor_prefix.index:
            continue
        vals = sensor_prefix.loc[sensor, key_cols]
        log(f"{sensor:<14}" + "".join(f"{v:>9.4f}" for v in vals))

    log("\n=== PER-SENSOR MEAN r AT KEY PREFIXES [DELTA] ===")
    sensor_prefix_d = delta_corr_df.groupby(["sensor", "prefix"])["spearman_r"].mean().unstack("prefix")
    log(f"{'Sensor':<14}" + "".join(f"{p:>8}%" for p in key_cols))
    log("-" * (14 + len(key_cols) * 9))
    for sensor in SENSORS:
        if sensor not in sensor_prefix_d.index:
            continue
        vals = sensor_prefix_d.loc[sensor, key_cols]
        log(f"{sensor:<14}" + "".join(f"{v:>9.4f}" for v in vals))

    # ── Plots ─────────────────────────────────────────────────────────────────
    log("\nGenerating plots...")
    plot_corr_heatmap(
        raw_corr_df,
        "Raw Feature × Input Ratio — Mean Spearman r with VB",
        out_dir / "plots" / "heatmap_raw_corr.png",
    )
    plot_corr_heatmap(
        delta_corr_df,
        "Delta Feature × Input Ratio — Mean Spearman r with VB",
        out_dir / "plots" / "heatmap_delta_corr.png",
    )
    plot_prefix_effect_lines(
        raw_corr_df,
        "Input Ratio Effect on Feature-VB Correlation (Raw, per sensor)",
        out_dir / "plots" / "prefix_effect_raw_by_sensor.png",
    )
    plot_prefix_effect_lines(
        delta_corr_df,
        "Input Ratio Effect on Feature-VB Correlation (Delta, per sensor)",
        out_dir / "plots" / "prefix_effect_delta_by_sensor.png",
    )
    plot_stat_prefix_lines(
        raw_corr_df,
        "Raw Feature-VB Correlation by Sensor × Stat across Input Ratios",
        out_dir / "plots" / "stat_prefix_raw.png",
    )
    plot_stat_prefix_lines(
        delta_corr_df,
        "Delta Feature-VB Correlation by Sensor × Stat across Input Ratios",
        out_dir / "plots" / "stat_prefix_delta.png",
    )
    plot_topk_features(
        raw_corr_df,
        "Top-6 Raw Features: Spearman r with VB across Input Ratios",
        out_dir / "plots" / "top6_raw_prefix_curve.png",
        topk=6,
    )
    plot_topk_features(
        delta_corr_df,
        "Top-6 Delta Features: Spearman r with VB across Input Ratios",
        out_dir / "plots" / "top6_delta_prefix_curve.png",
        topk=6,
    )
    plot_trajectory_grid(cache, proc_clean, out_dir / "plots")

    # ── Save summary JSON ──────────────────────────────────────────────────────
    summary = {
        "experiment": "H9_S1_prefix_feature_trajectory",
        "prefix_list": PREFIX_LIST,
        "sensors": SENSORS,
        "stats": STATS,
        "n_cases": len(CASE_SCOPE),
        "top10_raw_at_80pct": top10_raw[["sensor", "stat", "mean_r"]].to_dict("records"),
        "top10_delta_at_80pct": top10_delta[["sensor", "stat", "mean_r"]].to_dict("records"),
        "per_prefix_raw_mean_abs_r": per_prefix_raw[["prefix", "mean_abs"]].to_dict("records"),
        "per_prefix_delta_mean_abs_r": per_prefix_delta[["prefix", "mean_abs"]].to_dict("records"),
        "execution_dir": str(out_dir),
    }
    (out_dir / "analysis" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    (out_dir / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

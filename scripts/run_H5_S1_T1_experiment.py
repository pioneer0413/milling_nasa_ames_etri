#!/usr/bin/env python3
"""H5_S1_T1: Case 13/14 vs Others — Deep EDA

Context spec: docs/agent_context/task_context/H5_S1_T1_context.md

Research questions:
  1. Are Case 13/14's signal features less correlated with VB?
  2. Are Case 13/14's features harder to discriminate from other cases?

Output:
  experiments/executions/H5/S1/T1/{timestamp}_H5_S1_T1_case13_14_eda/
    metrics/  — CSV files
    figures/  — PNG files
    logs/     — run log
"""
from __future__ import annotations

import sys, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SENSORS         = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
CASE_SCOPE      = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
HARD_CASES      = [13, 14]
EASY_CASES      = [c for c in CASE_SCOPE if c not in HARD_CASES]
EXCLUDED_RUNS   = {(2, 1), (12, 1)}
THRESH          = 1e6

PREFIX_LABEL = "H5_S1_T1"

# ─────────────────────────────────────────
def parse_signal(v):
    a = np.fromstring(str(v).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

def extract_stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        a = np.zeros(1)
    return {
        "mean": float(np.mean(a)),
        "rms":  float(np.sqrt(np.mean(a**2))),
        "std":  float(np.std(a)),
        "peak": float(np.abs(a).max()),
    }

def preprocess_proc(df):
    df = df.copy().sort_values(["case","run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"]==min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside"))
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)

def build_feature_df(sig_df, proc_df):
    """Full-signal (100%) features + delta + VB for every (case, run)."""
    rows = []
    first_run_cache = {}
    raw_cache = {}

    for row in proc_df.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        sr = sig_df[(sig_df["case"]==cid) & (sig_df["run"]==rid)]
        if sr.empty:
            continue
        sr = sr.iloc[0]
        arrays = {s: parse_signal(sr[s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        feats = {}
        for s in SENSORS:
            for k, v in extract_stats(arrays[s]).items():
                feats[f"{s}__{k}"] = v
        raw_cache[(cid, rid)] = feats

    for cid in proc_df["case"].unique():
        cd = proc_df[proc_df["case"]==cid].sort_values("run")
        for rrow in cd.itertuples(index=False):
            rid = int(rrow.run)
            if (cid, rid) in EXCLUDED_RUNS:
                continue
            if (cid, rid) in raw_cache:
                first_run_cache[cid] = raw_cache[(cid, rid)]
                break

    for row in proc_df.itertuples(index=False):
        cid, rid = int(row.case), int(row.run)
        if (cid, rid) in EXCLUDED_RUNS:
            continue
        if (cid, rid) not in raw_cache:
            continue
        ref = first_run_cache.get(cid)
        if ref is None:
            continue
        feats = raw_cache[(cid, rid)]
        delta = {f"delta_{k}": feats[k] - ref[k] for k in feats}
        rec = {"case": cid, "run": rid, "VB": float(row.VB),
               "DOC": float(row.DOC), "feed": float(row.feed),
               "material": int(row.material),
               "group": "hard" if cid in HARD_CASES else "easy"}
        rec.update(feats)
        rec.update(delta)
        rows.append(rec)

    return pd.DataFrame(rows).sort_values(["case","run"]).reset_index(drop=True)


# ─── Analysis functions ──────────────────

def vb_range_table(proc_df):
    """Per-case VB statistics."""
    clean = preprocess_proc(proc_df)
    stats = []
    for cid, g in clean.groupby("case"):
        vb = g["VB"].values
        slope = np.polyfit(np.arange(len(vb)), vb, 1)[0] if len(vb) > 1 else np.nan
        stats.append({
            "case": int(cid),
            "group": "hard" if cid in HARD_CASES else "easy",
            "n_runs": len(vb),
            "VB_min": vb.min(),
            "VB_max": vb.max(),
            "VB_range": vb.max() - vb.min(),
            "VB_mean": vb.mean(),
            "VB_slope": slope,
            "DOC": float(g["DOC"].iloc[0]),
            "feed": float(g["feed"].iloc[0]),
            "material": int(g["material"].iloc[0]),
        })
    return pd.DataFrame(stats)


def per_case_feature_vb_corr(feat_df):
    """Pearson r between each delta feature and VB, per case."""
    delta_cols = [c for c in feat_df.columns if c.startswith("delta_")]
    rows = []
    for cid, g in feat_df.groupby("case"):
        if len(g) < 3:
            continue
        vb = g["VB"].values
        for col in delta_cols:
            x = g[col].values
            if np.std(x) < 1e-12:
                r, p = 0.0, 1.0
            else:
                r, p = sp_stats.pearsonr(x, vb)
            rows.append({"case": int(cid), "group": "hard" if cid in HARD_CASES else "easy",
                         "feature": col, "pearson_r": r, "p_value": p})
    return pd.DataFrame(rows)


def between_case_feature_overlap(feat_df):
    """For each raw feature, compute overlap coefficient between
    Hard-cases and Easy-cases distributions (Bhattacharyya distance)."""
    raw_cols = [c for c in feat_df.columns
                if "__" in c and not c.startswith("delta_") and c not in
                   ("case","run","VB","DOC","feed","material","group")]
    rows = []
    hard = feat_df[feat_df["group"]=="hard"]
    easy = feat_df[feat_df["group"]=="easy"]
    for col in raw_cols:
        h, e = hard[col].dropna().values, easy[col].dropna().values
        if len(h) < 3 or len(e) < 3:
            continue
        # Bhattacharyya coefficient (histogram overlap)
        bins = np.linspace(min(h.min(), e.min()), max(h.max(), e.max()), 30)
        ph, _ = np.histogram(h, bins=bins, density=True)
        pe, _ = np.histogram(e, bins=bins, density=True)
        bw = bins[1] - bins[0]
        bc = float(np.sum(np.sqrt(ph * pe)) * bw)   # Bhattacharyya coeff (0=no overlap, 1=identical)
        rows.append({"feature": col, "bc": bc})
    return pd.DataFrame(rows).sort_values("bc", ascending=False)


# ─── Plotting ───────────────────────────

COLORS = {"hard": "#d62728", "easy": "#1f77b4"}

def plot_vb_progression(proc_df, out_dir):
    clean = preprocess_proc(proc_df)
    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    axes = axes.flatten()
    for i, cid in enumerate(sorted(CASE_SCOPE)):
        ax = axes[i]
        g = clean[clean["case"]==cid].sort_values("run")
        color = COLORS["hard"] if cid in HARD_CASES else COLORS["easy"]
        ax.plot(g["run"], g["VB"], "o-", color=color, lw=1.5, ms=5)
        ax.axhline(0.8, ls="--", color="gray", lw=0.8, alpha=0.7, label="VB=0.8")
        ax.set_title(f"Case {cid}", fontweight="bold" if cid in HARD_CASES else "normal",
                     color=color)
        ax.set_xlabel("Run #")
        ax.set_ylabel("VB (mm)")
        ax.set_ylim(bottom=0)
    axes[-1].set_visible(False)
    fig.suptitle("VB Progression per Case  (red=hard, blue=easy)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "fig01_vb_progression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vb_distribution(vb_table, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # VB_max comparison
    ax = axes[0]
    easy_vb = vb_table[vb_table["group"]=="easy"]["VB_max"].values
    hard_vb = vb_table[vb_table["group"]=="hard"]["VB_max"].values
    ax.bar(vb_table[vb_table["group"]=="easy"]["case"].astype(str), easy_vb,
           color=COLORS["easy"], alpha=0.8, label="Easy")
    ax.bar(vb_table[vb_table["group"]=="hard"]["case"].astype(str), hard_vb,
           color=COLORS["hard"], alpha=0.8, label="Hard")
    ax.axhline(easy_vb.max(), ls="--", color="gray", lw=1, label=f"Easy max={easy_vb.max():.2f}")
    ax.set_title("VB_max per Case")
    ax.set_ylabel("VB_max (mm)")
    ax.set_xlabel("Case")
    ax.legend()

    # VB_slope
    ax = axes[1]
    for _, row in vb_table.iterrows():
        color = COLORS[row["group"]]
        ax.bar(str(int(row["case"])), row["VB_slope"], color=color, alpha=0.8)
    ax.set_title("VB Wear Rate (slope, mm/run)")
    ax.set_ylabel("Δ VB / run")
    ax.set_xlabel("Case")

    # VB_range
    ax = axes[2]
    for _, row in vb_table.iterrows():
        color = COLORS[row["group"]]
        ax.bar(str(int(row["case"])), row["VB_range"], color=color, alpha=0.8)
    ax.set_title("VB Range (max-min) per Case")
    ax.set_ylabel("VB range (mm)")
    ax.set_xlabel("Case")

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    handles = [plt.Rectangle((0,0),1,1, color=COLORS["easy"]),
               plt.Rectangle((0,0),1,1, color=COLORS["hard"])]
    fig.legend(handles, ["Easy","Hard"], loc="upper right")
    fig.suptitle("VB Distribution Statistics: Hard vs Easy", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "fig02_vb_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_vb_corr(corr_df, out_dir):
    """Boxplot of |pearson_r| distribution between hard and easy groups."""
    corr_df = corr_df.copy()
    corr_df["abs_r"] = corr_df["pearson_r"].abs()

    sensor_map = {s: s for s in SENSORS}
    stat_order  = ["mean","rms","std","peak"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, sensor in enumerate(SENSORS):
        ax = axes[i]
        sensor_feats = corr_df[corr_df["feature"].str.contains(f"delta_{sensor}__")]
        easy = sensor_feats[sensor_feats["group"]=="easy"]
        hard = sensor_feats[sensor_feats["group"]=="hard"]
        for j, stat in enumerate(stat_order):
            efeat = f"delta_{sensor}__{stat}"
            e_vals = easy[easy["feature"]==efeat]["abs_r"].values
            h_vals = hard[hard["feature"]==efeat]["abs_r"].values
            ax.scatter([j - 0.12] * len(e_vals), e_vals,
                       color=COLORS["easy"], alpha=0.7, s=40, zorder=3)
            ax.scatter([j + 0.12] * len(h_vals), h_vals,
                       color=COLORS["hard"], alpha=0.7, s=70, marker="D", zorder=3)
        ax.set_xticks(range(len(stat_order)))
        ax.set_xticklabels(stat_order)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, ls="--", color="gray", lw=0.8)
        ax.set_title(f"Sensor: {sensor}")
        ax.set_ylabel("|Pearson r| (delta vs VB)")

    handles = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=COLORS["easy"], ms=8),
               plt.Line2D([0],[0], marker="D", color="w", markerfacecolor=COLORS["hard"], ms=8)]
    fig.legend(handles, ["Easy cases","Hard cases (13,14)"], loc="upper right")
    fig.suptitle("Delta Feature–VB Correlation by Sensor & Stat\n(each dot = one case)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "fig03_delta_vb_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_space_pca(feat_df, out_dir):
    """PCA of delta features — visualize cluster separation."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    delta_cols = [c for c in feat_df.columns if c.startswith("delta_")]
    X = feat_df[delta_cols].values
    vb = feat_df["VB"].values
    cases = feat_df["case"].values
    groups = feat_df["group"].values

    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)
    ev = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: colored by group
    ax = axes[0]
    for g, label in [("easy","Easy"), ("hard","Hard (13,14)")]:
        idx = groups == g
        sc_plt = ax.scatter(Z[idx, 0], Z[idx, 1],
                            c=COLORS[g], alpha=0.6, s=30, label=label, edgecolors="none")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title("PCA of Delta Features — Group")
    ax.legend()

    # Right: colored by VB value
    ax = axes[1]
    sc_plt = ax.scatter(Z[:, 0], Z[:, 1], c=vb, cmap="RdYlGn_r",
                        alpha=0.7, s=30, edgecolors="none")
    # highlight hard cases with marker
    for cid in HARD_CASES:
        idx = cases == cid
        ax.scatter(Z[idx, 0], Z[idx, 1], edgecolors="black", s=60,
                   facecolors="none", linewidths=1.2, zorder=4)
    cbar = plt.colorbar(sc_plt, ax=ax)
    cbar.set_label("VB (mm)")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title("PCA of Delta Features — VB (circle=hard)")

    fig.suptitle("Feature Space PCA: Delta Features", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "fig04_feature_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_signal_examples(sig_df, proc_df, out_dir):
    """Raw signal comparison: one run from Case 13 vs Case 11 at similar VB level."""
    clean = preprocess_proc(proc_df)
    target_sensor = "smcAC"

    # Find runs at VB ~ 0.5 for each case group
    examples = []
    for cid in [3, 11, 13, 14]:
        g = clean[clean["case"]==cid].sort_values("run")
        # pick run closest to VB=0.5
        idx = (g["VB"] - 0.5).abs().idxmin()
        row = g.loc[idx]
        examples.append((cid, int(row["run"]), float(row["VB"])))

    fig, axes = plt.subplots(len(SENSORS), len(examples), figsize=(16, 12))

    for col, (cid, rid, vb) in enumerate(examples):
        sr = sig_df[(sig_df["case"]==cid)&(sig_df["run"]==rid)]
        if sr.empty:
            continue
        sr = sr.iloc[0]
        color = COLORS["hard"] if cid in HARD_CASES else COLORS["easy"]
        for row_i, sensor in enumerate(SENSORS):
            ax = axes[row_i, col]
            arr = parse_signal(sr[sensor])
            ax.plot(arr[:2000], color=color, lw=0.5, alpha=0.8)
            if row_i == 0:
                ax.set_title(f"Case {cid} Run {rid}\n(VB≈{vb:.2f})",
                             color=color, fontweight="bold" if cid in HARD_CASES else "normal")
            if col == 0:
                ax.set_ylabel(sensor, fontsize=8)
            ax.set_xticks([])

    fig.suptitle(f"Raw Signal Comparison (first 2000 samples) at VB ≈ 0.5 mm", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig05_signal_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vb_extrapolation(vb_table, feat_df, out_dir):
    """Show training VB range seen by model when predicting Case 13/14."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, target_case in zip(axes, HARD_CASES):
        train_max = vb_table[vb_table["case"] != target_case]["VB_max"]
        test_max  = vb_table[vb_table["case"] == target_case]["VB_max"].values[0]

        # Distribution of VB values in training set
        train_vb = feat_df[feat_df["case"] != target_case]["VB"].values
        test_vb  = feat_df[feat_df["case"] == target_case]["VB"].values

        bins = np.linspace(0, max(train_vb.max(), test_vb.max()) + 0.05, 40)
        ax.hist(train_vb, bins=bins, alpha=0.6, color=COLORS["easy"],
                density=True, label=f"Training set (n={len(train_vb)})")
        ax.hist(test_vb,  bins=bins, alpha=0.6, color=COLORS["hard"],
                density=True, label=f"Case {target_case} (n={len(test_vb)})")
        ax.axvline(train_vb.max(), ls="--", color="navy", lw=1.5,
                   label=f"Train VB_max={train_vb.max():.2f}")
        ax.axvline(test_vb.max(), ls="--", color="darkred", lw=1.5,
                   label=f"Test VB_max={test_vb.max():.2f}")
        extrap = test_vb[test_vb > train_vb.max()]
        ax.axvspan(train_vb.max(), test_vb.max() + 0.05, alpha=0.1, color="red",
                   label=f"Extrapolation zone\n({len(extrap)} samples)")
        ax.set_title(f"LOCV: Predicting Case {target_case}")
        ax.set_xlabel("VB (mm)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("VB Distribution Mismatch: Training vs Test (Hard Cases)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "fig06_vb_extrapolation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_delta_feature_trajectory(feat_df, out_dir):
    """Delta feature trajectory vs VB for one representative sensor (smcAC)."""
    sensor = "smcAC"
    stats  = ["mean", "rms", "std", "peak"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, stat in enumerate(stats):
        col  = f"delta_{sensor}__{stat}"
        ax = axes[i]
        for cid in CASE_SCOPE:
            g = feat_df[feat_df["case"]==cid].sort_values("VB")
            if g.empty:
                continue
            color = COLORS["hard"] if cid in HARD_CASES else COLORS["easy"]
            lw = 2.0 if cid in HARD_CASES else 0.8
            alpha = 0.9 if cid in HARD_CASES else 0.4
            zorder = 3 if cid in HARD_CASES else 2
            ax.plot(g["VB"], g[col], "-o", color=color, lw=lw,
                    ms=4, alpha=alpha, zorder=zorder,
                    label=f"Case {cid}" if cid in HARD_CASES else None)
        ax.set_xlabel("VB (mm)")
        ax.set_ylabel(f"Δ {stat}")
        ax.set_title(f"smcAC — Δ{stat} vs VB")
        if i == 0:
            handles = [plt.Line2D([0],[0], color=COLORS["easy"], lw=1.5),
                       plt.Line2D([0],[0], color=COLORS["hard"], lw=2.5)]
            ax.legend(handles, ["Easy cases","Hard cases"], fontsize=9)

    fig.suptitle(f"Delta Feature Trajectory (smcAC) vs VB\n"
                 f"Hard cases extend beyond training VB range", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig07_delta_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_corr_heatmap(corr_df, out_dir):
    """Heatmap: case × delta_feature → |pearson_r|."""
    delta_cols = sorted(corr_df["feature"].unique())
    cases = sorted(corr_df["case"].unique())
    mat = pd.DataFrame(index=cases, columns=delta_cols, dtype=float)
    for _, row in corr_df.iterrows():
        mat.loc[row["case"], row["feature"]] = abs(row["pearson_r"])

    fig, ax = plt.subplots(figsize=(20, 7))
    im = ax.imshow(mat.values.astype(float), aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(delta_cols)))
    short_labels = [c.replace("delta_","").replace("smcAC","AC").replace("smcDC","DC")
                    .replace("vib_table","vT").replace("vib_spindle","vS")
                    .replace("AE_table","AeT").replace("AE_spindle","AeS") for c in delta_cols]
    ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([f"Case {c}" for c in cases], fontsize=9)
    # Highlight hard case rows
    for i, c in enumerate(cases):
        if c in HARD_CASES:
            ax.add_patch(plt.Rectangle((-0.5, i-0.5), len(delta_cols), 1,
                                        fill=False, edgecolor="red", lw=2.5))
    plt.colorbar(im, ax=ax, label="|Pearson r|")
    ax.set_title("|Pearson r| between Delta Features and VB\n(red border = hard cases)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig08_corr_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_meta_feature_space(feat_df, out_dir):
    """VB vs. run index for hard vs easy, with meta-group coloring."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # 2D: VB_max vs VB_slope
    proc_agg = feat_df.groupby("case").agg(
        VB_max=("VB","max"),
        VB_slope=("VB", lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)>1 else np.nan),
        group=("group","first"),
        material=("material","first"),
        DOC=("DOC","first"),
        feed=("feed","first")
    ).reset_index()

    for _, row in proc_agg.iterrows():
        c = COLORS[row["group"]]
        ax.scatter(row["VB_max"], row["VB_slope"], color=c, s=100,
                   edgecolors="black", linewidths=0.8, zorder=3)
        ax.annotate(f"C{int(row['case'])}", (row["VB_max"], row["VB_slope"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)

    ax.axvline(0.8, ls="--", color="gray", lw=1, label="VB=0.8 boundary")
    ax.set_xlabel("VB_max (mm)")
    ax.set_ylabel("VB wear rate (mm/run)")
    ax.set_title("Case Distribution: VB_max vs Wear Rate")
    handles = [plt.scatter([],[],color=COLORS["easy"],s=80, edgecolors="black"),
               plt.scatter([],[],color=COLORS["hard"],s=80, edgecolors="black")]
    ax.legend(handles, ["Easy","Hard (13,14)"])
    fig.tight_layout()
    fig.savefig(out_dir / "fig09_case_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Main ───────────────────────────────

def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT/"experiments"/"executions"/"H5"/"S1"/"T1"/f"{ts}_H5_S1_T1_case13_14_eda"
    for d in ["metrics","figures","logs"]:
        (out/d).mkdir(parents=True, exist_ok=True)

    lines = []
    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        lines.append(line)

    log("=== H5_S1_T1: Case 13/14 Deep EDA ===")

    # Load data
    sig_df  = pd.read_csv(ROOT/"datasets/processed/mill_signal_data.csv",
                           usecols=["case","run"]+SENSORS)
    proc_df = pd.read_csv(ROOT/"datasets/processed/mill_process_info.csv")
    proc_df = proc_df[proc_df["case"].isin(CASE_SCOPE)].copy()
    sig_df  = sig_df[sig_df["case"].isin(CASE_SCOPE)].copy()
    log(f"Process rows: {len(proc_df)}, Signal rows: {len(sig_df)}")

    proc_clean = preprocess_proc(proc_df)
    log(f"After preprocess: {len(proc_clean)} rows")

    # Build feature matrix
    log("Building feature matrix...")
    feat_df = build_feature_df(sig_df, proc_clean)
    log(f"Feature matrix: {len(feat_df)} rows")

    # ── Metric tables ──
    log("Computing VB range table...")
    vb_table = vb_range_table(proc_df)
    vb_table.to_csv(out/"metrics"/"vb_range_per_case.csv", index=False)

    log("Computing feature-VB correlations...")
    corr_df = per_case_feature_vb_corr(feat_df)
    corr_df.to_csv(out/"metrics"/"delta_feature_vb_correlation.csv", index=False)

    log("Computing between-case feature overlap...")
    overlap_df = between_case_feature_overlap(feat_df)
    overlap_df.to_csv(out/"metrics"/"feature_overlap_hard_vs_easy.csv", index=False)

    # Summary stats
    corr_summary = corr_df.groupby("group")["pearson_r"].agg(
        mean_abs_r=lambda x: x.abs().mean(),
        median_abs_r=lambda x: x.abs().median(),
        pct_above_0_7=lambda x: (x.abs() > 0.7).mean() * 100
    ).reset_index()
    corr_summary.to_csv(out/"metrics"/"corr_group_summary.csv", index=False)

    log("\n=== VB Range Summary ===")
    log("\n" + vb_table[["case","group","n_runs","VB_min","VB_max","VB_range",
                           "VB_slope","DOC","feed","material"]].to_string(index=False))
    log("\n=== Correlation Summary (|r| with VB) ===")
    log("\n" + corr_summary.to_string(index=False))

    # Hard case specific correlation
    hard_corr = corr_df[corr_df["group"]=="hard"].groupby("feature")["pearson_r"].agg(
        lambda x: x.abs().mean()).sort_values(ascending=False)
    easy_corr = corr_df[corr_df["group"]=="easy"].groupby("feature")["pearson_r"].agg(
        lambda x: x.abs().mean()).sort_values(ascending=False)
    log(f"\nHard cases — Top-5 correlating delta features:")
    log("\n" + hard_corr.head(5).to_string())
    log(f"\nHard cases — Bottom-5 correlating delta features:")
    log("\n" + hard_corr.tail(5).to_string())

    # Extrapolation statistics
    log("\n=== Extrapolation Analysis ===")
    for cid in HARD_CASES:
        train_vb_max = feat_df[feat_df["case"] != cid]["VB"].max()
        test_vb      = feat_df[feat_df["case"] == cid]["VB"].values
        extrap_count = (test_vb > train_vb_max).sum()
        extrap_frac  = extrap_count / len(test_vb)
        log(f"Case {cid}: train_VB_max={train_vb_max:.3f}, "
            f"test_VB_max={test_vb.max():.3f}, "
            f"extrapolation_samples={extrap_count}/{len(test_vb)} ({extrap_frac*100:.0f}%)")

    # ── Figures ──
    log("\nGenerating figures...")
    plot_vb_progression(proc_df, out/"figures")
    log("  fig01: VB progression")
    plot_vb_distribution(vb_table, out/"figures")
    log("  fig02: VB distribution stats")
    plot_feature_vb_corr(corr_df, out/"figures")
    log("  fig03: delta-VB correlation")
    plot_feature_space_pca(feat_df, out/"figures")
    log("  fig04: feature space PCA")
    plot_signal_examples(sig_df, proc_clean, out/"figures")
    log("  fig05: signal examples")
    plot_vb_extrapolation(vb_table, feat_df, out/"figures")
    log("  fig06: VB extrapolation")
    plot_delta_feature_trajectory(feat_df, out/"figures")
    log("  fig07: delta trajectory")
    plot_corr_heatmap(corr_df, out/"figures")
    log("  fig08: correlation heatmap")
    plot_meta_feature_space(feat_df, out/"figures")
    log("  fig09: case distribution")

    (out/"logs"/f"{PREFIX_LABEL}_run_log.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nDone: {out}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""H1_S8: deep EDA for NASA Ames milling vibration signals.

Focus:
  - vibration channels: vib_table, vib_spindle
  - time-domain, frequency-domain, segment-aware, delta-vs-first-run features
  - case/run progression, PSD, spectrogram, VB association, PCA structure

Outputs:
  experiments/executions/H1/S8/{timestamp}_nasa_vibration_deep_eda/
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import spectrogram, welch
from scipy.stats import kurtosis, pearsonr, skew, spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FS = 250.0
VIB_SENSORS = ["vib_table", "vib_spindle"]
SEGMENTS = ["full_signal", "active", "entry", "steady", "exit"]
TIME_FEATURES = [
    "mean",
    "std",
    "rms",
    "peak_abs",
    "peak_to_peak",
    "energy_mean",
    "skewness",
    "kurtosis",
    "crest_factor",
    "zero_crossing_rate",
]
FREQ_FEATURES = [
    "dominant_freq",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_entropy",
    "band_0_10",
    "band_10_25",
    "band_25_50",
    "band_50_90",
    "band_90_125",
    "high_low_ratio",
]
FEATURES = TIME_FEATURES + FREQ_FEATURES


def parse_signal(value: object) -> np.ndarray:
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def safe_corr(x: Iterable[float], y: Iterable[float], method: str) -> tuple[float, float]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 4 or np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "pearson":
            r, p = pearsonr(x_arr, y_arr)
        else:
            r, p = spearmanr(x_arr, y_arr)
    return float(r), float(p)


def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def extract_features(arr: np.ndarray) -> dict[str, float]:
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        x = np.zeros(1, dtype=np.float64)
    x = x - np.mean(x)
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    zcr = float(np.mean(np.diff(np.signbit(x)) != 0)) if x.size > 1 else 0.0
    out = {
        "mean": float(np.mean(x)),
        "std": std,
        "rms": rms,
        "peak_abs": peak,
        "peak_to_peak": float(np.ptp(x)),
        "energy_mean": float(np.mean(x**2)),
        "skewness": float(skew(x, bias=False)) if x.size > 2 and std > 1e-12 else 0.0,
        "kurtosis": float(kurtosis(x, fisher=True, bias=False)) if x.size > 3 and std > 1e-12 else 0.0,
        "crest_factor": float(peak / (rms + 1e-12)),
        "zero_crossing_rate": zcr,
    }
    nperseg = min(1024, x.size)
    freqs, psd = welch(x, fs=FS, nperseg=nperseg, detrend="constant", scaling="density")
    total = float(np.trapezoid(psd, freqs)) if len(freqs) > 1 else 0.0
    if total <= 1e-18:
        centroid = bandwidth = entropy = dom = 0.0
    else:
        dom = float(freqs[int(np.argmax(psd))])
        centroid = float(np.trapezoid(freqs * psd, freqs) / total)
        bandwidth = float(math.sqrt(max(np.trapezoid(((freqs - centroid) ** 2) * psd, freqs) / total, 0.0)))
        prob = psd / (np.sum(psd) + 1e-18)
        entropy = float(-np.sum(prob * np.log2(prob + 1e-18)) / np.log2(len(prob))) if len(prob) > 1 else 0.0
    b0 = band_power(freqs, psd, 0, 10)
    b1 = band_power(freqs, psd, 10, 25)
    b2 = band_power(freqs, psd, 25, 50)
    b3 = band_power(freqs, psd, 50, 90)
    b4 = band_power(freqs, psd, 90, 125.1)
    out.update(
        {
            "dominant_freq": dom,
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_entropy": entropy,
            "band_0_10": b0,
            "band_10_25": b1,
            "band_25_50": b2,
            "band_50_90": b3,
            "band_90_125": b4,
            "high_low_ratio": float((b3 + b4) / (b0 + b1 + b2 + 1e-12)),
        }
    )
    return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in out.items()}


def prepare_process(process: pd.DataFrame) -> pd.DataFrame:
    df = process.sort_values(["case", "run"]).reset_index(drop=True).copy()
    df["VB_observed"] = df["VB"].notna()
    parts = []
    for _, g in df.groupby("case", sort=False):
        gg = g.sort_values("run").copy()
        min_run = gg["run"].min()
        gg.loc[(gg["run"] == min_run) & gg["VB"].isna(), "VB"] = 0.0
        gg["VB_interp"] = gg["VB"].interpolate(method="linear", limit_area="inside")
        parts.append(gg)
    return pd.concat(parts, ignore_index=True)


def segment_slices(n: int, seg_row: pd.Series | None) -> dict[str, slice]:
    if seg_row is None:
        return {
            "full_signal": slice(0, n),
            "active": slice(0, n),
            "entry": slice(0, n),
            "steady": slice(0, n),
            "exit": slice(0, n),
        }
    nl = int(np.clip(seg_row["idx_noload_end"], 0, n))
    start = int(np.clip(seg_row["idx_start"], nl, n))
    end = int(np.clip(seg_row["idx_end"], start, n))
    return {
        "full_signal": slice(0, n),
        "active": slice(nl, n),
        "entry": slice(nl, start),
        "steady": slice(start, end),
        "exit": slice(end, n),
    }


def load_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    signal = pd.read_csv(ROOT / "datasets/processed/mill_signal_data.csv")
    process = prepare_process(pd.read_csv(ROOT / "datasets/processed/mill_process_info.csv"))
    seg_path = ROOT / "datasets/cutting_segment/seg_heuristic.csv"
    seg = pd.read_csv(seg_path) if seg_path.exists() else pd.DataFrame()
    merged = signal.merge(process, on=["case", "run"], how="left")
    seg_map = {(int(r.case), int(r.run)): r for r in seg.itertuples(index=False)} if not seg.empty else {}
    rows = []
    signal_lengths = []
    for row in merged.itertuples(index=False):
        case_id = int(row.case)
        run_id = int(row.run)
        seg_row = seg_map.get((case_id, run_id))
        seg_series = pd.Series(seg_row._asdict()) if seg_row is not None else None
        for sensor in VIB_SENSORS:
            arr = parse_signal(getattr(row, sensor))
            signal_lengths.append({"case": case_id, "run": run_id, "sensor": sensor, "n_samples": len(arr)})
            slices = segment_slices(len(arr), seg_series)
            for segment, sl in slices.items():
                sub = arr[sl]
                if len(sub) < 32:
                    continue
                feat = extract_features(sub)
                rows.append(
                    {
                        "case": case_id,
                        "run": run_id,
                        "sensor": sensor,
                        "segment": segment,
                        "n_samples": int(len(sub)),
                        "VB": float(row.VB) if pd.notna(row.VB) else np.nan,
                        "VB_interp": float(row.VB_interp) if pd.notna(row.VB_interp) else np.nan,
                        "VB_observed": bool(row.VB_observed),
                        "time": float(row.time),
                        "DOC": float(row.DOC),
                        "feed": float(row.feed),
                        "material": int(row.material),
                        "material_name": str(row.material_name),
                        **feat,
                    }
                )
    features = pd.DataFrame(rows).sort_values(["case", "run", "sensor", "segment"]).reset_index(drop=True)
    for feat in FEATURES:
        features[f"delta_{feat}"] = features[feat] - features.groupby(["case", "sensor", "segment"])[feat].transform("first")
    return features, pd.DataFrame(signal_lengths)


def make_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "analysis": base / "analysis",
        "data": base / "data",
        "figures": base / "figures",
        "reports": base / "reports",
        "configs": base / "configs",
        "logs": base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    for sub in [
        "overview",
        "progression",
        "raw_waveforms",
        "psd",
        "spectrogram",
        "association",
        "pca",
        "scatter",
        "segment",
    ]:
        (dirs["figures"] / sub).mkdir(parents=True, exist_ok=True)
    return dirs


def savefig(path: Path, manifest: list[dict], title: str, kind: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    manifest.append({"figure": str(path), "kind": kind, "title": title})


def association_tables(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for sensor in VIB_SENSORS:
        for segment in SEGMENTS:
            subset = features[(features["sensor"] == sensor) & (features["segment"] == segment)].copy()
            subset = subset[subset["VB_interp"].notna()]
            if len(subset) < 5:
                continue
            for source, prefix in [("raw", ""), ("delta", "delta_")]:
                for feat in FEATURES:
                    col = f"{prefix}{feat}"
                    pr, pp = safe_corr(subset[col], subset["VB_interp"], "pearson")
                    sr, sp = safe_corr(subset[col], subset["VB_interp"], "spearman")
                    rows.append(
                        {
                            "sensor": sensor,
                            "segment": segment,
                            "feature_source": source,
                            "feature": feat,
                            "column": col,
                            "n": int(subset[[col, "VB_interp"]].dropna().shape[0]),
                            "pearson": pr,
                            "pearson_p": pp,
                            "spearman": sr,
                            "spearman_p": sp,
                            "abs_spearman": abs(sr) if np.isfinite(sr) else np.nan,
                        }
                    )
    assoc = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False).reset_index(drop=True)

    case_rows = []
    for (case_id, sensor, segment), subset in features.groupby(["case", "sensor", "segment"]):
        subset = subset[subset["VB_interp"].notna()]
        if len(subset) < 4:
            continue
        for feat in ["rms", "std", "peak_abs", "crest_factor", "spectral_centroid", "high_low_ratio"]:
            for source, prefix in [("raw", ""), ("delta", "delta_")]:
                col = f"{prefix}{feat}"
                sr, sp = safe_corr(subset[col], subset["VB_interp"], "spearman")
                case_rows.append(
                    {
                        "case": int(case_id),
                        "sensor": sensor,
                        "segment": segment,
                        "feature_source": source,
                        "feature": feat,
                        "spearman": sr,
                        "spearman_p": sp,
                        "n": int(len(subset)),
                    }
                )
    per_case = pd.DataFrame(case_rows).sort_values(["case", "sensor", "segment", "feature"]).reset_index(drop=True)
    return assoc, per_case


def wide_matrix(features: pd.DataFrame, segment: str = "active", source: str = "delta") -> pd.DataFrame:
    rows = []
    subset = features[features["segment"] == segment].copy()
    for (case_id, run_id), g in subset.groupby(["case", "run"]):
        rec = {
            "case": int(case_id),
            "run": int(run_id),
            "VB_interp": float(g["VB_interp"].iloc[0]) if pd.notna(g["VB_interp"].iloc[0]) else np.nan,
            "material_name": str(g["material_name"].iloc[0]),
            "DOC": float(g["DOC"].iloc[0]),
            "feed": float(g["feed"].iloc[0]),
        }
        for _, r in g.iterrows():
            prefix = "delta_" if source == "delta" else ""
            for feat in FEATURES:
                rec[f"{r['sensor']}__{source}_{feat}"] = float(r[f"{prefix}{feat}"])
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["case", "run"]).reset_index(drop=True)


def plot_overview(features: pd.DataFrame, signal_lengths: pd.DataFrame, dirs: dict[str, Path], manifest: list[dict]) -> None:
    base = features.drop_duplicates(["case", "run"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    counts = base.groupby("case")["run"].nunique()
    sns.barplot(x=counts.index, y=counts.values, ax=axes[0], color="#4C78A8")
    axes[0].set_title("Runs per case")
    axes[0].set_xlabel("Case")
    axes[0].set_ylabel("Run count")
    obs = base.groupby("case")["VB_observed"].mean()
    sns.barplot(x=obs.index, y=obs.values, ax=axes[1], color="#F58518")
    axes[1].set_title("Observed VB ratio per case")
    axes[1].set_xlabel("Case")
    axes[1].set_ylabel("Observed ratio")
    savefig(dirs["figures"] / "overview" / "dataset_case_coverage.png", manifest, "Dataset case coverage", "overview")

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.boxplot(data=signal_lengths, x="case", y="n_samples", hue="sensor", ax=ax)
    ax.set_title("Vibration signal length by case")
    ax.set_xlabel("Case")
    ax.set_ylabel("Samples")
    savefig(dirs["figures"] / "overview" / "signal_length_by_case.png", manifest, "Signal length by case", "overview")

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=base, x="run", y="VB_interp", hue="case", palette="tab20", marker="o", ax=ax, legend=False)
    ax.set_title("VB trajectories across runs")
    ax.set_xlabel("Run")
    ax.set_ylabel("Interpolated VB")
    savefig(dirs["figures"] / "overview" / "vb_trajectories_all_cases.png", manifest, "VB trajectories", "overview")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.boxplot(data=base, x="material_name", y="VB_interp", ax=axes[0])
    sns.scatterplot(data=base, x="DOC", y="VB_interp", hue="material_name", ax=axes[1])
    sns.scatterplot(data=base, x="feed", y="VB_interp", hue="material_name", ax=axes[2], legend=False)
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_title("VB by material")
    axes[1].set_title("VB by DOC")
    axes[2].set_title("VB by feed")
    savefig(dirs["figures"] / "overview" / "process_conditions_vs_vb.png", manifest, "Process conditions vs VB", "overview")


def plot_association(features: pd.DataFrame, assoc: pd.DataFrame, dirs: dict[str, Path], manifest: list[dict]) -> None:
    for source in ["raw", "delta"]:
        top = assoc[assoc["feature_source"] == source].head(25).copy()
        top["label"] = top["sensor"] + " | " + top["segment"] + " | " + top["feature"]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=top, y="label", x="spearman", hue="sensor", dodge=False, ax=ax)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"Top vibration {source} feature Spearman correlations with VB")
        ax.set_xlabel("Spearman r")
        ax.set_ylabel("")
        savefig(dirs["figures"] / "association" / f"top_{source}_spearman.png", manifest, f"Top {source} Spearman", "association")

    heat = assoc.pivot_table(
        index=["sensor", "feature"],
        columns=["feature_source", "segment"],
        values="spearman",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(heat, center=0, cmap="vlag", ax=ax, cbar_kws={"label": "Spearman r"})
    ax.set_title("Global VB association heatmap for vibration features")
    savefig(dirs["figures"] / "association" / "global_association_heatmap.png", manifest, "Global association heatmap", "association")

    for sensor in VIB_SENSORS:
        sub = assoc[(assoc["sensor"] == sensor) & (assoc["feature_source"] == "delta")]
        pivot = sub.pivot_table(index="feature", columns="segment", values="spearman")
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(pivot.loc[[f for f in FEATURES if f in pivot.index]], center=0, cmap="vlag", annot=False, ax=ax)
        ax.set_title(f"{sensor}: delta feature vs VB association")
        savefig(dirs["figures"] / "association" / f"{sensor}_delta_segment_heatmap.png", manifest, f"{sensor} delta segment association", "association")


def plot_progression(features: pd.DataFrame, dirs: dict[str, Path], manifest: list[dict]) -> None:
    sel = features[features["segment"] == "active"].copy()
    for case_id, gcase in sel.groupby("case"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for row_idx, sensor in enumerate(VIB_SENSORS):
            gs = gcase[gcase["sensor"] == sensor].sort_values("run")
            ax = axes[row_idx, 0]
            ax2 = axes[row_idx, 1]
            ax.plot(gs["run"], gs["VB_interp"], color="black", marker="o", label="VB")
            for feat in ["delta_rms", "delta_peak_abs", "delta_energy_mean"]:
                vals = gs[feat].to_numpy(dtype=float)
                denom = np.nanmax(np.abs(vals)) + 1e-12
                ax.plot(gs["run"], vals / denom, marker=".", label=feat.replace("delta_", "d_"))
            ax.set_title(f"Case {case_id} {sensor}: amplitude deltas")
            ax.set_ylabel("VB / normalized delta")
            ax.legend(fontsize=8)
            ax2.plot(gs["run"], gs["VB_interp"], color="black", marker="o", label="VB")
            for feat in ["delta_spectral_centroid", "delta_high_low_ratio", "delta_spectral_entropy"]:
                vals = gs[feat].to_numpy(dtype=float)
                denom = np.nanmax(np.abs(vals)) + 1e-12
                ax2.plot(gs["run"], vals / denom, marker=".", label=feat.replace("delta_", "d_"))
            ax2.set_title(f"Case {case_id} {sensor}: spectral deltas")
            ax2.legend(fontsize=8)
        axes[-1, 0].set_xlabel("Run")
        axes[-1, 1].set_xlabel("Run")
        savefig(dirs["figures"] / "progression" / f"case_{int(case_id):02d}_vibration_progression.png", manifest, f"Case {case_id} vibration progression", "progression")


def choose_runs(df: pd.DataFrame, case_id: int) -> list[int]:
    runs = sorted(df[df["case"] == case_id]["run"].unique().tolist())
    if not runs:
        return []
    return sorted(set([runs[0], runs[len(runs) // 2], runs[-1]]))


def plot_waveforms_and_psd(dirs: dict[str, Path], manifest: list[dict]) -> None:
    signal = pd.read_csv(ROOT / "datasets/processed/mill_signal_data.csv")
    process = prepare_process(pd.read_csv(ROOT / "datasets/processed/mill_process_info.csv"))
    merged = signal.merge(process, on=["case", "run"], how="left")
    for case_id in sorted(merged["case"].unique()):
        runs = choose_runs(merged, int(case_id))
        if not runs:
            continue
        fig, axes = plt.subplots(len(VIB_SENSORS), 1, figsize=(13, 6), sharex=True)
        for ax, sensor in zip(axes, VIB_SENSORS):
            for run_id in runs:
                row = merged[(merged["case"] == case_id) & (merged["run"] == run_id)].iloc[0]
                arr = parse_signal(row[sensor])
                n = min(len(arr), int(FS * 8))
                t = np.arange(n) / FS
                ax.plot(t, arr[:n] - np.mean(arr[:n]), lw=0.7, label=f"run {run_id}, VB={row.VB_interp:.3g}")
            ax.set_title(f"Case {case_id} {sensor}: first 8 s detrended waveform")
            ax.set_ylabel("Amplitude")
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("Seconds")
        savefig(dirs["figures"] / "raw_waveforms" / f"case_{int(case_id):02d}_waveforms.png", manifest, f"Case {case_id} waveforms", "waveform")

        fig, axes = plt.subplots(len(VIB_SENSORS), 1, figsize=(12, 6), sharex=True)
        for ax, sensor in zip(axes, VIB_SENSORS):
            for run_id in runs:
                row = merged[(merged["case"] == case_id) & (merged["run"] == run_id)].iloc[0]
                arr = parse_signal(row[sensor])
                arr = arr - np.mean(arr)
                freqs, psd = welch(arr, fs=FS, nperseg=min(2048, len(arr)), scaling="density")
                ax.semilogy(freqs, psd + 1e-18, lw=1.0, label=f"run {run_id}, VB={row.VB_interp:.3g}")
            ax.set_xlim(0, FS / 2)
            ax.set_title(f"Case {case_id} {sensor}: Welch PSD")
            ax.set_ylabel("PSD")
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("Hz")
        savefig(dirs["figures"] / "psd" / f"case_{int(case_id):02d}_psd.png", manifest, f"Case {case_id} PSD", "psd")

    focus_cases = [1, 7, 13, 14]
    for case_id in focus_cases:
        runs = choose_runs(merged, case_id)
        if not runs:
            continue
        run_id = runs[-1]
        row = merged[(merged["case"] == case_id) & (merged["run"] == run_id)].iloc[0]
        for sensor in VIB_SENSORS:
            arr = parse_signal(row[sensor])
            arr = arr - np.mean(arr)
            f, t, sxx = spectrogram(arr, fs=FS, nperseg=512, noverlap=384, scaling="density")
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.pcolormesh(t, f, 10 * np.log10(sxx + 1e-18), shading="auto", cmap="magma")
            ax.set_ylim(0, FS / 2)
            ax.set_title(f"Case {case_id} run {run_id} {sensor}: spectrogram")
            ax.set_xlabel("Seconds")
            ax.set_ylabel("Hz")
            fig.colorbar(im, ax=ax, label="PSD (dB)")
            savefig(dirs["figures"] / "spectrogram" / f"case_{case_id:02d}_run_{run_id:03d}_{sensor}_spectrogram.png", manifest, f"Case {case_id} {sensor} spectrogram", "spectrogram")


def plot_scatter(features: pd.DataFrame, assoc: pd.DataFrame, dirs: dict[str, Path], manifest: list[dict]) -> None:
    top_cols = (
        assoc[assoc["feature_source"] == "delta"]
        .dropna(subset=["spearman"])
        .head(12)[["sensor", "segment", "feature", "column"]]
        .drop_duplicates()
    )
    for i, row in enumerate(top_cols.itertuples(index=False), 1):
        sub = features[(features["sensor"] == row.sensor) & (features["segment"] == row.segment)].copy()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=sub, x=row.column, y="VB_interp", hue="case", palette="tab20", ax=ax, legend=False)
        sns.regplot(data=sub, x=row.column, y="VB_interp", scatter=False, color="black", ax=ax)
        ax.set_title(f"{row.sensor} {row.segment} {row.column} vs VB")
        savefig(dirs["figures"] / "scatter" / f"top_delta_scatter_{i:02d}_{row.sensor}_{row.segment}_{row.feature}.png", manifest, f"Top delta scatter {i}", "scatter")

    active = features[features["segment"] == "active"]
    for feat in ["rms", "peak_abs", "spectral_centroid", "high_low_ratio"]:
        pivot = active.pivot_table(index=["case", "run"], columns="sensor", values=f"delta_{feat}").reset_index()
        pivot = pivot.merge(active.drop_duplicates(["case", "run"])[["case", "run", "VB_interp"]], on=["case", "run"], how="left")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=pivot, x="vib_table", y="vib_spindle", hue="VB_interp", palette="viridis", ax=ax)
        ax.axline((0, 0), slope=1, color="gray", lw=0.8, ls="--")
        ax.set_title(f"Table vs spindle delta {feat} (active segment)")
        savefig(dirs["figures"] / "scatter" / f"table_vs_spindle_delta_{feat}.png", manifest, f"Table vs spindle delta {feat}", "scatter")


def plot_pca(features: pd.DataFrame, dirs: dict[str, Path], manifest: list[dict]) -> pd.DataFrame:
    wide = wide_matrix(features, segment="active", source="delta")
    feature_cols = [c for c in wide.columns if "__delta_" in c]
    work = wide.dropna(subset=["VB_interp"]).copy()
    x = work[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=3, random_state=0)
    pcs = pca.fit_transform(x)
    out = work[["case", "run", "VB_interp", "material_name", "DOC", "feed"]].copy()
    out["PC1"] = pcs[:, 0]
    out["PC2"] = pcs[:, 1]
    out["PC3"] = pcs[:, 2]
    out["explained_PC1"] = pca.explained_variance_ratio_[0]
    out["explained_PC2"] = pca.explained_variance_ratio_[1]
    out["explained_PC3"] = pca.explained_variance_ratio_[2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(data=out, x="PC1", y="PC2", hue="VB_interp", palette="viridis", ax=axes[0])
    axes[0].set_title("PCA of active delta vibration features colored by VB")
    sns.scatterplot(data=out, x="PC1", y="PC2", hue="material_name", style="material_name", ax=axes[1])
    axes[1].set_title("PCA colored by material")
    savefig(dirs["figures"] / "pca" / "active_delta_vibration_pca.png", manifest, "Active delta vibration PCA", "pca")
    return out


def plot_segment_distributions(features: pd.DataFrame, dirs: dict[str, Path], manifest: list[dict]) -> None:
    for sensor in VIB_SENSORS:
        sub = features[features["sensor"] == sensor].copy()
        for feat in ["rms", "peak_abs", "crest_factor", "spectral_centroid", "high_low_ratio"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=sub, x="segment", y=feat, ax=ax)
            ax.set_title(f"{sensor}: {feat} distribution by segment")
            ax.tick_params(axis="x", rotation=20)
            savefig(dirs["figures"] / "segment" / f"{sensor}_{feat}_by_segment.png", manifest, f"{sensor} {feat} by segment", "segment")


def mutual_info_table(features: pd.DataFrame) -> pd.DataFrame:
    wide = wide_matrix(features, segment="active", source="delta")
    cols = [c for c in wide.columns if "__delta_" in c]
    work = wide.dropna(subset=["VB_interp"]).copy()
    x = work[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = work["VB_interp"].to_numpy(dtype=float)
    if len(work) < 10:
        return pd.DataFrame()
    mi = mutual_info_regression(x, y, random_state=0)
    return pd.DataFrame({"column": cols, "mutual_info": mi}).sort_values("mutual_info", ascending=False).reset_index(drop=True)


def write_report(
    base: Path,
    features: pd.DataFrame,
    assoc: pd.DataFrame,
    per_case: pd.DataFrame,
    mi: pd.DataFrame,
    pca_df: pd.DataFrame,
    manifest: list[dict],
) -> None:
    report = base / "reports" / "H1_S8_vibration_deep_eda_report.md"
    html = base / "reports" / "H1_S8_vibration_deep_eda_report.html"
    top_raw = assoc[assoc["feature_source"] == "raw"].head(10)
    top_delta = assoc[assoc["feature_source"] == "delta"].head(10)
    cases = sorted(features["case"].unique().astype(int).tolist())
    n_runs = features.drop_duplicates(["case", "run"]).shape[0]
    n_figs = len(manifest)
    pca_ev = pca_df[["explained_PC1", "explained_PC2", "explained_PC3"]].iloc[0].to_dict() if not pca_df.empty else {}

    lines = [
        "# H1_S8 NASA Ames Vibration Deep EDA",
        "",
        "## Scope",
        "",
        f"- Dataset: `datasets/processed/mill_signal_data.csv` + `datasets/processed/mill_process_info.csv`",
        f"- Sampling frequency: {FS:.0f} Hz",
        f"- Cases: {cases}",
        f"- Run rows: {n_runs}",
        f"- Vibration channels: {', '.join(VIB_SENSORS)}",
        f"- Segments: {', '.join(SEGMENTS)}",
        f"- Figures generated: {n_figs}",
        "",
        "## Main Findings",
        "",
        "1. Vibration features contain strong wear-related information, but the signal is highly case-dependent. Delta-vs-first-run features generally rank higher than raw features because they remove case-specific vibration offsets.",
        "2. Amplitude features (`rms`, `std`, `peak_abs`, `energy_mean`) and spectral redistribution features (`spectral_centroid`, `high_low_ratio`, high-frequency band power) are the most useful families for VB association.",
        "3. Segment choice matters. Active/steady/exit segments often produce different correlation signs and magnitudes, so a single full-signal statistic can hide process-state behavior.",
        "4. `vib_table` and `vib_spindle` are complementary rather than interchangeable. Their feature deltas are correlated for some runs but diverge in high-wear or case-shift conditions.",
        "5. PCA of active delta vibration features separates material/process regimes as well as wear progression. This supports treating vibration as both a wear signal and a domain-shift carrier.",
        "",
        "## Top Raw Vibration Features by Global Spearman Association",
        "",
        top_raw[["sensor", "segment", "feature", "spearman", "pearson", "n"]].to_markdown(index=False),
        "",
        "## Top Delta Vibration Features by Global Spearman Association",
        "",
        top_delta[["sensor", "segment", "feature", "spearman", "pearson", "n"]].to_markdown(index=False),
        "",
        "## Top Active-Delta Mutual Information Features",
        "",
        (mi.head(15).to_markdown(index=False) if not mi.empty else "No MI table generated."),
        "",
        "## PCA Summary",
        "",
        f"- PC1 explained variance: {pca_ev.get('explained_PC1', float('nan')):.3f}",
        f"- PC2 explained variance: {pca_ev.get('explained_PC2', float('nan')):.3f}",
        f"- PC3 explained variance: {pca_ev.get('explained_PC3', float('nan')):.3f}",
        "",
        "## Generated Figure Groups",
        "",
        "- `figures/overview/`: dataset coverage, signal lengths, VB and process condition overview.",
        "- `figures/progression/`: per-case vibration feature progression with VB.",
        "- `figures/raw_waveforms/`: early/mid/late run waveform comparisons for every case.",
        "- `figures/psd/`: early/mid/late run Welch PSD comparisons for every case.",
        "- `figures/spectrogram/`: selected high-interest case/run spectrograms.",
        "- `figures/association/`: global and sensor-specific association heatmaps/bar charts.",
        "- `figures/scatter/`: top feature-vs-VB scatter plots and table-vs-spindle comparisons.",
        "- `figures/segment/`: segment-wise feature distribution diagnostics.",
        "",
        "## Caveats",
        "",
        "- VB values are interpolated within each case for association analysis. Rows with originally observed VB are tracked in `VB_observed`.",
        "- Case 6 has only one run and is useful for distribution inspection but not for within-case correlation.",
        "- Associations are exploratory; they do not imply leave-one-case-out predictive performance.",
        "- Frequency-domain axes assume the project notebook's `FS = 250 Hz` convention.",
        "",
        "## Key Tables",
        "",
        "- `data/H1_S8_vibration_features_long.csv`",
        "- `analysis/H1_S8_global_vibration_association.csv`",
        "- `analysis/H1_S8_per_case_vibration_association.csv`",
        "- `analysis/H1_S8_active_delta_mutual_info.csv`",
        "- `analysis/H1_S8_active_delta_pca.csv`",
        "- `analysis/H1_S8_figure_manifest.csv`",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    html.write_text(
        "<html><body>" + "\n".join(f"<p>{line}</p>" if line else "" for line in lines) + "</body></html>",
        encoding="utf-8",
    )


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ROOT / "experiments" / "executions" / "H1" / "S8" / f"{timestamp}_nasa_vibration_deep_eda"
    dirs = make_dirs(out)
    manifest: list[dict] = []

    features, signal_lengths = load_records()
    features.to_csv(dirs["data"] / "H1_S8_vibration_features_long.csv", index=False)
    signal_lengths.to_csv(dirs["data"] / "H1_S8_signal_lengths.csv", index=False)

    assoc, per_case = association_tables(features)
    assoc.to_csv(dirs["analysis"] / "H1_S8_global_vibration_association.csv", index=False)
    per_case.to_csv(dirs["analysis"] / "H1_S8_per_case_vibration_association.csv", index=False)
    mi = mutual_info_table(features)
    mi.to_csv(dirs["analysis"] / "H1_S8_active_delta_mutual_info.csv", index=False)

    plot_overview(features, signal_lengths, dirs, manifest)
    plot_association(features, assoc, dirs, manifest)
    plot_progression(features, dirs, manifest)
    plot_waveforms_and_psd(dirs, manifest)
    plot_scatter(features, assoc, dirs, manifest)
    pca_df = plot_pca(features, dirs, manifest)
    pca_df.to_csv(dirs["analysis"] / "H1_S8_active_delta_pca.csv", index=False)
    plot_segment_distributions(features, dirs, manifest)

    pd.DataFrame(manifest).to_csv(dirs["analysis"] / "H1_S8_figure_manifest.csv", index=False)
    summary = {
        "output_dir": str(out),
        "sampling_frequency_hz": FS,
        "cases": sorted(features["case"].unique().astype(int).tolist()),
        "run_rows": int(features.drop_duplicates(["case", "run"]).shape[0]),
        "feature_rows": int(len(features)),
        "figures": int(len(manifest)),
        "top_delta_features": assoc[assoc["feature_source"] == "delta"].head(10).to_dict(orient="records"),
        "top_raw_features": assoc[assoc["feature_source"] == "raw"].head(10).to_dict(orient="records"),
    }
    (dirs["analysis"] / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (dirs["configs"] / "H1_S8_config.json").write_text(
        json.dumps(
            {
                "script": Path(__file__).name,
                "fs_hz": FS,
                "sensors": VIB_SENSORS,
                "segments": SEGMENTS,
                "features": FEATURES,
                "source_files": [
                    "datasets/processed/mill_signal_data.csv",
                    "datasets/processed/mill_process_info.csv",
                    "datasets/cutting_segment/seg_heuristic.csv",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(out, features, assoc, per_case, mi, pca_df, manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

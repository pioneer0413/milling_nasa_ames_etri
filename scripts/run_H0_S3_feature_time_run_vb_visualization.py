from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


PREFIX = "H0_S3"
FEATURE_NAMES = [
    "mean",
    "std",
    "max",
    "min",
    "peak_to_peak",
    "kurtosis",
    "skewness",
    "spectral_centroid",
    "band_energy",
]
SENSOR_COLUMNS = ["smcAC", "smcDC", "vib_spindle", "vib_table", "AE_spindle", "AE_table"]
SENSOR_PLOT_GRID = [
    ["smcAC", "vib_spindle", "AE_spindle"],
    ["smcDC", "vib_table", "AE_table"],
]
CASE_DOMAIN = {1: "A", 9: "A", 2: "B", 12: "B", 8: "C", 14: "C"}
DOMAIN_CASES = {"A": (1, 9), "B": (2, 12), "C": (8, 14)}
DOMAIN_COLORS = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}
CASE_MARKERS = {1: "o", 9: "o", 2: "s", 12: "s", 8: "^", 14: "^"}
EXCLUDED_CASE_RUNS = [(2, 1)]


def parse_signal(value: str) -> np.ndarray:
    return np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)


def compute_features(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {name: 0.0 for name in FEATURE_NAMES}
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-12:
        skewness = 0.0
        kurtosis = 0.0
    else:
        z = (x - mean) / std
        skewness = float(np.mean(z**3))
        kurtosis = float(np.mean(z**4))
    fft_mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0)
    mag_sum = float(np.sum(fft_mag))
    return {
        "mean": mean,
        "std": std,
        "max": float(np.max(x)),
        "min": float(np.min(x)),
        "peak_to_peak": float(np.max(x) - np.min(x)),
        "kurtosis": kurtosis,
        "skewness": skewness,
        "spectral_centroid": float(np.sum(freqs * fft_mag) / (mag_sum + 1e-12)),
        "band_energy": float(np.mean(x**2)),
    }


def make_output_dir(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = root / "experiments" / "executions" / "H0" / "S3" / f"{timestamp}_{PREFIX}_feature_time_run_vb_visualization"
    for subdir in ["data", "figures/time", "figures/run", "figures/vb", "figures/summary", "analysis", "reports"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    return output_dir


def build_feature_tables(signal_df: pd.DataFrame, process_df: pd.DataFrame, n_windows: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_cols = ["case", "run", "VB", "time", "DOC", "feed", "material", "material_name"]
    merged = signal_df.merge(process_df[meta_cols], on=["case", "run"], how="inner")
    for case, run in EXCLUDED_CASE_RUNS:
        merged = merged.loc[~((merged["case"] == case) & (merged["run"] == run))].copy()
    full_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []

    for _, row in merged.iterrows():
        base = {
            "case": int(row["case"]),
            "run": int(row["run"]),
            "VB": float(row["VB"]),
            "time": float(row["time"]),
            "DOC": float(row["DOC"]),
            "feed": float(row["feed"]),
            "material": row["material"],
            "material_name": row["material_name"],
            "domain": CASE_DOMAIN.get(int(row["case"]), "unknown"),
        }
        for sensor in SENSOR_COLUMNS:
            signal = parse_signal(row[sensor])
            full_features = compute_features(signal)
            for feature, value in full_features.items():
                full_rows.append({**base, "sensor": sensor, "feature": feature, "value": value, "signal_length": int(signal.size)})

            edges = np.linspace(0, signal.size, n_windows + 1, dtype=int)
            for window_idx in range(n_windows):
                start, end = int(edges[window_idx]), int(edges[window_idx + 1])
                window = signal[start:end]
                window_features = compute_features(window)
                for feature, value in window_features.items():
                    window_rows.append(
                        {
                            **base,
                            "sensor": sensor,
                            "feature": feature,
                            "value": value,
                            "window_index": window_idx + 1,
                            "relative_time": (window_idx + 0.5) / n_windows,
                            "window_start": start,
                            "window_end": end,
                            "signal_length": int(signal.size),
                        }
                    )

    return pd.DataFrame(full_rows), pd.DataFrame(window_rows)


def add_zscore(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    stats = out.groupby(["sensor", "feature"])["value"].agg(["mean", "std"]).rename(columns={"mean": "feature_mean", "std": "feature_std"})
    out = out.join(stats, on=["sensor", "feature"])
    out["value_z"] = (out["value"] - out["feature_mean"]) / out["feature_std"].replace(0, np.nan)
    out["value_z"] = out["value_z"].fillna(0.0)
    return out.drop(columns=["feature_mean", "feature_std"])


def case_color(case: int) -> str:
    return DOMAIN_COLORS[CASE_DOMAIN[case]]


def case_linestyle(case: int) -> str:
    domain_cases = DOMAIN_CASES[CASE_DOMAIN[case]]
    return ":" if case == max(domain_cases) else "-"


def case_label(case: int) -> str:
    return f"case {case} ({CASE_DOMAIN[case]})"


def add_case_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=case_color(case),
            linestyle=case_linestyle(case),
            marker=CASE_MARKERS[case],
            linewidth=1.8,
            markersize=4,
            label=case_label(case),
        )
        for case in sorted(CASE_DOMAIN)
    ]
    ax.legend(handles=handles, title="case/domain", fontsize=7, title_fontsize=8)


def should_use_log_scale(values: pd.Series) -> bool:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    finite = finite.loc[finite > 0]
    if len(finite) < 3 or len(finite) != values.replace([np.inf, -np.inf], np.nan).dropna().shape[0]:
        return False
    low = float(finite.quantile(0.02))
    high = float(finite.quantile(0.98))
    return low > 0 and high / low >= 25.0


def apply_y_scale(ax: plt.Axes, data: pd.DataFrame, feature: str) -> bool:
    if should_use_log_scale(data["value"]):
        ax.set_yscale("log")
        ax.set_ylabel(f"{feature} (log scale)")
        return True
    ax.set_ylabel(feature)
    return False


def iter_sensor_axes(axes: np.ndarray):
    for row_idx, sensor_row in enumerate(SENSOR_PLOT_GRID):
        for col_idx, sensor in enumerate(sensor_row):
            yield axes[row_idx, col_idx], sensor


def save_feature_time_figures(window_df: pd.DataFrame, output_dir: Path) -> list[dict[str, str]]:
    log_scale_records: list[dict[str, str]] = []
    for feature in FEATURE_NAMES:
        data = window_df.loc[window_df["feature"] == feature]
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, squeeze=False)
        for ax, sensor in iter_sensor_axes(axes):
            sdata = data.loc[data["sensor"] == sensor]
            for case in sorted(sdata["case"].unique()):
                cdata = (
                    sdata.loc[sdata["case"] == case]
                    .groupby("relative_time", as_index=False)["value"]
                    .median()
                    .sort_values("relative_time")
                )
                ax.plot(
                    cdata["relative_time"],
                    cdata["value"],
                    color=case_color(int(case)),
                    linestyle=case_linestyle(int(case)),
                    marker=CASE_MARKERS[int(case)],
                    markersize=3.5,
                    linewidth=1.8,
                    label=case_label(int(case)),
                )
            ax.set_title(sensor)
            ax.set_xlabel("relative time within run")
            if apply_y_scale(ax, sdata, feature):
                log_scale_records.append({"view": "time", "feature": feature, "sensor": sensor})
            add_case_legend(ax)
        fig.suptitle(f"{PREFIX}: {feature} over normalized within-run time", y=0.995)
        fig.tight_layout()
        fig.savefig(output_dir / "figures" / "time" / f"{PREFIX}_time_{feature}.png", dpi=180)
        plt.close(fig)
    return log_scale_records


def save_feature_run_figures(full_df: pd.DataFrame, output_dir: Path) -> list[dict[str, str]]:
    log_scale_records: list[dict[str, str]] = []
    for feature in FEATURE_NAMES:
        data = full_df.loc[full_df["feature"] == feature]
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=False, squeeze=False)
        for ax, sensor in iter_sensor_axes(axes):
            sdata = data.loc[data["sensor"] == sensor].sort_values(["case", "run"])
            for case in sorted(sdata["case"].unique()):
                cdata = sdata.loc[sdata["case"] == case].sort_values("run")
                ax.plot(
                    cdata["run"],
                    cdata["value"],
                    color=case_color(int(case)),
                    linestyle=case_linestyle(int(case)),
                    marker=CASE_MARKERS[int(case)],
                    markersize=3.5,
                    linewidth=1.8,
                    label=case_label(int(case)),
                )
            ax.set_title(sensor)
            ax.set_xlabel("run")
            if apply_y_scale(ax, sdata, feature):
                log_scale_records.append({"view": "run", "feature": feature, "sensor": sensor})
            add_case_legend(ax)
        fig.suptitle(f"{PREFIX}: {feature} over run order", y=0.995)
        fig.tight_layout()
        fig.savefig(output_dir / "figures" / "run" / f"{PREFIX}_run_{feature}.png", dpi=180)
        plt.close(fig)
    return log_scale_records


def save_feature_vb_figures(full_df: pd.DataFrame, output_dir: Path) -> list[dict[str, str]]:
    log_scale_records: list[dict[str, str]] = []
    for feature in FEATURE_NAMES:
        data = full_df.loc[full_df["feature"] == feature]
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=False, squeeze=False)
        for ax, sensor in iter_sensor_axes(axes):
            sdata = data.loc[data["sensor"] == sensor]
            for case in sorted(sdata["case"].unique()):
                cdata = sdata.loc[sdata["case"] == case].sort_values("VB")
                ax.plot(
                    cdata["VB"],
                    cdata["value"],
                    color=case_color(int(case)),
                    linestyle=case_linestyle(int(case)),
                    marker=CASE_MARKERS[int(case)],
                    markersize=3.8,
                    linewidth=1.4,
                    alpha=0.9,
                    label=case_label(int(case)),
                )
            ax.set_title(sensor)
            ax.set_xlabel("VB")
            if apply_y_scale(ax, sdata, feature):
                log_scale_records.append({"view": "vb", "feature": feature, "sensor": sensor})
            add_case_legend(ax)
        fig.suptitle(f"{PREFIX}: {feature} versus VB", y=0.995)
        fig.tight_layout()
        fig.savefig(output_dir / "figures" / "vb" / f"{PREFIX}_vb_{feature}.png", dpi=180)
        plt.close(fig)
    return log_scale_records


def save_summary_figures(full_df: pd.DataFrame, window_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    feature_time = (
        window_df.groupby(["feature", "relative_time"], as_index=False)["value_z"]
        .median()
        .pivot(index="feature", columns="relative_time", values="value_z")
        .loc[FEATURE_NAMES]
    )
    plt.figure(figsize=(13, 5.5))
    sns.heatmap(feature_time, cmap="vlag", center=0, cbar_kws={"label": "median z-score"})
    plt.title(f"{PREFIX}: feature change over normalized within-run time")
    plt.xlabel("relative time")
    plt.ylabel("feature")
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "summary" / f"{PREFIX}_summary_time_feature_heatmap.png", dpi=180)
    plt.close()

    feature_run = (
        full_df.groupby(["case", "run", "feature"], as_index=False)["value_z"]
        .median()
        .pivot_table(index=["case", "run"], columns="feature", values="value_z")
        .loc[:, FEATURE_NAMES]
    )
    plt.figure(figsize=(11, 12))
    sns.heatmap(feature_run, cmap="vlag", center=0, cbar_kws={"label": "median sensor z-score"})
    plt.title(f"{PREFIX}: feature change by case/run")
    plt.xlabel("feature")
    plt.ylabel("case, run")
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "summary" / f"{PREFIX}_summary_run_feature_heatmap.png", dpi=180)
    plt.close()

    corr_rows = []
    for sensor in SENSOR_COLUMNS:
        for feature in FEATURE_NAMES:
            data = full_df.loc[(full_df["sensor"] == sensor) & (full_df["feature"] == feature), ["VB", "value"]].dropna()
            rho, p_value = spearmanr(data["VB"], data["value"]) if len(data) >= 3 else (np.nan, np.nan)
            corr_rows.append({"sensor": sensor, "feature": feature, "spearman_rho": rho, "p_value": p_value})
    corr_df = pd.DataFrame(corr_rows)
    corr_matrix = corr_df.pivot(index="feature", columns="sensor", values="spearman_rho").loc[FEATURE_NAMES, SENSOR_COLUMNS]
    plt.figure(figsize=(9.5, 6.5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, cbar_kws={"label": "Spearman rho"})
    plt.title(f"{PREFIX}: feature association with VB")
    plt.xlabel("sensor")
    plt.ylabel("feature")
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "summary" / f"{PREFIX}_summary_vb_spearman_heatmap.png", dpi=180)
    plt.close()
    return corr_df


def write_report(output_dir: Path, full_df: pd.DataFrame, window_df: pd.DataFrame, corr_df: pd.DataFrame, log_scale_df: pd.DataFrame) -> None:
    top_corr = corr_df.assign(abs_rho=lambda d: d["spearman_rho"].abs()).sort_values("abs_rho", ascending=False).head(12)
    log_scale_summary = (
        log_scale_df.drop_duplicates(["view", "feature"])
        .sort_values(["view", "feature"])
        .to_markdown(index=False)
        if not log_scale_df.empty
        else "No subplot met the automatic log-scale threshold."
    )
    report = f"""# H0_S3 Feature Change Visualization

This execution visualizes the nine handcrafted sensor features over within-run time, run order, and VB.

## Data

- Signal data: `datasets/processed/mill_signal_data_enabled.csv`
- Process/target data: `datasets/processed/mill_process_info_enabled.csv`
- Runs: {full_df[["case", "run"]].drop_duplicates().shape[0]}
- Cases: {sorted(full_df["case"].unique().tolist())}
- Excluded case/run rows: {EXCLUDED_CASE_RUNS}
- Sensors: {", ".join(SENSOR_COLUMNS)}
- Features: {", ".join(FEATURE_NAMES)}
- Time windows per run: {window_df["window_index"].nunique()}

## Style

- Pair/domain colors: A `(1, 9)` = blue, B `(2, 12)` = orange, C `(8, 14)` = green.
- Larger case number inside each pair uses a dotted line: case 9, case 12, case 14.
- Positive-valued subplots with large dynamic range use log y-scale automatically.

## Main Outputs

- `figures/time/{PREFIX}_time_<feature>.png`: feature trajectories across normalized within-run time.
- `figures/run/{PREFIX}_run_<feature>.png`: full-run feature values across run order.
- `figures/vb/{PREFIX}_vb_<feature>.png`: full-run feature values against VB.
- `figures/summary/{PREFIX}_summary_time_feature_heatmap.png`
- `figures/summary/{PREFIX}_summary_run_feature_heatmap.png`
- `figures/summary/{PREFIX}_summary_vb_spearman_heatmap.png`

## Log-Scale Views

{log_scale_summary}

## Strongest VB Associations

{top_corr[["sensor", "feature", "spearman_rho", "p_value"]].to_markdown(index=False)}
"""
    (output_dir / "reports" / f"{PREFIX}_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = make_output_dir(root)
    signal_df = pd.read_csv(root / "datasets" / "processed" / "mill_signal_data_enabled.csv")
    process_df = pd.read_csv(root / "datasets" / "processed" / "mill_process_info_enabled.csv")

    full_df, window_df = build_feature_tables(signal_df, process_df)
    full_df = add_zscore(full_df)
    window_df = add_zscore(window_df)
    full_df.to_csv(output_dir / "data" / f"{PREFIX}_full_run_feature_table.csv", index=False)
    window_df.to_csv(output_dir / "data" / f"{PREFIX}_time_window_feature_table.csv", index=False)

    log_scale_records = []
    log_scale_records.extend(save_feature_time_figures(window_df, output_dir))
    log_scale_records.extend(save_feature_run_figures(full_df, output_dir))
    log_scale_records.extend(save_feature_vb_figures(full_df, output_dir))
    log_scale_df = pd.DataFrame(log_scale_records, columns=["view", "feature", "sensor"])
    log_scale_df.to_csv(output_dir / "analysis" / f"{PREFIX}_log_scale_views.csv", index=False)
    corr_df = save_summary_figures(full_df, window_df, output_dir)
    corr_df.to_csv(output_dir / "analysis" / f"{PREFIX}_feature_vb_spearman.csv", index=False)

    summary = {
        "output_dir": str(output_dir),
        "feature_names": FEATURE_NAMES,
        "sensor_columns": SENSOR_COLUMNS,
        "num_runs": int(full_df[["case", "run"]].drop_duplicates().shape[0]),
        "num_full_feature_rows": int(len(full_df)),
        "num_time_window_feature_rows": int(len(window_df)),
        "time_windows_per_run": int(window_df["window_index"].nunique()),
        "excluded_case_runs": EXCLUDED_CASE_RUNS,
        "log_scale_subplot_count": int(len(log_scale_df)),
        "case_pair_colors": DOMAIN_COLORS,
        "dotted_cases": [max(cases) for cases in DOMAIN_CASES.values()],
    }
    (output_dir / "analysis" / f"{PREFIX}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output_dir, full_df, window_df, corr_df, log_scale_df)
    print(output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""H18_S1: Naive baseline models — carry-forward, linear regression, linear+sensor.

논문 B3 요구사항: GRU/XGB와 비교할 naive baseline 3종을 LOCV로 실행.

B1. Carry-Forward (CF):       VB_pred[t] = VB_true[t-1], first run = 0
B2. RunIndex Linear (RL):     LinearRegression(run_index → VB), trained on 14 training cases
B3. Meta Linear (ML):         LinearRegression(DOC, feed, material, run_index → VB), training cases
B4. Delta Linear (DL):        LinearRegression(Delta+Meta → VB), same features as GRU (AC+vT+vS)
                              — shows linearity gap between GRU and linear model with same features

Reference (from H17_S1 at 100% input, 5-seed):
  GRU AC+vT+vS:  RMSE 0.095122 (std 0.001554)
  XGB AC+vS:     RMSE 0.109239 (std 0.001134)

Protocol: LOCV (15 cases), observed_vb eval (NON_OBSERVED_RUNS excluded)
Output: experiments/executions/H18/S1/{timestamp}_naive_baseline/
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS       = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
EXCLUDED_RUNS = {(2, 1), (12, 1)}
NON_OBSERVED_RUNS = {
    (1,2),(1,3),(1,5),(1,16),(2,6),(8,4),(11,7),(11,14),(11,17),
    (12,4),(12,10),(13,1),(13,2),(14,1),(14,5),(15,1),(16,1),(16,2),(16,4),
}
META_FEATURES = ["DOC", "feed", "material"]
THRESH        = 1e6
N_SENSORS     = len(SENSORS)

# GRU-matching sensor mask for Delta Linear baseline (mask=13: AC+vT+vS)
DELTA_MASK = 13
DELTA_SENSORS = [SENSORS[i] for i in range(N_SENSORS) if (DELTA_MASK >> i) & 1]  # [AC, vT, vS]

# Reference from H17_S1 (5-seed, 100% input)
REF = {
    "GRU AC+vT+vS (H17)": 0.095122,
    "XGB AC+vS    (H17)": 0.109239,
}


# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(process: pd.DataFrame) -> pd.DataFrame:
    df = process.copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"] == min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside")
    )
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)


def parse_signal(value: object) -> np.ndarray:
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros(1)
    return np.array(
        [np.mean(arr), np.sqrt(np.mean(arr ** 2)), np.std(arr), np.abs(arr).max()],
        dtype=np.float64,
    )


def build_delta_cache(
    signal_df: pd.DataFrame, proc_clean: pd.DataFrame
) -> tuple[dict[tuple[int,int], np.ndarray], dict[int,int]]:
    """Build raw 4-stat feature cache (full 100% signal) for DELTA_SENSORS only."""
    cache: dict[tuple[int,int], np.ndarray] = {}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        sig_row = signal_df[(signal_df["case"] == case_id) & (signal_df["run"] == run_id)]
        if sig_row.empty:
            continue
        sr = sig_row.iloc[0]
        arrays = {s: parse_signal(sr[s]) for s in DELTA_SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        feats = np.concatenate([extract_features(arrays[s]) for s in DELTA_SENSORS])
        cache[(case_id, run_id)] = feats

    first_run: dict[int,int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c,r) in cache if c == case_id)
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


# ─── B1: Carry-Forward ────────────────────────────────────────────────────────
def run_carry_forward(proc_clean: pd.DataFrame) -> tuple[float, dict[int,float]]:
    """VB_pred[t] = VB_true[t-1]; first run of each case gets 0."""
    case_rmses: dict[int,float] = {}
    for tc in CASE_SCOPE:
        df = proc_clean[proc_clean["case"] == tc].sort_values("run")
        if df.empty:
            continue
        vb_true = df["VB"].to_numpy(float)
        runs    = df["run"].to_numpy(int)
        # shift: pred[i] = true[i-1], pred[0] = 0
        vb_pred = np.concatenate([[0.0], vb_true[:-1]])
        obs     = np.array([(tc, int(r)) not in NON_OBSERVED_RUNS for r in runs])
        if obs.sum() == 0:
            continue
        case_rmses[tc] = float(np.sqrt(mean_squared_error(vb_true[obs], vb_pred[obs])))
    return float(np.mean(list(case_rmses.values()))), case_rmses


# ─── B2: RunIndex Linear ──────────────────────────────────────────────────────
def run_runindex_linear(proc_clean: pd.DataFrame) -> tuple[float, dict[int,float]]:
    """LinearRegression(run_index → VB) trained on 14 training cases."""
    case_rmses: dict[int,float] = {}
    for tc in CASE_SCOPE:
        train_df = proc_clean[proc_clean["case"] != tc]
        test_df  = proc_clean[proc_clean["case"] == tc].sort_values("run")
        if test_df.empty or train_df.empty:
            continue
        X_train = train_df[["run"]].to_numpy(float)
        y_train = train_df["VB"].to_numpy(float)
        X_test  = test_df[["run"]].to_numpy(float)
        y_true  = test_df["VB"].to_numpy(float)
        runs    = test_df["run"].to_numpy(int)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0.0, None)

        obs = np.array([(tc, int(r)) not in NON_OBSERVED_RUNS for r in runs])
        if obs.sum() == 0:
            continue
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_true[obs], y_pred[obs])))
    return float(np.mean(list(case_rmses.values()))), case_rmses


# ─── B3: Meta+RunIndex Linear ─────────────────────────────────────────────────
def run_meta_linear(proc_clean: pd.DataFrame) -> tuple[float, dict[int,float]]:
    """LinearRegression(DOC, feed, material, run_index → VB) on training cases."""
    feat_cols = META_FEATURES + ["run"]
    case_rmses: dict[int,float] = {}
    for tc in CASE_SCOPE:
        train_df = proc_clean[proc_clean["case"] != tc]
        test_df  = proc_clean[proc_clean["case"] == tc].sort_values("run")
        if test_df.empty or train_df.empty:
            continue
        X_train = train_df[feat_cols].to_numpy(float)
        y_train = train_df["VB"].to_numpy(float)
        X_test  = test_df[feat_cols].to_numpy(float)
        y_true  = test_df["VB"].to_numpy(float)
        runs    = test_df["run"].to_numpy(int)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_pred = np.clip(y_pred, 0.0, None)

        obs = np.array([(tc, int(r)) not in NON_OBSERVED_RUNS for r in runs])
        if obs.sum() == 0:
            continue
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_true[obs], y_pred[obs])))
    return float(np.mean(list(case_rmses.values()))), case_rmses


# ─── B4: Delta+Meta Linear ────────────────────────────────────────────────────
def run_delta_linear(
    cache: dict[tuple[int,int], np.ndarray],
    first_run: dict[int,int],
    proc_clean: pd.DataFrame,
) -> tuple[float, dict[int,float]]:
    """LinearRegression(Delta+Meta → VB) — same features as GRU (AC+vT+vS, 15-dim)."""
    n_delta_feats = len(DELTA_SENSORS) * 4  # 3 sensors × 4 stats = 12
    rows: list[dict] = []
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id)
        if key not in cache:
            continue
        ref_key = (case_id, first_run.get(case_id, run_id))
        ref_vec = cache.get(ref_key, np.zeros(n_delta_feats))
        delta   = cache[key] - ref_vec
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        rec = {"case_id": case_id, "run": run_id, "VB": float(row.VB)}
        for mf in META_FEATURES:
            rec[mf] = float(getattr(row, mf, 0.0))
        for k, val in enumerate(delta):
            rec[f"d{k}"] = float(val)
        rows.append(rec)

    feat_df   = pd.DataFrame(rows)
    feat_cols = [f"d{k}" for k in range(n_delta_feats)] + META_FEATURES
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_df.fillna(0.0, inplace=True)

    case_rmses: dict[int,float] = {}
    for tc in CASE_SCOPE:
        train_df = feat_df[feat_df["case_id"] != tc]
        test_df  = feat_df[feat_df["case_id"] == tc].sort_values("run")
        if test_df.empty or train_df.empty:
            continue
        X_train = train_df[feat_cols].to_numpy(float)
        y_train = train_df["VB"].to_numpy(float)
        X_test  = test_df[feat_cols].to_numpy(float)
        y_true  = test_df["VB"].to_numpy(float)
        runs    = test_df["run"].to_numpy(int)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_pred = np.clip(y_pred, 0.0, None)

        obs = np.array([(tc, int(r)) not in NON_OBSERVED_RUNS for r in runs])
        if obs.sum() == 0:
            continue
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_true[obs], y_pred[obs])))
    return float(np.mean(list(case_rmses.values()))), case_rmses


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_comparison(
    results: dict[str, tuple[float, dict[int,float]]],
    out_dir: Path,
) -> None:
    models      = list(results.keys())
    mean_rmses  = [results[m][0] for m in models]
    case_rmses  = [results[m][1] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: aggregate RMSE bar chart (baseline + reference)
    ax = axes[0]
    all_labels = models + list(REF.keys())
    all_values = mean_rmses + list(REF.values())
    n_base = len(models)
    colors = (
        ["#d62728", "#ff7f0e", "#ffbb78", "#9467bd"] +   # baseline colors
        ["steelblue", "darkorange"]                        # GRU, XGB
    )
    bars = ax.bar(range(len(all_labels)), all_values,
                  color=colors[:len(all_labels)], alpha=0.85)
    for bar, val in zip(bars, all_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Observed-VB RMSE")
    ax.set_title("Aggregate RMSE: Baselines vs Feature-GRU / XGBoost")
    ax.grid(True, axis="y", alpha=0.3)
    # gap annotation
    gru_rmse = list(REF.values())[0]
    for i, (label, val) in enumerate(zip(all_labels[:n_base], mean_rmses)):
        if val > gru_rmse:
            ax.annotate(
                f"Δ={val - gru_rmse:+.4f}",
                xy=(i, val), xytext=(i + 0.1, val + 0.005),
                fontsize=7, color="gray",
            )

    # Right: per-case RMSE heatmap-like plot
    ax = axes[1]
    case_ids = sorted(set.union(*[set(cr.keys()) for cr in case_rmses]))
    data = np.array([
        [results[m][1].get(c, float("nan")) for c in case_ids]
        for m in models
    ])
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=0.3)
    ax.set_xticks(range(len(case_ids)))
    ax.set_xticklabels([f"C{c}" for c in case_ids], fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_title("Per-Case RMSE: Naive Baselines")
    plt.colorbar(im, ax=ax, label="RMSE")

    fig.suptitle("H18_S1: Naive Baseline Comparison — 100% Input, LOCV", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(out_dir / "naive_baseline_comparison.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "naive_baseline_comparison.svg"), bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "H18" / "S1" / f"{ts}_naive_baseline"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H18_S1: Naive Baseline — 100% Input, LOCV 15 Cases ===")
    log("B1: Carry-Forward (no training needed)")
    log("B2: RunIndex Linear (run_index → VB, training cases)")
    log("B3: Meta Linear (DOC+feed+mat+run_index → VB, training cases)")
    log("B4: Delta Linear (Delta+Meta 15-dim → VB, same feats as GRU, training cases)")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    log("Building delta feature cache (AC+vT+vS, 100%)...")
    cache, first_run = build_delta_cache(signal_df, proc_clean)
    log(f"Cache size: {len(cache)} runs")

    results: dict[str, tuple[float, dict[int,float]]] = {}

    # B1: Carry-Forward
    log("\n--- B1: Carry-Forward ---")
    mean_cf, case_cf = run_carry_forward(proc_clean)
    results["CF"] = (mean_cf, case_cf)
    log(f"  CF RMSE = {mean_cf:.6f}")

    # B2: RunIndex Linear
    log("\n--- B2: RunIndex Linear ---")
    mean_rl, case_rl = run_runindex_linear(proc_clean)
    results["RL"] = (mean_rl, case_rl)
    log(f"  RL RMSE = {mean_rl:.6f}")

    # B3: Meta Linear
    log("\n--- B3: Meta+RunIndex Linear ---")
    mean_ml, case_ml = run_meta_linear(proc_clean)
    results["ML"] = (mean_ml, case_ml)
    log(f"  ML RMSE = {mean_ml:.6f}")

    # B4: Delta Linear
    log("\n--- B4: Delta+Meta Linear (AC+vT+vS) ---")
    mean_dl, case_dl = run_delta_linear(cache, first_run, proc_clean)
    results["DL"] = (mean_dl, case_dl)
    log(f"  DL RMSE = {mean_dl:.6f}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    agg_rows = [
        {"model": name, "rmse": val, "description": desc}
        for (name, (val, _)), desc in zip(results.items(), [
            "Carry-forward: VB[t] = VB[t-1]",
            "RunIndex Linear: run_index → VB",
            "Meta Linear: DOC+feed+mat+run_index → VB",
            "Delta Linear: Delta+Meta 15-dim → VB (same feats as GRU, no sequence)",
        ])
    ]
    for k, rv in REF.items():
        agg_rows.append({"model": k, "rmse": rv, "description": "reference from H17_S1"})
    pd.DataFrame(agg_rows).to_csv(out_dir / "metrics" / "aggregate.csv", index=False)

    # Per-case table
    case_ids = sorted(set.union(*[set(cr.keys()) for _, (_, cr) in results.items()]))
    per_case: dict[str, list] = {"case": case_ids}
    for name, (_, case_r) in results.items():
        per_case[name] = [case_r.get(c, float("nan")) for c in case_ids]
    pd.DataFrame(per_case).to_csv(out_dir / "metrics" / "per_case.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_comparison(results, out_dir / "figures")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    log(f"{'Model':<28} {'RMSE':>10}  {'vs GRU':>10}")
    log("-" * 52)
    gru_ref = list(REF.values())[0]
    for name, (val, _) in results.items():
        log(f"{name:<28} {val:>10.6f}  {val - gru_ref:>+10.6f}")
    log("-" * 52)
    for k, rv in REF.items():
        log(f"{k:<28} {rv:>10.6f}  {'(ref)':>10}")

    summary = {
        "experiment": "H18_S1_naive_baseline",
        "baselines": {
            name: {"rmse": val, "per_case": case_r}
            for name, (val, case_r) in results.items()
        },
        "reference_H17_S1": REF,
        "execution_dir": str(out_dir),
    }
    (out_dir / "logs" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text(
        "\n".join(log_lines), encoding="utf-8")

    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

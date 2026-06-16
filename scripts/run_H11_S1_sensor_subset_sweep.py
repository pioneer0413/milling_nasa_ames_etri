#!/usr/bin/env python3
"""H11_S1: 2^6 - 1 = 63 sensor subset complete sweep → XGBoost + Random Forest.

Feature per subset:
  delta features: (mean, rms, std, peak) × n_included_sensors  (4n-dim)
  meta features : DOC, feed, material                            (3-dim)
  Total: 4n + 3 features

Sensors: smcAC, smcDC, vib_table, vib_spindle, AE_table, AE_spindle
Subset encoding: 6-bit mask (bit 0 = smcAC ... bit 5 = AE_spindle)

Models:
  XGBoost: g_lr0.03_n600_d3_mcw8 (H4_S9 best)
  RF:      n_estimators=500, max_features="sqrt", min_samples_leaf=2
  Prefix: 100%, Seeds: [0, 1, 2], LOCV 15 cases

Output:
  experiments/executions/H11/S1/{timestamp}_sensor_subset_sweep/
"""
from __future__ import annotations

import itertools
import json
import sys
import time as time_mod
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS       = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
SENSOR_ABBR   = {"smcAC": "AC", "smcDC": "DC", "vib_table": "vT",
                 "vib_spindle": "vS", "AE_table": "aT", "AE_spindle": "aS"}
EXCLUDED_RUNS = {(2, 1), (12, 1)}
NON_OBSERVED_RUNS = {
    (1,2),(1,3),(1,5),(1,16),(2,6),(8,4),(11,7),(11,14),(11,17),
    (12,4),(12,10),(13,1),(13,2),(14,1),(14,5),(15,1),(16,1),(16,2),(16,4),
}
META_FEATURES = ["DOC", "feed", "material"]
SEEDS         = [0, 1, 2]
PREFIX        = 100
THRESH        = 1e6
STATS         = ["mean", "rms", "std", "peak"]

XGB_CFG = dict(
    learning_rate=0.03, n_estimators=600, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.9,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, n_jobs=4,
)
RF_CFG = dict(
    n_estimators=500, max_features="sqrt", min_samples_leaf=2, n_jobs=4,
)

N_SENSORS = len(SENSORS)
ALL_MASKS = list(range(1, 2 ** N_SENSORS))  # 1..63


def mask_to_sensors(mask: int) -> list[str]:
    return [SENSORS[i] for i in range(N_SENSORS) if (mask >> i) & 1]


def mask_label(mask: int) -> str:
    return "+".join(SENSOR_ABBR[s] for s in mask_to_sensors(mask))


# ─── Signal parsing & feature extraction ──────────────────────────────────────
def parse_signal(value: object) -> np.ndarray:
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_4stat(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros(1)
    return np.array([
        np.mean(arr),
        np.sqrt(np.mean(arr ** 2)),
        np.std(arr),
        np.abs(arr).max(),
    ], dtype=np.float64)


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


# ─── Full feature cache (24-dim raw per run) ──────────────────────────────────
def build_raw_cache(
    signal_df: pd.DataFrame, process_df: pd.DataFrame
) -> tuple[dict[tuple[int, int], np.ndarray], dict[int, int]]:
    """raw_cache[(case, run)] = 24-dim raw feature vector (4-stat × 6 sensors)."""
    raw_cache: dict[tuple[int, int], np.ndarray] = {}
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
        end = {s: max(1, int(np.ceil(len(a) * PREFIX / 100.0))) for s, a in arrays.items()}
        raw_vec = np.concatenate([extract_4stat(arrays[s][:end[s]]) for s in SENSORS])
        raw_cache[(case_id, run_id)] = raw_vec

    first_run: dict[int, int] = {}
    for case_id in CASE_SCOPE:
        runs = sorted(r for (c, r) in raw_cache
                      if c == case_id and (c, r) not in EXCLUDED_RUNS)
        if runs:
            first_run[case_id] = runs[0]
    return raw_cache, first_run


def build_subset_df(
    raw_cache: dict[tuple[int, int], np.ndarray],
    first_run: dict[int, int],
    process_df: pd.DataFrame,
    mask: int,
) -> pd.DataFrame:
    """Build delta feature DataFrame for a given sensor mask."""
    sensor_indices = []
    for i, sensor in enumerate(SENSORS):
        if (mask >> i) & 1:
            sensor_indices.extend([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])

    rows: list[dict] = []
    for row in process_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        if (case_id, run_id) in EXCLUDED_RUNS:
            continue
        key = (case_id, run_id)
        if key not in raw_cache:
            continue
        ref_run = first_run.get(case_id, run_id)
        raw_vec  = raw_cache[key]
        ref_vec  = raw_cache.get((case_id, ref_run), np.zeros_like(raw_vec))
        delta    = (raw_vec - ref_vec)[sensor_indices]
        vb       = float(row.VB) if not pd.isna(row.VB) else 0.0
        rec: dict = {"case_id": case_id, "run": run_id, "VB": vb}
        for mf in META_FEATURES:
            rec[mf] = float(getattr(row, mf, 0.0))
        for k, val in enumerate(delta):
            rec[f"f{k}"] = float(val) if np.isfinite(val) else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


# ─── LOCV runner ──────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def run_locv_seed(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    model,
) -> float:
    rmses = []
    for tc in CASE_SCOPE:
        test_df  = feat_df[feat_df["case_id"] == tc]
        train_df = feat_df[feat_df["case_id"] != tc]
        if test_df.empty or train_df.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_df[feature_cols], train_df["VB"])
            y_pred = model.predict(test_df[feature_cols])
        obs = test_df.apply(
            lambda r: (int(r["case_id"]), int(r["run"])) not in NON_OBSERVED_RUNS, axis=1
        ).to_numpy()
        y_true = test_df["VB"].to_numpy(float)[obs]
        y_pred = y_pred[obs]
        if len(y_true) == 0:
            continue
        rmses.append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
    return float(np.mean(rmses)) if rmses else float("nan")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "H11" / "S1" / f"{ts}_sensor_subset_sweep"
    for sub in ["metrics", "analysis", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H11_S1: Sensor Subset Complete Sweep (2^6 - 1 = 63 subsets) ===")
    log(f"SENSORS = {SENSORS}")
    log(f"Prefix = {PREFIX}%, Seeds = {SEEDS}, LOCV = {len(CASE_SCOPE)} cases")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    log("Building raw feature cache (prefix=100%)...")
    raw_cache, first_run = build_raw_cache(signal_df, proc_clean)
    log(f"Cached {len(raw_cache)} runs")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    results: list[dict] = []
    t_start = time_mod.time()

    for mask_idx, mask in enumerate(ALL_MASKS):
        sensors  = mask_to_sensors(mask)
        label    = mask_label(mask)
        n_feat   = len(sensors) * 4 + 3
        feat_df  = build_subset_df(raw_cache, first_run, proc_clean, mask)
        feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat_df.fillna(0.0, inplace=True)
        delta_cols = [c for c in feat_df.columns if c.startswith("f") and c[1:].isdigit()]
        feat_cols  = delta_cols + META_FEATURES

        xgb_rmses, rf_rmses = [], []
        for seed in SEEDS:
            xgb = XGBRegressor(**XGB_CFG, random_state=seed, verbosity=0)
            rf  = RandomForestRegressor(**RF_CFG, random_state=seed)
            xgb_rmses.append(run_locv_seed(feat_df, feat_cols, xgb))
            rf_rmses.append(run_locv_seed(feat_df, feat_cols, rf))

        xgb_mean = float(np.mean(xgb_rmses))
        rf_mean  = float(np.mean(rf_rmses))
        results.append({
            "mask": mask, "label": label, "n_sensors": len(sensors),
            "n_feat": n_feat, "sensors": sensors,
            "xgb_mean_rmse": xgb_mean, "xgb_std_rmse": float(np.std(xgb_rmses)),
            "rf_mean_rmse":  rf_mean,  "rf_std_rmse":  float(np.std(rf_rmses)),
        })

        if (mask_idx + 1) % 10 == 0 or mask_idx == 0:
            elapsed = time_mod.time() - t_start
            log(f"  [{mask_idx+1:2d}/63]  {label:<32}  XGB={xgb_mean:.4f}  RF={rf_mean:.4f}  [{elapsed:.0f}s]")

    log(f"\nTotal sweep time: {time_mod.time()-t_start:.1f}s")

    # ── Results ───────────────────────────────────────────────────────────────
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "metrics" / "all_subset_results.csv", index=False)

    # Per-size statistics
    size_agg = res_df.groupby("n_sensors").agg(
        xgb_min=("xgb_mean_rmse", "min"),
        xgb_mean=("xgb_mean_rmse", "mean"),
        rf_min=("rf_mean_rmse", "min"),
        rf_mean=("rf_mean_rmse", "mean"),
        n_subsets=("mask", "count"),
    ).reset_index()
    size_agg.to_csv(out_dir / "analysis" / "size_summary.csv", index=False)

    # Top-10 per model
    top_xgb = res_df.nsmallest(10, "xgb_mean_rmse")
    top_rf  = res_df.nsmallest(10, "rf_mean_rmse")
    top_xgb.to_csv(out_dir / "analysis" / "top10_xgb.csv", index=False)
    top_rf.to_csv(out_dir / "analysis" / "top10_rf.csv", index=False)

    # Full-sensor baseline (mask = 63)
    full_mask = 2 ** N_SENSORS - 1  # 63
    full_row  = res_df[res_df["mask"] == full_mask].iloc[0]

    # ── Print ─────────────────────────────────────────────────────────────────
    log("\n=== TOP-10 SUBSETS (XGBoost, 3-seed mean LOCV RMSE) ===")
    log(f"{'Rank':<5} {'Subset':<34} {'n':>3} {'XGB_RMSE':>10} {'RF_RMSE':>10}")
    log("-" * 65)
    for rank, (_, row) in enumerate(top_xgb.iterrows(), 1):
        log(f"{rank:<5} {row['label']:<34} {int(row['n_sensors']):>3} "
            f"{row['xgb_mean_rmse']:>10.6f} {row['rf_mean_rmse']:>10.6f}")

    log("\n=== TOP-10 SUBSETS (Random Forest, 3-seed mean LOCV RMSE) ===")
    log(f"{'Rank':<5} {'Subset':<34} {'n':>3} {'XGB_RMSE':>10} {'RF_RMSE':>10}")
    log("-" * 65)
    for rank, (_, row) in enumerate(top_rf.iterrows(), 1):
        log(f"{rank:<5} {row['label']:<34} {int(row['n_sensors']):>3} "
            f"{row['xgb_mean_rmse']:>10.6f} {row['rf_mean_rmse']:>10.6f}")

    log("\n=== SINGLE-SENSOR PERFORMANCE ===")
    single = res_df[res_df["n_sensors"] == 1].sort_values("xgb_mean_rmse")
    log(f"{'Sensor':<14} {'n_feat':>6} {'XGB_RMSE':>10} {'RF_RMSE':>10}")
    log("-" * 44)
    for _, row in single.iterrows():
        log(f"{row['label']:<14} {int(row['n_feat']):>6} "
            f"{row['xgb_mean_rmse']:>10.6f} {row['rf_mean_rmse']:>10.6f}")

    log("\n=== PER-SIZE MIN/MEAN RMSE ===")
    log(f"{'n_sensors':>10} {'n_subsets':>10} {'XGB_min':>10} {'XGB_mean':>10} {'RF_min':>10} {'RF_mean':>10}")
    log("-" * 62)
    for _, row in size_agg.iterrows():
        log(f"{int(row['n_sensors']):>10} {int(row['n_subsets']):>10} "
            f"{row['xgb_min']:>10.6f} {row['xgb_mean']:>10.6f} "
            f"{row['rf_min']:>10.6f} {row['rf_mean']:>10.6f}")

    log(f"\n=== FULL SENSOR BASELINE (all 6 sensors) ===")
    log(f"  XGB RMSE = {full_row['xgb_mean_rmse']:.6f}")
    log(f"  RF  RMSE = {full_row['rf_mean_rmse']:.6f}")

    best_xgb = top_xgb.iloc[0]
    best_rf  = top_rf.iloc[0]
    log(f"\n=== BEST SUBSETS ===")
    log(f"  XGB best: {best_xgb['label']}  RMSE={best_xgb['xgb_mean_rmse']:.6f}"
        f"  vs full-sensor {full_row['xgb_mean_rmse']:.6f}"
        f"  Δ={best_xgb['xgb_mean_rmse']-full_row['xgb_mean_rmse']:+.6f}")
    log(f"  RF  best: {best_rf['label']}  RMSE={best_rf['rf_mean_rmse']:.6f}"
        f"  vs full-sensor {full_row['rf_mean_rmse']:.6f}"
        f"  Δ={best_rf['rf_mean_rmse']-full_row['rf_mean_rmse']:+.6f}")

    # Sensor appearance frequency in top-10
    log("\n=== SENSOR FREQUENCY IN TOP-10 (XGBoost) ===")
    freq: dict[str, int] = {s: 0 for s in SENSORS}
    for _, row in top_xgb.iterrows():
        for s in row["sensors"]:
            freq[s] += 1
    for s in sorted(SENSORS, key=lambda x: -freq[x]):
        log(f"  {SENSOR_ABBR[s]}: {freq[s]}/10")

    # Summary JSON
    summary = {
        "experiment": "H11_S1_sensor_subset_sweep",
        "prefix": PREFIX, "seeds": SEEDS, "n_subsets": len(ALL_MASKS),
        "best_xgb": {"label": best_xgb["label"], "rmse": best_xgb["xgb_mean_rmse"],
                     "n_sensors": int(best_xgb["n_sensors"])},
        "best_rf":  {"label": best_rf["label"],  "rmse": best_rf["rf_mean_rmse"],
                     "n_sensors": int(best_rf["n_sensors"])},
        "full_sensor_xgb_rmse": float(full_row["xgb_mean_rmse"]),
        "full_sensor_rf_rmse":  float(full_row["rf_mean_rmse"]),
        "top10_xgb": top_xgb[["label", "n_sensors", "xgb_mean_rmse", "rf_mean_rmse"]].to_dict("records"),
        "top10_rf":  top_rf[["label",  "n_sensors", "xgb_mean_rmse", "rf_mean_rmse"]].to_dict("records"),
        "size_summary": size_agg.to_dict("records"),
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

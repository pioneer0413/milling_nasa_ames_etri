#!/usr/bin/env python3
"""H10_S1: Extended feature set (26 types) → XGBoost + Random Forest, prefix=100%.

Feature types per sensor (25개, rank 3 Latent 제외):
  Time domain (15): min, max, sum, mean, rms, std, var, peak(=max_abs),
                    peak_to_peak, skewness, kurtosis, crest_factor,
                    shape_factor, impulse_factor, clearance_factor
  Power (3):        mp, rmsp, rvp
  Frequency (5):    gcf, msf, vf, spectral_skewness, spectral_kurtosis
  Wavelet DWT (4):  wt_d1, wt_d2, wt_d3, wt_a3  (db4, level-3 DWT band energy)
  → 27 features per sensor × 6 sensors = 162 raw features

Cross-correlation (15 pairs): max |Pearson r| between each sensor pair → 15 raw features
Meta (3): DOC, feed, material

Delta = raw - first_run_raw  (per-sensor 162 features)
Total input: 162 delta + 15 cross-corr (raw) + 3 meta = 180 features

Baseline: 24 delta (mean, rms, std, peak × 6 sensors) + 3 meta = 27 features

Models:
  XGBoost: g_lr0.03_n600_d3_mcw8 (H4_S9 best)
  RF:      n_estimators=500, max_features="sqrt", min_samples_leaf=2
  Both: prefix=100%, 3 seeds, LOCV 15 cases

Output:
  experiments/executions/H10/S1/{timestamp}_extended_features/
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis as scipy_kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS       = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
EXCLUDED_RUNS = {(2, 1), (12, 1)}
META_FEATURES = ["DOC", "feed", "material"]
SEEDS         = [0, 1, 2]
PREFIX        = 100
THRESH        = 1e6
WAVELET       = "db4"
WAVELET_LEVEL = 3  # → d1, d2, d3, a3 (4 bands)

XGB_CFG = dict(
    learning_rate=0.03, n_estimators=600, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.9,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, n_jobs=4,
)
RF_CFG = dict(
    n_estimators=500, max_features="sqrt", min_samples_leaf=2, n_jobs=4,
)


# ─── Signal parsing ───────────────────────────────────────────────────────────
def parse_signal(value: object) -> np.ndarray:
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# ─── Extended feature extraction (per sensor) ─────────────────────────────────
def extract_extended(arr: np.ndarray) -> np.ndarray:
    """27 features per sensor signal."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size < 4:
        return np.zeros(27)

    eps = 1e-10
    abs_arr    = np.abs(arr)
    arr_sq     = arr ** 2
    arr_power  = arr_sq           # instantaneous power
    mean_abs   = max(float(np.mean(abs_arr)), eps)
    mean_sqrt  = max(float(np.mean(np.sqrt(abs_arr))), eps)

    # Time domain (15)
    t_min        = float(np.min(arr))
    t_max        = float(np.max(arr))
    t_sum        = float(np.sum(arr))
    t_mean       = float(np.mean(arr))
    t_rms        = float(np.sqrt(np.mean(arr_sq)))
    t_std        = float(np.std(arr))
    t_var        = float(np.var(arr))
    t_peak       = float(np.max(abs_arr))          # max absolute value
    t_ptp        = float(t_max - t_min)            # peak-to-peak
    t_skew       = float(skew(arr))
    t_kurt       = float(scipy_kurtosis(arr))      # Fisher definition (excess kurtosis)
    t_crest      = t_peak / max(t_rms, eps)
    t_shape      = t_rms / mean_abs
    t_impulse    = t_peak / mean_abs
    t_clearance  = t_peak / (mean_sqrt ** 2)

    # Power statistics (3)
    p_mp   = float(np.mean(arr_power))             # mean power
    p_rmsp = float(np.sqrt(np.mean(arr_power ** 2)))  # rms of power (√E[x⁴])
    p_rvp  = float(np.sqrt(np.var(arr_power)))     # root variance power (std of power)

    # Frequency domain (5) — one-sided FFT magnitudes
    n     = len(arr)
    fft   = np.abs(np.fft.rfft(arr))
    freqs = np.fft.rfftfreq(n)                     # normalized freq [0, 0.5]
    fft2  = fft ** 2
    sum_fft2 = max(float(np.sum(fft2)), eps)

    gcf  = float(np.sum(freqs * fft2) / sum_fft2)
    msf  = float(np.sum(freqs ** 2 * fft2) / sum_fft2)
    vf   = float(np.sqrt(max(msf - gcf ** 2, 0.0)))
    sp_skew = float(skew(fft2))
    sp_kurt = float(scipy_kurtosis(fft2))

    # Wavelet DWT band energy (4 bands: d1, d2, d3, a3)
    coeffs   = pywt.wavedec(arr, WAVELET, level=WAVELET_LEVEL)
    # coeffs = [cA3, cD3, cD2, cD1]
    wt_a3 = float(np.sum(coeffs[0] ** 2))
    wt_d3 = float(np.sum(coeffs[1] ** 2))
    wt_d2 = float(np.sum(coeffs[2] ** 2))
    wt_d1 = float(np.sum(coeffs[3] ** 2))

    return np.array([
        t_min, t_max, t_sum, t_mean, t_rms, t_std, t_var, t_peak, t_ptp,
        t_skew, t_kurt, t_crest, t_shape, t_impulse, t_clearance,
        p_mp, p_rmsp, p_rvp,
        gcf, msf, vf, sp_skew, sp_kurt,
        wt_d1, wt_d2, wt_d3, wt_a3,
    ], dtype=np.float64)


EXT_STAT_NAMES = [
    "min", "max", "sum", "mean", "rms", "std", "var", "peak", "ptp",
    "skew", "kurt", "crest", "shape", "impulse", "clearance",
    "mp", "rmsp", "rvp",
    "gcf", "msf", "vf", "sp_skew", "sp_kurt",
    "wt_d1", "wt_d2", "wt_d3", "wt_a3",
]  # 27 names


def ext_col_names() -> list[str]:
    return [f"{s}__{stat}" for s in SENSORS for stat in EXT_STAT_NAMES]


SENSOR_ABBR_MAP = {
    "smcAC": "AC", "smcDC": "DC",
    "vib_table": "vT", "vib_spindle": "vS",
    "AE_table": "aT", "AE_spindle": "aS",
}


def cross_corr_col_names() -> list[str]:
    pairs = []
    for i, s1 in enumerate(SENSORS):
        for j, s2 in enumerate(SENSORS):
            if j > i:
                pairs.append(f"xcorr__{SENSOR_ABBR_MAP[s1]}_{SENSOR_ABBR_MAP[s2]}")
    return pairs  # 15 pairs


def baseline_col_names() -> list[str]:
    return [f"{s}__delta_{f}" for s in SENSORS for f in ["mean", "rms", "std", "peak"]]


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


# ─── Feature matrix builder ───────────────────────────────────────────────────
def build_feature_matrices(
    signal_df: pd.DataFrame, process_df: pd.DataFrame, log
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (extended_df, baseline_df)."""
    ext_cols   = ext_col_names()
    xcorr_cols = cross_corr_col_names()
    base_cols  = baseline_col_names()

    # Raw feature cache: (case, run) → 27×6=162 dim array
    raw_cache:  dict[tuple[int, int], np.ndarray] = {}
    arr_cache:  dict[tuple[int, int], dict[str, np.ndarray]] = {}

    total = len(process_df)
    for idx, row in enumerate(process_df.itertuples(index=False)):
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
        cropped = {s: a[:end[s]] for s, a in arrays.items()}

        ext_vec = np.concatenate([extract_extended(cropped[s]) for s in SENSORS])
        raw_cache[(case_id, run_id)]  = ext_vec
        arr_cache[(case_id, run_id)]  = cropped
        if (idx + 1) % 50 == 0:
            log(f"  cached {idx+1}/{total} runs")

    # first_run per case
    first_run: dict[int, int] = {}
    for case_id in CASE_SCOPE:
        runs = sorted(r for (c, r) in raw_cache if c == case_id and (c, r) not in EXCLUDED_RUNS)
        if runs:
            first_run[case_id] = runs[0]

    ext_rows:  list[dict] = []
    base_rows: list[dict] = []

    for row in process_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        if (case_id, run_id) in EXCLUDED_RUNS:
            continue
        if (case_id, run_id) not in raw_cache:
            continue
        ref_run = first_run.get(case_id, run_id)
        vb = float(row.VB) if not pd.isna(row.VB) else 0.0

        ext_vec = raw_cache[(case_id, run_id)]
        ref_vec = raw_cache.get((case_id, ref_run), np.zeros_like(ext_vec))
        delta_ext = ext_vec - ref_vec

        # Cross-correlation between sensor pairs (raw)
        arrs = arr_cache[(case_id, run_id)]
        xcorr_vals = []
        for i, s1 in enumerate(SENSORS):
            for j, s2 in enumerate(SENSORS):
                if j > i:
                    a1, a2 = arrs[s1], arrs[s2]
                    n = min(len(a1), len(a2))
                    if n < 4:
                        xcorr_vals.append(0.0)
                    else:
                        corr = float(np.corrcoef(a1[:n], a2[:n])[0, 1])
                        xcorr_vals.append(corr if np.isfinite(corr) else 0.0)

        # Extended feature row
        rec_ext: dict = {"case_id": case_id, "run": run_id, "VB": vb}
        for mf in META_FEATURES:
            rec_ext[mf] = float(getattr(row, mf, 0.0))
        for col, val in zip(ext_cols, delta_ext):
            rec_ext[col] = float(val) if np.isfinite(val) else 0.0
        for col, val in zip(xcorr_cols, xcorr_vals):
            rec_ext[col] = float(val)
        ext_rows.append(rec_ext)

        # Baseline feature row (delta: mean, rms, std, peak × 6 sensors)
        baseline_stats = ["mean", "rms", "std", "peak"]
        stat_indices   = [EXT_STAT_NAMES.index(s) for s in ["mean", "rms", "std", "peak"]]
        n_stats = len(EXT_STAT_NAMES)  # 27
        rec_base: dict = {"case_id": case_id, "run": run_id, "VB": vb}
        for mf in META_FEATURES:
            rec_base[mf] = float(getattr(row, mf, 0.0))
        for si, sensor in enumerate(SENSORS):
            for stat_name, stat_idx in zip(baseline_stats, stat_indices):
                col = f"{sensor}__delta_{stat_name}"
                val = delta_ext[si * n_stats + stat_idx]
                rec_base[col] = float(val) if np.isfinite(val) else 0.0
        base_rows.append(rec_base)

    ext_df  = pd.DataFrame(ext_rows)
    base_df = pd.DataFrame(base_rows)
    return ext_df, base_df


# ─── LOCV runners ─────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def run_locv(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    model_fn,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for tc in CASE_SCOPE:
        test_df  = feat_df[feat_df["case_id"] == tc]
        train_df = feat_df[feat_df["case_id"] != tc]
        if test_df.empty or train_df.empty:
            continue
        model = model_fn(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_df[feature_cols], train_df["VB"])
            y_pred = model.predict(test_df[feature_cols])
        rows.append({
            "target_case": tc,
            "n_runs": len(test_df),
            **compute_metrics(test_df["VB"].to_numpy(float), y_pred),
        })
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "H10" / "S1" / f"{ts}_extended_features"
    for sub in ["metrics", "analysis", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H10_S1: Extended Features → XGBoost + Random Forest ===")
    log(f"Prefix: {PREFIX}%, Seeds: {SEEDS}")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/processed/mill_signal_data.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/processed/mill_process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    log("\nBuilding feature matrices...")
    ext_df, base_df = build_feature_matrices(signal_df, proc_clean, log)

    ext_cols   = ext_col_names()
    xcorr_cols = cross_corr_col_names()
    base_cols  = baseline_col_names()
    ext_feat_cols  = ext_cols + xcorr_cols + META_FEATURES   # 162 delta + 15 xcorr + 3 meta = 180
    base_feat_cols = base_cols + META_FEATURES                # 24 delta + 3 meta = 27

    log(f"Extended feature dim: {len(ext_feat_cols)} (162 delta + 15 xcorr + 3 meta)")
    log(f"Baseline feature dim: {len(base_feat_cols)} (24 delta + 3 meta)")

    # Replace NaN/Inf in feature matrix
    for df in [ext_df, base_df]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0.0, inplace=True)

    def xgb_fn(seed):
        return XGBRegressor(**XGB_CFG, random_state=seed, verbosity=0)

    def rf_fn(seed):
        return RandomForestRegressor(**RF_CFG, random_state=seed)

    variants = [
        ("XGB_baseline",   base_df, base_feat_cols, xgb_fn),
        ("XGB_extended",   ext_df,  ext_feat_cols,  xgb_fn),
        ("RF_baseline",    base_df, base_feat_cols,  rf_fn),
        ("RF_extended",    ext_df,  ext_feat_cols,   rf_fn),
    ]

    all_rows: list[dict] = []
    import time as time_mod
    for variant, feat_df, feat_cols, model_fn in variants:
        log(f"\n--- {variant} ({len(feat_cols)} features) ---")
        t0 = time_mod.time()
        for seed in SEEDS:
            df = run_locv(feat_df, feat_cols, model_fn, seed)
            locv = float(df["rmse"].mean())
            log(f"  seed={seed}  LOCV_RMSE={locv:.6f}  [{time_mod.time()-t0:.0f}s]")
            for rec in df.to_dict("records"):
                all_rows.append({**rec, "seed": seed, "variant": variant})

    # ── Aggregate ─────────────────────────────────────────────────────────────
    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "metrics" / "all_per_case_metrics.csv", index=False)

    agg = (all_df.groupby(["variant", "seed"])["rmse"].mean()
           .groupby("variant").agg(mean_rmse="mean", std_rmse="std")
           .reset_index())
    agg["std_rmse"] = agg["std_rmse"].fillna(0.0)
    agg.to_csv(out_dir / "analysis" / "aggregate_comparison.csv", index=False)

    per_case = (all_df.groupby(["variant", "target_case"])["rmse"].mean()
                .unstack("variant").reindex(index=CASE_SCOPE).reset_index())
    per_case.columns.name = None
    per_case.to_csv(out_dir / "analysis" / "per_case_comparison.csv", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    ORDER = ["XGB_baseline", "XGB_extended", "RF_baseline", "RF_extended"]
    def get(v):
        r = agg[agg["variant"] == v]
        return (float(r["mean_rmse"].iloc[0]), float(r["std_rmse"].iloc[0])) if not r.empty else (float("nan"), 0.0)

    log("\n=== AGGREGATE RESULTS (3-seed mean LOCV RMSE) ===")
    log(f"{'Variant':<22} {'feat_dim':>8} {'mean_RMSE':>10} {'std_RMSE':>10}")
    log("-" * 54)
    dims = {"XGB_baseline": len(base_feat_cols), "XGB_extended": len(ext_feat_cols),
            "RF_baseline":  len(base_feat_cols), "RF_extended":  len(ext_feat_cols)}
    for v in ORDER:
        m, s = get(v)
        log(f"{v:<22} {dims[v]:>8} {m:>10.6f} {s:>10.6f}")

    log("\n=== DELTA vs baseline ===")
    for model in ["XGB", "RF"]:
        base_m, _ = get(f"{model}_baseline")
        ext_m,  _ = get(f"{model}_extended")
        d = ext_m - base_m
        sign = "+" if d > 0 else ""
        tag  = "WORSE" if d > 0 else "BETTER"
        log(f"[{model}] baseline={base_m:.6f}  extended={ext_m:.6f}  Δ={sign}{d:.6f} ({sign}{d/base_m*100:.2f}%)  [{tag}]")

    log("\n=== PER-CASE RMSE (3-seed mean) ===")
    cols = ORDER
    header = f"{'Case':>5}" + "".join(f" {c[:12]:>13}" for c in cols)
    log(header)
    log("-" * len(header))
    for _, row in per_case.iterrows():
        cid  = int(row["target_case"])
        vals = [row.get(c, float("nan")) for c in cols]
        line = f"{cid:>5}" + "".join(f" {v:>13.4f}" if pd.notna(v) else "           N/A" for v in vals)
        log(line)

    summary = {
        "experiment": "H10_S1_extended_features",
        "prefix": PREFIX, "seeds": SEEDS,
        "n_ext_features": len(ext_feat_cols), "n_base_features": len(base_feat_cols),
        "aggregate": agg.to_dict("records"),
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

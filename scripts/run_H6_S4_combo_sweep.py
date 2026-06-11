#!/usr/bin/env python3
"""H6_S4: Multi-dimensional combinatorial sweep.

Axes swept (all cross-products of selected values):
  feature_names   subset of {mean, std, peak, rms, skewness, kurtosis}
  sensors         subset of {smcAC, smcDC, AE_spindle, AE_table, vib_spindle, vib_table}
  prefix_percent  input ratio 10~100%
  feature_type    raw | delta | raw_meta | delta_meta
  model           xgboost (default) | rf | ridge | svr | lgbm

All models use per-run tabular features; LOCV over 15 cases, 3 seeds.

Feature types:
  raw        raw statistical features from signal
  delta      feature(run_t) - feature(run_1)  (change vs first run)
  raw_meta   raw + DOC / feed / material
  delta_meta delta + DOC / feed / material

SMOKE test: 2 features, 2 sensors, 1 prefix, 2 types, xgboost, seed=0, 5 cases.
  -> covers both raw and delta code-paths quickly.

Usage:
  python run_H6_S4_combo_sweep.py --smoke
  python run_H6_S4_combo_sweep.py --prefixes 50 80 100 --models xgboost rf
  python run_H6_S4_combo_sweep.py  # full default sweep

Output: experiments/executions/H6/S4/{timestamp}_{smoke_test|combo_sweep}/
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
ALL_SENSORS = ["smcAC", "smcDC", "AE_spindle", "AE_table", "vib_spindle", "vib_table"]
# Map user-facing sensor names to actual CSV column names
_SENSOR_COL = {
    "smcAC": "smcAC", "smcDC": "smcDC",
    "AE_spindle": "AE_spindle", "AE_table": "AE_table",
    "vib_spindle": "vib_spindle", "vib_table": "vib_table",
}
ALL_FEATURES = ["mean", "std", "peak", "rms", "skewness", "kurtosis"]
META_FEATURES = ["DOC", "feed", "material"]

# Abbreviations for compact terminal output
_FEAT_ABBR = {"mean": "Mn", "std": "Sd", "peak": "Pk", "rms": "RM", "skewness": "Sk", "kurtosis": "Ku"}
_SENS_ABBR = {
    "smcAC": "AC", "smcDC": "DC",
    "AE_spindle": "AS", "AE_table": "AT",
    "vib_spindle": "VS", "vib_table": "VT",
}
EXCLUDED_RUNS = {(2, 1), (12, 1)}
THRESH = 1e6

DEFAULT_PREFIXES = list(range(10, 101, 10))
DEFAULT_FEATURE_TYPES = ["raw", "delta", "raw_meta", "delta_meta", "raw_delta", "raw_delta_meta"]
DEFAULT_MODELS = ["xgboost"]

# ─── Signal parsing ───────────────────────────────────────────────────────────
def _parse_signal(value: object) -> np.ndarray:
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _extract_features(arr: np.ndarray) -> dict[str, float]:
    """Compute all 6 candidate statistics from a 1-D signal array."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros(1)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    if std_val > 1e-12:
        z = (arr - mean_val) / std_val
        skewness = float(np.mean(z ** 3))
        kurtosis = float(np.mean(z ** 4))
    else:
        skewness, kurtosis = 0.0, 0.0
    return {
        "mean": mean_val,
        "std": std_val,
        "peak": float(np.abs(arr).max()),
        "rms": float(np.sqrt(np.mean(arr ** 2))),
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


# ─── Preprocessing ────────────────────────────────────────────────────────────
def _preprocess(process: pd.DataFrame) -> pd.DataFrame:
    df = process.copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"] == min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside")
    )
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)


def build_master_matrix(
    signal_df: pd.DataFrame, process_df: pd.DataFrame, prefix_percents: list[int]
) -> pd.DataFrame:
    """Build master feature matrix for all sensors × features × prefixes.

    Returns a DataFrame with columns:
      case_id, run, signal_prefix_percent, VB, DOC, feed, material,
      {sensor}_{feat}           (raw statistical feature),
      {sensor}_{feat}__ref      (same feature on the first run of that case),
      {sensor}_{feat}__delta    (raw - ref).
    """
    sig_cols = list(_SENSOR_COL.values())  # actual CSV column names

    # 1. Parse all signals into memory
    sig_cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for row in signal_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        if (case_id, run_id) in EXCLUDED_RUNS:
            continue
        arrays: dict[str, np.ndarray] = {}
        valid = True
        for col in sig_cols:
            arr = _parse_signal(getattr(row, col))
            if np.abs(arr).max() > THRESH:
                valid = False
                break
            arrays[col] = arr
        if valid:
            sig_cache[(case_id, run_id)] = arrays

    # 2. Identify first (reference) run per case
    first_run: dict[int, int] = {}
    for case_id in process_df["case"].unique():
        runs = sorted(r for (c, r) in sig_cache if c == int(case_id))
        if runs:
            first_run[int(case_id)] = runs[0]

    # 3. Build rows
    rows: list[dict] = []
    for row in process_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        if (case_id, run_id) in EXCLUDED_RUNS:
            continue
        arrays = sig_cache.get((case_id, run_id))
        if arrays is None:
            continue
        ref_run = first_run.get(case_id, run_id)
        ref_arrays = sig_cache.get((case_id, ref_run), arrays)
        vb = float(row.VB) if not pd.isna(row.VB) else 0.0
        base_len = min(len(a) for a in arrays.values())
        ref_len = min(len(a) for a in ref_arrays.values())
        for pct in prefix_percents:
            end = max(1, int(np.ceil(base_len * pct / 100.0)))
            ref_end = max(1, int(np.ceil(ref_len * pct / 100.0)))
            record: dict = {
                "case_id": case_id, "run": run_id,
                "signal_prefix_percent": float(pct),
                "VB": vb,
                "DOC": float(getattr(row, "DOC", 0.0)),
                "feed": float(getattr(row, "feed", 0.0)),
                "material": int(getattr(row, "material", 0)),
            }
            for sensor_key, col in _SENSOR_COL.items():
                raw_f = _extract_features(arrays[col][:end])
                ref_f = _extract_features(ref_arrays[col][:ref_end])
                for feat in ALL_FEATURES:
                    record[f"{sensor_key}_{feat}"] = raw_f[feat]
                    record[f"{sensor_key}_{feat}__ref"] = ref_f[feat]
                    record[f"{sensor_key}_{feat}__delta"] = raw_f[feat] - ref_f[feat]
            rows.append(record)

    return pd.DataFrame(rows).sort_values(
        ["signal_prefix_percent", "case_id", "run"]
    ).reset_index(drop=True)


# ─── Feature column selection ─────────────────────────────────────────────────
def get_feature_cols(
    sensors: list[str], feature_names: list[str], feature_type: str
) -> list[str]:
    raw_cols = [f"{s}_{f}" for s in sensors for f in feature_names]
    delta_cols = [f"{s}_{f}__delta" for s in sensors for f in feature_names]
    if feature_type == "raw":
        base = raw_cols
    elif feature_type == "delta":
        base = delta_cols
    elif feature_type == "raw_meta":
        base = raw_cols
    elif feature_type == "delta_meta":
        base = delta_cols
    elif feature_type == "raw_delta":
        base = raw_cols + delta_cols
    elif feature_type == "raw_delta_meta":
        base = raw_cols + delta_cols
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    return base + META_FEATURES if "meta" in feature_type else base


# ─── Model factory ────────────────────────────────────────────────────────────
def make_model(model_name: str, seed: int):
    if model_name == "xgboost":
        return XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.9,
            random_state=seed, verbosity=0, n_jobs=4,
        )
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=4
        )
    if model_name == "ridge":
        return Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    if model_name == "svr":
        from sklearn.svm import SVR  # noqa: PLC0415
        return Pipeline([("sc", StandardScaler()), ("reg", SVR(C=1.0, epsilon=0.01))])
    if model_name == "lgbm":
        import lightgbm as lgb  # noqa: PLC0415
        return lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            random_state=seed, n_jobs=4, verbose=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


# ─── Metrics ──────────────────────────────────────────────────────────────────
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# ─── LOCV ─────────────────────────────────────────────────────────────────────
def run_locv(
    subset: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    seed: int,
    case_scope: list[int],
) -> list[dict]:
    """Leave-one-case-out CV for a single combo. Returns per-case metric dicts."""
    np.random.seed(seed)
    rows: list[dict] = []
    for test_case in case_scope:
        test_df = subset[subset["case_id"] == test_case]
        train_df = subset[subset["case_id"] != test_case]
        if test_df.empty or train_df.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = make_model(model_name, seed)
            model.fit(train_df[feature_cols], train_df["VB"])
            y_pred = model.predict(test_df[feature_cols])
        y_true = test_df["VB"].to_numpy(dtype=float)
        rows.append({"target_case": test_case, "n_runs": len(y_true),
                     **_metrics(y_true, y_pred)})
    return rows


# ─── Checkpoint helpers ───────────────────────────────────────────────────────
_CHECKPOINT = "analysis/checkpoint.txt"  # one combo key per line


def _combo_key(feat_names: list[str], sensors: list[str],
               pct: int, feat_type: str, model_name: str) -> str:
    return f"{'+'.join(feat_names)}|{'+'.join(sensors)}|{pct}|{feat_type}|{model_name}"


def _load_completed(out: Path) -> set[str]:
    cp = out / _CHECKPOINT
    if not cp.exists():
        return set()
    return set(cp.read_text(encoding="utf-8").splitlines())


def _mark_done(out: Path, key: str) -> None:
    with (out / _CHECKPOINT).open("a", encoding="utf-8") as f:
        f.write(key + "\n")


def _flush_csv(out: Path, seed_rows: list, per_case_rows: list, combo_rows: list) -> None:
    """Overwrite CSVs with current accumulated results."""
    pd.DataFrame(seed_rows).to_csv(out / "metrics" / "seed_metrics.csv", index=False)
    pd.DataFrame(per_case_rows).to_csv(out / "metrics" / "per_case_metrics.csv", index=False)
    df = (pd.DataFrame(combo_rows).sort_values("mean_rmse").reset_index(drop=True))
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df.to_csv(out / "analysis" / "combo_ranking.csv", index=False)


# ─── Calibration ─────────────────────────────────────────────────────────────
def _calibrate(master: pd.DataFrame, case_scope: list[int],
               pct: int, feat_type: str, model_name: str,
               n_cal: int = 3) -> float:
    """Time N_CAL LOCV fits using the full feature/sensor set → seconds per fit."""
    feature_cols = get_feature_cols(ALL_SENSORS, ALL_FEATURES, feat_type)
    subset = master[master["signal_prefix_percent"] == float(pct)].copy()
    times: list[float] = []
    for tc in case_scope[:n_cal]:
        test_df = subset[subset["case_id"] == tc]
        train_df = subset[subset["case_id"] != tc]
        if test_df.empty or train_df.empty:
            continue
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = make_model(model_name, 0)
            m.fit(train_df[feature_cols], train_df["VB"])
            m.predict(test_df[feature_cols])
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)) if times else 0.05


# ─── CLI ──────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H6_S4 combinatorial sweep.")
    p.add_argument("--features", nargs="+", default=ALL_FEATURES,
                   choices=ALL_FEATURES, metavar="FEAT",
                   help="Statistical features to extract (default: all 6).")
    p.add_argument("--sensors", nargs="+", default=ALL_SENSORS,
                   choices=ALL_SENSORS, metavar="SENSOR",
                   help="Sensors to use (default: all 6).")
    p.add_argument("--prefixes", type=int, nargs="+", default=DEFAULT_PREFIXES,
                   metavar="PCT",
                   help="Input-ratio percentages to sweep (default: 10-100 step 10).")
    p.add_argument("--feature-types", nargs="+", default=DEFAULT_FEATURE_TYPES,
                   choices=["raw", "delta", "raw_meta", "delta_meta", "raw_delta", "raw_delta_meta"],
                   dest="feature_types", metavar="TYPE",
                   help="Feature types to sweep (default: all 4).")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   choices=["xgboost", "rf", "ridge", "svr", "lgbm"],
                   help="Models to compare (default: xgboost).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Random seeds for LOCV (default: 0 1 2).")
    p.add_argument("--max-feature-size", type=int, default=None, dest="max_feature_size",
                   help="Max feature subset size (default: len(features) = all subsets).")
    p.add_argument("--max-sensor-size", type=int, default=None, dest="max_sensor_size",
                   help="Max sensor subset size (default: len(sensors) = all subsets).")
    p.add_argument("--smoke", action="store_true",
                   help=(
                       "Quick smoke test: mean+rms, smcAC+vib_table, prefix=50, "
                       "types=raw+delta_meta, xgboost, seed=0, cases 1-5, size=1."
                   ))
    p.add_argument("--resume", default=None, metavar="DIR",
                   help="Resume an interrupted run from the given execution directory.")
    p.add_argument("--output-root", default="experiments/executions")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    if args.smoke:
        args.features = ["mean", "rms"]
        args.sensors = ["smcAC", "vib_table"]
        args.prefixes = [50]
        args.feature_types = ["raw", "delta_meta"]
        args.models = ["xgboost"]
        args.seeds = [0]
        # no max-size override: explore all subsets of the 2-item smoke feature/sensor sets
        case_scope = CASE_SCOPE[:5]
    else:
        case_scope = CASE_SCOPE

    if args.resume:
        out = Path(args.resume).resolve()
        if not out.exists():
            raise SystemExit(f"Resume directory not found: {out}")
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        suffix = "smoke_test" if args.smoke else "combo_sweep"
        out = ROOT / args.output_root / "H6" / "S4" / f"{ts}_{suffix}"
    for sub in ["analysis", "metrics", "figures", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(m: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log_lines.append(line)

    # Build all subsets for features and sensors
    max_fs = args.max_feature_size if args.max_feature_size else len(args.features)
    max_ss = args.max_sensor_size if args.max_sensor_size else len(args.sensors)
    feat_subsets = [c for r in range(1, max_fs + 1)
                    for c in itertools.combinations(args.features, r)]
    sensor_subsets = [c for r in range(1, max_ss + 1)
                      for c in itertools.combinations(args.sensors, r)]

    combos = list(itertools.product(
        feat_subsets, sensor_subsets, args.prefixes, args.feature_types, args.models,
    ))
    n_combos = len(combos)
    n_total = n_combos * len(args.seeds) * len(case_scope)

    log(f"=== H6_S4 {'SMOKE TEST' if args.smoke else 'combo sweep'} ===")
    log(f"features      : {args.features}  (max subset size: {max_fs} → {len(feat_subsets)} subsets)")
    log(f"sensors       : {args.sensors}  (max subset size: {max_ss} → {len(sensor_subsets)} subsets)")
    log(f"prefixes      : {args.prefixes}")
    log(f"feature_types : {args.feature_types}")
    log(f"models        : {args.models}")
    log(f"seeds         : {args.seeds}  |  cases: {case_scope}")
    log(f"combos        : {n_combos}  (total LOCV evaluations: {n_total})")

    # ── Load data ──────────────────────────────────────────────────────────────
    log("Loading data...")
    sig = pd.read_csv(
        ROOT / "datasets/processed/mill_signal_data.csv",
        usecols=["case", "run"] + list(_SENSOR_COL.values()),
    )
    proc = pd.read_csv(ROOT / "datasets/processed/mill_process_info.csv")
    proc = proc[proc["case"].isin(case_scope)].copy()
    sig = sig[sig["case"].isin(case_scope)].copy()
    proc_clean = _preprocess(proc)
    log(f"Rows: signal={len(sig)}, process={len(proc_clean)}")

    # ── Build master feature matrix (once) ─────────────────────────────────────
    log(f"Building master feature matrix for {len(args.prefixes)} prefix(es)...")
    t_build = time.time()
    master = build_master_matrix(sig, proc_clean, args.prefixes)
    log(f"Master matrix: {len(master)} rows × {len(master.columns)} cols  [{time.time()-t_build:.1f}s]")

    if not args.resume:
        (out / "analysis" / "config.json").write_text(
            json.dumps({
                "experiment": "H6_S4_combo_sweep", "smoke": args.smoke,
                "features": args.features, "sensors": args.sensors,
                "prefixes": args.prefixes, "feature_types": args.feature_types,
                "models": args.models, "seeds": args.seeds,
                "case_scope": case_scope, "n_combos": n_combos,
            }, indent=2),
            encoding="utf-8",
        )

    # ── Resume: load already-completed results ─────────────────────────────────
    completed = _load_completed(out)
    n_done = len(completed)
    if args.resume:
        log(f"Resuming from: {out}")
        sd_path = out / "metrics" / "seed_metrics.csv"
        pc_path = out / "metrics" / "per_case_metrics.csv"
        cr_path = out / "analysis" / "combo_ranking.csv"
        seed_rows: list[dict] = pd.read_csv(sd_path).to_dict("records") if sd_path.exists() else []
        per_case_rows: list[dict] = pd.read_csv(pc_path).to_dict("records") if pc_path.exists() else []
        combo_rows: list[dict] = (pd.read_csv(cr_path).drop(columns=["rank"], errors="ignore")
                                  .to_dict("records") if cr_path.exists() else [])
        log(f"Already completed : {n_done}/{n_combos} combos")
    else:
        seed_rows, per_case_rows, combo_rows = [], [], []

    # ── Calibration: estimate remaining time ───────────────────────────────────
    n_remaining = (n_combos - n_done) * len(args.seeds) * len(case_scope)
    if n_remaining > 0:
        cal_pct = args.prefixes[len(args.prefixes) // 2]
        cal_type = args.feature_types[0]
        cal_model = args.models[0]
        log(f"Calibrating with full feature set ({len(ALL_FEATURES)}F×{len(ALL_SENSORS)}S, "
            f"pct={cal_pct}%, {cal_model}, 3 folds)...")
        t_per_fit = _calibrate(master, case_scope, cal_pct, cal_type, cal_model, n_cal=3)
        est_s = t_per_fit * n_remaining
        def _fmt_dur(s: float) -> str:
            if s < 120:   return f"{s:.0f}s"
            if s < 7200:  return f"{s/60:.1f}min"
            return f"{s/3600:.1f}h"
        log(f"Calibration      : {t_per_fit*1000:.1f} ms/fit  (worst-case upper bound)")
        log(f"Remaining fits   : {n_remaining:,}  ({n_combos - n_done} combos × "
            f"{len(args.seeds)} seeds × {len(case_scope)} folds)")
        log(f"Estimated time   : ≤ {_fmt_dur(est_s)}  "
            f"(actual shorter: smaller feature/sensor subsets are faster)")

    # ── Sweep ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    _w = len(str(n_combos))

    for ci, (feat_names_t, sensors_t, pct, feat_type, model_name) in enumerate(combos, 1):
        feat_names = list(feat_names_t)
        sensors = list(sensors_t)
        key = _combo_key(feat_names, sensors, pct, feat_type, model_name)

        if key in completed:
            fa = "+".join(_FEAT_ABBR[f] for f in feat_names)
            sa = "+".join(_SENS_ABBR[s] for s in sensors)
            log(
                f"[{ci:>{_w}}/{n_combos}] "
                f"{'SKIP':<14} "
                f"pct={pct:>3}%  "
                f"{model_name:<8}  "
                f"F=[{fa:<17}]  "
                f"S=[{sa:<17}]  "
                f"(already done)"
            )
            continue

        feature_cols = get_feature_cols(sensors, feat_names, feat_type)
        subset = master[master["signal_prefix_percent"] == float(pct)].copy()

        seed_rmses: list[float] = []
        seed_r2s: list[float] = []
        seed_maes: list[float] = []

        for seed in args.seeds:
            case_rows = run_locv(subset, feature_cols, model_name, seed, case_scope)
            if not case_rows:
                continue
            locv_rmse = float(np.mean([r["rmse"] for r in case_rows]))
            locv_r2 = float(np.mean([r["r2"] for r in case_rows]))
            locv_mae = float(np.mean([r["mae"] for r in case_rows]))
            seed_rmses.append(locv_rmse)
            seed_r2s.append(locv_r2)
            seed_maes.append(locv_mae)
            common = {
                "features": "+".join(feat_names), "sensors": "+".join(sensors),
                "prefix": pct, "feature_type": feat_type, "model": model_name,
            }
            for r in case_rows:
                per_case_rows.append({**common, "seed": seed, **r})
            seed_rows.append({**common, "seed": seed,
                              "locv_rmse": locv_rmse, "locv_r2": locv_r2, "locv_mae": locv_mae})

        if not seed_rmses:
            continue
        mean_rmse = float(np.mean(seed_rmses))
        std_rmse = float(np.std(seed_rmses)) if len(seed_rmses) > 1 else 0.0
        combo_rows.append({
            "features": "+".join(feat_names), "sensors": "+".join(sensors),
            "prefix": pct, "feature_type": feat_type, "model": model_name,
            "n_features_per_sensor": len(feat_names),
            "n_sensors": len(sensors),
            "input_dim": len(feature_cols),
            "mean_rmse": mean_rmse, "std_rmse": std_rmse,
            "mean_r2": float(np.mean(seed_r2s)),
            "mean_mae": float(np.mean(seed_maes)),
            "n_seeds": len(seed_rmses),
        })

        # Checkpoint + incremental CSV save
        _mark_done(out, key)
        _flush_csv(out, seed_rows, per_case_rows, combo_rows)

        fa = "+".join(_FEAT_ABBR[f] for f in feat_names)
        sa = "+".join(_SENS_ABBR[s] for s in sensors)
        log(
            f"[{ci:>{_w}}/{n_combos}] "
            f"{feat_type:<14} "
            f"pct={pct:>3}%  "
            f"{model_name:<8}  "
            f"F=[{fa:<17}]  "
            f"S=[{sa:<17}]  "
            f"dim={len(feature_cols):>3}  "
            f"RMSE={mean_rmse:.6f}  "
            f"R²={float(np.mean(seed_r2s)):+.4f}  "
            f"[{time.time()-t0:>5.0f}s]"
        )

    # ── Load final CSVs (already written incrementally) ────────────────────────
    combo_df = (pd.DataFrame(combo_rows)
                .sort_values("mean_rmse")
                .reset_index(drop=True))
    if not combo_df.empty:
        combo_df.insert(0, "rank", np.arange(1, len(combo_df) + 1))
        combo_df.to_csv(out / "analysis" / "combo_ranking.csv", index=False)
    per_case_df = pd.DataFrame(per_case_rows)

    if combo_df.empty:
        log("No results produced."); return

    best = combo_df.iloc[0]

    # ── Figures ────────────────────────────────────────────────────────────────
    fig_id = 0

    def next_fig() -> int:
        nonlocal fig_id
        fig_id += 1
        return fig_id

    # feature_type breakdown
    if combo_df["feature_type"].nunique() > 1:
        fid = next_fig()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        agg = combo_df.groupby("feature_type")["mean_rmse"].mean().sort_values()
        ax.bar(agg.index, agg.values, color="#4c72b0", alpha=0.85)
        ax.set_ylabel("Mean LOCV RMSE"); ax.set_xlabel("Feature type")
        ax.set_title("H6_S4: Mean RMSE by feature type (all prefixes / models averaged)")
        ax.grid(True, axis="y", color="#eee", lw=0.5)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"{fid:02d}_rmse_by_feature_type.png", dpi=150)
        plt.close(fig)

    # prefix curve
    if combo_df["prefix"].nunique() > 1:
        fid = next_fig()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pref_agg = (combo_df.groupby("prefix")
                    .agg(mean_rmse=("mean_rmse", "mean"), std_rmse=("mean_rmse", "std"))
                    .reset_index())
        pref_agg["std_rmse"] = pref_agg["std_rmse"].fillna(0.0)
        ax.errorbar(pref_agg["prefix"], pref_agg["mean_rmse"], yerr=pref_agg["std_rmse"],
                    marker="o", capsize=3, color="#d62728")
        ax.set_xlabel("Input ratio (%)"); ax.set_ylabel("Mean LOCV RMSE")
        ax.set_title("H6_S4: RMSE vs input ratio (feature types / models averaged)")
        ax.grid(True, color="#eee", lw=0.5)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"{fid:02d}_rmse_by_prefix.png", dpi=150)
        plt.close(fig)

    # model breakdown
    if combo_df["model"].nunique() > 1:
        fid = next_fig()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        model_agg = combo_df.groupby("model")["mean_rmse"].mean().sort_values()
        ax.bar(model_agg.index, model_agg.values, color="#2ca02c", alpha=0.85)
        ax.set_ylabel("Mean LOCV RMSE"); ax.set_xlabel("Model")
        ax.set_title("H6_S4: Mean RMSE by model")
        ax.grid(True, axis="y", color="#eee", lw=0.5)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"{fid:02d}_rmse_by_model.png", dpi=150)
        plt.close(fig)

    # per-case RMSE for best combo
    if not per_case_df.empty:
        fid = next_fig()
        q = per_case_df[
            (per_case_df["features"] == best["features"])
            & (per_case_df["sensors"] == best["sensors"])
            & (per_case_df["prefix"] == best["prefix"])
            & (per_case_df["feature_type"] == best["feature_type"])
            & (per_case_df["model"] == best["model"])
        ]
        if not q.empty:
            pc = q.groupby("target_case")["rmse"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(pc["target_case"].astype(str), pc["rmse"], color="#4c72b0", alpha=0.85)
            ax.set_xlabel("Hold-out case"); ax.set_ylabel("LOCV RMSE (seed mean)")
            ax.set_title(
                f"H6_S4 best combo per-case RMSE\n"
                f"{best['feature_type']} | prefix={int(best['prefix'])}% | {best['model']}"
            )
            ax.grid(True, axis="y", color="#eee", lw=0.5)
            fig.tight_layout()
            fig.savefig(out / "figures" / f"{fid:02d}_best_combo_per_case_rmse.png", dpi=150)
            plt.close(fig)

    # ── Summary ────────────────────────────────────────────────────────────────
    summary = {
        "experiment": "H6_S4_combo_sweep",
        "smoke": args.smoke,
        "total_combos": len(combo_df),
        "total_elapsed_s": round(time.time() - t0, 1),
        "best": {k: (int(v) if isinstance(v, np.integer) else
                     float(v) if isinstance(v, (np.floating, float)) else v)
                 for k, v in best.items()},
        "top10": combo_df.head(10).to_dict("records"),
    }
    (out / "analysis" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )

    # ── Console report ─────────────────────────────────────────────────────────
    log("\n=== TOP combos (by mean LOCV RMSE) ===")
    show_cols = ["rank", "feature_type", "prefix", "model", "features", "sensors",
                 "input_dim", "mean_rmse", "std_rmse", "mean_r2"]
    log("\n" + combo_df.head(min(10, len(combo_df)))[show_cols].to_string(index=False))
    log(
        f"\nBest: {best['feature_type']} | prefix={int(best['prefix'])}% | {best['model']}"
        f"\n      features={best['features']}  sensors={best['sensors']}"
        f"\n      RMSE={best['mean_rmse']:.6f}  R²={best['mean_r2']:.4f}  MAE={best['mean_mae']:.6f}"
    )

    (out / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nDone. Results in: {out}")
    print(f"EXECUTION_DIR={out}", flush=True)


if __name__ == "__main__":
    main()

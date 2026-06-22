#!/usr/bin/env python3
"""B5_S1: Sensor Subset Search — ML models (Ridge, SVR, XGBoost, RandomForest)

[B5] Phase 1. [B1]/[B2]에서 sensor subset이 model/segment 선택보다 성능에 더 큰
영향을 준다는 것이 확인됨에 따라, 4개 ML 모델에 대해 6센서의 63개 non-empty
subset(mask=1~63) 전수 탐색을 수행하고 모델별 최적 subset을 찾는다.

고정: Delta+Meta, Segment=Full, LOCV-15, observed-VB-only.
Seed: RandomForest 5-seed mean±std, Ridge/SVR/XGBoost 1회(결정론적).

Output: experiments/executions/B5/S1/{timestamp}_sensor_subset_search_ml/
"""
from __future__ import annotations

import json
import sys
import time as time_mod
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
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
SEEDS         = [0, 1, 2, 3, 4]
THRESH        = 1e6
N_SENSORS     = len(SENSORS)
ALL_MASKS     = list(range(1, 64))   # 2^6 - 1 = 63 non-empty subsets
MODELS        = ["Ridge", "SVR", "XGBoost", "RandomForest"]
STOCHASTIC    = {"RandomForest"}

XGB_CFG = dict(
    learning_rate=0.03, n_estimators=600, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.9,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, n_jobs=4,
)
RF_CFG  = dict(n_estimators=200, max_depth=None, n_jobs=4)


# ─── Utils ────────────────────────────────────────────────────────────────────
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


def mask_sensor_indices(mask: int) -> list[int]:
    return [
        i * 4 + j
        for i in range(N_SENSORS) if (mask >> i) & 1
        for j in range(4)
    ]


def mask_label(mask: int) -> str:
    return "+".join(SENSOR_ABBR[SENSORS[i]] for i in range(N_SENSORS) if (mask >> i) & 1)


def obs_mask(case_id: int, runs: np.ndarray) -> np.ndarray:
    return np.array([(case_id, int(r)) not in NON_OBSERVED_RUNS for r in runs])


# ─── Full 6-sensor raw cache (built once, sliced per mask) ───────────────────
def build_full_raw_cache(
    signal_df: pd.DataFrame, proc_clean: pd.DataFrame
) -> tuple[dict[tuple[int,int], np.ndarray], dict[int,int]]:
    """Cache[(case,run)] = 24-dim raw 4-stat vector for all 6 sensors (Full segment)."""
    cache: dict[tuple[int,int], np.ndarray] = {}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        sig_row = signal_df[(signal_df["case"] == case_id) & (signal_df["run"] == run_id)]
        if sig_row.empty:
            continue
        sr = sig_row.iloc[0]
        arrays = {s: parse_signal(sr[s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        base_len = min(len(a) for a in arrays.values())
        cache[(case_id, run_id)] = np.concatenate(
            [extract_features(arrays[s][:base_len]) for s in SENSORS])

    first_run: dict[int,int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c, r) in cache if c == case_id)
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


def build_feat_df_for_mask(
    full_cache: dict[tuple[int,int], np.ndarray],
    first_run: dict[int,int],
    proc_clean: pd.DataFrame,
    mask: int,
) -> tuple[pd.DataFrame, list[str]]:
    sensor_indices = mask_sensor_indices(mask)
    rows = []
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id)
        if key not in full_cache:
            continue
        raw_vec = full_cache[key][sensor_indices]
        ref_vec = full_cache.get((case_id, first_run.get(case_id, run_id)),
                                  np.zeros(24))[sensor_indices]
        delta = raw_vec - ref_vec
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        meta = np.array([float(getattr(row, mf, 0.0)) for mf in META_FEATURES])
        feat = np.concatenate([delta, meta])
        rec = {"case_id": case_id, "run": run_id, "VB": float(row.VB) if not pd.isna(row.VB) else 0.0}
        for k, v in enumerate(feat):
            rec[f"f{k}"] = float(v)
        rows.append(rec)
    df = pd.DataFrame(rows)
    feature_cols = [f"f{k}" for k in range(bin(mask).count("1") * 4 + len(META_FEATURES))]
    return df, feature_cols


# ─── Tabular LOCV ─────────────────────────────────────────────────────────────
def run_tabular_locv(feat_df: pd.DataFrame, feature_cols: list[str], model_type: str, seed: int) -> float:
    case_rmses: list[float] = []
    use_scaler = model_type in ("Ridge", "SVR")
    for tc in CASE_SCOPE:
        test_df  = feat_df[feat_df["case_id"] == tc].sort_values("run")
        train_df = feat_df[feat_df["case_id"] != tc]
        if test_df.empty or train_df.empty:
            continue
        X_tr_raw = train_df[feature_cols].values
        X_te_raw = test_df[feature_cols].values
        if use_scaler:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)
        else:
            X_tr, X_te = X_tr_raw, X_te_raw
        y_tr = train_df["VB"].values

        if model_type == "Ridge":
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        elif model_type == "SVR":
            model = SVR(kernel="rbf", C=10.0, gamma="scale")
        elif model_type == "XGBoost":
            model = XGBRegressor(**XGB_CFG, random_state=seed, verbosity=0)
        else:
            model = RandomForestRegressor(**RF_CFG, random_state=seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
        if use_scaler:
            y_pred = np.clip(y_pred, 0.0, None)

        obs = obs_mask(tc, test_df["run"].to_numpy(int))
        y_true = test_df["VB"].to_numpy(float)[obs]
        y_pred_obs = y_pred[obs]
        if len(y_true) == 0:
            continue
        case_rmses.append(float(np.sqrt(mean_squared_error(y_true, y_pred_obs))))
    return float(np.mean(case_rmses)) if case_rmses else float("nan")


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_results(results: dict[str, list[dict]], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, model_type in zip(axes.flat, MODELS):
        recs = sorted(results[model_type], key=lambda r: r["mean"])
        means = [r["mean"] for r in recs]
        ax.plot(range(len(means)), means, marker=".", linewidth=0.8, markersize=3)
        best = recs[0]
        ax.axhline(best["mean"], color="crimson", linestyle="--", linewidth=1,
                   label=f"best={best['label']} ({best['mean']:.4f})")
        ax.set_title(model_type)
        ax.set_xlabel("subset rank (best→worst)")
        ax.set_ylabel("Observed-VB RMSE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_dir / "sensor_subset_search_ml.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "sensor_subset_search_ml.svg"), bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B5" / "S1" / f"{ts}_sensor_subset_search_ml"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== B5_S1: Sensor Subset Search (ML) ===")
    log(f"Models={MODELS}, Masks=1..63 (63 subsets), Segment=Full, Features=Delta+Meta")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    log("Building full 6-sensor raw cache (once)...")
    full_cache, first_run = build_full_raw_cache(signal_df, proc_clean)
    log(f"Cache size: {len(full_cache)} runs")

    results: dict[str, list[dict]] = {m: [] for m in MODELS}
    t_total = time_mod.time()

    for mi, mask in enumerate(ALL_MASKS):
        feat_df, feature_cols = build_feat_df_for_mask(full_cache, first_run, proc_clean, mask)
        label = mask_label(mask)
        for model_type in MODELS:
            seeds_to_run = SEEDS if model_type in STOCHASTIC else [0]
            seed_rmses = [run_tabular_locv(feat_df, feature_cols, model_type, s) for s in seeds_to_run]
            mean_ = float(np.mean(seed_rmses))
            std_  = float(np.std(seed_rmses))
            results[model_type].append({
                "mask": mask, "label": label, "n_sensors": bin(mask).count("1"),
                "mean": mean_, "std": std_,
            })
        if (mi + 1) % 10 == 0 or mi == len(ALL_MASKS) - 1:
            log(f"  [{mi+1}/{len(ALL_MASKS)}] mask={mask} ({label})  [{time_mod.time()-t_total:.0f}s elapsed]")

    # ── Save full grid CSV ───────────────────────────────────────────────────
    all_rows = []
    for model_type, recs in results.items():
        for r in recs:
            all_rows.append({"model": model_type, **r})
    grid_df = pd.DataFrame(all_rows)
    grid_df.to_csv(out_dir / "metrics" / "full_grid.csv", index=False)

    plot_results(results, out_dir / "figures")

    # ── Summary: best subset per model ───────────────────────────────────────
    log("\n=== SUMMARY: Best sensor subset per model ===")
    best_per_model = {}
    for model_type, recs in results.items():
        best = min(recs, key=lambda r: r["mean"])
        best_per_model[model_type] = best
        log(f"  {model_type:<14} best={best['label']:<20} (n_sensors={best['n_sensors']})  "
            f"RMSE={best['mean']:.6f}  std={best['std']:.6f}")

    # overlap check
    best_labels = {m: b["label"] for m, b in best_per_model.items()}
    log(f"\nBest subsets identical across all ML models? "
        f"{'YES' if len(set(best_labels.values())) == 1 else 'NO'}")
    log(f"Best subsets: {best_labels}")

    overall_best = min(best_per_model.items(), key=lambda kv: kv[1]["mean"])
    log(f"\nOverall best (ML): {overall_best[0]} / {overall_best[1]['label']} "
        f"RMSE={overall_best[1]['mean']:.6f}")

    summary = {
        "experiment": "B5_S1_sensor_subset_search_ml",
        "models": MODELS, "n_masks": len(ALL_MASKS),
        "best_per_model": best_per_model,
        "best_labels": best_labels,
        "overall_best": {"model": overall_best[0], **overall_best[1]},
        "execution_dir": str(out_dir),
        "total_elapsed_s": time_mod.time() - t_total,
    }
    (out_dir / "logs" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text(
        "\n".join(log_lines), encoding="utf-8")

    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

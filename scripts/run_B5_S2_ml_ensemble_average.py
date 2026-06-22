#!/usr/bin/env python3
"""B5_S2: ML Ensemble (simple average) — Ridge + SVR + XGBoost + RandomForest

[B5] Phase 1 후속. 4개 ML 모델을 각자의 B5_S1 최적 sensor subset으로 학습시킨 뒤,
LOCV test case에서의 예측값을 단순 평균(equal weight)해 앙상블 성능을 측정한다.

  Ridge        : AC+vT       (개별 RMSE=0.122213)
  SVR          : AC+vS+aS    (개별 RMSE=0.109065)
  XGBoost      : AC+vS       (개별 RMSE=0.108260, ML 단일 최선)
  RandomForest : AC+vT+aS    (개별 RMSE=0.119037)

비교 시나리오:
  A) Own-best subset ensemble — 각 모델 자신의 최적 subset 사용 (메인)
  B) Uniform subset ensemble  — 4개 모델 전부 ML 전체 최선 subset(AC+vS) 사용 (대조)

평가: LOCV 15 cases, observed-VB-only RMSE. RandomForest 5-seed(앙상블 RMSE
mean±std), 나머지 고정.

Output: experiments/executions/B5/S2/{timestamp}_ml_ensemble_average/
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
ABBR_TO_IDX   = {v: i for i, v in enumerate(SENSOR_ABBR.values())}
EXCLUDED_RUNS = {(2, 1), (12, 1)}
NON_OBSERVED_RUNS = {
    (1,2),(1,3),(1,5),(1,16),(2,6),(8,4),(11,7),(11,14),(11,17),
    (12,4),(12,10),(13,1),(13,2),(14,1),(14,5),(15,1),(16,1),(16,2),(16,4),
}
META_FEATURES = ["DOC", "feed", "material"]
SEEDS         = [0, 1, 2, 3, 4]
THRESH        = 1e6
N_SENSORS     = len(SENSORS)
MODELS        = ["Ridge", "SVR", "XGBoost", "RandomForest"]

OWN_BEST = {"Ridge": "AC+vT", "SVR": "AC+vS+aS", "XGBoost": "AC+vS", "RandomForest": "AC+vT+aS"}
UNIFORM_BEST = "AC+vS"   # overall ML best (XGBoost's subset)
INDIVIDUAL_RMSE = {"Ridge": 0.122213, "SVR": 0.109065, "XGBoost": 0.108260, "RandomForest": 0.119037}

XGB_CFG = dict(
    learning_rate=0.03, n_estimators=600, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.9,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, n_jobs=4,
)
RF_CFG  = dict(n_estimators=200, max_depth=None, n_jobs=4)


def label_to_mask(label: str) -> int:
    mask = 0
    for abbr in label.split("+"):
        mask |= (1 << ABBR_TO_IDX[abbr])
    return mask


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


def build_full_raw_cache(signal_df, proc_clean):
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


def build_feat_df_for_mask(full_cache, first_run, proc_clean, mask):
    sensor_indices = mask_sensor_indices(mask)
    rows = []
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id)
        if key not in full_cache:
            continue
        raw_vec = full_cache[key][sensor_indices]
        ref_vec = full_cache.get((case_id, first_run.get(case_id, run_id)), np.zeros(24))[sensor_indices]
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


def fit_model(model_type: str, seed: int):
    if model_type == "Ridge":
        return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    if model_type == "SVR":
        return SVR(kernel="rbf", C=10.0, gamma="scale")
    if model_type == "XGBoost":
        return XGBRegressor(**XGB_CFG, random_state=seed, verbosity=0)
    return RandomForestRegressor(**RF_CFG, random_state=seed)


def predict_model(model_type: str, train_df, test_df, feature_cols, seed: int) -> np.ndarray:
    use_scaler = model_type in ("Ridge", "SVR")
    X_tr_raw = train_df[feature_cols].values
    X_te_raw = test_df[feature_cols].values
    if use_scaler:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)
    else:
        X_tr, X_te = X_tr_raw, X_te_raw
    y_tr = train_df["VB"].values
    model = fit_model(model_type, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
    if use_scaler:
        y_pred = np.clip(y_pred, 0.0, None)
    return y_pred


def obs_mask(case_id: int, runs: np.ndarray) -> np.ndarray:
    return np.array([(case_id, int(r)) not in NON_OBSERVED_RUNS for r in runs])


# ─── Ensemble LOCV ────────────────────────────────────────────────────────────
def run_ensemble_locv(
    feat_dfs: dict[str, tuple[pd.DataFrame, list[str]]], rf_seed: int,
) -> tuple[float, dict[int, dict]]:
    case_rmses: list[float] = []
    per_case: dict[int, dict] = {}
    for tc in CASE_SCOPE:
        model_preds: dict[str, np.ndarray] = {}
        y_true = None
        runs = None
        for model_type in MODELS:
            df, feature_cols = feat_dfs[model_type]
            test_df  = df[df["case_id"] == tc].sort_values("run")
            train_df = df[df["case_id"] != tc]
            if test_df.empty or train_df.empty:
                model_preds = {}
                break
            seed = rf_seed if model_type == "RandomForest" else 0
            model_preds[model_type] = predict_model(model_type, train_df, test_df, feature_cols, seed)
            if y_true is None:
                y_true = test_df["VB"].to_numpy(float)
                runs = test_df["run"].to_numpy(int)
        if not model_preds:
            continue
        ensemble_pred = np.mean(np.stack(list(model_preds.values())), axis=0)
        obs = obs_mask(tc, runs)
        if obs.sum() == 0:
            continue
        rmse = float(np.sqrt(mean_squared_error(y_true[obs], ensemble_pred[obs])))
        case_rmses.append(rmse)
        per_case[tc] = {"run": runs, "y_true": y_true, "y_pred": ensemble_pred,
                         **{f"pred_{m}": p for m, p in model_preds.items()}}
    return (float(np.mean(case_rmses)) if case_rmses else float("nan")), per_case


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B5" / "S2" / f"{ts}_ml_ensemble_average"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== B5_S2: ML Ensemble (simple average) ===")
    log(f"Models={MODELS}")
    log(f"Scenario A (own-best): {OWN_BEST}")
    log(f"Scenario B (uniform):  all models @ {UNIFORM_BEST}")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    full_cache, first_run = build_full_raw_cache(signal_df, proc_clean)
    log(f"Clean runs: {len(proc_clean)}, cache size: {len(full_cache)}")

    results = {}
    for scenario_name, subset_map in [("own_best", OWN_BEST),
                                       ("uniform_ACvS", {m: UNIFORM_BEST for m in MODELS})]:
        log(f"\n--- Scenario: {scenario_name} ---")
        feat_dfs = {
            m: build_feat_df_for_mask(full_cache, first_run, proc_clean, label_to_mask(subset_map[m]))
            for m in MODELS
        }
        seed_rmses = []
        last_per_case = None
        for seed in SEEDS:
            rmse, per_case = run_ensemble_locv(feat_dfs, seed)
            seed_rmses.append(rmse)
            last_per_case = per_case
        mean_, std_ = float(np.mean(seed_rmses)), float(np.std(seed_rmses))
        log(f"  Ensemble RMSE: mean={mean_:.6f}  std={std_:.6f}  (5-seed, RF seed varies)")
        results[scenario_name] = {
            "subset_map": subset_map, "mean": mean_, "std": std_, "seed_rmses": seed_rmses,
            "per_case_sample": last_per_case,
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    log(f"{'Scenario':<16} {'Mean':>10} {'Std':>10}")
    log("-" * 38)
    for name, r in results.items():
        log(f"{name:<16} {r['mean']:>10.6f} {r['std']:>10.6f}")

    log(f"\nIndividual model RMSE (own-best subset, B5_S1 reference):")
    for m, rmse in INDIVIDUAL_RMSE.items():
        log(f"  {m:<14} {rmse:.6f}")
    best_individual = min(INDIVIDUAL_RMSE, key=INDIVIDUAL_RMSE.get)
    log(f"Best individual model: {best_individual} = {INDIVIDUAL_RMSE[best_individual]:.6f}")

    ens_own = results["own_best"]["mean"]
    delta = ens_own - INDIVIDUAL_RMSE[best_individual]
    log(f"\nEnsemble(own-best) vs best individual({best_individual}): "
        f"{ens_own:.6f} vs {INDIVIDUAL_RMSE[best_individual]:.6f}  Δ={delta:+.6f} "
        f"({'개선' if delta < 0 else '악화'})")

    # ── Plot: case-level comparison for own_best scenario ───────────────────
    per_case = results["own_best"]["per_case_sample"]
    cases = sorted(per_case.keys())
    ncols = 5
    nrows = int(np.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for idx, case_id in enumerate(cases):
        ax = axes[idx // ncols][idx % ncols]
        d = per_case[case_id]
        runs, y_true, y_pred = d["run"], d["y_true"], d["y_pred"]
        is_obs = np.array([(case_id, int(r)) not in NON_OBSERVED_RUNS for r in runs])
        ax.plot(runs, y_pred, "-", color="crimson", marker="o", markersize=4, linewidth=1.3, label="Ensemble pred")
        ax.plot(runs[is_obs], y_true[is_obs], "-", color="black", marker="o", markersize=4, linewidth=1.0, label="VB (observed)")
        if (~is_obs).any():
            ax.plot(runs[~is_obs], y_true[~is_obs], "o", mfc="none", mec="gray", markersize=5, linewidth=0, label="VB (interpolated)")
        ax.set_title(f"Case {case_id}", fontsize=10)
        ax.set_xlabel("run", fontsize=8); ax.set_ylabel("VB", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")
    for idx in range(len(cases), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(f"ML Ensemble (own-best subset avg) — RMSE={ens_own:.6f}", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(out_dir / "figures" / "trajectory_ensemble.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "figures" / "trajectory_ensemble.svg"), bbox_inches="tight")
    plt.close(fig)

    # ── Bar chart: individual vs ensemble ───────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    names = list(INDIVIDUAL_RMSE.keys()) + ["Ensemble(own-best)", "Ensemble(uniform AC+vS)"]
    vals  = list(INDIVIDUAL_RMSE.values()) + [results["own_best"]["mean"], results["uniform_ACvS"]["mean"]]
    colors = ["#4878d0"]*4 + ["#d65f5f", "#ee854a"]
    bars = ax2.bar(range(len(names)), vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, f"{v:.4f}", ha="center", fontsize=8)
    ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylabel("Observed-VB RMSE")
    ax2.set_title("B5_S2: Individual ML models vs Ensemble (avg)")
    ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig2.savefig(str(out_dir / "figures" / "ensemble_vs_individual.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    summary = {
        "experiment": "B5_S2_ml_ensemble_average",
        "individual_rmse": INDIVIDUAL_RMSE,
        "ensemble_results": {k: {kk: vv for kk, vv in v.items() if kk != "per_case_sample"}
                              for k, v in results.items()},
        "execution_dir": str(out_dir),
    }
    (out_dir / "logs" / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""B5_S1 (addendum): VB Prediction Trajectory — best sensor subset per ML model

[B5_S1] 결과(model별 best sensor subset)로 LOCV 예측값을 뽑아, case별 VB
trajectory(실측 vs 예측)를 시각화한다.

  Ridge        : AC+vT       (RMSE=0.122213)
  SVR          : AC+vS+aS    (RMSE=0.109065)
  XGBoost      : AC+vS       (RMSE=0.108260)
  RandomForest : AC+vT+aS    (RMSE=0.119037)

Non-observed run(보간/대치 VB)은 open marker/회색으로 구분
([[feedback_prediction_trajectory_plot]] 컨벤션).

Output: experiments/executions/B5/S1/{timestamp}_prediction_trajectory/
"""
from __future__ import annotations

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
THRESH        = 1e6
N_SENSORS     = len(SENSORS)

# Best subset per model from B5_S1 full grid search
BEST_CONFIG = {
    "Ridge":        {"mask_label": "AC+vT",    "rmse_ref": 0.122213},
    "SVR":          {"mask_label": "AC+vS+aS", "rmse_ref": 0.109065},
    "XGBoost":      {"mask_label": "AC+vS",    "rmse_ref": 0.108260},
    "RandomForest": {"mask_label": "AC+vT+aS", "rmse_ref": 0.119037},
}

XGB_CFG = dict(
    learning_rate=0.03, n_estimators=600, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.9,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, n_jobs=4,
)
RF_CFG  = dict(n_estimators=200, max_depth=None, n_jobs=4, random_state=0)


def label_to_mask(label: str) -> int:
    mask = 0
    for abbr in label.split("+"):
        mask |= (1 << ABBR_TO_IDX[abbr])
    return mask


# ─── Utils (shared with B5_S1) ────────────────────────────────────────────────
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


def fit_model(model_type: str):
    if model_type == "Ridge":
        return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    if model_type == "SVR":
        return SVR(kernel="rbf", C=10.0, gamma="scale")
    if model_type == "XGBoost":
        return XGBRegressor(**XGB_CFG, random_state=0, verbosity=0)
    return RandomForestRegressor(**RF_CFG)


def run_locv_predictions(feat_df: pd.DataFrame, feature_cols: list[str], model_type: str) -> dict[int, dict]:
    """Returns {case_id: {"run": arr, "y_true": arr, "y_pred": arr}}"""
    use_scaler = model_type in ("Ridge", "SVR")
    preds: dict[int, dict] = {}
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

        model = fit_model(model_type)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
        if use_scaler:
            y_pred = np.clip(y_pred, 0.0, None)

        preds[tc] = {
            "run": test_df["run"].to_numpy(int),
            "y_true": test_df["VB"].to_numpy(float),
            "y_pred": y_pred,
        }
    return preds


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_trajectories(model_type: str, preds: dict[int, dict], rmse_ref: float, mask_label: str, out_dir: Path) -> None:
    cases = sorted(preds.keys())
    ncols = 5
    nrows = int(np.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, case_id in enumerate(cases):
        ax = axes[idx // ncols][idx % ncols]
        d = preds[case_id]
        runs, y_true, y_pred = d["run"], d["y_true"], d["y_pred"]
        is_obs = np.array([(case_id, int(r)) not in NON_OBSERVED_RUNS for r in runs])

        ax.plot(runs, y_pred, "-", color="crimson", marker="o", markersize=4,
                linewidth=1.3, label="Predicted")
        ax.plot(runs[is_obs], y_true[is_obs], "-", color="black", marker="o",
                markersize=4, linewidth=1.0, label="VB (observed)")
        if (~is_obs).any():
            ax.plot(runs[~is_obs], y_true[~is_obs], "o", mfc="none", mec="gray",
                    markersize=5, linewidth=0, label="VB (interpolated)")

        ax.set_title(f"Case {case_id}", fontsize=10)
        ax.set_xlabel("run", fontsize=8)
        ax.set_ylabel("VB", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    for idx in range(len(cases), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"{model_type} — best sensor subset {mask_label} (LOCV RMSE={rmse_ref:.6f})", fontsize=13)
    plt.tight_layout()
    fname = f"trajectory_{model_type}"
    fig.savefig(str(out_dir / f"{fname}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / f"{fname}.svg"), bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B5" / "S1" / f"{ts}_prediction_trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    full_cache, first_run = build_full_raw_cache(signal_df, proc_clean)
    print(f"Clean runs: {len(proc_clean)}, cache size: {len(full_cache)}")

    for model_type, cfg in BEST_CONFIG.items():
        mask = label_to_mask(cfg["mask_label"])
        feat_df, feature_cols = build_feat_df_for_mask(full_cache, first_run, proc_clean, mask)
        print(f"\n{model_type}: best subset={cfg['mask_label']} (mask={mask}), running LOCV predictions...")
        preds = run_locv_predictions(feat_df, feature_cols, model_type)
        plot_trajectories(model_type, preds, cfg["rmse_ref"], cfg["mask_label"], out_dir)
        print(f"  Saved trajectory_{model_type}.png")

    print(f"\nDone. Figures in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}")


if __name__ == "__main__":
    main()

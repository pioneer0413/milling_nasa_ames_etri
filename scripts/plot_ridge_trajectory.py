#!/usr/bin/env python3
"""Ridge LOCV prediction trajectory (own-best subset AC+vT, Delta+Meta).

B5_S1 Ridge own-best = AC+vT (LOCV observed-VB RMSE ref 0.122213). 각 case별
예측 궤적을 산출 — 관측 VB는 검정 실선, 보간/대치 VB는 회색 open marker로 구분.

Output: experiments/executions/B5/S2/{ts}_ridge_trajectory/
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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
SUBSET        = "AC+vT"
REF_RMSE      = 0.122213


def label_to_mask(label):
    mask = 0
    for abbr in label.split("+"):
        mask |= (1 << ABBR_TO_IDX[abbr])
    return mask


def parse_signal(value):
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros(1)
    return np.array([np.mean(arr), np.sqrt(np.mean(arr ** 2)), np.std(arr), np.abs(arr).max()])


def preprocess(process):
    df = process.copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"] == min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside"))
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)


def mask_sensor_indices(mask):
    return [i * 4 + j for i in range(N_SENSORS) if (mask >> i) & 1 for j in range(4)]


def build_full_raw_cache(signal_df, proc_clean):
    cache = {}
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
    first_run = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c, r) in cache if c == case_id)
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


def build_feat_df(full_cache, first_run, proc_clean, mask):
    sensor_indices = mask_sensor_indices(mask)
    rows = []
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        if (case_id, run_id) not in full_cache:
            continue
        raw_vec = full_cache[(case_id, run_id)][sensor_indices]
        ref_vec = full_cache.get((case_id, first_run.get(case_id, run_id)), np.zeros(24))[sensor_indices]
        delta = raw_vec - ref_vec
        delta = np.where(np.isfinite(delta), delta, 0.0)
        meta = np.array([float(getattr(row, mf, 0.0)) for mf in META_FEATURES])
        feat = np.concatenate([delta, meta])
        rec = {"case_id": case_id, "run": run_id, "VB": float(row.VB) if not pd.isna(row.VB) else 0.0}
        for k, v in enumerate(feat):
            rec[f"f{k}"] = float(v)
        rows.append(rec)
    df = pd.DataFrame(rows)
    cols = [f"f{k}" for k in range(bin(mask).count("1") * 4 + len(META_FEATURES))]
    return df, cols


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B5" / "S2" / f"{ts}_ridge_trajectory"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    full_cache, first_run = build_full_raw_cache(signal_df, proc_clean)
    df, cols = build_feat_df(full_cache, first_run, proc_clean, label_to_mask(SUBSET))

    per_case, case_rmses, pred_rows = {}, {}, []
    for tc in CASE_SCOPE:
        test = df[df["case_id"] == tc].sort_values("run")
        train = df[df["case_id"] != tc]
        if test.empty or train.empty:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train[cols].values)
        X_te = scaler.transform(test[cols].values)
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, train["VB"].values)
            y_pred = np.clip(model.predict(X_te), 0.0, None)
        runs = test["run"].to_numpy(int)
        y_true = test["VB"].to_numpy(float)
        obs = np.array([(tc, int(r)) not in NON_OBSERVED_RUNS for r in runs])
        per_case[tc] = {"run": runs, "y_true": y_true, "y_pred": y_pred, "obs": obs}
        if obs.sum():
            case_rmses[tc] = float(np.sqrt(mean_squared_error(y_true[obs], y_pred[obs])))
        for r, yt, yp, o in zip(runs, y_true, y_pred, obs):
            pred_rows.append({"case": tc, "run": int(r), "observed": bool(o),
                              "y_true": float(yt), "ridge_pred": float(yp)})

    overall = float(np.mean(list(case_rmses.values())))
    pd.DataFrame(pred_rows).to_csv(out_dir / "metrics" / "predictions.csv", index=False)
    pd.DataFrame([{"case": c, "rmse": r} for c, r in case_rmses.items()]).to_csv(
        out_dir / "metrics" / "case_rmse.csv", index=False)

    cases = sorted(per_case)
    ncols, nrows = 5, int(np.ceil(len(cases) / 5))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for idx, c in enumerate(cases):
        ax = axes[idx // ncols][idx % ncols]
        d = per_case[c]
        runs, y_true, y_pred, obs = d["run"], d["y_true"], d["y_pred"], d["obs"]
        ax.plot(runs, y_pred, "-", color="crimson", marker="o", markersize=4, linewidth=1.3, label="Ridge pred")
        ax.plot(runs[obs], y_true[obs], "-", color="black", marker="o", markersize=4, linewidth=1.0, label="VB (observed)")
        if (~obs).any():
            ax.plot(runs[~obs], y_true[~obs], "o", mfc="none", mec="gray", markersize=5, linewidth=0, label="VB (interpolated)")
        ax.set_title(f"Case {c}  (RMSE={case_rmses.get(c, float('nan')):.4f})", fontsize=9)
        ax.set_xlabel("run", fontsize=8); ax.set_ylabel("VB", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")
    for idx in range(len(cases), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(f"Ridge LOCV trajectory ({SUBSET}, Delta+Meta) — observed RMSE={overall:.6f} "
                 f"(B5_S1 ref={REF_RMSE})", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(out_dir / "figures" / "ridge_trajectory.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "figures" / "ridge_trajectory.svg"), bbox_inches="tight")
    plt.close(fig)

    print(f"Overall observed RMSE = {overall:.6f}  (ref {REF_RMSE})")
    for c in cases:
        print(f"  Case {c:>2}: RMSE={case_rmses.get(c, float('nan')):.4f}")
    print(f"EXECUTION_DIR={out_dir}")


if __name__ == "__main__":
    main()

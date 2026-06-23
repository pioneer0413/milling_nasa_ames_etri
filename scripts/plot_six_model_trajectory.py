#!/usr/bin/env python3
"""6-model LOCV trajectory overlay: B6 base 5종 + Ridge.

B6_S1 predictions.csv(FeatGRU/FeatLSTM/XGBoost/RandomForest/SVR seed-평균 예측)에
Ridge(AC+vT, Delta+Meta) LOCV 예측을 추가해 case별 6개 모델 궤적을 겹쳐 그린다.
관측 VB=검정 실선, 보간/대치 VB=회색 open marker.

Output: experiments/executions/B6/S1/{ts}_six_model_trajectory/
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# reuse Ridge LOCV helpers
from plot_ridge_trajectory import (  # noqa: E402
    CASE_SCOPE, NON_OBSERVED_RUNS, SENSORS, SUBSET,
    label_to_mask, preprocess, build_full_raw_cache, build_feat_df,
)
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import mean_squared_error  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import warnings  # noqa: E402

B6_PRED = ROOT / "experiments/executions/B6/S1/2026-06-22_193945_id_swe/metrics/predictions.csv"

MODELS = [
    ("FeatGRU",      "FeatGRU_pred",      "#4878d0"),
    ("FeatLSTM",     "FeatLSTM_pred",     "#6acc65"),
    ("XGBoost",      "XGBoost_pred",      "#ee854a"),
    ("RandomForest", "RandomForest_pred", "#956cb4"),
    ("SVR",          "SVR_pred",          "#8c613c"),
    ("Ridge",        "Ridge_pred",        "#d65f5f"),
]


def ridge_locv_preds() -> dict[tuple[int, int], float]:
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    full_cache, first_run = build_full_raw_cache(signal_df, proc_clean)
    df, cols = build_feat_df(full_cache, first_run, proc_clean, label_to_mask(SUBSET))
    preds = {}
    for tc in CASE_SCOPE:
        test = df[df["case_id"] == tc].sort_values("run")
        train = df[df["case_id"] != tc]
        if test.empty or train.empty:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(train[cols].values)
        X_te = sc.transform(test[cols].values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(X_tr, train["VB"].values)
            yp = np.clip(m.predict(X_te), 0.0, None)
        for r, p in zip(test["run"].to_numpy(int), yp):
            preds[(tc, int(r))] = float(p)
    return preds


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B6" / "S1" / f"{ts}_six_model_trajectory"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(B6_PRED)
    ridge = ridge_locv_preds()
    pred["Ridge_pred"] = pred.apply(lambda r: ridge.get((int(r["case"]), int(r["run"])), np.nan), axis=1)

    # per-model observed RMSE (macro over cases)
    rmse = {}
    for name, col, _ in MODELS:
        rr = []
        for c, g in pred[pred["observed"]].groupby("case"):
            gg = g.dropna(subset=[col])
            if len(gg):
                rr.append(np.sqrt(mean_squared_error(gg["y_true"], gg[col])))
        rmse[name] = float(np.mean(rr))

    cases = sorted(pred["case"].unique())
    ncols, nrows = 5, int(np.ceil(len(cases) / 5))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows), squeeze=False)
    for idx, c in enumerate(cases):
        ax = axes[idx // ncols][idx % ncols]
        d = pred[pred["case"] == c].sort_values("run")
        runs = d["run"].to_numpy(int)
        obs = d["observed"].to_numpy(bool)
        yt = d["y_true"].to_numpy(float)
        for name, col, color in MODELS:
            ax.plot(runs, d[col].to_numpy(float), "-", color=color, linewidth=1.1,
                    marker="o", markersize=2.5, alpha=0.9, label=name)
        ax.plot(runs[obs], yt[obs], "-", color="black", marker="o", markersize=4,
                linewidth=1.4, label="VB (observed)", zorder=10)
        if (~obs).any():
            ax.plot(runs[~obs], yt[~obs], "o", mfc="none", mec="gray", markersize=5,
                    linewidth=0, label="VB (interpolated)", zorder=10)
        ax.set_title(f"Case {c}", fontsize=10)
        ax.set_xlabel("run", fontsize=8); ax.set_ylabel("VB", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left", ncol=2)
    for idx in range(len(cases), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    sub = "  ".join(f"{n}={rmse[n]:.4f}" for n, _, _ in MODELS)
    fig.suptitle(f"6-model LOCV trajectory (B6 base 5 + Ridge)\n{sub}", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(out_dir / "figures" / "six_model_trajectory.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "figures" / "six_model_trajectory.svg"), bbox_inches="tight")
    plt.close(fig)

    print("Per-model observed RMSE:")
    for n, _, _ in MODELS:
        print(f"  {n:<13} {rmse[n]:.6f}")
    print(f"EXECUTION_DIR={out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""XGBoost prediction trajectory: sweep input length sample-by-sample from noload_end.

No-load 이후부터 sample 1개씩 늘리면서 XGBoost의 예측값 변화를 추적.
각 step L에서 signal[noload_end : noload_end+L]로부터 feature 추출 후 LOCV.

Key optimization: precompute ALL features for ALL L values as numpy arrays
(cumulative sum/sq/peak vectorization) → eliminates Python loop overhead in main sweep.
Python loop in original script was ~1.6s/step; vectorized version ~0s/step.

Model: XGBoost (AC+vS, mask=9), n_estimators=50, seed=0.
Step: 1 sample (user request).
Output: RMSE trajectory + per-run prediction trajectory.
"""
from __future__ import annotations

import sys
import time as time_mod
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

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
XGB_MASK      = 9       # AC+vS
SEED          = 0
N_ESTIMATORS  = 50      # fast for trajectory (shape matters, not absolute accuracy)
STEP          = 1       # sample step
SEG_CSV       = ROOT / "datasets/nasa/cutting_segment_v2/seg_peng2026_steady5_exitfix_reverse_kurtosis.csv"

XGB_CFG = dict(
    learning_rate=0.03, n_estimators=N_ESTIMATORS, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.9, n_jobs=4, verbosity=0,
    random_state=SEED,
)


# ─── Data utils ───────────────────────────────────────────────────────────────
def parse_signal(value: object) -> np.ndarray:
    return np.nan_to_num(
        np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"] == min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside")
    )
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    excl = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~excl].reset_index(drop=True)


def mask_sensor_indices(mask: int) -> list[int]:
    return [i * 4 + j for i in range(N_SENSORS) if (mask >> i) & 1 for j in range(4)]


# ─── Vectorized cumulative feature precomputation ─────────────────────────────
def precompute_all_features(
    signal_df: pd.DataFrame,
    proc_clean: pd.DataFrame,
    seg_idx: dict[tuple[int, int], dict],
    first_run: dict[int, int],
    sensor_indices: list[int],
    meta_arr: np.ndarray,          # [n_runs, 3] meta features
    max_L: int,
) -> np.ndarray:
    """
    Returns all_features: np.ndarray of shape [n_runs, max_L, n_delta+n_meta].

    all_features[run_idx, L, :] = delta features at window length L+1,
    where window = signal[noload_end : noload_end + (L+1)].

    Fully vectorized using cumulative sums — no per-step Python loops.
    """
    n_runs    = len(proc_clean)
    n_delta   = len(sensor_indices)
    n_meta    = len(META_FEATURES)
    n_feat    = n_delta + n_meta

    all_features = np.zeros((n_runs, max_L, n_feat), dtype=np.float32)

    # First pass: compute raw feature arrays for each run
    raw_per_run: dict[int, np.ndarray] = {}  # run_order → [max_L, 24]

    for run_idx, row in enumerate(proc_clean.itertuples(index=False)):
        case_id, run_id = int(row.case), int(row.run)
        sig_row = signal_df[
            (signal_df["case"] == case_id) & (signal_df["run"] == run_id)
        ]
        if sig_row.empty:
            raw_per_run[run_idx] = None
            continue
        sr = sig_row.iloc[0]
        arrays = {s: parse_signal(sr[s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            raw_per_run[run_idx] = None
            continue
        base_len = min(len(a) for a in arrays.values())

        seg = seg_idx.get((case_id, run_id))
        nl_end  = min(int(seg["idx_noload_end"]), base_len) if seg else 0
        cut_end = min(int(seg["idx_end"]), base_len)       if seg else base_len

        # Available cutting signal length for this run
        avail = cut_end - nl_end  # number of samples after noload_end up to idx_end

        # For each sensor: vectorize over L
        # raw[L, sensor_idx*4 + stat_idx] where stat = mean/rms/std/peak
        raw_run = np.zeros((max_L, N_SENSORS * 4), dtype=np.float32)

        for s_i, s in enumerate(SENSORS):
            arr = arrays[s][nl_end:cut_end].astype(np.float64)  # [avail]
            T   = min(avail, max_L)
            if T == 0:
                continue
            # Cumulative arrays
            cum_sum  = np.cumsum(arr[:T])          # [T]
            cum_sq   = np.cumsum(arr[:T] ** 2)    # [T]
            cum_peak = np.maximum.accumulate(np.abs(arr[:T]))  # [T]
            L_range  = np.arange(1, T + 1, dtype=np.float64)

            means = cum_sum  / L_range                                    # [T]
            rms   = np.sqrt(np.clip(cum_sq / L_range, 0, None))         # [T]
            var   = np.clip(cum_sq / L_range - means ** 2, 0, None)
            std   = np.sqrt(var)                                          # [T]
            peak  = cum_peak                                              # [T]

            base_col = s_i * 4
            raw_run[:T, base_col]     = means.astype(np.float32)
            raw_run[:T, base_col + 1] = rms.astype(np.float32)
            raw_run[:T, base_col + 2] = std.astype(np.float32)
            raw_run[:T, base_col + 3] = peak.astype(np.float32)

            if T < max_L:
                # Saturate: repeat last value for L > avail
                raw_run[T:, base_col]     = float(means[-1])
                raw_run[T:, base_col + 1] = float(rms[-1])
                raw_run[T:, base_col + 2] = float(std[-1])
                raw_run[T:, base_col + 3] = float(peak[-1])

        raw_per_run[run_idx] = raw_run  # [max_L, 24]

    # Second pass: compute delta (current - first_run) for selected sensor_indices
    case_to_first_idx: dict[int, int] = {}
    for run_idx, row in enumerate(proc_clean.itertuples(index=False)):
        case_id, run_id = int(row.case), int(row.run)
        fr = first_run.get(case_id)
        if fr == run_id:
            case_to_first_idx[case_id] = run_idx

    for run_idx, row in enumerate(proc_clean.itertuples(index=False)):
        case_id = int(row.case)
        raw = raw_per_run.get(run_idx)
        if raw is None:
            all_features[run_idx] = 0.0
            continue
        ref_idx = case_to_first_idx.get(case_id, run_idx)
        ref     = raw_per_run.get(ref_idx)
        if ref is None:
            ref = raw
        delta = (raw - ref)[:, sensor_indices]   # [max_L, n_delta]
        delta = np.where(np.isfinite(delta), delta, 0.0)
        all_features[run_idx, :, :n_delta] = delta.astype(np.float32)
        all_features[run_idx, :, n_delta:] = meta_arr[run_idx].astype(np.float32)

    return all_features   # [n_runs, max_L, n_feat]


# ─── LOCV (no-DataFrame, pure numpy) ─────────────────────────────────────────
def locv_step(
    X: np.ndarray,            # [n_runs, n_feat]
    vb: np.ndarray,           # [n_runs]
    case_arr: np.ndarray,     # [n_runs] int case IDs
    run_arr:  np.ndarray,     # [n_runs] int run IDs
    obs_mask: np.ndarray,     # [n_runs] bool — observed VB
) -> tuple[float, np.ndarray]:
    """LOCV at one L. Returns (mean_obs_rmse, pred_vb[n_runs])."""
    preds = np.full(len(vb), np.nan, dtype=np.float32)
    rmses: list[float] = []

    for tc in CASE_SCOPE:
        tr_mask = case_arr != tc
        te_mask = case_arr == tc
        if not tr_mask.any() or not te_mask.any():
            continue
        model = XGBRegressor(**XGB_CFG)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[tr_mask], vb[tr_mask])
            y_pred = model.predict(X[te_mask]).astype(np.float32)
        preds[te_mask] = y_pred
        obs_te = obs_mask[te_mask]
        if obs_te.any():
            rmses.append(float(np.sqrt(mean_squared_error(
                vb[te_mask][obs_te], y_pred[obs_te]
            ))))
    return float(np.mean(rmses)) if rmses else float("nan"), preds


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "analysis" / f"{ts}_xgb_timestamp_trajectory"
    for sub in ["data", "figures"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    t_start = time_mod.time()
    def log(msg: str) -> None:
        print(f"[{time_mod.time()-t_start:6.0f}s] {msg}", flush=True)

    log("=== XGBoost Timestamp Trajectory (vectorized) ===")
    log(f"Sensor: AC+vS (mask={XGB_MASK}), n_estimators={N_ESTIMATORS}, seed={SEED}, step={STEP}")

    log("Loading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    n_runs     = len(proc_clean)
    log(f"Clean runs: {n_runs}")

    seg_df = pd.read_csv(SEG_CSV)
    seg_df = seg_df[seg_df["case"].isin(CASE_SCOPE) & (seg_df["status"] == "labeled")]
    seg_idx: dict[tuple[int, int], dict] = {
        (int(r.case), int(r.run)): {
            "idx_noload_end": int(r.idx_noload_end),
            "idx_start":      int(r.idx_start),
            "idx_end":        int(r.idx_end),
        }
        for r in seg_df.itertuples(index=False)
    }

    # Segment stats for markers
    entry_lens  = [s["idx_start"] - s["idx_noload_end"] for s in seg_idx.values()]
    steady_lens = [s["idx_end"]   - s["idx_start"]      for s in seg_idx.values()]
    cut_lens    = [s["idx_end"]   - s["idx_noload_end"] for s in seg_idx.values()]
    mean_entry  = int(np.mean(entry_lens))
    mean_steady = int(np.mean(steady_lens))
    max_L       = max(cut_lens)
    log(f"Mean Entry zone: {mean_entry} samples | Mean Steady: {mean_steady} samples")
    log(f"max_L = {max_L}  (step={STEP} → {max_L // STEP} sweep points)")

    first_run: dict[int, int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(proc_clean[proc_clean["case"] == case_id]["run"].tolist())
        if runs:
            first_run[int(case_id)] = runs[0]

    # Meta and lookup arrays
    case_arr = proc_clean["case"].to_numpy(int)
    run_arr  = proc_clean["run"].to_numpy(int)
    vb_arr   = proc_clean["VB"].to_numpy(float)
    obs_mask = np.array(
        [(int(r.case), int(r.run)) not in NON_OBSERVED_RUNS
         for r in proc_clean.itertuples(index=False)], dtype=bool
    )
    meta_arr = proc_clean[META_FEATURES].to_numpy(float)

    sensor_indices = mask_sensor_indices(XGB_MASK)
    n_delta        = len(sensor_indices)

    log("\nPrecomputing feature matrix for all L values (vectorized)...")
    t_precomp = time_mod.time()
    all_features = precompute_all_features(
        signal_df, proc_clean, seg_idx, first_run,
        sensor_indices, meta_arr, max_L,
    )
    log(f"Feature matrix shape: {all_features.shape}  "
        f"[{all_features.nbytes / 1e6:.1f} MB]  "
        f"({time_mod.time()-t_precomp:.1f}s)")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    L_values    = list(range(1, max_L + 1, STEP))
    n_steps     = len(L_values)
    log(f"\nSweeping {n_steps} steps (L=1..{max_L}, step={STEP}) ...")
    log_interval = max(1, n_steps // 20)

    rmse_records:          list[dict]                   = []
    # Per-run predictions saved at reduced frequency
    pred_step              = max(1, n_steps // 500)
    pred_snapshots:        list[dict]                   = []   # {L, case, run, pred, true}

    t_sweep = time_mod.time()
    for step_i, L in enumerate(L_values):
        X         = all_features[:, L - 1, :]            # [n_runs, n_feat]
        X_finite  = np.where(np.isfinite(X), X, 0.0)
        mean_rmse, preds = locv_step(X_finite, vb_arr, case_arr, run_arr, obs_mask)
        rmse_records.append({"L": L, "rmse": mean_rmse})

        if step_i % pred_step == 0 or step_i == n_steps - 1:
            for ri in range(n_runs):
                pred_snapshots.append({
                    "L":           L,
                    "case_id":     int(case_arr[ri]),
                    "run":         int(run_arr[ri]),
                    "pred_vb":     float(preds[ri]) if np.isfinite(preds[ri]) else 0.0,
                    "true_vb":     float(vb_arr[ri]),
                    "is_observed": bool(obs_mask[ri]),
                })

        if (step_i + 1) % log_interval == 0 or step_i == n_steps - 1:
            elapsed  = time_mod.time() - t_sweep
            eta      = elapsed / (step_i + 1) * (n_steps - step_i - 1)
            log(f"  L={L:5d}/{max_L}  RMSE={mean_rmse:.4f}  ETA={eta:.0f}s")

    log("Sweep done.")

    rmse_df = pd.DataFrame(rmse_records)
    pred_df = pd.DataFrame(pred_snapshots)
    rmse_df.to_csv(out_dir / "data" / "rmse_trajectory.csv", index=False)
    pred_df.to_csv(out_dir / "data" / "pred_trajectory.csv", index=False)
    log(f"Saved: rmse_trajectory.csv ({len(rmse_df)} rows), "
        f"pred_trajectory.csv ({len(pred_df)} rows)")

    # ── Plot 1: RMSE trajectory ────────────────────────────────────────────────
    smooth_w = max(1, n_steps // 200)
    fig, ax  = plt.subplots(figsize=(13, 5))
    ax.plot(rmse_df["L"], rmse_df["rmse"], color="darkorange", linewidth=0.6, alpha=0.5)
    smooth   = rmse_df["rmse"].rolling(smooth_w, center=True).mean()
    ax.plot(rmse_df["L"], smooth, color="darkred", linewidth=2.0,
            label=f"RMSE (smooth w={smooth_w})")
    ax.axvline(mean_entry, color="steelblue", linestyle="--", linewidth=1.2,
               label=f"Mean Entry end ({mean_entry})")
    ax.axvline(mean_entry + mean_steady, color="green", linestyle="--", linewidth=1.2,
               label=f"Mean Steady end ({mean_entry+mean_steady})")
    ax.axhline(0.103660, color="gray",  linestyle=":", linewidth=1.2,
               label="prefix-90% best (0.1037)")
    ax.axhline(0.104067, color="black", linestyle=":", linewidth=1.0,
               label="Excl_Exit H13 (0.1041)")
    best_rmse = rmse_df["rmse"].min()
    best_L    = int(rmse_df.loc[rmse_df["rmse"].idxmin(), "L"])
    ax.axvline(best_L, color="crimson", linestyle=":", linewidth=1.2,
               label=f"Best L={best_L}  RMSE={best_rmse:.4f}")
    ax.set_xlabel("Samples from noload_end (L)")
    ax.set_ylabel("Observed-VB RMSE (XGBoost LOCV)")
    ax.set_title(f"XGBoost RMSE vs Input Length  "
                 f"[AC+vS, n_est={N_ESTIMATORS}, step={STEP}]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "figures" / "rmse_trajectory.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "figures" / "rmse_trajectory.svg"), bbox_inches="tight")
    plt.close(fig)
    log("Saved: rmse_trajectory.png")

    # ── Plot 2: Per-run prediction trajectories ────────────────────────────────
    obs_pred = pred_df[pred_df["is_observed"]].copy()

    fig, axes = plt.subplots(3, 5, figsize=(20, 12), sharex=True, sharey=True)
    for ax_i, case_id in enumerate(CASE_SCOPE):
        ax        = axes.flatten()[ax_i]
        case_runs = obs_pred[obs_pred["case_id"] == case_id]
        run_ids   = sorted(case_runs["run"].unique())
        colors    = cm.viridis(np.linspace(0.1, 0.9, max(len(run_ids), 1)))
        for run_id, color in zip(run_ids, colors):
            rdf = case_runs[case_runs["run"] == run_id].sort_values("L")
            if rdf.empty:
                continue
            true_vb     = rdf["true_vb"].iloc[0]
            smooth_pred = rdf["pred_vb"].rolling(3, center=True).mean()
            ax.plot(rdf["L"], smooth_pred, color=color, linewidth=0.8, alpha=0.8)
            ax.axhline(true_vb, color=color, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.axvline(mean_entry, color="steelblue", linestyle="--",
                   linewidth=0.7, alpha=0.5)
        ax.axvline(mean_entry + mean_steady, color="green", linestyle="--",
                   linewidth=0.7, alpha=0.5)
        ax.set_title(f"Case {case_id}", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.1, 0.7)
    for ax in axes.flatten():
        ax.set_xlabel("L (samples from noload_end)", fontsize=7)
        ax.set_ylabel("Predicted VB", fontsize=7)
    fig.suptitle(
        f"XGBoost Per-Run Prediction Trajectory  [AC+vS, n_est={N_ESTIMATORS}]\n"
        "Solid: smoothed pred | Dotted: true VB | "
        "Blue dashed: Entry end | Green dashed: Steady end",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(str(out_dir / "figures" / "pred_trajectory_per_case.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)
    log("Saved: pred_trajectory_per_case.png")

    # ── Summary ────────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    log(f"Best RMSE: {best_rmse:.6f} at L={best_L}")
    for mark_L in [mean_entry, mean_entry + mean_steady, max_L]:
        nearest = rmse_df.iloc[(rmse_df["L"] - mark_L).abs().argsort().iloc[0]]
        log(f"  RMSE at L={mark_L}: {nearest['rmse']:.6f}")
    log(f"prefix-90% reference: 0.103660")
    log(f"\nTotal time: {time_mod.time()-t_start:.0f}s")
    log(f"Results: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""H12_S1: Input ratio sweep for H11 top-1 sensor subset configurations.

T1: Feature-GRU — AC+vT+vS (mask=13), Delta+Meta (15-dim), H11_S3 best config
    H11 baseline: prefix=80% → RMSE=0.081977
T2: XGBoost    — AC+vS   (mask= 9), Delta+Meta (11-dim), H11_S1 best config
    H11 baseline: prefix=100% → RMSE=0.109247

Sweep: input_length_pct ∈ {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
Protocol: LOCV (15 cases), seeds=[0,1,2], observed_vb eval

Output: experiments/executions/H12/S1/{timestamp}_input_ratio_sweep/
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
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
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
RATIOS        = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
THRESH        = 1e6
N_SENSORS     = len(SENSORS)

GRU_MASK = 13   # AC+vT+vS: bit0(AC)=1, bit2(vT)=4, bit3(vS)=8 → 13
XGB_MASK = 9    # AC+vS:    bit0(AC)=1, bit3(vS)=8             → 9

H11_GRU_BASELINE = {"pct": 80,  "rmse": 0.081977}   # H11_S3_T1
H11_XGB_BASELINE = {"pct": 100, "rmse": 0.109247}   # H11_S1_T1

GRU_CFG = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0,
)
XGB_CFG = dict(
    learning_rate=0.03, n_estimators=600, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.9,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, n_jobs=4,
)


# ─── Signal / feature utils ───────────────────────────────────────────────────
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
    indices: list[int] = []
    for i in range(N_SENSORS):
        if (mask >> i) & 1:
            indices.extend([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])
    return indices


def mask_label(mask: int) -> str:
    return "+".join(SENSOR_ABBR[SENSORS[i]] for i in range(N_SENSORS) if (mask >> i) & 1)


# ─── Multi-prefix feature cache ───────────────────────────────────────────────
def build_multi_prefix_cache(
    signal_df: pd.DataFrame,
    proc_clean: pd.DataFrame,
    pcts: list[int],
) -> tuple[dict[tuple[int, int, int], np.ndarray], dict[int, int]]:
    """Build cache[(case_id, run_id, pct)] = 24-dim raw feature (all 6 sensors)."""
    cache: dict[tuple[int, int, int], np.ndarray] = {}
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
        for pct in pcts:
            end = max(1, int(np.ceil(base_len * pct / 100.0)))
            feats = np.concatenate([extract_features(arrays[s][:end]) for s in SENSORS])
            cache[(case_id, run_id, pct)] = feats

    first_run: dict[int, int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c, r, p) in cache if c == case_id and p == pcts[0])
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


# ─── GRU data builder ─────────────────────────────────────────────────────────
def build_gru_sequences(
    cache: dict[tuple[int, int, int], np.ndarray],
    first_run: dict[int, int],
    proc_clean: pd.DataFrame,
    mask: int,
    pct: int,
) -> dict[int, dict]:
    sensor_indices = mask_sensor_indices(mask)
    case_rows: dict[int, list[dict]] = {c: [] for c in CASE_SCOPE}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id, pct)
        if key not in cache:
            continue
        ref_run = first_run.get(case_id, run_id)
        ref_vec = cache.get((case_id, ref_run, pct), np.zeros(N_SENSORS * 4))
        delta = (cache[key] - ref_vec)[sensor_indices]
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        meta = np.array([float(getattr(row, mf, 0.0)) for mf in META_FEATURES], dtype=np.float64)
        feat = np.concatenate([delta, meta]).astype(np.float32)
        vb = float(row.VB) if not pd.isna(row.VB) else 0.0
        case_rows[case_id].append({"feat": feat, "vb": vb, "run": run_id})

    cases: dict[int, dict] = {}
    for case_id, rows in case_rows.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["run"])
        seq  = np.stack([r["feat"] for r in rows_sorted])
        vb   = np.array([r["vb"]  for r in rows_sorted], dtype=np.float32)
        runs = np.array([r["run"] for r in rows_sorted], dtype=int)
        cases[case_id] = {"seq": seq, "vb": vb, "runs": runs, "n_runs": len(rows_sorted)}
    return cases


# ─── GRU model ────────────────────────────────────────────────────────────────
class DeltaMetaGRU(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        cfg = GRU_CFG
        self.gru = nn.GRU(
            input_dim, cfg["hidden_size"], cfg["num_layers"],
            batch_first=True,
            dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg["hidden_size"], cfg["head_hidden"]),
            nn.ReLU(),
            nn.Linear(cfg["head_hidden"], 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out).squeeze(-1)


def fit_predict_gru(
    train_cases: dict[int, dict],
    test_cases: dict[int, dict],
    input_dim: int,
    device: torch.device,
    seed: int,
) -> dict[int, np.ndarray]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = DeltaMetaGRU(input_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=GRU_CFG["lr"], weight_decay=GRU_CFG["weight_decay"])
    epochs    = GRU_CFG["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_seqs = [info["seq"] for info in train_cases.values()]
    train_vbs  = [info["vb"]  for info in train_cases.values()]
    all_vb = np.concatenate(train_vbs)
    y_mean = float(all_vb.mean())
    y_std  = max(float(all_vb.std()), 1e-8)

    tensors = [torch.tensor(s, dtype=torch.float32) for s in train_seqs]
    lengths = torch.tensor([len(s) for s in train_seqs])
    x_train = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0).to(device)
    tgt     = [torch.tensor((t - y_mean) / y_std, dtype=torch.float32) for t in train_vbs]
    y_train = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=float("nan")).to(device)
    mask_tr = ~torch.isnan(y_train)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train, lengths)
        loss = ((pred[mask_tr] - y_train[mask_tr]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRU_CFG["grad_clip"])
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        return {
            cid: (
                model(
                    torch.tensor(info["seq"], dtype=torch.float32).unsqueeze(0).to(device),
                    torch.tensor([info["n_runs"]]),
                ).squeeze(0).cpu().numpy() * y_std + y_mean
            )
            for cid, info in test_cases.items()
        }


def run_gru_locv_seed(
    all_cases: dict[int, dict], input_dim: int, device: torch.device, seed: int
) -> float:
    rmses: list[float] = []
    for tc in CASE_SCOPE:
        if tc not in all_cases:
            continue
        train = {c: all_cases[c] for c in CASE_SCOPE if c != tc and c in all_cases}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = fit_predict_gru(train, {tc: all_cases[tc]}, input_dim, device, seed)
        runs = all_cases[tc]["runs"]
        obs  = np.array([(tc, int(r)) not in NON_OBSERVED_RUNS for r in runs])
        y_pred = preds[tc][obs]
        y_true = all_cases[tc]["vb"][obs]
        if len(y_true) == 0:
            continue
        rmses.append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
    return float(np.mean(rmses)) if rmses else float("nan")


# ─── XGB data builder ─────────────────────────────────────────────────────────
def build_xgb_df(
    cache: dict[tuple[int, int, int], np.ndarray],
    first_run: dict[int, int],
    proc_clean: pd.DataFrame,
    mask: int,
    pct: int,
) -> pd.DataFrame:
    sensor_indices = mask_sensor_indices(mask)
    rows: list[dict] = []
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id, pct)
        if key not in cache:
            continue
        ref_run = first_run.get(case_id, run_id)
        ref_vec = cache.get((case_id, ref_run, pct), np.zeros(N_SENSORS * 4))
        delta   = (cache[key] - ref_vec)[sensor_indices]
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        vb  = float(row.VB) if not pd.isna(row.VB) else 0.0
        rec: dict = {"case_id": case_id, "run": run_id, "VB": vb}
        for mf in META_FEATURES:
            rec[mf] = float(getattr(row, mf, 0.0))
        for k, val in enumerate(delta):
            rec[f"f{k}"] = float(val)
        rows.append(rec)
    return pd.DataFrame(rows)


def run_xgb_locv_seed(feat_df: pd.DataFrame, feature_cols: list[str], seed: int) -> float:
    rmses: list[float] = []
    for tc in CASE_SCOPE:
        test_df  = feat_df[feat_df["case_id"] == tc]
        train_df = feat_df[feat_df["case_id"] != tc]
        if test_df.empty or train_df.empty:
            continue
        model = XGBRegressor(**XGB_CFG, random_state=seed, verbosity=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_df[feature_cols], train_df["VB"])
            y_pred = model.predict(test_df[feature_cols])
        obs = test_df.apply(
            lambda r: (int(r["case_id"]), int(r["run"])) not in NON_OBSERVED_RUNS, axis=1
        ).to_numpy()
        y_true = test_df["VB"].to_numpy(float)[obs]
        y_pred_obs = y_pred[obs]
        if len(y_true) == 0:
            continue
        rmses.append(float(np.sqrt(mean_squared_error(y_true, y_pred_obs))))
    return float(np.mean(rmses)) if rmses else float("nan")


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_ratio_curve(
    gru_results: list[dict],
    xgb_results: list[dict],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, results, label, color, baseline in [
        (axes[0], gru_results, "Feature-GRU (AC+vT+vS)", "steelblue", H11_GRU_BASELINE),
        (axes[1], xgb_results, "XGBoost (AC+vS)",        "darkorange", H11_XGB_BASELINE),
    ]:
        pcts  = [r["pct"]  for r in results]
        means = [r["mean_rmse"] for r in results]
        stds  = [r["std_rmse"]  for r in results]
        ax.errorbar(pcts, means, yerr=stds, marker="o", capsize=4,
                    color=color, label=label, linewidth=1.8)
        ax.axhline(baseline["rmse"], color="gray", linestyle="--", linewidth=1.2,
                   label=f"H11 baseline ({baseline['pct']}%): {baseline['rmse']:.4f}")
        best = min(results, key=lambda r: r["mean_rmse"])
        ax.axvline(best["pct"], color="crimson", linestyle=":", linewidth=1.2,
                   label=f"Best={best['pct']}%  RMSE={best['mean_rmse']:.4f}")
        ax.set_xlabel("Input length (%)")
        ax.set_ylabel("Observed-VB RMSE (3-seed mean)")
        ax.set_title(label)
        ax.set_xticks(pcts)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("H12_S1: Input Ratio Sweep — H11 Top-1 Configs", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(out_path.with_suffix(".png")), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".svg")), bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "H12" / "S1" / f"{ts}_input_ratio_sweep"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H12_S1: Input Ratio Sweep for H11 Top-1 Configs ===")
    log(f"T1: Feature-GRU  mask={GRU_MASK} ({mask_label(GRU_MASK)})  baseline={H11_GRU_BASELINE}")
    log(f"T2: XGBoost      mask={XGB_MASK} ({mask_label(XGB_MASK)})  baseline={H11_XGB_BASELINE}")
    log(f"Ratios={RATIOS}, Seeds={SEEDS}, LOCV={len(CASE_SCOPE)} cases")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    log(f"Building multi-prefix cache (pcts={RATIOS})...")
    cache, first_run = build_multi_prefix_cache(signal_df, proc_clean, RATIOS)
    log(f"Cache size: {len(cache)} entries ({len(RATIOS)} pcts × ~{len(cache)//len(RATIOS)} runs)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ── T1: Feature-GRU sweep ─────────────────────────────────────────────────
    gru_n_sensors = bin(GRU_MASK).count("1")
    gru_input_dim = gru_n_sensors * 4 + len(META_FEATURES)
    log(f"\n--- T1: Feature-GRU  input_dim={gru_input_dim}  label={mask_label(GRU_MASK)} ---")
    gru_results: list[dict] = []
    t0 = time_mod.time()

    for pct in RATIOS:
        all_cases = build_gru_sequences(cache, first_run, proc_clean, GRU_MASK, pct)
        seed_rmses: list[float] = []
        for seed in SEEDS:
            rmse = run_gru_locv_seed(all_cases, gru_input_dim, device, seed)
            seed_rmses.append(rmse)
        mean_rmse = float(np.mean(seed_rmses))
        std_rmse  = float(np.std(seed_rmses))
        gru_results.append({"pct": pct, "mean_rmse": mean_rmse, "std_rmse": std_rmse,
                            "seed_rmses": seed_rmses})
        elapsed = time_mod.time() - t0
        log(f"  GRU pct={pct:3d}%  RMSE={mean_rmse:.6f}  std={std_rmse:.6f}  [{elapsed:.0f}s]")

    gru_best = min(gru_results, key=lambda r: r["mean_rmse"])
    log(f"  GRU best: pct={gru_best['pct']}%  RMSE={gru_best['mean_rmse']:.6f}"
        f"  H11 baseline={H11_GRU_BASELINE['rmse']:.6f}"
        f"  Δ={gru_best['mean_rmse']-H11_GRU_BASELINE['rmse']:+.6f}")

    # ── T2: XGBoost sweep ─────────────────────────────────────────────────────
    n_delta = bin(XGB_MASK).count("1") * 4
    delta_cols   = [f"f{k}" for k in range(n_delta)]
    feature_cols = delta_cols + META_FEATURES
    log(f"\n--- T2: XGBoost  n_feat={len(feature_cols)}  label={mask_label(XGB_MASK)} ---")
    xgb_results: list[dict] = []
    t0 = time_mod.time()

    for pct in RATIOS:
        feat_df = build_xgb_df(cache, first_run, proc_clean, XGB_MASK, pct)
        feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat_df.fillna(0.0, inplace=True)
        seed_rmses = []
        for seed in SEEDS:
            rmse = run_xgb_locv_seed(feat_df, feature_cols, seed)
            seed_rmses.append(rmse)
        mean_rmse = float(np.mean(seed_rmses))
        std_rmse  = float(np.std(seed_rmses))
        xgb_results.append({"pct": pct, "mean_rmse": mean_rmse, "std_rmse": std_rmse,
                            "seed_rmses": seed_rmses})
        elapsed = time_mod.time() - t0
        log(f"  XGB pct={pct:3d}%  RMSE={mean_rmse:.6f}  std={std_rmse:.6f}  [{elapsed:.0f}s]")

    xgb_best = min(xgb_results, key=lambda r: r["mean_rmse"])
    log(f"  XGB best: pct={xgb_best['pct']}%  RMSE={xgb_best['mean_rmse']:.6f}"
        f"  H11 baseline={H11_XGB_BASELINE['rmse']:.6f}"
        f"  Δ={xgb_best['mean_rmse']-H11_XGB_BASELINE['rmse']:+.6f}")

    # ── Save results ──────────────────────────────────────────────────────────
    gru_df = pd.DataFrame([
        {"model": "feature_gru", "mask": GRU_MASK, "sensor_subset": mask_label(GRU_MASK),
         "pct": r["pct"], "mean_rmse": r["mean_rmse"], "std_rmse": r["std_rmse"],
         **{f"seed{i}_rmse": r["seed_rmses"][i] for i in range(len(SEEDS))}}
        for r in gru_results
    ])
    xgb_df = pd.DataFrame([
        {"model": "xgboost", "mask": XGB_MASK, "sensor_subset": mask_label(XGB_MASK),
         "pct": r["pct"], "mean_rmse": r["mean_rmse"], "std_rmse": r["std_rmse"],
         **{f"seed{i}_rmse": r["seed_rmses"][i] for i in range(len(SEEDS))}}
        for r in xgb_results
    ])
    all_df = pd.concat([gru_df, xgb_df], ignore_index=True)
    all_df.to_csv(out_dir / "metrics" / "input_ratio_results.csv", index=False)
    gru_df.to_csv(out_dir / "metrics" / "T1_gru_results.csv", index=False)
    xgb_df.to_csv(out_dir / "metrics" / "T2_xgb_results.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_ratio_curve(gru_results, xgb_results, out_dir / "figures" / "input_ratio_curve")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    log(f"{'':6} {'pct':>5}  {'GRU_RMSE':>10}  {'XGB_RMSE':>10}")
    log("-" * 36)
    for g, x in zip(gru_results, xgb_results):
        g_best = "★" if g["pct"] == gru_best["pct"] else " "
        x_best = "★" if x["pct"] == xgb_best["pct"] else " "
        log(f"{g_best+x_best:6} {g['pct']:>3d}%  {g['mean_rmse']:>10.6f}  {x['mean_rmse']:>10.6f}")
    log(f"\nGRU best: {gru_best['pct']}%  RMSE={gru_best['mean_rmse']:.6f}"
        f"  (H11: {H11_GRU_BASELINE['pct']}% → {H11_GRU_BASELINE['rmse']:.6f})")
    log(f"XGB best: {xgb_best['pct']}%  RMSE={xgb_best['mean_rmse']:.6f}"
        f"  (H11: {H11_XGB_BASELINE['pct']}% → {H11_XGB_BASELINE['rmse']:.6f})")

    summary = {
        "experiment": "H12_S1_input_ratio_sweep",
        "gru": {
            "mask": GRU_MASK, "sensor_subset": mask_label(GRU_MASK),
            "best_pct": gru_best["pct"], "best_rmse": gru_best["mean_rmse"],
            "h11_baseline": H11_GRU_BASELINE,
            "results": gru_results,
        },
        "xgb": {
            "mask": XGB_MASK, "sensor_subset": mask_label(XGB_MASK),
            "best_pct": xgb_best["pct"], "best_rmse": xgb_best["mean_rmse"],
            "h11_baseline": H11_XGB_BASELINE,
            "results": xgb_results,
        },
        "seeds": SEEDS, "ratios": RATIOS,
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

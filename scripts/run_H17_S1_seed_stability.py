#!/usr/bin/env python3
"""H17_S1: Seed stability experiment — 100% input, 5 seeds.

논문 보고 수치의 안정성 검증. Prefix/segmentation 없이 전체 신호(100%) 사용.

T1: Feature-GRU — AC+vT+vS (mask=13), Delta+Meta 15-dim
    Ref: H12_S1_T1 pct=100% 3-seed mean RMSE=0.095010
T2: XGBoost    — AC+vS   (mask= 9), Delta+Meta 11-dim
    Ref: H12_S1_T2 pct=100% 3-seed mean RMSE=0.109247

Seeds: [0, 1, 2, 3, 4]
Protocol: LOCV (15 cases), observed_vb eval
Output: experiments/executions/H17/S1/{timestamp}_seed_stability/
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
SEEDS         = [0, 1, 2, 3, 4]
PCT           = 100
THRESH        = 1e6
N_SENSORS     = len(SENSORS)

GRU_MASK = 13   # AC+vT+vS
XGB_MASK = 9    # AC+vS

REF_GRU_3SEED = 0.095010   # H12_S1_T1 pct=100%, 3-seed
REF_XGB_3SEED = 0.109247   # H12_S1_T2 pct=100%, 3-seed

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
    return [
        i * 4 + j
        for i in range(N_SENSORS) if (mask >> i) & 1
        for j in range(4)
    ]


def mask_label(mask: int) -> str:
    return "+".join(SENSOR_ABBR[SENSORS[i]] for i in range(N_SENSORS) if (mask >> i) & 1)


# ─── Feature cache (single pct) ───────────────────────────────────────────────
def build_cache(
    signal_df: pd.DataFrame,
    proc_clean: pd.DataFrame,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[int, int]]:
    cache: dict[tuple[int, int], np.ndarray] = {}
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
        end = max(1, int(np.ceil(base_len * PCT / 100.0)))
        feats = np.concatenate([extract_features(arrays[s][:end]) for s in SENSORS])
        cache[(case_id, run_id)] = feats

    first_run: dict[int, int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c, r) in cache if c == case_id)
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


# ─── GRU data builder ─────────────────────────────────────────────────────────
def build_gru_sequences(
    cache: dict[tuple[int, int], np.ndarray],
    first_run: dict[int, int],
    proc_clean: pd.DataFrame,
    mask: int,
) -> dict[int, dict]:
    sensor_indices = mask_sensor_indices(mask)
    case_rows: dict[int, list[dict]] = {c: [] for c in CASE_SCOPE}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id)
        if key not in cache:
            continue
        ref_run = first_run.get(case_id, run_id)
        ref_vec = cache.get((case_id, ref_run), np.zeros(N_SENSORS * 4))
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
) -> tuple[float, dict[int, float]]:
    """Returns (mean_rmse, {case: rmse})."""
    case_rmses: dict[int, float] = {}
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
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mean_rmse = float(np.mean(list(case_rmses.values()))) if case_rmses else float("nan")
    return mean_rmse, case_rmses


# ─── XGB data builder ─────────────────────────────────────────────────────────
def build_xgb_df(
    cache: dict[tuple[int, int], np.ndarray],
    first_run: dict[int, int],
    proc_clean: pd.DataFrame,
    mask: int,
) -> pd.DataFrame:
    sensor_indices = mask_sensor_indices(mask)
    rows: list[dict] = []
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id)
        if key not in cache:
            continue
        ref_run = first_run.get(case_id, run_id)
        ref_vec = cache.get((case_id, ref_run), np.zeros(N_SENSORS * 4))
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


def run_xgb_locv_seed(
    feat_df: pd.DataFrame, feature_cols: list[str], seed: int
) -> tuple[float, dict[int, float]]:
    """Returns (mean_rmse, {case: rmse})."""
    case_rmses: dict[int, float] = {}
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
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_true, y_pred_obs)))
    mean_rmse = float(np.mean(list(case_rmses.values()))) if case_rmses else float("nan")
    return mean_rmse, case_rmses


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_stability(
    gru_seed_rmses: list[float],
    xgb_seed_rmses: list[float],
    gru_case_matrix: dict[int, list[float]],
    xgb_case_matrix: dict[int, list[float]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: seed-level aggregate RMSE
    ax = axes[0]
    x = np.arange(len(SEEDS))
    w = 0.35
    ax.bar(x - w/2, gru_seed_rmses, w, label=f"Feature-GRU ({mask_label(GRU_MASK)})",
           color="steelblue", alpha=0.85)
    ax.bar(x + w/2, xgb_seed_rmses, w, label=f"XGBoost ({mask_label(XGB_MASK)})",
           color="darkorange", alpha=0.85)
    ax.axhline(np.mean(gru_seed_rmses), color="steelblue", linestyle="--", linewidth=1.2,
               label=f"GRU mean={np.mean(gru_seed_rmses):.4f}")
    ax.axhline(np.mean(xgb_seed_rmses), color="darkorange", linestyle="--", linewidth=1.2,
               label=f"XGB mean={np.mean(xgb_seed_rmses):.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("Observed-VB RMSE")
    ax.set_title("Aggregate RMSE per Seed")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: per-case RMSE box plot (GRU)
    ax = axes[1]
    case_ids = sorted(gru_case_matrix.keys())
    data = [gru_case_matrix[c] for c in case_ids]
    bp = ax.boxplot(data, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.5),
                    medianprops=dict(color="navy"))
    ax.set_xticks(range(1, len(case_ids) + 1))
    ax.set_xticklabels([f"C{c}" for c in case_ids], fontsize=8)
    ax.set_ylabel("RMSE (5 seeds)")
    ax.set_title(f"Feature-GRU Per-Case RMSE Distribution ({len(SEEDS)} seeds)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"H17_S1: Seed Stability — 100% Input, {len(SEEDS)} Seeds", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(out_dir / "seed_stability.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "seed_stability.svg"), bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "H17" / "S1" / f"{ts}_seed_stability"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H17_S1: Seed Stability — 100% Input, 5 Seeds ===")
    log(f"T1: Feature-GRU  mask={GRU_MASK} ({mask_label(GRU_MASK)})  ref_3seed={REF_GRU_3SEED}")
    log(f"T2: XGBoost      mask={XGB_MASK} ({mask_label(XGB_MASK)})  ref_3seed={REF_XGB_3SEED}")
    log(f"PCT={PCT}%, Seeds={SEEDS}, LOCV={len(CASE_SCOPE)} cases")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/processed/mill_signal_data.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/processed/mill_process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    log("Building feature cache (pct=100%)...")
    cache, first_run = build_cache(signal_df, proc_clean)
    log(f"Cache size: {len(cache)} runs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ── T1: Feature-GRU ──────────────────────────────────────────────────────
    gru_n_sensors = bin(GRU_MASK).count("1")
    gru_input_dim = gru_n_sensors * 4 + len(META_FEATURES)
    log(f"\n--- T1: Feature-GRU  input_dim={gru_input_dim}  label={mask_label(GRU_MASK)} ---")
    all_cases = build_gru_sequences(cache, first_run, proc_clean, GRU_MASK)

    gru_seed_rmses: list[float] = []
    gru_case_matrix: dict[int, list[float]] = {c: [] for c in CASE_SCOPE}
    t0 = time_mod.time()

    for seed in SEEDS:
        mean_rmse, case_rmses = run_gru_locv_seed(all_cases, gru_input_dim, device, seed)
        gru_seed_rmses.append(mean_rmse)
        for c, r in case_rmses.items():
            gru_case_matrix[c].append(r)
        elapsed = time_mod.time() - t0
        log(f"  seed={seed}  RMSE={mean_rmse:.6f}  [{elapsed:.0f}s]")

    gru_mean = float(np.mean(gru_seed_rmses))
    gru_std  = float(np.std(gru_seed_rmses))
    gru_cv   = gru_std / gru_mean if gru_mean > 0 else float("nan")
    log(f"  GRU 5-seed: mean={gru_mean:.6f}  std={gru_std:.6f}  CV={gru_cv:.4f}"
        f"  (3-seed ref={REF_GRU_3SEED:.6f})")

    # ── T2: XGBoost ──────────────────────────────────────────────────────────
    n_delta = bin(XGB_MASK).count("1") * 4
    delta_cols   = [f"f{k}" for k in range(n_delta)]
    feature_cols = delta_cols + META_FEATURES
    log(f"\n--- T2: XGBoost  n_feat={len(feature_cols)}  label={mask_label(XGB_MASK)} ---")
    feat_df = build_xgb_df(cache, first_run, proc_clean, XGB_MASK)
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_df.fillna(0.0, inplace=True)

    xgb_seed_rmses: list[float] = []
    xgb_case_matrix: dict[int, list[float]] = {c: [] for c in CASE_SCOPE}
    t0 = time_mod.time()

    for seed in SEEDS:
        mean_rmse, case_rmses = run_xgb_locv_seed(feat_df, feature_cols, seed)
        xgb_seed_rmses.append(mean_rmse)
        for c, r in case_rmses.items():
            xgb_case_matrix[c].append(r)
        elapsed = time_mod.time() - t0
        log(f"  seed={seed}  RMSE={mean_rmse:.6f}  [{elapsed:.0f}s]")

    xgb_mean = float(np.mean(xgb_seed_rmses))
    xgb_std  = float(np.std(xgb_seed_rmses))
    xgb_cv   = xgb_std / xgb_mean if xgb_mean > 0 else float("nan")
    log(f"  XGB 5-seed: mean={xgb_mean:.6f}  std={xgb_std:.6f}  CV={xgb_cv:.4f}"
        f"  (3-seed ref={REF_XGB_3SEED:.6f})")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    # Aggregate seed table
    agg_df = pd.DataFrame({
        "seed": SEEDS,
        "gru_rmse": gru_seed_rmses,
        "xgb_rmse": xgb_seed_rmses,
    })
    agg_df.to_csv(out_dir / "metrics" / "seed_aggregate.csv", index=False)

    # Per-case table (GRU)
    gru_rows = []
    for c in CASE_SCOPE:
        if not gru_case_matrix[c]:
            continue
        row = {"case": c}
        for i, s in enumerate(SEEDS):
            row[f"seed{s}"] = gru_case_matrix[c][i] if i < len(gru_case_matrix[c]) else float("nan")
        vals = [v for v in gru_case_matrix[c] if not np.isnan(v)]
        row["mean"] = float(np.mean(vals)) if vals else float("nan")
        row["std"]  = float(np.std(vals))  if vals else float("nan")
        gru_rows.append(row)
    gru_case_df = pd.DataFrame(gru_rows)
    gru_case_df.to_csv(out_dir / "metrics" / "gru_per_case.csv", index=False)

    # Per-case table (XGB)
    xgb_rows = []
    for c in CASE_SCOPE:
        if not xgb_case_matrix[c]:
            continue
        row = {"case": c}
        for i, s in enumerate(SEEDS):
            row[f"seed{s}"] = xgb_case_matrix[c][i] if i < len(xgb_case_matrix[c]) else float("nan")
        vals = [v for v in xgb_case_matrix[c] if not np.isnan(v)]
        row["mean"] = float(np.mean(vals)) if vals else float("nan")
        row["std"]  = float(np.std(vals))  if vals else float("nan")
        xgb_rows.append(row)
    xgb_case_df = pd.DataFrame(xgb_rows)
    xgb_case_df.to_csv(out_dir / "metrics" / "xgb_per_case.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_stability(
        gru_seed_rmses, xgb_seed_rmses,
        {c: v for c, v in gru_case_matrix.items() if v},
        {c: v for c, v in xgb_case_matrix.items() if v},
        out_dir / "figures",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    log(f"{'Model':<20} {'Mean':>10} {'Std':>10} {'CV':>8} {'3-seed ref':>12} {'Δ':>10}")
    log("-" * 74)
    log(f"{'GRU '+mask_label(GRU_MASK):<20} {gru_mean:>10.6f} {gru_std:>10.6f}"
        f" {gru_cv:>8.4f} {REF_GRU_3SEED:>12.6f} {gru_mean-REF_GRU_3SEED:>+10.6f}")
    log(f"{'XGB '+mask_label(XGB_MASK):<20} {xgb_mean:>10.6f} {xgb_std:>10.6f}"
        f" {xgb_cv:>8.4f} {REF_XGB_3SEED:>12.6f} {xgb_mean-REF_XGB_3SEED:>+10.6f}")

    log("\nGRU per-seed:")
    for s, r in zip(SEEDS, gru_seed_rmses):
        log(f"  seed={s}  {r:.6f}")
    log("\nXGB per-seed:")
    for s, r in zip(SEEDS, xgb_seed_rmses):
        log(f"  seed={s}  {r:.6f}")

    summary = {
        "experiment": "H17_S1_seed_stability",
        "pct": PCT,
        "seeds": SEEDS,
        "gru": {
            "mask": GRU_MASK, "sensor_subset": mask_label(GRU_MASK),
            "input_dim": gru_input_dim,
            "seed_rmses": gru_seed_rmses,
            "mean": gru_mean, "std": gru_std, "cv": gru_cv,
            "ref_3seed": REF_GRU_3SEED,
        },
        "xgb": {
            "mask": XGB_MASK, "sensor_subset": mask_label(XGB_MASK),
            "n_features": len(feature_cols),
            "seed_rmses": xgb_seed_rmses,
            "mean": xgb_mean, "std": xgb_std, "cv": xgb_cv,
            "ref_3seed": REF_XGB_3SEED,
        },
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

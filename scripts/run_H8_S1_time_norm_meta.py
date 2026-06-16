#!/usr/bin/env python3
"""H8_S1: Add case-wise normalized cumulative time as 4th meta feature.

Hypothesis: Cumulative machining time (case-wise min-max normalized to [0,1])
provides additional temporal context beyond run order and improves GRU/XGBoost.

Experiment:
  - Feature set A (baseline): Delta+Meta 27 = 24 delta + [DOC, feed, material]
  - Feature set B (+time_norm): Delta+Meta 28 = 24 delta + [DOC, feed, material, time_norm]
  - GRU best config:     head32 (head_hidden=32), prefix=80%, 3 seeds
  - XGBoost best config: g_lr0.03_n600_d3_mcw8, prefix=100%, 3 seeds
  - Evaluation: LOCV 15 cases

time_norm = (time - case_min) / (case_max - case_min)
computed after preprocess() so EXCLUDED_RUNS are already removed.

Output:
  experiments/executions/H8/S1/{timestamp}_time_norm_meta/
"""
from __future__ import annotations

import json
import sys
import time as time_mod
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
EXCLUDED_RUNS = {(2, 1), (12, 1)}
META_BASE = ["DOC", "feed", "material"]
META_TIME = ["DOC", "feed", "material", "time_norm"]
SEEDS = [0, 1, 2]
GRU_PREFIX = 80
XGB_PREFIX = 100

# Best GRU config (H4_S5 head32)
GRU_CFG = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0, scheduler="cosine",
)

# Best XGBoost config (H4_S9 g_lr0.03_n600_d3_mcw8)
XGB_CFG = dict(
    learning_rate=0.03, n_estimators=600, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.9,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
    n_jobs=4,
)


# ─── Feature extraction ───────────────────────────────────────────────────────
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


def delta_col_names() -> list[str]:
    return [f"{s}__delta_{f}" for s in SENSORS for f in ["mean", "rms", "std", "peak"]]


def preprocess(process: pd.DataFrame) -> pd.DataFrame:
    df = process.copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run_per_case = df.groupby("case")["run"].transform("min")
    is_run1 = df["run"] == min_run_per_case
    df.loc[is_run1 & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside")
    )
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    df = df[~mask].reset_index(drop=True)

    # Case-wise time normalization (applied after EXCLUDED_RUNS removal)
    df["time_norm"] = df.groupby("case")["time"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0
    )
    return df


def build_feature_matrix(
    signal_df: pd.DataFrame,
    process_df: pd.DataFrame,
    prefix_percents: list[int],
    meta_features: list[str],
) -> pd.DataFrame:
    delta_cols = delta_col_names()
    THRESH = 1e6
    cache: dict[tuple[int, int, int], np.ndarray] = {}
    for row in process_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        sig_row = signal_df[(signal_df["case"] == case_id) & (signal_df["run"] == run_id)]
        if sig_row.empty:
            continue
        sig_row = sig_row.iloc[0]
        arrays = {s: parse_signal(sig_row[s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        base_len = min(len(a) for a in arrays.values())
        for pct in prefix_percents:
            end = max(1, int(np.ceil(base_len * pct / 100.0)))
            feats = np.concatenate([extract_features(arrays[s][:end]) for s in SENSORS])
            cache[(case_id, run_id, pct)] = feats

    first_run: dict[int, int] = {}
    for case_id in process_df["case"].unique():
        runs = sorted(r for (c, r, _) in cache if c == case_id and (c, r) not in EXCLUDED_RUNS)
        if runs:
            first_run[int(case_id)] = runs[0]

    rows: list[dict] = []
    for row in process_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        if (case_id, run_id) in EXCLUDED_RUNS:
            continue
        ref_run = first_run.get(case_id, run_id)
        vb = float(row.VB) if not pd.isna(row.VB) else 0.0
        for pct in prefix_percents:
            raw_feats = cache.get((case_id, run_id, pct))
            ref_feats = cache.get((case_id, ref_run, pct))
            if raw_feats is None or ref_feats is None:
                continue
            delta_feats = raw_feats - ref_feats
            record: dict = {"case_id": case_id, "run": run_id, "signal_prefix_percent": float(pct), "VB": vb}
            for mf in meta_features:
                record[mf] = float(getattr(row, mf, 0.0))
            for col, val in zip(delta_cols, delta_feats):
                record[col] = float(val)
            rows.append(record)
    return pd.DataFrame(rows).sort_values(
        ["signal_prefix_percent", "case_id", "run"]
    ).reset_index(drop=True)


# ─── GRU model ────────────────────────────────────────────────────────────────
class DeltaMetaGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int,
                 dropout: float, head_hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out).squeeze(-1)


def build_sequences(feature_df: pd.DataFrame, delta_cols: list[str],
                    meta_features: list[str], pct: float) -> dict[int, dict]:
    subset = feature_df[feature_df["signal_prefix_percent"] == float(pct)].copy()
    cases: dict[int, dict] = {}
    for case_id in subset["case_id"].unique():
        c = subset[subset["case_id"] == case_id].sort_values("run").reset_index(drop=True)
        delta = c[delta_cols].to_numpy(dtype=np.float32)
        meta = c[meta_features].to_numpy(dtype=np.float32)
        seq = np.concatenate([delta, meta], axis=1)
        cases[int(case_id)] = {"seq": seq, "vb": c["VB"].to_numpy(dtype=np.float32), "n_runs": len(c)}
    return cases


def fit_predict_gru(
    train_cases: dict[int, dict], test_cases: dict[int, dict],
    input_dim: int, device: torch.device, cfg: dict, seed: int,
) -> dict[int, np.ndarray]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = DeltaMetaGRU(
        input_dim=input_dim, hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"], dropout=cfg["dropout"],
        head_hidden=cfg["head_hidden"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    epochs = cfg["epochs"]
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if cfg.get("scheduler") == "cosine" else None
    )

    train_seqs = [info["seq"] for info in train_cases.values()]
    train_vbs = [info["vb"] for info in train_cases.values()]
    all_vb = np.concatenate(train_vbs)
    y_mean = float(all_vb.mean())
    y_std = float(all_vb.std()) if all_vb.std() > 1e-8 else 1.0

    tensors = [torch.tensor(s, dtype=torch.float32) for s in train_seqs]
    lengths = torch.tensor([len(s) for s in train_seqs])
    x_train = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0).to(device)
    tgt = [torch.tensor((t - y_mean) / y_std, dtype=torch.float32) for t in train_vbs]
    y_train = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=float("nan")).to(device)
    mask_train = ~torch.isnan(y_train)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train, lengths)
        loss = ((pred[mask_train] - y_train[mask_train]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    model.eval()
    predictions: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for case_id, info in test_cases.items():
            seq_t = torch.tensor(info["seq"], dtype=torch.float32).unsqueeze(0).to(device)
            length_t = torch.tensor([info["n_runs"]])
            pred_norm = model(seq_t, length_t).squeeze(0).cpu().numpy()
            predictions[case_id] = pred_norm * y_std + y_mean
    return predictions


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def run_gru_locv(all_case_data: dict[int, dict], input_dim: int,
                 device: torch.device, seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    for test_case in CASE_SCOPE:
        if test_case not in all_case_data:
            continue
        train_ids = [c for c in CASE_SCOPE if c != test_case and c in all_case_data]
        train_data = {c: all_case_data[c] for c in train_ids}
        test_data = {test_case: all_case_data[test_case]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = fit_predict_gru(train_data, test_data, input_dim, device, GRU_CFG, seed)
        y_pred = preds[test_case]
        y_true = all_case_data[test_case]["vb"]
        rows.append({"target_case": test_case, "n_runs": len(y_true), **compute_metrics(y_true, y_pred)})
    return pd.DataFrame(rows)


def run_xgb_locv(feat_df: pd.DataFrame, feature_cols: list[str], pct: int, seed: int) -> pd.DataFrame:
    subset = feat_df[feat_df["signal_prefix_percent"] == float(pct)].copy()
    rows: list[dict] = []
    for tc in CASE_SCOPE:
        test_df = subset[subset["case_id"] == tc]
        train_df = subset[subset["case_id"] != tc]
        if test_df.empty or train_df.empty:
            continue
        m = XGBRegressor(**XGB_CFG, random_state=seed, verbosity=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(train_df[feature_cols], train_df["VB"])
            y_pred = m.predict(test_df[feature_cols])
        y_true = test_df["VB"].to_numpy(dtype=float)
        rows.append({"target_case": tc, "n_runs": len(y_true), **compute_metrics(y_true, y_pred)})
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "H8" / "S1" / f"{ts}_time_norm_meta"
    for sub in ["metrics", "analysis", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H8_S1: time_norm meta feature experiment ===")
    log(f"GRU: head32, prefix={GRU_PREFIX}%, seeds={SEEDS}")
    log(f"XGB: g_lr0.03_n600_d3_mcw8, prefix={XGB_PREFIX}%, seeds={SEEDS}")

    # ── Load data ─────────────────────────────────────────────────────────────
    log("Loading data...")
    signal_df = pd.read_csv(
        ROOT / "datasets/nasa/raw_signal.csv",
        usecols=["case", "run"] + SENSORS,
    )
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()

    log("Preprocessing (VB interpolation + excluded runs + case-wise time_norm)...")
    process_clean = preprocess(process_df)

    # Verify time_norm
    log("time_norm stats per case (sample):")
    for cid in [1, 11, 12]:
        sub = process_clean[process_clean["case"] == cid][["run", "time", "time_norm"]]
        log(f"  Case {cid}: {sub.to_dict('records')}")

    delta_cols = delta_col_names()

    # ── Build feature matrices (baseline and +time_norm) ──────────────────────
    log("Building feature matrix (baseline, prefix 80 & 100)...")
    feat_base_gru = build_feature_matrix(signal_df, process_clean, [GRU_PREFIX], META_BASE)
    feat_base_xgb = build_feature_matrix(signal_df, process_clean, [XGB_PREFIX], META_BASE)
    log("Building feature matrix (+time_norm, prefix 80 & 100)...")
    feat_time_gru = build_feature_matrix(signal_df, process_clean, [GRU_PREFIX], META_TIME)
    feat_time_xgb = build_feature_matrix(signal_df, process_clean, [XGB_PREFIX], META_TIME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    results: dict[str, list[dict]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # GRU baseline (27 features)
    # ─────────────────────────────────────────────────────────────────────────
    log("\n--- GRU baseline (27 features, prefix=80%) ---")
    input_dim_base = len(delta_cols) + len(META_BASE)
    all_cases_base_gru = build_sequences(feat_base_gru, delta_cols, META_BASE, float(GRU_PREFIX))
    gru_base_rows: list[dict] = []
    t0 = time_mod.time()
    for seed in SEEDS:
        df = run_gru_locv(all_cases_base_gru, input_dim_base, device, seed)
        locv_rmse = float(df["rmse"].mean())
        log(f"  seed={seed}  LOCV_RMSE={locv_rmse:.6f}  [{time_mod.time()-t0:.0f}s]")
        for rec in df.to_dict("records"):
            gru_base_rows.append({**rec, "seed": seed, "variant": "gru_base"})
    results["gru_base"] = gru_base_rows

    # ─────────────────────────────────────────────────────────────────────────
    # GRU +time_norm (28 features)
    # ─────────────────────────────────────────────────────────────────────────
    log("\n--- GRU +time_norm (28 features, prefix=80%) ---")
    input_dim_time = len(delta_cols) + len(META_TIME)
    all_cases_time_gru = build_sequences(feat_time_gru, delta_cols, META_TIME, float(GRU_PREFIX))
    gru_time_rows: list[dict] = []
    t0 = time_mod.time()
    for seed in SEEDS:
        df = run_gru_locv(all_cases_time_gru, input_dim_time, device, seed)
        locv_rmse = float(df["rmse"].mean())
        log(f"  seed={seed}  LOCV_RMSE={locv_rmse:.6f}  [{time_mod.time()-t0:.0f}s]")
        for rec in df.to_dict("records"):
            gru_time_rows.append({**rec, "seed": seed, "variant": "gru_time"})
    results["gru_time"] = gru_time_rows

    # ─────────────────────────────────────────────────────────────────────────
    # XGBoost baseline (27 features)
    # ─────────────────────────────────────────────────────────────────────────
    log("\n--- XGBoost baseline (27 features, prefix=100%) ---")
    xgb_feat_cols_base = delta_cols + META_BASE
    xgb_base_rows: list[dict] = []
    t0 = time_mod.time()
    for seed in SEEDS:
        df = run_xgb_locv(feat_base_xgb, xgb_feat_cols_base, XGB_PREFIX, seed)
        locv_rmse = float(df["rmse"].mean())
        log(f"  seed={seed}  LOCV_RMSE={locv_rmse:.6f}  [{time_mod.time()-t0:.0f}s]")
        for rec in df.to_dict("records"):
            xgb_base_rows.append({**rec, "seed": seed, "variant": "xgb_base"})
    results["xgb_base"] = xgb_base_rows

    # ─────────────────────────────────────────────────────────────────────────
    # XGBoost +time_norm (28 features)
    # ─────────────────────────────────────────────────────────────────────────
    log("\n--- XGBoost +time_norm (28 features, prefix=100%) ---")
    xgb_feat_cols_time = delta_cols + META_TIME
    xgb_time_rows: list[dict] = []
    t0 = time_mod.time()
    for seed in SEEDS:
        df = run_xgb_locv(feat_time_xgb, xgb_feat_cols_time, XGB_PREFIX, seed)
        locv_rmse = float(df["rmse"].mean())
        log(f"  seed={seed}  LOCV_RMSE={locv_rmse:.6f}  [{time_mod.time()-t0:.0f}s]")
        for rec in df.to_dict("records"):
            xgb_time_rows.append({**rec, "seed": seed, "variant": "xgb_time"})
    results["xgb_time"] = xgb_time_rows

    # ─────────────────────────────────────────────────────────────────────────
    # Save & report
    # ─────────────────────────────────────────────────────────────────────────
    all_rows = gru_base_rows + gru_time_rows + xgb_base_rows + xgb_time_rows
    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "metrics" / "all_per_case_metrics.csv", index=False)

    # Aggregate comparison
    agg = (
        all_df.groupby(["variant", "seed"])["rmse"].mean()
        .groupby("variant").agg(mean_rmse="mean", std_rmse="std")
        .reset_index()
    )
    agg["std_rmse"] = agg["std_rmse"].fillna(0.0)
    agg.to_csv(out_dir / "analysis" / "aggregate_comparison.csv", index=False)

    # Per-case comparison (3-seed mean per variant)
    per_case = (
        all_df.groupby(["variant", "target_case"])["rmse"].mean()
        .unstack("variant")
        .reindex(index=CASE_SCOPE)
        .reset_index()
    )
    per_case.columns.name = None
    per_case.to_csv(out_dir / "analysis" / "per_case_comparison.csv", index=False)

    # Delta table: +time_norm vs baseline
    def make_delta(variant_base: str, variant_time: str, label: str) -> dict:
        base_rmse = float(agg[agg["variant"] == variant_base]["mean_rmse"].iloc[0])
        time_rmse = float(agg[agg["variant"] == variant_time]["mean_rmse"].iloc[0])
        return {
            "model": label,
            "baseline_rmse": base_rmse,
            "time_norm_rmse": time_rmse,
            "delta": time_rmse - base_rmse,
            "pct_change": (time_rmse - base_rmse) / base_rmse * 100,
        }

    delta_rows = [
        make_delta("gru_base", "gru_time", "Feature-GRU (head32, prefix=80%)"),
        make_delta("xgb_base", "xgb_time", "XGBoost (g_lr0.03_n600_d3_mcw8, prefix=100%)"),
    ]
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "analysis" / "delta_summary.csv", index=False)

    log("\n=== AGGREGATE RESULTS (3-seed mean LOCV RMSE) ===")
    log(f"{'Variant':<45} {'mean_RMSE':>10} {'std_RMSE':>10}")
    log("-" * 68)
    variant_order = ["gru_base", "gru_time", "xgb_base", "xgb_time"]
    labels = {
        "gru_base": "GRU baseline (27 feat, prefix=80%)",
        "gru_time": "GRU +time_norm (28 feat, prefix=80%)",
        "xgb_base": "XGB baseline (27 feat, prefix=100%)",
        "xgb_time": "XGB +time_norm (28 feat, prefix=100%)",
    }
    for v in variant_order:
        row = agg[agg["variant"] == v].iloc[0]
        log(f"{labels[v]:<45} {row['mean_rmse']:>10.6f} {row['std_rmse']:>10.6f}")

    log("\n=== DELTA (time_norm vs baseline) ===")
    for d in delta_rows:
        sign = "+" if d["delta"] > 0 else ""
        direction = "WORSE" if d["delta"] > 0 else "BETTER"
        log(f"{d['model']}")
        log(f"  baseline={d['baseline_rmse']:.6f}  +time_norm={d['time_norm_rmse']:.6f}"
            f"  Δ={sign}{d['delta']:.6f} ({sign}{d['pct_change']:.2f}%)  [{direction}]")

    log("\n=== PER-CASE RMSE (3-seed mean) ===")
    header = f"{'Case':>6} {'GRU_base':>10} {'GRU_time':>10} {'GRU_Δ':>8} {'XGB_base':>10} {'XGB_time':>10} {'XGB_Δ':>8}"
    log(header)
    log("-" * len(header))
    for _, row in per_case.iterrows():
        cid = int(row["target_case"])
        gb = row.get("gru_base", float("nan"))
        gt = row.get("gru_time", float("nan"))
        xb = row.get("xgb_base", float("nan"))
        xt = row.get("xgb_time", float("nan"))
        gd = gt - gb if pd.notna(gt) and pd.notna(gb) else float("nan")
        xd = xt - xb if pd.notna(xt) and pd.notna(xb) else float("nan")
        gd_str = f"{gd:+.4f}" if pd.notna(gd) else "  N/A"
        xd_str = f"{xd:+.4f}" if pd.notna(xd) else "  N/A"
        log(f"{cid:>6} {gb:>10.4f} {gt:>10.4f} {gd_str:>8} {xb:>10.4f} {xt:>10.4f} {xd_str:>8}")

    summary = {
        "experiment": "H8_S1_time_norm_meta",
        "time_norm_method": "case-wise min-max: (time - case_min) / (case_max - case_min)",
        "gru_config": "head32",
        "gru_prefix": GRU_PREFIX,
        "xgb_config": "g_lr0.03_n600_d3_mcw8",
        "xgb_prefix": XGB_PREFIX,
        "seeds": SEEDS,
        "aggregate_results": agg.to_dict("records"),
        "delta_summary": delta_rows,
        "execution_dir": str(out_dir),
    }
    (out_dir / "analysis" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    (out_dir / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()

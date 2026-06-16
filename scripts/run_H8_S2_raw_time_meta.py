#!/usr/bin/env python3
"""H8_S2: Compare baseline vs time_norm vs raw_time as meta feature.

Variants:
  - baseline:   Delta+Meta 27 = 24 delta + [DOC, feed, material]
  - time_norm:  Delta+Meta 28 = 24 delta + [DOC, feed, material, time_norm]
                time_norm = case-wise (time - case_min) / (case_max - case_min)
                → requires knowing final run's time (leakage in prospective setting)
  - raw_time:   Delta+Meta 28 = 24 delta + [DOC, feed, material, time]
                raw cumulative machining time in minutes (always known at prediction time)

Best configs:
  GRU:     head32 (head_hidden=32), prefix=80%, seeds=[0,1,2]
  XGBoost: g_lr0.03_n600_d3_mcw8, prefix=100%, seeds=[0,1,2]

Output:
  experiments/executions/H8/S2/{timestamp}_raw_time_meta/
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
META_BASE      = ["DOC", "feed", "material"]
META_TIME_NORM = ["DOC", "feed", "material", "time_norm"]
META_RAW_TIME  = ["DOC", "feed", "material", "time"]
SEEDS = [0, 1, 2]
GRU_PREFIX = 80
XGB_PREFIX = 100

GRU_CFG = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0, scheduler="cosine",
)
XGB_CFG = dict(
    learning_rate=0.03, n_estimators=600, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.9,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, n_jobs=4,
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
    # Case-wise time_norm (after EXCLUDED_RUNS removal)
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
    THRESH = 1e6
    delta_cols = delta_col_names()
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
            record: dict = {"case_id": case_id, "run": run_id,
                            "signal_prefix_percent": float(pct), "VB": vb}
            for mf in meta_features:
                record[mf] = float(getattr(row, mf, 0.0))
            for col, val in zip(delta_cols, delta_feats):
                record[col] = float(val)
            rows.append(record)
    return pd.DataFrame(rows).sort_values(
        ["signal_prefix_percent", "case_id", "run"]
    ).reset_index(drop=True)


# ─── GRU ─────────────────────────────────────────────────────────────────────
class DeltaMetaGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int,
                 dropout: float, head_hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out).squeeze(-1)


def build_sequences(feature_df: pd.DataFrame, delta_cols: list[str],
                    meta_features: list[str], pct: float) -> dict[int, dict]:
    subset = feature_df[feature_df["signal_prefix_percent"] == float(pct)].copy()
    cases: dict[int, dict] = {}
    for case_id in subset["case_id"].unique():
        c = subset[subset["case_id"] == case_id].sort_values("run").reset_index(drop=True)
        seq = np.concatenate(
            [c[delta_cols].to_numpy(dtype=np.float32),
             c[meta_features].to_numpy(dtype=np.float32)], axis=1
        )
        cases[int(case_id)] = {"seq": seq, "vb": c["VB"].to_numpy(dtype=np.float32),
                                "n_runs": len(c)}
    return cases


def fit_predict_gru(train_cases: dict[int, dict], test_cases: dict[int, dict],
                    input_dim: int, device: torch.device, seed: int) -> dict[int, np.ndarray]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = DeltaMetaGRU(input_dim, GRU_CFG["hidden_size"], GRU_CFG["num_layers"],
                         GRU_CFG["dropout"], GRU_CFG["head_hidden"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_CFG["lr"],
                                 weight_decay=GRU_CFG["weight_decay"])
    epochs = GRU_CFG["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_seqs = [info["seq"] for info in train_cases.values()]
    train_vbs  = [info["vb"]  for info in train_cases.values()]
    all_vb = np.concatenate(train_vbs)
    y_mean, y_std = float(all_vb.mean()), max(float(all_vb.std()), 1e-8)

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
        nn.utils.clip_grad_norm_(model.parameters(), GRU_CFG["grad_clip"])
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        return {
            cid: (model(torch.tensor(info["seq"], dtype=torch.float32).unsqueeze(0).to(device),
                        torch.tensor([info["n_runs"]])).squeeze(0).cpu().numpy() * y_std + y_mean)
            for cid, info in test_cases.items()
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def run_gru_locv(all_case_data: dict[int, dict], input_dim: int,
                 device: torch.device, seed: int) -> pd.DataFrame:
    rows = []
    for tc in CASE_SCOPE:
        if tc not in all_case_data:
            continue
        train = {c: all_case_data[c] for c in CASE_SCOPE if c != tc and c in all_case_data}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = fit_predict_gru(train, {tc: all_case_data[tc]}, input_dim, device, seed)
        y_pred, y_true = preds[tc], all_case_data[tc]["vb"]
        rows.append({"target_case": tc, "n_runs": len(y_true), **compute_metrics(y_true, y_pred)})
    return pd.DataFrame(rows)


def run_xgb_locv(feat_df: pd.DataFrame, feature_cols: list[str], pct: int, seed: int) -> pd.DataFrame:
    subset = feat_df[feat_df["signal_prefix_percent"] == float(pct)].copy()
    rows = []
    for tc in CASE_SCOPE:
        test_df  = subset[subset["case_id"] == tc]
        train_df = subset[subset["case_id"] != tc]
        if test_df.empty or train_df.empty:
            continue
        m = XGBRegressor(**XGB_CFG, random_state=seed, verbosity=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(train_df[feature_cols], train_df["VB"])
            y_pred = m.predict(test_df[feature_cols])
        rows.append({"target_case": tc, "n_runs": len(test_df),
                     **compute_metrics(test_df["VB"].to_numpy(float), y_pred)})
    return pd.DataFrame(rows)


# ─── Runner helper ────────────────────────────────────────────────────────────
def run_gru_variant(feat_df, delta_cols, meta_features, pct, device, variant_name, log):
    input_dim = len(delta_cols) + len(meta_features)
    all_cases = build_sequences(feat_df, delta_cols, meta_features, float(pct))
    rows, t0 = [], time_mod.time()
    for seed in SEEDS:
        df = run_gru_locv(all_cases, input_dim, device, seed)
        locv = float(df["rmse"].mean())
        log(f"  seed={seed}  LOCV_RMSE={locv:.6f}  [{time_mod.time()-t0:.0f}s]")
        for rec in df.to_dict("records"):
            rows.append({**rec, "seed": seed, "variant": variant_name})
    return rows


def run_xgb_variant(feat_df, delta_cols, meta_features, pct, variant_name, log):
    feature_cols = delta_cols + meta_features
    rows, t0 = [], time_mod.time()
    for seed in SEEDS:
        df = run_xgb_locv(feat_df, feature_cols, pct, seed)
        locv = float(df["rmse"].mean())
        log(f"  seed={seed}  LOCV_RMSE={locv:.6f}  [{time_mod.time()-t0:.0f}s]")
        for rec in df.to_dict("records"):
            rows.append({**rec, "seed": seed, "variant": variant_name})
    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "H8" / "S2" / f"{ts}_raw_time_meta"
    for sub in ["metrics", "analysis", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H8_S2: baseline vs time_norm vs raw_time meta feature ===")

    log("Loading & preprocessing...")
    signal_df = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                            usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)

    log(f"raw time range: {proc_clean['time'].min()} – {proc_clean['time'].max()} min")
    log("time_norm: case-wise min-max (requires knowing last run at prediction time)")
    log("raw_time:  actual cumulative minutes (always known at prediction time)")

    delta_cols = delta_col_names()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    log("Building feature matrices...")
    feat_base_g  = build_feature_matrix(signal_df, proc_clean, [GRU_PREFIX], META_BASE)
    feat_tnorm_g = build_feature_matrix(signal_df, proc_clean, [GRU_PREFIX], META_TIME_NORM)
    feat_raw_g   = build_feature_matrix(signal_df, proc_clean, [GRU_PREFIX], META_RAW_TIME)
    feat_base_x  = build_feature_matrix(signal_df, proc_clean, [XGB_PREFIX], META_BASE)
    feat_tnorm_x = build_feature_matrix(signal_df, proc_clean, [XGB_PREFIX], META_TIME_NORM)
    feat_raw_x   = build_feature_matrix(signal_df, proc_clean, [XGB_PREFIX], META_RAW_TIME)

    all_rows: list[dict] = []

    # ── GRU ──────────────────────────────────────────────────────────────────
    for label, feat_df, meta in [
        ("gru_base",      feat_base_g,  META_BASE),
        ("gru_time_norm", feat_tnorm_g, META_TIME_NORM),
        ("gru_raw_time",  feat_raw_g,   META_RAW_TIME),
    ]:
        log(f"\n--- GRU {label} ({len(meta)} meta, prefix={GRU_PREFIX}%) ---")
        all_rows += run_gru_variant(feat_df, delta_cols, meta, GRU_PREFIX, device, label, log)

    # ── XGBoost ──────────────────────────────────────────────────────────────
    for label, feat_df, meta in [
        ("xgb_base",      feat_base_x,  META_BASE),
        ("xgb_time_norm", feat_tnorm_x, META_TIME_NORM),
        ("xgb_raw_time",  feat_raw_x,   META_RAW_TIME),
    ]:
        log(f"\n--- XGBoost {label} ({len(meta)} meta, prefix={XGB_PREFIX}%) ---")
        all_rows += run_xgb_variant(feat_df, delta_cols, meta, XGB_PREFIX, label, log)

    # ── Save ─────────────────────────────────────────────────────────────────
    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "metrics" / "all_per_case_metrics.csv", index=False)

    agg = (all_df.groupby(["variant", "seed"])["rmse"].mean()
           .groupby("variant").agg(mean_rmse="mean", std_rmse="std")
           .reset_index())
    agg["std_rmse"] = agg["std_rmse"].fillna(0.0)
    agg.to_csv(out_dir / "analysis" / "aggregate_comparison.csv", index=False)

    per_case = (all_df.groupby(["variant", "target_case"])["rmse"].mean()
                .unstack("variant").reindex(index=CASE_SCOPE).reset_index())
    per_case.columns.name = None
    per_case.to_csv(out_dir / "analysis" / "per_case_comparison.csv", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    VARIANT_ORDER = [
        "gru_base", "gru_time_norm", "gru_raw_time",
        "xgb_base", "xgb_time_norm", "xgb_raw_time",
    ]
    LABELS = {
        "gru_base":      "GRU   baseline   (27 feat)",
        "gru_time_norm": "GRU   +time_norm (28 feat) [case-wise, leakage]",
        "gru_raw_time":  "GRU   +raw_time  (28 feat) [minutes, no leakage]",
        "xgb_base":      "XGB   baseline   (27 feat)",
        "xgb_time_norm": "XGB   +time_norm (28 feat) [case-wise, leakage]",
        "xgb_raw_time":  "XGB   +raw_time  (28 feat) [minutes, no leakage]",
    }

    log("\n=== AGGREGATE RESULTS (3-seed mean LOCV RMSE) ===")
    log(f"{'Variant':<52} {'mean_RMSE':>10} {'std_RMSE':>10}")
    log("-" * 74)
    for v in VARIANT_ORDER:
        row = agg[agg["variant"] == v]
        if row.empty:
            continue
        row = row.iloc[0]
        log(f"{LABELS[v]:<52} {row['mean_rmse']:>10.6f} {row['std_rmse']:>10.6f}")

    def get_rmse(v):
        r = agg[agg["variant"] == v]
        return float(r["mean_rmse"].iloc[0]) if not r.empty else float("nan")

    log("\n=== DELTA vs baseline ===")
    for model in ["gru", "xgb"]:
        base = get_rmse(f"{model}_base")
        tnorm = get_rmse(f"{model}_time_norm")
        raw   = get_rmse(f"{model}_raw_time")
        log(f"[{model.upper()}] baseline={base:.6f}")
        for name, val in [("time_norm", tnorm), ("raw_time", raw)]:
            d = val - base
            sign = "+" if d > 0 else ""
            tag = "WORSE" if d > 0 else "BETTER"
            log(f"  +{name:<10}: {val:.6f}  Δ={sign}{d:.6f} ({sign}{d/base*100:.2f}%)  [{tag}]")

    log("\n=== PER-CASE RMSE (3-seed mean) ===")
    cols = ["gru_base", "gru_time_norm", "gru_raw_time",
            "xgb_base", "xgb_time_norm", "xgb_raw_time"]
    header = f"{'Case':>5} {'G_base':>8} {'G_tnorm':>8} {'G_raw':>8}  {'X_base':>8} {'X_tnorm':>8} {'X_raw':>8}"
    log(header)
    log("-" * len(header))
    for _, row in per_case.iterrows():
        cid = int(row["target_case"])
        vals = [row.get(c, float("nan")) for c in cols]
        fmt = " ".join(f"{v:8.4f}" if pd.notna(v) else "     N/A" for v in vals)
        log(f"{cid:>5} {fmt}")

    summary = {
        "experiment": "H8_S2_raw_time_meta",
        "variants": {
            "baseline":   "Delta+Meta 27, no time feature",
            "time_norm":  "Delta+Meta 28, case-wise (time-min)/(max-min), leakage in prospective",
            "raw_time":   "Delta+Meta 28, raw cumulative minutes, no leakage",
        },
        "gru_config": "head32, prefix=80%",
        "xgb_config": "g_lr0.03_n600_d3_mcw8, prefix=100%",
        "seeds": SEEDS,
        "aggregate": agg.to_dict("records"),
        "execution_dir": str(out_dir),
    }
    (out_dir / "analysis" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    (out_dir / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

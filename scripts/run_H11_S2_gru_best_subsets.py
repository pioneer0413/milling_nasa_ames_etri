#!/usr/bin/env python3
"""H11_S2: Feature-GRU on H11_S1 best sensor subsets.

Variants (GRU best config: head32, prefix=80%, seeds=[0,1,2]):
  - baseline:           all 6 sensors → input_dim = 24+3 = 27
  - xgb_best (AC+vS):  smcAC + vib_spindle → input_dim = 8+3 = 11
  - rf_best  (AC+vT+vS+aS): smcAC + vib_table + vib_spindle + AE_spindle → input_dim = 16+3 = 19

All variants use:
  delta features: (mean, rms, std, peak) × n_sensors relative to first non-excluded run
  meta features : DOC, feed, material

H11_S1 baselines for reference:
  XGB baseline (all 6, prefix=100%): RMSE=0.1040
  RF  baseline (all 6, prefix=100%): RMSE=0.1128
  GRU baseline (all 6, prefix=80%):  RMSE≈0.0908

Output:
  experiments/executions/H11/S2/{timestamp}_gru_best_subsets/
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
ALL_SENSORS   = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
EXCLUDED_RUNS = {(2, 1), (12, 1)}
NON_OBSERVED_RUNS = {
    (1,2),(1,3),(1,5),(1,16),(2,6),(8,4),(11,7),(11,14),(11,17),
    (12,4),(12,10),(13,1),(13,2),(14,1),(14,5),(15,1),(16,1),(16,2),(16,4),
}
META_FEATURES = ["DOC", "feed", "material"]
SEEDS         = [0, 1, 2]
GRU_PREFIX    = 80
THRESH        = 1e6

VARIANTS = [
    ("baseline",   ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]),
    ("AC+vS",      ["smcAC", "vib_spindle"]),
    ("AC+vT+vS+aS",["smcAC", "vib_table", "vib_spindle", "AE_spindle"]),
]

GRU_CFG = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0, scheduler="cosine",
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
    return df[~mask].reset_index(drop=True)


def build_feature_matrix(
    signal_df: pd.DataFrame,
    process_df: pd.DataFrame,
    sensors: list[str],
) -> pd.DataFrame:
    delta_cols = [f"{s}__delta_{f}" for s in sensors for f in ["mean", "rms", "std", "peak"]]
    cache: dict[tuple[int, int], np.ndarray] = {}

    for row in process_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        sig_row = signal_df[(signal_df["case"] == case_id) & (signal_df["run"] == run_id)]
        if sig_row.empty:
            continue
        sig_row = sig_row.iloc[0]
        arrays = {s: parse_signal(sig_row[s]) for s in sensors}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        base_len = min(len(a) for a in arrays.values())
        end = max(1, int(np.ceil(base_len * GRU_PREFIX / 100.0)))
        feats = np.concatenate([extract_features(arrays[s][:end]) for s in sensors])
        cache[(case_id, run_id)] = feats

    first_run: dict[int, int] = {}
    for case_id in process_df["case"].unique():
        runs = sorted(r for (c, r) in cache if c == case_id and (c, r) not in EXCLUDED_RUNS)
        if runs:
            first_run[int(case_id)] = runs[0]

    rows: list[dict] = []
    for row in process_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        if (case_id, run_id) in EXCLUDED_RUNS:
            continue
        ref_run = first_run.get(case_id, run_id)
        raw_feats = cache.get((case_id, run_id))
        ref_feats = cache.get((case_id, ref_run))
        if raw_feats is None or ref_feats is None:
            continue
        delta = raw_feats - ref_feats
        vb = float(row.VB) if not pd.isna(row.VB) else 0.0
        record: dict = {"case_id": case_id, "run": run_id, "VB": vb}
        for mf in META_FEATURES:
            record[mf] = float(getattr(row, mf, 0.0))
        for col, val in zip(delta_cols, delta):
            record[col] = float(val) if np.isfinite(val) else 0.0
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["case_id", "run"]).reset_index(drop=True)


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


def build_sequences(feat_df: pd.DataFrame, delta_cols: list[str]) -> dict[int, dict]:
    cases: dict[int, dict] = {}
    for case_id in feat_df["case_id"].unique():
        c = feat_df[feat_df["case_id"] == case_id].sort_values("run").reset_index(drop=True)
        seq = np.concatenate(
            [c[delta_cols].to_numpy(dtype=np.float32),
             c[META_FEATURES].to_numpy(dtype=np.float32)], axis=1
        )
        cases[int(case_id)] = {"seq": seq, "vb": c["VB"].to_numpy(dtype=np.float32),
                                "runs": c["run"].to_numpy(dtype=int),
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
            cid: (model(
                torch.tensor(info["seq"], dtype=torch.float32).unsqueeze(0).to(device),
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
        runs = all_case_data[tc]["runs"]
        obs  = np.array([(tc, int(r)) not in NON_OBSERVED_RUNS for r in runs])
        y_pred = preds[tc][obs]
        y_true = all_case_data[tc]["vb"][obs]
        rows.append({"target_case": tc, "n_runs": int(obs.sum()), **compute_metrics(y_true, y_pred)})
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "H11" / "S2" / f"{ts}_gru_best_subsets"
    for sub in ["metrics", "analysis", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H11_S2: Feature-GRU on Best Sensor Subsets ===")
    log(f"Prefix = {GRU_PREFIX}%, Seeds = {SEEDS}, LOCV = {len(CASE_SCOPE)} cases")
    for name, sensors in VARIANTS:
        log(f"  {name}: {sensors}  (input_dim={len(sensors)*4+3})")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/processed/mill_signal_data.csv",
                             usecols=["case", "run"] + ALL_SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/processed/mill_process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    all_rows: list[dict] = []
    t_total = time_mod.time()

    for variant_name, sensors in VARIANTS:
        delta_cols = [f"{s}__delta_{f}" for s in sensors for f in ["mean", "rms", "std", "peak"]]
        input_dim  = len(delta_cols) + len(META_FEATURES)
        n_feat     = input_dim

        log(f"\nBuilding feature matrix for [{variant_name}]  ({n_feat} features)...")
        feat_df = build_feature_matrix(signal_df, proc_clean, sensors)
        feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat_df.fillna(0.0, inplace=True)
        all_cases = build_sequences(feat_df, delta_cols)

        log(f"\n--- GRU [{variant_name}] (prefix={GRU_PREFIX}%, input_dim={input_dim}) ---")
        t0 = time_mod.time()
        seed_rmses = []
        for seed in SEEDS:
            locv_df = run_gru_locv(all_cases, input_dim, device, seed)
            locv_rmse = float(locv_df["rmse"].mean())
            seed_rmses.append(locv_rmse)
            log(f"  seed={seed}  LOCV_RMSE={locv_rmse:.6f}  [{time_mod.time()-t0:.0f}s]")
            for rec in locv_df.to_dict("records"):
                all_rows.append({**rec, "seed": seed, "variant": variant_name})
        log(f"  3-seed mean = {np.mean(seed_rmses):.6f}  std = {np.std(seed_rmses):.6f}")

    log(f"\nTotal time: {time_mod.time()-t_total:.1f}s")

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
    VARIANT_ORDER = [v for v, _ in VARIANTS]
    LABELS = {
        "baseline":    "GRU  baseline     (all 6,        27 feat)",
        "AC+vS":       "GRU  AC+vS        (XGB best,     11 feat)",
        "AC+vT+vS+aS": "GRU  AC+vT+vS+aS (RF best,      19 feat)",
    }

    def get_rmse(v: str) -> float:
        r = agg[agg["variant"] == v]
        return float(r["mean_rmse"].iloc[0]) if not r.empty else float("nan")

    baseline_rmse = get_rmse("baseline")

    log("\n=== AGGREGATE RESULTS (3-seed mean LOCV RMSE) ===")
    log(f"{'Variant':<48} {'mean_RMSE':>10} {'std_RMSE':>10}")
    log("-" * 70)
    for v in VARIANT_ORDER:
        row = agg[agg["variant"] == v]
        if row.empty:
            continue
        r = row.iloc[0]
        log(f"{LABELS[v]:<48} {r['mean_rmse']:>10.6f} {r['std_rmse']:>10.6f}")

    log(f"\n=== DELTA vs GRU baseline (all 6 sensors, RMSE={baseline_rmse:.6f}) ===")
    for v in VARIANT_ORDER[1:]:
        rmse = get_rmse(v)
        d = rmse - baseline_rmse
        sign = "+" if d > 0 else ""
        tag = "WORSE" if d > 0 else "BETTER"
        log(f"  {LABELS[v]}: {rmse:.6f}  Δ={sign}{d:.6f} ({sign}{d/baseline_rmse*100:.2f}%)  [{tag}]")

    log("\n=== PER-CASE RMSE (3-seed mean) ===")
    hdr = f"{'Case':>5} {'G_base':>8} {'G_AC+vS':>9} {'Δ':>8}   {'G_AC+vT+vS+aS':>14} {'Δ':>8}"
    log(hdr)
    log("-" * len(hdr))
    for _, row in per_case.iterrows():
        cid  = int(row["target_case"])
        base = row.get("baseline",    float("nan"))
        acvs = row.get("AC+vS",       float("nan"))
        acrf = row.get("AC+vT+vS+aS", float("nan"))
        d1   = acvs - base if pd.notna(acvs) and pd.notna(base) else float("nan")
        d2   = acrf - base if pd.notna(acrf) and pd.notna(base) else float("nan")
        log(f"{cid:>5} {base:8.4f} {acvs:9.4f} {d1:+8.4f}   {acrf:14.4f} {d2:+8.4f}")

    summary = {
        "experiment": "H11_S2_gru_best_subsets",
        "variants": {v: s for v, s in VARIANTS},
        "gru_config":  f"head32, prefix={GRU_PREFIX}%",
        "seeds":       SEEDS,
        "aggregate":   agg.to_dict("records"),
        "execution_dir": str(out_dir),
    }
    (out_dir / "analysis" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

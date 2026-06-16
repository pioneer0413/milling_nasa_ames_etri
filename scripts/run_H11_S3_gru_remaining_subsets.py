#!/usr/bin/env python3
"""H11_S3: Feature-GRU on remaining 60 sensor subsets (63 - 3 already tested in H11_S2).

H11_S2 already tested (excluded here):
  mask  9 = AC+vS         (smcAC + vib_spindle)           → RMSE=0.0813
  mask 45 = AC+vT+vS+aS   (smcAC + vib_table + vib_spindle + AE_spindle) → RMSE=0.0835
  mask 63 = baseline       (all 6 sensors)                  → RMSE=0.0908

This script tests all remaining 60 masks.

Sensor bit mapping:
  bit0 = smcAC (AC),  bit1 = smcDC (DC),  bit2 = vib_table (vT),
  bit3 = vib_spindle (vS),  bit4 = AE_table (aT),  bit5 = AE_spindle (aS)

GRU config (best, head32): hidden=256, layers=3, dropout=0.1,
  lr=1e-3, wd=1e-4, epochs=200, cosine scheduler, prefix=80%, seeds=[0,1,2], LOCV

Output:
  experiments/executions/H11/S3/{timestamp}_gru_remaining_subsets/
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
from sklearn.metrics import mean_squared_error

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
PREFIX        = 80
THRESH        = 1e6
N_SENSORS     = len(SENSORS)

# H11_S2 already tested these 3 masks
EXCLUDE_MASKS = {9, 45, 63}
REMAINING_MASKS = [m for m in range(1, 2 ** N_SENSORS) if m not in EXCLUDE_MASKS]

GRU_CFG = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0,
)


def mask_to_sensors(mask: int) -> list[str]:
    return [SENSORS[i] for i in range(N_SENSORS) if (mask >> i) & 1]


def mask_label(mask: int) -> str:
    return "+".join(SENSOR_ABBR[s] for s in mask_to_sensors(mask))


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
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"] == min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside")
    )
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    mask = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~mask].reset_index(drop=True)


def build_raw_cache(
    signal_df: pd.DataFrame, proc_clean: pd.DataFrame
) -> tuple[dict[tuple[int, int], np.ndarray], dict[int, int]]:
    """Build 24-dim (6 sensors × 4 stats) feature cache at prefix=80%."""
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
        end = max(1, int(np.ceil(base_len * PREFIX / 100.0)))
        feats = np.concatenate([extract_features(arrays[s][:end]) for s in SENSORS])
        cache[(case_id, run_id)] = feats

    first_run: dict[int, int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c, r) in cache if c == case_id)
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


def build_sequences_for_mask(
    raw_cache: dict[tuple[int, int], np.ndarray],
    first_run: dict[int, int],
    proc_clean: pd.DataFrame,
    mask: int,
) -> dict[int, dict]:
    """Build per-case GRU sequences by slicing raw_cache for active sensors."""
    # Each sensor occupies 4 consecutive slots (mean, rms, std, peak)
    sensor_indices: list[int] = []
    for i in range(N_SENSORS):
        if (mask >> i) & 1:
            sensor_indices.extend([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])

    # Group rows by case
    case_rows: dict[int, list[dict]] = {c: [] for c in CASE_SCOPE}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id)
        if key not in raw_cache:
            continue
        ref_run = first_run.get(case_id, run_id)
        ref_vec = raw_cache.get((case_id, ref_run), np.zeros(N_SENSORS * 4))
        delta   = (raw_cache[key] - ref_vec)[sensor_indices]
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        meta = np.array([float(getattr(row, mf, 0.0)) for mf in META_FEATURES],
                        dtype=np.float64)
        feat = np.concatenate([delta, meta]).astype(np.float32)
        vb   = float(row.VB) if not pd.isna(row.VB) else 0.0
        case_rows[case_id].append({"feat": feat, "vb": vb, "run": run_id})

    cases: dict[int, dict] = {}
    for case_id, rows in case_rows.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["run"])
        seq = np.stack([r["feat"] for r in rows_sorted])
        vb   = np.array([r["vb"]  for r in rows_sorted], dtype=np.float32)
        runs = np.array([r["run"] for r in rows_sorted], dtype=int)
        cases[case_id] = {"seq": seq, "vb": vb, "runs": runs, "n_runs": len(rows_sorted)}
    return cases


# ─── GRU ─────────────────────────────────────────────────────────────────────
class DeltaMetaGRU(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        cfg = GRU_CFG
        self.gru = nn.GRU(input_dim, cfg["hidden_size"], cfg["num_layers"],
                          batch_first=True,
                          dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0)
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
    test_cases:  dict[int, dict],
    input_dim:   int,
    device:      torch.device,
    seed:        int,
) -> dict[int, np.ndarray]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = DeltaMetaGRU(input_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=GRU_CFG["lr"], weight_decay=GRU_CFG["weight_decay"])
    epochs = GRU_CFG["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_seqs = [info["seq"] for info in train_cases.values()]
    train_vbs  = [info["vb"]  for info in train_cases.values()]
    all_vb = np.concatenate(train_vbs)
    y_mean = float(all_vb.mean())
    y_std  = max(float(all_vb.std()), 1e-8)

    tensors = [torch.tensor(s, dtype=torch.float32) for s in train_seqs]
    lengths = torch.tensor([len(s) for s in train_seqs])
    x_train = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0).to(device)
    tgt = [torch.tensor((t - y_mean) / y_std, dtype=torch.float32) for t in train_vbs]
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


def run_locv_seed(all_cases: dict[int, dict], input_dim: int,
                  device: torch.device, seed: int) -> float:
    rmses = []
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
        rmses.append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
    return float(np.mean(rmses)) if rmses else float("nan")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "H11" / "S3" / f"{ts}_gru_remaining_subsets"
    for sub in ["metrics", "analysis", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H11_S3: Feature-GRU Remaining 60 Sensor Subsets ===")
    log(f"Total masks: 63 - 3 already tested (H11_S2) = {len(REMAINING_MASKS)} remaining")
    log(f"Excluded: mask9=AC+vS, mask45=AC+vT+vS+aS, mask63=baseline")
    log(f"Prefix={PREFIX}%, Seeds={SEEDS}, LOCV={len(CASE_SCOPE)} cases")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    log(f"Building raw feature cache (prefix={PREFIX}%)...")
    raw_cache, first_run = build_raw_cache(signal_df, proc_clean)
    log(f"Cached {len(raw_cache)} runs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    results: list[dict] = []
    t_start = time_mod.time()

    for idx, mask in enumerate(REMAINING_MASKS):
        sensors   = mask_to_sensors(mask)
        label     = mask_label(mask)
        input_dim = len(sensors) * 4 + len(META_FEATURES)
        all_cases = build_sequences_for_mask(raw_cache, first_run, proc_clean, mask)

        seed_rmses = []
        for seed in SEEDS:
            rmse = run_locv_seed(all_cases, input_dim, device, seed)
            seed_rmses.append(rmse)

        mean_rmse = float(np.mean(seed_rmses))
        std_rmse  = float(np.std(seed_rmses))
        results.append({
            "mask": mask, "label": label, "n_sensors": len(sensors),
            "n_feat": input_dim, "sensors": sensors,
            "gru_mean_rmse": mean_rmse, "gru_std_rmse": std_rmse,
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time_mod.time() - t_start
            eta = elapsed / (idx + 1) * (len(REMAINING_MASKS) - idx - 1)
            log(f"  [{idx+1:2d}/{len(REMAINING_MASKS)}]  {label:<32}  "
                f"GRU={mean_rmse:.4f}  [{elapsed:.0f}s, ETA~{eta:.0f}s]")

    log(f"\nTotal sweep time: {time_mod.time()-t_start:.1f}s")

    # ── Combine with H11_S2 known results ─────────────────────────────────────
    known = [
        {"mask": 63, "label": "AC+DC+vT+vS+aT+aS", "n_sensors": 6, "n_feat": 27,
         "sensors": SENSORS, "gru_mean_rmse": 0.090773, "gru_std_rmse": 0.001895},
        {"mask": 9,  "label": "AC+vS",       "n_sensors": 2, "n_feat": 11,
         "sensors": ["smcAC","vib_spindle"],  "gru_mean_rmse": 0.081307, "gru_std_rmse": 0.001209},
        {"mask": 45, "label": "AC+vT+vS+aS", "n_sensors": 4, "n_feat": 19,
         "sensors": ["smcAC","vib_table","vib_spindle","AE_spindle"],
         "gru_mean_rmse": 0.083516, "gru_std_rmse": 0.002813},
    ]
    all_results = results + known
    res_df      = pd.DataFrame(all_results)
    res_df_sorted = res_df.sort_values("gru_mean_rmse").reset_index(drop=True)
    res_df_sorted.to_csv(out_dir / "metrics" / "all_subset_results.csv", index=False)

    # Per-size stats (all 63)
    size_agg = res_df.groupby("n_sensors").agg(
        n_subsets=("mask", "count"),
        gru_min=("gru_mean_rmse", "min"),
        gru_mean=("gru_mean_rmse", "mean"),
    ).reset_index()
    size_agg.to_csv(out_dir / "analysis" / "per_size_stats.csv", index=False)

    # Sensor frequency in top-10
    top10 = res_df_sorted.head(10)
    freq  = {s: 0 for s in SENSORS}
    for _, row in top10.iterrows():
        for s in row["sensors"]:
            freq[s] += 1
    freq_sorted = sorted(freq.items(), key=lambda x: -x[1])

    # ── Report ────────────────────────────────────────────────────────────────
    log("\n=== TOP-10 SUBSETS (GRU, 3-seed mean LOCV RMSE, all 63 subsets) ===")
    log(f"{'Rank':<5} {'Subset':<32} {'n':>3}  {'GRU_RMSE':>10}")
    log("-" * 56)
    for rank, (_, row) in enumerate(res_df_sorted.head(10).iterrows(), 1):
        log(f"{rank:<5} {row['label']:<32} {row['n_sensors']:>3}  {row['gru_mean_rmse']:>10.6f}")

    log(f"\n=== SINGLE-SENSOR PERFORMANCE ===")
    single = res_df[res_df["n_sensors"] == 1].sort_values("gru_mean_rmse")
    log(f"{'Sensor':<12} {'GRU_RMSE':>10}")
    log("-" * 24)
    for _, row in single.iterrows():
        log(f"{row['label']:<12} {row['gru_mean_rmse']:>10.6f}")

    log(f"\n=== PER-SIZE MIN/MEAN RMSE (all 63 subsets) ===")
    log(f"{'n_sensors':>10} {'n_subsets':>10} {'GRU_min':>10} {'GRU_mean':>10}")
    log("-" * 44)
    for _, row in size_agg.iterrows():
        log(f"{int(row['n_sensors']):>10} {int(row['n_subsets']):>10} "
            f"{row['gru_min']:>10.6f} {row['gru_mean']:>10.6f}")

    baseline_rmse = 0.090773
    best_row = res_df_sorted.iloc[0]
    log(f"\n=== BEST SUBSET vs BASELINE ===")
    log(f"  GRU best: {best_row['label']}  RMSE={best_row['gru_mean_rmse']:.6f}"
        f"  vs baseline {baseline_rmse:.6f}"
        f"  Δ={best_row['gru_mean_rmse']-baseline_rmse:+.6f}"
        f"  ({(best_row['gru_mean_rmse']-baseline_rmse)/baseline_rmse*100:+.2f}%)")

    log(f"\n=== SENSOR FREQUENCY IN TOP-10 (GRU) ===")
    for s, cnt in freq_sorted:
        log(f"  {SENSOR_ABBR[s]}: {cnt}/10")

    summary = {
        "experiment": "H11_S3_gru_remaining_subsets",
        "n_masks_tested": len(REMAINING_MASKS),
        "excluded_masks": list(EXCLUDE_MASKS),
        "gru_config": f"head32, prefix={PREFIX}%",
        "seeds": SEEDS,
        "top10": res_df_sorted.head(10)[["label", "n_sensors", "gru_mean_rmse"]].to_dict("records"),
        "execution_dir": str(out_dir),
    }
    (out_dir / "analysis" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

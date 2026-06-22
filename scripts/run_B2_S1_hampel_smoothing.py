#!/usr/bin/env python3
"""B2_S1 (variant): All-sensors FeatGRU/FeatLSTM with Hampel smoothing

vib_table, vib_spindle, AE_table, AE_spindle 4개 센서에 Hampel filter(median 기반
outlier 스무딩)를 적용한 뒤, all-sensors(mask=63) Delta+Meta Full-segment
FeatGRU/FeatLSTM을 재실행해 [B2_S1]/[I1_S2] 레퍼런스와 비교한다.

  smcAC, smcDC        : 필터링 안 함 (그대로)
  vib_table/spindle   : Hampel(window=11, n_sigmas=3) 적용
  AE_table/spindle    : Hampel(window=11, n_sigmas=3) 적용

Reference (필터링 전, all-sensors, Delta+Meta, Full):
  FeatGRU  = 0.114839
  FeatLSTM = 0.117919

Output: experiments/executions/B2/S1/{timestamp}_hampel_smoothing/
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
from scipy.ndimage import median_filter
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS       = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
HAMPEL_SENSORS = {"vib_table", "vib_spindle", "AE_table", "AE_spindle"}
SENSOR_ABBR   = {"smcAC": "AC", "smcDC": "DC", "vib_table": "vT",
                 "vib_spindle": "vS", "AE_table": "aT", "AE_spindle": "aS"}
EXCLUDED_RUNS = {(2, 1), (12, 1)}
NON_OBSERVED_RUNS = {
    (1,2),(1,3),(1,5),(1,16),(2,6),(8,4),(11,7),(11,14),(11,17),
    (12,4),(12,10),(13,1),(13,2),(14,1),(14,5),(15,1),(16,1),(16,2),(16,4),
}
META_FEATURES = ["DOC", "feed", "material"]
SEEDS         = [0, 1, 2, 3, 4]
THRESH        = 1e6
N_SENSORS     = len(SENSORS)
MASK          = 63   # all sensors
MODELS        = ["FeatGRU", "FeatLSTM"]
HAMPEL_WINDOW   = 11
HAMPEL_NSIGMAS  = 3
REF_NO_HAMPEL = {"FeatGRU": 0.114839, "FeatLSTM": 0.117919}   # I1_S2 / B2_S1 all-sensors Full

RNN_CFG = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0,
)


# ─── Hampel filter ─────────────────────────────────────────────────────────────
def hampel_filter(x: np.ndarray, window: int = HAMPEL_WINDOW, n_sigmas: int = HAMPEL_NSIGMAS) -> np.ndarray:
    """Median-based outlier smoothing. Replaces points deviating from the local
    median by more than n_sigmas*1.4826*MAD with the local median."""
    if x.size < window:
        return x
    k = 1.4826  # MAD -> std scale factor for Gaussian-consistent estimate
    med = median_filter(x, size=window, mode="reflect")
    mad = median_filter(np.abs(x - med), size=window, mode="reflect")
    threshold = n_sigmas * k * mad
    outlier = np.abs(x - med) > threshold
    out = x.copy()
    out[outlier] = med[outlier]
    return out


# ─── Utils ────────────────────────────────────────────────────────────────────
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


def build_raw_cache(signal_df: pd.DataFrame, proc_clean: pd.DataFrame, apply_hampel: bool):
    sensor_indices = mask_sensor_indices(MASK)
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
        if apply_hampel:
            for s in HAMPEL_SENSORS:
                arrays[s] = hampel_filter(arrays[s])
        base_len = min(len(a) for a in arrays.values())
        full = np.concatenate([extract_features(arrays[s][:base_len]) for s in SENSORS])
        cache[(case_id, run_id)] = full[sensor_indices]

    first_run: dict[int,int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c, r) in cache if c == case_id)
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


def build_delta_meta(raw_cache, first_run, proc_clean):
    n_sensor_feats = bin(MASK).count("1") * 4
    case_rows: dict[int, list[dict]] = {c: [] for c in CASE_SCOPE}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id)
        if key not in raw_cache:
            continue
        raw_vec = raw_cache[key]
        ref_key = (case_id, first_run.get(case_id, run_id))
        ref_vec = raw_cache.get(ref_key, np.zeros(n_sensor_feats))
        delta = raw_vec - ref_vec
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        meta = np.array([float(getattr(row, mf, 0.0)) for mf in META_FEATURES])
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


# ─── RNN model ────────────────────────────────────────────────────────────────
class DeltaMetaRNN(nn.Module):
    def __init__(self, input_dim: int, cell_type: str) -> None:
        super().__init__()
        cfg = RNN_CFG
        rnn_cls = nn.GRU if cell_type == "FeatGRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_dim, cfg["hidden_size"], cfg["num_layers"],
            batch_first=True, dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg["hidden_size"], cfg["head_hidden"]), nn.ReLU(),
            nn.Linear(cfg["head_hidden"], 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out).squeeze(-1)


def fit_predict_rnn(train_cases, test_cases, input_dim, cell_type, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = DeltaMetaRNN(input_dim, cell_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=RNN_CFG["lr"], weight_decay=RNN_CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RNN_CFG["epochs"])

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
    for _ in range(RNN_CFG["epochs"]):
        optimizer.zero_grad()
        pred = model(x_train, lengths)
        loss = ((pred[mask_tr] - y_train[mask_tr]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), RNN_CFG["grad_clip"])
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


def run_locv_seed(all_cases, input_dim, cell_type, device, seed):
    rmses: list[float] = []
    for tc in CASE_SCOPE:
        if tc not in all_cases:
            continue
        train = {c: all_cases[c] for c in CASE_SCOPE if c != tc and c in all_cases}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = fit_predict_rnn(train, {tc: all_cases[tc]}, input_dim, cell_type, device, seed)
        runs = all_cases[tc]["runs"]
        obs  = np.array([(tc, int(r)) not in NON_OBSERVED_RUNS for r in runs])
        y_pred = preds[tc][obs]
        y_true = all_cases[tc]["vb"][obs]
        if len(y_true) == 0:
            continue
        rmses.append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
    return float(np.mean(rmses)) if rmses else float("nan")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B2" / "S1" / f"{ts}_hampel_smoothing"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== B2_S1 variant: All-sensors FeatGRU/FeatLSTM with Hampel smoothing ===")
    log(f"Hampel sensors: {sorted(HAMPEL_SENSORS)} (window={HAMPEL_WINDOW}, n_sigmas={HAMPEL_NSIGMAS})")
    log(f"Unfiltered: smcAC, smcDC")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    input_dim = bin(MASK).count("1") * 4 + len(META_FEATURES)

    log("\nBuilding raw cache WITH Hampel smoothing...")
    raw_cache_h, first_run_h = build_raw_cache(signal_df, proc_clean, apply_hampel=True)
    all_cases_h = build_delta_meta(raw_cache_h, first_run_h, proc_clean)

    results: dict[str, dict] = {}
    t_total = time_mod.time()
    for model_type in MODELS:
        t0 = time_mod.time()
        seed_rmses = [run_locv_seed(all_cases_h, input_dim, model_type, device, s) for s in SEEDS]
        mean_, std_ = float(np.mean(seed_rmses)), float(np.std(seed_rmses))
        ref = REF_NO_HAMPEL[model_type]
        log(f"  {model_type:<10} (Hampel)  mean={mean_:.6f}  std={std_:.6f}  "
            f"[{time_mod.time()-t0:.0f}s]  vs no-Hampel ref={ref:.6f}  Δ={mean_-ref:+.6f}")
        results[model_type] = {"mean": mean_, "std": std_, "seed_rmses": seed_rmses, "ref_no_hampel": ref}

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    log(f"{'Model':<10} {'No-Hampel(ref)':>15} {'Hampel':>12} {'Δ':>12}")
    log("-" * 52)
    for m in MODELS:
        r = results[m]
        delta = r["mean"] - r["ref_no_hampel"]
        tag = "개선" if delta < 0 else "악화"
        log(f"{m:<10} {r['ref_no_hampel']:>15.6f} {r['mean']:>12.6f} {delta:>+12.6f} ({tag})")

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame([
        {"model": m, "mean": r["mean"], "std": r["std"], "ref_no_hampel": r["ref_no_hampel"],
         "delta": r["mean"] - r["ref_no_hampel"]}
        for m, r in results.items()
    ]).to_csv(out_dir / "metrics" / "hampel_vs_nohampel.csv", index=False)

    seed_rows = []
    for m, r in results.items():
        for i, v in enumerate(r["seed_rmses"]):
            seed_rows.append({"model": m, "seed": SEEDS[i], "rmse": v})
    pd.DataFrame(seed_rows).to_csv(out_dir / "metrics" / "seed_detail.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    names = MODELS + [f"{m}(no-Hampel ref)" for m in MODELS]
    vals = [results[m]["mean"] for m in MODELS] + [REF_NO_HAMPEL[m] for m in MODELS]
    colors = ["#d65f5f", "#d65f5f", "#4878d0", "#4878d0"]
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Observed-VB RMSE")
    ax.set_title("Hampel smoothing (vT/vS/aT/aS) vs no-Hampel — All sensors, Full")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_dir / "figures" / "hampel_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "experiment": "B2_S1_hampel_smoothing",
        "hampel_sensors": sorted(HAMPEL_SENSORS), "window": HAMPEL_WINDOW, "n_sigmas": HAMPEL_NSIGMAS,
        "mask": MASK, "sensor_subset": mask_label(MASK),
        "results": results,
        "execution_dir": str(out_dir),
        "total_elapsed_s": time_mod.time() - t_total,
    }
    (out_dir / "logs" / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

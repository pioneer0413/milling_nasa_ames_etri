#!/usr/bin/env python3
"""B2_S1 (variant): All-sensors FeatGRU/FeatLSTM with sensor-specific 2-stat features

기존엔 6개 센서 전부에 동일한 4-stat(mean/rms/std/max)을 적용했으나, 센서 물리적
특성에 맞춰 센서 유형별로 가장 적합한 2-stat만 사용:

  smcAC, smcDC (전류, 부하 proxy)        : mean, std
  vib_table, vib_spindle (진동, 과도 피크) : rms, crest_factor(=max|x|/rms)
  AE_table, AE_spindle (음향방출, 충격성)  : rms, kurtosis

Raw dim = 6 sensors x 2 = 12, Delta+Meta = 12 + 3(meta) = 15-dim.

비교 대상:
  all-sensors 4-stat(24-dim)      : FeatGRU=0.114839, FeatLSTM=0.117919
  AC+vT+vS 4-stat(15-dim, 동일차원): FeatGRU=0.095122, FeatLSTM=0.092217
  all-sensors Hampel-smoothed      : FeatGRU=0.140917, FeatLSTM=0.141668 (negative result)

Output: experiments/executions/B2/S1/{timestamp}_sensor_specific_features/
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
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS       = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
CURRENT_SENSORS = {"smcAC", "smcDC"}
VIB_SENSORS     = {"vib_table", "vib_spindle"}
AE_SENSORS      = {"AE_table", "AE_spindle"}
EXCLUDED_RUNS = {(2, 1), (12, 1)}
NON_OBSERVED_RUNS = {
    (1,2),(1,3),(1,5),(1,16),(2,6),(8,4),(11,7),(11,14),(11,17),
    (12,4),(12,10),(13,1),(13,2),(14,1),(14,5),(15,1),(16,1),(16,2),(16,4),
}
META_FEATURES = ["DOC", "feed", "material"]
SEEDS         = [0, 1, 2, 3, 4]
THRESH        = 1e6
N_STATS_PER_SENSOR = 2
MODELS        = ["FeatGRU", "FeatLSTM"]

REF = {
    "all6_4stat":   {"FeatGRU": 0.114839, "FeatLSTM": 0.117919},
    "ACvTvS_4stat": {"FeatGRU": 0.095122, "FeatLSTM": 0.092217},
    "all6_hampel":  {"FeatGRU": 0.140917, "FeatLSTM": 0.141668},
}

RNN_CFG = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0,
)


# ─── Sensor-specific feature extraction ──────────────────────────────────────
def extract_features_for_sensor(sensor: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size < 2:
        return np.zeros(2, dtype=np.float64)
    if sensor in CURRENT_SENSORS:
        return np.array([np.mean(arr), np.std(arr)])
    rms = np.sqrt(np.mean(arr ** 2))
    if sensor in VIB_SENSORS:
        crest = np.abs(arr).max() / rms if rms > 1e-12 else 0.0
        return np.array([rms, crest])
    # AE_SENSORS
    kurt = scipy_kurtosis(arr, fisher=True, bias=True) if arr.size > 3 else 0.0
    kurt = 0.0 if not np.isfinite(kurt) else kurt
    return np.array([rms, kurt])


# ─── Utils ────────────────────────────────────────────────────────────────────
def parse_signal(value: object) -> np.ndarray:
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


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


def build_raw_cache(signal_df: pd.DataFrame, proc_clean: pd.DataFrame):
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
        base_len = min(len(a) for a in arrays.values())
        full = np.concatenate(
            [extract_features_for_sensor(s, arrays[s][:base_len]) for s in SENSORS])
        cache[(case_id, run_id)] = full   # 12-dim (6 sensors x 2 stats)

    first_run: dict[int,int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c, r) in cache if c == case_id)
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


def build_delta_meta(raw_cache, first_run, proc_clean):
    n_feat = len(SENSORS) * N_STATS_PER_SENSOR
    case_rows: dict[int, list[dict]] = {c: [] for c in CASE_SCOPE}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id)
        if key not in raw_cache:
            continue
        raw_vec = raw_cache[key]
        ref_key = (case_id, first_run.get(case_id, run_id))
        ref_vec = raw_cache.get(ref_key, np.zeros(n_feat))
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
    out_dir = ROOT / "experiments" / "executions" / "B2" / "S1" / f"{ts}_sensor_specific_features"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== B2_S1 variant: Sensor-specific 2-stat features (all 6 sensors) ===")
    log("Current(mean,std) / Vibration(rms,crest) / AE(rms,kurtosis)")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    raw_cache, first_run = build_raw_cache(signal_df, proc_clean)
    all_cases = build_delta_meta(raw_cache, first_run, proc_clean)
    input_dim = len(SENSORS) * N_STATS_PER_SENSOR + len(META_FEATURES)
    log(f"Input dim: {input_dim} (12 sensor-specific delta + 3 meta)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    results: dict[str, dict] = {}
    t_total = time_mod.time()
    for model_type in MODELS:
        t0 = time_mod.time()
        seed_rmses = [run_locv_seed(all_cases, input_dim, model_type, device, s) for s in SEEDS]
        mean_, std_ = float(np.mean(seed_rmses)), float(np.std(seed_rmses))
        log(f"  {model_type:<10} mean={mean_:.6f}  std={std_:.6f}  [{time_mod.time()-t0:.0f}s]")
        results[model_type] = {"mean": mean_, "std": std_, "seed_rmses": seed_rmses}

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    log(f"{'Model':<10} {'Sensor-specific':>16} {'all6_4stat':>12} {'ACvTvS_4stat':>14} {'all6_hampel':>13}")
    log("-" * 70)
    for m in MODELS:
        r = results[m]["mean"]
        log(f"{m:<10} {r:>16.6f} {REF['all6_4stat'][m]:>12.6f} {REF['ACvTvS_4stat'][m]:>14.6f} {REF['all6_hampel'][m]:>13.6f}")

    for m in MODELS:
        d_all6 = results[m]["mean"] - REF["all6_4stat"][m]
        d_acvtvs = results[m]["mean"] - REF["ACvTvS_4stat"][m]
        log(f"\n{m}: vs all6_4stat Δ={d_all6:+.6f} ({'개선' if d_all6<0 else '악화'}), "
            f"vs ACvTvS_4stat(동일차원) Δ={d_acvtvs:+.6f} ({'개선' if d_acvtvs<0 else '악화'})")

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame([
        {"model": m, "mean": r["mean"], "std": r["std"],
         "ref_all6_4stat": REF["all6_4stat"][m], "ref_ACvTvS_4stat": REF["ACvTvS_4stat"][m],
         "ref_all6_hampel": REF["all6_hampel"][m]}
        for m, r in results.items()
    ]).to_csv(out_dir / "metrics" / "sensor_specific_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    names, vals, colors = [], [], []
    for m in MODELS:
        names += [f"{m}\nsensor-specific", f"{m}\nall6_4stat", f"{m}\nACvTvS_4stat", f"{m}\nall6_hampel"]
        vals  += [results[m]["mean"], REF["all6_4stat"][m], REF["ACvTvS_4stat"][m], REF["all6_hampel"][m]]
        colors += ["#d65f5f", "#4878d0", "#6acc65", "#ee854a"]
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, f"{v:.4f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Observed-VB RMSE")
    ax.set_title("Sensor-specific 2-stat vs generic 4-stat / Hampel")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_dir / "figures" / "sensor_specific_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "experiment": "B2_S1_sensor_specific_features",
        "feature_scheme": {"current": ["mean","std"], "vibration": ["rms","crest_factor"], "AE": ["rms","kurtosis"]},
        "input_dim": input_dim,
        "results": results, "references": REF,
        "execution_dir": str(out_dir), "total_elapsed_s": time_mod.time() - t_total,
    }
    (out_dir / "logs" / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

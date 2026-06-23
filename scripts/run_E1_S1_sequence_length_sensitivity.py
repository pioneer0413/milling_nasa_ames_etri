#!/usr/bin/env python3
"""E1_S1: Sequence Length Sensitivity — 이전 N run 이력 의존성

E1 논문화 보강 실험. GRU에 가장 최근 N개 run만 입력했을 때 N에 따른 성능 변화.
N이 커질수록 이력이 늘어나며, N=all이 현재 채택 조건.

N 조건: [1, 2, 3, 5, 10, all]
  N=1   : 현재 run만 (시퀀스 효과 없음)
  N=all : 전체 run 이력 (B4/I1 ref = 0.095122)

설정: GRU AC+vT+vS (mask=13), Delta+Meta, 5-seed, 100% input, LOCV 15 cases

Output: experiments/executions/E1/S1/{timestamp}_sequence_length_sensitivity/
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
GRU_MASK      = 13    # AC+vT+vS

# N=None means use all available runs
N_CONDITIONS  = [1, 2, 3, 5, 10, None]   # None = all
N_LABELS      = ["N=1", "N=2", "N=3", "N=5", "N=10", "N=all"]

REF_ALL_MEAN  = 0.095122
REF_ALL_STD   = 0.001554

GRU_CFG = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0,
)


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
    return [i * 4 + j for i in range(N_SENSORS) if (mask >> i) & 1 for j in range(4)]


def mask_label(mask: int) -> str:
    return "+".join(SENSOR_ABBR[SENSORS[i]] for i in range(N_SENSORS) if (mask >> i) & 1)


# ─── Feature cache ────────────────────────────────────────────────────────────
def build_cache(
    signal_df: pd.DataFrame, proc_clean: pd.DataFrame
) -> tuple[dict[tuple[int,int], np.ndarray], dict[int,int]]:
    sensor_indices = mask_sensor_indices(GRU_MASK)
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
        end = max(1, int(np.ceil(base_len * PCT / 100.0)))
        full = np.concatenate([extract_features(arrays[s][:end]) for s in SENSORS])
        cache[(case_id, run_id)] = full[sensor_indices]

    first_run: dict[int,int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c, r) in cache if c == case_id)
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


# ─── Sequence builder with N-truncation ──────────────────────────────────────
def build_sequences(
    cache: dict[tuple[int,int], np.ndarray],
    first_run: dict[int,int],
    proc_clean: pd.DataFrame,
    n_limit: int | None,
) -> dict[int, dict]:
    """Build GRU sequences, keeping only the last n_limit runs per case."""
    n_sensor_feats = bin(GRU_MASK).count("1") * 4
    case_rows: dict[int, list[dict]] = {c: [] for c in CASE_SCOPE}

    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id)
        if key not in cache:
            continue
        ref_key = (case_id, first_run.get(case_id, run_id))
        ref_vec = cache.get(ref_key, np.zeros(n_sensor_feats))
        delta = cache[key] - ref_vec
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        meta = np.array([float(getattr(row, mf, 0.0)) for mf in META_FEATURES])
        feat = np.concatenate([delta, meta]).astype(np.float32)
        vb   = float(row.VB) if not pd.isna(row.VB) else 0.0
        case_rows[case_id].append({"feat": feat, "vb": vb, "run": run_id})

    cases: dict[int, dict] = {}
    for case_id, rows in case_rows.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["run"])
        # Keep only last n_limit rows
        if n_limit is not None and len(rows_sorted) > n_limit:
            rows_sorted = rows_sorted[-n_limit:]
        seq  = np.stack([r["feat"] for r in rows_sorted])
        vb   = np.array([r["vb"]  for r in rows_sorted], dtype=np.float32)
        runs = np.array([r["run"] for r in rows_sorted], dtype=int)
        cases[case_id] = {"seq": seq, "vb": vb, "runs": runs, "n_runs": len(rows_sorted)}
    return cases


# ─── GRU ─────────────────────────────────────────────────────────────────────
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=GRU_CFG["epochs"])

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
    for _ in range(GRU_CFG["epochs"]):
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


def run_locv_seed(
    all_cases: dict[int, dict],
    input_dim: int,
    device: torch.device,
    seed: int,
) -> tuple[float, dict[int,float]]:
    case_rmses: dict[int,float] = {}
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


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_results(
    n_labels: list[str],
    means: list[float],
    stds: list[float],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean ± std line chart
    ax = axes[0]
    x = np.arange(len(n_labels))
    ax.fill_between(x, np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds), alpha=0.2, color="steelblue")
    ax.plot(x, means, "o-", color="steelblue", linewidth=2, markersize=7)
    ax.axhline(REF_ALL_MEAN, color="gray", linestyle="--", linewidth=1,
               label=f"N=all ref={REF_ALL_MEAN:.4f}")
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.annotate(f"{m:.4f}", (xi, m + s + 0.003), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(n_labels, fontsize=10)
    ax.set_ylabel("Observed-VB RMSE (5-seed mean ± std)")
    ax.set_title("RMSE vs. Sequence Length N")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: relative degradation vs N=all
    ax = axes[1]
    deltas = [m - REF_ALL_MEAN for m in means]
    colors = ["tomato" if d > 0 else "steelblue" for d in deltas]
    ax.bar(x, deltas, color=colors, alpha=0.85)
    for xi, d in enumerate(deltas):
        ax.text(xi, d + (0.001 if d >= 0 else -0.003),
                f"{d:+.4f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(n_labels, fontsize=10)
    ax.set_ylabel("RMSE − N=all (positive = worse)")
    ax.set_title("Degradation vs. N=all")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"E1_S1: Sequence Length Sensitivity — GRU {mask_label(GRU_MASK)}, Delta+Meta, 5-seed",
        fontsize=13)
    plt.tight_layout()
    fig.savefig(str(out_dir / "sequence_length_sensitivity.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "sequence_length_sensitivity.svg"), bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = (ROOT / "experiments" / "executions" / "E1" / "S1"
               / f"{ts}_sequence_length_sensitivity")
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== E1_S1: Sequence Length Sensitivity ===")
    log(f"GRU: {mask_label(GRU_MASK)} (mask={GRU_MASK}), Delta+Meta, Seeds={SEEDS}, PCT={PCT}%")
    log(f"N conditions: {N_LABELS}")
    log(f"Ref (N=all, B4): mean={REF_ALL_MEAN}, std={REF_ALL_STD}")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    log("Building feature cache...")
    cache, first_run = build_cache(signal_df, proc_clean)
    log(f"Cache size: {len(cache)} runs")

    n_sensor  = bin(GRU_MASK).count("1")
    input_dim = n_sensor * 4 + len(META_FEATURES)   # 15
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}, input_dim={input_dim}")

    results: dict[str, dict] = {}
    t_total = time_mod.time()

    for n_limit, n_label in zip(N_CONDITIONS, N_LABELS):
        log(f"\n--- {n_label} ---")
        all_cases = build_sequences(cache, first_run, proc_clean, n_limit)

        # Report actual sequence lengths
        lens = [info["n_runs"] for info in all_cases.values()]
        log(f"  Actual seq lengths: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.1f}")

        seed_rmses: list[float] = []
        case_matrix: dict[int,list[float]] = {c: [] for c in CASE_SCOPE}
        t0 = time_mod.time()

        for seed in SEEDS:
            mean_rmse, case_rmses = run_locv_seed(all_cases, input_dim, device, seed)
            seed_rmses.append(mean_rmse)
            for c, r in case_rmses.items():
                case_matrix[c].append(r)
            log(f"  seed={seed}  RMSE={mean_rmse:.6f}  [{time_mod.time()-t0:.0f}s]")

        mean_ = float(np.mean(seed_rmses))
        std_  = float(np.std(seed_rmses))
        delta = mean_ - REF_ALL_MEAN
        log(f"  {n_label}: mean={mean_:.6f}  std={std_:.6f}  vs N=all: {delta:+.6f}")

        results[n_label] = {
            "n_limit": n_limit,
            "seed_rmses": seed_rmses,
            "mean": mean_, "std": std_,
            "vs_all": delta,
            "seq_len_min": int(min(lens)),
            "seq_len_max": int(max(lens)),
            "seq_len_mean": float(np.mean(lens)),
            "case_matrix": {c: vs for c, vs in case_matrix.items() if vs},
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    agg_rows = [
        {"n_label": lbl, "n_limit": r["n_limit"],
         "mean": r["mean"], "std": r["std"], "vs_all": r["vs_all"]}
        for lbl, r in results.items()
    ]
    pd.DataFrame(agg_rows).to_csv(out_dir / "metrics" / "aggregate.csv", index=False)

    seed_rows = []
    for lbl, r in results.items():
        for s, v in zip(SEEDS, r["seed_rmses"]):
            seed_rows.append({"n_label": lbl, "seed": s, "rmse": v})
    pd.DataFrame(seed_rows).to_csv(out_dir / "metrics" / "seed_detail.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    means = [results[lbl]["mean"] for lbl in N_LABELS]
    stds  = [results[lbl]["std"]  for lbl in N_LABELS]
    plot_results(N_LABELS, means, stds, out_dir / "figures")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    log(f"{'N':<8} {'Mean':>10} {'Std':>10} {'vs N=all':>12}")
    log("-" * 44)
    for lbl in N_LABELS:
        r = results[lbl]
        marker = " ←" if lbl == "N=all" else ""
        log(f"{lbl:<8} {r['mean']:>10.6f} {r['std']:>10.6f} {r['vs_all']:>+12.6f}{marker}")

    best = min(results, key=lambda l: results[l]["mean"])
    log(f"\nBest: {best}  mean={results[best]['mean']:.6f}")

    # Saturation analysis: find smallest N within 1% of N=all
    for lbl in N_LABELS:
        r = results[lbl]
        if abs(r["vs_all"]) / REF_ALL_MEAN < 0.01:
            log(f"Saturation (within 1% of N=all): {lbl}")
            break

    summary = {
        "experiment": "E1_S1_sequence_length_sensitivity",
        "gru_mask": GRU_MASK, "sensor_subset": mask_label(GRU_MASK),
        "seeds": SEEDS, "pct": PCT,
        "n_conditions": {lbl: results[lbl] for lbl in N_LABELS},
        "best_n": best,
        "ref_all": {"mean": REF_ALL_MEAN, "std": REF_ALL_STD},
        "execution_dir": str(out_dir),
        "total_elapsed_s": time_mod.time() - t_total,
    }
    (out_dir / "logs" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text(
        "\n".join(log_lines), encoding="utf-8")

    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

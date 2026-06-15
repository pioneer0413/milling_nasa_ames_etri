#!/usr/bin/env python3
"""H19_S1: Shuffled-Sequence Ablation — does GRU actually use run ordering?

B2 논문화 블로커 해소. 동일 GRU 아키텍처(AC+vT+vS)에서 run 순서를 무작위 치환하여
순서 정보 제거 시 성능이 얼마나 하락하는지 정량화.

설계:
  - Case별로 run 순서를 무작위 permutation (훈련+테스트 모두 동일하게 적용)
  - seed별로 다른 permutation → 평균 성능 = "ordering 없는 GRU" 기댓값
  - ordered GRU(H17_S1): mean=0.095122, std=0.001554
  - shuffled GRU: if RMSE >> ordered → GRU는 실제로 순서를 이용함

설정:
  GRU: AC+vT+vS (mask=13), input_dim=15, same hyperparams as H17
  Seeds: [0, 1, 2, 3, 4]
  Input: 100% (no prefix/segment)
  Protocol: LOCV 15 cases, observed_vb eval

Output: experiments/executions/H19/S1/{timestamp}_shuffled_ablation/
"""
from __future__ import annotations

import json
import sys
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

GRU_MASK = 13   # AC+vT+vS

# Reference from H17_S1 (ordered, 5-seed)
REF_ORDERED_MEAN = 0.095122
REF_ORDERED_STD  = 0.001554

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
    return [
        i * 4 + j
        for i in range(N_SENSORS) if (mask >> i) & 1
        for j in range(4)
    ]


def mask_label(mask: int) -> str:
    return "+".join(SENSOR_ABBR[SENSORS[i]] for i in range(N_SENSORS) if (mask >> i) & 1)


# ─── Feature cache ────────────────────────────────────────────────────────────
def build_cache(
    signal_df: pd.DataFrame, proc_clean: pd.DataFrame
) -> tuple[dict[tuple[int,int], np.ndarray], dict[int,int]]:
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
        feats = np.concatenate([extract_features(arrays[s][:end]) for s in SENSORS])
        cache[(case_id, run_id)] = feats

    first_run: dict[int,int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c,r) in cache if c == case_id)
        if runs:
            first_run[int(case_id)] = runs[0]
    return cache, first_run


# ─── Sequence builder (ordered) ───────────────────────────────────────────────
def build_gru_sequences(
    cache: dict[tuple[int,int], np.ndarray],
    first_run: dict[int,int],
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
        vb   = float(row.VB) if not pd.isna(row.VB) else 0.0
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


def shuffle_sequences(all_cases: dict[int, dict], rng: np.random.Generator) -> dict[int, dict]:
    """Return a new dict with each case's run sequence randomly permuted."""
    shuffled: dict[int, dict] = {}
    for case_id, info in all_cases.items():
        n = info["n_runs"]
        perm = rng.permutation(n)
        shuffled[case_id] = {
            "seq":    info["seq"][perm],
            "vb":     info["vb"][perm],
            "runs":   info["runs"][perm],
            "n_runs": n,
        }
    return shuffled


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
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Returns {case_id: (predictions_in_shuffled_order, run_ids_in_shuffled_order)}."""
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
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with torch.no_grad():
        for cid, info in test_cases.items():
            pred = (
                model(
                    torch.tensor(info["seq"], dtype=torch.float32).unsqueeze(0).to(device),
                    torch.tensor([info["n_runs"]]),
                ).squeeze(0).cpu().numpy() * y_std + y_mean
            )
            results[cid] = (pred, info["runs"])
    return results


def run_locv_seed(
    all_cases_ordered: dict[int, dict],
    input_dim: int,
    device: torch.device,
    seed: int,
    shuffle: bool,
) -> tuple[float, dict[int,float]]:
    """Run LOCV for one seed; optionally shuffle run order per case."""
    rng = np.random.default_rng(seed)

    if shuffle:
        all_cases = shuffle_sequences(all_cases_ordered, rng)
    else:
        all_cases = all_cases_ordered

    case_rmses: dict[int,float] = {}
    for tc in CASE_SCOPE:
        if tc not in all_cases:
            continue
        train = {c: all_cases[c] for c in CASE_SCOPE if c != tc and c in all_cases}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = fit_predict_gru(train, {tc: all_cases[tc]}, input_dim, device, seed)

        pred_vb, pred_runs = preds[tc]
        # Map back: match by run id to get obs mask
        tc_ordered = all_cases_ordered[tc]
        run_to_obs = {
            int(r): (tc, int(r)) not in NON_OBSERVED_RUNS
            for r in tc_ordered["runs"]
        }
        run_to_true = {
            int(r): float(v)
            for r, v in zip(tc_ordered["runs"], tc_ordered["vb"])
        }
        y_pred_list, y_true_list = [], []
        for pred_val, run_id in zip(pred_vb, pred_runs):
            rid = int(run_id)
            if run_to_obs.get(rid, False):
                y_pred_list.append(float(pred_val))
                y_true_list.append(run_to_true[rid])

        if not y_true_list:
            continue
        case_rmses[tc] = float(np.sqrt(
            mean_squared_error(np.array(y_true_list), np.array(y_pred_list))
        ))

    mean_rmse = float(np.mean(list(case_rmses.values()))) if case_rmses else float("nan")
    return mean_rmse, case_rmses


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_results(
    ordered_rmses: list[float],
    shuffled_rmses: list[float],
    ordered_case: dict[int,list[float]],
    shuffled_case: dict[int,list[float]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: seed-level RMSE comparison
    ax = axes[0]
    x = np.arange(len(SEEDS))
    w = 0.35
    ax.bar(x - w/2, ordered_rmses,  w, label="Ordered (H17 ref)",  color="steelblue", alpha=0.85)
    ax.bar(x + w/2, shuffled_rmses, w, label="Shuffled (H19)",     color="tomato",    alpha=0.85)
    ax.axhline(np.mean(ordered_rmses),  color="steelblue", linestyle="--", linewidth=1.2,
               label=f"Ordered mean={np.mean(ordered_rmses):.4f}")
    ax.axhline(np.mean(shuffled_rmses), color="tomato",    linestyle="--", linewidth=1.2,
               label=f"Shuffled mean={np.mean(shuffled_rmses):.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("Observed-VB RMSE")
    ax.set_title("Ordered vs Shuffled: Aggregate RMSE per Seed")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: per-case mean RMSE comparison (ordered vs shuffled)
    ax = axes[1]
    case_ids = sorted(ordered_case.keys())
    ord_means  = [np.mean(ordered_case[c])  if ordered_case[c]  else float("nan") for c in case_ids]
    shuf_means = [np.mean(shuffled_case[c]) if shuffled_case[c] else float("nan") for c in case_ids]
    x2 = np.arange(len(case_ids))
    ax.bar(x2 - w/2, ord_means,  w, color="steelblue", alpha=0.85, label="Ordered")
    ax.bar(x2 + w/2, shuf_means, w, color="tomato",    alpha=0.85, label="Shuffled")
    ax.set_xticks(x2)
    ax.set_xticklabels([f"C{c}" for c in case_ids], fontsize=8)
    ax.set_ylabel("Mean RMSE (5 seeds)")
    ax.set_title("Per-Case RMSE: Ordered vs Shuffled")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    delta = np.mean(shuffled_rmses) - np.mean(ordered_rmses)
    fig.suptitle(
        f"H19_S1: Shuffled-Sequence Ablation  |  Δ RMSE = {delta:+.4f}  "
        f"({delta/np.mean(ordered_rmses)*100:+.1f}%)",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(str(out_dir / "shuffled_ablation.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "shuffled_ablation.svg"), bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "H19" / "S1" / f"{ts}_shuffled_ablation"
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== H19_S1: Shuffled-Sequence Ablation ===")
    log(f"GRU: AC+vT+vS (mask={GRU_MASK}), input_dim={bin(GRU_MASK).count('1')*4+len(META_FEATURES)}")
    log(f"Seeds: {SEEDS}, PCT={PCT}%, LOCV={len(CASE_SCOPE)} cases")
    log(f"Ordered ref (H17_S1): mean={REF_ORDERED_MEAN}, std={REF_ORDERED_STD}")

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

    n_sensors  = bin(GRU_MASK).count("1")
    input_dim  = n_sensors * 4 + len(META_FEATURES)
    all_cases_ordered = build_gru_sequences(cache, first_run, proc_clean, GRU_MASK)
    log(f"Cases built: {len(all_cases_ordered)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ── Run: Shuffled ─────────────────────────────────────────────────────────
    log("\n--- Shuffled GRU ---")
    shuffled_seed_rmses: list[float] = []
    shuffled_case_matrix: dict[int,list[float]] = {c: [] for c in CASE_SCOPE}

    import time as time_mod
    t0 = time_mod.time()
    for seed in SEEDS:
        mean_rmse, case_rmses = run_locv_seed(
            all_cases_ordered, input_dim, device, seed, shuffle=True)
        shuffled_seed_rmses.append(mean_rmse)
        for c, r in case_rmses.items():
            shuffled_case_matrix[c].append(r)
        elapsed = time_mod.time() - t0
        log(f"  seed={seed}  RMSE={mean_rmse:.6f}  [{elapsed:.0f}s]")

    shuf_mean = float(np.mean(shuffled_seed_rmses))
    shuf_std  = float(np.std(shuffled_seed_rmses))
    shuf_cv   = shuf_std / shuf_mean if shuf_mean > 0 else float("nan")
    log(f"  Shuffled 5-seed: mean={shuf_mean:.6f}  std={shuf_std:.6f}  CV={shuf_cv:.4f}")

    # Reference from H17 (use stored per-seed values)
    # Re-run ordered for per-case matrix comparison
    log("\n--- Ordered GRU (for per-case comparison) ---")
    ordered_seed_rmses: list[float] = []
    ordered_case_matrix: dict[int,list[float]] = {c: [] for c in CASE_SCOPE}
    t0 = time_mod.time()
    for seed in SEEDS:
        mean_rmse, case_rmses = run_locv_seed(
            all_cases_ordered, input_dim, device, seed, shuffle=False)
        ordered_seed_rmses.append(mean_rmse)
        for c, r in case_rmses.items():
            ordered_case_matrix[c].append(r)
        elapsed = time_mod.time() - t0
        log(f"  seed={seed}  RMSE={mean_rmse:.6f}  [{elapsed:.0f}s]")

    ord_mean = float(np.mean(ordered_seed_rmses))
    ord_std  = float(np.std(ordered_seed_rmses))
    log(f"  Ordered 5-seed: mean={ord_mean:.6f}  std={ord_std:.6f}")

    # ── Delta analysis ────────────────────────────────────────────────────────
    delta_abs = shuf_mean - ord_mean
    delta_rel = delta_abs / ord_mean * 100.0

    log(f"\n  Δ RMSE (shuffled - ordered) = {delta_abs:+.6f}  ({delta_rel:+.1f}%)")
    if delta_abs > 2 * ord_std:
        log("  → RMSE 상승이 ordered std의 2배 초과: GRU는 run 순서를 유의하게 이용함 [CONFIRMED]")
    elif delta_abs > 0:
        log("  → RMSE 소폭 상승: ordering이 일부 기여하나 통계적 유의성 낮음")
    else:
        log("  → RMSE 감소 또는 동등: ordering이 GRU 성능에 기여하지 않음")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    agg_df = pd.DataFrame({
        "seed":          SEEDS,
        "ordered_rmse":  ordered_seed_rmses,
        "shuffled_rmse": shuffled_seed_rmses,
        "delta":         [s - o for s, o in zip(shuffled_seed_rmses, ordered_seed_rmses)],
    })
    agg_df.to_csv(out_dir / "metrics" / "seed_aggregate.csv", index=False)

    per_case_rows = []
    for c in CASE_SCOPE:
        if not ordered_case_matrix[c]:
            continue
        per_case_rows.append({
            "case":          c,
            "ordered_mean":  float(np.mean(ordered_case_matrix[c])),
            "ordered_std":   float(np.std(ordered_case_matrix[c])),
            "shuffled_mean": float(np.mean(shuffled_case_matrix[c])) if shuffled_case_matrix[c] else float("nan"),
            "shuffled_std":  float(np.std(shuffled_case_matrix[c]))  if shuffled_case_matrix[c] else float("nan"),
        })
    per_case_df = pd.DataFrame(per_case_rows)
    per_case_df["delta"] = per_case_df["shuffled_mean"] - per_case_df["ordered_mean"]
    per_case_df.to_csv(out_dir / "metrics" / "per_case.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(
        ordered_seed_rmses, shuffled_seed_rmses,
        {c: v for c, v in ordered_case_matrix.items()  if v},
        {c: v for c, v in shuffled_case_matrix.items() if v},
        out_dir / "figures",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    log(f"{'Condition':<20} {'Mean':>10} {'Std':>10} {'CV':>8}")
    log("-" * 52)
    log(f"{'Ordered (H19)':<20} {ord_mean:>10.6f} {ord_std:>10.6f} {ord_std/ord_mean:>8.4f}")
    log(f"{'Shuffled (H19)':<20} {shuf_mean:>10.6f} {shuf_std:>10.6f} {shuf_cv:>8.4f}")
    log(f"{'Δ (shuf-ord)':<20} {delta_abs:>+10.6f} {'':<10} {delta_rel:>+7.1f}%")
    log(f"\nOrdered H17 ref:  mean={REF_ORDERED_MEAN}  std={REF_ORDERED_STD}")

    log("\nPer-seed delta:")
    for s, o, sh in zip(SEEDS, ordered_seed_rmses, shuffled_seed_rmses):
        log(f"  seed={s}  ordered={o:.6f}  shuffled={sh:.6f}  Δ={sh-o:+.6f}")

    summary = {
        "experiment":   "H19_S1_shuffled_ablation",
        "ordered": {
            "seed_rmses": ordered_seed_rmses, "mean": ord_mean, "std": ord_std,
        },
        "shuffled": {
            "seed_rmses": shuffled_seed_rmses, "mean": shuf_mean, "std": shuf_std, "cv": shuf_cv,
        },
        "delta_abs":    delta_abs,
        "delta_rel_pct": delta_rel,
        "ref_H17_ordered_mean": REF_ORDERED_MEAN,
        "ref_H17_ordered_std":  REF_ORDERED_STD,
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

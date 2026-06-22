#!/usr/bin/env python3
"""I1_S2: Feature Component Ablation — All Sensors x {FeatGRU, FeatLSTM}

[I1] Ablation: Feature component 보완 필요 항목 해소.
H20_S1(=I1_S1, GRU AC+vT+vS만)의 잔여 TODO:
  - GRU AC+vT+vS -> All sensors(mask=63)로 변경
  - FeatLSTM도 동일 ablation에 적용

6조건 (All sensors mask=63, 5-seed, 100% input, LOCV 15 cases) x 2 모델(GRU/LSTM):
  Raw           : 절대값 통계 (mean, rms, std, peak) — no delta, no meta    dim=24
  Delta         : run-to-run 변화량만                — no raw,   no meta    dim=24
  Raw+Delta     : 절대값 + 변화량                    — no meta              dim=48
  Raw+Meta      : 절대값 + DOC/feed/material          — no delta             dim=27
  Delta+Meta    : 변화량 + meta (AC+vT+vS 기준 현재 채택)                      dim=27
  Raw+Delta+Meta: 절대값 + 변화량 + meta                                      dim=51

Reference (H20_S1=I1_S1, GRU AC+vT+vS, Delta+Meta): mean=0.095122, std=0.001554

Output: experiments/executions/I1/S2/{timestamp}_feature_component_all_sensors/
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
MASK          = 63   # all 6 sensors: 0b111111
MODELS        = ["FeatGRU", "FeatLSTM"]

REF_DELTA_META_MEAN = 0.095122   # H20_S1 (=I1_S1) GRU AC+vT+vS
REF_DELTA_META_STD  = 0.001554

RNN_CFG = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0,
)

# Feature conditions: (name, use_raw, use_delta, use_meta)
FEAT_CONDITIONS = [
    ("Raw",             True,  False, False),
    ("Delta",           False, True,  False),
    ("Raw+Delta",       True,  True,  False),
    ("Raw+Meta",        True,  False, True),
    ("Delta+Meta",      False, True,  True),
    ("Raw+Delta+Meta",  True,  True,  True),
]


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


# ─── Feature cache (raw 4-stat per run) ──────────────────────────────────────
def build_raw_cache(
    signal_df: pd.DataFrame, proc_clean: pd.DataFrame
) -> tuple[dict[tuple[int,int], np.ndarray], dict[int,int]]:
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


# ─── Sequence builder ─────────────────────────────────────────────────────────
def build_sequences(
    raw_cache: dict[tuple[int,int], np.ndarray],
    first_run: dict[int,int],
    proc_clean: pd.DataFrame,
    use_raw: bool,
    use_delta: bool,
    use_meta: bool,
) -> dict[int, dict]:
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
        delta_vec = raw_vec - ref_vec

        parts: list[np.ndarray] = []
        if use_raw:
            parts.append(raw_vec)
        if use_delta:
            d = delta_vec.copy()
            if not np.all(np.isfinite(d)):
                d = np.where(np.isfinite(d), d, 0.0)
            parts.append(d)
        if use_meta:
            meta = np.array([float(getattr(row, mf, 0.0)) for mf in META_FEATURES])
            parts.append(meta)

        feat = np.concatenate(parts).astype(np.float32)
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


# ─── Model ────────────────────────────────────────────────────────────────────
class DeltaMetaRNN(nn.Module):
    def __init__(self, input_dim: int, cell_type: str) -> None:
        super().__init__()
        cfg = RNN_CFG
        rnn_cls = nn.GRU if cell_type == "FeatGRU" else nn.LSTM
        self.rnn = rnn_cls(
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
        out_packed, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out).squeeze(-1)


def fit_predict(
    train_cases: dict[int, dict],
    test_cases: dict[int, dict],
    input_dim: int,
    cell_type: str,
    device: torch.device,
    seed: int,
) -> dict[int, np.ndarray]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = DeltaMetaRNN(input_dim, cell_type).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=RNN_CFG["lr"], weight_decay=RNN_CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=RNN_CFG["epochs"])

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


def run_locv_seed(
    all_cases: dict[int, dict],
    input_dim: int,
    cell_type: str,
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
            preds = fit_predict(train, {tc: all_cases[tc]}, input_dim, cell_type, device, seed)
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
def plot_results(results: dict[str, dict], out_dir: Path) -> None:
    cond_names = [c[0] for c in FEAT_CONDITIONS]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    colors = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f", "#956cb4", "#8c613c"]

    for ax, model_type in zip(axes, MODELS):
        means = [results[model_type][n]["mean"] for n in cond_names]
        stds  = [results[model_type][n]["std"]  for n in cond_names]
        x = np.arange(len(cond_names))
        bars = ax.bar(x, means, color=colors[:len(cond_names)], alpha=0.85,
                      yerr=stds, capsize=4)
        dm_idx = cond_names.index("Delta+Meta")
        bars[dm_idx].set_edgecolor("black")
        bars[dm_idx].set_linewidth(2)
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + std + 0.003,
                    f"{mean:.4f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(REF_DELTA_META_MEAN, color="black", linestyle="--", linewidth=1,
                   label=f"GRU AC+vT+vS Delta+Meta ref={REF_DELTA_META_MEAN:.4f} (H20_S1)")
        ax.set_xticks(x)
        ax.set_xticklabels(cond_names, fontsize=9, rotation=15)
        ax.set_title(f"{model_type}  All Sensors (mask={MASK})")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Observed-VB RMSE (5-seed mean ± std)")
    fig.suptitle("I1_S2: Feature Component Ablation — All Sensors x {FeatGRU, FeatLSTM}", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(out_dir / "feature_component_all_sensors.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "feature_component_all_sensors.svg"), bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ts      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = (ROOT / "experiments" / "executions" / "I1" / "S2"
               / f"{ts}_feature_component_all_sensors")
    for sub in ["metrics", "figures", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== I1_S2: Feature Component Ablation — All Sensors x {FeatGRU, FeatLSTM} ===")
    log(f"Sensors: {mask_label(MASK)} (mask={MASK}), Models={MODELS}, Seeds={SEEDS}, PCT={PCT}%")
    log(f"Conditions: {[c[0] for c in FEAT_CONDITIONS]}")
    log(f"Ref (H20_S1=I1_S1, GRU AC+vT+vS, Delta+Meta): mean={REF_DELTA_META_MEAN}, std={REF_DELTA_META_STD}")

    log("\nLoading data...")
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)
    log(f"Clean runs: {len(proc_clean)}")

    log("Building raw feature cache (all sensors)...")
    raw_cache, first_run = build_raw_cache(signal_df, proc_clean)
    log(f"Cache size: {len(raw_cache)} runs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    results: dict[str, dict[str, dict]] = {m: {} for m in MODELS}
    t_total = time_mod.time()

    for model_type in MODELS:
        for cond_name, use_raw, use_delta, use_meta in FEAT_CONDITIONS:
            n_sensor = bin(MASK).count("1")
            input_dim = (n_sensor * 4 if use_raw else 0) + \
                        (n_sensor * 4 if use_delta else 0) + \
                        (len(META_FEATURES) if use_meta else 0)
            log(f"\n--- {model_type} / {cond_name}  input_dim={input_dim} ---")

            all_cases = build_sequences(raw_cache, first_run, proc_clean,
                                        use_raw, use_delta, use_meta)
            seed_rmses: list[float] = []
            case_matrix: dict[int, list[float]] = {c: [] for c in CASE_SCOPE}
            t0 = time_mod.time()

            for seed in SEEDS:
                mean_rmse, case_rmses = run_locv_seed(
                    all_cases, input_dim, model_type, device, seed)
                seed_rmses.append(mean_rmse)
                for c, r in case_rmses.items():
                    case_matrix[c].append(r)
                log(f"  seed={seed}  RMSE={mean_rmse:.6f}  [{time_mod.time()-t0:.0f}s]")

            mean_ = float(np.mean(seed_rmses))
            std_  = float(np.std(seed_rmses))
            cv_   = std_ / mean_ if mean_ > 0 else float("nan")
            delta = mean_ - REF_DELTA_META_MEAN
            log(f"  {model_type}/{cond_name}: mean={mean_:.6f}  std={std_:.6f}  CV={cv_:.4f}  "
                f"vs GRU-AC+vT+vS ref: {delta:+.6f}")

            results[model_type][cond_name] = {
                "use_raw": use_raw, "use_delta": use_delta, "use_meta": use_meta,
                "input_dim": input_dim,
                "seed_rmses": seed_rmses,
                "mean": mean_, "std": std_, "cv": cv_,
                "vs_global_ref": delta,
                "case_matrix": {c: vs for c, vs in case_matrix.items() if vs},
            }

    # ── vs own Delta+Meta (within-model contrast) ───────────────────────────
    for model_type in MODELS:
        own_ref = results[model_type]["Delta+Meta"]["mean"]
        for cond_name in results[model_type]:
            results[model_type][cond_name]["vs_own_delta_meta"] = (
                results[model_type][cond_name]["mean"] - own_ref
            )

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    agg_rows = [
        {
            "model": model_type, "condition": name,
            "use_raw": r["use_raw"], "use_delta": r["use_delta"], "use_meta": r["use_meta"],
            "input_dim": r["input_dim"],
            "mean": r["mean"], "std": r["std"], "cv": r["cv"],
            "vs_own_delta_meta": r["vs_own_delta_meta"],
            "vs_global_ref": r["vs_global_ref"],
        }
        for model_type in MODELS for name, r in results[model_type].items()
    ]
    pd.DataFrame(agg_rows).to_csv(out_dir / "metrics" / "aggregate.csv", index=False)

    seed_rows = []
    for model_type in MODELS:
        for name, r in results[model_type].items():
            for s, v in zip(SEEDS, r["seed_rmses"]):
                seed_rows.append({"model": model_type, "condition": name, "seed": s, "rmse": v})
    pd.DataFrame(seed_rows).to_csv(out_dir / "metrics" / "seed_detail.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(results, out_dir / "figures")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    for model_type in MODELS:
        log(f"\n[{model_type}]  {'Condition':<18} {'dim':>4} {'Mean':>10} {'Std':>10} {'vs own D+M':>12} {'vs GRU-ref':>12}")
        log("-" * 75)
        for name, _, _, _ in FEAT_CONDITIONS:
            r = results[model_type][name]
            marker = " ←" if name == "Delta+Meta" else ""
            log(f"{'':>10}{name:<18} {r['input_dim']:>4} {r['mean']:>10.6f} {r['std']:>10.6f}"
                f" {r['vs_own_delta_meta']:>+12.6f} {r['vs_global_ref']:>+12.6f}{marker}")

    best_overall = min(
        ((m, n) for m in MODELS for n in results[m]),
        key=lambda mn: results[mn[0]][mn[1]]["mean"],
    )
    log(f"\nBest overall: {best_overall[0]}/{best_overall[1]}  "
        f"mean={results[best_overall[0]][best_overall[1]]['mean']:.6f}")

    summary = {
        "experiment": "I1_S2_feature_component_all_sensors",
        "mask": MASK, "sensor_subset": mask_label(MASK),
        "models": MODELS, "seeds": SEEDS, "pct": PCT,
        "results": results,
        "best_overall": {"model": best_overall[0], "condition": best_overall[1],
                          "mean": results[best_overall[0]][best_overall[1]]["mean"]},
        "ref_h20_s1_gru_acvtvs_delta_meta": {"mean": REF_DELTA_META_MEAN, "std": REF_DELTA_META_STD},
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

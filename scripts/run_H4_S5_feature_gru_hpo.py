#!/usr/bin/env python3
"""H4_S5: feature_gru hyperparameter optimization on the rank-1 S1_T4 setting.

Base setting (H4 leader board rank #1):
  scenario S1, model feature_gru, Delta+Meta features (24 delta + 3 meta = 27),
  prefix input strategy at 80% signal length, LOCV over 15 cases, single seed=0,
  GRU hidden=256, layers=3, dropout=0.1, head=16, epochs=200, lr=1e-3, wd=1e-4,
  cosine schedule, grad clip 1.0  ->  LOCV mean RMSE = 0.094549.

This script reproduces that exact data pipeline and GRU, then evaluates several
hyperparameter subsets. Each candidate config is run with 3 random seeds; the
per-seed LOCV mean RMSE is averaged across seeds to rank configs and pick the best.

Output:
  experiments/executions/H4/S5/{timestamp}_feature_gru_hpo/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

# ─── Constants (identical to H4_S1_T4) ─────────────────────────────────────────
CASE_SCOPE = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SENSORS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
EXCLUDED_RUNS = {(2, 1), (12, 1)}
META_FEATURES = ["DOC", "feed", "material"]
THRESH = 1e6
BASE_PREFIX = 80


# ─── Feature extraction ────────────────────────────────────────────────────────
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
    return df[~mask].reset_index(drop=True)


def build_feature_matrix(
    signal_df: pd.DataFrame, process_df: pd.DataFrame, prefix_percents: list[int]
) -> pd.DataFrame:
    """Delta+Meta feature matrix (24 delta + 3 meta), identical to H4_S1_T4."""
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
        doc = float(getattr(row, "DOC", 0.0))
        feed = float(getattr(row, "feed", 0.0))
        material = int(getattr(row, "material", 0))
        for pct in prefix_percents:
            raw_feats = cache.get((case_id, run_id, pct))
            ref_feats = cache.get((case_id, ref_run, pct))
            if raw_feats is None or ref_feats is None:
                continue
            delta_feats = raw_feats - ref_feats
            record: dict = {
                "case_id": case_id, "run": run_id, "signal_prefix_percent": float(pct),
                "VB": vb, "DOC": doc, "feed": feed, "material": material,
            }
            for col, val in zip(delta_cols, delta_feats):
                record[col] = float(val)
            rows.append(record)
    return pd.DataFrame(rows).sort_values(
        ["signal_prefix_percent", "case_id", "run"]
    ).reset_index(drop=True)


# ─── Parametric GRU ────────────────────────────────────────────────────────────
class DeltaMetaGRU(nn.Module):
    def __init__(
        self, input_dim: int, hidden_size: int, num_layers: int,
        dropout: float, head_hidden: int,
        use_metadata_film: bool = False,
        metadata_feature_dim: int = 3,
        film_hidden_dim: int | None = None,
        film_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.use_metadata_film = bool(use_metadata_film)
        self.metadata_feature_dim = int(metadata_feature_dim)
        if self.use_metadata_film:
            if self.metadata_feature_dim <= 0:
                raise ValueError(f"metadata_feature_dim must be > 0, got {metadata_feature_dim}")
            if self.metadata_feature_dim >= self.input_dim:
                raise ValueError(
                    "metadata_feature_dim must be smaller than input_dim so signal features remain unmodified: "
                    f"input_dim={input_dim}, metadata_feature_dim={metadata_feature_dim}"
                )
            self.signal_feature_dim = self.input_dim - self.metadata_feature_dim
            film_hidden = int(film_hidden_dim or max(8, self.metadata_feature_dim * 4))
            film_layers: list[nn.Module] = [
                nn.Linear(self.signal_feature_dim, film_hidden),
                nn.ReLU(),
            ]
            if float(film_dropout) > 0.0:
                film_layers.append(nn.Dropout(float(film_dropout)))
            film_layers.append(nn.Linear(film_hidden, self.metadata_feature_dim * 2))
            self.metadata_film = nn.Sequential(*film_layers)
            last = self.metadata_film[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        else:
            self.signal_feature_dim = self.input_dim - self.metadata_feature_dim
            self.metadata_film = None
        self.gru = nn.GRU(
            input_dim, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1)
        )

    def apply_metadata_film(self, x: torch.Tensor) -> torch.Tensor:
        if self.metadata_film is None:
            return x
        signal_features = x[..., : self.signal_feature_dim]
        metadata_features = x[..., self.signal_feature_dim :]
        gamma_beta = self.metadata_film(signal_features)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        modulated_metadata = metadata_features * (1.0 + gamma) + beta
        return torch.cat([signal_features, modulated_metadata], dim=-1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.apply_metadata_film(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out).squeeze(-1)


def build_sequences(feature_df: pd.DataFrame, delta_cols: list[str], pct: float) -> dict[int, dict]:
    subset = feature_df[feature_df["signal_prefix_percent"] == float(pct)].copy()
    cases: dict[int, dict] = {}
    for case_id in subset["case_id"].unique():
        c = subset[subset["case_id"] == case_id].sort_values("run").reset_index(drop=True)
        delta = c[delta_cols].to_numpy(dtype=np.float32)
        meta = c[META_FEATURES].to_numpy(dtype=np.float32)
        seq = np.concatenate([delta, meta], axis=1)
        cases[int(case_id)] = {
            "seq": seq, "vb": c["VB"].to_numpy(dtype=np.float32), "n_runs": len(c)
        }
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
        use_metadata_film=bool(cfg.get("use_metadata_film", False)),
        metadata_feature_dim=int(cfg.get("metadata_feature_dim", len(META_FEATURES))),
        film_hidden_dim=cfg.get("film_hidden_dim"),
        film_dropout=float(cfg.get("film_dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    epochs = cfg["epochs"]
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if cfg.get("scheduler", "cosine") == "cosine" else None
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


def run_locv_seed(
    all_case_data: dict[int, dict], input_dim: int, device: torch.device,
    cfg: dict, seed: int,
) -> pd.DataFrame:
    """One LOCV pass for a single seed -> per-case metrics."""
    rows: list[dict] = []
    for test_case in CASE_SCOPE:
        if test_case not in all_case_data:
            continue
        train_ids = [c for c in CASE_SCOPE if c != test_case and c in all_case_data]
        train_data = {c: all_case_data[c] for c in train_ids}
        test_data = {test_case: all_case_data[test_case]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = fit_predict_gru(train_data, test_data, input_dim, device, cfg, seed)
        y_pred = preds[test_case]
        y_true = all_case_data[test_case]["vb"]
        rows.append({"target_case": test_case, "n_runs": len(y_true), **compute_metrics(y_true, y_pred)})
    return pd.DataFrame(rows)


# ─── Hyperparameter search space ───────────────────────────────────────────────
BASE = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=16,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0, scheduler="cosine",
)
MODEL_CHOICES = ("feature_gru", "feature_film_gru")


def with_model_options(
    cfg: dict,
    model_type: str,
    film_hidden_dim: int | None = None,
    film_dropout: float = 0.0,
) -> dict:
    model_cfg = {**cfg, "model_type": model_type}
    if model_type == "feature_film_gru":
        model_cfg.update(
            {
                "use_metadata_film": True,
                "metadata_feature_dim": len(META_FEATURES),
                "film_dropout": float(film_dropout),
            }
        )
        if film_hidden_dim is not None:
            model_cfg["film_hidden_dim"] = int(film_hidden_dim)
    return model_cfg


def build_config_list() -> list[dict]:
    """Several hyperparameter subsets, each a one- or two-factor change from BASE."""
    configs: list[dict] = []
    seen: set[tuple] = set()

    def add(name: str, **overrides):
        cfg = {**BASE, **overrides, "name": name}
        key = tuple(sorted((k, v) for k, v in cfg.items() if k != "name"))
        if key in seen:
            return
        seen.add(key)
        configs.append(cfg)

    add("baseline_rank1")
    # Capacity: hidden size
    for h in [64, 128, 192, 384]:
        add(f"hidden{h}", hidden_size=h)
    # Depth
    for nl in [1, 2, 4]:
        add(f"layers{nl}", num_layers=nl)
    # Dropout
    for d in [0.0, 0.2, 0.3]:
        add(f"dropout{d}", dropout=d)
    # Weight decay
    for wd in [0.0, 1e-3, 3e-3]:
        add(f"wd{wd}", weight_decay=wd)
    # Learning rate
    for lr in [5e-4, 2e-3, 3e-3]:
        add(f"lr{lr}", lr=lr)
    # Epochs
    for ep in [120, 300, 400]:
        add(f"epochs{ep}", epochs=ep)
    # Head hidden
    add("head32", head_hidden=32)
    # Promising combinations (lower-capacity + regularization, tuned lr/epochs)
    add("combo_small_reg", hidden_size=128, num_layers=2, dropout=0.2, weight_decay=1e-3, epochs=300)
    add("combo_mid_reg", hidden_size=192, num_layers=2, dropout=0.2, weight_decay=1e-3, lr=5e-4, epochs=300)
    add("combo_big_lowlr", hidden_size=256, num_layers=3, dropout=0.2, lr=5e-4, weight_decay=1e-3, epochs=300)
    add("combo_small_long", hidden_size=128, num_layers=2, dropout=0.1, lr=1e-3, weight_decay=1e-3, epochs=400)
    add("combo_compact", hidden_size=96, num_layers=2, dropout=0.15, weight_decay=1e-3, lr=1e-3, epochs=300)
    return configs


def build_refine_list() -> list[dict]:
    """Stage-2: combine the winning single-factor changes from stage-1.

    Stage-1 winners over baseline: head32, hidden192, dropout0.0, layers2, wd1e-3.
    """
    configs: list[dict] = []
    seen: set[tuple] = set()

    def add(name: str, **overrides):
        cfg = {**BASE, **overrides, "name": name}
        key = tuple(sorted((k, v) for k, v in cfg.items() if k != "name"))
        if key in seen:
            return
        seen.add(key)
        configs.append(cfg)

    add("baseline_rank1")                                                   # reference
    add("head32")                                                          # stage-1 best, reference
    add("r_head32_h192", head_hidden=32, hidden_size=192)
    add("r_head32_d0", head_hidden=32, dropout=0.0)
    add("r_head32_h192_d0", head_hidden=32, hidden_size=192, dropout=0.0)
    add("r_head32_h192_l2", head_hidden=32, hidden_size=192, num_layers=2)
    add("r_head32_wd1e3", head_hidden=32, weight_decay=1e-3)
    add("r_head32_h192_wd1e3", head_hidden=32, hidden_size=192, weight_decay=1e-3)
    add("r_head32_h192_d0_wd1e3", head_hidden=32, hidden_size=192, dropout=0.0, weight_decay=1e-3)
    add("r_head32_l2_d0", head_hidden=32, num_layers=2, dropout=0.0)
    add("r_head32_h192_l2_d0", head_hidden=32, hidden_size=192, num_layers=2, dropout=0.0)
    return configs


# ─── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H4_S5 feature_gru hyperparameter optimization.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--prefix", type=int, default=BASE_PREFIX)
    p.add_argument("--model", choices=MODEL_CHOICES, default="feature_gru")
    p.add_argument("--film-hidden-dim", type=int, default=None)
    p.add_argument("--film-dropout", type=float, default=0.0)
    p.add_argument("--only", nargs="+", default=None, help="Run only these config names.")
    p.add_argument("--refine", action="store_true", help="Stage-2: run combined winning configs.")
    p.add_argument("--smoke", action="store_true", help="Baseline only, 1 seed, for timing.")
    p.add_argument("--output-root", default="experiments/executions")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / args.output_root / "H4" / "S5" / f"{timestamp}_{args.model}_hpo"
    for sub in ["configs", "metrics", "analysis", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    seeds = args.seeds[:1] if args.smoke else args.seeds
    configs = build_refine_list() if args.refine else build_config_list()
    if args.smoke:
        configs = [c for c in build_config_list() if c["name"] == "baseline_rank1"]
    elif args.only:
        configs = [c for c in configs if c["name"] in set(args.only)]
    configs = [
        with_model_options(c, args.model, args.film_hidden_dim, args.film_dropout)
        for c in configs
    ]

    log(f"=== H4_S5 {args.model} HPO === prefix={args.prefix}% seeds={seeds} configs={len(configs)}")

    log("Loading data...")
    signal_df = pd.read_csv(
        ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + SENSORS
    )
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()

    log("Preprocessing + building Delta+Meta feature matrix...")
    process_clean = preprocess(process_df)
    feat_df = build_feature_matrix(signal_df, process_clean, [args.prefix])
    delta_cols = delta_col_names()
    input_dim = len(delta_cols) + len(META_FEATURES)
    all_case_data = build_sequences(feat_df, delta_cols, float(args.prefix))
    log(f"Feature matrix {len(feat_df)} rows; {len(all_case_data)} cases; input_dim={input_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    seed_label = f"{len(seeds)}-seed"

    per_case_rows: list[dict] = []
    seed_rows: list[dict] = []
    config_rows: list[dict] = []

    t0 = time.time()
    for ci, cfg in enumerate(configs, 1):
        seed_rmses, seed_r2s, seed_maes = [], [], []
        for seed in seeds:
            df = run_locv_seed(all_case_data, input_dim, device, cfg, seed)
            locv_rmse = float(df["rmse"].mean())
            locv_r2 = float(df["r2"].mean())
            locv_mae = float(df["mae"].mean())
            seed_rmses.append(locv_rmse)
            seed_r2s.append(locv_r2)
            seed_maes.append(locv_mae)
            seed_rows.append({"config": cfg["name"], "seed": seed,
                              "locv_rmse": locv_rmse, "locv_r2": locv_r2, "locv_mae": locv_mae})
            df.insert(0, "config", cfg["name"])
            df.insert(1, "seed", seed)
            per_case_rows.extend(df.to_dict("records"))
        rec = {
            "config": cfg["name"],
            "model_type": cfg.get("model_type", "feature_gru"),
            "use_metadata_film": bool(cfg.get("use_metadata_film", False)),
            "metadata_feature_dim": int(cfg.get("metadata_feature_dim", 0)),
            "film_hidden_dim": cfg.get("film_hidden_dim"),
            "film_dropout": float(cfg.get("film_dropout", 0.0)),
            **{k: cfg[k] for k in ["hidden_size", "num_layers", "dropout", "head_hidden",
                                   "lr", "weight_decay", "epochs", "grad_clip", "scheduler"]},
            "mean_rmse": float(np.mean(seed_rmses)),
            "std_rmse": float(np.std(seed_rmses)),
            "mean_r2": float(np.mean(seed_r2s)),
            "mean_mae": float(np.mean(seed_maes)),
            "n_seeds": len(seeds),
        }
        config_rows.append(rec)
        log(f"[{ci}/{len(configs)}] {cfg['name']:<18} "
            f"mean_rmse={rec['mean_rmse']:.6f} (+/-{rec['std_rmse']:.6f}) "
            f"mean_r2={rec['mean_r2']:.4f}  [{time.time()-t0:.0f}s]")

    results = pd.DataFrame(config_rows).sort_values("mean_rmse").reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    pd.DataFrame(per_case_rows).to_csv(out_dir / "metrics" / "H4_S5_per_case_metrics.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out_dir / "metrics" / "H4_S5_seed_metrics.csv", index=False)
    results.to_csv(out_dir / "analysis" / "H4_S5_config_ranking.csv", index=False)

    best = results.iloc[0]
    baseline = results[results["config"] == "baseline_rank1"]
    config_meta = {
        "experiment": f"H4_S5_{args.model}_hpo",
        "model_type": args.model,
        "base_setting": f"S1_T4 Delta+Meta prefix-80 LOCV {args.model}",
        "prefix_percent": args.prefix, "seeds": seeds,
        "leaderboard_rank1_rmse": 0.094549,
        "search_space": {k: v for k, v in BASE.items()},
        "film": {
            "target": "metadata_features_only",
            "metadata_features": META_FEATURES,
            "signal_features_preserved": True,
            "film_hidden_dim": args.film_hidden_dim,
            "film_dropout": args.film_dropout,
        } if args.model == "feature_film_gru" else None,
        "best_config": best.to_dict(),
    }
    (out_dir / "configs" / "H4_S5_input_config.json").write_text(
        json.dumps(config_meta, indent=2, default=float), encoding="utf-8"
    )

    log(f"\n=== TOP CONFIGS (by {seed_label} mean LOCV RMSE) ===")
    show = results.head(10)[["rank", "config", "hidden_size", "num_layers", "dropout",
                             "lr", "weight_decay", "epochs", "mean_rmse", "std_rmse", "mean_r2"]]
    log("\n" + show.to_string(index=False))
    if not baseline.empty:
        log(f"\nBaseline (rank1 config, {seed_label}): mean_rmse={float(baseline['mean_rmse'].iloc[0]):.6f}")
    log(f"Best config: {best['config']} -> mean_rmse={best['mean_rmse']:.6f}, mean_r2={best['mean_r2']:.4f}")
    log(f"Leaderboard rank-1 (single seed): 0.094549")

    # Report
    lines = [
        f"# H4_S5 {args.model} Hyperparameter Optimization", "",
        f"- Base setting: **S1_T4 Delta+Meta, prefix {args.prefix}%, LOCV (15 cases), {args.model}**",
        f"- Selection: each config run with seeds `{seeds}`; ranked by {seed_label} mean LOCV RMSE.",
        f"- Leader board rank-1 (single seed=0): **RMSE 0.094549**", "",
        "## Best configuration", "",
        f"- **{best['config']}**: hidden={int(best['hidden_size'])}, layers={int(best['num_layers'])}, "
        f"dropout={best['dropout']}, head={int(best['head_hidden'])}, lr={best['lr']}, "
        f"wd={best['weight_decay']}, epochs={int(best['epochs'])}",
        f"- **mean RMSE = {best['mean_rmse']:.6f}** (+/- {best['std_rmse']:.6f}), "
        f"mean R2 = {best['mean_r2']:.4f}, mean MAE = {best['mean_mae']:.6f}", "",
        "## Full ranking", "",
        "| rank | config | hidden | layers | dropout | lr | wd | epochs | mean_rmse | std_rmse | mean_r2 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results.itertuples(index=False):
        lines.append(
            f"| {r.rank} | {r.config} | {int(r.hidden_size)} | {int(r.num_layers)} | {r.dropout} | "
            f"{r.lr} | {r.weight_decay} | {int(r.epochs)} | {r.mean_rmse:.6f} | {r.std_rmse:.6f} | {r.mean_r2:.4f} |"
        )
    (out_dir / "analysis" / "H4_S5_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "logs" / "H4_S5_run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()

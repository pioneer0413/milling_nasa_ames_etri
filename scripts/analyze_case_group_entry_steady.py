#!/usr/bin/env python3
"""Per-case comparison: Entry_Steady(100%) vs Prefix best for two case groups.

Group A: [1, 2, 5, 8, 9, 12, 14, 16]
Group B: [3, 4, 7, 10, 11, 13, 15]

T1 GRU:  ES-100% vs Prefix-80%  (mask=13, AC+vT+vS)
T2 XGB:  ES-100% vs Prefix-90%  (mask=9,  AC+vS)
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
CASE_SCOPE    = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
GROUP_A       = {1, 2, 5, 8, 9, 12, 14, 16}
GROUP_B       = {c for c in CASE_SCOPE if c not in GROUP_A}
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
THRESH        = 1e6
N_SENSORS     = len(SENSORS)
GRU_MASK      = 13   # AC+vT+vS
XGB_MASK      = 9    # AC+vS
GRU_PREFIX    = 80
XGB_PREFIX    = 90
SEG_CSV       = ROOT / "datasets/nasa/cutting_segment_v2/seg_peng2026_steady5_exitfix_reverse_kurtosis.csv"

GRU_CFG = dict(
    hidden_size=256, num_layers=3, dropout=0.1, head_hidden=32,
    lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0,
)
XGB_CFG = dict(
    learning_rate=0.03, n_estimators=600, max_depth=3, min_child_weight=8,
    subsample=0.8, colsample_bytree=0.9, n_jobs=4, verbosity=0,
)


# ─── Data utils ───────────────────────────────────────────────────────────────
def parse_signal(value: object) -> np.ndarray:
    return np.nan_to_num(
        np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


def extract_features(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros(1)
    return np.array(
        [np.mean(arr), np.sqrt(np.mean(arr**2)), np.std(arr), np.abs(arr).max()],
        dtype=np.float64,
    )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["case", "run"]).reset_index(drop=True)
    min_run = df.groupby("case")["run"].transform("min")
    df.loc[(df["run"] == min_run) & df["VB"].isna(), "VB"] = 0.0
    df["VB"] = df.groupby("case")["VB"].transform(
        lambda s: s.interpolate(method="index", limit_area="inside")
    )
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    excl = df.apply(lambda r: (int(r["case"]), int(r["run"])) in EXCLUDED_RUNS, axis=1)
    return df[~excl].reset_index(drop=True)


def mask_sensor_indices(mask: int) -> list[int]:
    return [i * 4 + j for i in range(N_SENSORS) if (mask >> i) & 1 for j in range(4)]


# ─── Dual-mode cache: Entry_Steady(100%) + Prefix(pct%) ───────────────────────
def build_dual_cache(
    signal_df: pd.DataFrame,
    proc_clean: pd.DataFrame,
    seg_idx: dict[tuple[int, int], dict],
    prefix_pct: int,
) -> dict[tuple[int, int, str], np.ndarray]:
    """cache[(case, run, 'es')] and cache[(case, run, 'px')] = 24-dim feature."""
    cache: dict[tuple[int, int, str], np.ndarray] = {}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        sig_row = signal_df[
            (signal_df["case"] == case_id) & (signal_df["run"] == run_id)
        ]
        if sig_row.empty:
            continue
        sr = sig_row.iloc[0]
        arrays = {s: parse_signal(sr[s]) for s in SENSORS}
        if any(np.abs(a).max() > THRESH for a in arrays.values()):
            continue
        base_len = min(len(a) for a in arrays.values())

        # Prefix
        px_end = max(1, math.ceil(base_len * prefix_pct / 100.0))
        cache[(case_id, run_id, "px")] = np.concatenate(
            [extract_features(arrays[s][:px_end]) for s in SENSORS]
        )

        # Entry_Steady (100%)
        seg = seg_idx.get((case_id, run_id))
        if seg is None:
            # fallback: full signal
            cache[(case_id, run_id, "es")] = np.concatenate(
                [extract_features(arrays[s]) for s in SENSORS]
            )
        else:
            es_start = min(int(seg["idx_noload_end"]), base_len)
            es_end   = min(int(seg["idx_end"]),        base_len)
            cache[(case_id, run_id, "es")] = np.concatenate(
                [extract_features(arrays[s][es_start:es_end]) for s in SENSORS]
            )
    return cache


def get_first_run(cache: dict, proc_clean: pd.DataFrame, mode: str) -> dict[int, int]:
    first: dict[int, int] = {}
    for case_id in proc_clean["case"].unique():
        runs = sorted(r for (c, r, m) in cache if c == case_id and m == mode)
        if runs:
            first[int(case_id)] = runs[0]
    return first


# ─── GRU ──────────────────────────────────────────────────────────────────────
def build_gru_sequences(
    cache, first_run, proc_clean, mask, mode
) -> dict[int, dict]:
    si = mask_sensor_indices(mask)
    case_rows: dict[int, list[dict]] = {c: [] for c in CASE_SCOPE}
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id, mode)
        if key not in cache:
            continue
        ref_run = first_run.get(case_id, run_id)
        ref_vec = cache.get((case_id, ref_run, mode), np.zeros(N_SENSORS * 4))
        delta   = (cache[key] - ref_vec)[si]
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        meta = np.array([float(getattr(row, mf, 0.0)) for mf in META_FEATURES], dtype=np.float64)
        feat = np.concatenate([delta, meta]).astype(np.float32)
        vb   = float(row.VB) if not pd.isna(row.VB) else 0.0
        case_rows[case_id].append({"feat": feat, "vb": vb, "run": run_id})
    return {
        c: {
            "seq":    np.stack([r["feat"] for r in sorted(rows, key=lambda r: r["run"])]),
            "vb":     np.array([r["vb"]   for r in sorted(rows, key=lambda r: r["run"])], dtype=np.float32),
            "runs":   np.array([r["run"]  for r in sorted(rows, key=lambda r: r["run"])], dtype=int),
            "n_runs": len(rows),
        }
        for c, rows in case_rows.items() if rows
    }



def fit_predict_gru(train_cases, test_cases, input_dim, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    class _GRU(nn.Module):
        def __init__(self):
            super().__init__()
            cfg = GRU_CFG
            self.gru = nn.GRU(input_dim, cfg["hidden_size"], cfg["num_layers"],
                               batch_first=True,
                               dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0)
            self.head = nn.Sequential(
                nn.Linear(cfg["hidden_size"], cfg["head_hidden"]), nn.ReLU(),
                nn.Linear(cfg["head_hidden"], 1),
            )
        def forward(self, x, lengths):
            pk   = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(pk)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            return self.head(out).squeeze(-1)

    model     = _GRU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_CFG["lr"], weight_decay=GRU_CFG["weight_decay"])
    epochs    = GRU_CFG["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_seqs = [info["seq"] for info in train_cases.values()]
    train_vbs  = [info["vb"]  for info in train_cases.values()]
    all_vb  = np.concatenate(train_vbs)
    y_mean  = float(all_vb.mean())
    y_std   = max(float(all_vb.std()), 1e-8)
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
    preds = {}
    with torch.no_grad():
        for cid, info in test_cases.items():
            x = torch.tensor(info["seq"], dtype=torch.float32).unsqueeze(0).to(device)
            l = torch.tensor([info["n_runs"]])
            preds[cid] = model(x, l).squeeze(0).cpu().numpy() * y_std + y_mean
    return preds


def locv_per_case_gru(all_cases, input_dim, device, seed) -> dict[int, float]:
    per_case: dict[int, float] = {}
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
        per_case[tc] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return per_case


# ─── XGB ──────────────────────────────────────────────────────────────────────
def build_xgb_df(cache, first_run, proc_clean, mask, mode) -> pd.DataFrame:
    si   = mask_sensor_indices(mask)
    rows = []
    for row in proc_clean.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        key = (case_id, run_id, mode)
        if key not in cache:
            continue
        ref_run = first_run.get(case_id, run_id)
        ref_vec = cache.get((case_id, ref_run, mode), np.zeros(N_SENSORS * 4))
        delta   = (cache[key] - ref_vec)[si]
        if not np.all(np.isfinite(delta)):
            delta = np.where(np.isfinite(delta), delta, 0.0)
        vb  = float(row.VB) if not pd.isna(row.VB) else 0.0
        rec = {"case_id": case_id, "run": run_id, "VB": vb}
        for mf in META_FEATURES:
            rec[mf] = float(getattr(row, mf, 0.0))
        for k, v in enumerate(delta):
            rec[f"f{k}"] = float(v)
        rows.append(rec)
    return pd.DataFrame(rows)


def locv_per_case_xgb(feat_df, feature_cols, seed) -> dict[int, float]:
    per_case: dict[int, float] = {}
    for tc in CASE_SCOPE:
        test_df  = feat_df[feat_df["case_id"] == tc]
        train_df = feat_df[feat_df["case_id"] != tc]
        if test_df.empty or train_df.empty:
            continue
        model = XGBRegressor(**XGB_CFG, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_df[feature_cols], train_df["VB"])
            y_pred = model.predict(test_df[feature_cols])
        obs    = test_df.apply(
            lambda r: (int(r["case_id"]), int(r["run"])) not in NON_OBSERVED_RUNS, axis=1
        ).to_numpy()
        y_true = test_df["VB"].to_numpy(float)[obs]
        if len(y_true) == 0:
            continue
        per_case[tc] = float(np.sqrt(mean_squared_error(y_true, y_pred[obs])))
    return per_case


# ─── Main ─────────────────────────────────────────────────────────────────────
def mean_group(per_case: dict[int, float], group: set[int]) -> float:
    vals = [per_case[c] for c in group if c in per_case]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    print("Loading data...", flush=True)
    signal_df  = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv",
                             usecols=["case", "run"] + SENSORS)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df  = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    proc_clean = preprocess(process_df)

    seg_df = pd.read_csv(SEG_CSV)
    seg_df = seg_df[seg_df["case"].isin(CASE_SCOPE) & (seg_df["status"] == "labeled")]
    seg_idx: dict[tuple[int, int], dict] = {
        (int(r.case), int(r.run)): {
            "idx_noload_end": int(r.idx_noload_end),
            "idx_start":      int(r.idx_start),
            "idx_end":        int(r.idx_end),
        }
        for r in seg_df.itertuples(index=False)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── GRU ───────────────────────────────────────────────────────────────────
    gru_input_dim = bin(GRU_MASK).count("1") * 4 + len(META_FEATURES)  # 15

    print(f"\nBuilding GRU cache (prefix={GRU_PREFIX}%)...", flush=True)
    gru_cache = build_dual_cache(signal_df, proc_clean, seg_idx, GRU_PREFIX)
    fr_gru_px = get_first_run(gru_cache, proc_clean, "px")
    fr_gru_es = get_first_run(gru_cache, proc_clean, "es")

    gru_es_per_seed: list[dict[int, float]] = []
    gru_px_per_seed: list[dict[int, float]] = []

    for seed in SEEDS:
        print(f"  GRU seed={seed} ...", flush=True)
        cases_es = build_gru_sequences(gru_cache, fr_gru_es, proc_clean, GRU_MASK, "es")
        cases_px = build_gru_sequences(gru_cache, fr_gru_px, proc_clean, GRU_MASK, "px")
        gru_es_per_seed.append(locv_per_case_gru(cases_es, gru_input_dim, device, seed))
        gru_px_per_seed.append(locv_per_case_gru(cases_px, gru_input_dim, device, seed))

    # 3-seed average per case
    gru_es_avg = {c: float(np.mean([s[c] for s in gru_es_per_seed if c in s]))
                  for c in CASE_SCOPE}
    gru_px_avg = {c: float(np.mean([s[c] for s in gru_px_per_seed if c in s]))
                  for c in CASE_SCOPE}

    # ── XGB ───────────────────────────────────────────────────────────────────
    n_delta      = bin(XGB_MASK).count("1") * 4
    feature_cols = [f"f{k}" for k in range(n_delta)] + META_FEATURES

    print(f"\nBuilding XGB cache (prefix={XGB_PREFIX}%)...", flush=True)
    xgb_cache = build_dual_cache(signal_df, proc_clean, seg_idx, XGB_PREFIX)
    fr_xgb_px = get_first_run(xgb_cache, proc_clean, "px")
    fr_xgb_es = get_first_run(xgb_cache, proc_clean, "es")

    xgb_es_per_seed: list[dict[int, float]] = []
    xgb_px_per_seed: list[dict[int, float]] = []

    for seed in SEEDS:
        print(f"  XGB seed={seed} ...", flush=True)
        df_es = build_xgb_df(xgb_cache, fr_xgb_es, proc_clean, XGB_MASK, "es")
        df_px = build_xgb_df(xgb_cache, fr_xgb_px, proc_clean, XGB_MASK, "px")
        for df in (df_es, df_px):
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0.0, inplace=True)
        xgb_es_per_seed.append(locv_per_case_xgb(df_es, feature_cols, seed))
        xgb_px_per_seed.append(locv_per_case_xgb(df_px, feature_cols, seed))

    xgb_es_avg = {c: float(np.mean([s[c] for s in xgb_es_per_seed if c in s]))
                  for c in CASE_SCOPE}
    xgb_px_avg = {c: float(np.mean([s[c] for s in xgb_px_per_seed if c in s]))
                  for c in CASE_SCOPE}

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("Per-Case RMSE: Entry_Steady(100%) vs Prefix")
    print(f"GRU prefix={GRU_PREFIX}%  |  XGBoost prefix={XGB_PREFIX}%")
    print("=" * 78)
    print(f"{'Case':>5}  {'Grp':>3}  {'GRU_ES':>8}  {'GRU_PX':>8}  {'Δ_GRU':>8}  "
          f"{'XGB_ES':>8}  {'XGB_PX':>8}  {'Δ_XGB':>8}")
    print("-" * 78)
    for c in CASE_SCOPE:
        grp = "A" if c in GROUP_A else "B"
        ge  = gru_es_avg.get(c, float("nan"))
        gp  = gru_px_avg.get(c, float("nan"))
        xe  = xgb_es_avg.get(c, float("nan"))
        xp  = xgb_px_avg.get(c, float("nan"))
        print(f"{c:>5}  {grp:>3}  {ge:>8.4f}  {gp:>8.4f}  {ge-gp:>+8.4f}  "
              f"{xe:>8.4f}  {xp:>8.4f}  {xe-xp:>+8.4f}")

    print("=" * 78)
    print("\nGroup averages:")
    print(f"{'Group':>10}  {'GRU_ES':>8}  {'GRU_PX':>8}  {'Δ_GRU':>8}  "
          f"{'XGB_ES':>8}  {'XGB_PX':>8}  {'Δ_XGB':>8}")
    print("-" * 60)
    for grp_name, grp in [("A", GROUP_A), ("B", GROUP_B), ("All", set(CASE_SCOPE))]:
        ge = mean_group(gru_es_avg, grp)
        gp = mean_group(gru_px_avg, grp)
        xe = mean_group(xgb_es_avg, grp)
        xp = mean_group(xgb_px_avg, grp)
        print(f"{grp_name:>10}  {ge:>8.4f}  {gp:>8.4f}  {ge-gp:>+8.4f}  "
              f"{xe:>8.4f}  {xp:>8.4f}  {xe-xp:>+8.4f}")
    print(f"\nGroup A = {sorted(GROUP_A)}")
    print(f"Group B = {sorted(GROUP_B)}")
    print(f"\nΔ = ES - Prefix  (양수: ES 열위, 음수: ES 우세)")


if __name__ == "__main__":
    main()

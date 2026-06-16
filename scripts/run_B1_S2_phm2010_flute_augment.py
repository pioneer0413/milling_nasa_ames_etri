#!/usr/bin/env python3
"""B1_S2: PHM2010 flute-label augmentation

가설: PHM2010 sequence 모델의 병목은 fold당 학습 궤적이 2개(case 2개)뿐이라는 점.
flute_1/2/3를 각각 개별 "VB" 라벨로 취급하면 데이터가 3배가 된다.
  - run-independent (RF 등): fold당 630 → 1890 샘플
  - sequence (FeatGRU 등):   fold당 2 → 6 궤적

테스트 라벨은 여전히 VB_max = max(flute_1, flute_2, flute_3).

두 가지 증강 방식(Arm)을 비교한다:

  Arm A — 순수 pooling (literal):
    같은 feature vector(70)에 flute_1/2/3를 개별 라벨로 부여. flute-id 없음.
    모델은 세 라벨을 구분할 입력이 없어 사실상 "평균 flute"를 예측 → VB_max 대비 저편향.
    Test: feature 1회 입력 → run당 1예측 → VB_max와 비교.

  Arm B — flute-id 부여 (편향 보정):
    입력 = 70 features + flute one-hot(3) = 73차원. 라벨 = 해당 flute.
    Test: test case를 flute-id=1/2/3로 3번 예측 → run별 element-wise max → VB_max 예측.
    데이터 3배 + VB_max를 직접 타깃 → B1(VB_max 학습) leaderboard와 apples-to-apples.

비교 기준: B1_S1 (VB_max 직접 학습) leaderboard.

Output: experiments/executions/B1/S2/{timestamp}_flute_augment/
"""
from __future__ import annotations

import importlib.util
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Import B1_S1 module (reuse model classes, hyperparams, data loaders) ──────
_spec = importlib.util.spec_from_file_location(
    "b1", ROOT / "scripts" / "run_B1_S1_phm2010_feature_baseline.py"
)
_b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b1)

CASE_SCOPE = _b1.CASE_SCOPE
SEEDS      = _b1.SEEDS                 # [0,1,2,3,4]  → 5-seed
FEAT_COLS  = _b1.FEAT_COLS             # 70 features
FLUTES     = [1, 2, 3]
ARMS       = ["A", "B"]

# B1_S1 reference (VB_max 직접 학습, LOCV-3 5-seed mean RMSE µm)
BASELINE_VBMAX = {
    "Ridge": 40.81, "SVR": 27.20, "RF": 20.35, "MLP_Feat": 70.20,
    "FeatRNN": 23.64, "FeatLSTM": 21.14, "FeatGRU": 22.46, "CF": 2.79,
}


# ─── Augmentation helpers ──────────────────────────────────────────────────────
def _append_onehot(X: np.ndarray, k: int) -> np.ndarray:
    """Append flute one-hot (k∈{1,2,3}) as 3 extra columns (unscaled)."""
    oh = np.zeros((X.shape[0], 3), dtype=X.dtype)
    oh[:, k - 1] = 1.0
    return np.hstack([X, oh])


def _input_dim(arm: str) -> int:
    return len(FEAT_COLS) + (3 if arm == "B" else 0)


# ─── Tabular models (Ridge / SVR / RF / MLP) under augmentation ────────────────
def run_tabular_aug(df, arm, model_fn, device=None, seed=None, torch_mlp=False):
    """LOCV-3 with flute-label augmentation for tabular / MLP models."""
    case_rmses: dict[int, float] = {}
    for tc in CASE_SCOPE:
        tr = df[df["case"] != tc]
        te = df[df["case"] == tc].sort_values("run")
        scaler = StandardScaler().fit(tr[FEAT_COLS].values)

        # ── Augmented training set: 3× rows (one per flute) ──
        X_parts, y_parts = [], []
        for k in FLUTES:
            Xk = scaler.transform(tr[FEAT_COLS].values).astype(np.float32)
            if arm == "B":
                Xk = _append_onehot(Xk, k)
            X_parts.append(Xk)
            y_parts.append(tr[f"VB_flute_{k}"].values.astype(np.float32))
        X_tr = np.vstack(X_parts).astype(np.float32)
        y_tr = np.concatenate(y_parts).astype(np.float32)

        y_te = te["VB_max"].values
        X_te_base = scaler.transform(te[FEAT_COLS].values).astype(np.float32)

        # ── Fit ──
        if torch_mlp:
            pred = _fit_predict_mlp(X_tr, y_tr, X_te_base, arm, device, seed)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = model_fn()
                if arm == "A":
                    m.fit(X_tr, y_tr)
                    pred = m.predict(X_te_base)
                else:  # Arm B: predict each flute, take max
                    m.fit(X_tr, y_tr)
                    preds = [m.predict(_append_onehot(X_te_base, k)) for k in FLUTES]
                    pred = np.max(preds, axis=0)
        pred = np.clip(pred, 0.0, None)
        case_rmses[tc] = float(np.sqrt(mean_squared_error(y_te, pred)))
    return float(np.mean(list(case_rmses.values()))), case_rmses


def _fit_predict_mlp(X_tr, y_tr, X_te_base, arm, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = _b1.FeatMLP(X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=_b1.MLP_LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_b1.MLP_EPOCHS)
    x_t = torch.tensor(X_tr).to(device)
    y_t = torch.tensor(y_tr).to(device)
    model.train()
    for _ in range(_b1.MLP_EPOCHS):
        opt.zero_grad()
        ((model(x_t) - y_t) ** 2).mean().backward()
        opt.step()
        sch.step()
    model.eval()
    with torch.no_grad():
        if arm == "A":
            return model(torch.tensor(X_te_base).to(device)).cpu().numpy()
        preds = []
        for k in FLUTES:
            xk = torch.tensor(_append_onehot(X_te_base, k)).to(device)
            preds.append(model(xk).cpu().numpy())
        return np.max(preds, axis=0)


# ─── Sequence models (RNN / LSTM / GRU) under augmentation ─────────────────────
def _build_seq_aug(df, scaler):
    """Per-case scaled feature sequence + per-flute label sequences + VB_max."""
    cases: dict[int, dict] = {}
    for c in CASE_SCOPE:
        rows = df[df["case"] == c].sort_values("run")
        cases[c] = {
            "feat":   scaler.transform(rows[FEAT_COLS].values).astype(np.float32),
            "flutes": {k: rows[f"VB_flute_{k}"].values.astype(np.float32) for k in FLUTES},
            "vbmax":  rows["VB_max"].values.astype(np.float32),
            "n_runs": len(rows),
        }
    return cases


def _tile_onehot_seq(feat: np.ndarray, k: int) -> np.ndarray:
    """Append flute one-hot broadcast over all timesteps. feat:(n,70)→(n,73)."""
    oh = np.zeros((feat.shape[0], 3), dtype=feat.dtype)
    oh[:, k - 1] = 1.0
    return np.hstack([feat, oh])


def run_feat_seq_aug(df, device, seed, cell, arm):
    """LOCV-3 sequence model with flute-label augmentation (2→6 trajectories)."""
    input_dim = _input_dim(arm)
    case_rmses: dict[int, float] = {}

    for tc in CASE_SCOPE:
        tr_df = df[df["case"] != tc]
        scaler = StandardScaler().fit(tr_df[FEAT_COLS].values)
        all_cases = _build_seq_aug(df, scaler)
        train_cases = [c for c in CASE_SCOPE if c != tc]

        # y-normalization over pooled flute values of training cases
        all_flute_vals = np.concatenate(
            [all_cases[c]["flutes"][k] for c in train_cases for k in FLUTES]
        )
        y_mean = float(all_flute_vals.mean())
        y_std  = max(float(all_flute_vals.std()), 1e-8)

        # ── Build training trajectories: case × flute = 6 sequences ──
        seqs, tgts = [], []
        for c in train_cases:
            feat = all_cases[c]["feat"]
            for k in FLUTES:
                s = _tile_onehot_seq(feat, k) if arm == "B" else feat
                seqs.append(torch.tensor(s))
                y = (all_cases[c]["flutes"][k] - y_mean) / y_std
                tgts.append(torch.tensor(y, dtype=torch.float32))

        lengths = torch.tensor([len(s) for s in seqs])
        x_train = nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(device)
        y_train = nn.utils.rnn.pad_sequence(
            tgts, batch_first=True, padding_value=float("nan")).to(device)
        mask_tr = ~torch.isnan(y_train)

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = _b1.FeatureSeqModel(input_dim, cell=cell).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=_b1.SEQ_LR, weight_decay=_b1.SEQ_WD)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_b1.SEQ_EPOCHS)

        model.train()
        for _ in range(_b1.SEQ_EPOCHS):
            opt.zero_grad()
            pred = model(x_train, lengths)
            loss = ((pred[mask_tr] - y_train[mask_tr]) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), _b1.SEQ_GRAD_CLIP)
            opt.step()
            sch.step()

        # ── Test → VB_max ──
        info = all_cases[tc]
        model.eval()
        with torch.no_grad():
            if arm == "A":
                x_te = torch.tensor(info["feat"]).unsqueeze(0).to(device)
                pred = model(x_te, torch.tensor([info["n_runs"]])).squeeze(0).cpu().numpy()
                y_pred = pred * y_std + y_mean
            else:  # Arm B: predict each flute trajectory, take element-wise max
                flute_preds = []
                for k in FLUTES:
                    xk = torch.tensor(_tile_onehot_seq(info["feat"], k)).unsqueeze(0).to(device)
                    pk = model(xk, torch.tensor([info["n_runs"]])).squeeze(0).cpu().numpy()
                    flute_preds.append(pk * y_std + y_mean)
                y_pred = np.max(flute_preds, axis=0)
        y_pred = np.clip(y_pred, 0.0, None)
        case_rmses[tc] = float(np.sqrt(mean_squared_error(info["vbmax"], y_pred)))

    return float(np.mean(list(case_rmses.values()))), case_rmses


# ─── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experiments" / "executions" / "B1" / "S2" / f"{ts}_flute_augment"
    for sub in ["metrics", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== B1_S2: PHM2010 flute-label augmentation ===")
    log(f"Cases={CASE_SCOPE}, Seeds={SEEDS}, Arms={ARMS}")
    log("Train labels: flute_1/2/3 (3×).  Test label: VB_max.")

    process_df, feat_df = _b1.load_data()
    base = _b1.build_merged(process_df, feat_df)
    # build_merged keeps flute columns + VB_max + FEAT_COLS
    df = base
    log(f"Total rows: {len(df)}  → augmented train rows/fold ≈ {int(len(df)*2/3*3)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    DET = [
        ("Ridge", lambda arm: run_tabular_aug(
            df, arm, lambda: RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]))),
        ("SVR",   lambda arm: run_tabular_aug(
            df, arm, lambda: SVR(kernel="rbf", C=10.0, gamma="scale"))),
    ]
    STO = [
        ("RF",       lambda arm, s: run_tabular_aug(
            df, arm, lambda: RandomForestRegressor(
                n_estimators=200, random_state=s, n_jobs=4), seed=s)),
        ("MLP_Feat", lambda arm, s: run_tabular_aug(
            df, arm, None, device=device, seed=s, torch_mlp=True)),
        ("FeatRNN",  lambda arm, s: run_feat_seq_aug(df, device, s, "rnn",  arm)),
        ("FeatLSTM", lambda arm, s: run_feat_seq_aug(df, device, s, "lstm", arm)),
        ("FeatGRU",  lambda arm, s: run_feat_seq_aug(df, device, s, "gru",  arm)),
    ]
    MODEL_ORDER = ["Ridge", "SVR", "RF", "MLP_Feat", "FeatRNN", "FeatLSTM", "FeatGRU"]

    results: dict[str, dict[str, tuple[float, float, dict]]] = {a: {} for a in ARMS}
    t_total = time_mod.time()

    for arm in ARMS:
        log(f"\n########## ARM {arm} "
            f"({'pure pooling' if arm=='A' else 'flute-id + max'}) ##########")
        for name, fn in DET:
            t0 = time_mod.time()
            m, cr = fn(arm)
            results[arm][name] = (m, 0.0, cr)
            log(f"  {name:<10} {m:>7.2f} µm  [{time_mod.time()-t0:.1f}s]")
        for name, fn in STO:
            t0 = time_mod.time()
            seed_res = [fn(arm, s) for s in SEEDS]
            m, s_, cr = _b1.aggregate_seeds(seed_res)
            results[arm][name] = (m, s_, cr)
            log(f"  {name:<10} {m:>7.2f} µm  ±{s_:.2f}  [{time_mod.time()-t0:.0f}s]")

    # ── Summary: Arm A vs Arm B vs B1_S1 baseline ──
    log("\n=== SUMMARY (LOCV-3 mean RMSE µm) ===")
    log(f"{'Model':<10} {'B1_S1':>8} {'ArmA':>8} {'ArmB':>8} "
        f"{'A-base':>8} {'B-base':>8}")
    log("-" * 56)
    for name in MODEL_ORDER:
        b = BASELINE_VBMAX.get(name, float('nan'))
        a_m = results["A"][name][0]
        b_m = results["B"][name][0]
        log(f"{name:<10} {b:>8.2f} {a_m:>8.2f} {b_m:>8.2f} "
            f"{a_m-b:>+8.2f} {b_m-b:>+8.2f}")
    log(f"\n(CF reference, VB_max oracle, unchanged): {BASELINE_VBMAX['CF']:.2f} µm")

    # ── Save ──
    rows = []
    for arm in ARMS:
        for name in MODEL_ORDER:
            m, s_, cr = results[arm][name]
            rows.append({"arm": arm, "model": name, "mean_um": m, "std_um": s_,
                         **{f"case_{c}": cr.get(c, float('nan')) for c in CASE_SCOPE},
                         "baseline_vbmax": BASELINE_VBMAX.get(name, float('nan'))})
    pd.DataFrame(rows).to_csv(out_dir / "metrics" / "augment_results.csv", index=False)

    summary = {
        "arms": ARMS, "seeds": SEEDS, "model_order": MODEL_ORDER,
        "baseline_vbmax": BASELINE_VBMAX,
        "results": {a: {n: {"mean_um": results[a][n][0], "std_um": results[a][n][1],
                            "per_case": results[a][n][2]} for n in MODEL_ORDER}
                    for a in ARMS},
        "total_elapsed_s": time_mod.time() - t_total,
        "execution_dir": str(out_dir),
    }
    (out_dir / "logs" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "logs" / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    log(f"\nDone. Results in: {out_dir}")
    print(f"EXECUTION_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()

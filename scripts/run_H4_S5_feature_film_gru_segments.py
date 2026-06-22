#!/usr/bin/env python3
"""H4_S5 feature_film_gru segment check: Full vs Entry_Steady.

Uses the same H4_S5 all-sensor Delta+Meta per-timestep GRU frame, changing only
the signal slicing rule:
  Full         : [0, signal_length)
  Entry_Steady : [idx_noload_end, idx_exit_start)

Segment source: datasets/nasa/cutting_segment_index.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_H4_S5_feature_gru_hpo import (  # noqa: E402
    BASE,
    CASE_SCOPE,
    EXCLUDED_RUNS,
    META_FEATURES,
    SENSORS,
    THRESH,
    fit_predict_gru,
    preprocess,
)

SEGMENT_CSV = ROOT / "datasets/nasa/cutting_segment_index.csv"
SEGMENTS = ("Full", "Entry_Steady")
DEFAULT_SENSORS = SENSORS


def parse_signal(value: object) -> np.ndarray:
    arr = np.fromstring(str(value).strip()[1:-1], sep=",", dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        arr = np.zeros(1)
    return np.array(
        [np.mean(arr), np.sqrt(np.mean(arr**2)), np.std(arr), np.abs(arr).max()],
        dtype=np.float64,
    )


def delta_col_names(sensors: list[str]) -> list[str]:
    return [f"{sensor}__delta_{name}" for sensor in sensors for name in ["mean", "rms", "std", "peak"]]


def segment_bounds(segment: str, seg: dict[str, int], base_len: int) -> tuple[int, int]:
    if segment == "Full":
        return 0, base_len
    if segment == "Entry_Steady":
        start = min(max(0, int(seg["idx_noload_end"])), base_len)
        end = min(max(0, int(seg["idx_exit_start"])), base_len)
        return start, max(start + 1, end)
    raise ValueError(f"Unknown segment: {segment}")


def build_segment_feature_matrix(
    signal_df: pd.DataFrame,
    process_df: pd.DataFrame,
    segment_index_df: pd.DataFrame,
    sensors: list[str],
) -> pd.DataFrame:
    delta_cols = delta_col_names(sensors)
    seg_lookup = {
        (int(row.case), int(row.run)): {
            "idx_noload_end": int(row.idx_noload_end),
            "idx_exit_start": int(row.idx_exit_start),
        }
        for row in segment_index_df.itertuples(index=False)
    }

    cache: dict[tuple[int, int, str], np.ndarray] = {}
    for row in process_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        sig_row = signal_df[(signal_df["case"] == case_id) & (signal_df["run"] == run_id)]
        if sig_row.empty:
            continue
        sig_row = sig_row.iloc[0]
        arrays = {sensor: parse_signal(sig_row[sensor]) for sensor in sensors}
        if any(np.abs(arr).max() > THRESH for arr in arrays.values()):
            continue
        base_len = min(len(arr) for arr in arrays.values())
        seg = seg_lookup.get((case_id, run_id))

        for segment in SEGMENTS:
            if segment == "Full" or seg is None:
                start, end = 0, base_len
            else:
                start, end = segment_bounds(segment, seg, base_len)
            cache[(case_id, run_id, segment)] = np.concatenate(
                [extract_features(arrays[sensor][start:end]) for sensor in sensors]
            )

    first_run: dict[tuple[int, str], int] = {}
    for case_id in process_df["case"].unique():
        for segment in SEGMENTS:
            runs = sorted(r for (c, r, s) in cache if c == int(case_id) and s == segment)
            if runs:
                first_run[(int(case_id), segment)] = runs[0]

    rows: list[dict] = []
    for row in process_df.itertuples(index=False):
        case_id, run_id = int(row.case), int(row.run)
        if (case_id, run_id) in EXCLUDED_RUNS:
            continue
        vb = float(row.VB) if not pd.isna(row.VB) else 0.0
        record_base = {
            "case_id": case_id,
            "run": run_id,
            "VB": vb,
            "DOC": float(getattr(row, "DOC", 0.0)),
            "feed": float(getattr(row, "feed", 0.0)),
            "material": int(getattr(row, "material", 0)),
        }
        for segment in SEGMENTS:
            raw_feats = cache.get((case_id, run_id, segment))
            ref_run = first_run.get((case_id, segment), run_id)
            ref_feats = cache.get((case_id, ref_run, segment))
            if raw_feats is None or ref_feats is None:
                continue
            row_record = {**record_base, "segment": segment}
            for col, val in zip(delta_cols, raw_feats - ref_feats):
                row_record[col] = float(val)
            rows.append(row_record)

    return pd.DataFrame(rows).sort_values(["segment", "case_id", "run"]).reset_index(drop=True)


def build_sequences(feature_df: pd.DataFrame, delta_cols: list[str], segment: str) -> dict[int, dict]:
    subset = feature_df[feature_df["segment"] == segment].copy()
    cases: dict[int, dict] = {}
    for case_id in subset["case_id"].unique():
        case_df = subset[subset["case_id"] == case_id].sort_values("run").reset_index(drop=True)
        delta = case_df[delta_cols].to_numpy(dtype=np.float32)
        meta = case_df[META_FEATURES].to_numpy(dtype=np.float32)
        cases[int(case_id)] = {
            "seq": np.concatenate([delta, meta], axis=1),
            "vb": case_df["VB"].to_numpy(dtype=np.float32),
            "runs": case_df["run"].to_numpy(dtype=int),
            "n_runs": len(case_df),
        }
    return cases


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def run_locv_seed(
    all_case_data: dict[int, dict],
    input_dim: int,
    device: torch.device,
    cfg: dict,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for test_case in CASE_SCOPE:
        if test_case not in all_case_data:
            continue
        train_data = {c: all_case_data[c] for c in CASE_SCOPE if c != test_case and c in all_case_data}
        preds = fit_predict_gru(train_data, {test_case: all_case_data[test_case]}, input_dim, device, cfg, seed)
        y_true = all_case_data[test_case]["vb"]
        y_pred = preds[test_case]
        rows.append({"target_case": test_case, "n_runs": len(y_true), **compute_metrics(y_true, y_pred)})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H4_S5 feature_film_gru Full vs Entry_Steady segment check.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--segments", nargs="+", choices=SEGMENTS, default=list(SEGMENTS))
    parser.add_argument("--sensors", nargs="+", default=DEFAULT_SENSORS)
    parser.add_argument("--output-root", default="experiments/executions")
    parser.add_argument("--film-hidden-dim", type=int, default=None)
    parser.add_argument("--film-dropout", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / args.output_root / "H4" / "S5" / f"{timestamp}_feature_film_gru_segments"
    for subdir in ["configs", "metrics", "analysis", "logs"]:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        log_lines.append(line)

    cfg = {
        **BASE,
        "name": "baseline_rank1",
        "model_type": "feature_film_gru",
        "use_metadata_film": True,
        "metadata_feature_dim": len(META_FEATURES),
        "film_hidden_dim": args.film_hidden_dim,
        "film_dropout": float(args.film_dropout),
    }

    log("=== H4_S5 feature_film_gru segment check: Full vs Entry_Steady ===")
    sensors = list(args.sensors)
    unknown_sensors = sorted(set(sensors) - set(DEFAULT_SENSORS))
    if unknown_sensors:
        raise ValueError(f"Unknown sensors: {unknown_sensors}. Supported: {DEFAULT_SENSORS}")
    log(f"Seeds={args.seeds}; segments={args.segments}; sensors={sensors}")
    signal_df = pd.read_csv(ROOT / "datasets/nasa/raw_signal.csv", usecols=["case", "run"] + sensors)
    process_df = pd.read_csv(ROOT / "datasets/nasa/process_info.csv")
    segment_index_df = pd.read_csv(SEGMENT_CSV)
    process_df = process_df[process_df["case"].isin(CASE_SCOPE)].copy()
    signal_df = signal_df[signal_df["case"].isin(CASE_SCOPE)].copy()
    segment_index_df = segment_index_df[segment_index_df["case"].isin(CASE_SCOPE)].copy()
    process_clean = preprocess(process_df)

    log("Building segment Delta+Meta feature matrix...")
    feature_df = build_segment_feature_matrix(signal_df, process_clean, segment_index_df, sensors)
    delta_cols = delta_col_names(sensors)
    input_dim = len(delta_cols) + len(META_FEATURES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Feature rows={len(feature_df)}; input_dim={input_dim}; device={device}")

    segment_rows: list[dict] = []
    seed_rows: list[dict] = []
    per_case_rows: list[dict] = []
    t0 = time.time()
    for segment in args.segments:
        all_case_data = build_sequences(feature_df, delta_cols, segment)
        log(f"\nSegment={segment}; cases={len(all_case_data)}")
        rmse_values: list[float] = []
        r2_values: list[float] = []
        mae_values: list[float] = []
        for seed in args.seeds:
            fold_df = run_locv_seed(all_case_data, input_dim, device, cfg, seed)
            locv_rmse = float(fold_df["rmse"].mean())
            locv_r2 = float(fold_df["r2"].mean())
            locv_mae = float(fold_df["mae"].mean())
            rmse_values.append(locv_rmse)
            r2_values.append(locv_r2)
            mae_values.append(locv_mae)
            seed_rows.append(
                {"segment": segment, "seed": seed, "locv_rmse": locv_rmse, "locv_r2": locv_r2, "locv_mae": locv_mae}
            )
            fold_df.insert(0, "segment", segment)
            fold_df.insert(1, "seed", seed)
            per_case_rows.extend(fold_df.to_dict("records"))
            log(f"  seed={seed}: rmse={locv_rmse:.6f}, r2={locv_r2:.4f}, mae={locv_mae:.6f}")
        summary = {
            "segment": segment,
            "mean_rmse": float(np.mean(rmse_values)),
            "std_rmse": float(np.std(rmse_values)),
            "mean_r2": float(np.mean(r2_values)),
            "mean_mae": float(np.mean(mae_values)),
            "n_seeds": len(args.seeds),
            "n_cases": len(all_case_data),
        }
        segment_rows.append(summary)
        log(
            f"Segment={segment} summary: mean_rmse={summary['mean_rmse']:.6f} "
            f"(+/-{summary['std_rmse']:.6f}), mean_r2={summary['mean_r2']:.4f}"
        )

    summary_df = pd.DataFrame(segment_rows).sort_values("mean_rmse").reset_index(drop=True)
    summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))
    pd.DataFrame(seed_rows).to_csv(out_dir / "metrics" / "H4_S5_feature_film_gru_segment_seed_metrics.csv", index=False)
    pd.DataFrame(per_case_rows).to_csv(out_dir / "metrics" / "H4_S5_feature_film_gru_segment_per_case_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "analysis" / "H4_S5_feature_film_gru_segment_summary.csv", index=False)

    config = {
        "experiment": "H4_S5_feature_film_gru_segments",
        "segment_source": str(SEGMENT_CSV.relative_to(ROOT)),
        "segments": {
            "Full": "[0, signal_length)",
            "Entry_Steady": "[idx_noload_end, idx_exit_start)",
        },
        "case_scope": CASE_SCOPE,
        "excluded_runs": sorted([list(item) for item in EXCLUDED_RUNS]),
        "sensors": sensors,
        "metadata_features": META_FEATURES,
        "model_config": cfg,
        "seeds": args.seeds,
        "elapsed_seconds": time.time() - t0,
        "ranking": summary_df.to_dict("records"),
    }
    (out_dir / "configs" / "H4_S5_feature_film_gru_segment_config.json").write_text(
        json.dumps(config, indent=2, default=float),
        encoding="utf-8",
    )
    (out_dir / "logs" / "H4_S5_feature_film_gru_segment_log.txt").write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )

    log("\n=== SUMMARY ===")
    log(summary_df.to_string(index=False))
    log(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SENSOR_COLUMNS = [
    "force_x",
    "force_y",
    "force_z",
    "vib_x",
    "vib_y",
    "vib_z",
    "ae_rms",
]


def parse_run_id(path: Path) -> int:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected run filename: {path.name}")
    return int(parts[-1])


def load_wear_map(wear_path: Path) -> dict[int, dict[str, float]]:
    wear = pd.read_csv(wear_path)
    expected = {"cut", "flute_1", "flute_2", "flute_3"}
    missing = expected.difference(wear.columns)
    if missing:
        raise ValueError(f"Wear file missing columns {sorted(missing)}: {wear_path}")
    wear_map: dict[int, dict[str, float]] = {}
    for row in wear.itertuples(index=False):
        wear_map[int(row.cut)] = {
            "VB_flute_1": float(row.flute_1),
            "VB_flute_2": float(row.flute_2),
            "VB_flute_3": float(row.flute_3),
        }
    return wear_map


def read_signal_csv(path: Path) -> dict[str, list[float]]:
    data = pd.read_csv(path, header=None)
    if data.shape[1] != len(SENSOR_COLUMNS):
        raise ValueError(f"Expected {len(SENSOR_COLUMNS)} columns, got {data.shape[1]} in {path}")
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    columns = {}
    for idx, name in enumerate(SENSOR_COLUMNS):
        columns[name] = data.iloc[:, idx].astype(float).tolist()
    return columns


def build_rows(raw_root: Path, cutters: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_rows: list[dict[str, Any]] = []
    process_rows: list[dict[str, Any]] = []
    heuristic_rows: list[dict[str, Any]] = []

    for cutter in cutters:
        cutter_id = int(cutter.lstrip("c"))
        wear_path = raw_root / cutter / f"{cutter}_wear.csv"
        wear_map = load_wear_map(wear_path)
        signal_dir = raw_root / cutter / cutter
        run_files = sorted(signal_dir.glob("*.csv"))
        if not run_files:
            raise FileNotFoundError(f"No signal files found in {signal_dir}")

        for run_file in run_files:
            run_id = parse_run_id(run_file)
            signals = read_signal_csv(run_file)
            signal_length = len(next(iter(signals.values())))
            wear = wear_map.get(run_id)
            if wear is None:
                raise ValueError(f"Wear label missing for {run_file.name} (cut {run_id})")

            signal_rows.append(
                {
                    "case": cutter_id,
                    "run": run_id,
                    "enable": True,
                    **{name: json.dumps(signals[name]) for name in SENSOR_COLUMNS},
                }
            )

            process_rows.append(
                {
                    "case": cutter_id,
                    "run": run_id,
                    "VB": float("nan"),
                    **wear,
                    "time": float(run_id),
                    "DOC": float("nan"),
                    "feed": float("nan"),
                    "material": float("nan"),
                    "material_name": "phm2010",
                    "enable": True,
                }
            )

            heuristic_rows.append(
                {
                    "case": cutter_id,
                    "run": run_id,
                    "idx_noload_end": 0,
                    "idx_start": 0,
                    "idx_end": signal_length,
                    "signal_length": signal_length,
                    "status": "full_length_only",
                }
            )

    signal_df = pd.DataFrame(signal_rows).sort_values(["case", "run"]).reset_index(drop=True)
    process_df = pd.DataFrame(process_rows).sort_values(["case", "run"]).reset_index(drop=True)
    heuristic_df = pd.DataFrame(heuristic_rows).sort_values(["case", "run"]).reset_index(drop=True)
    return signal_df, process_df, heuristic_df


def write_outputs(output_root: Path, signal_df: pd.DataFrame, process_df: pd.DataFrame, heuristic_df: pd.DataFrame) -> None:
    processed_dir = output_root / "datasets" / "processed"
    metadata_dir = output_root / "datasets" / "metadata"
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    signal_path = processed_dir / "phm2010_signal_data_enabled.csv"
    process_path = processed_dir / "phm2010_process_info_enabled.csv"
    heuristic_path = metadata_dir / "phm2010_heuristic_sequence.csv"

    signal_df.to_csv(signal_path, index=False)
    process_df.to_csv(process_path, index=False)
    heuristic_df.to_csv(heuristic_path, index=False)


def summarize(signal_df: pd.DataFrame, process_df: pd.DataFrame, heuristic_df: pd.DataFrame) -> None:
    print("PHM2010 preprocessing summary")
    print(f"signal rows: {len(signal_df)}")
    print(f"process rows: {len(process_df)}")
    print(f"heuristic rows: {len(heuristic_df)}")
    print("cases:", sorted(signal_df["case"].unique().tolist()))
    print("runs per case:", signal_df.groupby("case")["run"].nunique().to_dict())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PHM2010 dataset for the milling framework.")
    parser.add_argument("--raw-root", default="datasets/raw/phm2010")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--cutters", nargs="+", default=["c1", "c4", "c6"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    signal_df, process_df, heuristic_df = build_rows(raw_root, args.cutters)
    write_outputs(output_root, signal_df, process_df, heuristic_df)
    summarize(signal_df, process_df, heuristic_df)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.analysis.feature_quality import run_feature_quality_analysis
from milling_experiment_framework.experiments.h2_s4_process_information_execution import PROCESS_FACTORS
from milling_experiment_framework.experiments.s1_segment_execution import (
    DOMAIN_CASES,
    FEATURE_NAMES,
    S1RunConfig,
    S1SegmentExecution,
)


PAIR_BY_CASE = {case: pair for pair, cases in DOMAIN_CASES.items() for case in cases}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H3.S0 feature quality diagnostic analysis for VB prediction.")
    parser.add_argument("--config", default="configs/experiments/H2_S4_process_information_combination_effect_on_segment_aware_VB_prediction.yaml")
    parser.add_argument("--output-root", default="experiments/executions")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    root = Path.cwd()
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_H3_S0_feature_quality_analysis_for_VB_prediction")
    output_dir = root / args.output_root / experiment_id
    for dirname in ["configs", "data", "analysis", "figures", "reports", "logs"]:
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs" / "H3_S0_run.log"
    error_path = output_dir / "logs" / "H3_S0_error.log"

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")
        print(message)

    try:
        log(f"H3.S0 execution started: {experiment_id}")
        config_path = Path(args.config)
        run_config = _resolve_run_config(config_path)
        helper = S1SegmentExecution(config_path, root=root, dry_run=True, seed_mode="initial")
        dataset = helper._load_dataset(run_config)
        sensors = helper._sensor_columns(dataset)
        feature_table = helper._build_feature_table(dataset, sensors)
        wide = build_wide_feature_table(dataset, feature_table, sensors)
        target_col = "VB"
        metadata_cols = ["sample_id", "dataset_run_id", "case_id", "run", "pair_id", "domain_id"]
        feature_cols = [c for c in wide.columns if c not in set(metadata_cols + [target_col])]
        log(
            "Dry-run passed: "
            f"samples={len(wide)}, sensors={sensors}, features={len(feature_cols)}, "
            f"target={target_col}, metadata_cols={metadata_cols}"
        )
        (output_dir / "data" / "H3_S0_feature_matrix_preview.csv").write_text(
            wide.head(20).to_csv(index=False),
            encoding="utf-8",
        )
        summary = {
            "experiment_id": experiment_id,
            "source_config": str(config_path),
            "samples": int(len(wide)),
            "features": int(len(feature_cols)),
            "target_col": target_col,
            "metadata_cols": metadata_cols,
            "sensors": sensors,
            "process_features": ["DoC", "Feed", "Material", "Time"],
            "sensor_feature_names": FEATURE_NAMES,
        }
        (output_dir / "data" / "H3_S0_feature_matrix_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        result = run_feature_quality_analysis(
            wide,
            target_col=target_col,
            feature_cols=feature_cols,
            output_dir=str(output_dir),
            metadata_cols=metadata_cols,
            experiment_id=experiment_id,
            top_n=args.top_n,
            random_state=args.random_state,
        )
        log(f"H3.S0 execution finished: {experiment_id}")
        print(json.dumps({"experiment_id": experiment_id, "execution_dir": str(output_dir), "result": result}, indent=2, ensure_ascii=False, default=str))
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _resolve_run_config(config_path: Path) -> S1RunConfig:
    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    dataset_cfg = config["dataset"]
    return S1RunConfig(
        process_info_path=Path(dataset_cfg["process_info_path"]),
        signal_data_path=Path(dataset_cfg["signal_data_path"]),
        heuristic_sequence_path=Path(dataset_cfg["heuristic_sequence_path"]),
        seeds=[],
        models=[],
        random_forest_params={},
        mlp_params={},
    )


def build_wide_feature_table(dataset: pd.DataFrame, feature_table: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    sample_cols = ["sample_id", "dataset_run_id", "case", "run", "domain_id", "VB"]
    process_cols = ["DOC", "feed", "material_name", "time"]
    base = dataset[sample_cols + process_cols].drop_duplicates("sample_id").copy()
    base = base.rename(columns={"case": "case_id", "DOC": "DoC", "feed": "Feed", "material_name": "Material", "time": "Time"})
    base["pair_id"] = base["case_id"].map(PAIR_BY_CASE)
    base = base[["sample_id", "dataset_run_id", "case_id", "run", "pair_id", "domain_id", "VB", "DoC", "Feed", "Material", "Time"]]
    wide = base.copy()
    for segment in sorted(feature_table["segment_setting"].unique()):
        segment_rows = feature_table.loc[feature_table["segment_setting"] == segment]
        for sensor in sensors:
            rows = segment_rows.loc[segment_rows["sensor"] == sensor, ["sample_id"] + FEATURE_NAMES].copy()
            rows = rows.rename(columns={feature: f"{sensor}__{segment}__{feature}" for feature in FEATURE_NAMES})
            wide = wide.merge(rows, on="sample_id", how="inner", validate="one_to_one")
    return wide


if __name__ == "__main__":
    main()

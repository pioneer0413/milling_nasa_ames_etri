#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_H1_S3_association_suitability_segment_cross_analysis import (  # noqa: E402
    CASE_SCOPE,
    PAIR_DEFINITION,
    SCHEMA_VERSION,
    SEGMENT_SETTINGS,
    TARGET_FEATURES,
    collect_environment,
    create_execution_dir,
    find_or_create_entry_steady_exit_features,
    identify_sensors,
    load_dataset,
    stable_hash,
    write_json,
    write_yaml_like,
)


PREFIX = "H1_S3"
TREND_METHOD = "centered_rolling_mean"
TREND_WINDOW = 3
DENOMINATOR_EPSILON = 1e-12


def rolling_average_trend(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype="float64")
    )


def robustness_for_sequence(values: np.ndarray, window: int, epsilon: float) -> dict[str, Any]:
    values = np.asarray(values, dtype="float64")
    finite = np.isfinite(values)
    values = values[finite]
    if len(values) < 2:
        return {
            "robustness": np.nan,
            "sequence_length": int(len(values)),
            "status": "skipped_n_less_than_2",
            "mean_abs_relative_deviation": np.nan,
            "median_abs_relative_deviation": np.nan,
            "zero_denominator_count": int(np.sum(np.abs(values) < epsilon)),
        }
    trend = rolling_average_trend(values, window)
    denominator = np.maximum(np.abs(values), epsilon)
    relative_deviation = np.abs((values - trend) / denominator)
    scores = np.exp(-relative_deviation)
    return {
        "robustness": float(np.mean(scores)),
        "sequence_length": int(len(values)),
        "status": "ok",
        "mean_abs_relative_deviation": float(np.mean(relative_deviation)),
        "median_abs_relative_deviation": float(np.median(relative_deviation)),
        "zero_denominator_count": int(np.sum(np.abs(values) < epsilon)),
    }


def compute_case_sensor_robustness(feature_long: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "case_id",
        "pair_id",
        "sensor_name",
        "sensor_group",
        "segment_setting",
        "feature_name",
        "feature_group",
    ]
    for keys, group in feature_long.groupby(group_cols, dropna=False):
        case_id, pair_id, sensor_name, sensor_group, segment_setting, feature_name, feature_group = keys
        group = group.sort_values(["run", "sample_id"], kind="mergesort")
        values = group["feature_value"].to_numpy(dtype="float64")
        result = robustness_for_sequence(values, TREND_WINDOW, DENOMINATOR_EPSILON)
        rows.append(
            {
                "experiment_id": experiment_id,
                "case_id": int(case_id),
                "pair_id": pair_id,
                "sensor_name": sensor_name,
                "sensor_group": sensor_group,
                "segment_setting": segment_setting,
                "feature_name": feature_name,
                "feature_group": feature_group,
                "robustness": result["robustness"],
                "mean_abs_relative_deviation": result["mean_abs_relative_deviation"],
                "median_abs_relative_deviation": result["median_abs_relative_deviation"],
                "sequence_length": result["sequence_length"],
                "zero_denominator_count": result["zero_denominator_count"],
                "calculation_status": result["status"],
                "trend_method": TREND_METHOD,
                "trend_window": TREND_WINDOW,
                "denominator_epsilon": DENOMINATOR_EPSILON,
            }
        )
    return pd.DataFrame(rows)


def aggregate_segment_feature(case_sensor: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    ok = case_sensor.loc[case_sensor["calculation_status"].eq("ok")].copy()
    out = (
        ok.groupby(["segment_setting", "feature_name", "feature_group"], dropna=False)
        .agg(
            mean_robustness=("robustness", "mean"),
            median_robustness=("robustness", "median"),
            std_robustness=("robustness", "std"),
            min_robustness=("robustness", "min"),
            max_robustness=("robustness", "max"),
            mean_abs_relative_deviation=("mean_abs_relative_deviation", "mean"),
            num_case_sensor_sequences=("robustness", "size"),
            num_cases=("case_id", "nunique"),
            num_sensors=("sensor_name", "nunique"),
            total_points=("sequence_length", "sum"),
            zero_denominator_count=("zero_denominator_count", "sum"),
        )
        .reset_index()
    )
    out["experiment_id"] = experiment_id
    out["std_robustness"] = out["std_robustness"].fillna(0.0)
    out["rank_by_mean_robustness"] = out["mean_robustness"].rank(ascending=False, method="first").astype(int)
    out["trend_method"] = TREND_METHOD
    out["trend_window"] = TREND_WINDOW
    out["denominator_epsilon"] = DENOMINATOR_EPSILON
    return out.sort_values("rank_by_mean_robustness").reset_index(drop=True)


def aggregate_sensor_feature_segment(case_sensor: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    ok = case_sensor.loc[case_sensor["calculation_status"].eq("ok")].copy()
    out = (
        ok.groupby(["sensor_name", "sensor_group", "segment_setting", "feature_name", "feature_group"], dropna=False)
        .agg(
            mean_robustness=("robustness", "mean"),
            median_robustness=("robustness", "median"),
            std_robustness=("robustness", "std"),
            min_robustness=("robustness", "min"),
            max_robustness=("robustness", "max"),
            mean_abs_relative_deviation=("mean_abs_relative_deviation", "mean"),
            num_cases=("case_id", "nunique"),
            total_points=("sequence_length", "sum"),
            zero_denominator_count=("zero_denominator_count", "sum"),
        )
        .reset_index()
    )
    out["experiment_id"] = experiment_id
    out["std_robustness"] = out["std_robustness"].fillna(0.0)
    out["rank_by_mean_robustness"] = out["mean_robustness"].rank(ascending=False, method="first").astype(int)
    return out.sort_values("rank_by_mean_robustness").reset_index(drop=True)


def aggregate_pair_segment_feature(case_sensor: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    ok = case_sensor.loc[case_sensor["calculation_status"].eq("ok")].copy()
    out = (
        ok.groupby(["pair_id", "segment_setting", "feature_name", "feature_group"], dropna=False)
        .agg(
            mean_robustness=("robustness", "mean"),
            median_robustness=("robustness", "median"),
            std_robustness=("robustness", "std"),
            num_case_sensor_sequences=("robustness", "size"),
            num_cases=("case_id", "nunique"),
            num_sensors=("sensor_name", "nunique"),
            total_points=("sequence_length", "sum"),
        )
        .reset_index()
    )
    out["experiment_id"] = experiment_id
    out["std_robustness"] = out["std_robustness"].fillna(0.0)
    out["rank_within_pair"] = out.groupby("pair_id")["mean_robustness"].rank(ascending=False, method="first").astype(int)
    return out.sort_values(["pair_id", "rank_within_pair"]).reset_index(drop=True)


def write_heatmap(segment_feature: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    matrix = (
        segment_feature.pivot_table(
            index="feature_name",
            columns="segment_setting",
            values="mean_robustness",
            aggfunc="mean",
        )
        .reindex(index=TARGET_FEATURES, columns=SEGMENT_SETTINGS)
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.imshow(matrix.to_numpy(dtype="float64"), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value > 0.55 else "black", fontsize=8)
    ax.set_title("H1_S3 Segment x Feature Robustness")
    fig.colorbar(image, ax=ax, label="Mean robustness")
    plt.tight_layout()
    plt.savefig(figure_dir / f"{PREFIX}_segment_feature_robustness_heatmap.png", dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    out = frame.head(max_rows).copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    header = "| " + " | ".join(map(str, out.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in out.astype(str).to_numpy()]
    return "\n".join([header, separator, *rows])


def write_report(output_dir: Path, context: dict[str, Any], segment_feature: pd.DataFrame, sensor_level: pd.DataFrame) -> None:
    top_segment_feature = segment_feature.sort_values("mean_robustness", ascending=False).head(12)
    top_sensor = sensor_level.sort_values("mean_robustness", ascending=False).head(12)
    text = f"""# H1_S3 robustness_segment_feature_analysis

## Summary

Robustness was computed for each case-level feature sequence and then averaged to segment x feature level.

- Heuristic segment file: `{context['heuristic_sequence_path']}`
- Trend estimate x_k^T: `{TREND_METHOD}`, window `{TREND_WINDOW}`, within each case/run sequence
- Denominator guard: `max(abs(x_k), {DENOMINATOR_EPSILON})`
- Segment settings: `{SEGMENT_SETTINGS}`
- Features: `{TARGET_FEATURES}`
- Case scope: `{CASE_SCOPE}`
- Sensors: `{context['sensors']}`
- no-load segment rows: `{context['no_load_rows']}`
- full_length includes no-load: `true`

## Formula

`Rob(X) = mean(exp(-abs((x_k - x_k^T) / x_k)))`

Because the paper does not define the average trend estimator, this run uses centered rolling mean as the explicit estimator. When `x_k` is numerically zero, the denominator is guarded with epsilon and the count is reported in the output tables.

## Top Segment x Feature

{markdown_table(top_segment_feature[['segment_setting', 'feature_name', 'mean_robustness', 'median_robustness', 'std_robustness', 'num_case_sensor_sequences', 'rank_by_mean_robustness']], 12)}

## Top Sensor x Segment x Feature

{markdown_table(top_sensor[['sensor_name', 'sensor_group', 'segment_setting', 'feature_name', 'mean_robustness', 'std_robustness', 'num_cases', 'rank_by_mean_robustness']], 12)}

## Outputs

- `analysis/{PREFIX}_robustness_segment_feature.csv`
- `analysis/{PREFIX}_robustness_case_sensor_feature_segment.csv`
- `analysis/{PREFIX}_robustness_sensor_feature_segment.csv`
- `analysis/{PREFIX}_robustness_pair_segment_feature.csv`
- `figures/{PREFIX}_segment_feature_robustness_heatmap.png`
"""
    report_path = output_dir / "reports" / f"{PREFIX}_robustness_report.md"
    report_path.write_text(text, encoding="utf-8")
    html_text = "<!doctype html><html><head><meta charset='utf-8'><title>H1_S3 Robustness</title></head><body>" + html.escape(text).replace("\n", "<br>\n") + "</body></html>"
    (output_dir / "reports" / f"{PREFIX}_robustness_report.html").write_text(html_text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    experiment_id = f"{timestamp}_H1_S3_robustness_segment_feature_analysis"
    path_config = {
        "experiment": {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "hypothesis_id": "H1",
            "scenario_id": "S3",
            "experiment_topic": "robustness_segment_feature_analysis",
        }
    }
    output_dir = Path(create_execution_dir(path_config, root=root / "experiments" / "executions"))
    for dirname in ["configs", "data", "analysis", "figures", "reports", "logs"]:
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)
    error_log = output_dir / "logs" / f"{PREFIX}_robustness_error.log"
    error_log.touch()
    run_log = output_dir / "logs" / f"{PREFIX}_robustness_run.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line)
        with run_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    try:
        process_info_path = root / args.process_info_path
        signal_data_path = root / args.signal_data_path
        heuristic_sequence_path = root / args.heuristic_sequence_path
        log(f"started {experiment_id}")
        data = load_dataset(process_info_path, signal_data_path, heuristic_sequence_path)
        sensors_df = identify_sensors(data)
        sensors = sensors_df.loc[sensors_df["used"], "sensor_name"].tolist()
        feature_long, segment_construction = find_or_create_entry_steady_exit_features(data, sensors)

        no_load_rows = int(feature_long["segment_setting"].astype(str).str.contains("no_load|no-load", case=False, na=False).sum())
        case_sensor = compute_case_sensor_robustness(feature_long, experiment_id)
        segment_feature = aggregate_segment_feature(case_sensor, experiment_id)
        sensor_level = aggregate_sensor_feature_segment(case_sensor, experiment_id)
        pair_level = aggregate_pair_segment_feature(case_sensor, experiment_id)

        validation = {
            "ok": bool(
                len(segment_feature) == len(SEGMENT_SETTINGS) * len(TARGET_FEATURES)
                and case_sensor["calculation_status"].eq("ok").all()
                and segment_feature["mean_robustness"].between(0, 1).all()
            ),
            "segment_feature_rows": int(len(segment_feature)),
            "expected_segment_feature_rows": int(len(SEGMENT_SETTINGS) * len(TARGET_FEATURES)),
            "case_sensor_rows": int(len(case_sensor)),
            "status_counts": case_sensor["calculation_status"].value_counts().to_dict(),
            "robustness_in_range": bool(segment_feature["mean_robustness"].between(0, 1).all()),
            "no_load_rows": no_load_rows,
            "full_length_includes_no_load": True,
            "trend_method": TREND_METHOD,
            "trend_window": TREND_WINDOW,
            "denominator_epsilon": DENOMINATOR_EPSILON,
        }
        if not validation["ok"]:
            raise ValueError(f"Robustness validation failed: {validation}")

        config = {
            "experiment": path_config["experiment"],
            "process_info_path": str(process_info_path),
            "signal_data_path": str(signal_data_path),
            "heuristic_sequence_path": str(heuristic_sequence_path),
            "case_scope": CASE_SCOPE,
            "pair_definition": PAIR_DEFINITION,
            "segments": SEGMENT_SETTINGS,
            "features": TARGET_FEATURES,
            "robustness": {
                "formula": "mean(exp(-abs((x_k - x_k_T) / x_k)))",
                "trend_method": TREND_METHOD,
                "trend_window": TREND_WINDOW,
                "denominator_epsilon": DENOMINATOR_EPSILON,
            },
            "schema_version": SCHEMA_VERSION,
        }
        config["config_hash"] = stable_hash(config)

        write_yaml_like(output_dir / "configs" / f"{PREFIX}_robustness_input_config.yaml", config)
        write_yaml_like(output_dir / "configs" / f"{PREFIX}_robustness_resolved_config.yaml", {**config, "validation": validation})
        write_json(output_dir / "configs" / f"{PREFIX}_robustness_config_validation.json", validation)
        write_json(
            output_dir / "data" / f"{PREFIX}_robustness_dataset_summary.json",
            {
                "rows": int(len(data)),
                "case_scope": CASE_SCOPE,
                "rows_by_case": data.groupby("case_id").size().to_dict(),
                "sensors": sensors,
                "num_feature_long_rows": int(len(feature_long)),
                "heuristic_sequence_path": str(heuristic_sequence_path),
                "segment_construction": segment_construction,
            },
        )
        sensors_df.to_csv(output_dir / "data" / f"{PREFIX}_robustness_sensor_mapping.csv", index=False)

        case_sensor.to_csv(output_dir / "analysis" / f"{PREFIX}_robustness_case_sensor_feature_segment.csv", index=False)
        segment_feature.to_csv(output_dir / "analysis" / f"{PREFIX}_robustness_segment_feature.csv", index=False)
        sensor_level.to_csv(output_dir / "analysis" / f"{PREFIX}_robustness_sensor_feature_segment.csv", index=False)
        pair_level.to_csv(output_dir / "analysis" / f"{PREFIX}_robustness_pair_segment_feature.csv", index=False)

        summary = {
            "experiment_id": experiment_id,
            "execution_dir": str(output_dir),
            "top_segment_feature": segment_feature.head(10).to_dict(orient="records"),
            "top_sensor_feature_segment": sensor_level.head(10).to_dict(orient="records"),
            "validation": validation,
        }
        write_json(output_dir / "analysis" / f"{PREFIX}_robustness_analysis_summary.json", summary)
        write_json(output_dir / "logs" / f"{PREFIX}_robustness_environment.json", collect_environment(str(root)))
        write_heatmap(segment_feature, output_dir / "figures")
        write_report(
            output_dir,
            {
                "heuristic_sequence_path": str(heuristic_sequence_path),
                "sensors": sensors,
                "no_load_rows": no_load_rows,
            },
            segment_feature,
            sensor_level,
        )
        log(f"finished {experiment_id}")
        return summary
    except Exception:
        error_log.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H1_S3 robustness over segment x feature.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--process-info-path", default="datasets/processed/mill_process_info_enabled.csv")
    parser.add_argument("--signal-data-path", default="datasets/processed/mill_signal_data_enabled.csv")
    parser.add_argument("--heuristic-sequence-path", default="datasets/metadata/heuristic_sequence_peng2026.csv")
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

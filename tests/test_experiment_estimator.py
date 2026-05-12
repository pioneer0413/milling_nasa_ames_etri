from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from milling_experiment_framework.experiments.estimator import estimate_experiment_from_config


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_h2_s2_style_grid_atomic_count():
    with tempfile.TemporaryDirectory() as tmp:
        result = estimate_experiment_from_config(str(FIXTURES / "estimator_h2_s2.yaml"), output_dir=tmp)
        assert result["total_atomic_executions"] == 11760


def test_h2_s4_conditional_grid_count():
    with tempfile.TemporaryDirectory() as tmp:
        result = estimate_experiment_from_config(str(FIXTURES / "estimator_h2_s4.yaml"), output_dir=tmp)
        assert result["total_atomic_executions"] == 30480
        summary = pd.read_csv(result["atomic_count_summary_path"])
        full = summary[summary["phase"] == "full"]
        by_group = full.groupby("condition_group")["atomic_executions"].sum().to_dict()
        assert by_group["sensor_only"] == 1680
        assert by_group["process_only"] == 3600
        assert by_group["sensor_plus_process"] == 25200


def test_cumulative_s_dnn_subrun_count():
    with tempfile.TemporaryDirectory() as tmp:
        result = estimate_experiment_from_config(str(FIXTURES / "estimator_cumulative_s_dnn.yaml"), output_dir=tmp)
        assert result["total_atomic_executions"] == 882
        assert result["total_sub_runs"] == 18522
        summary = pd.read_csv(result["atomic_count_summary_path"])
        full = summary[summary["phase"] == "full"].iloc[0]
        assert full["sub_runs_per_atomic"] == 21


def test_runtime_override_and_warnings_and_reduced_grid():
    with tempfile.TemporaryDirectory() as tmp:
        result = estimate_experiment_from_config(
            str(FIXTURES / "estimator_h2_s2.yaml"),
            output_dir=tmp,
            runtime_overrides={"__default__": 10},
        )
        runtime = pd.read_csv(result["runtime_estimate_path"])
        full_hours = runtime[runtime["phase"] == "full"]["estimated_total_hours"].sum()
        assert round(full_hours, 6) == round(11760 * 10 / 3600, 6)
        assert result["warnings"]
        assert result["reduced_grid_suggestion"]["estimated_atomic_executions"] < result["total_atomic_executions"]


def test_output_files_created_and_compare_inputs_can_be_estimated():
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        first = estimate_experiment_from_config(str(FIXTURES / "estimator_h2_s2.yaml"), output_dir=tmp1)
        second = estimate_experiment_from_config(str(FIXTURES / "estimator_cumulative_s_dnn.yaml"), output_dir=tmp2)
        for key in ["atomic_count_summary_path", "runtime_estimate_path", "report_path"]:
            assert Path(first[key]).exists()
            assert Path(second[key]).exists()
        assert second["total_sub_runs"] > second["total_atomic_executions"]


from __future__ import annotations

from pathlib import Path

from milling_experiment_framework.experiments.execution_path import (
    build_execution_dir,
    find_experiment_dirs,
    parse_experiment_identifier,
)


def test_parse_legacy_timestamp_first_name():
    parsed = parse_experiment_identifier(
        "2026-05-06_140501_964643_H2_S1_all_models_all_sensors_all_segments_all_shifts_seeds_0_19_combined"
    )
    assert parsed["hypothesis_id"] == "H2"
    assert parsed["scenario_id"] == "S1"
    assert parsed["timestamp"] == "2026-05-06_140501_964643"
    assert parsed["experiment_topic"] == "all_models_all_sensors_all_segments_all_shifts_seeds_0_19_combined"


def test_parse_compact_timestamp_name():
    parsed = parse_experiment_identifier("20260507_161435_H1_S3_keep_feature_subset_VB_prediction")
    assert parsed["hypothesis_id"] == "H1"
    assert parsed["scenario_id"] == "S3"
    assert parsed["timestamp"] == "20260507_161435"
    assert parsed["experiment_topic"] == "keep_feature_subset_VB_prediction"


def test_parse_hs_prefix_name():
    parsed = parse_experiment_identifier("H1_S4_feature_segment_suitability_analysis")
    assert parsed["hypothesis_id"] == "H1"
    assert parsed["scenario_id"] == "S4"
    assert parsed["timestamp"] is None
    assert parsed["experiment_topic"] == "feature_segment_suitability_analysis"


def test_build_execution_dir():
    path = build_execution_dir(
        "experiments/executions",
        "H2",
        "S1",
        "2026-05-06_140501_964643",
        "all_models_all_sensors_all_segments_all_shifts_seeds_0_19_combined",
    )
    assert path == (
        "experiments/executions/H2/S1/"
        "2026-05-06_140501_964643_all_models_all_sensors_all_segments_all_shifts_seeds_0_19_combined"
    )


def test_legacy_and_new_search(tmp_path: Path):
    root = tmp_path / "experiments" / "executions"
    legacy = root / "2026-05-06_140501_964643_H2_S1_xxx"
    new = root / "H2" / "S1" / "2026-05-06_140501_964643_xxx"
    legacy.mkdir(parents=True)
    new.mkdir(parents=True)

    found = find_experiment_dirs(root, keyword="H2_S1")
    assert str(legacy) in found
    assert str(new) in found

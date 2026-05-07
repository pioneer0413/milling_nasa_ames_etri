# Changelog

## v0.1.0 - 2026-05-06

- Added MVP manufacturing experiment framework with config validation, example dataset loader, random and dataset-run-wise splits, leakage checks, train-only normalization, windowing, RandomForest, CNN1D, evaluator, CLI scripts, experiment index, smoke outputs, and patch note.
- Added fixed case-pair domain-shift test execution for enabled milling processed data with six A/B/C source-target scenarios and `shift_metrics.csv` aggregation.
- Added S1 segment-setting effect execution over sensors, seven segment settings, RandomForest/MLP, six domain shifts, and seed merging for 20-seed reports.
- Added H1.S1 no-load-excluded segment feature to VB association EDA with Pearson, Spearman, mutual information, baseline comparison, sensor consistency, pair-level summaries, and figures.
- Added S2 segment-aware VB prediction execution for current/vibration/acoustic sensor group combinations with 20-seed batch merging, sensor-combination effect analysis, segment-effect analysis, R2-positive summaries, figures, and reports.
- Added H2.S3 all-sensors segment-aware VB prediction execution for statistics/shape/frequency feature group combinations with 20-seed batch merging, feature contribution analysis, segment-effect analysis, R2-positive summaries, figures, and reports.
- Added H2.S4 all-sensors segment-aware VB prediction execution for process information combinations, train-only numeric/categorical process preprocessing, process addition effects, process factor contribution, R2-positive summaries, prefixed outputs, figures, and reports.

# Virtual environment Usage
path: `/home/hwkang/manufacturing_data_exercise/02_milling_v2/venv312`

# CLI Usage

Run a single experiment:

```bash
python scripts/run_experiment.py --config configs/experiments/exp001_cnn1d_smoke.yaml
```

Dry-run:

```bash
python scripts/run_experiment.py --config configs/experiments/exp001_cnn1d_smoke.yaml --dry-run
```

Smoke test:

```bash
python scripts/smoke_test.py --config configs/experiments/exp001_cnn1d_smoke.yaml
```

Feature-based RandomForest example:

```bash
python scripts/run_experiment.py --config configs/experiments/exp002_random_forest_features.yaml
```

Fixed six-scenario case-pair domain-shift execution:

```bash
python scripts/run_domain_shift_experiment.py --config configs/experiments/exp003_guideline_domain_shift_rf.yaml
```

S1 segment-setting effect execution:

```bash
python scripts/run_s1_experiment.py --config configs/experiments/S1_segment_setting_effect_on_VB_prediction.yaml --dry-run --seed-mode initial
python scripts/run_s1_experiment.py --config configs/experiments/S1_segment_setting_effect_on_VB_prediction.yaml --seed-mode initial
python scripts/run_s1_experiment.py --config configs/experiments/S1_segment_setting_effect_on_VB_prediction.yaml --seed-mode extra
python scripts/merge_s1_executions.py --config configs/experiments/S1_segment_setting_effect_on_VB_prediction.yaml --execution-dirs <initial_execution_dir> <extra_execution_dir>
```

H1.S1 segment feature to VB association EDA:

```bash
python scripts/run_h1_s1_eda.py --config configs/experiments/H1_S1_segment_feature_VB_association_no_noload.yaml --dry-run
python scripts/run_h1_s1_eda.py --config configs/experiments/H1_S1_segment_feature_VB_association_no_noload.yaml
```

## Experiment Execution Guidelines

These guidelines define the intended execution contract for the milling domain-shift experiments. Use `scripts/run_domain_shift_experiment.py` to run the fixed six-scenario protocol in one execution.

### 1. Case Scope

Only the following cases are used in this experiment:

- Case 1
- Case 2
- Case 8
- Case 9
- Case 12
- Case 14

The enabled case data is available from:

- `datasets/processed/mill_process_info_enabled.csv`
- `datasets/processed/mill_signal_data_enabled.csv`

During experiment execution, these files are the reference dataset. Unless explicitly requested, cases outside `1`, `2`, `8`, `9`, `12`, and `14` must not be used for training, validation, or evaluation.

### 2. Case Pair Definition

Individual cases are not used directly as source or target domains. Instead, each case pair is treated as one domain group:

- Pair A: Case 1, Case 9
- Pair B: Case 2, Case 12
- Pair C: Case 8, Case 14

Domain definitions:

```text
Domain A = {Case 1, Case 9}
Domain B = {Case 2, Case 12}
Domain C = {Case 8, Case 14}
```

Each pair is handled as a single domain.

### 3. Domain Shift Scenarios

Domain shift is defined by changing the source and target domain across case pairs.

The experiment must run all six domain-shift scenarios:

1. A to B
2. A to C
3. B to A
4. B to C
5. C to A
6. C to B

### 4. Train/Test Policy

Each domain-shift scenario follows these rules:

- Only cases included in the source domain are used for training.
- Only cases included in the target domain are used for evaluation.
- Source-domain cases and target-domain cases must not overlap.
- Target-domain labels are used only for evaluation.
- Unless explicitly requested, target-domain data must not be used for training.
- If a validation set is needed, it must be split only from the source domain.
- For source-domain validation, use the smaller case number as validation unless otherwise specified.
- Target-domain data must not be used for validation or hyperparameter tuning.

### 5. Metric Aggregation

The final performance of one experiment setting is calculated by averaging the results from all six domain-shift scenarios.

For one experiment setting, all six results must be produced:

```text
final_metric =
mean([
  metric(A -> B),
  metric(A -> C),
  metric(B -> A),
  metric(B -> C),
  metric(C -> A),
  metric(C -> B)
])
```

For classification tasks, the same averaging rule applies independently to each metric, such as `accuracy` and `f1_macro`.

For regression tasks, the same averaging rule applies independently to each metric, such as `mae`, `rmse`, and `r2`.

### 6. Result Storage

Each domain-shift scenario result must be stored separately, and the final averaged result must also be stored separately.

Recommended output structure:

```text
experiments/executions/{experiment_id}/metrics/
├── shift_metrics.csv
├── metrics.json
└── domain_metrics.csv
```

`shift_metrics.csv` must include at least:

```text
experiment_id
source_domain
target_domain
source_cases
target_cases
metric_name
metric_value
```

`metrics.json` stores the average over the six shift scenarios as the final performance:

```json
{
  "aggregation": "mean_over_6_domain_shifts",
  "shift_scenarios": ["A_to_B", "A_to_C", "B_to_A", "B_to_C", "C_to_A", "C_to_B"],
  "final_metrics": {
    "rmse": 0.89,
    "r2": 0.87,
    "mae": 0.55,
  }
}
```

### 7. Execution Notes

- Cases outside `1`, `2`, `8`, `9`, `12`, and `14` must be excluded.
- Pair A, B, and C definitions are fixed for all experiments.
- All six domain-shift scenarios must be executed.
- If any shift scenario fails, do not calculate the final average; clearly report the failed scenario.
- Target-domain data must not be used for anything except evaluation.
- Validation, normalization fitting, feature selection, and hyperparameter tuning must use source-domain data only.
- The final report must include both per-shift metrics and averaged metrics.

## S2 Sensor Combination Experiment

Run the no-load-excluded, segment-aware VB prediction experiment with sensor group combinations:

```bash
python scripts/run_s2_experiment.py \
  --config configs/experiments/S2_sensor_combination_effect_on_segment_aware_VB_prediction.yaml \
  --dry-run \
  --seed-mode initial
```

After dry-run validation succeeds, run the initial and extra seed batches:

```bash
python scripts/run_s2_experiment.py \
  --config configs/experiments/S2_sensor_combination_effect_on_segment_aware_VB_prediction.yaml \
  --seed-mode initial

python scripts/run_s2_experiment.py \
  --config configs/experiments/S2_sensor_combination_effect_on_segment_aware_VB_prediction.yaml \
  --seed-mode extra
```

Merge both batches into one 20-seed result:

```bash
python scripts/merge_s2_executions.py \
  --config configs/experiments/S2_sensor_combination_effect_on_segment_aware_VB_prediction.yaml \
  --execution-dirs \
  experiments/executions/{initial_execution_id} \
  experiments/executions/{extra_execution_id}
```

The S2 execution writes `sensor_group_mapping.csv`, `shift_metrics.csv`, `seed_metrics.csv`, `sensor_combination_metrics.csv`, sensor-combination effect analysis, segment-effect analysis, figures, and `report.md`.

## H2.S3 Feature Combination Experiment

Run the all-sensors, no-load-excluded, segment-aware VB prediction experiment over feature group combinations:

```bash
python scripts/run_h2_s3_experiment.py \
  --config configs/experiments/H2_S3_feature_combination_effect_on_segment_aware_VB_prediction_all_sensors.yaml \
  --dry-run \
  --seed-mode initial
```

After dry-run validation succeeds, run the initial and extra seed batches:

```bash
python scripts/run_h2_s3_experiment.py \
  --config configs/experiments/H2_S3_feature_combination_effect_on_segment_aware_VB_prediction_all_sensors.yaml \
  --seed-mode initial

python scripts/run_h2_s3_experiment.py \
  --config configs/experiments/H2_S3_feature_combination_effect_on_segment_aware_VB_prediction_all_sensors.yaml \
  --seed-mode extra
```

Merge both batches into one 20-seed result:

```bash
python scripts/merge_h2_s3_executions.py \
  --config configs/experiments/H2_S3_feature_combination_effect_on_segment_aware_VB_prediction_all_sensors.yaml \
  --execution-dirs \
  experiments/executions/{initial_execution_id} \
  experiments/executions/{extra_execution_id}
```

The H2.S3 execution fixes `sensor_setting=all_sensors` and writes feature-combination metrics, feature group contribution analysis, segment-effect analysis, figures, and `report.md`.

## H2.S4 Process Information Experiment

Run the all-sensors, no-load-excluded, segment-aware VB prediction experiment over process information combinations:

```bash
python scripts/run_h2_s4_experiment.py \
  --config configs/experiments/H2_S4_process_information_combination_effect_on_segment_aware_VB_prediction.yaml \
  --dry-run \
  --seed-mode initial
```

After dry-run validation succeeds, run the initial and extra seed batches:

```bash
python scripts/run_h2_s4_experiment.py \
  --config configs/experiments/H2_S4_process_information_combination_effect_on_segment_aware_VB_prediction.yaml \
  --seed-mode initial

python scripts/run_h2_s4_experiment.py \
  --config configs/experiments/H2_S4_process_information_combination_effect_on_segment_aware_VB_prediction.yaml \
  --seed-mode extra
```

Merge both batches into one 20-seed result:

```bash
python scripts/merge_h2_s4_executions.py \
  --config configs/experiments/H2_S4_process_information_combination_effect_on_segment_aware_VB_prediction.yaml \
  --execution-dirs \
  experiments/executions/{initial_execution_id} \
  experiments/executions/{extra_execution_id}
```

The H2.S4 execution writes prefixed outputs such as `H2_S4_shift_metrics.csv`, `H2_S4_process_information_mapping.csv`, process-effect analysis, process-factor contribution analysis, figures, and `H2_S4_report.md`.

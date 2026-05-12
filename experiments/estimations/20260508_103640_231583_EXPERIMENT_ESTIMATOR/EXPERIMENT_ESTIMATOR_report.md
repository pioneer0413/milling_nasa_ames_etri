# Experiment Estimator Report

## 1. Executive Summary

- experiment_name: `estimator_cumulative_s_dnn`
- total atomic executions: `882`
- total sub-runs: `18522`
- estimated wall-clock hours: `147.000`
- warning level: `CRITICAL`
- recommended execution plan: `reduced` first

## 2. Grid Summary

- models: 1
- seeds: 3
- shifts: 6
- sensor_combinations: 7
- segment_settings: 7
- feature_combinations: 1
- process_combinations: 1
- input_representations: 1
- modes: 1
- base_learners: 4
- sequence_modes: 1
- hyperparameter_grid: 1

## 3. Phase-wise Estimate

- full / default / cumulative_s_dnn: 882 atomic, 18522 sub-runs, 147.000 h
- initial / default / cumulative_s_dnn: 882 atomic, 18522 sub-runs, 147.000 h
- extra_seed / default / cumulative_s_dnn: 294 atomic, 6174 sub-runs, 49.000 h
- smoke / default / cumulative_s_dnn: 1 atomic, 21 sub-runs, 0.167 h
- reduced / default / cumulative_s_dnn: 216 atomic, 4536 sub-runs, 36.000 h
- optional_ablation / default / cumulative_s_dnn: 882 atomic, 18522 sub-runs, 147.000 h

## 4. Model-specific Estimate

- cumulative_s_dnn: config_runtime_per_atomic (estimation.runtime_per_atomic_sec.cumulative_s_dnn)
- cumulative_s_dnn: config_runtime_per_atomic (estimation.runtime_per_atomic_sec.cumulative_s_dnn)
- cumulative_s_dnn: config_runtime_per_atomic (estimation.runtime_per_atomic_sec.cumulative_s_dnn)
- cumulative_s_dnn: config_runtime_per_atomic (estimation.runtime_per_atomic_sec.cumulative_s_dnn)
- cumulative_s_dnn: config_runtime_per_atomic (estimation.runtime_per_atomic_sec.cumulative_s_dnn)
- cumulative_s_dnn: config_runtime_per_atomic (estimation.runtime_per_atomic_sec.cumulative_s_dnn)

## 5. Risk Warnings

- CRITICAL `RUNTIME_HOURS`: estimated runtime is high: 147.0 Suggested: Use reduced grid or fewer seeds.
- WARNING `DISK_USAGE`: estimated disk usage 185309.4 MB exceeds threshold Suggested: Reduce checkpoints or max_keep.
- HIGH `STACKING_SUBRUNS`: stacking/OOF model has many sub-runs per atomic execution Suggested: Profile one condition and run reduced grid first.

## 6. Reduced Grid Recommendation

- reason: large grid or cautious initial execution
- estimated_atomic_executions: 216
- estimated_runtime_hours: 36.0
- recommended_grid: `{'models': ['cumulative_s_dnn'], 'seeds': [0, 1, 2], 'shifts': ['A_to_B', 'A_to_C', 'B_to_A', 'B_to_C', 'C_to_A', 'C_to_B'], 'sensor_combinations': ['current', 'acoustic', 'current_acoustic', 'current_vibration_acoustic'], 'segment_settings': ['full_length', 'steady', 'entry_exit'], 'feature_combinations': ['default_feature'], 'process_combinations': ['none'], 'input_representations': ['cumulative'], 'modes': ['s_dnn'], 'base_learners': ['rnn', 'gru', 'lstm', 'cnn'], 'sequence_modes': ['segment_mode'], 'hyperparameter_grid': ['default_hparam']}`

## 7. Assumptions

- Runtime defaults are heuristic unless config/user overrides are provided.
- GPU/CPU hours are approximated from model type and requested device assumptions.
- Disk usage uses checkpoint and row-count heuristics.

## 8. CLI Command Examples

```bash
python scripts/estimate_experiment.py --config <config.yaml>
python scripts/estimate_experiment.py --config <config.yaml> --runtime-per-atomic-sec 30
python scripts/estimate_experiment.py --config H2_S2.yaml --compare-config H3_S1.yaml
```

# Experiment Estimator Report

## 1. Executive Summary

- experiment_name: `cumulative_s_dnn`
- total atomic executions: `2`
- total sub-runs: `42`
- estimated wall-clock hours: `0.333`
- warning level: `HIGH`
- recommended execution plan: `reduced` first

## 2. Grid Summary

- models: 1
- seeds: 1
- shifts: 1
- sensor_combinations: 1
- segment_settings: 1
- feature_combinations: 1
- process_combinations: 1
- input_representations: 1
- modes: 2
- base_learners: 4
- sequence_modes: 1
- hyperparameter_grid: 1

## 3. Phase-wise Estimate

- full / default / cumulative_s_dnn: 1 atomic, 21 sub-runs, 0.167 h
- full / default / cumulative_s_dnn: 1 atomic, 21 sub-runs, 0.167 h
- initial / default / cumulative_s_dnn: 1 atomic, 21 sub-runs, 0.167 h
- initial / default / cumulative_s_dnn: 1 atomic, 21 sub-runs, 0.167 h
- extra_seed / default / cumulative_s_dnn: 1 atomic, 21 sub-runs, 0.167 h
- extra_seed / default / cumulative_s_dnn: 1 atomic, 21 sub-runs, 0.167 h
- smoke / default / cumulative_s_dnn: 1 atomic, 21 sub-runs, 0.167 h
- reduced / default / cumulative_s_dnn: 1 atomic, 21 sub-runs, 0.167 h
- reduced / default / cumulative_s_dnn: 1 atomic, 21 sub-runs, 0.167 h
- optional_ablation / default / cumulative_s_dnn: 1 atomic, 21 sub-runs, 0.167 h
- optional_ablation / default / cumulative_s_dnn: 1 atomic, 21 sub-runs, 0.167 h

## 4. Model-specific Estimate

- cumulative_s_dnn: heuristic_default (built-in heuristic)
- cumulative_s_dnn: heuristic_default (built-in heuristic)
- cumulative_s_dnn: heuristic_default (built-in heuristic)
- cumulative_s_dnn: heuristic_default (built-in heuristic)
- cumulative_s_dnn: heuristic_default (built-in heuristic)
- cumulative_s_dnn: heuristic_default (built-in heuristic)
- cumulative_s_dnn: heuristic_default (built-in heuristic)
- cumulative_s_dnn: heuristic_default (built-in heuristic)
- cumulative_s_dnn: heuristic_default (built-in heuristic)
- cumulative_s_dnn: heuristic_default (built-in heuristic)
- cumulative_s_dnn: heuristic_default (built-in heuristic)

## 5. Risk Warnings

- HIGH `STACKING_SUBRUNS`: stacking/OOF model has many sub-runs per atomic execution Suggested: Profile one condition and run reduced grid first.

## 6. Reduced Grid Recommendation

- reason: large grid or cautious initial execution
- estimated_atomic_executions: 2
- estimated_runtime_hours: 0.3333333333333333
- recommended_grid: `{'models': ['cumulative_s_dnn'], 'seeds': [0], 'shifts': ['A_to_B'], 'sensor_combinations': ['all_sensors'], 'segment_settings': ['full_length'], 'feature_combinations': ['default_feature'], 'process_combinations': ['none'], 'input_representations': ['cumulative'], 'modes': ['base_only', 's_dnn'], 'base_learners': ['rnn', 'gru', 'lstm', 'cnn'], 'sequence_modes': ['paper_style_sliding_window'], 'hyperparameter_grid': ['default_hparam']}`

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

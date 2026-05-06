# Architecture

This repository implements the MVP of the milling experiment framework described in `design_specification.md`.

The Python package is `milling_experiment_framework`. Runtime experiment outputs are written under `experiments/executions/{experiment_id}` so that config, split, preprocessing, model, metrics, predictions, logs, reports, and checkpoints are grouped by `experiment_id`.

The package uses registries for extension points:

- `DATASET_REGISTRY`
- `PREPROCESSOR_REGISTRY`
- `SPLIT_REGISTRY`
- `MODEL_REGISTRY`
- `FEATURE_REGISTRY`

The framework intentionally avoids a top-level Python package named `logging` to prevent shadowing the Python standard library. Logging-related framework modules live under `milling_experiment_framework/experiment_logging`.

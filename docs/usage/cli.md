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

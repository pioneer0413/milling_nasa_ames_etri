```
python scripts/estimate_experiment.py \
  --config configs/experiments/example.yaml

python scripts/estimate_experiment.py \
  --config tests/fixtures/estimator_h2_s2.yaml \
  --output experiments/estimations/

python scripts/estimate_experiment.py \
  --config tests/fixtures/estimator_h2_s2.yaml \
  --compare-config tests/fixtures/estimator_cumulative_s_dnn.yaml

python scripts/run_experiment.py \
  --config configs/experiments/example.yaml \
  --estimate-only
```
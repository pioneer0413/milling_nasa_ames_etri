# H2 Model Catalog

This document records the model set prepared for H2/S1, H2/S2, H2/S3, and H2/S4 VB regression experiments. It documents the runnable model names, estimator classes, pipeline structure, and default parameters used by the H2 execution code.

## Scope

- H2/S1: segment-setting effect on VB prediction.
- H2/S2: sensor-combination effect on segment-aware VB prediction.
- H2/S3: feature-combination effect on segment-aware VB prediction.
- H2/S4: process-information combination effect on segment-aware VB prediction.

The full experiments are not executed by this preparation step. The models are wired into the existing H2 executors and can be selected through each H2 experiment config's `models` list.

## Runnable Model Names

| model name | family | estimator | random seed handling |
| --- | --- | --- | --- |
| `random_forest` | Feature-based ML | `sklearn.ensemble.RandomForestRegressor` | `random_state=seed` |
| `xgboost` | Feature-based ML | `xgboost.XGBRegressor` | `random_state=seed` |
| `svr` | Feature-based ML | `sklearn.svm.SVR` | deterministic for fixed data and params |
| `linear_regression` | Feature-based ML | `sklearn.linear_model.LinearRegression` | deterministic |
| `mlp` | Neural Network | `sklearn.neural_network.MLPRegressor` | `random_state=seed` |

Supported aliases inside the H2 helper are `rf`, `randomforest`, `xgb`, `xgb_regressor`, `support_vector_regression`, and `linear`.

## Pipeline Structure

### H2/S1, H2/S2, H2/S3

These experiments use numeric run-level feature tables. Every model uses the same feature pipeline:

```text
StandardScaler(fit on train split only)
-> estimator
```

The scaler is intentionally kept for every model, including tree-based models, so all H2 models share one input contract and SVR/MLP receive scaled numeric inputs.

### H2/S4

H2/S4 can combine sensor features with process columns. Its pipeline is:

```text
ColumnTransformer(
  numeric: SimpleImputer(strategy="median") -> StandardScaler(),
  categorical: SimpleImputer(strategy="most_frequent") -> OneHotEncoder(handle_unknown="ignore")
)
-> estimator
```

The `ColumnTransformer` uses `sparse_threshold=0.0` so the estimator receives dense tabular arrays. This keeps MLP, SVR, linear regression, random forest, and XGBoost on the same H2/S4 input path.

## Default Parameters

The defaults live in `milling_experiment_framework/models/h2_regressors.py` and are overridden by each experiment config's `model_defaults` when provided.

### Random Forest

```yaml
random_forest:
  n_estimators: 100
  n_jobs: -1
```

At runtime, `random_state` is set to the experiment seed.

### XGBoost

```yaml
xgboost:
  n_estimators: 200
  max_depth: 3
  learning_rate: 0.05
  subsample: 0.9
  colsample_bytree: 0.9
  objective: reg:squarederror
  tree_method: hist
  n_jobs: -1
```

At runtime, `random_state` is set to the experiment seed.

### SVR

```yaml
svr:
  kernel: rbf
  C: 10.0
  epsilon: 0.03
  gamma: scale
```

### Linear Regression

```yaml
linear_regression: {}
```

This uses scikit-learn defaults for `LinearRegression`.

### MLP

```yaml
mlp:
  hidden_layer_sizes: [64, 32, 16]
  max_iter: 300
  learning_rate_init: 0.001
```

At runtime, `random_state` is set to the experiment seed.

## Config Files Updated

The following configs now list all five models and their parameter defaults:

- `configs/experiments/H2_S1_segment_setting_effect_on_VB_prediction.yaml`
- `configs/experiments/H2_S1_segment_setting_effect_on_VB_prediction_peng2026.yaml`
- `configs/experiments/H2_S2_sensor_combination_effect_on_segment_aware_VB_prediction.yaml`
- `configs/experiments/H2_S2_sensor_combination_effect_on_segment_aware_VB_prediction_peng2026.yaml`
- `configs/experiments/H2_S3_feature_combination_effect_on_segment_aware_VB_prediction_all_sensors.yaml`
- `configs/experiments/H2_S3_feature_combination_effect_on_segment_aware_VB_prediction_all_sensors_peng2026.yaml`
- `configs/experiments/H2_S4_process_information_combination_effect_on_segment_aware_VB_prediction.yaml`
- `configs/experiments/H2_S4_process_information_combination_effect_on_segment_aware_VB_prediction_peng2026.yaml`

## Execution Note

Dry-run mode still only validates data/config/planned grid shape and does not train the full model grid. Use a small smoke test to check model construction, fitting, and prediction without launching H2/S1-S4 full experiments.

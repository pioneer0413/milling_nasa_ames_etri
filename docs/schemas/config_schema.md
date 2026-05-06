# Config Schema

MVP required top-level keys:

- `experiment`
- `dataset`
- `task`
- `preprocessing`
- `split`
- `model`
- `training`
- `evaluation`

Compatibility rules:

- `preprocessing.output_type: features` requires `model.input_type: feature-based`
- `preprocessing.output_type: timeseries` requires `model.input_type: timeseries-based`
- `preprocessing.output_type: hybrid` requires `model.input_type: hybrid`

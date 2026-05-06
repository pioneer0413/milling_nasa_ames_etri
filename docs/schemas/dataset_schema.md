# Dataset Schema

Required metadata columns:

- `sample_id`
- `label`
- `dataset_run_id`
- `sequence_index` or `timestamp`

The internal timeseries shape is `[num_samples, num_channels, sequence_length]`.

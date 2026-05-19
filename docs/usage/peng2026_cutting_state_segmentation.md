# peng2026-Inspired Cutting State Segmentation

This implementation extends Figure 8's unsupervised steady-state extraction idea from
“Frequency-aware and bionic-aligned collaborative modeling for cross-domain tool wear
monitoring under small-sample conditions” into four milling process states:

- `no_load`
- `entry`
- `steady`
- `exit`

It is not a full-length mode and it is not sliding-window inference. It first detects
low-energy no-load samples, then detects the minimum-fluctuation steady region, and
derives entry/exit from those two boundaries.

## Paper-Specified Values

- `xi_values = [2, 2.5, 3, 3.5, 4, 4.5, 5]`
- `target_length = 4000` as the default steady extraction length

`target_length` remains configurable.

## Implementation Hyperparameters

The paper describes the conceptual procedure but does not provide fixed values for
the following parameters. They are implementation choices and should be tuned or
recorded per experiment:

- `window_size`
- `hop_size`
- `idle_delta` or `idle_delta_ratio`
- `statistic`: `rms`, `mean_abs`, `std`, or `energy`
- `reference_mode`: `middle`, `whole`, or `local`
- `middle_window_mode`
- `min_steady_length`
- `candidate_merge_gap`
- `detect_tail_idle`
- `padding_mode`
- `target_length`

## Usage

```python
from milling_experiment_framework.preprocessing.segmentation import CuttingStateSegmenter

segmenter = CuttingStateSegmenter(
    window_size=64,
    hop_size=1,
    target_length=4000,
    idle_delta_ratio=0.35,
    statistic="rms",
    reference_mode="middle",
    min_steady_length=4000,
    candidate_merge_gap=32,
    detect_tail_idle=True,
)

result = segmenter.segment(signal)
segmenter.plot(signal, result)
```

The result contains absolute raw-signal indices:

```python
{
    "segments": {
        "no_load": (0, cutting_start_idx),
        "entry": (cutting_start_idx, steady_start_idx),
        "steady": (steady_start_idx, steady_end_idx),
        "exit": (steady_end_idx, cutting_end_idx),
    },
    "signals": {...},
    "diagnostics": {...},
}
```

Tail no-load, when detected, is reported in `diagnostics["tail_no_load"]` rather
than as one of the four primary states.

## Metadata Generation

The generated NASA milling metadata file is:

```bash
venv312/bin/python scripts/generate_heuristic_sequence_peng2026.py
```

Default generation uses `vib_table`, `window_size=64`, `hop_size=1`,
`idle_delta_ratio=0.35`, `statistic=rms`, `reference_mode=middle`,
`min_steady_length=4000`, and `candidate_merge_gap=32`. These defaults are
implementation choices, not reported paper constants.

The output follows the existing metadata schema:

```text
datasets/metadata/heuristic_sequence_peng2026.csv
```

## Synthetic Example

```bash
venv312/bin/python scripts/synthetic_peng2026_segmentation_example.py
venv312/bin/python scripts/synthetic_peng2026_segmentation_example.py --plot
```

import numpy as np

from milling_experiment_framework.preprocessing.windowing.overlapped_windowing import window_run


def test_window_run_creates_four_overlapped_windows():
    signal = np.arange(9000 * 2, dtype=np.float32).reshape(9000, 2)

    result = window_run(signal)

    assert result.windows.shape == (4, 3000, 2)
    assert result.masks.shape == (4, 3000)
    assert result.start_indices == [0, 2000, 4000, 6000]
    assert result.end_indices == [3000, 5000, 7000, 9000]
    assert result.valid_lengths == [3000, 3000, 3000, 3000]
    np.testing.assert_array_equal(result.windows[1, 0], signal[2000])


def test_window_run_pads_short_run_to_target_length():
    signal = np.ones((2500, 3), dtype=np.float32)

    result = window_run(signal)

    assert result.was_padded is True
    assert result.padding_length == 6500
    assert result.windows.shape == (4, 3000, 3)
    assert result.valid_lengths == [2500, 500, 0, 0]
    assert result.masks[0].sum() == 2500
    assert result.masks[2].sum() == 0

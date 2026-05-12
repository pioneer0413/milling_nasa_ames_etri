import numpy as np

from milling_experiment_framework.preprocessing.windowing.run_windowing import window_signal_channel_first


def test_window_signal_channel_first_overlap_starts_include_end():
    x = np.arange(2 * 9000, dtype=np.float32).reshape(2, 9000)

    result = window_signal_channel_first(x, window_length=3000, num_windows=5)

    assert result.windows.shape == (5, 2, 3000)
    assert result.stride == 1500
    assert result.window_start_indices == [0, 1500, 3000, 4500, 6000]
    assert result.window_end_indices == [3000, 4500, 6000, 7500, 9000]
    assert not result.padded
    np.testing.assert_array_equal(result.windows[-1], x[:, 6000:9000])


def test_window_signal_channel_first_short_run_pads_and_repeats():
    x = np.ones((6, 120), dtype=np.float32)

    result = window_signal_channel_first(x, window_length=300, num_windows=5)

    assert result.windows.shape == (5, 6, 300)
    assert result.padded
    assert result.padding_length == 180
    assert result.windowing_status == "padded_repeated_short_run"
    np.testing.assert_array_equal(result.windows[0], result.windows[4])
    assert np.all(result.windows[:, :, 120:] == 0.0)

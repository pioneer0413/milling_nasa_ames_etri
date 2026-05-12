import numpy as np

from milling_experiment_framework.preprocessing.features.window_feature_extractor import (
    FEATURE_NAMES,
    extract_multisensor_features,
)


def test_extract_multisensor_features_is_mask_aware_and_finite():
    signal = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [999.0, 999.0],
        ],
        dtype=np.float32,
    )
    mask = np.array([True, True, False])

    row, names = extract_multisensor_features(signal, ["s1", "s2"], mask=mask, prefix="window_1")

    assert row.shape == (len(FEATURE_NAMES) * 2,)
    assert len(names) == len(FEATURE_NAMES) * 2
    assert np.isfinite(row).all()
    assert names[0] == "window_1__s1__mean"
    assert row[0] == 1.5

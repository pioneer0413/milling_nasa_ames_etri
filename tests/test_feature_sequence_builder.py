import numpy as np
import pandas as pd

from milling_experiment_framework.preprocessing.sequences.feature_sequence_builder import (
    FeatureSequenceBuilder,
    fit_transform_feature_preprocessor,
)


def test_feature_sequence_builder_left_padding_and_no_cross_case():
    features = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    meta = pd.DataFrame(
        {
            "case_id": [1, 1, 1, 2, 2],
            "run_id": [1, 2, 3, 1, 2],
            "run_order": [1, 2, 3, 1, 2],
        }
    )
    builder = FeatureSequenceBuilder(sequence_size=3, padding_value=0.0)

    out = builder.build_sequences(features, meta, indices=[0, 1, 3, 4])

    assert out.x_seq.shape == (4, 3, 2)
    assert out.mask.tolist()[0] == [0.0, 0.0, 1.0]
    assert out.mask.tolist()[1] == [0.0, 1.0, 1.0]
    assert out.mask.tolist()[2] == [0.0, 0.0, 1.0]
    np.testing.assert_array_equal(out.x_seq[3, 1], features[3])
    np.testing.assert_array_equal(out.x_seq[3, 2], features[4])


def test_feature_preprocessor_fits_source_train_only():
    features = np.array([[1.0, np.nan], [2.0, 5.0], [100.0, 100.0]], dtype=float)

    transformed, imputer, scaler = fit_transform_feature_preprocessor(features, train_indices=[0, 1])

    assert transformed.shape == features.shape
    assert np.isfinite(transformed).all()
    assert imputer.statistics_[0] == 1.5
    assert imputer.statistics_[1] == 5.0
    # The target-like outlier row is transformed by source-train statistics, not fitted into them.
    assert scaler.mean_[0] == 1.5

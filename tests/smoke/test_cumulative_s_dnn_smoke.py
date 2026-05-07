from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from milling_experiment_framework.models.dl.base_sequence_learners import CNN1DRegressor, GRURegressor, LSTMRegressor, RNNRegressor
from milling_experiment_framework.models.dl.cumulative_descriptor import CumulativeDescriptorLayer
from milling_experiment_framework.models.dl.stacking_meta_learner import RidgeAdaBoostStackingRegressor


def test_cumulative_s_dnn_forward_backward_checkpoint_smoke():
    torch.manual_seed(0)
    batch_size = 4
    sequence_length = 128
    channels = 6
    x = torch.randn(batch_size, sequence_length, channels)
    y = torch.randn(batch_size, 1)
    descriptor = CumulativeDescriptorLayer(input_representation="cumulative")
    x_cum = descriptor(x)
    assert x_cum.shape == x.shape
    assert torch.isfinite(x_cum).all()
    learners = {
        "rnn": RNNRegressor(channels, hidden_size=8, num_layers=1, head_hidden_dims=[8]),
        "gru": GRURegressor(channels, hidden_size=8, num_layers=1, head_hidden_dims=[8]),
        "lstm": LSTMRegressor(channels, hidden_size=8, num_layers=1, head_hidden_dims=[8]),
        "cnn": CNN1DRegressor(channels, channels=[8], head_hidden_dims=[8]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, model in learners.items():
            out = model(x_cum)
            assert out.shape == (batch_size, 1)
            loss = nn.MSELoss()(out, y)
            loss.backward()
            path = Path(tmp) / f"{name}.pt"
            torch.save({"model_state": model.state_dict()}, path)
            assert path.exists()
    x_meta = np.random.randn(12, 4).astype("float32")
    y_meta = np.random.randn(12).astype("float32")
    meta = RidgeAdaBoostStackingRegressor(n_estimators=3, learning_rate=0.01, random_state=0).fit(x_meta, y_meta)
    pred = meta.predict(x_meta[:5])
    assert pred.shape == (5,)


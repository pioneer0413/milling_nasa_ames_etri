from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from milling_experiment_framework.models.dl.hybrid_lstm_process import HybridLSTMProcessRegressor


def test_hybrid_lstm_process_forward_backward_checkpoint_smoke():
    torch.manual_seed(0)
    batch_size = 4
    sequence_length = 128
    num_sensors = 6
    num_process_features = 4
    model = HybridLSTMProcessRegressor(
        num_sensors=num_sensors,
        process_feature_dim=num_process_features,
        hidden_size=32,
        num_layers=2,
        regression_hidden_dims=[16, 8],
        use_process_info=True,
    )
    sensor_sequence = torch.randn(batch_size, sequence_length, num_sensors)
    process_features = torch.randn(batch_size, num_process_features)
    target = torch.randn(batch_size, 1)
    y_pred = model(sensor_sequence, process_features)
    assert y_pred.shape == (batch_size, 1)
    loss = torch.nn.MSELoss()(y_pred, target)
    loss.backward()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "hybrid_lstm_process_smoke.pt"
        torch.save({"model_state": model.state_dict()}, path)
        loaded = HybridLSTMProcessRegressor(
            num_sensors=num_sensors,
            process_feature_dim=num_process_features,
            hidden_size=32,
            num_layers=2,
            regression_hidden_dims=[16, 8],
            use_process_info=True,
        )
        loaded.load_state_dict(torch.load(path, map_location="cpu")["model_state"])
        loaded_pred = loaded(sensor_sequence, process_features)
        assert loaded_pred.shape == (batch_size, 1)


def test_hybrid_lstm_process_sensor_only_and_process_only_smoke():
    sensor_only = HybridLSTMProcessRegressor(
        num_sensors=2,
        process_feature_dim=0,
        hidden_size=16,
        num_layers=1,
        regression_hidden_dims=[8],
        use_process_info=False,
        input_mode="sensor_only",
    )
    out = sensor_only(torch.randn(3, 64, 2), None)
    assert out.shape == (3, 1)

    process_only = HybridLSTMProcessRegressor(
        num_sensors=1,
        process_feature_dim=3,
        hidden_size=16,
        num_layers=1,
        regression_hidden_dims=[8],
        use_process_info=True,
        input_mode="process_only",
    )
    out = process_only(None, torch.randn(3, 3))
    assert out.shape == (3, 1)

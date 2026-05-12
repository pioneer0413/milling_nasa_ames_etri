import torch

from milling_experiment_framework.models.dl.cnn_gru_regressor import CNNGRURegressor
from milling_experiment_framework.models.dl.cnn_lag_concat_regressor import CNNLagConcatRegressor
from milling_experiment_framework.models.dl.cnn_only_regressor import CNNOnlyRegressor
from milling_experiment_framework.models.dl.windowed_cnn_encoder import WindowedCNNEncoder


def _config(window_length=300):
    return {
        "sequence": {"sequence_size": 3},
        "model_design": {
            "run_encoder": {
                "window_length": window_length,
                "num_windows": 5,
                "latent_dim": 32,
                "window_aggregation": "max_pool",
                "short_run_policy": "pad_to_window_length",
            },
            "cnn_encoder": {
                "channels": [16, 32],
                "kernel_size": 5,
                "use_batch_norm": True,
                "dropout": 0.0,
            },
            "cnn_only": {"head_hidden_dim": 32, "dropout": 0.0},
            "cnn_lag_concat": {"sequence_size": 3, "include_mask": True, "head_hidden_dim": 64, "dropout": 0.0},
            "cnn_gru": {"sequence_size": 3, "gru_hidden_size": 32, "gru_num_layers": 1, "bidirectional": False, "head_hidden_dim": 32},
        },
    }


def _backward_step(model, args):
    y = torch.randn(4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pred = model(*args)
    assert pred.shape == (4,)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    optimizer.step()


def test_windowed_cnn_encoder_forward_shapes():
    encoder = WindowedCNNEncoder(input_channels=6, window_length=300, num_windows=5, latent_dim=32)
    x = torch.randn(4, 5, 6, 300)

    z, window_latents = encoder(x, return_window_latents=True)

    assert z.shape == (4, 32)
    assert window_latents.shape == (4, 5, 32)


def test_cnn_only_regressor_backward_step():
    model = CNNOnlyRegressor.from_config(_config(), input_channels=6)
    x = torch.randn(4, 5, 6, 300)

    _backward_step(model, (x,))


def test_cnn_lag_concat_regressor_backward_step():
    model = CNNLagConcatRegressor.from_config(_config(), input_channels=6)
    x = torch.randn(4, 3, 5, 6, 300)
    mask = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32)

    _backward_step(model, (x, mask))


def test_cnn_gru_regressor_backward_step():
    model = CNNGRURegressor.from_config(_config(), input_channels=6)
    x = torch.randn(4, 3, 5, 6, 300)
    mask = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32)

    _backward_step(model, (x, mask))

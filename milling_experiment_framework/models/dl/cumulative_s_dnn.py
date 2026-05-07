from __future__ import annotations

from typing import Any

from torch import nn

from milling_experiment_framework.models.dl.base_sequence_learners import CNN1DRegressor, GRURegressor, LSTMRegressor, RNNRegressor
from milling_experiment_framework.models.dl.cumulative_descriptor import CumulativeDescriptorLayer
from milling_experiment_framework.registry.model_registry import MODEL_REGISTRY


class CumulativeSDNN(nn.Module):
    """Container for the four S-DNN base learners.

    The stacking meta learner is intentionally handled outside this module by
    the experiment entrypoint so out-of-fold predictions can be created strictly
    within the source-domain training data.
    """

    def __init__(self, input_channels: int, config: dict[str, Any] | None = None):
        super().__init__()
        config = config or {}
        head = config.get("base_regression_head", {})
        head_hidden = head.get("hidden_dims", [64])
        dropout = float(head.get("dropout", 0.5))
        learners = config.get("base_learners", {})
        rnn_cfg = learners.get("rnn", {})
        gru_cfg = learners.get("gru", {})
        lstm_cfg = learners.get("lstm", {})
        cnn_cfg = learners.get("cnn", {})
        self.rnn = RNNRegressor(input_size=input_channels, hidden_size=int(rnn_cfg.get("hidden_size", rnn_cfg.get("hidden_sizes", [128])[-1])), num_layers=int(rnn_cfg.get("num_layers", 2)), dropout=float(rnn_cfg.get("dropout", dropout)), head_hidden_dims=head_hidden)
        self.gru = GRURegressor(input_size=input_channels, hidden_size=int(gru_cfg.get("hidden_size", 32)), num_layers=int(gru_cfg.get("num_layers", 1)), dropout=float(gru_cfg.get("dropout", dropout)), head_hidden_dims=head_hidden)
        self.lstm = LSTMRegressor(input_size=input_channels, hidden_size=int(lstm_cfg.get("hidden_size", 32)), num_layers=int(lstm_cfg.get("num_layers", 1)), dropout=float(lstm_cfg.get("dropout", dropout)), head_hidden_dims=head_hidden)
        self.cnn = CNN1DRegressor(input_channels=input_channels, channels=cnn_cfg.get("channels", [128, 64, 32]), kernel_size=int(cnn_cfg.get("kernel_size", 3)), dropout=float(cnn_cfg.get("dropout", dropout)), pooling=cnn_cfg.get("pooling", "max"), head_hidden_dims=head_hidden)

    def forward(self, sensor_sequence):
        return {
            "rnn": self.rnn(sensor_sequence),
            "gru": self.gru(sensor_sequence),
            "lstm": self.lstm(sensor_sequence),
            "cnn": self.cnn(sensor_sequence),
        }


@MODEL_REGISTRY.register("cumulative_s_dnn")
class CumulativeSDNNModel:
    model_type = "DL"
    input_type = "timeseries"

    def __init__(self, config: dict, task_config: dict):
        params = dict(config.get("params", {}))
        input_channels = params.get("input_channels", config.get("input_channels", 1))
        if str(input_channels).lower() == "auto":
            input_channels = 1
        self.task_config = task_config
        self.descriptor = CumulativeDescriptorLayer(input_representation=config.get("input_representation", "cumulative"))
        self.module = CumulativeSDNN(input_channels=int(input_channels), config=config)

    def fit(self, X, y, **kwargs):
        raise RuntimeError("CumulativeSDNNModel requires source-domain-only OOF stacking; use scripts/run_cumulative_s_dnn_experiment.py.")

    def predict(self, X):
        raise RuntimeError("Use scripts/run_cumulative_s_dnn_experiment.py for cumulative S-DNN prediction.")


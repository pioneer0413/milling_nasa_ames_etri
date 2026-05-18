import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.run_H4_S1_cnn_gru_domain_generalization import RunSequenceDataset, train_dl_model


def test_feature_gru_training_smoke_with_learning_history_and_checkpoint(tmp_path):
    config = {
        "experiment": {"experiment_id": "feature_gru_smoke"},
        "data": {"sensor_columns": ["s1", "s2"]},
        "sequence": {"sequence_size": 3, "padding": "left"},
        "feature_gru": {
            "model": {
                "gru_hidden_size": 8,
                "gru_num_layers": 1,
                "bidirectional": False,
                "dropout": 0.0,
                "regression_head_hidden_dim": 8,
            }
        },
        "training": {
            "learning_rate": 0.001,
            "max_epochs": 2,
            "early_stopping": {"patience": 10},
        },
    }
    x = torch.randn(8, 3, 6).numpy()
    mask = torch.ones(8, 3).numpy()
    y = torch.randn(8).numpy()
    train_loader = DataLoader(RunSequenceDataset(x[:6], mask[:6], y[:6]), batch_size=2, shuffle=True)
    val_loader = DataLoader(RunSequenceDataset(x[6:], mask[6:], y[6:]), batch_size=2, shuffle=False)

    model, best_metrics, elapsed, history, best_epoch = train_dl_model(
        "feature_gru",
        train_loader,
        val_loader,
        config,
        torch.device("cpu"),
        y_mean=0.0,
        y_std=1.0,
        checkpoint_dir=tmp_path,
        checkpoint_stem="feature_gru_smoke",
        scenario_name="A_to_B",
        source_domain="A",
        target_domain="B",
        seed=0,
        max_epochs=2,
    )

    assert best_metrics["RMSE"] >= 0.0
    assert elapsed >= 0.0
    assert isinstance(history, pd.DataFrame)
    assert set(["epoch", "train_loss", "val_RMSE", "is_best_epoch"]).issubset(history.columns)
    assert best_epoch["model_name"] == "feature_gru"
    assert (tmp_path / "feature_gru_smoke_best.pt").exists()
    assert (tmp_path / "feature_gru_smoke_last.pt").exists()

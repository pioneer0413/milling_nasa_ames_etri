import torch

from milling_experiment_framework.models.dl.feature_gru_regressor import FeatureGRURegressor


def test_feature_gru_forward_backward_and_checkpoint(tmp_path):
    model = FeatureGRURegressor(input_dim=54, gru_hidden_size=32, regression_head_hidden_dim=32)
    x = torch.randn(4, 3, 54)
    mask = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32)
    y = torch.randn(4)

    pred = model(x, mask)
    assert pred.shape == (4,)

    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.step()

    ckpt = tmp_path / "feature_gru.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1, "best_val_RMSE": 1.0}, ckpt)
    loaded = FeatureGRURegressor(input_dim=54)
    loaded.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])


def test_feature_gru_rejects_wrong_feature_dim():
    model = FeatureGRURegressor(input_dim=10)
    x = torch.randn(2, 3, 9)
    try:
        model(x)
    except ValueError as exc:
        assert "feature dimension mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for feature_dim mismatch")

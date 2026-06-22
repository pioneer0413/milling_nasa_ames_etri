import torch

from milling_experiment_framework.models.dl.feature_gru_regressor import FeatureFiLMGRURegressor, FeatureGRURegressor


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


def test_feature_film_gru_modulates_only_metadata_features():
    model = FeatureFiLMGRURegressor(
        input_dim=10,
        metadata_feature_dim=3,
        gru_hidden_size=8,
        regression_head_hidden_dim=8,
    )
    x = torch.randn(2, 4, 10)

    modulated = model.apply_metadata_film(x)

    assert torch.allclose(modulated[..., :7], x[..., :7])
    assert torch.allclose(modulated[..., 7:], x[..., 7:])

    with torch.no_grad():
        last = model.film.generator[-1]
        last.bias[3:] = torch.tensor([0.1, -0.2, 0.3])
    shifted = model.apply_metadata_film(x)
    assert torch.allclose(shifted[..., :7], x[..., :7])
    assert not torch.allclose(shifted[..., 7:], x[..., 7:])


def test_feature_film_gru_forward_backward_and_checkpoint(tmp_path):
    model = FeatureFiLMGRURegressor(
        input_dim=12,
        metadata_feature_dim=3,
        gru_hidden_size=8,
        regression_head_hidden_dim=8,
    )
    x = torch.randn(4, 3, 12)
    mask = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32)
    y = torch.randn(4)

    pred = model(x, mask)
    assert pred.shape == (4,)

    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.step()

    ckpt = tmp_path / "feature_film_gru.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1, "best_val_RMSE": 1.0}, ckpt)
    loaded = FeatureFiLMGRURegressor(
        input_dim=12,
        metadata_feature_dim=3,
        gru_hidden_size=8,
        regression_head_hidden_dim=8,
    )
    loaded.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])


def test_feature_film_gru_rejects_metadata_only_input():
    try:
        FeatureFiLMGRURegressor(input_dim=3, metadata_feature_dim=3)
    except ValueError as exc:
        assert "metadata_feature_dim must be smaller than input_dim" in str(exc)
    else:
        raise AssertionError("Expected ValueError when no signal features remain")

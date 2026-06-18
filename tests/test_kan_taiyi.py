import torch

from extension.model.kan.KAN_layer import KANLinear


def test_kan_linear_exposes_taiyi_residual_states():
    layer = KANLinear(3, 4)
    x = torch.randn(5, 3)

    y = layer(x)
    state = layer.residual_states["default"]

    assert y.shape == (5, 4)
    assert set(state.keys()) == {"stream", "branch", "output"}
    assert state["stream"].shape == (5, 4)
    assert state["branch"].shape == (5, 4)
    assert state["output"].shape == (5, 4)
    assert torch.allclose(state["output"], state["stream"] + state["branch"], atol=1e-6, rtol=1e-6)

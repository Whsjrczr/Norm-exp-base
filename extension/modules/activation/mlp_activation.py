import torch
import torch.nn as nn


def _make_inner_activation(name, hidden_dim, activation_cfg=None):
    activation_cfg = dict(activation_cfg or {})
    name = str(name).lower()

    if name == "relu":
        return nn.ReLU(inplace=bool(activation_cfg.get("inplace", False)))
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "no":
        return nn.Identity()

    raise ValueError(
        f"MLPActivation inner activation '{name}' is not supported. "
        "Choose from: relu, sigmoid, tanh, silu, gelu, no."
    )


class MLPActivation(nn.Module):
    """Shared scalar learnable activation implemented as a 1-n-1 MLP."""

    def __init__(self, num_features=None, hidden_dim=16, n=None, act="relu", act_cfg=None, bias=True):
        super().__init__()
        if n is not None:
            hidden_dim = n
        hidden_dim = int(hidden_dim)
        if hidden_dim < 1:
            raise ValueError(f"MLPActivation expects hidden_dim >= 1, but got hidden_dim={hidden_dim}.")

        self.hidden_dim = hidden_dim
        self.act_name = str(act).lower()
        self.fc1 = nn.Linear(1, hidden_dim, bias=bias)
        self.act = _make_inner_activation(self.act_name, hidden_dim, act_cfg)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=bias)

    def forward(self, x):
        y = self.fc1(x.unsqueeze(-1))
        y = self.act(y)
        y = self.fc2(y)
        return y.squeeze(-1)

    def extra_repr(self):
        return f"hidden_dim={self.hidden_dim}, act={self.act_name}"

"""
(p,q) - Normalization
This is a PyTorch implementation of the (p,q) normalization as described in the paper [2]. 

Reference:
[1] Ni Y, Guo Y, Jia J, et al. On the nonlinearity of layer normalization[J]. arXiv preprint arXiv:2406.01255, 2024.
[2] Ni Y, Liu Y, Sun W, et al. Parallel layer normalization for universal approximation[J]. arXiv preprint arXiv:2505.13142, 2025.
"""

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter


class PQNorm(nn.Module):
    def __init__(
        self,
        num_features,
        num_per_group=None,
        p=2,
        q=2,
        eps=1e-5,
        centering=True,
        affine=True,
        dim=4,
        *args,
        **kwargs,
    ):
        super().__init__()
        if p < 1 or q < 1:
            raise ValueError(f"PQNorm expects p >= 1 and q >= 1, but got p={p}, q={q}.")

        self.num_features = num_features
        self.num_per_group = num_per_group if num_per_group is not None else num_features
        self.p = p
        self.q = q
        self.eps = eps
        self.centering = centering
        self.affine = affine
        self.dim = dim
        if self.num_features % self.num_per_group != 0:
            raise ValueError("num_features must be divisible by num_per_group")

        if self.affine:
            if self.dim == 4:
                self.shape = [1, self.num_features, 1, 1]
            elif self.dim == 3:
                self.shape = [1, 1, self.num_features]
            elif self.dim == 2:
                self.shape = [1, self.num_features]
            else:
                raise ValueError(f"Unsupported dim={self.dim} for PQNorm.")
            self.weight = Parameter(torch.empty(*self.shape))
            self.bias = Parameter(torch.empty(*self.shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _resolve_runtime_dim(self, x: torch.Tensor):
        runtime_dim = x.dim()
        if runtime_dim not in (2, 3, 4):
            raise RuntimeError(f"PQNorm only supports 2D, 3D, or 4D inputs, but got {tuple(x.shape)}.")
        if x.shape[-1] == self.num_features:
            return runtime_dim, False
        if runtime_dim >= 3 and x.shape[1] == self.num_features:
            return runtime_dim, True
        raise RuntimeError(
            f"PQNorm expected num_features={self.num_features} on channel dim 1 or last dim, "
            f"but got shape {tuple(x.shape)}."
        )

    def forward(self, x: torch.Tensor):
        runtime_dim, channel_first = self._resolve_runtime_dim(x)
        if channel_first:
            if runtime_dim == 4:
                x = x.permute(0, 2, 3, 1)
            elif runtime_dim == 3:
                x = x.permute(0, 2, 1)

        size = x.shape
        x = x.reshape(-1, self.num_features // self.num_per_group, self.num_per_group)
        if self.centering:
            x = x - x.mean(dim=-1, keepdim=True)
        ratio = self.p / self.q
        numerator = torch.sign(x) * torch.abs(x).pow(ratio)
        denominator = torch.mean(torch.abs(x).pow(self.p), dim=-1, keepdim=True)
        y = numerator / (denominator + self.eps).pow(1.0 / self.q)
        y = y.reshape(size)

        if channel_first:
            if runtime_dim == 4:
                y = y.permute(0, 3, 1, 2)
            elif runtime_dim == 3:
                y = y.permute(0, 2, 1)

        if self.affine:
            if runtime_dim == 4:
                shape = [1, self.num_features, 1, 1] if channel_first else [1, 1, 1, self.num_features]
            elif runtime_dim == 3:
                shape = [1, self.num_features, 1] if channel_first else [1, 1, self.num_features]
            else:
                shape = [1, self.num_features]
            weight = self.weight.view(*shape) if tuple(self.weight.shape) != tuple(shape) else self.weight
            bias = self.bias.view(*shape) if tuple(self.bias.shape) != tuple(shape) else self.bias
            y = y * weight + bias

        return y

    def extra_repr(self):
        return "num_features={}, num_per_group={}, p={}, q={}, eps={}, centering={}, affine={}".format(
            self.num_features, self.num_per_group, self.p, self.q, self.eps, self.centering, self.affine
        )

"""
Parallel Layer Normalization
This is a PyTorch implementation of the Parallel Layer Normalization (PLN) as described in the paper [2]. 

Reference:
[1] Ni Y, Guo Y, Jia J, et al. On the nonlinearity of layer normalization[J]. arXiv preprint arXiv:2406.01255, 2024.
[2] Ni Y, Liu Y, Sun W, et al. Parallel layer normalization for universal approximation[J]. arXiv preprint arXiv:2505.13142, 2025.
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ParallelLN(nn.Module):
    def __init__(
        self,
        num_features,
        num_per_group,
        eps=1e-5,
        centering=True,
        affine=True,
        dim=4,
        p=2,
        *args,
        **kwargs
    ):
        super(ParallelLN, self).__init__()
        self.num_features = num_features
        self.num_per_group = num_per_group
        self.eps = eps
        self.dim = dim
        self.affine = affine
        self.centering = centering
        self.p = p
        if self.affine:
            if self.dim == 4:
                self.shape = [1, self.num_features, 1, 1]
            elif self.dim == 3:
                self.shape = [1, 1, self.num_features]
            elif self.dim == 2:
                self.shape = [1, self.num_features]
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def _resolve_runtime_dim(self, x: torch.Tensor):
        runtime_dim = x.dim()
        if runtime_dim not in (2, 3, 4):
            raise RuntimeError(f"ParallelLN only supports 2D, 3D, or 4D inputs, but got {tuple(x.shape)}.")
        if x.shape[-1] == self.num_features:
            return runtime_dim, False
        if runtime_dim >= 3 and x.shape[1] == self.num_features:
            return runtime_dim, True
        raise RuntimeError(
            f"ParallelLN expected num_features={self.num_features} on channel dim 1 or last dim, "
            f"but got shape {tuple(x.shape)}."
        )

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        runtime_dim, channel_first = self._resolve_runtime_dim(x)
        if channel_first:
            if runtime_dim == 4:
                x = x.permute(0, 2, 3, 1)
            elif runtime_dim == 3:
                x = x.permute(0, 2, 1)
        size = x.size()
        assert self.num_features % self.num_per_group == 0, "num_features must be divisible by num_per_group"
        x = x.reshape(-1, self.num_features // self.num_per_group, self.num_per_group)
        if self.centering:
            mean = x.mean(dim=2, keepdim=True)
            xc = x - mean
        else:
            xc = x
        std = torch.mean(torch.abs(xc) ** self.p, dim=2, keepdim=True)
        norm = 1 / (std + self.eps) ** (1 / self.p)
        xn = xc * norm
        xn = xn.reshape(size)
        if channel_first:
            if runtime_dim == 4:
                xn = xn.permute(0, 3, 1, 2)
            elif runtime_dim == 3:
                xn = xn.permute(0, 2, 1)
        if self.affine:
            if runtime_dim == 4:
                shape = [1, self.num_features, 1, 1] if channel_first else [1, 1, 1, self.num_features]
            elif runtime_dim == 3:
                shape = [1, self.num_features, 1] if channel_first else [1, 1, self.num_features]
            else:
                shape = [1, self.num_features]
            weight = self.weight.view(*shape) if tuple(self.weight.shape) != tuple(shape) else self.weight
            bias = self.bias.view(*shape) if tuple(self.bias.shape) != tuple(shape) else self.bias
            xn = xn * weight + bias
        return xn

    def extra_repr(self):
        return "{num_features}, {num_per_group}, affine={affine}, centering={centering}".format(
            **self.__dict__
        )


if __name__ == "__main__":
    x = torch.randn((1, 9, 3, 3))
    f = ParallelLN(num_features=9, num_per_group=9, dim=4)
    fx = f(x)
    print(fx.mean(dim=(1)))
    print((fx**2).mean(dim=(1)))
    print("-----------------------------------------------")
    x = torch.randn((1, 9, 3, 3))
    f = ParallelLN(num_features=9, num_per_group=3, dim=4)
    fx = f(x)
    fx = fx.view(1, 3, 3, 3, 3)
    print(fx.mean(dim=(2)))
    print((fx**2).mean(dim=(2)))
    print("-----------------------------------------------")
    x = torch.randn((3, 3, 8))
    f = ParallelLN(num_features=8, num_per_group=4, dim=3)
    fx = f(x)
    fx = fx.view(3, 3, 2, 4)
    print(fx.mean(dim=(3)))
    print((fx**2).mean(dim=(3)))
    print("-----------------------------------------------")
    x = torch.randn((4, 16))
    f = ParallelLN(num_features=16, num_per_group=8, dim=2)
    fx = f(x)
    fx = fx.view(4, 2, 8)
    print(fx.mean(dim=(2)))
    print((fx**2).mean(dim=(2)))
    print("-----------------------------------------------")

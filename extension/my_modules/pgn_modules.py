import torch
from torch import Size, Tensor
import torch.nn as nn
from torch.nn import init as init
from torch.nn.parameter import Parameter

from typing import List, Optional, Tuple, Union


_shape_t = Union[int, List[int], Size]

class PointwiseGroupNormCentering(nn.Module):
    __constants__ = ["num_groups", "num_channels", "eps", "affine"]
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        size = input.shape
        # assert input.size(1) == self.num_channels
        input = input.view(size[0], self.num_groups, self.num_channels // self.num_groups, *size[2:])
        length = len(input.shape)
        mean = input.mean(dim=2, keepdim=True)
        output = input - mean
        output = output.view(*size)
        
        if self.affine:
            output = output * self.weight + self.bias
        return output
    

class PointwiseGroupNormScaling(nn.Module):
    __constants__ = ["num_groups", "num_channels", "eps", "affine"]
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        size = input.shape
        # assert input.size(1) == self.num_channels
        input = input.view(size[0], self.num_groups, self.num_channels // self.num_groups, *size[2:])
        length = len(input.shape)
        var = input.var(dim=2, keepdim=True, unbiased=False)

        output = input / torch.sqrt(var + self.eps)
        output = output.view(*size)
        
        if self.affine:
            output = output * self.weight + self.bias
        return output
    

if __name__ == '__main__':

    x = torch.randn(16, 32, 4)

    size = x.size()
    print(size)
    x_reshape = x.view(size[0], 4, size[1]//4, *size[2:])
    print(x_reshape.size())

    print()

    gc = GroupNormCentering(2,32,affine=False)
    gs = GroupNormScaling(2,32,affine=False)
    gn = nn.GroupNorm(2, 32, affine=False)

    ln = nn.LayerNorm([32, 4], elementwise_affine=False)

    print("orgin")
    print(x)
    print("gn")
    y = gn(x)
    print(y)
    print("gn1")
    z = gs(gc(x))
    print(z)
    print(y-z)


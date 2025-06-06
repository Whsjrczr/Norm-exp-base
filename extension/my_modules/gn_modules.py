import math

import torch
from torch import Size, Tensor
import torch.nn as nn
from torch.nn import init as init
from torch.nn.parameter import Parameter

from typing import List, Optional, Tuple, Union


_shape_t = Union[int, List[int], Size]

class GroupNormCentering(nn.Module):
    __constants__ = ["num_groups", "num_channels", "affine"]
    num_groups: int
    num_channels: int
    affine: bool

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
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
        self.affine = affine
        if self.affine:
            self.bias = Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        size = input.shape
        # assert input.size(1) == self.num_channels
        input = input.view(size[0], self.num_groups, self.num_channels // self.num_groups, *size[2:])
        length = len(input.shape)
        dims_to_var = [i for i in range (2, length)]
        mean = input.mean(dim=dims_to_var, keepdim=True)
        output = input - mean
        output = output.view(*size)
        
        if self.affine:
            dims_to_affine = [1 for i in range (2, len(size))]
            bias = self.bias.view(1, self.num_channels,*dims_to_affine)
            output = output + bias
        return output
    
    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, " "affine={affine}".format(
            **self.__dict__
        )
    

class GroupNormScaling(nn.Module):
    __constants__ = ["num_groups", "num_channels", "eps", "affine", "bias"]
    num_groups: int
    num_channels: int
    eps: float
    affine: bool
    bias: bool


    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        bias: bool = False,
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
        self.affine_bias = bias
        if self.affine:
            self.weight = Parameter(torch.empty(num_channels, **factory_kwargs))
            if self.affine_bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            if self.affine_bias:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        size = input.shape
        # assert input.size(1) == self.num_channels
        input = input.view(size[0], self.num_groups, self.num_channels // self.num_groups, *size[2:])
        length = len(input.shape)
        dims_to_var = [i for i in range (2, length)]

        var = input.var(dim=dims_to_var, keepdim=True, unbiased=False)
        output = input / torch.sqrt(var + self.eps)
        output = output.view(*size)
        
        if self.affine:
            dims_to_affine = [1 for i in range (2, len(size))]
            weight = self.weight.view(1, self.num_channels,*dims_to_affine)
            output = output * weight
            if self.affine_bias:
                bias = self.bias.view(1, self.num_channels,*dims_to_affine)
                output = output + bias
        return output
    
    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, " "affine={affine}".format(
            **self.__dict__
        )
    

class GroupNormScalingRMS(nn.Module):
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
        else:
            self.register_parameter("weight", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        size = input.shape
        input = input.view(size[0], self.num_groups, self.num_channels // self.num_groups, *size[2:])
        length = len(input.shape)
        dims_to_norm = [i for i in range (2, length)]

        norm = input.norm(p=2, dim=dims_to_norm, keepdim=True)
        frac = math.prod(input.shape[2:])

        output = input / (norm / ( frac ** (1/2)) + self.eps)
        output = output.view(*size)
        
        if self.affine:
            dims_to_affine = [1 for i in range (2, len(size))]
            weight = self.weight.view(1, self.num_channels,*dims_to_affine)
            output = output * weight
        return output
    
    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, " "affine={affine}".format(
            **self.__dict__
        )

if __name__ == '__main__':

    x = torch.randn(16, 32, 4)

    size = x.size()
    print(size)
    x_reshape = x.view(size[0], 4, size[1]//4, *size[2:])
    print(x_reshape.size())

    print()

    gc = GroupNormCentering(2,32,affine=True)
    gs = GroupNormScaling(2,32,affine=True)
    gn = nn.GroupNorm(2, 32, affine=False)
    grms = GroupNormScalingRMS(2,32,affine=True)

    ln = nn.LayerNorm([32, 4], elementwise_affine=False)

    # print("orgin")
    # print(x)
    # print("gn")
    # y = gn(x)
    # print(y)
    # print("gn1")
    # z = grms(gc(x))
    # print(z)
    # print(y-z)

    print(gc)
    print(gs)
    print(grms)

    print(gc(x))
    print(gs(x))
    print(grms(x))

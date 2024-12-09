import torch
from torch import Size, Tensor
import torch.nn as nn
from torch.nn import init as init
from torch.nn.parameter import Parameter

from typing import List, Optional, Tuple, Union


_shape_t = Union[int, List[int], Size]

class PointwiseGroupNorm(nn.Module):
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
        var = input.var(dim=2, keepdim=True, unbiased=False)
        mean = input.mean(dim=2, keepdim=True)
        output = (input - mean) / torch.sqrt(var + self.eps)
        output = output.view(*size)
        print(mean.shape)
        
        if self.affine:
            output = output * self.weight + self.bias
        return output
    
    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, " "affine={affine}".format(
            **self.__dict__
        )
    



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
        mean = input.mean(dim=2, keepdim=True)
        output = input - mean
        output = output.view(*size)
        
        if self.affine:
            output = output * self.weight + self.bias
        return output
    
    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, " "affine={affine}".format(
            **self.__dict__
        )
    
    

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
        var = input.var(dim=2, keepdim=True, unbiased=False)

        output = input / torch.sqrt(var + self.eps)
        output = output.view(*size)
        
        if self.affine:
            output = output * self.weight + self.bias
        return output

    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, " "affine={affine}".format(
            **self.__dict__
        )
    


class PointwiseGroupNormScalingRMS(nn.Module):
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
        input = input.view(size[0], self.num_groups, self.num_channels // self.num_groups, *size[2:])
        norm = input.norm(p=2, dim=2, keepdim=True)
        frac = input.shape[2]

        output = input / (norm / (frac ** (1/2)) + self.eps)
        output = output.view(*size)
        
        if self.affine:
            output = output * self.weight + self.bias
        return output

    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, " "affine={affine}".format(
            **self.__dict__
        )
    
    

if __name__ == '__main__':

    x = torch.randn(2, 6, 4, 2)

    pgn = PointwiseGroupNorm(2, 6, affine=False)

    print(x)
    print(pgn(x))


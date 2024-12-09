import numbers
import math

import torch
from torch import Size, Tensor
import torch.nn as nn
from torch.nn import init as init
from torch.nn.parameter import Parameter

from typing import List, Optional, Tuple, Union


_shape_t = Union[int, List[int], Size]


class LayerNormCentering(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, 
        normalized_shape: _shape_t,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        length = len(self.normalized_shape)
        dims_to_mean = [i for i in range (-length, 0)]
        mean = torch.mean(input, dim=dims_to_mean, keepdim=True)
        centered_tensor = input - mean
        if self.elementwise_affine:
            centered_tensor = centered_tensor * self.weight + self.bias
        return centered_tensor
    
class LayerNormScaling(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool
    
    def __init__(self, 
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        length = len(self.normalized_shape)
        dims_to_var = [i for i in range (-length, 0)]
        var = torch.var(input, dim=dims_to_var, unbiased=False, keepdim=True)
        centered_tensor = input / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            centered_tensor = centered_tensor * self.weight + self.bias
        return centered_tensor
    

class LayerNormScalingRMS(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, 
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        length = len(self.normalized_shape)
        dims_to_norm = [i for i in range (-length, 0)]
        norm = torch.norm(input, p=2, dim=dims_to_norm, keepdim=True)
        frac = math.prod(input.shape[-length:])
        centered_tensor = input / (norm / (frac ** (1/2)) + self.eps)
        if self.elementwise_affine:
            centered_tensor = centered_tensor * self.weight + self.bias
        return centered_tensor
    


if __name__ == '__main__':

    x = torch.randn(1, 4, 3, 6)

    lc = LayerNormCentering([4,3,6], elementwise_affine=False)
    rms = LayerNormScalingRMS([4,3,6], elementwise_affine=False)
    ls = LayerNormScaling([4,3,6], elementwise_affine=False)
    ln = nn.LayerNorm([3,6], elementwise_affine=False)

    print(lc.__class__.__name__)

    # print("orgin")
    # print(x)
    # print("ln")
    # y = ln(x)
    # print(y)
    # print("lc+ls")
    # z = ls(lc(x))
    # # print(z)
    # print(y-z)
    # print("lc+rms")
    # a = rms(lc(x))
    # # print(a)
    # print(y-a)

    # print()
    # print(z-a)


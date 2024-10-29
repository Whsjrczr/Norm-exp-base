import torch
import torch.nn as nn
import numbers
from typing import Tuple
from torch.nn import functional as F


class SOLayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        super(SOLayerNorm, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    '''
    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)
    '''

    def forward(self, input_tensor):
        var = torch.var(input_tensor, dim=-1, unbiased=False, keepdim=True)
        normalized_tensor = input_tensor / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            normalized_tensor = normalized_tensor * self.weight + self.bias
        return normalized_tensor

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class SOGroupNorm(nn.Module):
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input_tensor):
        # 将输入张量沿通道维度分成多个组
        groups = torch.chunk(input_tensor, self.num_groups, dim=1)
        scaled_groups = []
        for group in groups:
            # 计算组内沿通道维度的标准差
            var = torch.var(group, dim=1, unbiased=False, keepdim=True)
            # 对组内的每个样本进行放缩
            scaled_group = group / torch.sqrt(var + self.eps)
            if self.affine:
                scaled_group = scaled_group * self.weight + self.bias
            scaled_groups.append(scaled_group)
        # 将放缩后的组合并成affi一个张量
        scaled_tensor = torch.cat(scaled_groups, dim=1)
        return scaled_tensor

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={ne}'.format(**self.__dict__)



# BN 1d2d3d
# LRN/IN

# if __name__ == '__main__':

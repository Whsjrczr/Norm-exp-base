import math
import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from .culculate_tools import my_calculate_fan_in_and_fan_out
from .normalization_scalingonly import SOLayerNorm



# 这里CCLinear可以直接替代nn.Linear函数
# 记得写GN对饮的GCCLinear
class CCLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = my_calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        column_means = torch.mean(self.weight, dim=0)
        compute_weight = torch.sub(self.weight, column_means)
        bias_mean = torch.mean(self.bias, dim=0)
        compute_bias = torch.sub(self.bias, bias_mean)
        return F.linear(input, compute_weight, compute_bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


if __name__ == '__main__':
    dbn = CCLinear(16, 4)
    dby = nn.Linear(16, 4)
    dby.weight = dbn.weight
    dby.bias = dbn.bias
    d = nn.LayerNorm(4, elementwise_affine=False)
    c = SOLayerNorm(4, elementwise_affine=False)
    x = torch.ones(4, 16)
    print("CCL")
    y = dbn(x)
    print(y)
    print("+LN")
    z = d(y)
    print(z)
    print("+SOLN")
    y = c(y)
    print(y)
    print("L+LN")
    x = torch.ones(4, 16)
    y = dby(x)
    print(y)
    y = d(y)
    print(y)
'''    y = y.view(y.size(0), dbn.num_groups, y.size(1) // dbn.num_groups, *y.size()[2:])
    y = y.view(y.size(0), dbn.num_groups, -1)
    print('y reshaped:', y.size())
    z = y.matmul(y.transpose(1,2))/y.size(2)
    #print('train mode:', z.diag())
    print('z_ins:', z)
    y = y.transpose(0, 1).contiguous().view(dbn.num_groups, -1)
    print('y reshaped:', y.size())
    z = y.matmul(y.transpose(0,1))/y.size(1)
    print('z_batch:', z)
    # print(__file__)'''
import re

import torch
import torch.nn as nn

from ..activation import Activation
from ..normalization import Norm

class Copy(nn.Module):
    def __init__(self, thickness=1, data_dim=2, copy_dim=None):
        super(Copy, self).__init__()
        copy_list = {2:1, 3:2, 4:1}
        self.copy_dim = copy_list[data_dim] if copy_dim is None else copy_dim
        self.thickness = thickness

    def forward(self, x):
        return x.repeat_interleave(self.thickness, dim=self.copy_dim)

    def extra_repr(self):
        return "thickness={}, copy_dim={}".format(self.thickness, self.copy_dim)

class Bias(nn.Module):
    def __init__(self, num_features=32, dim=2, initial_mode="zeros", requires_grad=True):
        super(Bias, self).__init__()
        self.num_features = num_features
        self.dim = dim
        if initial_mode == "zeros":
            if dim == 2:
                self.bias = nn.Parameter(torch.zeros(1, num_features), requires_grad=requires_grad)
            elif dim == 3:
                self.bias = nn.Parameter(torch.zeros(1, 1, num_features), requires_grad=requires_grad)
            elif dim == 4:
                self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1), requires_grad=requires_grad)
        elif initial_mode == "uniform":
            if dim == 2:
                self.bias = nn.Parameter(torch.rand(1, num_features)*6-3, requires_grad=requires_grad)
            elif dim == 3:
                self.bias = nn.Parameter(torch.rand(1, 1, num_features)*6-3, requires_grad=requires_grad)
            elif dim == 4:
                self.bias = nn.Parameter(torch.rand(1, num_features, 1, 1)*6-3, requires_grad=requires_grad)

    def forward(self, x):
        return x + self.bias

    def extra_repr(self):
        return "num_features={}, dim={}".format(self.num_features, self.dim)

class Weight(nn.Module):
    def __init__(self, in_features=32, out_features=64, dim=2):
        super(Weight, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dim = dim
        if dim == 2:
            self.weight = nn.Linear(in_features, out_features, bias=False)
        elif dim == 3:
            self.weight = nn.Linear(in_features, out_features, bias=False)
        elif dim == 4:
            self.weight = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)

    def forward(self, x):
        return self.weight(x)

    def extra_repr(self):
        return "in_features={}, out_features={}, dim={}".format(self.in_features, self.out_features, self.dim)
    

class MLP_Basic_Layer(nn.Module):
    '''
    kan layer：mode=cbaw
    mlp layer：mode=awb
    '''

    def __init__(
        self,
        in_features,
        out_features,
        thickness=1,
        mode="wbac",
        dim=2,
        bias_initial_mode="zeros",
        bias_requires_grad=True,
        residual=False,
    ):
        super(MLP_Basic_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.thickness = thickness
        self.mode = mode
        self.layers = nn.ModuleList()
        self.residual = residual
        input_dim = in_features
        if re.search(r"w.*c", mode):
            raise ValueError(
                f"Invalid mode '{mode}': 'Copy' must appear before 'Weight' (found 'w' before 'c')"
            )

        for m in mode:
            if m == "w": # weight
                self.layers.append(Weight(input_dim, out_features, dim=dim))
                input_dim = out_features
            elif m == "b": # bias
                self.layers.append(
                    Bias(
                        input_dim,
                        dim=dim,
                        initial_mode=bias_initial_mode,
                        requires_grad=bias_requires_grad,
                    )
                )
            elif m == "a": # activation
                self.layers.append(Activation(input_dim))
            elif m == "c": # copy operation
                self.layers.append(Copy(thickness=thickness, data_dim=dim))
                input_dim *= thickness
            elif m == "n": # normalization
                self.layers.append(Norm(input_dim, dim=dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        if self.residual:
            return x + self.layers(x)
        else:
            return self.layers(x)

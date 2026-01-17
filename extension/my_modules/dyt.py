import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


# class LNPositionNorm(nn.Module):
#     def __init__(self, num_features, num_groups=4, num_channels=0,eps=1e-5, frozen=False, affine=True, *args, **kwargs):
#         super(LNPositionNorm, self).__init__()
#         self.num_features = num_features
#         self.num_groups = num_groups
#         self.eps = eps
#         self.shape = [1, 1, 1, self.num_groups, self.num_features // self.num_groups]
#         self.affine = affine
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(*self.shape))
#             self.bias = Parameter(torch.Tensor(*self.shape))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         if self.affine:
#             nn.init.ones_(self.weight)
#             nn.init.zeros_(self.bias)
#
#     def forward(self, input: torch.Tensor):
#         size = input.size()
#         x = input.reshape(size[0], size[2], size[3], self.num_groups, self.num_features // self.num_groups)
#         mean = x.mean(-1, keepdim=True)
#         xc = x - mean
#         std = xc.std(-1, keepdim=True) + self.eps
#         xn = xc/std
#         #output = xn.view(input.size(1), input.size(0), *input.size()[2:]).transpose(0, 1).contiguous()
#         if self.affine:
#             xn = xn * self.weight + self.bias
#         output = xn.reshape_as(input)
#         return output
#
#     def extra_repr(self):
#         return '{num_features}, groups={num_groups}, affine={affine}'.format(**self.__dict__)


class DynamicTanh(nn.Module):
    def __init__(self, num_features=0, alpha_init_value=0.5, *args, **kwargs):
        super(DynamicTanh, self).__init__()
        self.num_features=num_features
        self.alpha = Parameter(torch.ones(1) * alpha_init_value)
        self.weight = Parameter(torch.ones([1,1,num_features]))
        self.bias = Parameter(torch.zeros([1,1,num_features]))

    def forward(self, input: torch.Tensor):
        x = torch.tanh(self.alpha * input)
        return x * self.weight + self.bias

    def extra_repr(self):
        return "{num_features}".format(**self.__dict__)


if __name__ == "__main__":
    x = torch.randn((1, 3, 3, 3))
    f = DynamicTanh(num_features=3)
    fx = f(x)
    print(fx)

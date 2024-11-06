import torch

from torch import Tensor
import torch.nn as nn
from torch.nn import init as init
from torch.nn.parameter import Parameter

class BatchNorm1dCentering(nn.Module):
    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float = 1e-5
    momentum: float | None
    affine: bool
    track_running_stats: bool
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, **factory_kwargs)
            )
            self.running_mean: Tensor | None
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    0,
                    dtype=torch.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )
            self.num_batches_tracked: Tensor | None
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.num_batches_tracked.zero_()
    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None: 
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None)

        assert input.dim() in (2, 3)
        if input.dim() == 2:
            cal_dim = 0
        elif input.dim() == 3:
            cal_dim = [0,2]

        if bn_training:
            mean = torch.mean(input, dim=cal_dim, keepdim=True)
            self.running_mean = (exponential_average_factor * self.running_mean + (1 - exponential_average_factor) * mean)
        else:
            mean = self.running_mean
        output = input - mean
        if self.affine:
            output = output * self.weight + self.bias
        return output
    
class BatchNorm1dScaling(nn.Module):
    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: float | None
    affine: bool
    track_running_stats: bool
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_var", torch.zeros(num_features, **factory_kwargs)
            )
            self.running_var: Tensor | None
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    0,
                    dtype=torch.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )
            self.num_batches_tracked: Tensor | None
        else:
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None: 
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_var is None)

        assert input.dim() in (2, 3)
        if input.dim() == 2:
            cal_dim = 0
        elif input.dim() == 3:
            cal_dim = [0,2]

        if bn_training:
            var = torch.var(input, dim=cal_dim, unbiased=False, keepdim=True)
            self.running_var = (exponential_average_factor * self.running_var + (1 - exponential_average_factor) * var)
        else:
            var = self.running_var
        output = input / torch.sqrt(var + self.eps)
        if self.affine:
            output = output * self.weight + self.bias
        return output
    
if __name__ == '__main__':

    x = torch.randn(3, 4, 5)

    bc = BatchNorm1dCentering(4, affine=False)
    bs = BatchNorm1dScaling(4, affine=False)
    bn = nn.BatchNorm1d(4, affine=False)

    print("orgin")
    print(x)
    print("bn")
    y = bn(x)
    print(y)
    print("bc+bs")
    z = bs(bc(x))
    print(z)
    print(y-z)

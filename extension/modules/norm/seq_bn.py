import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import init as init
from torch.nn.parameter import Parameter


def _resolve_sequence_dim(input: Tensor, layout: str | None, num_features: int | None = None):
    if input.dim() < 3:
        raise RuntimeError(
            f"Expected input.dim() >= 3 for sequence batch norm, but got shape {tuple(input.shape)}."
        )

    if layout is None:
        if num_features is None:
            raise RuntimeError("layout must be specified when num_features is not provided.")
        first_match = input.shape[1] == num_features
        last_match = input.shape[-1] == num_features
        if first_match and not last_match:
            sequence_dim = 1
        elif last_match and not first_match:
            sequence_dim = input.dim() - 1
        elif first_match and last_match:
            raise RuntimeError(
                "Ambiguous sequence axis: both dim 1 and the last dim match "
                f"num_features={num_features} for shape {tuple(input.shape)}. "
                "Pass layout='first' or layout='last' explicitly."
            )
        else:
            raise RuntimeError(
                f"Expected sequence length={num_features} on dim 1 or the last dim, "
                f"but got shape {tuple(input.shape)}."
            )
    elif layout == "first":
        sequence_dim = input.dim() - 1
    elif layout == "last":
        sequence_dim = 1
    else:
        raise ValueError(f"Unsupported sequence layout: {layout}. Expected 'first' or 'last'.")

    if num_features is not None and input.shape[sequence_dim] != num_features:
        raise RuntimeError(
            f"Expected sequence length={num_features} on dim {sequence_dim}, "
            f"but got shape {tuple(input.shape)}."
        )

    reduce_dims = tuple(dim for dim in range(input.dim()) if dim != sequence_dim)
    view_shape = [1] * input.dim()
    view_shape[sequence_dim] = input.shape[sequence_dim]

    return sequence_dim, reduce_dims, view_shape


def _resolve_feature_dim(input: Tensor, layout: str | None, num_features: int | None = None):
    if input.dim() < 3:
        raise RuntimeError(
            f"Expected input.dim() >= 3 for sequence norm, but got shape {tuple(input.shape)}."
        )

    if layout is None or layout == "last":
        feature_dim = input.dim() - 1
    elif layout == "first":
        feature_dim = 1
    else:
        raise ValueError(f"Unsupported sequence layout: {layout}. Expected 'first' or 'last'.")

    if num_features is not None and input.shape[feature_dim] != num_features:
        raise RuntimeError(
            f"Expected num_features={num_features} on dim {feature_dim}, "
            f"but got shape {tuple(input.shape)}."
        )

    view_shape = [1] * input.dim()
    view_shape[feature_dim] = input.shape[feature_dim]
    return feature_dim, view_shape


def _prefix_stats(input: Tensor, sequence_dim: int, eps: float | None = None, reduce_non_sequence: bool = False):
    if reduce_non_sequence:
        reduce_dims = tuple(dim for dim in range(input.dim()) if dim != sequence_dim)
        summed = input.sum(dim=reduce_dims, keepdim=True)
        squared = input.square().sum(dim=reduce_dims, keepdim=True)
        sample_count = 1
        for dim in reduce_dims:
            sample_count *= input.shape[dim]
    else:
        summed = input
        squared = input.square()
        sample_count = 1

    prefix_sum = summed.cumsum(dim=sequence_dim)
    prefix_square_sum = squared.cumsum(dim=sequence_dim)
    length_shape = [1] * input.dim()
    length_shape[sequence_dim] = input.shape[sequence_dim]
    lengths = torch.arange(1, input.shape[sequence_dim] + 1, device=input.device, dtype=input.dtype).view(*length_shape)
    counts = lengths * sample_count
    mean = prefix_sum / counts
    square_mean = prefix_square_sum / counts
    var = (square_mean - mean.square()).clamp_min(0.0)
    if eps is not None:
        var = var + eps
    return mean, var


class SequenceBatchNorm1dCentering(nn.Module):
    _version = 2
    __constants__ = ["track_running_stats", "momentum", "num_features", "affine"]

    def __init__(
        self,
        num_features: int,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        layout: str | None = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.layout = layout
        if self.affine:
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features, **factory_kwargs))
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(0, dtype=torch.long, **{k: v for k, v in factory_kwargs.items() if k != "dtype"}),
            )
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
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        exponential_average_factor = 0.0 if self.momentum is None else self.momentum
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)

        bn_training = self.training or self.running_mean is None
        _, reduce_dims, shape = _resolve_sequence_dim(input, self.layout, num_features=self.num_features)

        if bn_training:
            mean = torch.mean(input, dim=reduce_dims, keepdim=True)
            if self.track_running_stats:
                self.running_mean.mul_(1 - exponential_average_factor).add_(
                    mean.reshape(self.num_features), alpha=exponential_average_factor
                )
        else:
            mean = self.running_mean.view(*shape)

        output = input - mean
        if self.affine:
            output = output + self.bias.view(*shape)
        return output

    def extra_repr(self):
        return (
            "{num_features}, momentum={momentum}, affine={affine}, "
            "layout={layout}, track_running_stats={track_running_stats}".format(**self.__dict__)
        )


class SequenceBatchNorm1dScaling(nn.Module):
    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine", "bias"]

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        bias: bool = False,
        track_running_stats: bool = True,
        layout: str | None = None,
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
        self.layout = layout
        self.affine_bias = bias
        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            if self.affine_bias:
                self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_var", torch.ones(num_features, **factory_kwargs))
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(0, dtype=torch.long, **{k: v for k, v in factory_kwargs.items() if k != "dtype"}),
            )
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
            if self.affine_bias:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        exponential_average_factor = 0.0 if self.momentum is None else self.momentum
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)

        bn_training = self.training or self.running_var is None
        _, reduce_dims, shape = _resolve_sequence_dim(input, self.layout, num_features=self.num_features)

        if bn_training:
            var = torch.var(input, dim=reduce_dims, unbiased=False, keepdim=True)
            if self.track_running_stats:
                self.running_var.mul_(1 - exponential_average_factor).add_(
                    var.reshape(self.num_features), alpha=exponential_average_factor
                )
        else:
            var = self.running_var.view(*shape)

        output = input / torch.sqrt(var + self.eps)
        if self.affine:
            output = output * self.weight.view(*shape)
            if self.affine_bias:
                output = output + self.bias.view(*shape)
        return output

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "layout={layout}, track_running_stats={track_running_stats}".format(**self.__dict__)
        )


class SequenceBatchNorm1d(nn.Sequential):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        layout: str | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            SequenceBatchNorm1dCentering(
                num_features,
                momentum=momentum,
                affine=False,
                track_running_stats=track_running_stats,
                layout=layout,
                device=device,
                dtype=dtype,
            ),
            SequenceBatchNorm1dScaling(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                bias=affine,
                track_running_stats=track_running_stats,
                layout=layout,
                device=device,
                dtype=dtype,
            ),
        )


class SequenceDimBatchNorm1dCentering(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, layout: str = "last", causal: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.layout = layout
        self.causal = causal
        if self.affine:
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        sequence_dim, _, _ = _resolve_sequence_dim(input, self.layout)
        _, feature_shape = _resolve_feature_dim(input, self.layout, self.num_features)
        if self.causal:
            mean, _ = _prefix_stats(input, sequence_dim, reduce_non_sequence=False)
        else:
            mean = input.mean(dim=sequence_dim, keepdim=True)
        output = input - mean
        if self.affine:
            output = output + self.bias.view(*feature_shape)
        return output


class SequenceDimBatchNorm1dScaling(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        bias: bool = False,
        layout: str = "last",
        causal: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.affine_bias = bias
        self.layout = layout
        self.causal = causal
        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            if self.affine_bias:
                self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            if self.affine_bias:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        sequence_dim, _, _ = _resolve_sequence_dim(input, self.layout)
        _, feature_shape = _resolve_feature_dim(input, self.layout, self.num_features)
        if self.causal:
            _, var = _prefix_stats(input, sequence_dim, reduce_non_sequence=False)
        else:
            var = input.var(dim=sequence_dim, unbiased=False, keepdim=True)
        output = input / torch.sqrt(var + self.eps)
        if self.affine:
            output = output * self.weight.view(*feature_shape)
            if self.affine_bias:
                output = output + self.bias.view(*feature_shape)
        return output


class SequenceDimBatchNorm1d(nn.Sequential):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        layout: str = "last",
        causal: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            SequenceDimBatchNorm1dCentering(
                num_features,
                affine=False,
                layout=layout,
                causal=causal,
                device=device,
                dtype=dtype,
            ),
            SequenceDimBatchNorm1dScaling(
                num_features,
                eps=eps,
                affine=affine,
                bias=affine,
                layout=layout,
                causal=causal,
                device=device,
                dtype=dtype,
            ),
        )


class CausalSequenceBatchNorm1dCentering(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, layout: str = "last", device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.layout = layout
        if self.affine:
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        sequence_dim, _, shape = _resolve_sequence_dim(input, self.layout, num_features=self.num_features)
        mean, _ = _prefix_stats(input, sequence_dim, reduce_non_sequence=True)
        output = input - mean
        if self.affine:
            output = output + self.bias.view(*shape)
        return output


class CausalSequenceBatchNorm1dScaling(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        bias: bool = False,
        layout: str = "last",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.affine_bias = bias
        self.layout = layout
        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            if self.affine_bias:
                self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            if self.affine_bias:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        sequence_dim, _, shape = _resolve_sequence_dim(input, self.layout, num_features=self.num_features)
        _, var = _prefix_stats(input, sequence_dim, reduce_non_sequence=True)
        output = input / torch.sqrt(var + self.eps)
        if self.affine:
            output = output * self.weight.view(*shape)
            if self.affine_bias:
                output = output + self.bias.view(*shape)
        return output


class CausalSequenceBatchNorm1d(nn.Sequential):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, layout: str = "last", device=None, dtype=None) -> None:
        super().__init__(
            CausalSequenceBatchNorm1dCentering(
                num_features,
                affine=False,
                layout=layout,
                device=device,
                dtype=dtype,
            ),
            CausalSequenceBatchNorm1dScaling(
                num_features,
                eps=eps,
                affine=affine,
                bias=affine,
                layout=layout,
                device=device,
                dtype=dtype,
            ),
        )


class DynamicSequenceBatchNorm1dCentering(nn.Module):
    def __init__(self, affine: bool = False, layout: str = "last", device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.affine = affine
        self.layout = layout
        if self.affine:
            self.bias = Parameter(torch.empty(1, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        _, reduce_dims, _ = _resolve_sequence_dim(input, self.layout)
        mean = torch.mean(input, dim=reduce_dims, keepdim=True)
        output = input - mean
        if self.affine:
            output = output + self.bias
        return output


class DynamicSequenceBatchNorm1dScaling(nn.Module):
    def __init__(self, eps: float = 1e-5, affine: bool = False, bias: bool = False, layout: str = "last", device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.affine_bias = bias
        self.layout = layout
        if self.affine:
            self.weight = Parameter(torch.empty(1, **factory_kwargs))
            if self.affine_bias:
                self.bias = Parameter(torch.empty(1, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            if self.affine_bias:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        _, reduce_dims, _ = _resolve_sequence_dim(input, self.layout)
        var = torch.var(input, dim=reduce_dims, unbiased=False, keepdim=True)
        output = input / torch.sqrt(var + self.eps)
        if self.affine:
            output = output * self.weight
            if self.affine_bias:
                output = output + self.bias
        return output


class DynamicSequenceBatchNorm1d(nn.Sequential):
    def __init__(self, eps: float = 1e-5, affine: bool = False, layout: str = "last", device=None, dtype=None) -> None:
        super().__init__(
            DynamicSequenceBatchNorm1dCentering(
                affine=False,
                layout=layout,
                device=device,
                dtype=dtype,
            ),
            DynamicSequenceBatchNorm1dScaling(
                eps=eps,
                affine=affine,
                bias=affine,
                layout=layout,
                device=device,
                dtype=dtype,
            ),
        )


class CausalDynamicSequenceBatchNorm1dCentering(nn.Module):
    def __init__(self, affine: bool = False, layout: str = "last", device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.affine = affine
        self.layout = layout
        if self.affine:
            self.bias = Parameter(torch.empty(1, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        sequence_dim, _, _ = _resolve_sequence_dim(input, self.layout)
        mean, _ = _prefix_stats(input, sequence_dim, reduce_non_sequence=True)
        output = input - mean
        if self.affine:
            output = output + self.bias
        return output


class CausalDynamicSequenceBatchNorm1dScaling(nn.Module):
    def __init__(self, eps: float = 1e-5, affine: bool = False, bias: bool = False, layout: str = "last", device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.affine_bias = bias
        self.layout = layout
        if self.affine:
            self.weight = Parameter(torch.empty(1, **factory_kwargs))
            if self.affine_bias:
                self.bias = Parameter(torch.empty(1, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            if self.affine_bias:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        sequence_dim, _, _ = _resolve_sequence_dim(input, self.layout)
        _, var = _prefix_stats(input, sequence_dim, reduce_non_sequence=True)
        output = input / torch.sqrt(var + self.eps)
        if self.affine:
            output = output * self.weight
            if self.affine_bias:
                output = output + self.bias
        return output


class CausalDynamicSequenceBatchNorm1d(nn.Sequential):
    def __init__(self, eps: float = 1e-5, affine: bool = False, layout: str = "last", device=None, dtype=None) -> None:
        super().__init__(
            CausalDynamicSequenceBatchNorm1dCentering(
                affine=False,
                layout=layout,
                device=device,
                dtype=dtype,
            ),
            CausalDynamicSequenceBatchNorm1dScaling(
                eps=eps,
                affine=affine,
                bias=affine,
                layout=layout,
                device=device,
                dtype=dtype,
            ),
        )

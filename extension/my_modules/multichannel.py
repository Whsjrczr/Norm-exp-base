import re

import torch
import torch.nn as nn


def _to_channel_list(indices, num_channels):
    if indices is None:
        return None
    if isinstance(indices, int):
        indices = [indices]
    out = sorted({int(idx) for idx in indices})
    for idx in out:
        if idx < 0 or idx >= num_channels:
            raise ValueError(f"Channel index {idx} is out of range for num_channels={num_channels}.")
    return out


class MultiChannelLinear(nn.Module):
    """Block-diagonal linear layer with no interaction across channels."""

    def __init__(self, in_features, out_features, num_channels=1, bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_channels = int(num_channels)
        if self.num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, but got {self.num_channels}.")
        if self.in_features % self.num_channels != 0:
            raise ValueError(
                f"in_features={self.in_features} must be divisible by num_channels={self.num_channels}."
            )
        if self.out_features % self.num_channels != 0:
            raise ValueError(
                f"out_features={self.out_features} must be divisible by num_channels={self.num_channels}."
            )

        self.in_features_per_channel = self.in_features // self.num_channels
        self.out_features_per_channel = self.out_features // self.num_channels

        self.weight = nn.Parameter(
            torch.empty(
                self.num_channels,
                self.out_features_per_channel,
                self.in_features_per_channel,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_channels, self.out_features_per_channel))
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "weight_train_mask",
            torch.ones(self.num_channels, 1, 1, dtype=torch.bool),
        )
        self.register_buffer("weight_frozen_value", torch.empty_like(self.weight))
        if self.bias is not None:
            self.register_buffer(
                "bias_train_mask",
                torch.ones(self.num_channels, 1, dtype=torch.bool),
            )
            self.register_buffer("bias_frozen_value", torch.empty_like(self.bias))
        else:
            self.register_buffer("bias_train_mask", None)
            self.register_buffer("bias_frozen_value", None)

        self.reset_parameters()

    def reset_parameters(self):
        for channel_idx in range(self.num_channels):
            nn.init.kaiming_uniform_(self.weight[channel_idx], a=5 ** 0.5)
        if self.bias is not None:
            bound = self.in_features_per_channel ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)
        self.weight_frozen_value.copy_(self.weight.detach())
        if self.bias is not None:
            self.bias_frozen_value.copy_(self.bias.detach())

    def set_trainable_channels(self, trainable_channels=None, frozen_channels=None):
        trainable_channels = _to_channel_list(trainable_channels, self.num_channels)
        frozen_channels = _to_channel_list(frozen_channels, self.num_channels)

        mask = torch.ones(self.num_channels, dtype=torch.bool, device=self.weight.device)
        if trainable_channels is not None:
            mask.zero_()
            mask[trainable_channels] = True
        if frozen_channels is not None:
            mask[frozen_channels] = False

        self.weight_train_mask.copy_(mask.view(self.num_channels, 1, 1))
        self.weight_frozen_value.copy_(self.weight.detach())
        if self.bias is not None:
            self.bias_train_mask.copy_(mask.view(self.num_channels, 1))
            self.bias_frozen_value.copy_(self.bias.detach())

    def effective_weight(self):
        return torch.where(self.weight_train_mask, self.weight, self.weight_frozen_value)

    def effective_bias(self):
        if self.bias is None:
            return None
        return torch.where(self.bias_train_mask, self.bias, self.bias_frozen_value)

    def as_dense_weight(self):
        blocks = [self.effective_weight()[idx] for idx in range(self.num_channels)]
        return torch.block_diag(*blocks)

    def forward(self, x):
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.num_channels, self.in_features_per_channel)
        y = torch.einsum("bci,coi->bco", x, self.effective_weight())
        bias = self.effective_bias()
        if bias is not None:
            y = y + bias.unsqueeze(0)
        y = y.reshape(*original_shape, self.out_features)
        return y

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_channels={self.num_channels}, bias={self.bias is not None}"
        )


class MultiChannelMLP(nn.Module):
    """Grouped MLP trunk with independent channels and an optional merge head."""

    def __init__(
        self,
        input_size,
        output_size,
        width=100,
        depth=4,
        num_channels=1,
        activation_factory=None,
        norm_factory=None,
        dropout_prob=0.0,
        bias=True,
        final_layer="merge",
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.width = int(width)
        self.depth = int(depth)
        self.num_channels = int(num_channels)
        self.final_layer = str(final_layer).lower()

        if self.input_size % self.num_channels != 0:
            raise ValueError(
                f"input_size={self.input_size} must be divisible by num_channels={self.num_channels} "
                "to keep channels independent."
            )
        if self.width % self.num_channels != 0:
            raise ValueError(
                f"width={self.width} must be divisible by num_channels={self.num_channels}."
            )
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, but got depth={self.depth}.")

        if activation_factory is None:
            activation_factory = lambda num_features: nn.ReLU()
        if norm_factory is None:
            norm_factory = None

        layers = [nn.Flatten(start_dim=1)]
        layers.append(MultiChannelLinear(self.input_size, self.width, num_channels=self.num_channels, bias=bias))
        if norm_factory is not None:
            layers.append(MultiChannelNorm(self.width, num_channels=self.num_channels, norm_factory=norm_factory))
        layers.append(activation_factory(self.width))

        for _ in range(self.depth - 1):
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            layers.append(MultiChannelLinear(self.width, self.width, num_channels=self.num_channels, bias=bias))
            if norm_factory is not None:
                layers.append(MultiChannelNorm(self.width, num_channels=self.num_channels, norm_factory=norm_factory))
            layers.append(activation_factory(self.width))

        self.trunk = nn.Sequential(*layers)
        if self.final_layer == "grouped":
            self.head = MultiChannelLinear(
                self.width,
                self.output_size,
                num_channels=self.num_channels,
                bias=bias,
            )
        elif self.final_layer == "merge":
            self.head = nn.Linear(self.width, self.output_size, bias=bias)
        else:
            raise ValueError(f"Unsupported final_layer={final_layer}. Choose from: merge, grouped.")

    def set_trainable_channels(self, trainable_channels=None, frozen_channels=None):
        for module in self.modules():
            if isinstance(module, MultiChannelLinear):
                module.set_trainable_channels(
                    trainable_channels=trainable_channels,
                    frozen_channels=frozen_channels,
                )

    def forward(self, x):
        x = self.trunk(x)
        return self.head(x)

    def extra_repr(self):
        return (
            f"input_size={self.input_size}, output_size={self.output_size}, width={self.width}, "
            f"depth={self.depth}, num_channels={self.num_channels}, final_layer={self.final_layer}"
        )


class MultiChannelNorm(nn.Module):
    """Apply one normalization module per channel group."""

    def __init__(self, num_features, num_channels=1, norm_factory=None):
        super().__init__()
        self.num_features = int(num_features)
        self.num_channels = int(num_channels)
        if self.num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, but got {self.num_channels}.")
        if self.num_features % self.num_channels != 0:
            raise ValueError(
                f"num_features={self.num_features} must be divisible by num_channels={self.num_channels}."
            )
        if norm_factory is None:
            raise ValueError("MultiChannelNorm requires a norm_factory.")
        self.features_per_channel = self.num_features // self.num_channels
        self.norms = nn.ModuleList(
            norm_factory(self.features_per_channel) for _ in range(self.num_channels)
        )

    def forward(self, x):
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.num_channels, self.features_per_channel)
        outputs = []
        for idx, norm in enumerate(self.norms):
            outputs.append(norm(x[:, idx, :]))
        y = torch.stack(outputs, dim=1)
        return y.reshape(*original_shape, self.num_features)

    def extra_repr(self):
        return f"num_features={self.num_features}, num_channels={self.num_channels}"


def apply_parameter_freeze(model, freeze_patterns=None, train_patterns=None):
    freeze_patterns = list(freeze_patterns or [])
    train_patterns = list(train_patterns or [])
    compiled_freeze = [re.compile(pattern) for pattern in freeze_patterns]
    compiled_train = [re.compile(pattern) for pattern in train_patterns]

    for name, param in model.named_parameters():
        if any(pattern.search(name) for pattern in compiled_freeze):
            param.requires_grad = False
        if any(pattern.search(name) for pattern in compiled_train):
            param.requires_grad = True


def apply_channel_freeze(model, trainable_channels=None, frozen_channels=None):
    has_multichannel_module = False
    for module in model.modules():
        if isinstance(module, MultiChannelLinear):
            has_multichannel_module = True
            module.set_trainable_channels(
                trainable_channels=trainable_channels,
                frozen_channels=frozen_channels,
            )
    return has_multichannel_module


def summarize_trainability(model):
    trainable = []
    frozen = []
    for name, param in model.named_parameters():
        item = f"{name}: shape={tuple(param.shape)}"
        if param.requires_grad:
            trainable.append(item)
        else:
            frozen.append(item)
    return {"trainable": trainable, "frozen": frozen}


def collect_multichannel_state(model):
    states = []
    for module_name, module in model.named_modules():
        if not isinstance(module, MultiChannelLinear):
            continue
        mask = module.weight_train_mask[:, 0, 0].detach().cpu().tolist()
        trainable_channels = [idx for idx, enabled in enumerate(mask) if enabled]
        frozen_channels = [idx for idx, enabled in enumerate(mask) if not enabled]
        states.append(
            {
                "module": module_name,
                "num_channels": module.num_channels,
                "trainable_channels": trainable_channels,
                "frozen_channels": frozen_channels,
            }
        )
    return states

import math
import torch.nn as nn


INIT_PRESETS = (
    "default",
    "none",
    "xavier",
    "xavier_uniform",
    "xavier_normal",
    "kaiming",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
    "trunc_normal",
    "normal",
)


def add_initialization_arguments(group):
    group.add_argument(
        "--init-preset",
        default="default",
        choices=INIT_PRESETS,
        help=(
            "preset initialization for trainable weights; "
            "default/none keeps each model's native init behavior"
        ),
    )
    group.add_argument(
        "--init-gain",
        type=float,
        default=1.0,
        help="gain used by xavier/orthogonal initialization presets",
    )
    group.add_argument(
        "--init-std",
        type=float,
        default=0.02,
        help="std used by normal/trunc_normal initialization presets",
    )
    group.add_argument(
        "--init-bias",
        type=float,
        default=0.0,
        help="constant value used to initialize layer bias when present",
    )
    return group


def normalize_init_preset(name):
    preset = str(name).lower()
    if preset in {"default", "none", "origin"}:
        return "default"
    if preset == "xavier":
        return "xavier_uniform"
    if preset == "kaiming":
        return "kaiming_uniform"
    return preset


def maybe_map_kan_init(cfg, family: str):
    preset = normalize_init_preset(getattr(cfg, "init_preset", "default"))
    if family != "kan":
        return
    if getattr(cfg, "kan_init", "origin") != "origin":
        return
    if preset == "xavier_uniform":
        cfg.kan_init = "xavier"
    elif preset == "kaiming_uniform":
        cfg.kan_init = "kaiming"


def _should_init_module(module):
    if isinstance(
        module,
        (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Embedding,
        ),
    ):
        return True
    return module.__class__.__name__ == "MultiChannelLinear"


def _apply_weight_init(weight, preset, gain, std):
    if preset == "xavier_uniform":
        nn.init.xavier_uniform_(weight, gain=gain)
    elif preset == "xavier_normal":
        nn.init.xavier_normal_(weight, gain=gain)
    elif preset == "kaiming_uniform":
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5), nonlinearity="relu")
    elif preset == "kaiming_normal":
        nn.init.kaiming_normal_(weight, a=math.sqrt(5), nonlinearity="relu")
    elif preset == "orthogonal":
        nn.init.orthogonal_(weight, gain=gain)
    elif preset == "trunc_normal":
        nn.init.trunc_normal_(weight, std=std)
    elif preset == "normal":
        nn.init.normal_(weight, std=std)
    else:
        raise ValueError(f"Unsupported init preset: {preset}")


def apply_init_preset(model, cfg):
    preset = normalize_init_preset(getattr(cfg, "init_preset", "default"))
    if preset == "default":
        return {
            "preset": "default",
            "num_weight_tensors": 0,
            "num_bias_tensors": 0,
        }

    gain = float(getattr(cfg, "init_gain", 1.0))
    std = float(getattr(cfg, "init_std", 0.02))
    bias_value = float(getattr(cfg, "init_bias", 0.0))

    num_weight_tensors = 0
    num_bias_tensors = 0
    for module in model.modules():
        if not _should_init_module(module):
            continue
        if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
            _apply_weight_init(module.weight, preset, gain, std)
            num_weight_tensors += 1
        if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
            nn.init.constant_(module.bias, bias_value)
            num_bias_tensors += 1

    return {
        "preset": preset,
        "num_weight_tensors": num_weight_tensors,
        "num_bias_tensors": num_bias_tensors,
    }


__all__ = [
    "INIT_PRESETS",
    "add_initialization_arguments",
    "apply_init_preset",
    "maybe_map_kan_init",
    "normalize_init_preset",
]

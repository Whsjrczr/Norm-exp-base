import argparse

from .utils import str2list
from .my_modules.multichannel import (
    MultiChannelLinear,
    MultiChannelMLP,
    MultiChannelNorm,
    apply_channel_freeze,
    collect_multichannel_state,
    apply_parameter_freeze,
    summarize_trainability,
)


class _config:
    multi_channels = 1
    multi_final_layer = "merge"
    multi_norm = False
    freeze_patterns = []
    train_patterns = []
    freeze_channels = None
    trainable_channels = None


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("MultiChannel Option:")
    group.add_argument("--multi-channels", type=int, default=1, help="number of independent channels in grouped MLP")
    group.add_argument(
        "--multi-final-layer",
        type=str,
        default="merge",
        choices=["merge", "grouped"],
        help="final layer type for MultiChannelMLP",
    )
    group.add_argument(
        "--multi-norm",
        action="store_true",
        help="apply one independent normalization module per channel after each grouped linear",
    )
    group.add_argument(
        "--freeze-patterns",
        type=str2list,
        default=[],
        metavar="LIST",
        help="comma-separated regex patterns for fully frozen parameters",
    )
    group.add_argument(
        "--train-patterns",
        type=str2list,
        default=[],
        metavar="LIST",
        help="comma-separated regex patterns to re-enable trainable parameters",
    )
    group.add_argument(
        "--freeze-channels",
        type=str2list,
        default=None,
        metavar="LIST",
        help="comma-separated grouped channel indices to freeze inside MultiChannelLinear",
    )
    group.add_argument(
        "--trainable-channels",
        type=str2list,
        default=None,
        metavar="LIST",
        help="comma-separated grouped channel indices to keep trainable inside MultiChannelLinear",
    )
    return group


def setting(cfg: argparse.Namespace):
    for key, value in vars(cfg).items():
        if key in _config.__dict__:
            setattr(_config, key, value)
    flag = f"mc{_config.multi_channels}_{_config.multi_final_layer}"
    if _config.multi_norm:
        flag += "_mnorm"
    if _config.trainable_channels is not None:
        flag += "_tc" + "-".join(str(v) for v in _config.trainable_channels)
    if _config.freeze_channels is not None:
        flag += "_fc" + "-".join(str(v) for v in _config.freeze_channels)
    return flag


def configure_model(model, cfg: argparse.Namespace):
    apply_parameter_freeze(
        model,
        freeze_patterns=getattr(cfg, "freeze_patterns", []),
        train_patterns=getattr(cfg, "train_patterns", []),
    )
    apply_channel_freeze(
        model,
        trainable_channels=getattr(cfg, "trainable_channels", None),
        frozen_channels=getattr(cfg, "freeze_channels", None),
    )
    return model


def summarize_freeze_state(model):
    trainable_params = 0
    frozen_params = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()

    named_summary = summarize_trainability(model)
    channel_state = collect_multichannel_state(model)
    summary = {
        "trainable_params": int(trainable_params),
        "frozen_params": int(frozen_params),
        "trainable_tensors": len(named_summary["trainable"]),
        "frozen_tensors": len(named_summary["frozen"]),
        "trainable_names": named_summary["trainable"],
        "frozen_names": named_summary["frozen"],
        "multichannel_modules": channel_state,
    }
    summary["has_frozen_params"] = summary["frozen_params"] > 0 or any(
        item["frozen_channels"] for item in channel_state
    )
    return summary


def format_freeze_summary(summary):
    lines = [
        "Freeze summary:",
        f"  trainable_params={summary['trainable_params']}",
        f"  frozen_params={summary['frozen_params']}",
        f"  trainable_tensors={summary['trainable_tensors']}",
        f"  frozen_tensors={summary['frozen_tensors']}",
    ]
    if summary["multichannel_modules"]:
        lines.append("  multichannel_modules:")
        for item in summary["multichannel_modules"]:
            lines.append(
                "    "
                + f"{item['module']}: trainable_channels={item['trainable_channels']}, "
                + f"frozen_channels={item['frozen_channels']}"
            )
    if summary["frozen_names"]:
        lines.append("  frozen_parameter_names:")
        for name in summary["frozen_names"]:
            lines.append("    " + name)
    return "\n".join(lines)


def get_runtime_config(cfg: argparse.Namespace):
    return {
        "multichannel": getattr(cfg, "arch", None) == "MultiChannelMLP",
        "multichannel_cfg": setting(cfg) if getattr(cfg, "arch", None) == "MultiChannelMLP" else None,
        "multi_channels": getattr(cfg, "multi_channels", None),
        "multi_final_layer": getattr(cfg, "multi_final_layer", None),
        "multi_norm": getattr(cfg, "multi_norm", False),
        "freeze_patterns": getattr(cfg, "freeze_patterns", []),
        "train_patterns": getattr(cfg, "train_patterns", []),
        "freeze_channels": getattr(cfg, "freeze_channels", None),
        "trainable_channels": getattr(cfg, "trainable_channels", None),
    }


def log_runtime_summary(logger, cfg: argparse.Namespace, freeze_summary: dict):
    runtime_cfg = get_runtime_config(cfg)
    logger("==> config: {}".format(cfg))
    logger("==> multichannel config: {}".format(runtime_cfg))
    logger("==> " + format_freeze_summary(freeze_summary).replace("\n", "\n==> "))


__all__ = [
    "MultiChannelLinear",
    "MultiChannelMLP",
    "MultiChannelNorm",
    "add_arguments",
    "setting",
    "configure_model",
    "get_runtime_config",
    "log_runtime_summary",
    "summarize_freeze_state",
    "format_freeze_summary",
    "summarize_trainability",
]

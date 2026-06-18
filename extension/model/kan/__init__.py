import argparse
import torch.nn as nn

from extension.utils import str2list


KAN_MODEL_NAMES = ["KAN", "MLP"]


def add_kan_arguments(parser):
    group = parser.add_argument_group("KAN Model Options")
    group.add_argument("--layers-hidden", type=str2list, default=None)
    group.add_argument("--input-dim", type=int, default=3)
    group.add_argument("--output-dim", type=int, default=1)
    group.add_argument("--grid-size", type=int, default=5)
    group.add_argument("--spline-order", type=int, default=3)
    group.add_argument("--scale-noise", type=float, default=0.01)
    group.add_argument("--scale-base", type=float, default=1.0)
    group.add_argument("--scale-spline", type=float, default=1.0)
    group.add_argument("--grid-eps", type=float, default=0.02)
    group.add_argument("--grid-range", type=str2list, default="-1,1")
    group.add_argument("--update-grid", action="store_true")
    group.add_argument("--weight-norm", action="store_true")
    group.add_argument("--kan-regularization", type=float, default=0.0)
    group.add_argument("--kan-init", default="origin", choices=["origin", "xavier", "kaiming"])
    group.add_argument(
        "--residual-activation",
        default="same",
        choices=["same", "relu", "sigmoid", "tanh", "silu", "no"],
        help="activation used on the KAN residual/base branch; 'same' follows --activation",
    )
    group.add_argument(
        "--disable-residual-branch",
        action="store_true",
        help="disable the KAN residual/base branch and keep only the spline branch",
    )
    group.add_argument(
        "--no-base-branch",
        dest="disable_residual_branch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return group


def _resolve_input_dim(cfg):
    if getattr(cfg, "layers_hidden", None):
        return int(cfg.layers_hidden[0])
    if getattr(cfg, "input_dim", None) is not None:
        return int(cfg.input_dim)
    if getattr(cfg, "im_size", None) is not None:
        input_size = 1
        for dim in cfg.im_size:
            input_size *= int(dim)
        return input_size
    return 3


def _resolve_output_dim(cfg):
    if getattr(cfg, "layers_hidden", None):
        return int(cfg.layers_hidden[-1])
    if getattr(cfg, "dataset_classes", None) is not None:
        return int(cfg.dataset_classes)
    if getattr(cfg, "output_dim", None) is not None:
        return int(cfg.output_dim)
    return 1


def _resolve_layers_hidden(cfg):
    if getattr(cfg, "layers_hidden", None):
        return list(cfg.layers_hidden)

    depth = max(int(getattr(cfg, "depth", 3)), 1)
    width = int(getattr(cfg, "width", 64))
    input_dim = _resolve_input_dim(cfg)
    output_dim = _resolve_output_dim(cfg)
    return [input_dim] + [width] * depth + [output_dim]


def _activation_name_to_module(activation_name: str):
    activation_map = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "no": nn.Identity,
    }
    return activation_map.get(activation_name, nn.SiLU)


def _resolve_base_activation(cfg):
    activation_name = str(getattr(cfg, "activation", "relu")).lower()
    residual_activation = str(getattr(cfg, "residual_activation", "same")).lower()
    if residual_activation != "same":
        activation_name = residual_activation
    return _activation_name_to_module(activation_name)


def get_kan_model(cfg):
    import extension as ext
    from .KAN import KAN_norm
    from .MLP import MLP

    model_name = getattr(cfg, "arch", None) or "KAN"
    layers_hidden = _resolve_layers_hidden(cfg)
    disable_residual_branch = bool(getattr(cfg, "disable_residual_branch", False))
    norm_2d = ext.make_norm_factory(dim=2)

    if model_name == "KAN":
        return KAN_norm(
            layers_hidden=layers_hidden,
            grid_size=getattr(cfg, "grid_size", 5),
            spline_order=getattr(cfg, "spline_order", 3),
            scale_noise=getattr(cfg, "scale_noise", 0.01),
            scale_base=getattr(cfg, "scale_base", 1.0),
            scale_spline=getattr(cfg, "scale_spline", 1.0),
            base_activation=_resolve_base_activation(cfg),
            grid_eps=getattr(cfg, "grid_eps", 0.02),
            grid_range=getattr(cfg, "grid_range", [-1, 1]),
            norm=norm_2d,
            upgrade_grid=getattr(cfg, "update_grid", False),
            weight_norm=getattr(cfg, "weight_norm", False),
            init=getattr(cfg, "kan_init", "origin"),
            dropout_rate=getattr(cfg, "dropout", 0.0),
            use_base_branch=not disable_residual_branch,
        )

    if model_name == "MLP":
        return MLP(
            layers_hidden,
            norm=norm_2d,
            activation=_activation_name_to_module(str(getattr(cfg, "activation", "relu")).lower()),
        )

    raise ValueError(f"Invalid KAN model name: {model_name}. Choose from {KAN_MODEL_NAMES}.")


__all__ = [
    "KAN_MODEL_NAMES",
    "add_kan_arguments",
    "get_kan_model",
]

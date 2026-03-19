import os
import sys

import torch
import torch.nn as nn

from .KAN import KAN_norm
from .MLP import MLP

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import extension as ext


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


def _resolve_base_activation(cfg):
    activation_name = str(getattr(cfg, "activation", "relu")).lower()
    activation_map = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "no": nn.Identity,
    }
    return activation_map.get(activation_name, nn.SiLU)


def get_model(cfg):
    model_name = getattr(cfg, "arch", "KAN")
    layers_hidden = _resolve_layers_hidden(cfg)

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
            norm=ext.normalization.Norm,
            upgrade_grid=getattr(cfg, "update_grid", False),
            weight_norm=getattr(cfg, "weight_norm", False),
            init=getattr(cfg, "kan_init", "origin"),
            dropout_rate=getattr(cfg, "dropout", 0.0),
            use_base_branch=not getattr(cfg, "no_base_branch", False),
        )

    if model_name == "MLP":
        return MLP(layers_hidden, norm=ext.normalization.Norm, activation=_resolve_base_activation(cfg))

    raise ValueError(f"Invalid model name: {model_name}. Choose 'KAN' or 'MLP'.")

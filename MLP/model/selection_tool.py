import argparse

from . import *


CLASSIFICATION_MODEL_NAMES = [
    "MLP",
    "ResCenDropScalingMLP",
    "CenDropScalingMLP",
    "CenDropScalingPreNormMLP",
    "LinearModel",
    "Linear",
    "resnet18",
    "resnet34",
    "resnet50",
    "MLPReLU",
    "PreNormMLP",
    "ConvBN",
    "ConvBNPre",
    "ConvBNRes",
    "ConvBNResPre",
    "ConvLN",
    "ConvLNPre",
    "ConvLNRes",
    "ConvLNResPre",
]

PDE_MODEL_NAMES = [
    "MLP",
    "PreNormMLP",
    "CenDropScalingMLP",
    "CenDropScalingPreNormMLP",
    "ResCenDropScalingMLP",
]


def add_model_arguments(parser: argparse.ArgumentParser, task: str = "classification"):
    if task == "pde":
        model_names = PDE_MODEL_NAMES
        default_arch = "MLP"
        default_width = 50
        default_depth = 3
        default_dropout = 0.0
    else:
        model_names = CLASSIFICATION_MODEL_NAMES
        default_arch = "MLP"
        default_width = 100
        default_depth = 4
        default_dropout = 0.0

    group = parser.add_argument_group("Model Options")
    group.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default=default_arch,
        choices=model_names,
        help="model architecture: " + " | ".join(model_names),
    )
    group.add_argument("-width", "--width", type=int, default=default_width)
    group.add_argument("-depth", "--depth", type=int, default=default_depth)
    group.add_argument("-dropout", "--dropout", type=float, default=default_dropout)
    return group


def _resolve_input_size(cfg):
    input_size = 1
    for dim in cfg.im_size:
        input_size *= dim
    return input_size


def get_model(cfg):
    model_name = cfg.arch
    model_width = cfg.width
    model_depth = cfg.depth
    output_size = cfg.dataset_classes
    dropout_prob = cfg.dropout
    input_size = _resolve_input_size(cfg)

    if model_name == "MLP":
        model_out = MLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size, dropout_prob=dropout_prob)
    elif model_name == "CenDropScalingMLP":
        model_out = CenDropScalingMLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size, dropout_prob=dropout_prob)
    elif model_name == "CenDropScalingPreNormMLP":
        model_out = CenDropScalingPreNormMLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size, dropout_prob=dropout_prob)
    elif model_name == "PreNormMLP":
        model_out = PreNormMLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size, dropout_prob=dropout_prob)
    elif model_name == "resnet18":
        model_out = resnet18(num_classes=output_size)
    elif model_name == "resnet34":
        model_out = resnet34(num_classes=output_size)
    elif model_name == "resnet50":
        model_out = resnet50(num_classes=output_size)
    elif model_name == "ResCenDropScalingMLP":
        model_out = ResCenDropScalingMLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size, dropout_prob=dropout_prob)
    elif model_name == "ResMLP":
        model_out = ResMLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size, dropout_prob=dropout_prob)
    elif model_name == "ConvBN":
        model_out = ConvBN(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size)
    elif model_name == "ConvBNPre":
        model_out = ConvBNPre(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size)
    elif model_name == "ConvBNRes":
        model_out = ConvBNRes(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size)
    elif model_name == "ConvBNResPre":
        model_out = ConvBNResPre(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size)
    elif model_name == "ConvLN":
        model_out = ConvLN(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size)
    elif model_name == "ConvLNPre":
        model_out = ConvLNPre(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size)
    elif model_name == "ConvLNRes":
        model_out = ConvLNRes(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size)
    elif model_name == "ConvLNResPre":
        model_out = ConvLNResPre(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size)
    else:
        model_out = MLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size)
    return model_out

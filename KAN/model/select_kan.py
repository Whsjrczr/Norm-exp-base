import argparse

from extension.model import add_model_arguments as _add_model_arguments
from extension.model import get_model as _get_model
from extension.model.kan import KAN_MODEL_NAMES as MODEL_NAMES


def add_model_arguments(parser: argparse.ArgumentParser):
    group = _add_model_arguments(parser, task="classification", default_family="kan")
    parser.set_defaults(width=32, depth=3, arch="KAN")
    return group


def get_model(cfg):
    if getattr(cfg, "model_family", None) is None:
        cfg.model_family = "kan"
    return _get_model(cfg)

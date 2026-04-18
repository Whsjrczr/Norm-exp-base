import argparse

from extension.model import add_model_arguments as _add_model_arguments
from extension.model import build_vit_norm_layer
from extension.model import get_model as _get_model
from extension.model.vit import VIT_MODEL_NAMES as MODEL_NAMES


def add_model_arguments(parser: argparse.ArgumentParser):
    return _add_model_arguments(parser, task="classification", default_family="vit")


def get_model(cfg):
    if getattr(cfg, "model_family", None) is None:
        cfg.model_family = "vit"
    return _get_model(cfg)

import argparse

from extension.model import add_model_arguments as _add_model_arguments
from extension.model import get_model as _get_model
from extension.model.mlp import MLP_CLASSIFICATION_MODEL_NAMES as CLASSIFICATION_MODEL_NAMES
from extension.model.mlp import MLP_PDE_MODEL_NAMES as PDE_MODEL_NAMES


def add_model_arguments(parser: argparse.ArgumentParser, task: str = "classification"):
    return _add_model_arguments(parser, task=task, default_family="mlp")


def get_model(cfg):
    if getattr(cfg, "model_family", None) is None:
        cfg.model_family = "mlp"
    return _get_model(cfg)

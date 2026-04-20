import argparse

from .kan import KAN_MODEL_NAMES, add_kan_arguments, get_kan_model
from .initialization import (
    add_initialization_arguments,
    apply_init_preset,
    maybe_map_kan_init,
)
from .mlp import MLP_CLASSIFICATION_MODEL_NAMES, MLP_PDE_MODEL_NAMES, add_mlp_arguments, get_mlp_model
from .vit import VIT_MODEL_NAMES, add_vit_arguments, build_vit_norm_layer, get_vit_model


MODEL_FAMILIES = ("mlp", "kan", "vit")


def _all_model_names():
    return sorted(
        set(MLP_CLASSIFICATION_MODEL_NAMES)
        | set(MLP_PDE_MODEL_NAMES)
        | set(KAN_MODEL_NAMES)
        | set(VIT_MODEL_NAMES)
    )


def _infer_family_from_arch(arch: str):
    if arch in VIT_MODEL_NAMES:
        return "vit"
    if arch in KAN_MODEL_NAMES:
        return "kan"
    if arch in MLP_CLASSIFICATION_MODEL_NAMES or arch in MLP_PDE_MODEL_NAMES:
        return "mlp"
    return None


def resolve_model_family(cfg, default_family: str = "mlp"):
    family = getattr(cfg, "model_family", None)
    if family:
        return str(family).lower()

    inferred = _infer_family_from_arch(getattr(cfg, "arch", None))
    if inferred is not None:
        return inferred
    return default_family


def add_model_arguments(
    parser: argparse.ArgumentParser,
    task: str = "classification",
    default_family: str = "mlp",
):
    default_arch = {
        "mlp": "MLP",
        "kan": "KAN",
        "vit": "vit_small",
    }[default_family]
    group = parser.add_argument_group("Model Options")
    group.add_argument(
        "--model-family",
        dest="model_family",
        default=default_family,
        choices=MODEL_FAMILIES,
        help="model family: " + " | ".join(MODEL_FAMILIES),
    )
    group.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default=default_arch,
        help="model architecture, available names: " + " | ".join(_all_model_names()),
    )
    add_initialization_arguments(group)

    add_mlp_arguments(parser, task=task)
    add_kan_arguments(parser)
    add_vit_arguments(parser)
    return group


def get_model(cfg):
    family = resolve_model_family(cfg)
    maybe_map_kan_init(cfg, family)

    if family == "mlp":
        model = get_mlp_model(cfg)
    elif family == "kan":
        model = get_kan_model(cfg)
    elif family == "vit":
        model = get_vit_model(cfg)
    else:
        raise ValueError(f"Unknown model family: {family}")

    init_summary = apply_init_preset(model, cfg)
    setattr(model, "init_summary", init_summary)
    return model


__all__ = [
    "MODEL_FAMILIES",
    "add_model_arguments",
    "apply_init_preset",
    "build_vit_norm_layer",
    "get_model",
    "resolve_model_family",
]

import torch.nn as nn

import extension as ext
from extension.model.nanogpt import MeanShiftNormWrapper

from . import vision_transformer as vits


VIT_MODEL_NAMES = ["vit_tiny", "vit_small", "vit_base"]

FIXED_SEQUENCE_NORM_NAMES = {"SeqBN", "SeqBNc", "SeqBNs", "CSeqBN", "CSeqBNc", "CSeqBNs"}
DYNAMIC_SEQUENCE_NORM_NAMES = {"DSeqBN", "DSeqBNc", "DSeqBNs", "CDSeqBN", "CDSeqBNc", "CDSeqBNs"}


def add_vit_arguments(parser):
    group = parser.add_argument_group("ViT Model Options")
    group.add_argument("--patch-size", dest="patch_size", type=int, default=16)
    group.add_argument("--in-chans", dest="in_chans", type=int, default=3)
    group.add_argument("--num_classes", type=int, default=None)
    group.add_argument("--drop-path-rate", dest="drop_path_rate", type=float, default=0.1)
    return group


def _resolve_image_size(cfg):
    image_size = getattr(cfg, "im_size", [getattr(cfg, "image_size", 224)])
    if isinstance(image_size, (tuple, list)):
        return int(image_size[-1])
    return int(image_size)


def _resolve_vit_sequence_length(cfg):
    image_size = _resolve_image_size(cfg)
    patch_size = int(getattr(cfg, "patch_size", 16))
    num_patches = (image_size // patch_size) * (image_size // patch_size)
    return num_patches + 1


def _split_csv(value):
    if value is None or value == "":
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _selected_norm_sites(cfg):
    valid = ("norm1", "norm2", "final")
    site = getattr(cfg, "norm_site", None)
    sites = _split_csv(getattr(cfg, "norm_sites", None))
    if site in (None, "none") and not sites:
        return None
    if site == "all":
        selected = set(valid)
    else:
        selected = {site} if site not in (None, "none") else set()
    selected.update(sites)
    return {item for item in selected if item in valid}


def _slot_norm_name(cfg, slot):
    selected = _selected_norm_sites(cfg)
    if selected is not None:
        return getattr(cfg, "norm", "LN") if slot in selected else "LN"
    return getattr(cfg, "norm", "LN")


def _full_norm_name(norm_name):
    names = [name.strip() for name in str(norm_name).split("+") if name.strip()]
    out = []
    for name in names:
        candidate = name[:-1] if name.endswith("s") else name
        out.append(candidate if candidate in ext.normalization._config.norm_methods else name)
    return "+".join(out)


def _range_rescue_applies(cfg, block_idx, rescue):
    if block_idx is None or rescue not in {"early", "middle", "late"}:
        return False
    n_layer = 12
    third = max(1, n_layer // 3)
    if rescue == "early":
        return block_idx < third
    if rescue == "middle":
        return third <= block_idx < min(n_layer, 2 * third)
    return block_idx >= min(n_layer, 2 * third)


def _rescue_applies(cfg, slot, block_idx):
    rescue = getattr(cfg, "centering_rescue", "none") or "none"
    if rescue == "none":
        return False
    if rescue == "all":
        return True
    if rescue in {"norm1", "norm2", "final"}:
        return slot == rescue
    if rescue == "attn":
        return slot == "norm1"
    if rescue == "mlp":
        return slot == "norm2"
    return _range_rescue_applies(cfg, block_idx, rescue)


def intervention_suffix(cfg):
    parts = []
    selected = _selected_norm_sites(cfg)
    if selected is not None:
        raw_site = getattr(cfg, "norm_site", None)
        raw_sites = getattr(cfg, "norm_sites", None)
        if raw_site == "all" and not raw_sites:
            parts.append("siteall")
        else:
            parts.append("site" + "-".join(sorted(selected)) if selected else "sitenone")
    alpha_value = getattr(cfg, "mean_shift_alpha", None)
    alpha = float(alpha_value or 0.0)
    if alpha_value is not None:
        parts.append(f"shift{getattr(cfg, 'mean_shift_target', 'pre_norm')}{alpha:g}")
    rescue = getattr(cfg, "centering_rescue", "none") or "none"
    if rescue != "none":
        parts.append(f"rescue{rescue}")
    if getattr(cfg, "norm_no_affine", False):
        parts.append("noaffine")
    return "_" + "_".join(parts) if parts else ""


def _norm_cfg(cfg):
    norm_cfg = dict(getattr(cfg, "norm_cfg", {}) or {})
    if getattr(cfg, "norm_no_affine", False):
        norm_cfg["affine"] = False
    return norm_cfg


def _build_named_vit_norm_layer(cfg, norm_name):
    bound_kwargs = dict(_norm_cfg(cfg), dim=3, layout="last")
    names = [name.strip() for name in str(norm_name).split("+") if name.strip()]

    if len(names) == 1 and norm_name in FIXED_SEQUENCE_NORM_NAMES:
        seq_len = _resolve_vit_sequence_length(cfg)

        def norm_layer(_normalized_shape):
            return ext.normalization._make_composite_norm(names, seq_len, **bound_kwargs)

        return norm_layer

    if len(names) == 1 and norm_name in DYNAMIC_SEQUENCE_NORM_NAMES:
        def norm_layer(_normalized_shape):
            return ext.normalization._make_composite_norm(names, **bound_kwargs)

        return norm_layer

    def norm_layer(normalized_shape):
        return ext.normalization._make_composite_norm(names, normalized_shape, **bound_kwargs)

    return norm_layer


def build_vit_norm_layer(cfg):
    return _build_named_vit_norm_layer(cfg, getattr(cfg, "norm", "LN"))


def _build_vit_site_norm_layer(cfg, slot):
    norm_name = _slot_norm_name(cfg, slot)
    base_factory = _build_named_vit_norm_layer(cfg, norm_name)
    rescue_name = _full_norm_name(norm_name)
    rescue_factory = _build_named_vit_norm_layer(cfg, rescue_name)
    alpha = float(getattr(cfg, "mean_shift_alpha", 0.0) or 0.0)
    shift_target = getattr(cfg, "mean_shift_target", "pre_norm")

    def factory(num_features, block_idx=None, site=None):
        use_rescue = _rescue_applies(cfg, slot, block_idx) and rescue_name != norm_name
        module = (rescue_factory if use_rescue else base_factory)(num_features)
        selected = _selected_norm_sites(cfg)
        shift_site_selected = selected is None or slot in selected
        if shift_site_selected and alpha != 0.0 and shift_target in {"pre_norm", "post_norm"}:
            module = MeanShiftNormWrapper(module, alpha=alpha, target=shift_target)
        return module

    return factory


def build_vit_norm_layers(cfg):
    return {
        "norm1_layer": _build_vit_site_norm_layer(cfg, "norm1"),
        "norm2_layer": _build_vit_site_norm_layer(cfg, "norm2"),
        "final_norm_layer": _build_vit_site_norm_layer(cfg, "final"),
    }


def get_vit_model(cfg):
    model_name = getattr(cfg, "arch", None) or "vit_small"
    if model_name not in vits.__dict__:
        raise ValueError(f"Unknown ViT architecture: {model_name}")

    num_classes = getattr(cfg, "dataset_classes", None)
    if num_classes is None:
        num_classes = getattr(cfg, "num_classes", None)

    norm_layers = build_vit_norm_layers(cfg)
    return vits.__dict__[model_name](
        img_size=getattr(cfg, "im_size", [getattr(cfg, "image_size", 224)]),
        patch_size=getattr(cfg, "patch_size", 16),
        in_chans=getattr(cfg, "in_chans", 3),
        num_classes=num_classes,
        drop_rate=getattr(cfg, "dropout", 0.0),
        drop_path_rate=getattr(cfg, "drop_path_rate", 0.0),
        norm_layer=build_vit_norm_layer(cfg),
        act_layer=ext.activation.Activation,
        mean_shift_alpha=getattr(cfg, "mean_shift_alpha", 0.0),
        mean_shift_target=getattr(cfg, "mean_shift_target", "pre_norm"),
        **norm_layers,
    )


__all__ = [
    "VIT_MODEL_NAMES",
    "add_vit_arguments",
    "build_vit_norm_layer",
    "build_vit_norm_layers",
    "get_vit_model",
]


import extension as ext

from ViT.model_vit import vision_transformer as vits


VIT_MODEL_NAMES = ["vit_tiny", "vit_small", "vit_base"]

FIXED_SEQUENCE_NORM_NAMES = {"SeqBN", "SeqBNc", "SeqBNs"}
DYNAMIC_SEQUENCE_NORM_NAMES = {"DSeqBN", "DSeqBNc", "DSeqBNs"}


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


def build_vit_norm_layer(cfg):
    norm_name = getattr(cfg, "norm", None)
    bound_kwargs = dict(dim=3, layout="last")

    if norm_name in FIXED_SEQUENCE_NORM_NAMES:
        seq_len = _resolve_vit_sequence_length(cfg)

        def norm_layer(_normalized_shape):
            return ext.normalization.Norm(seq_len, **bound_kwargs)

        return norm_layer

    if norm_name in DYNAMIC_SEQUENCE_NORM_NAMES:
        def norm_layer(_normalized_shape):
            return ext.normalization.Norm(**bound_kwargs)

        return norm_layer

    return ext.make_norm_factory(**bound_kwargs)


def get_vit_model(cfg):
    model_name = getattr(cfg, "arch", None) or "vit_small"
    if model_name not in vits.__dict__:
        raise ValueError(f"Unknown ViT architecture: {model_name}")

    num_classes = getattr(cfg, "dataset_classes", None)
    if num_classes is None:
        num_classes = getattr(cfg, "num_classes", None)

    norm_layer = build_vit_norm_layer(cfg)
    return vits.__dict__[model_name](
        img_size=getattr(cfg, "im_size", [getattr(cfg, "image_size", 224)]),
        patch_size=getattr(cfg, "patch_size", 16),
        in_chans=getattr(cfg, "in_chans", 3),
        num_classes=num_classes,
        drop_rate=getattr(cfg, "dropout", 0.0),
        drop_path_rate=getattr(cfg, "drop_path_rate", 0.0),
        norm_layer=norm_layer,
        act_layer=ext.activation.Activation,
    )


__all__ = [
    "VIT_MODEL_NAMES",
    "add_vit_arguments",
    "build_vit_norm_layer",
    "get_vit_model",
]

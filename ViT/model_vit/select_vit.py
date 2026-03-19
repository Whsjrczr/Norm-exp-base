import sys

sys.path.append("../")
import extension as ext

from . import vision_transformer as vits


def get_model(cfg):
    model_name = cfg.arch
    if model_name not in vits.__dict__:
        raise ValueError(f"Unknown ViT architecture: {model_name}")

    num_classes = cfg.dataset_classes if getattr(cfg, "dataset_classes", None) is not None else cfg.num_classes
    return vits.__dict__[model_name](
        img_size=getattr(cfg, "im_size", [getattr(cfg, "image_size", 224)]),
        patch_size=getattr(cfg, "patch_size", 16),
        in_chans=getattr(cfg, "in_chans", 3),
        num_classes=num_classes,
        drop_rate=getattr(cfg, "dropout", 0.0),
        drop_path_rate=getattr(cfg, "drop_path_rate", 0.0),
        norm_layer=ext.normalization.Norm,
        act_layer=ext.activation.Activation,
    )

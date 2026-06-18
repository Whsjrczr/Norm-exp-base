import extension as ext


MLP_CLASSIFICATION_MODEL_NAMES = [
    "MLP",
    "ResMLP",
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
    "MultiChannelMLP",
]

MLP_PDE_MODEL_NAMES = [
    "MLP",
    "PreNormMLP",
    "CenDropScalingMLP",
    "CenDropScalingPreNormMLP",
    "ResCenDropScalingMLP",
    "MultiChannelMLP",
]


def add_mlp_arguments(parser, task: str = "classification"):
    default_width = 50 if task == "pde" else 100
    default_depth = 3 if task == "pde" else 4

    group = parser.add_argument_group("MLP Model Options")
    group.add_argument("-width", "--width", type=int, default=default_width)
    group.add_argument("-depth", "--depth", type=int, default=default_depth)
    group.add_argument("-dropout", "--dropout", type=float, default=0.0)
    ext.multichannel.add_arguments(parser)
    return group


def _resolve_input_size(cfg):
    input_size = 1
    for dim in getattr(cfg, "im_size", [1]):
        input_size *= dim
    return input_size


def get_mlp_model(cfg, task: str = "classification"):
    from .MLP import (
        CenDropScalingMLP,
        CenDropScalingPreNormMLP,
        MLP,
        PreNormMLP,
        ResCenDropScalingMLP,
        ResMLP,
    )
    from .resnet import resnet18, resnet34, resnet50
    from .test_bn import ConvBN, ConvBNPre, ConvBNRes, ConvBNResPre
    from .test_ln import ConvLN, ConvLNPre, ConvLNRes, ConvLNResPre

    model_name = getattr(cfg, "arch", None) or "MLP"
    model_width = getattr(cfg, "width", 100)
    model_depth = getattr(cfg, "depth", 4)
    output_size = getattr(cfg, "dataset_classes", 1)
    dropout_prob = getattr(cfg, "dropout", 0.0)
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
    elif model_name == "MultiChannelMLP":
        model_out = ext.MultiChannelMLP(
            width=model_width,
            depth=model_depth,
            input_size=input_size,
            output_size=output_size,
            dropout_prob=dropout_prob,
            num_channels=getattr(cfg, "multi_channels", 1),
            final_layer=getattr(cfg, "multi_final_layer", "merge"),
            activation_factory=ext.Activation,
        )
    else:
        model_out = MLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size, dropout_prob=dropout_prob)
    return ext.multichannel.configure_model(model_out, cfg)


__all__ = [
    "MLP_CLASSIFICATION_MODEL_NAMES",
    "MLP_PDE_MODEL_NAMES",
    "add_mlp_arguments",
    "get_mlp_model",
]

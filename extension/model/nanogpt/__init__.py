import torch.nn as nn

import extension as ext

from .gpt import GPT


NANOGPT_MODEL_NAMES = ["nanoGPT"]

_FEATURE_SAFE_NORMS = {
    "LN",
    "LNc",
    "LNs",
    "RMS",
    "CDS",
    "PLN",
    "PLS",
    "PQN",
}
_CAUSAL_SEQUENCE_NORMS = {
    "CSBN",
    "CSBNc",
    "CSBNs",
    "CDSeqBN",
    "CDSeqBNc",
    "CDSeqBNs",
}


def add_nanogpt_arguments(parser):
    group = parser.add_argument_group("nanoGPT Model Options")
    group.add_argument("--n-layer", dest="n_layer", type=int, default=6)
    group.add_argument("--n-head", dest="n_head", type=int, default=6)
    group.add_argument("--n-embd", dest="n_embd", type=int, default=384)
    group.add_argument("--block-size", dest="block_size", type=int, default=256)
    group.add_argument("--vocab-size", dest="vocab_size", type=int, default=None)
    group.add_argument("--bias", action="store_true", help="use bias terms in Linear and LayerNorm modules")
    return group


def _norm_names(cfg):
    return [name.strip() for name in str(getattr(cfg, "norm", "LN")).split("+") if name.strip()]


def build_nanogpt_norm_layer(cfg):
    names = _norm_names(cfg)
    if names in (["No"], ["no"]):
        return lambda _num_features: nn.Identity()

    unsupported = [
        name for name in names if name not in _FEATURE_SAFE_NORMS and name not in _CAUSAL_SEQUENCE_NORMS
    ]
    if unsupported:
        raise ValueError(
            "nanoGPT only supports feature-axis norms or causal sequence norms. "
            f"Unsupported for causal language modeling: {', '.join(unsupported)}."
        )

    bound_kwargs = dict(dim=3, layout="last")
    if any(name.startswith("CDSeqBN") for name in names):
        if len(names) != 1:
            raise ValueError("CDSeqBN variants cannot be composed with feature-axis normalization.")

        def norm_layer(_num_features):
            return ext.normalization.Norm(**bound_kwargs)

        return norm_layer

    return ext.make_norm_factory(**bound_kwargs)


def get_nanogpt_model(cfg):
    if getattr(cfg, "arch", "nanoGPT") not in NANOGPT_MODEL_NAMES:
        raise ValueError(f"Unknown nanoGPT architecture: {cfg.arch}")
    if getattr(cfg, "vocab_size", None) is None:
        raise ValueError("nanoGPT requires cfg.vocab_size from the prepared dataset metadata.")
    if cfg.n_embd % cfg.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head.")

    return GPT(
        vocab_size=int(cfg.vocab_size),
        block_size=int(cfg.block_size),
        n_layer=int(cfg.n_layer),
        n_head=int(cfg.n_head),
        n_embd=int(cfg.n_embd),
        dropout=float(cfg.dropout),
        bias=bool(cfg.bias),
        norm_layer=build_nanogpt_norm_layer(cfg),
        act_layer=ext.activation.Activation,
    )


__all__ = [
    "GPT",
    "NANOGPT_MODEL_NAMES",
    "add_nanogpt_arguments",
    "build_nanogpt_norm_layer",
    "get_nanogpt_model",
]

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
    "CSeqBN",
    "CSeqBNc",
    "CSeqBNs",
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
    group.add_argument("--attn-norm", dest="attn_norm", default=None, help="normalization before causal self-attention")
    group.add_argument("--attn-norm-cfg", dest="attn_norm_cfg", type=ext.utils.str2dict, default=None, metavar="DICT")
    group.add_argument("--mlp-norm", dest="mlp_norm", default=None, help="normalization before the MLP block")
    group.add_argument("--mlp-norm-cfg", dest="mlp_norm_cfg", type=ext.utils.str2dict, default=None, metavar="DICT")
    group.add_argument("--final-norm", dest="final_norm", default=None, help="normalization before the language-model head")
    group.add_argument("--final-norm-cfg", dest="final_norm_cfg", type=ext.utils.str2dict, default=None, metavar="DICT")
    group.add_argument("--mlp-activation", dest="mlp_activation", default=None, help="activation inside the Transformer MLP")
    group.add_argument("--mlp-activation-cfg", dest="mlp_activation_cfg", type=ext.utils.str2dict, default=None, metavar="DICT")
    group.add_argument(
        "--allow-noncausal-norm",
        action="store_true",
        help="allow BN/SeqBN-style controls that mix future tokens; use only for explicit baselines",
    )
    return group


def _norm_names(norm):
    return [name.strip() for name in str(norm).split("+") if name.strip()]


def _merged_slot_cfg(cfg, slot):
    base = dict(getattr(cfg, "norm_cfg", {}) or {})
    override = getattr(cfg, f"{slot}_norm_cfg", None)
    if override:
        base.update(override)
    return base


def _slot_norm_name(cfg, slot):
    return getattr(cfg, f"{slot}_norm", None) or getattr(cfg, "norm", "LN")


def _validate_norm_names(names, allow_noncausal=False):
    if names in (["No"], ["no"]):
        return
    unsupported = [
        name for name in names if name not in _FEATURE_SAFE_NORMS and name not in _CAUSAL_SEQUENCE_NORMS
    ]
    if unsupported:
        if allow_noncausal and all(name in ext.normalization._config.norm_methods for name in unsupported):
            return
        raise ValueError(
            "nanoGPT only supports feature-axis norms or causal sequence norms. "
            f"Unsupported for causal language modeling: {', '.join(unsupported)}."
        )
    if any(name.startswith("CDSeqBN") for name in names) and len(names) != 1:
        raise ValueError("CDSeqBN variants cannot be composed with feature-axis normalization.")


def _build_norm_layer(norm_name, norm_cfg, block_size=None):
    names = _norm_names(norm_name)
    if names in (["No"], ["no"]):
        return lambda _num_features: nn.Identity()

    bound_kwargs = dict(norm_cfg or {})
    bound_kwargs.setdefault("dim", 3)
    bound_kwargs.setdefault("layout", "last")
    allow_noncausal = bool(bound_kwargs.pop("allow_noncausal_norm", False))
    _validate_norm_names(names, allow_noncausal=allow_noncausal)

    if len(names) == 1 and (
        (names[0].startswith("SeqBN") and not names[0].startswith("DSeqBN"))
        or (names[0].startswith("CSeqBN") and not names[0].startswith("CDSeqBN"))
    ):
        if block_size is None:
            raise ValueError("Fixed sequence norm variants require block_size for sequence length binding.")

        def norm_layer(_num_features):
            return ext.normalization._make_composite_norm(names, int(block_size), **bound_kwargs)

        return norm_layer

    if len(names) == 1 and (names[0].startswith("DSeqBN") or names[0].startswith("CDSeqBN")):
        def norm_layer(_num_features):
            return ext.normalization._make_composite_norm(names, **bound_kwargs)

        return norm_layer

    def norm_layer(num_features):
        return ext.normalization._make_composite_norm(names, num_features, **bound_kwargs)

    return norm_layer


def format_nanogpt_norm_setting(cfg):
    global_norm = getattr(cfg, "norm", "LN")
    global_cfg = dict(getattr(cfg, "norm_cfg", {}) or {})
    slots = ("attn", "mlp", "final")
    uses_global = all(
        getattr(cfg, f"{slot}_norm", None) is None and getattr(cfg, f"{slot}_norm_cfg", None) in (None, {})
        for slot in slots
    )
    if uses_global:
        return ext.normalization.setting(cfg)

    def slot_flag(slot):
        norm_name = _slot_norm_name(cfg, slot)
        norm_cfg = _merged_slot_cfg(cfg, slot)
        extra = "".join(f"_{key}{norm_cfg[key]}" for key in sorted(norm_cfg) if norm_cfg[key] != global_cfg.get(key))
        return f"{slot}{norm_name}{extra}"

    return "_".join(slot_flag(slot) for slot in slots)


def format_nanogpt_activation_setting(cfg):
    if getattr(cfg, "mlp_activation", None) is None and getattr(cfg, "mlp_activation_cfg", None) in (None, {}):
        return ext.activation.setting(cfg)
    act_name = getattr(cfg, "mlp_activation", None) or getattr(cfg, "activation", "gelu")
    act_cfg = dict(getattr(cfg, "activation_cfg", {}) or {})
    if getattr(cfg, "mlp_activation_cfg", None):
        act_cfg.update(cfg.mlp_activation_cfg)
    extra = "".join(f"_{key}{act_cfg[key]}" for key in sorted(act_cfg))
    return f"mlp{act_name}{extra}"


def build_nanogpt_norm_layer(cfg):
    norm_cfg = dict(getattr(cfg, "norm_cfg", {}) or {})
    norm_cfg["allow_noncausal_norm"] = getattr(cfg, "allow_noncausal_norm", False)
    return _build_norm_layer(
        getattr(cfg, "norm", "LN"),
        norm_cfg,
        block_size=getattr(cfg, "block_size", None),
    )


def build_nanogpt_norm_layers(cfg):
    return {
        "attn_norm_layer": _build_norm_layer(
            _slot_norm_name(cfg, "attn"),
            {**_merged_slot_cfg(cfg, "attn"), "allow_noncausal_norm": getattr(cfg, "allow_noncausal_norm", False)},
            block_size=getattr(cfg, "block_size", None),
        ),
        "mlp_norm_layer": _build_norm_layer(
            _slot_norm_name(cfg, "mlp"),
            {**_merged_slot_cfg(cfg, "mlp"), "allow_noncausal_norm": getattr(cfg, "allow_noncausal_norm", False)},
            block_size=getattr(cfg, "block_size", None),
        ),
        "final_norm_layer": _build_norm_layer(
            _slot_norm_name(cfg, "final"),
            {**_merged_slot_cfg(cfg, "final"), "allow_noncausal_norm": getattr(cfg, "allow_noncausal_norm", False)},
            block_size=getattr(cfg, "block_size", None),
        ),
    }


def build_nanogpt_activation_layer(cfg):
    act_name = getattr(cfg, "mlp_activation", None) or getattr(cfg, "activation", "gelu")
    act_cfg = dict(getattr(cfg, "activation_cfg", {}) or {})
    if getattr(cfg, "mlp_activation_cfg", None):
        act_cfg.update(cfg.mlp_activation_cfg)
    methods = ext.activation._config._methods
    if act_name not in methods:
        raise ValueError(f"Unknown nanoGPT MLP activation: {act_name}. Expected one of {sorted(methods)}.")

    def act_layer(num_features):
        return methods[act_name](num_features, **act_cfg)

    return act_layer


def get_nanogpt_model(cfg):
    if getattr(cfg, "arch", "nanoGPT") not in NANOGPT_MODEL_NAMES:
        raise ValueError(f"Unknown nanoGPT architecture: {cfg.arch}")
    if getattr(cfg, "vocab_size", None) is None:
        raise ValueError("nanoGPT requires cfg.vocab_size from the prepared dataset metadata.")
    if cfg.n_embd % cfg.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head.")

    norm_layers = build_nanogpt_norm_layers(cfg)
    return GPT(
        vocab_size=int(cfg.vocab_size),
        block_size=int(cfg.block_size),
        n_layer=int(cfg.n_layer),
        n_head=int(cfg.n_head),
        n_embd=int(cfg.n_embd),
        dropout=float(cfg.dropout),
        bias=bool(cfg.bias),
        norm_layer=build_nanogpt_norm_layer(cfg),
        act_layer=build_nanogpt_activation_layer(cfg),
        **norm_layers,
    )


__all__ = [
    "GPT",
    "NANOGPT_MODEL_NAMES",
    "add_nanogpt_arguments",
    "build_nanogpt_activation_layer",
    "build_nanogpt_norm_layer",
    "build_nanogpt_norm_layers",
    "format_nanogpt_activation_setting",
    "format_nanogpt_norm_setting",
    "get_nanogpt_model",
]

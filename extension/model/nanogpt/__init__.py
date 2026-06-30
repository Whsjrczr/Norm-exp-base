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
    "EMASBN",
    "EMASBNc",
    "EMASBNs",
    "CCFBN",
    "CCFBNc",
    "CCFBNs",
    "EMACFBN",
    "EMACFBNc",
    "EMACFBNs",
}


class MeanShiftNormWrapper(nn.Module):
    def __init__(self, module, alpha=0.0, target="pre_norm"):
        super().__init__()
        self.module = module
        self.alpha = float(alpha or 0.0)
        self.target = target

    def forward(self, x):
        if self.target == "pre_norm" and self.alpha != 0.0:
            x = x + self.alpha
        y = self.module(x)
        if self.target == "post_norm" and self.alpha != 0.0:
            y = y + self.alpha
        return y


def _split_csv(value):
    if value is None or value == "":
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _selected_norm_sites(cfg, valid_sites):
    site = getattr(cfg, "norm_site", None)
    sites = _split_csv(getattr(cfg, "norm_sites", None))
    if site in (None, "none") and not sites:
        return None
    if site == "all":
        selected = set(valid_sites)
    else:
        selected = {site} if site not in (None, "none") else set()
    selected.update(sites)
    return {item for item in selected if item in valid_sites}


def _full_norm_name(norm_name):
    names = _norm_names(norm_name)
    out = []
    for name in names:
        candidate = name[:-1] if name.endswith("s") else name
        out.append(candidate if candidate in ext.normalization._config.norm_methods else name)
    return "+".join(out)


def _range_rescue_applies(cfg, block_idx, rescue):
    if block_idx is None or rescue not in {"early", "middle", "late"}:
        return False
    n_layer = max(1, int(getattr(cfg, "n_layer", 1)))
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
    if rescue in {"attn", "mlp", "final"}:
        return slot == rescue
    return _range_rescue_applies(cfg, block_idx, rescue)


def _intervention_suffix(cfg):
    parts = []
    selected = _selected_norm_sites(cfg, ("attn", "mlp", "final"))
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
    if getattr(cfg, "norm_no_affine", False):
        base["affine"] = False
    override = getattr(cfg, f"{slot}_norm_cfg", None)
    if override:
        base.update(override)
    return base


def _slot_norm_name(cfg, slot):
    selected = _selected_norm_sites(cfg, ("attn", "mlp", "final"))
    explicit = getattr(cfg, f"{slot}_norm", None)
    if selected is not None:
        return explicit or (getattr(cfg, "norm", "LN") if slot in selected else "LN")
    return explicit or getattr(cfg, "norm", "LN")


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
        return ext.normalization.setting(cfg) + _intervention_suffix(cfg)

    def slot_flag(slot):
        norm_name = _slot_norm_name(cfg, slot)
        norm_cfg = _merged_slot_cfg(cfg, slot)
        extra = "".join(f"_{key}{norm_cfg[key]}" for key in sorted(norm_cfg) if norm_cfg[key] != global_cfg.get(key))
        return f"{slot}{norm_name}{extra}"

    return "_".join(slot_flag(slot) for slot in slots) + _intervention_suffix(cfg)


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
    if getattr(cfg, "norm_no_affine", False):
        norm_cfg["affine"] = False
    norm_cfg["allow_noncausal_norm"] = getattr(cfg, "allow_noncausal_norm", False)
    return _build_norm_layer(
        getattr(cfg, "norm", "LN"),
        norm_cfg,
        block_size=getattr(cfg, "block_size", None),
    )


def _build_nanogpt_site_norm_layer(cfg, slot):
    norm_name = _slot_norm_name(cfg, slot)
    norm_cfg = {**_merged_slot_cfg(cfg, slot), "allow_noncausal_norm": getattr(cfg, "allow_noncausal_norm", False)}
    base_factory = _build_norm_layer(norm_name, norm_cfg, block_size=getattr(cfg, "block_size", None))
    rescue_name = _full_norm_name(norm_name)
    rescue_factory = _build_norm_layer(rescue_name, norm_cfg, block_size=getattr(cfg, "block_size", None))
    alpha = float(getattr(cfg, "mean_shift_alpha", 0.0) or 0.0)
    shift_target = getattr(cfg, "mean_shift_target", "pre_norm")

    def factory(num_features, block_idx=None, site=None):
        use_rescue = _rescue_applies(cfg, slot, block_idx) and rescue_name != norm_name
        module = (rescue_factory if use_rescue else base_factory)(num_features)
        selected = _selected_norm_sites(cfg, ("attn", "mlp", "final"))
        shift_site_selected = selected is None or slot in selected
        if shift_site_selected and alpha != 0.0 and shift_target in {"pre_norm", "post_norm"}:
            module = MeanShiftNormWrapper(module, alpha=alpha, target=shift_target)
        return module

    return factory


def build_nanogpt_norm_layers(cfg):
    return {
        "attn_norm_layer": _build_nanogpt_site_norm_layer(cfg, "attn"),
        "mlp_norm_layer": _build_nanogpt_site_norm_layer(cfg, "mlp"),
        "final_norm_layer": _build_nanogpt_site_norm_layer(cfg, "final"),
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
        mean_shift_alpha=getattr(cfg, "mean_shift_alpha", 0.0),
        mean_shift_target=getattr(cfg, "mean_shift_target", "pre_norm"),
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











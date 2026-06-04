import torch.nn as nn

import extension as ext

from .model import BertTranslationModel


BERT_MODEL_NAMES = ["BERTTranslation", "bert_translation"]


def add_bert_arguments(parser):
    group = parser.add_argument_group("BERT Translation Model Options")
    group.add_argument("--bert-layers", dest="bert_layers", type=int, default=4)
    group.add_argument("--bert-heads", dest="bert_heads", type=int, default=4)
    group.add_argument("--bert-embd", dest="bert_embd", type=int, default=256)
    group.add_argument("--bert-ffn-mult", dest="bert_ffn_mult", type=int, default=4)
    group.add_argument("--max-src-len", dest="max_src_len", type=int, default=64)
    group.add_argument("--max-tgt-len", dest="max_tgt_len", type=int, default=64)
    group.add_argument("--pad-token-id", dest="pad_token_id", type=int, default=0)
    group.add_argument("--bos-token-id", dest="bos_token_id", type=int, default=1)
    group.add_argument("--eos-token-id", dest="eos_token_id", type=int, default=2)
    group.add_argument("--unk-token-id", dest="unk_token_id", type=int, default=3)
    return group


def _norm_names(norm):
    return [name.strip() for name in str(norm).split("+") if name.strip()]


def build_bert_norm_layer(cfg):
    norm_name = getattr(cfg, "norm", "LN")
    names = _norm_names(norm_name)
    if names in (["No"], ["no"]):
        return lambda _num_features: nn.Identity()

    norm_cfg = dict(getattr(cfg, "norm_cfg", {}) or {})
    norm_cfg.setdefault("dim", 3)
    norm_cfg.setdefault("layout", "last")

    sequence_len = max(int(getattr(cfg, "max_src_len", 0)), int(getattr(cfg, "max_tgt_len", 0)))
    fixed_sequence_norm = (
        len(names) == 1
        and names[0].startswith("SeqBN")
        and not names[0].startswith("DSeqBN")
    )

    def norm_layer(num_features):
        if fixed_sequence_norm:
            if sequence_len <= 0:
                raise ValueError("SeqBN requires max_src_len/max_tgt_len for sequence length binding.")
            return ext.normalization._make_composite_norm(names, sequence_len, **norm_cfg)
        return ext.normalization._make_composite_norm(names, num_features, **norm_cfg)

    return norm_layer


def build_bert_activation_layer(cfg):
    act_name = getattr(cfg, "activation", "gelu")
    act_cfg = dict(getattr(cfg, "activation_cfg", {}) or {})
    methods = ext.activation._config._methods
    if act_name not in methods:
        raise ValueError(f"Unknown BERT activation: {act_name}. Expected one of {sorted(methods)}.")

    def act_layer(num_features):
        return methods[act_name](num_features, **act_cfg)

    return act_layer


def get_bert_model(cfg):
    if getattr(cfg, "arch", "BERTTranslation") not in BERT_MODEL_NAMES:
        raise ValueError(f"Unknown BERT architecture: {cfg.arch}")
    if getattr(cfg, "vocab_size", None) is None:
        raise ValueError("BERTTranslation requires cfg.vocab_size from translation dataset metadata.")
    if cfg.bert_embd % cfg.bert_heads != 0:
        raise ValueError("bert_embd must be divisible by bert_heads.")
    return BertTranslationModel(
        vocab_size=int(cfg.vocab_size),
        max_src_len=int(cfg.max_src_len),
        max_tgt_len=int(cfg.max_tgt_len),
        pad_token_id=int(cfg.pad_token_id),
        bos_token_id=int(cfg.bos_token_id),
        eos_token_id=int(cfg.eos_token_id),
        n_layer=int(cfg.bert_layers),
        n_head=int(cfg.bert_heads),
        n_embd=int(cfg.bert_embd),
        ffn_mult=int(cfg.bert_ffn_mult),
        dropout=float(cfg.dropout),
        norm_layer=build_bert_norm_layer(cfg),
        act_layer=build_bert_activation_layer(cfg),
    )


__all__ = [
    "BERT_MODEL_NAMES",
    "BertTranslationModel",
    "add_bert_arguments",
    "build_bert_activation_layer",
    "build_bert_norm_layer",
    "get_bert_model",
]

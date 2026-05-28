from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import extension as ext
from nanoGPT.tinyshakespeare import TinyShakespeareBatches, prepare_tinyshakespeare


def _cfg(norm="LN"):
    return SimpleNamespace(
        arch="nanoGPT",
        vocab_size=19,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
        norm=norm,
        norm_cfg={},
        attn_norm=None,
        attn_norm_cfg=None,
        mlp_norm=None,
        mlp_norm_cfg=None,
        final_norm=None,
        final_norm_cfg=None,
        activation="gelu",
        activation_cfg={},
        mlp_activation=None,
        mlp_activation_cfg=None,
        init_preset="default",
    )


@pytest.mark.parametrize("norm", ["LN", "RMS", "CSBN", "CSeqBN", "CDSeqBN"])
def test_nanogpt_forward_backward_and_generate(norm):
    cfg = _cfg(norm)
    ext.normalization.setting(cfg)
    ext.activation.setting(cfg)
    model = ext.model.get_model(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))

    logits, loss = model(tokens, tokens)

    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert torch.isfinite(loss)
    loss.backward()
    prompt = tokens if norm.startswith("CSeqBN") and not norm.startswith("CDSeqBN") else tokens[:, :2]
    generated = model.generate(prompt, max_new_tokens=3, top_k=5)
    assert generated.shape[1] == prompt.shape[1] + 3


@pytest.mark.parametrize("norm", ["BN", "SeqBN", "DSeqBN"])
def test_nanogpt_rejects_noncausal_token_mixing_norms(norm):
    cfg = _cfg(norm)
    ext.normalization.setting(cfg)
    ext.activation.setting(cfg)
    with pytest.raises(ValueError, match="causal language modeling"):
        ext.model.get_model(cfg)


def test_nanogpt_supports_independent_norm_and_activation_slots():
    cfg = _cfg("LN")
    cfg.attn_norm = "RMS"
    cfg.mlp_norm = "PQN"
    cfg.mlp_norm_cfg = {"num_per_group": 4, "p": 2, "q": 2}
    cfg.final_norm = "No"
    cfg.mlp_activation = "pqact"
    cfg.mlp_activation_cfg = {"p": 2, "q": 2}
    ext.normalization.setting(cfg)
    ext.activation.setting(cfg)
    model = ext.model.get_model(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))

    logits, loss = model(tokens, tokens)

    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert torch.isfinite(loss)
    assert isinstance(model.transformer.ln_f, torch.nn.Identity)


def test_tinyshakespeare_prepare_and_batch(tmp_path):
    text = "To be, or not to be.\n" * 30
    (tmp_path / "input.txt").write_text(text, encoding="utf-8")

    metadata = prepare_tinyshakespeare(tmp_path, train_ratio=0.8, download=False)
    dataset = TinyShakespeareBatches(tmp_path, block_size=8, device=torch.device("cpu"))
    inputs, targets = dataset.get_batch("train", batch_size=3)

    assert metadata["vocab_size"] == len(set(text))
    assert inputs.shape == (3, 8)
    assert targets.shape == (3, 8)
    assert dataset.decode(dataset.encode("To be")) == "To be"

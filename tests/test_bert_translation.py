from types import SimpleNamespace
import zipfile

import pytest

torch = pytest.importorskip("torch")

import extension as ext
from BERT.download_opus_books import convert_opus_zip
from BERT.translation_data import TranslationBatches, prepare_translation_data


def _cfg(vocab_size=32):
    return SimpleNamespace(
        model_family="bert",
        arch="BERTTranslation",
        vocab_size=vocab_size,
        bert_layers=2,
        bert_heads=2,
        bert_embd=32,
        bert_ffn_mult=2,
        max_src_len=8,
        max_tgt_len=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=3,
        dropout=0.0,
        norm="LN",
        norm_cfg={},
        activation="gelu",
        activation_cfg={},
        init_preset="default",
    )


def test_translation_prepare_and_batch(tmp_path):
    input_file = tmp_path / "pairs.tsv"
    input_file.write_text(
        "\n".join(
            [
                "hello\tbonjour",
                "good night\tbonne nuit",
                "thank you\tmerci",
                "yes\toui",
            ]
        ),
        encoding="utf-8",
    )

    meta = prepare_translation_data(tmp_path / "data", input_file=input_file, train_ratio=0.75)
    dataset = TranslationBatches(tmp_path / "data", max_src_len=8, max_tgt_len=8, device=torch.device("cpu"))
    src, tgt_input, tgt_labels = dataset.get_batch("train", batch_size=2)

    assert meta["vocab_size"] >= 8
    assert src.shape == (2, 8)
    assert tgt_input.shape == (2, 8)
    assert tgt_labels.shape == (2, 8)
    assert dataset.vocab.decode(dataset.vocab.encode("hello", max_len=8)) == "hello"


def test_convert_opus_zip_to_pairs(tmp_path):
    zip_path = tmp_path / "en-fr.txt.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("Books.en-fr.en", "hello\nvery long source sentence should be filtered\n")
        archive.writestr("Books.en-fr.fr", "bonjour\nphrase cible\n")

    output_tsv = tmp_path / "pairs.tsv"
    count, src_member, tgt_member = convert_opus_zip(
        zip_path,
        output_tsv,
        source_lang="en",
        target_lang="fr",
        pair_name="en-fr",
        max_pairs=10,
        max_src_tokens=2,
        max_tgt_tokens=4,
    )

    assert count == 1
    assert src_member == "Books.en-fr.en"
    assert tgt_member == "Books.en-fr.fr"
    assert output_tsv.read_text(encoding="utf-8").strip() == "hello\tbonjour"


@pytest.mark.parametrize("norm", ["LN", "BN", "SBN", "SeqBN", "CFBN"])
def test_bert_translation_forward_backward_and_generate(norm):
    cfg = _cfg()
    cfg.norm = norm
    ext.normalization.setting(cfg)
    ext.activation.setting(cfg)
    model = ext.model.get_model(cfg)
    src = torch.randint(4, cfg.vocab_size, (2, cfg.max_src_len))
    tgt_input = torch.randint(4, cfg.vocab_size, (2, cfg.max_tgt_len))
    tgt_input[:, 0] = cfg.bos_token_id
    tgt_labels = torch.randint(4, cfg.vocab_size, (2, cfg.max_tgt_len))
    tgt_labels[:, -1] = cfg.eos_token_id

    logits, loss = model(src, tgt_input, tgt_labels)

    assert logits.shape == (2, cfg.max_tgt_len, cfg.vocab_size)
    assert torch.isfinite(loss)
    loss.backward()
    generated = model.generate(src[:, :4], max_new_tokens=3)
    assert generated.shape[0] == 2
    assert generated.shape[1] <= 4

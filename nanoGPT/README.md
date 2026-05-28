# nanoGPT + TinyShakespeare

This task trains a character-level decoder-only Transformer from scratch on
TinyShakespeare. It follows the existing project task layout and reuses:

- `extension.model.get_model(cfg)` for model construction.
- `extension.normalization` and `extension.activation` for experimental layers.
- `extension.optimizer`, `extension.scheduler`, `extension.checkpoint`,
  `extension.tracking`, and `extension.measurement` for training utilities.

## Files

- `nanogpt.py`: training, validation, checkpoint, and generation entry point.
- `tinyshakespeare.py`: data preparation and random token batch provider.
- `prepare_tinyshakespeare.py`: explicit dataset preparation command.
- `run_tinyshakespeare_4090.ps1`: recommended RTX 4090 training command.
- `run_nanogpt_sequence_bn_batch.sh`: SeqBN-family traversal script.
- `extension/model/nanogpt/`: causal Transformer implementation and model factory.

## Dataset

Prepare the 1 MB TinyShakespeare corpus and its fixed 90/10 split:

```powershell
python nanoGPT/prepare_tinyshakespeare.py --data-dir ./dataset/tinyshakespeare
```

This produces:

```text
dataset/tinyshakespeare/
  input.txt
  train.bin
  val.bin
  meta.json
```

`nanogpt.py` also prepares these files automatically when they are absent,
unless `--no-auto-prepare` is specified.

## 4090 Training

The default architecture matches the small nanoGPT Shakespeare setup:

| Option | Value |
| --- | ---: |
| Parameters | about 10.7 M |
| Layers | 6 |
| Heads | 6 |
| Embedding width | 384 |
| Context length | 256 |
| Precision | `bfloat16` |
| Optimizer | AdamW |

Run the recommended single-GPU configuration:

```powershell
python nanoGPT/nanogpt.py `
  --data-dir ./dataset/tinyshakespeare `
  --batch-size 128,128 `
  --iters-per-epoch 100 `
  --eval-iters 50 `
  --epochs 50 `
  --optimizer adamw `
  --lr 6e-4 `
  --lr-method cos `
  --weight-decay 0.1 `
  --norm LN `
  --activation gelu `
  --dtype bfloat16 `
  --compile
```

For a quick pipeline check:

```powershell
python nanoGPT/nanogpt.py `
  --n-layer 2 --n-head 2 --n-embd 64 --block-size 32 `
  --batch-size 4,4 --iters-per-epoch 2 --eval-iters 2 --epochs 1 `
  --dtype float32 --no-save-checkpoint --sample-tokens 20
```

## Outputs

Training outputs follow the repository result convention:

```text
results/
  nanoGPT_tinyshakespeare_.../
    log.txt
    checkpoint.pth
    best.pth
    sample_epoch_*.txt
    sample_final.txt
```

Evaluate and generate from the best checkpoint:

```powershell
python nanoGPT/nanogpt.py `
  --test `
  --load ./results/<run-name>/best.pth `
  --data-dir ./dataset/tinyshakespeare `
  --sample-prompt "ROMEO:" `
  --sample-tokens 500
```

## Normalization Experiments

`LN` is the default. Feature-axis choices such as `LNc`, `LNs`, `RMS`,
`PLN`, `PLS`, and `PQN` are safe for autoregressive generation. Causal
sequence options `CSBN`, `CSBNc`, `CSBNs`, `CDSeqBN`, `CDSeqBNc`, and
`CDSeqBNs` are also supported.

Non-causal batch or sequence normalizations are rejected in this task because
they expose future tokens during language-model training.

The coarse setting applies one norm/activation choice to all Transformer blocks:

```powershell
python nanoGPT/nanogpt.py `
  --norm PQN `
  --norm-cfg "num_per_group=8,p=4,q=2" `
  --activation pqact `
  --activation-cfg "p=4,q=2"
```

You can also override each GPT slot independently:

```powershell
python nanoGPT/nanogpt.py `
  --norm LN `
  --attn-norm RMS `
  --mlp-norm PQN `
  --mlp-norm-cfg "num_per_group=8,p=4,q=2" `
  --final-norm No `
  --mlp-activation mlpact `
  --mlp-activation-cfg "hidden_dim=16,act=gelu"
```

Slot-specific options inherit `--norm-cfg` first, then apply their own
`--attn-norm-cfg`, `--mlp-norm-cfg`, or `--final-norm-cfg` overrides.

Batch traversal for causal SeqBN variants:

```bash
bash nanoGPT/run_nanogpt_sequence_bn_batch.sh
```

The script mirrors `ViT/run_vit_sequence_bn_batch.sh`, but only traverses
autoregressive-safe variants:

```text
CSBN CSBNc CSBNs
CSeqBN CSeqBNc CSeqBNs
CDSeqBN CDSeqBNc CDSeqBNs
```

It runs each norm on `attn`, `mlp`, `final`, and `all` slots. Override common
settings through environment variables, for example:

```bash
PYTHON_BIN=./.conda/python.exe EPOCHS=5 BATCH_SIZE=32,32 bash nanoGPT/run_nanogpt_sequence_bn_batch.sh
```

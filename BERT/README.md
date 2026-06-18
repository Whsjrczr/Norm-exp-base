# BERT-style Text Translation

This task trains a small Transformer encoder-decoder for text translation. The
encoder reads the source sentence bidirectionally in a BERT-style stack, and the
decoder generates the target sentence with a causal mask.

`nanoGPT` remains a decoder-only language-model task. It can be adapted to
translation by formatting data as `source <sep> target`, but the standard
translation setup needs an encoder-decoder model, so this folder adds a BERT
translation task instead of changing the TinyShakespeare entry point.

## Files

- `bert_translation.py`: training, validation, checkpoint, and sample translation entry point.
- `translation_data.py`: TSV preparation, vocabulary, split files, and random batches.
- `prepare_translation.py`: explicit dataset preparation command.
- `download_opus_books.py`: download OPUS Books English-French Moses files and convert them to TSV.
- `run_bert_translation_norm_lr_batch.sh`: generate OPUS Books norm/LR sweep jobs.
- `summarize_translation_results.py`: parse result logs into a CSV summary.
- `EXPERIMENTS.md`: experiment design, sweep variables, and analysis plan.
- `extension/model/bert/`: BERT-style translation model and model factory.

## Dataset

Use a TSV file with one pair per line:

```text
source sentence<TAB>target sentence
```

Prepare train/val files and vocabulary:

```powershell
python BERT/prepare_translation.py `
  --data-dir ./dataset/text_translation `
  --input-file ./dataset/my_translation_pairs.tsv `
  --train-ratio 0.9
```

If no `--input-file` is given and `pairs.tsv` is absent, a tiny English-French
demo dataset is created so the pipeline can be tested.

Recommended experiment dataset: OPUS Books `en-fr`. It is a public translation
dataset of aligned copyright-free books. The Hugging Face card marks it as a
Translation/Text dataset, and the `en-fr` subset has about 127k rows.

Download and convert the OPUS Moses zip into this task's TSV format:

```powershell
python BERT/download_opus_books.py `
  --data-dir ./dataset/opus_books_en_fr `
  --max-pairs 50000 `
  --max-src-tokens 64 `
  --max-tgt-tokens 64
```

Then prepare train/val splits:

```powershell
python BERT/prepare_translation.py `
  --data-dir ./dataset/opus_books_en_fr `
  --input-file ./dataset/opus_books_en_fr/pairs.tsv `
  --min-freq 2 `
  --overwrite
```

## Quick Check

```powershell
powershell -ExecutionPolicy Bypass -File BERT/run_bert_translation_quick.ps1
```

## Training

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File BERT/run_bert_translation_train.ps1 `
  -InputFile ./dataset/my_translation_pairs.tsv `
  -DataDir ./dataset/text_translation `
  -SampleSrc "hello"
```

Bash:

```bash
INPUT_FILE=./dataset/my_translation_pairs.tsv \
DATA_DIR=./dataset/text_translation \
SAMPLE_SRC="hello" \
bash BERT/run_bert_translation_train.sh
```

Equivalent raw command:

```powershell
python BERT/bert_translation.py `
  --data-dir ./dataset/text_translation `
  --input-file ./dataset/my_translation_pairs.tsv `
  --bert-layers 4 --bert-heads 4 --bert-embd 256 `
  --max-src-len 64 --max-tgt-len 64 `
  --batch-size 64,64 --iters-per-epoch 100 --eval-iters 50 --epochs 50 `
  --optimizer adamw --lr 3e-4 --lr-method cos --weight-decay 0.01 `
  --norm LN --activation gelu --dtype bfloat16
```

## Norm/LR Sweep

Generate jobs for `LN`, `BN/BNc/BNs`, `SBN/SBNc/SBNs`,
`SeqBN/SeqBNc/SeqBNs`, and `CFBN/CFBNc/CFBNs` across multiple learning rates
and seeds:

```bash
DATA_DIR=./dataset/opus_books_en_fr \
OUTPUT_ROOT=./results/exp-bert-opus-books-norm-lr \
MAX_PAIRS=50000 \
MIN_FREQ=2 \
NUM_ONCE=1 \
bash BERT/run_bert_translation_norm_lr_batch.sh
```

Then run the generated launcher:

```bash
cd BERT/exp-bert-opus-books-norm-lr
bash z_bash_execute.sh
```

Summarize finished runs:

```bash
python BERT/summarize_translation_results.py \
  --results-root ./results/exp-bert-opus-books-norm-lr \
  --output ./results/exp-bert-opus-books-norm-lr/summary.csv
```

Outputs follow the repository result convention:

```text
results/
  BERT_translation_.../
    log.txt
    checkpoint.pth
    best.pth
    sample_epoch_*.txt
    sample_final.txt
```

Evaluate and sample from a checkpoint:

```powershell
python BERT/bert_translation.py `
  --test `
  --load ./results/<run-name>/best.pth `
  --data-dir ./dataset/text_translation `
  --sample-src "hello"
```

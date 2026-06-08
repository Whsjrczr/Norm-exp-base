# BERT Translation Norm Experiments

## Goal

Measure how `SBN`, `SeqBN`, `LN`, `BN`, and `CFBN` affect BERT-style
encoder-decoder translation under different learning rates.

## Dataset

Use OPUS Books English-French (`en-fr`) as the default translation dataset.
The batch script downloads the OPUS Moses zip, converts aligned text files into
`source<TAB>target`, then builds the train/validation split and vocabulary.

Default preprocessing:

- `MAX_PAIRS=50000`
- `MAX_SRC_LEN=64`
- `MAX_TGT_LEN=64`
- `MIN_FREQ=2`
- train/val split: `90/10`

## Controlled Setup

Default model/training settings:

- model: `BERTTranslation`
- layers: `4`
- heads: `4`
- embedding width: `256`
- FFN multiplier: `4`
- optimizer: `adamw`
- weight decay: `0.01`
- dropout: `0.1`
- activation: `gelu`
- dtype: `bfloat16`
- batch size: `64,64`
- epochs: `50`
- iters per epoch: `100`
- eval iters: `50`

## Sweep Variables

Norms:

```text
LN
BN BNc BNs
SBN SBNc SBNs
SeqBN SeqBNc SeqBNs
CFBN CFBNc CFBNs
```

Learning rates:

```text
1e-4 3e-4 6e-4 1e-3
```

Seeds:

```text
0 1 2
```

This produces `13 * 4 * 3 = 156` jobs with the default single optimizer.

## Metrics

Primary:

- validation loss
- validation perplexity
- validation token accuracy

Secondary:

- train loss
- generated sample text
- training stability and failed runs

Use validation loss as the main ranking metric. Use token accuracy and sample
quality as sanity checks, not as the only basis for selecting a norm.

## Run

Generate jobs:

```bash
DATA_DIR=./dataset/opus_books_en_fr \
OUTPUT_ROOT=./results/exp-bert-opus-books-norm-lr \
MAX_PAIRS=50000 \
MIN_FREQ=2 \
NUM_ONCE=1 \
bash BERT/run_bert_translation_norm_lr_batch.sh
```

Run generated jobs:

```bash
cd BERT/exp-bert-opus-books-norm-lr
bash z_bash_execute.sh
```

Summarize results:

```bash
python BERT/summarize_translation_results.py \
  --results-root ./results/exp-bert-opus-books-norm-lr \
  --output ./results/exp-bert-opus-books-norm-lr/summary.csv
```

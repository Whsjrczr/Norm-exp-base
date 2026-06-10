#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
CUDA_VISIBLE_DEVICES=0 python /e/norm-exp/nanoGPT/nanogpt.py \
  --arch=nanoGPT \
  --data-dir=/e/norm-exp/dataset/tinyshakespeare \
  --no-auto-prepare \
  --n-layer=8 \
  --n-head=8 \
  --n-embd=512 \
  --block-size=256 \
  --batch-size=64,64 \
  --epochs=50 \
  --iters-per-epoch=100 \
  --eval-iters=50 \
  --lr=6e-4 \
  --lr-method=cos \
  --optimizer=adamw \
  --weight-decay=0.1 \
  --dropout=0.2 \
  --attn-norm=LN \
  --mlp-norm=CSBN \
  --final-norm=LN \
  --activation=gelu \
  --dtype=bfloat16 \
  --seed=2 \
  --sample-tokens=0 \
  --sample-every=0 \
  --print-f=1 \
  --output=/e/norm-exp/results/exp-nanogpt-sequence-bn-followup \
  --visualize \
  --wandb_project=nanoGPT-Norm-Scaling-Followup \
  --no-save-checkpoint

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

python_bin="${PYTHON_BIN:-${repo_root}/.conda/python.exe}"
dataset_root="${DATASET_ROOT:-${repo_root}/dataset/tinyshakespeare}"
output_root="${OUTPUT_ROOT:-${repo_root}/results/nanogpt-cfbn}"
wandb_project="${WANDB_PROJECT:-nanoGPT-CFBN}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"

n_layer="${N_LAYER:-6}"
n_head="${N_HEAD:-6}"
n_embd="${N_EMBD:-384}"
block_size="${BLOCK_SIZE:-256}"
batch_size="${BATCH_SIZE:-64,64}"
epochs="${EPOCHS:-50}"
iters_per_epoch="${ITERS_PER_EPOCH:-100}"
eval_iters="${EVAL_ITERS:-50}"
optimizer="${OPTIMIZER:-adamw}"
lr="${LR:-6e-4}"
lr_method="${LR_METHOD:-cos}"
weight_decay="${WEIGHT_DECAY:-0.1}"
dropout="${DROPOUT:-0.2}"
activation="${ACTIVATION:-gelu}"
dtype="${DTYPE:-bfloat16}"
seed="${SEED:-0}"
sample_tokens="${SAMPLE_TOKENS:-0}"
sample_every="${SAMPLE_EVERY:-0}"
display_every="${DISPLAY_EVERY:-1}"

norms=(${NORMS:-CCFBN CCFBNc CCFBNs CSBN CDSeqBN LN RMS})
slots=(${SLOTS:-attn mlp final all})

mkdir -p "${output_root}"

run_one() {
  local norm="$1"
  local slot="$2"
  local extra_args=()

  if [ "${slot}" = "attn" ]; then
    extra_args=(--attn-norm="${norm}" --mlp-norm=LN --final-norm=LN)
  elif [ "${slot}" = "mlp" ]; then
    extra_args=(--attn-norm=LN --mlp-norm="${norm}" --final-norm=LN)
  elif [ "${slot}" = "final" ]; then
    extra_args=(--attn-norm=LN --mlp-norm=LN --final-norm="${norm}")
  elif [ "${slot}" = "all" ]; then
    extra_args=(--norm="${norm}")
  else
    echo "Unknown slot: ${slot}" >&2
    exit 1
  fi

  echo "Running nanoGPT ${slot} ${norm} lr=${lr}"
  CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${python_bin}" "${repo_root}/nanoGPT/nanogpt.py" \
    --arch=nanoGPT \
    --data-dir="${dataset_root}" \
    --no-auto-prepare \
    --n-layer="${n_layer}" \
    --n-head="${n_head}" \
    --n-embd="${n_embd}" \
    --block-size="${block_size}" \
    --batch-size="${batch_size}" \
    --epochs="${epochs}" \
    --iters-per-epoch="${iters_per_epoch}" \
    --eval-iters="${eval_iters}" \
    --lr="${lr}" \
    --lr-method="${lr_method}" \
    --optimizer="${optimizer}" \
    --weight-decay="${weight_decay}" \
    --dropout="${dropout}" \
    "${extra_args[@]}" \
    --activation="${activation}" \
    --dtype="${dtype}" \
    --seed="${seed}" \
    --sample-tokens="${sample_tokens}" \
    --sample-every="${sample_every}" \
    --print-f="${display_every}" \
    --output="${output_root}" \
    --visualize \
    --wandb_project="${wandb_project}" \
    --no-save-checkpoint
}

for norm in "${norms[@]}"; do
  for slot in "${slots[@]}"; do
    if [ "${norm}" = "LN" ] && [ "${slot}" != "all" ]; then
      continue
    fi
    run_one "${norm}" "${slot}"
  done
done

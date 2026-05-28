#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

python_bin="${PYTHON_BIN:-python}"
data_dir="${DATA_DIR:-${repo_root}/dataset/tinyshakespeare}"
output_root="${OUTPUT_ROOT:-${repo_root}/results/nanogpt-sequence-bn}"
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
weight_decay="${WEIGHT_DECAY:-0.1}"
dropout="${DROPOUT:-0.2}"
activation="${ACTIVATION:-gelu}"
dtype="${DTYPE:-bfloat16}"
seed="${SEED:-0}"
sample_tokens="${SAMPLE_TOKENS:-0}"
sample_every="${SAMPLE_EVERY:-0}"

slots=(attn mlp final all)
norms=(
  CSBN CSBNc CSBNs
  CSeqBN CSeqBNc CSeqBNs
  CDSeqBN CDSeqBNc CDSeqBNs
)

mkdir -p "${output_root}"

if [[ ! -f "${data_dir}/train.bin" || ! -f "${data_dir}/val.bin" || ! -f "${data_dir}/meta.json" ]]; then
  "${python_bin}" "${repo_root}/nanoGPT/prepare_tinyshakespeare.py" --data-dir "${data_dir}"
fi

for slot in "${slots[@]}"; do
  for norm in "${norms[@]}"; do
    echo "Running nanoGPT TinyShakespeare slot=${slot} norm=${norm}"
    norm_args=()
    case "${slot}" in
      attn)
        norm_args=(--attn-norm="${norm}" --mlp-norm=LN --final-norm=LN)
        ;;
      mlp)
        norm_args=(--attn-norm=LN --mlp-norm="${norm}" --final-norm=LN)
        ;;
      final)
        norm_args=(--attn-norm=LN --mlp-norm=LN --final-norm="${norm}")
        ;;
      all)
        norm_args=(--norm="${norm}")
        ;;
      *)
        echo "Unknown slot: ${slot}" >&2
        exit 1
        ;;
    esac

    CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${python_bin}" "${repo_root}/nanoGPT/nanogpt.py" \
      --data-dir="${data_dir}" \
      --no-auto-prepare \
      --n-layer="${n_layer}" \
      --n-head="${n_head}" \
      --n-embd="${n_embd}" \
      --block-size="${block_size}" \
      --batch-size="${batch_size}" \
      --epochs="${epochs}" \
      --iters-per-epoch="${iters_per_epoch}" \
      --eval-iters="${eval_iters}" \
      --optimizer="${optimizer}" \
      --lr="${lr}" \
      --lr-method=cos \
      --weight-decay="${weight_decay}" \
      --dropout="${dropout}" \
      --activation="${activation}" \
      --dtype="${dtype}" \
      --seed="${seed}" \
      --sample-tokens="${sample_tokens}" \
      --sample-every="${sample_every}" \
      --output="${output_root}" \
      "${norm_args[@]}"
  done
done

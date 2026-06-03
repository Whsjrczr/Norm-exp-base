#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

python_bin="${PYTHON_BIN:-python}"
dataset="${DATASET:-cifar10}"
dataset_root="${DATASET_ROOT:-${repo_root}/dataset}"
output_root="${OUTPUT_ROOT:-${repo_root}/results/vit-cfbn}"
wandb_project="${WANDB_PROJECT:-ViT-CFBN}"
arch="${ARCH:-vit_small}"
image_size="${IMAGE_SIZE:-32}"
batch_size="${BATCH_SIZE:-256}"
epochs="${EPOCHS:-200}"
optimizer="${OPTIMIZER:-adamw}"
weight_decay="${WEIGHT_DECAY:-0.1}"
dropout="${DROPOUT:-0.0}"
drop_path_rate="${DROP_PATH_RATE:-0.1}"
seed="${SEED:-0}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"

patch_sizes=(${PATCH_SIZES:-4 8 16})
norms=(${NORMS:-CFBN CFBNc CFBNs BN BNs SBN SBNs DSeqBN DSeqBNs LN RMS})
activations=(${ACTIVATIONS:-relu})
lrs=(${LRS:-1e-4})

mkdir -p "${output_root}"

for patch_size in "${patch_sizes[@]}"; do
  if (( patch_size > image_size )); then
    continue
  fi

  for norm in "${norms[@]}"; do
    for activation in "${activations[@]}"; do
      for lr in "${lrs[@]}"; do
        echo "Running ${arch} ${dataset} img${image_size} patch${patch_size} ${norm} ${activation} lr=${lr}"
        CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${python_bin}" "${repo_root}/ViT/vit.py" \
          --arch="${arch}" \
          --dataset="${dataset}" \
          --dataset-root="${dataset_root}" \
          --im-size="${image_size},${image_size}" \
          --patch-size="${patch_size}" \
          --batch-size="${batch_size}" \
          --epochs="${epochs}" \
          --optimizer="${optimizer}" \
          --lr="${lr}" \
          --lr-method=cos \
          --weight-decay="${weight_decay}" \
          --dropout="${dropout}" \
          --drop-path-rate="${drop_path_rate}" \
          --norm="${norm}" \
          --activation="${activation}" \
          --seed="${seed}" \
          --output="${output_root}" \
          --visualize \
          --wandb_project="${wandb_project}"
      done
    done
  done
done

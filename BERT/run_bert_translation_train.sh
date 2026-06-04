#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

python_bin="${PYTHON_BIN:-${repo_root}/.conda/python.exe}"
if [ ! -x "${python_bin}" ]; then
  python_bin="${PYTHON_BIN:-python}"
fi

input_file="${INPUT_FILE:-}"
data_dir="${DATA_DIR:-${repo_root}/dataset/text_translation}"
output_root="${OUTPUT_ROOT:-${repo_root}/results}"
sample_src="${SAMPLE_SRC:-hello}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"

extra_args=()
if [ -n "${input_file}" ]; then
  extra_args+=(--input-file "${input_file}" --overwrite-data)
fi

CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${python_bin}" "${repo_root}/BERT/bert_translation.py" \
  --data-dir="${data_dir}" \
  "${extra_args[@]}" \
  --bert-layers=4 \
  --bert-heads=4 \
  --bert-embd=256 \
  --max-src-len=64 \
  --max-tgt-len=64 \
  --batch-size=64,64 \
  --iters-per-epoch=100 \
  --eval-iters=50 \
  --epochs=50 \
  --optimizer=adamw \
  --lr=3e-4 \
  --lr-method=cos \
  --weight-decay=0.01 \
  --norm=LN \
  --activation=gelu \
  --dtype=bfloat16 \
  --sample-src="${sample_src}" \
  --output="${output_root}"

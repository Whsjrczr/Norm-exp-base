#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"
while [[ "${repo_root}" != "/" && ! -d "${repo_root}/extension" ]]; do
  repo_root="$(dirname "${repo_root}")"
done
if [[ ! -d "${repo_root}/extension" ]]; then
  echo "Cannot find repository root from ${script_dir}" >&2
  exit 1
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${repo_root}/.conda/python.exe" ]]; then
    python_bin="${repo_root}/.conda/python.exe"
  elif [[ -x "${repo_root}/.conda/bin/python" ]]; then
    python_bin="${repo_root}/.conda/bin/python"
  else
    python_bin="python"
  fi
else
  python_bin="${PYTHON_BIN}"
fi

output_dir="${OUTPUT_DIR:-${repo_root}/results/vit-norm-dimension-probe}"
data_mode="${DATA_MODE:-synthetic}"
dataset_root="${DATASET_ROOT:-${repo_root}/dataset}"

"${python_bin}" "${repo_root}/ViT/vit_norm_dimension_probe.py" \
  --data-mode "${data_mode}" \
  --dataset-root "${dataset_root}" \
  --output-dir "${output_dir}" \
  --norms "${NORMS:-LN,SBN,CSBN,SeqBN,CSeqBN,DSeqBN,CDSeqBN,CFBN,CCFBN}" \
  --image-size "${IMAGE_SIZE:-32}" \
  --patch-size "${PATCH_SIZE:-8}" \
  --num-classes "${NUM_CLASSES:-10}" \
  --embed-dim "${EMBED_DIM:-64}" \
  --depth "${DEPTH:-2}" \
  --num-heads "${NUM_HEADS:-2}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --probe-batch-size "${PROBE_BATCH_SIZE:-32}" \
  --train-steps "${TRAIN_STEPS:-100}" \
  --eval-iters "${EVAL_ITERS:-10}" \
  --stat-every "${STAT_EVERY:-10}" \
  --lr "${LR:-1e-3}" \
  --weight-decay "${WEIGHT_DECAY:-0.05}" \
  --seed "${SEED:-0}" \
  "$@"

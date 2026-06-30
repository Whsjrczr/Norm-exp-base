#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
output_dir="${OUTPUT_DIR:-${repo_root}/results/norm-dimension-probe}"
data_mode="${DATA_MODE:-tinyshakespeare}"
data_dir="${DATA_DIR:-${repo_root}/dataset/tinyshakespeare}"

"${python_bin}" "${repo_root}/nanoGPT/norm_dimension_probe.py" \
  --data-mode "${data_mode}" \
  --data-dir "${data_dir}" \
  --output-dir "${output_dir}" \
  --norms "${NORMS:-LN,SBN,CSBN,SeqBN,CSeqBN,DSeqBN,CDSeqBN,CFBN,CCFBN}" \
  --n-layer "${N_LAYER:-2}" \
  --n-head "${N_HEAD:-2}" \
  --n-embd "${N_EMBD:-64}" \
  --block-size "${BLOCK_SIZE:-64}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --probe-batch-size "${PROBE_BATCH_SIZE:-32}" \
  --train-steps "${TRAIN_STEPS:-100}" \
  --eval-iters "${EVAL_ITERS:-10}" \
  --stat-every "${STAT_EVERY:-10}" \
  --lr "${LR:-6e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.1}" \
  --seed "${SEED:-0}"

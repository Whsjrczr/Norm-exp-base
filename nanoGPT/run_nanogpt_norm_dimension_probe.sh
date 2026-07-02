#!/usr/bin/env bash
set -euo pipefail

# Generate nanoGPT norm-dimension probe job scripts in the same style as the
# nanoGPT EMA-SBN experiment launcher.

# Default sweep:
#   one job per norm and seed, with isolated output directories so CSV/JSONL
#   artifacts can be compared without overwriting each other.
# Extend with:
#   SLOTS="all attn mlp final"
#   LRS="6e-4 1e-3"
#   BLOCK_SIZES="64 128 256"
#   NORM_CFGS="- momentum=0.05 momentum=0.10"

dir_name="${DIR_NAME:-exp-nanogpt-norm-dimension-probe}"

script_path="${BASH_SOURCE[0]}"
if [[ "${script_path}" == */* ]]; then
  script_dir="$(cd "${script_path%/*}" && pwd)"
else
  script_dir="$(pwd)"
fi

if [[ -d "${script_dir}/nanoGPT" && -d "${script_dir}/extension" ]]; then
  repo_root="${script_dir}"
else
  repo_root="$(cd "${script_dir}/.." && pwd)"
fi

gen_dir="${script_dir}/${dir_name}"
mkdir -p "${gen_dir}"
cp "$0" "${gen_dir}/gen_script.sh"

norms_value="${NORMS:-LN,SBN,CSBN,SeqBN,CSeqBN,DSeqBN,CDSeqBN,CFBN,CCFBN}"
norms_value="${norms_value//,/ }"
read -r -a norms <<< "${norms_value}"

slots_value="${SLOTS:-all}"
slots_value="${slots_value//,/ }"
read -r -a slots <<< "${slots_value}"

seeds_value="${SEEDS:-0}"
seeds_value="${seeds_value//,/ }"
read -r -a seeds <<< "${seeds_value}"

lrs_value="${LRS:-${LR:-6e-4}}"
lrs_value="${lrs_value//,/ }"
read -r -a lrs <<< "${lrs_value}"

block_sizes_value="${BLOCK_SIZES:-${BLOCK_SIZE:-64}}"
block_sizes_value="${block_sizes_value//,/ }"
read -r -a block_sizes <<< "${block_sizes_value}"

norm_cfgs_value="${NORM_CFGS:--}"
norm_cfgs_value="${norm_cfgs_value//,/ }"
read -r -a norm_cfgs <<< "${norm_cfgs_value}"

max_jobs="${MAX_JOBS:-}"
phase="${PHASE:-dimension_probe}"

data_mode="${DATA_MODE:-tinyshakespeare}"
data_dir="${DATA_DIR:-/home/dlth/norm-exp-code/dataset/tinyshakespeare}"
output_root="${OUTPUT_ROOT:-/home/dlth/norm-exp-code/Norm-exp-base/nanoGPT/results/${dir_name}}"
python_bin="${PYTHON_BIN:-/home/dlth/miniconda3/envs/norm-base/bin/python}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-1}"
num_once="${NUM_ONCE:-1}"

device="${DEVICE:-auto}"
vocab_size="${VOCAB_SIZE:-65}"
n_layer="${N_LAYER:-2}"
n_head="${N_HEAD:-2}"
n_embd="${N_EMBD:-64}"
batch_size="${BATCH_SIZE:-32}"
probe_batch_size="${PROBE_BATCH_SIZE:-32}"
train_steps="${TRAIN_STEPS:-100}"
eval_iters="${EVAL_ITERS:-10}"
stat_every="${STAT_EVERY:-10}"
weight_decay="${WEIGHT_DECAY:-0.1}"
grad_clip="${GRAD_CLIP:-1.0}"
dropout="${DROPOUT:-0.0}"
activation="${ACTIVATION:-gelu}"
causal_change_pos="${CAUSAL_CHANGE_POS:--}"
auto_prepare="${AUTO_PREPARE:-1}"
bias="${BIAS:-0}"
norm_no_affine="${NORM_NO_AFFINE:-0}"

launch_cnt=0
: > "${gen_dir}/z_bash_execute.sh"

sanitize() {
  local value="$1"
  if [ "${value}" = "-" ]; then
    printf 'none'
    return
  fi
  value="${value//+/p}"
  value="${value//,/x}"
  value="${value//./p}"
  value="${value//=/}"
  value="${value//\//-}"
  value="${value// /}"
  printf '%s' "${value}"
}

should_generate_job() {
  if [[ -n "${max_jobs}" ]] && (( launch_cnt >= max_jobs )); then
    return 1
  fi
  return 0
}

append_launcher() {
  local file_name="$1"
  local base_string="$2"

  echo "nohup bash ${file_name} > output_${base_string}.out 2>&1 &" >> "${gen_dir}/z_bash_execute.sh"
  launch_cnt=$((launch_cnt + 1))
  if (( launch_cnt % num_once == 0 )); then
    echo "wait" >> "${gen_dir}/z_bash_execute.sh"
  fi
}

norm_site_arg_for() {
  local slot="$1"
  if [ "${slot}" != "-" ]; then
    printf '  --norm-site=%s \\\n' "${slot}"
  fi
}

optional_probe_args() {
  local norm_cfg="$1"

  if [ "${norm_cfg}" != "-" ]; then
    printf '  --norm-cfg=%s \\\n' "${norm_cfg}"
  fi
  if [ "${causal_change_pos}" != "-" ]; then
    printf '  --causal-change-pos=%s \\\n' "${causal_change_pos}"
  fi
  if [ "${auto_prepare}" = "0" ]; then
    printf '  --no-auto-prepare \\\n'
  fi
  if [ "${bias}" = "1" ]; then
    printf '  --bias \\\n'
  fi
  if [ "${norm_no_affine}" = "1" ]; then
    printf '  --norm-no-affine \\\n'
  fi
}

generate_job() {
  should_generate_job || return 0

  local norm="$1"
  local slot="$2"
  local lr="$3"
  local seed="$4"
  local local_block_size="$5"
  local norm_cfg="$6"

  local norm_tag slot_tag lr_tag cfg_tag base_string file_name
  norm_tag="$(sanitize "${norm}")"
  slot_tag="$(sanitize "${slot}")"
  lr_tag="$(sanitize "${lr}")"
  cfg_tag="$(sanitize "${norm_cfg}")"

  base_string="execute_${phase}_nanogpt_${data_mode}_L${n_layer}_H${n_head}_D${n_embd}_ctx${local_block_size}_${slot_tag}${norm_tag}_lr${lr_tag}_bs${batch_size}_steps${train_steps}_s${seed}"
  if [ "${norm_cfg}" != "-" ]; then
    base_string="${base_string}_${cfg_tag}"
  fi
  file_name="${base_string}.sh"

  echo "Generating ${base_string}"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd "%s"\n' "${repo_root}"
    printf 'CUDA_VISIBLE_DEVICES=%s %s %s \\\n' "${cuda_visible_devices}" "${python_bin}" "${repo_root}/nanoGPT/norm_dimension_probe.py"
    printf '  --data-mode=%s \\\n' "${data_mode}"
    printf '  --data-dir=%s \\\n' "${data_dir}"
    printf '  --output-dir=%s \\\n' "${output_root}/${base_string}"
    printf '  --norms=%s \\\n' "${norm}"
    norm_site_arg_for "${slot}"
    optional_probe_args "${norm_cfg}"
    printf '  --device=%s \\\n' "${device}"
    printf '  --vocab-size=%s \\\n' "${vocab_size}"
    printf '  --n-layer=%s \\\n' "${n_layer}"
    printf '  --n-head=%s \\\n' "${n_head}"
    printf '  --n-embd=%s \\\n' "${n_embd}"
    printf '  --block-size=%s \\\n' "${local_block_size}"
    printf '  --batch-size=%s \\\n' "${batch_size}"
    printf '  --probe-batch-size=%s \\\n' "${probe_batch_size}"
    printf '  --train-steps=%s \\\n' "${train_steps}"
    printf '  --eval-iters=%s \\\n' "${eval_iters}"
    printf '  --stat-every=%s \\\n' "${stat_every}"
    printf '  --lr=%s \\\n' "${lr}"
    printf '  --weight-decay=%s \\\n' "${weight_decay}"
    printf '  --grad-clip=%s \\\n' "${grad_clip}"
    printf '  --dropout=%s \\\n' "${dropout}"
    printf '  --activation=%s \\\n' "${activation}"
    printf '  --seed=%s\n' "${seed}"
  } > "${gen_dir}/${file_name}"
  chmod +x "${gen_dir}/${file_name}"
  append_launcher "${file_name}" "${base_string}"
}

for seed in "${seeds[@]}"; do
  for local_block_size in "${block_sizes[@]}"; do
    for lr in "${lrs[@]}"; do
      for norm_cfg in "${norm_cfgs[@]}"; do
        for slot in "${slots[@]}"; do
          for norm in "${norms[@]}"; do
            generate_job "${norm}" "${slot}" "${lr}" "${seed}" "${local_block_size}" "${norm_cfg}"
          done
        done
      done
    done
  done
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

chmod +x "${gen_dir}/z_bash_execute.sh"
echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
echo "Run with: cd ${gen_dir} && bash z_bash_execute.sh"

#!/usr/bin/env bash
set -euo pipefail

# Generate centering-causality experiment job scripts in the same style as the
# nanoGPT EMA-SBN experiment launcher.

# Phases:
#   nanoGPT mechanism:
#     LN/RMS controls, CSBN-family MLP/site sweeps, mean-shift tests, and
#     centering-rescue ablations.
#   ViT mechanism:
#     Patch/LR grid over normalization families plus site, mean-shift, and
#     centering-rescue ablations.

dir_name="${DIR_NAME:-exp-centering-causality}"

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

target="${TARGET:-both}"
seeds_value="${SEEDS:-0 1 2}"
seeds_value="${seeds_value//,/ }"
read -r -a seeds <<< "${seeds_value}"
max_jobs="${MAX_JOBS:-}"

nanogpt_arch="${ARCH:-nanoGPT}"
nanogpt_dataset="${DATASET:-tinyshakespeare}"
n_layer="${N_LAYER:-6}"
n_head="${N_HEAD:-6}"
n_embd="${N_EMBD:-384}"
block_size="${BLOCK_SIZE:-256}"
nanogpt_batch_size="${BATCH_SIZE:-64,64}"
nanogpt_epochs="${EPOCHS:-20}"
iters_per_epoch="${ITERS_PER_EPOCH:-50}"
nanogpt_eval_iters="${EVAL_ITERS:-20}"
nanogpt_lr_method="${LR_METHOD:-cos}"
nanogpt_weight_decay="${WEIGHT_DECAY:-0.1}"
nanogpt_dropout="${DROPOUT:-0.2}"
nanogpt_activation="${ACTIVATION:-gelu}"
dtype="${DTYPE:-bfloat16}"
sample_tokens="${SAMPLE_TOKENS:-0}"
sample_every="${SAMPLE_EVERY:-0}"
display_every="${DISPLAY_EVERY:-1}"

vit_arch="${VIT_ARCH:-vit_small}"
vit_dataset="${VIT_DATASET:-cifar10}"
image_size="${IMAGE_SIZE:-32}"
vit_batch_size="${VIT_BATCH_SIZE:-256}"
vit_epochs="${VIT_EPOCHS:-80}"
vit_lr_method="${VIT_LR_METHOD:-cos}"
vit_optimizer="${VIT_OPTIMIZER:-adam}"
vit_weight_decay="${VIT_WEIGHT_DECAY:-0.1}"
vit_dropout="${VIT_DROPOUT:-0.0}"
drop_path="${DROP_PATH_RATE:-0.1}"
vit_activation="${VIT_ACTIVATION:-relu}"

subjectname="${WANDB_PROJECT:-Taiyi-centering-causality}"
wandb_entity="${WANDB_ENTITY:-whsjrc-buaa}"
dataset_root="${DATASET_ROOT:-/home/dlth/norm-exp-code/dataset}"
output_root="${OUTPUT_ROOT:-/home/dlth/norm-exp-code/Norm-exp-base/results/${dir_name}}"
python_bin="${PYTHON_BIN:-/home/dlth/miniconda3/envs/norm-base/bin/python}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-1}"
num_once="${NUM_ONCE:-1}"

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

norm_cfg_arg() {
  local norm_cfg="$1"
  if [ "${norm_cfg}" != "-" ]; then
    printf '  --norm-cfg=%s \\\n' "${norm_cfg}"
  fi
}

centering_args() {
  local intervention="$1"
  local alpha="$2"
  local rescue="$3"

  if [ "${alpha}" != "-" ]; then
    printf '  --mean-shift-alpha=%s \\\n' "${alpha}"
    printf '  --mean-shift-target=pre_norm \\\n'
  fi
  if [ "${rescue}" != "none" ]; then
    printf '  --centering-rescue=%s \\\n' "${rescue}"
  fi
  if [ "${intervention}" = "mean_shift" ] && [ "${alpha}" = "-" ]; then
    echo "mean_shift jobs require alpha" >&2
    exit 1
  fi
}

generate_nanogpt_job() {
  should_generate_job || return 0

  local phase="$1"
  local norm="$2"
  local site="$3"
  local lr="$4"
  local seed="$5"
  local intervention="$6"
  local alpha="$7"
  local rescue="$8"
  local context="$9"
  local norm_cfg="${10}"

  local lr_tag cfg_tag name_extra pc base_string file_name
  lr_tag="$(sanitize "${lr}")"
  cfg_tag="$(sanitize "${norm_cfg}")"
  pc="ctx${context}"
  name_extra=""
  if [ "${intervention}" != "none" ]; then
    name_extra="_${intervention}"
  fi
  if [ "${alpha}" != "-" ]; then
    name_extra="${name_extra}_alpha$(sanitize "${alpha}")"
  fi
  if [ "${rescue}" != "none" ]; then
    name_extra="${name_extra}_rescue${rescue}"
  fi
  if [ "${norm_cfg}" != "-" ]; then
    name_extra="${name_extra}_${cfg_tag}"
  fi

  base_string="execute_${phase}_nanogpt_${norm}_site${site}_${pc}_lr${lr_tag}_s${seed}${name_extra}"
  file_name="${base_string}.sh"

  echo "Generating ${base_string}"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd "%s"\n' "${repo_root}"
    printf 'export WANDB_PROJECT=%s\n' "${subjectname}"
    printf 'export WANDB_ENTITY=%s\n' "${wandb_entity}"
    printf 'CUDA_VISIBLE_DEVICES=%s %s %s \\\n' "${cuda_visible_devices}" "${python_bin}" "${repo_root}/nanoGPT/nanogpt.py"
    printf '  --arch=%s \\\n' "${nanogpt_arch}"
    printf '  --data-dir=%s \\\n' "${dataset_root}/${nanogpt_dataset}"
    printf '  --no-auto-prepare \\\n'
    printf '  --n-layer=%s \\\n' "${n_layer}"
    printf '  --n-head=%s \\\n' "${n_head}"
    printf '  --n-embd=%s \\\n' "${n_embd}"
    printf '  --block-size=%s \\\n' "${context}"
    printf '  --batch-size=%s \\\n' "${nanogpt_batch_size}"
    printf '  --epochs=%s \\\n' "${nanogpt_epochs}"
    printf '  --iters-per-epoch=%s \\\n' "${iters_per_epoch}"
    printf '  --eval-iters=%s \\\n' "${nanogpt_eval_iters}"
    printf '  --lr=%s \\\n' "${lr}"
    printf '  --lr-method=%s \\\n' "${nanogpt_lr_method}"
    printf '  --optimizer=adamw \\\n'
    printf '  --weight-decay=%s \\\n' "${nanogpt_weight_decay}"
    printf '  --dropout=%s \\\n' "${nanogpt_dropout}"
    printf '  --norm=%s \\\n' "${norm}"
    printf '  --norm-site=%s \\\n' "${site}"
    norm_cfg_arg "${norm_cfg}"
    centering_args "${intervention}" "${alpha}" "${rescue}"
    printf '  --activation=%s \\\n' "${nanogpt_activation}"
    printf '  --dtype=%s \\\n' "${dtype}"
    printf '  --seed=%s \\\n' "${seed}"
    printf '  --sample-tokens=%s \\\n' "${sample_tokens}"
    printf '  --sample-every=%s \\\n' "${sample_every}"
    printf '  --print-f=%s \\\n' "${display_every}"
    printf '  --output=%s \\\n' "${output_root}/nanoGPT"
    printf '  --wandb_project=%s \\\n' "${subjectname}"
    printf '  --diagnostics \\\n'
    printf '  --no-save-checkpoint\n'
  } > "${gen_dir}/${file_name}"
  chmod +x "${gen_dir}/${file_name}"
  append_launcher "${file_name}" "${base_string}"
}

generate_vit_job() {
  should_generate_job || return 0

  local phase="$1"
  local patch_size="$2"
  local norm="$3"
  local site="$4"
  local lr="$5"
  local seed="$6"
  local intervention="$7"
  local alpha="$8"
  local rescue="$9"

  if (( patch_size > image_size )); then
    return 0
  fi

  local lr_tag dp_tag wd_tag name_extra base_string file_name
  lr_tag="$(sanitize "${lr}")"
  dp_tag="$(sanitize "${drop_path}")"
  wd_tag="$(sanitize "${vit_weight_decay}")"
  name_extra=""
  if [ "${intervention}" != "none" ]; then
    name_extra="_${intervention}"
  fi
  if [ "${alpha}" != "-" ]; then
    name_extra="${name_extra}_alpha$(sanitize "${alpha}")"
  fi
  if [ "${rescue}" != "none" ]; then
    name_extra="${name_extra}_rescue${rescue}"
  fi

  base_string="execute_${phase}_vit_${vit_arch}_${vit_dataset}_img${image_size}_patch${patch_size}_${norm}_site${site}_lr${lr_tag}_bs${vit_batch_size}_dpath${dp_tag}_wd${wd_tag}_s${seed}${name_extra}"
  file_name="${base_string}.sh"

  echo "Generating ${base_string}"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd "%s"\n' "${repo_root}"
    printf 'export WANDB_PROJECT=%s\n' "${subjectname}"
    printf 'export WANDB_ENTITY=%s\n' "${wandb_entity}"
    printf 'CUDA_VISIBLE_DEVICES=%s %s %s \\\n' "${cuda_visible_devices}" "${python_bin}" "${repo_root}/ViT/vit.py"
    printf '  --arch=%s \\\n' "${vit_arch}"
    printf '  --dataset=%s \\\n' "${vit_dataset}"
    printf '  --dataset-root=%s \\\n' "${dataset_root}"
    printf '  --im-size=%s,%s \\\n' "${image_size}" "${image_size}"
    printf '  --patch-size=%s \\\n' "${patch_size}"
    printf '  --batch-size=%s \\\n' "${vit_batch_size}"
    printf '  --epochs=%s \\\n' "${vit_epochs}"
    printf '  --lr=%s \\\n' "${lr}"
    printf '  --lr-method=%s \\\n' "${vit_lr_method}"
    printf '  --optimizer=%s \\\n' "${vit_optimizer}"
    printf '  --weight-decay=%s \\\n' "${vit_weight_decay}"
    printf '  --dropout=%s \\\n' "${vit_dropout}"
    printf '  --drop-path-rate=%s \\\n' "${drop_path}"
    printf '  --norm=%s \\\n' "${norm}"
    printf '  --norm-site=%s \\\n' "${site}"
    centering_args "${intervention}" "${alpha}" "${rescue}"
    printf '  --activation=%s \\\n' "${vit_activation}"
    printf '  --seed=%s \\\n' "${seed}"
    printf '  --print-f=%s \\\n' "${display_every}"
    printf '  --output=%s \\\n' "${output_root}/ViT"
    printf '  --wandb_project=%s \\\n' "${subjectname}"
    printf '  --diagnostics \\\n'
    printf '  --no-save-checkpoint\n'
  } > "${gen_dir}/${file_name}"
  chmod +x "${gen_dir}/${file_name}"
  append_launcher "${file_name}" "${base_string}"
}

generate_nanogpt_jobs() {
  local seed spec norm site lr cfg alpha rescue
  local context="${block_size}"

  local baseline_specs=(
    "LN all 3e-4 -"
    "RMS all 3e-4 -"
  )
  local mlp_specs=(
    "CSBN mlp 6e-4 -"
    "CSBNc mlp 6e-4 -"
    "CSBNs mlp 6e-4 -"
    "EMACFBN mlp 3e-4 momentum=0.20"
    "EMACFBNc mlp 3e-4 momentum=0.20"
    "EMACFBNs mlp 3e-4 momentum=0.20"
  )
  local site_specs=(
    "CSBN attn 6e-4 -"
    "CSBN mlp 6e-4 -"
    "CSBN final 6e-4 -"
    "CSBN all 6e-4 -"
    "CSBNs attn 6e-4 -"
    "CSBNs mlp 6e-4 -"
    "CSBNs final 6e-4 -"
    "CSBNs all 6e-4 -"
  )

  for seed in "${seeds[@]}"; do
    for spec in "${baseline_specs[@]}" "${mlp_specs[@]}" "${site_specs[@]}"; do
      read -r norm site lr cfg <<< "${spec}"
      generate_nanogpt_job "mechanism" "${norm}" "${site}" "${lr}" "${seed}" "none" "-" "none" "${context}" "${cfg}"
    done
    for alpha in 0 0.01 0.03 0.1 0.3 1.0; do
      generate_nanogpt_job "mean_shift" "CSBNs" "mlp" "6e-4" "${seed}" "mean_shift" "${alpha}" "none" "${context}" "-"
    done
    for rescue in early middle late mlp attn final all; do
      generate_nanogpt_job "centering_rescue" "CSBNs" "mlp" "6e-4" "${seed}" "centering_rescue" "-" "${rescue}" "${context}" "-"
    done
  done
}

generate_vit_jobs() {
  local seed patch_size lr norm site alpha rescue
  local norms=(LN RMS BN BNc BNs SBN SBNc SBNs CFBN CFBNc CFBNs SeqBN DSeqBN)

  for seed in "${seeds[@]}"; do
    for patch_size in 4 16; do
      for lr in 1e-4 1e-3; do
        for norm in "${norms[@]}"; do
          generate_vit_job "mechanism" "${patch_size}" "${norm}" "all" "${lr}" "${seed}" "none" "-" "none"
        done
        for site in norm1 norm2 final all; do
          generate_vit_job "site_ablation" "${patch_size}" "SBNs" "${site}" "${lr}" "${seed}" "site_ablation" "-" "none"
        done
        for alpha in 0 0.01 0.03 0.1 0.3 1.0; do
          generate_vit_job "mean_shift" "${patch_size}" "SBNs" "all" "${lr}" "${seed}" "mean_shift" "${alpha}" "none"
        done
        for rescue in early middle late norm1 norm2 final all; do
          generate_vit_job "centering_rescue" "${patch_size}" "SBNs" "all" "${lr}" "${seed}" "centering_rescue" "-" "${rescue}"
        done
      done
    done
  done
}

case "${target}" in
  nanogpt)
    generate_nanogpt_jobs
    ;;
  vit)
    generate_vit_jobs
    ;;
  both)
    generate_nanogpt_jobs
    generate_vit_jobs
    ;;
  *)
    echo "Unknown TARGET: ${target}" >&2
    exit 1
    ;;
esac

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

chmod +x "${gen_dir}/z_bash_execute.sh"
echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
echo "Run with: cd ${gen_dir} && bash z_bash_execute.sh"

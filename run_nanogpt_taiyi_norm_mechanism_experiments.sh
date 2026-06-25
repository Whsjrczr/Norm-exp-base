#!/usr/bin/env bash
set -euo pipefail

# Taiyi mechanism diagnostics for SBN/CFBN and centering ablations in nanoGPT.
#
# Phases:
#   1. mechanism_probe:
#      Seed-0 local/global comparison of LN/RMS, CSBN c/s/full, and EMACFBN c/s/full.
#   2. context_probe:
#      Context length boundary for centering versus scaling-only variants.
#   3. multiseed_confirm:
#      Small multiseed confirmation for the strongest mechanism hypotheses.

dir_name="${DIR_NAME:-exp-nanogpt-taiyi-norm-mechanism}"

script_path="${BASH_SOURCE[0]}"
if [[ "${script_path}" == */* ]]; then
  script_dir="$(cd "${script_path%/*}" && pwd)"
else
  script_dir="$(pwd)"
fi

if [[ -d "${script_dir}/nanoGPT" ]]; then
  repo_root="${script_dir}"
else
  repo_root="$(cd "${script_dir}/.." && pwd)"
fi

gen_dir="${script_dir}/${dir_name}"
mkdir -p "${gen_dir}"
cp "$0" "${gen_dir}/gen_script.sh"

arch="${ARCH:-nanoGPT}"
dataset="${DATASET:-tinyshakespeare}"
n_layer="${N_LAYER:-6}"
n_head="${N_HEAD:-6}"
n_embd="${N_EMBD:-384}"
block_size="${BLOCK_SIZE:-256}"
batch_size="${BATCH_SIZE:-64,64}"
activation="${ACTIVATION:-gelu}"
epochs="${EPOCHS:-20}"
iters_per_epoch="${ITERS_PER_EPOCH:-50}"
eval_iters="${EVAL_ITERS:-20}"
display_every="${DISPLAY_EVERY:-1}"
weight_decay="${WEIGHT_DECAY:-0.1}"
dropout="${DROPOUT:-0.2}"
lr_method="${LR_METHOD:-cos}"
dtype="${DTYPE:-bfloat16}"
sample_tokens="${SAMPLE_TOKENS:-0}"
sample_every="${SAMPLE_EVERY:-0}"
subjectname="${WANDB_PROJECT:-nanoGPT-Taiyi-norm-mechanism}"
offline="${OFFLINE:-0}"

dataset_root="${DATASET_ROOT:-/home/dlth/norm-exp-code/dataset}"
output_root="${OUTPUT_ROOT:-/home/dlth/norm-exp-code/Norm-exp-base/nanoGPT/results/${dir_name}}"
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
  printf '%s' "${value}"
}

norm_args_for() {
  local norm="$1"
  local slot="$2"
  local norm_cfg="$3"

  case "${slot}" in
    baseline|all)
      printf '  --norm=%s \\\n' "${norm}"
      if [ "${norm_cfg}" != "-" ]; then
        printf '  --norm-cfg=%s \\\n' "${norm_cfg}"
      fi
      ;;
    attn)
      printf '  --attn-norm=%s \\\n  --mlp-norm=LN \\\n  --final-norm=LN \\\n' "${norm}"
      if [ "${norm_cfg}" != "-" ]; then
        printf '  --attn-norm-cfg=%s \\\n' "${norm_cfg}"
      fi
      ;;
    mlp)
      printf '  --attn-norm=LN \\\n  --mlp-norm=%s \\\n  --final-norm=LN \\\n' "${norm}"
      if [ "${norm_cfg}" != "-" ]; then
        printf '  --mlp-norm-cfg=%s \\\n' "${norm_cfg}"
      fi
      ;;
    final)
      printf '  --attn-norm=LN \\\n  --mlp-norm=LN \\\n  --final-norm=%s \\\n' "${norm}"
      if [ "${norm_cfg}" != "-" ]; then
        printf '  --final-norm-cfg=%s \\\n' "${norm_cfg}"
      fi
      ;;
    *)
      echo "Unknown slot: ${slot}" >&2
      exit 1
      ;;
  esac
}

tag_for() {
  local norm="$1"
  local slot="$2"
  local norm_cfg="$3"
  local norm_tag cfg_tag
  norm_tag="$(sanitize "${norm}")"
  cfg_tag="$(sanitize "${norm_cfg}")"

  case "${slot}" in
    baseline) printf 'control%s' "${norm_tag}" ;;
    all) printf 'all%s' "${norm_tag}" ;;
    attn|mlp|final) printf '%s%s' "${slot}" "${norm_tag}" ;;
    *)
      echo "Unknown slot: ${slot}" >&2
      exit 1
      ;;
  esac

  if [ "${norm_cfg}" != "-" ]; then
    printf '_%s' "${cfg_tag}"
  fi
}

generate_job() {
  local phase="$1"
  local norm="$2"
  local slot="$3"
  local lr="$4"
  local seed="$5"
  local norm_cfg="$6"
  local local_block_size="${7:-${block_size}}"

  local norm_tag lr_tag batch_tag
  norm_tag="$(tag_for "${norm}" "${slot}" "${norm_cfg}")"
  lr_tag="$(sanitize "${lr}")"
  batch_tag="$(sanitize "${batch_size}")"

  local base_string="execute_${phase}_${arch}_${dataset}_L${n_layer}_H${n_head}_D${n_embd}_ctx${local_block_size}_${norm_tag}_${activation}_lr${lr_tag}_bs${batch_tag}_drop${dropout}_wd${weight_decay}_s${seed}_adamw"
  local file_name="${base_string}.sh"

  echo "Generating ${base_string}"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd "%s"\n' "${repo_root}"
    printf 'CUDA_VISIBLE_DEVICES=%s %s %s \\\n' "${cuda_visible_devices}" "${python_bin}" "${repo_root}/nanoGPT/nanogpt.py"
    printf '  --arch=%s \\\n' "${arch}"
    printf '  --data-dir=%s \\\n' "${dataset_root}"
    printf '  --no-auto-prepare \\\n'
    printf '  --n-layer=%s \\\n' "${n_layer}"
    printf '  --n-head=%s \\\n' "${n_head}"
    printf '  --n-embd=%s \\\n' "${n_embd}"
    printf '  --block-size=%s \\\n' "${local_block_size}"
    printf '  --batch-size=%s \\\n' "${batch_size}"
    printf '  --epochs=%s \\\n' "${epochs}"
    printf '  --iters-per-epoch=%s \\\n' "${iters_per_epoch}"
    printf '  --eval-iters=%s \\\n' "${eval_iters}"
    printf '  --lr=%s \\\n' "${lr}"
    printf '  --lr-method=%s \\\n' "${lr_method}"
    printf '  --optimizer=adamw \\\n'
    printf '  --weight-decay=%s \\\n' "${weight_decay}"
    printf '  --dropout=%s \\\n' "${dropout}"
    norm_args_for "${norm}" "${slot}" "${norm_cfg}"
    printf '  --activation=%s \\\n' "${activation}"
    printf '  --dtype=%s \\\n' "${dtype}"
    printf '  --seed=%s \\\n' "${seed}"
    printf '  --sample-tokens=%s \\\n' "${sample_tokens}"
    printf '  --sample-every=%s \\\n' "${sample_every}"
    printf '  --print-f=%s \\\n' "${display_every}"
    printf '  --output=%s \\\n' "${output_root}"
    printf '  --wandb_project=%s \\\n' "${subjectname}"
    printf '  --diagnostics \\\n'
    if [ "${offline}" = "1" ]; then
      printf '  --offline \\\n'
    fi
    printf '  --no-save-checkpoint\n'
  } > "${gen_dir}/${file_name}"
  chmod +x "${gen_dir}/${file_name}"

  echo "nohup bash ${file_name} > output_${base_string}.out 2>&1 &" >> "${gen_dir}/z_bash_execute.sh"
  launch_cnt=$((launch_cnt + 1))
  if (( launch_cnt % num_once == 0 )); then
    echo "wait" >> "${gen_dir}/z_bash_execute.sh"
  fi
}

mechanism_specs=(
  "LN baseline 3e-4 -"
  "RMS baseline 3e-4 -"
  "CSBN mlp 6e-4 -"
  "CSBN mlp 1e-3 -"
  "CSBNc mlp 6e-4 -"
  "CSBNs mlp 6e-4 -"
  "CSBN all 6e-4 -"
  "CSBNs all 6e-4 -"
  "EMACFBN mlp 3e-4 momentum=0.20"
  "EMACFBNc mlp 3e-4 momentum=0.20"
  "EMACFBNs mlp 3e-4 momentum=0.20"
  "EMACFBN all 3e-4 momentum=0.20"
  "EMACFBNs all 3e-4 momentum=0.20"
)

for spec in "${mechanism_specs[@]}"; do
  read -r norm slot lr norm_cfg <<< "${spec}"
  generate_job "mechanism_probe" "${norm}" "${slot}" "${lr}" "0" "${norm_cfg}"
done

context_sizes=(128 512)
context_specs=(
  "LN baseline 3e-4 -"
  "CSBN mlp 6e-4 -"
  "CSBNs mlp 6e-4 -"
  "EMACFBN mlp 3e-4 momentum=0.20"
  "EMACFBNs mlp 3e-4 momentum=0.20"
)

for ctx in "${context_sizes[@]}"; do
  for spec in "${context_specs[@]}"; do
    read -r norm slot lr norm_cfg <<< "${spec}"
    generate_job "context_probe" "${norm}" "${slot}" "${lr}" "0" "${norm_cfg}" "${ctx}"
  done
done

confirm_seeds=(1 2)
confirm_specs=(
  "LN baseline 3e-4 -"
  "CSBN mlp 6e-4 -"
  "CSBNs mlp 6e-4 -"
  "EMACFBN mlp 3e-4 momentum=0.20"
  "EMACFBNs mlp 3e-4 momentum=0.20"
)

for seed in "${confirm_seeds[@]}"; do
  for spec in "${confirm_specs[@]}"; do
    read -r norm slot lr norm_cfg <<< "${spec}"
    generate_job "multiseed_confirm" "${norm}" "${slot}" "${lr}" "${seed}" "${norm_cfg}"
  done
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

chmod +x "${gen_dir}/z_bash_execute.sh"
echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
echo "Run with: cd ${gen_dir} && bash z_bash_execute.sh"

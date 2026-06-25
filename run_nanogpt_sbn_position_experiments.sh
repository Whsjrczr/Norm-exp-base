#!/usr/bin/env bash
set -euo pipefail

# Focused nanoGPT follow-up experiments for the current SBN/EMA-CFBN conclusion.
#
# Current writing target:
#   1. Main nanoGPT claim:
#      mlp/CSBN is the most stable causal norm replacement so far; it beats LN,
#      all-slot CSBN, and the CSeqBN/CDSeqBN family.
#   2. Secondary nanoGPT claim:
#      mlp/EMACFBN is the strongest competitor, especially momentum=0.20 and
#      lr=3e-4 over four seeds.
#   3. Structural claim:
#      Local replacement is better than global replacement; MLP pre-norm is the
#      critical site.
#   4. Negative claim:
#      SeqBN/DSeqBN are not first tier; scaling-only and some centering-only
#      variants are unstable.
#
# Design:
#   1. csbn_primary_confirm:
#      Four-seed confirmation of LN, mlp/CSBN, all/CSBN, CSeqBN/CDSeqBN controls.
#   2. ema_cfbn_confirm:
#      Four-seed confirmation of mlp/EMACFBN around m=0.20, lr=3e-4 plus local
#      vs global EMA-CFBN controls.
#   3. ema_cfbn_local_refine:
#      Seed-0 local LR x momentum grid near the strong m=0.20, lr=3e-4 point.
#   4. negative_ablation:
#      Scaling-only, centering-only, and SeqBN-family negative controls.
#   5. context_boundary:
#      Short/long context check for the two recommended language-model choices.

dir_name="${DIR_NAME:-exp-nanogpt-sbn-current-followup}"

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
epochs="${EPOCHS:-50}"
iters_per_epoch="${ITERS_PER_EPOCH:-100}"
eval_iters="${EVAL_ITERS:-50}"
display_every="${DISPLAY_EVERY:-1}"
weight_decay="${WEIGHT_DECAY:-0.1}"
dropout="${DROPOUT:-0.2}"
lr_method="${LR_METHOD:-cos}"
dtype="${DTYPE:-bfloat16}"
sample_tokens="${SAMPLE_TOKENS:-0}"
sample_every="${SAMPLE_EVERY:-0}"
subjectname="${WANDB_PROJECT:-nanoGPT-SBN-current-followup}"

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

optimizer_extra_args_for() {
  local optimizer="$1"
  case "${optimizer}" in
    sgd)
      printf '  --optimizer-config=momentum=0.9 \\\n'
      ;;
  esac
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
    baseline)
      printf 'control%s' "${norm_tag}"
      ;;
    all)
      printf 'all%s' "${norm_tag}"
      ;;
    attn|mlp|final)
      printf '%s%s' "${slot}" "${norm_tag}"
      ;;
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
  local optimizer="$5"
  local seed="$6"
  local norm_cfg="$7"

  local norm_tag lr_tag batch_tag
  norm_tag="$(tag_for "${norm}" "${slot}" "${norm_cfg}")"
  lr_tag="$(sanitize "${lr}")"
  batch_tag="$(sanitize "${batch_size}")"

  local base_string="execute_${phase}_${arch}_${dataset}_L${n_layer}_H${n_head}_D${n_embd}_ctx${block_size}_${norm_tag}_${activation}_lr${lr_tag}_bs${batch_tag}_drop${dropout}_wd${weight_decay}_s${seed}_${optimizer}"
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
    printf '  --block-size=%s \\\n' "${block_size}"
    printf '  --batch-size=%s \\\n' "${batch_size}"
    printf '  --epochs=%s \\\n' "${epochs}"
    printf '  --iters-per-epoch=%s \\\n' "${iters_per_epoch}"
    printf '  --eval-iters=%s \\\n' "${eval_iters}"
    printf '  --lr=%s \\\n' "${lr}"
    printf '  --lr-method=%s \\\n' "${lr_method}"
    printf '  --optimizer=%s \\\n' "${optimizer}"
    optimizer_extra_args_for "${optimizer}"
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
    printf '  --visualize \\\n'
    printf '  --wandb_project=%s \\\n' "${subjectname}"
    printf '  --no-save-checkpoint\n'
  } > "${gen_dir}/${file_name}"
  chmod +x "${gen_dir}/${file_name}"

  echo "nohup bash ${file_name} > output_${base_string}.out 2>&1 &" >> "${gen_dir}/z_bash_execute.sh"
  launch_cnt=$((launch_cnt + 1))
  if (( launch_cnt % num_once == 0 )); then
    echo "wait" >> "${gen_dir}/z_bash_execute.sh"
  fi
}

generate_with_block_size() {
  local local_block_size="$1"
  shift

  local saved_block_size="${block_size}"
  block_size="${local_block_size}"
  generate_job "$@"
  block_size="${saved_block_size}"
}

primary_seeds=(0 1 2 3)
primary_specs=(
  "LN baseline 3e-4 -"
  "LN baseline 6e-4 -"
  "LN baseline 1e-3 -"
  "CSBN mlp 6e-4 -"
  "CSBN mlp 1e-3 -"
  "CSBN all 6e-4 -"
  "CSBN all 1e-3 -"
  "CSeqBN mlp 6e-4 -"
  "CSeqBN all 6e-4 -"
  "CDSeqBN mlp 3e-4 -"
  "CDSeqBN mlp 6e-4 -"
  "CDSeqBN all 3e-4 -"
  "CDSeqBN all 6e-4 -"
)

for seed in "${primary_seeds[@]}"; do
  for spec in "${primary_specs[@]}"; do
    read -r norm slot lr norm_cfg <<< "${spec}"
    generate_job "csbn_primary_confirm" "${norm}" "${slot}" "${lr}" "adamw" "${seed}" "${norm_cfg}"
  done
done

ema_confirm_seeds=(0 1 2 3)
ema_confirm_specs=(
  "EMACFBN mlp 2e-4 momentum=0.20"
  "EMACFBN mlp 3e-4 momentum=0.10"
  "EMACFBN mlp 3e-4 momentum=0.20"
  "EMACFBN mlp 4e-4 momentum=0.20"
  "EMACFBN attn 3e-4 momentum=0.20"
  "EMACFBN final 3e-4 momentum=0.20"
  "EMACFBN all 3e-4 momentum=0.20"
  "CCFBN mlp 6e-4 -"
  "CSBN mlp 1e-3 -"
  "LN baseline 3e-4 -"
)

for seed in "${ema_confirm_seeds[@]}"; do
  for spec in "${ema_confirm_specs[@]}"; do
    read -r norm slot lr norm_cfg <<< "${spec}"
    generate_job "ema_cfbn_confirm" "${norm}" "${slot}" "${lr}" "adamw" "${seed}" "${norm_cfg}"
  done
done

ema_refine_momenta=(0.10 0.15 0.20 0.25)
ema_refine_lrs=(2e-4 3e-4 4e-4 6e-4)

for momentum in "${ema_refine_momenta[@]}"; do
  for lr in "${ema_refine_lrs[@]}"; do
    generate_job "ema_cfbn_local_refine" "EMACFBN" "mlp" "${lr}" "adamw" "0" "momentum=${momentum}"
  done
done
generate_job "ema_cfbn_local_refine" "EMACFBNc" "mlp" "3e-4" "adamw" "0" "momentum=0.20"
generate_job "ema_cfbn_local_refine" "EMACFBNs" "mlp" "3e-4" "adamw" "0" "momentum=0.20"

negative_seeds=(0 1 2)
negative_specs=(
  "CSBNc mlp 6e-4 -"
  "CSBNs mlp 6e-4 -"
  "CSeqBNc mlp 6e-4 -"
  "CSeqBNs mlp 6e-4 -"
  "CDSeqBNc mlp 6e-4 -"
  "CDSeqBNs mlp 6e-4 -"
  "EMACFBNc mlp 3e-4 momentum=0.20"
  "EMACFBNs mlp 3e-4 momentum=0.20"
)

for seed in "${negative_seeds[@]}"; do
  for spec in "${negative_specs[@]}"; do
    read -r norm slot lr norm_cfg <<< "${spec}"
    generate_job "negative_ablation" "${norm}" "${slot}" "${lr}" "adamw" "${seed}" "${norm_cfg}"
  done
done

context_sizes=(128 512)
context_seeds=(0 1 2)
context_specs=(
  "LN baseline 3e-4 -"
  "CSBN mlp 1e-3 -"
  "CSBN all 1e-3 -"
  "EMACFBN mlp 3e-4 momentum=0.20"
  "EMACFBN all 3e-4 momentum=0.20"
)

for ctx in "${context_sizes[@]}"; do
  for seed in "${context_seeds[@]}"; do
    for spec in "${context_specs[@]}"; do
      read -r norm slot lr norm_cfg <<< "${spec}"
      generate_with_block_size "${ctx}" "context_boundary" "${norm}" "${slot}" "${lr}" "adamw" "${seed}" "${norm_cfg}"
    done
  done
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

chmod +x "${gen_dir}/z_bash_execute.sh"
echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
echo "Run with: cd ${gen_dir} && bash z_bash_execute.sh"

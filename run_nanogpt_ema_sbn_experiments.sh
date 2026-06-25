#!/usr/bin/env bash
set -euo pipefail

# Focused EMA-SBN / EMA-CFBN experiments for nanoGPT/TinyShakespeare.
#
# Design:
#   1. controls: keep the strongest previous baselines in the same script
#      (LN, mlp/CSBN, mlp/CDSeqBN, and the best CCFBN slots) so W&B
#      comparisons do not depend on memory.
#   2. ema_sbn_mlp_sweep: test whether replacing CSBN's uniform prefix statistics
#      with exponential prefix statistics improves the strongest prior slot,
#      mlp/CSBN. Momentum is the current-token EMA weight.
#   3. ema_cfbn_primary_sweep: test EMA-CFBN on the previously strong CCFBN
#      slots: mlp/CCFBN, attn/CCFBN, and final/CCFBNs.
#   4. ema_slot_probe: test whether the best EMA memory should stay in MLP or
#      move to attention/final/all once the statistics are less long-range.
#   5. ema_multiseed: add seeds for the most likely winners and matching
#      previous controls.

dir_name="${DIR_NAME:-exp-nanogpt-ema-norm}"

script_path="${BASH_SOURCE[0]}"
if [[ "${script_path}" == */* ]]; then
  script_dir="$(cd "${script_path%/*}" && pwd)"
else
  script_dir="$(pwd)"
fi
repo_root="$(cd "${script_dir}/.." && pwd)"
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
subjectname="${WANDB_PROJECT:-nanoGPT-EMA-Norm}"

dataset_root="/home/dlth/norm-exp-code/dataset"
output_root="/home/dlth/norm-exp-code/Norm-exp-base/nanoGPT/results/${dir_name}"
python_bin="/home/dlth/miniconda3/envs/norm-base/bin/python"
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
    printf 'cd "$(dirname "$0")/../.."\n'
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

controls=(
  "control LN baseline 3e-4 adamw 0 -"
  "control LN baseline 6e-4 adamw 0 -"
  "control LN baseline 1e-3 adamw 0 -"
  "control CSBN mlp 6e-4 adamw 0 -"
  "control CSBN mlp 1e-3 adamw 0 -"
  "control CDSeqBN mlp 3e-4 adamw 0 -"
  "control CDSeqBN mlp 6e-4 adamw 0 -"
  "control CCFBN mlp 6e-4 adamw 0 -"
  "control CCFBN attn 3e-4 adamw 0 -"
  "control CCFBN attn 6e-4 adamw 0 -"
  "control CCFBNs final 3e-4 adamw 0 -"
  "control CCFBNs final 6e-4 adamw 0 -"
  "control CCFBNc mlp 6e-4 adamw 0 -"
)

for spec in "${controls[@]}"; do
  read -r phase norm slot lr optimizer seed norm_cfg <<< "${spec}"
  generate_job "${phase}" "${norm}" "${slot}" "${lr}" "${optimizer}" "${seed}" "${norm_cfg}"
done

ema_momenta=(0.02 0.05 0.1 0.2)
ema_lrs=(6e-4 1e-3)

for momentum in "${ema_momenta[@]}"; do
  for lr in "${ema_lrs[@]}"; do
    generate_job "ema_sbn_mlp_sweep" "EMASBN" "mlp" "${lr}" "adamw" "0" "momentum=${momentum}"
    generate_job "ema_sbn_mlp_sweep" "EMASBNc" "mlp" "${lr}" "adamw" "0" "momentum=${momentum}"
  done
done

# Scaling-only is high-risk because the first token has near-zero EMA variance,
# so keep it as a small diagnostic slice rather than a full sweep.
for momentum in 0.05 0.1; do
  generate_job "ema_scaling_diag" "EMASBNs" "mlp" "6e-4" "adamw" "0" "momentum=${momentum}"
done

ema_cfbn_primary_specs=(
  "EMACFBN mlp"
  "EMACFBN attn"
  "EMACFBNs final"
  "EMACFBNc mlp"
)
ema_cfbn_lrs=(3e-4 6e-4)

for momentum in "${ema_momenta[@]}"; do
  for lr in "${ema_cfbn_lrs[@]}"; do
    for spec in "${ema_cfbn_primary_specs[@]}"; do
      read -r norm slot <<< "${spec}"
      generate_job "ema_cfbn_primary_sweep" "${norm}" "${slot}" "${lr}" "adamw" "0" "momentum=${momentum}"
    done
  done
done

# Keep all-slot EMA-CFBN as a diagnostic: CCFBN all was not the strongest
# setting, but EMA may reduce the stale-statistics cost enough to make it useful.
for momentum in 0.05 0.1; do
  for norm in EMACFBN EMACFBNc EMACFBNs; do
    generate_job "ema_cfbn_all_diag" "${norm}" "all" "6e-4" "adamw" "0" "momentum=${momentum}"
  done
done

for momentum in 0.05 0.1; do
  for norm in EMASBN EMASBNc; do
    for slot in attn mlp final all; do
      generate_job "ema_slot_probe" "${norm}" "${slot}" "6e-4" "adamw" "0" "momentum=${momentum}"
    done
  done
done

multiseed_specs=(
  "EMASBN mlp 1e-3 momentum=0.05"
  "EMASBN mlp 1e-3 momentum=0.1"
  "EMASBNc mlp 6e-4 momentum=0.05"
  "EMACFBN mlp 6e-4 momentum=0.05"
  "EMACFBN attn 3e-4 momentum=0.05"
  "EMACFBNs final 3e-4 momentum=0.05"
  "EMACFBNc mlp 6e-4 momentum=0.05"
  "CCFBN mlp 6e-4 -"
  "CCFBN attn 3e-4 -"
  "CCFBNs final 3e-4 -"
  "CSBN mlp 1e-3 -"
  "LN baseline 3e-4 -"
)
multiseed_seeds=(1 2 3)

for seed in "${multiseed_seeds[@]}"; do
  for spec in "${multiseed_specs[@]}"; do
    read -r norm slot lr norm_cfg <<< "${spec}"
    generate_job "ema_multiseed" "${norm}" "${slot}" "${lr}" "adamw" "${seed}" "${norm_cfg}"
  done
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

chmod +x "${gen_dir}/z_bash_execute.sh"
echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
echo "Run with: cd ${gen_dir} && bash z_bash_execute.sh"

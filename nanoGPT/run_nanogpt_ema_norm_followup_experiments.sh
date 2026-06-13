#!/usr/bin/env bash
set -euo pipefail

# Follow-up experiments for whsjrc-buaa/nanoGPT-EMA-Norm.
#
# This script is intentionally narrower than the first EMA sweep:
#   1. multiseed_validate: validate seed0-strong EMA-CFBN candidates.
#   2. local_lr_refine: refine the LR x momentum neighborhood without re-running
#      points that already exist in the first EMA-Norm sweep by default.
#   3. early_stop_probe: rerun short 35-epoch jobs to test whether EMA-CFBN's
#      weakness is late drift rather than poor best validation loss.
#
# To regenerate the full local LR grid including already-ran points, use:
#   INCLUDE_EXISTING_GRID=1 bash nanoGPT/run_nanogpt_ema_norm_followup_experiments.sh

dir_name="${DIR_NAME:-exp-nanogpt-ema-norm-followup}"

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
short_epochs="${SHORT_EPOCHS:-35}"
iters_per_epoch="${ITERS_PER_EPOCH:-100}"
eval_iters="${EVAL_ITERS:-50}"
display_every="${DISPLAY_EVERY:-1}"
weight_decay="${WEIGHT_DECAY:-0.1}"
dropout="${DROPOUT:-0.2}"
lr_method="${LR_METHOD:-cos}"
dtype="${DTYPE:-bfloat16}"
sample_tokens="${SAMPLE_TOKENS:-0}"
sample_every="${SAMPLE_EVERY:-0}"
subjectname="${WANDB_PROJECT:-nanoGPT-EMA-Norm-followup}"
include_existing_grid="${INCLUDE_EXISTING_GRID:-0}"

dataset_root="${DATASET_ROOT:-${repo_root}/dataset/tinyshakespeare}"
output_root="${OUTPUT_ROOT:-${repo_root}/results/${dir_name}}"
python_bin="${PYTHON_BIN:-python}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
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
  local seed="$5"
  local norm_cfg="$6"
  local job_epochs="$7"

  local norm_tag lr_tag batch_tag
  norm_tag="$(tag_for "${norm}" "${slot}" "${norm_cfg}")"
  lr_tag="$(sanitize "${lr}")"
  batch_tag="$(sanitize "${batch_size}")"

  local base_string="execute_${phase}_${arch}_${dataset}_L${n_layer}_H${n_head}_D${n_embd}_ctx${block_size}_${norm_tag}_${activation}_lr${lr_tag}_bs${batch_tag}_drop${dropout}_wd${weight_decay}_e${job_epochs}_s${seed}_adamw"
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
    printf '  --epochs=%s \\\n' "${job_epochs}"
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

already_ran_grid_point() {
  local norm="$1"
  local slot="$2"
  local lr="$3"
  local momentum="$4"

  case "${norm} ${slot} ${lr} ${momentum}" in
    "EMACFBN mlp 3e-4 0.10"|"EMACFBN mlp 6e-4 0.10"|\
    "EMACFBN mlp 3e-4 0.20"|"EMACFBN mlp 6e-4 0.20"|\
    "EMACFBNs final 3e-4 0.10"|"EMACFBNs final 6e-4 0.10"|\
    "EMACFBNs final 3e-4 0.20"|"EMACFBNs final 6e-4 0.20")
      return 0
      ;;
  esac
  return 1
}

# 1. Validate seed0-strong candidates with seeds 1/2/3.
multiseed_seeds=(1 2 3)
multiseed_specs=(
  "EMACFBN mlp 3e-4 momentum=0.20"
  "EMACFBNs final 3e-4 momentum=0.20"
  "EMACFBN mlp 3e-4 momentum=0.10"
)

for seed in "${multiseed_seeds[@]}"; do
  for spec in "${multiseed_specs[@]}"; do
    read -r norm slot lr norm_cfg <<< "${spec}"
    generate_job "multiseed_validate" "${norm}" "${slot}" "${lr}" "${seed}" "${norm_cfg}" "${epochs}"
  done
done

# 2. Refine the LR x momentum neighborhood around the promising EMA-CFBN points.
local_specs=(
  "EMACFBN mlp 0.10 2e-4 3e-4 4e-4 6e-4"
  "EMACFBN mlp 0.20 2e-4 3e-4 4e-4 6e-4"
  "EMACFBNs final 0.10 2e-4 3e-4 4e-4"
  "EMACFBNs final 0.20 2e-4 3e-4 4e-4"
)

for spec in "${local_specs[@]}"; do
  read -r norm slot momentum lr1 lr2 lr3 lr4 <<< "${spec}"
  for lr in "${lr1}" "${lr2}" "${lr3}" "${lr4:-}"; do
    if [ -z "${lr}" ]; then
      continue
    fi
    if [ "${include_existing_grid}" != "1" ] && already_ran_grid_point "${norm}" "${slot}" "${lr}" "${momentum}"; then
      continue
    fi
    generate_job "local_lr_refine" "${norm}" "${slot}" "${lr}" "0" "momentum=${momentum}" "${epochs}"
  done
done

# 3. Short-horizon reruns to determine whether EMA-CFBN mainly suffers from
# late drift. These intentionally rerun known points with shorter training.
early_stop_specs=(
  "EMACFBN mlp 6e-4 momentum=0.05"
  "EMACFBN mlp 3e-4 momentum=0.20"
  "CCFBN mlp 6e-4 -"
  "CSBN mlp 1e-3 -"
)

for spec in "${early_stop_specs[@]}"; do
  read -r norm slot lr norm_cfg <<< "${spec}"
  generate_job "early_stop_probe" "${norm}" "${slot}" "${lr}" "0" "${norm_cfg}" "${short_epochs}"
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

chmod +x "${gen_dir}/z_bash_execute.sh"
echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
echo "Run with: cd ${gen_dir} && bash z_bash_execute.sh"

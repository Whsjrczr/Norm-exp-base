#!/usr/bin/env bash
set -euo pipefail

# Three focused experiment groups for nanoGPT/TinyShakespeare:
#   1. lr_sweep: AdamW learning-rate sensitivity for LN, CSBN, CSeqBN, CDSeqBN.
#   2. opt_compare: small optimizer comparison for LN, CSBN-MLP, CDSeqBN-MLP.
#   3. stat_variants: causal sample-wise statistics and current runnable
#      LN-hybrid proxy CSBNc+LN.
#
# EMA/running-average SBN and CFBN variants are covered separately by
# run_nanogpt_ema_norm_experiments.sh.

dir_name="${DIR_NAME:-exp-nanogpt-sbn-seqbn-lr-optimizer}"

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
subjectname="${WANDB_PROJECT:-nanoGPT-SBN-SeqBN-lr-optimizer}"

dataset_root="${DATASET_ROOT:-${repo_root}/dataset/tinyshakespeare}"
output_root="${OUTPUT_ROOT:-${repo_root}/results/${dir_name}}"
python_bin="${PYTHON_BIN:-python}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
num_once="${NUM_ONCE:-1}"

launch_cnt=0
: > "${gen_dir}/z_bash_execute.sh"

sanitize() {
  local value="$1"
  value="${value//+/p}"
  value="${value//,/x}"
  value="${value//./p}"
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

  case "${slot}" in
    baseline|all)
      printf '  --norm=%s \\\n' "${norm}"
      ;;
    attn)
      printf '  --attn-norm=%s \\\n  --mlp-norm=LN \\\n  --final-norm=LN \\\n' "${norm}"
      ;;
    mlp)
      printf '  --attn-norm=LN \\\n  --mlp-norm=%s \\\n  --final-norm=LN \\\n' "${norm}"
      ;;
    final)
      printf '  --attn-norm=LN \\\n  --mlp-norm=LN \\\n  --final-norm=%s \\\n' "${norm}"
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
  local norm_tag
  norm_tag="$(sanitize "${norm}")"

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
}

generate_job() {
  local phase="$1"
  local norm="$2"
  local slot="$3"
  local lr="$4"
  local optimizer="$5"
  local seed="$6"

  local norm_tag lr_tag
  norm_tag="$(tag_for "${norm}" "${slot}")"
  lr_tag="$(sanitize "${lr}")"

  local base_string="execute_${phase}_${arch}_${dataset}_L${n_layer}_H${n_head}_D${n_embd}_ctx${block_size}_${norm_tag}_${activation}_lr${lr_tag}_bs$(sanitize "${batch_size}")_drop${dropout}_wd${weight_decay}_s${seed}_${optimizer}"
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
    norm_args_for "${norm}" "${slot}"
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

lr_sweep_specs=(
  "LN baseline"
  "CSBN mlp"
  "CSBN all"
  "CSeqBN mlp"
  "CDSeqBN mlp"
)
lr_sweep_lrs=(1e-4 3e-4 6e-4 1e-3)
lr_sweep_seeds=(0)

for seed in "${lr_sweep_seeds[@]}"; do
  for spec in "${lr_sweep_specs[@]}"; do
    read -r norm slot <<< "${spec}"
    for lr in "${lr_sweep_lrs[@]}"; do
      generate_job "lr_sweep" "${norm}" "${slot}" "${lr}" "adamw" "${seed}"
    done
  done
done

opt_compare_specs=(
  "LN baseline"
  "CSBN mlp"
  "CDSeqBN mlp"
)
opt_compare_optimizer_lrs=(
  "adamw 6e-4"
  "adam 6e-4"
  "sgd 3e-3"
  "sgd 1e-3"
)
opt_compare_seeds=(0)

for seed in "${opt_compare_seeds[@]}"; do
  for spec in "${opt_compare_specs[@]}"; do
    read -r norm slot <<< "${spec}"
    for opt_lr in "${opt_compare_optimizer_lrs[@]}"; do
      read -r optimizer lr <<< "${opt_lr}"
      generate_job "opt_compare" "${norm}" "${slot}" "${lr}" "${optimizer}" "${seed}"
    done
  done
done

stat_variant_specs=(
  "LN baseline"
  "RMS attn"
  "CSBN mlp"
  "CSBNc mlp"
  "CSBNs mlp"
  "CSBNc+LN mlp"
  "CSeqBN mlp"
  "CSeqBNc mlp"
  "CSeqBNs mlp"
  "CDSeqBN mlp"
  "CDSeqBNc mlp"
  "CDSeqBNs mlp"
)
stat_variant_seeds=(0)

for seed in "${stat_variant_seeds[@]}"; do
  for spec in "${stat_variant_specs[@]}"; do
    read -r norm slot <<< "${spec}"
    generate_job "stat_variants" "${norm}" "${slot}" "6e-4" "adamw" "${seed}"
  done
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

chmod +x "${gen_dir}/z_bash_execute.sh"
echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
echo "Run with: cd ${gen_dir} && bash z_bash_execute.sh"

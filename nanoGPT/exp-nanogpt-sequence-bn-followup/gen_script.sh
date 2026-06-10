#!/usr/bin/env bash
set -euo pipefail

# Focused follow-up experiments for nanoGPT/TinyShakespeare norm comparison.
#
# Existing W&B runs show:
#   - attnLN + mlpCSBN + finalLN is the strongest and most stable setting.
#   - CDSeqBNc helps on L6 but is unstable on L8.
#   - CCFBN / final CCFBNs need learning-rate and slot checks.
#   - Plain LN remains the baseline.
#
# This generator keeps the original one-script-per-run workflow, but replaces
# the broad old traversal with a smaller matrix that directly tests those gaps.

dir_name="${DIR_NAME:-exp-nanogpt-sequence-bn-followup}"

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
block_size="${BLOCK_SIZE:-256}"
batch_size="${BATCH_SIZE:-64,64}"
activation="${ACTIVATION:-gelu}"
epochs="${EPOCHS:-50}"
iters_per_epoch="${ITERS_PER_EPOCH:-100}"
eval_iters="${EVAL_ITERS:-50}"
display_every="${DISPLAY_EVERY:-1}"
optimizer="${OPTIMIZER:-adamw}"
weight_decay="${WEIGHT_DECAY:-0.1}"
dropout="${DROPOUT:-0.2}"
lr_method="${LR_METHOD:-cos}"
dtype="${DTYPE:-bfloat16}"
sample_tokens="${SAMPLE_TOKENS:-0}"
sample_every="${SAMPLE_EVERY:-0}"
subjectname="${WANDB_PROJECT:-nanoGPT-Norm-Scaling-Followup}"

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
    * )
      echo "Unknown slot: ${slot}" >&2
      exit 1
      ;;
  esac
}

allow_noncausal_for() {
  local norm="$1"

  case "${norm}" in
    BN|BNc|BNs|SBN|SBNc|SBNs|SeqBN|SeqBNc|SeqBNs|CFBN|CFBNc|CFBNs)
      printf '  --allow-noncausal-norm \\\n'
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
    * )
      echo "Unknown slot: ${slot}" >&2
      exit 1
      ;;
  esac
}

generate_job() {
  local phase="$1"
  local n_layer="$2"
  local n_head="$3"
  local n_embd="$4"
  local norm="$5"
  local slot="$6"
  local lr="$7"
  local seed="$8"

  if (( n_embd % n_head != 0 )); then
    echo "Skip invalid size: L${n_layer} H${n_head} D${n_embd}" >&2
    return
  fi

  local norm_tag lr_tag batch_tag base_string file_name
  norm_tag="$(tag_for "${norm}" "${slot}")"
  lr_tag="$(sanitize "${lr}")"
  batch_tag="$(sanitize "${batch_size}")"
  base_string="execute_${phase}_${arch}_${dataset}_L${n_layer}_H${n_head}_D${n_embd}_ctx${block_size}_${norm_tag}_${activation}_lr${lr_tag}_bs${batch_tag}_drop${dropout}_wd${weight_decay}_s${seed}_${optimizer}"
  file_name="${base_string}.sh"

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
    printf '  --weight-decay=%s \\\n' "${weight_decay}"
    printf '  --dropout=%s \\\n' "${dropout}"
    norm_args_for "${norm}" "${slot}"
    allow_noncausal_for "${norm}"
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

sizes=(
  "L6 6 6 384"
  "L8 8 8 512"
)

seeds=(0 1 2)

# Main replication/extension matrix, matching the real W&B comparison.
primary_specs=(
  "LN baseline 6e-4"
  "CSBN mlp 6e-4"
  "CDSeqBNc mlp 6e-4"
  "CCFBN mlp 6e-4"
  "CCFBNs final 3e-4"
  "CCFBNs final 6e-4"
)

for size_spec in "${sizes[@]}"; do
  read -r size_tag n_layer n_head n_embd <<< "${size_spec}"
  for seed in "${seeds[@]}"; do
    for spec in "${primary_specs[@]}"; do
      read -r norm slot lr <<< "${spec}"
      generate_job "primary_${size_tag}" "${n_layer}" "${n_head}" "${n_embd}" "${norm}" "${slot}" "${lr}" "${seed}"
    done
  done
done

# Learning-rate sensitivity for the two most informative sequence norms.
lr_sweep_specs=(
  "CSBN mlp"
  "CDSeqBNc mlp"
)
lr_sweep_lrs=(3e-4 6e-4 1e-3)

for seed in 0; do
  for spec in "${lr_sweep_specs[@]}"; do
    read -r norm slot <<< "${spec}"
    for lr in "${lr_sweep_lrs[@]}"; do
      generate_job "lr_sweep_L6" 6 6 384 "${norm}" "${slot}" "${lr}" "${seed}"
      generate_job "lr_sweep_L8" 8 8 512 "${norm}" "${slot}" "${lr}" "${seed}"
    done
  done
done

# Slot sweep on L6 to check whether the CSBN/CDSeqBNc gain is MLP-specific.
slot_sweep_norms=(CSBN CDSeqBNc)
slot_sweep_slots=(attn mlp final all)

for norm in "${slot_sweep_norms[@]}"; do
  for slot in "${slot_sweep_slots[@]}"; do
    generate_job "slot_sweep_L6" 6 6 384 "${norm}" "${slot}" "6e-4" 0
  done
done

# Suffix/statistic variants that were missing from the current W&B project.
variant_specs=(
  "CSBNc mlp"
  "CSBNs mlp"
  "CSeqBN mlp"
  "CSeqBNc mlp"
  "CSeqBNs mlp"
  "CDSeqBN mlp"
  "CDSeqBNs mlp"
)

for spec in "${variant_specs[@]}"; do
  read -r norm slot <<< "${spec}"
  generate_job "suffix_variants_L6" 6 6 384 "${norm}" "${slot}" "6e-4" 0
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

chmod +x "${gen_dir}/z_bash_execute.sh"
echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
echo "Run with: cd ${gen_dir} && bash z_bash_execute.sh"

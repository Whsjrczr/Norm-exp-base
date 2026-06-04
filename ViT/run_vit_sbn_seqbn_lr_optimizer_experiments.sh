#!/usr/bin/env bash
set -euo pipefail

# Three focused experiment groups for ViT/CIFAR-10:
#   1. lr_sweep: AdamW learning-rate sensitivity for LN/SBN/SeqBN/BN/DSeqBN.
#   2. opt_compare: small optimizer comparison for LN/SBN/SeqBN.
#   3. stat_variants: no-batch/token-axis statistics and current runnable
#      LN-hybrid proxies such as SBNc+LN and DSeqBNc+LN.
#
# True EMA/running-average sequence norms are not implemented yet. Add them to
# stat_variant_norms after registering the corresponding norm names.

dir_name="${DIR_NAME:-exp-vit-sbn-seqbn-lr-optimizer}"

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

arch="${ARCH:-vit_small}"
dataset="${DATASET:-cifar10}"
image_size="${IMAGE_SIZE:-32}"
patch_size="${PATCH_SIZE:-4}"
batch_size="${BATCH_SIZE:-256}"
epochs="${EPOCHS:-200}"
activation="${ACTIVATION:-relu}"
weight_decay="${WEIGHT_DECAY:-0.1}"
dropout="${DROPOUT:-0.0}"
drop_path_rate="${DROP_PATH_RATE:-0.1}"
lr_method="${LR_METHOD:-cos}"
subjectname="${WANDB_PROJECT:-ViT-SBN-SeqBN-lr-optimizer}"

dataset_root="${DATASET_ROOT:-${repo_root}/dataset}"
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

generate_job() {
  local phase="$1"
  local norm="$2"
  local lr="$3"
  local optimizer="$4"
  local seed="$5"

  local norm_tag lr_tag
  norm_tag="$(sanitize "${norm}")"
  lr_tag="$(sanitize "${lr}")"

  local base_string="execute_${phase}_${arch}_${dataset}_img${image_size}_patch${patch_size}_${norm_tag}_${activation}_lr${lr_tag}_bs${batch_size}_drop${dropout}_dpath${drop_path_rate}_wd${weight_decay}_s${seed}_${optimizer}"
  local file_name="${base_string}.sh"

  echo "Generating ${base_string}"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd "$(dirname "$0")/../.."\n'
    printf 'CUDA_VISIBLE_DEVICES=%s %s %s \\\n' "${cuda_visible_devices}" "${python_bin}" "${repo_root}/ViT/vit.py"
    printf '  --arch=%s \\\n' "${arch}"
    printf '  --dataset=%s \\\n' "${dataset}"
    printf '  --dataset-root=%s \\\n' "${dataset_root}"
    printf '  --im-size=%s,%s \\\n' "${image_size}" "${image_size}"
    printf '  --patch-size=%s \\\n' "${patch_size}"
    printf '  --batch-size=%s \\\n' "${batch_size}"
    printf '  --epochs=%s \\\n' "${epochs}"
    printf '  --optimizer=%s \\\n' "${optimizer}"
    optimizer_extra_args_for "${optimizer}"
    printf '  --lr=%s \\\n' "${lr}"
    printf '  --lr-method=%s \\\n' "${lr_method}"
    printf '  --weight-decay=%s \\\n' "${weight_decay}"
    printf '  --dropout=%s \\\n' "${dropout}"
    printf '  --drop-path-rate=%s \\\n' "${drop_path_rate}"
    printf '  --norm=%s \\\n' "${norm}"
    printf '  --activation=%s \\\n' "${activation}"
    printf '  --seed=%s \\\n' "${seed}"
    printf '  --output=%s \\\n' "${output_root}"
    printf '  --visualize \\\n'
    printf '  --wandb_project=%s\n' "${subjectname}"
  } > "${gen_dir}/${file_name}"
  chmod +x "${gen_dir}/${file_name}"

  echo "nohup bash ${file_name} > output_${base_string}.out 2>&1 &" >> "${gen_dir}/z_bash_execute.sh"
  launch_cnt=$((launch_cnt + 1))
  if (( launch_cnt % num_once == 0 )); then
    echo "wait" >> "${gen_dir}/z_bash_execute.sh"
  fi
}

lr_sweep_norms=(LN SBN SeqBN BN DSeqBN)
lr_sweep_lrs=(3e-5 1e-4 3e-4 1e-3)
lr_sweep_seeds=(0 1 2)

for seed in "${lr_sweep_seeds[@]}"; do
  for norm in "${lr_sweep_norms[@]}"; do
    for lr in "${lr_sweep_lrs[@]}"; do
      generate_job "lr_sweep" "${norm}" "${lr}" "adamw" "${seed}"
    done
  done
done

opt_compare_norms=(LN SBN SeqBN)
opt_compare_specs=(
  "adamw 1e-4"
  "adam 1e-4"
  "sgd 1e-2"
  "sgd 3e-3"
)
opt_compare_seeds=(0 1 2)

for seed in "${opt_compare_seeds[@]}"; do
  for norm in "${opt_compare_norms[@]}"; do
    for spec in "${opt_compare_specs[@]}"; do
      read -r optimizer lr <<< "${spec}"
      generate_job "opt_compare" "${norm}" "${lr}" "${optimizer}" "${seed}"
    done
  done
done

stat_variant_norms=(
  LN RMS
  SBN SBNc SBNs SBNc+LN
  SeqBN
  DSeqBN DSeqBNc DSeqBNs DSeqBNc+LN
)
stat_variant_seeds=(0 1 2)

for seed in "${stat_variant_seeds[@]}"; do
  for norm in "${stat_variant_norms[@]}"; do
    generate_job "stat_variants" "${norm}" "1e-4" "adamw" "${seed}"
  done
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

chmod +x "${gen_dir}/z_bash_execute.sh"
echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
echo "Run with: cd ${gen_dir} && bash z_bash_execute.sh"

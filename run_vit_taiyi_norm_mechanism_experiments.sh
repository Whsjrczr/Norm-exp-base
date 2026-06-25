#!/usr/bin/env bash
set -euo pipefail

# Taiyi mechanism diagnostics for SBN/CFBN and centering ablations in ViT.
#
# Phases:
#   1. mechanism_grid:
#      Patch/LR grid over BN, SBN, CFBN, SeqBN/DSeqBN, LN/RMS.
#   2. multiseed_confirm:
#      Small multiseed confirmation for the strongest full-vs-scaling-only pairs.

dir_name="${DIR_NAME:-exp-vit-taiyi-norm-mechanism}"

script_path="${BASH_SOURCE[0]}"
if [[ "${script_path}" == */* ]]; then
  script_dir="$(cd "${script_path%/*}" && pwd)"
else
  script_dir="$(pwd)"
fi

if [[ -d "${script_dir}/ViT" ]]; then
  repo_root="${script_dir}"
else
  repo_root="$(cd "${script_dir}/.." && pwd)"
fi

gen_dir="${script_dir}/${dir_name}"
mkdir -p "${gen_dir}"
cp "$0" "${gen_dir}/gen_script.sh"

arch="${ARCH:-vit_small}"
dataset="${DATASET:-cifar10}"
image_size="${IMAGE_SIZE:-32}"
batch_size="${BATCH_SIZE:-256}"
epochs="${EPOCHS:-80}"
display_every="${DISPLAY_EVERY:-1}"
optimizer="${OPTIMIZER:-adam}"
weight_decay="${WEIGHT_DECAY:-0.1}"
dropout="${DROPOUT:-0.0}"
drop_path="${DROP_PATH_RATE:-0.1}"
lr_method="${LR_METHOD:-cos}"
activation="${ACTIVATION:-relu}"
subjectname="${WANDB_PROJECT:-ViT-Taiyi-norm-mechanism}"
offline="${OFFLINE:-0}"

dataset_root="${DATASET_ROOT:-/home/dlth/norm-exp-code/dataset}"
output_root="${OUTPUT_ROOT:-/home/dlth/norm-exp-code/Norm-exp-base/ViT/results/${dir_name}}"
python_bin="${PYTHON_BIN:-/home/dlth/miniconda3/envs/norm-base/bin/python}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-1}"
num_once="${NUM_ONCE:-1}"

launch_cnt=0
: > "${gen_dir}/z_bash_execute.sh"

sanitize() {
  local value="$1"
  value="${value//+/p}"
  value="${value//,/x}"
  value="${value//./p}"
  value="${value//=/}"
  printf '%s' "${value}"
}

generate_job() {
  local phase="$1"
  local patch_size="$2"
  local norm="$3"
  local lr="$4"
  local seed="$5"

  if (( patch_size > image_size )); then
    return
  fi

  local lr_tag dp_tag wd_tag
  lr_tag="$(sanitize "${lr}")"
  dp_tag="$(sanitize "${drop_path}")"
  wd_tag="$(sanitize "${weight_decay}")"

  local base_string="execute_${phase}_${arch}_${dataset}_img${image_size}_patch${patch_size}_${norm}_${activation}_lr${lr_tag}_bs${batch_size}_drop${dropout}_dpath${dp_tag}_wd${wd_tag}_s${seed}_${optimizer}"
  local file_name="${base_string}.sh"

  echo "Generating ${base_string}"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd "%s"\n' "${repo_root}"
    printf 'CUDA_VISIBLE_DEVICES=%s %s %s \\\n' "${cuda_visible_devices}" "${python_bin}" "${repo_root}/ViT/vit.py"
    printf '  --arch=%s \\\n' "${arch}"
    printf '  --dataset=%s \\\n' "${dataset}"
    printf '  --dataset-root=%s \\\n' "${dataset_root}"
    printf '  --im-size=%s,%s \\\n' "${image_size}" "${image_size}"
    printf '  --patch-size=%s \\\n' "${patch_size}"
    printf '  --batch-size=%s \\\n' "${batch_size}"
    printf '  --epochs=%s \\\n' "${epochs}"
    printf '  --lr=%s \\\n' "${lr}"
    printf '  --lr-method=%s \\\n' "${lr_method}"
    printf '  --optimizer=%s \\\n' "${optimizer}"
    printf '  --weight-decay=%s \\\n' "${weight_decay}"
    printf '  --dropout=%s \\\n' "${dropout}"
    printf '  --drop-path-rate=%s \\\n' "${drop_path}"
    printf '  --norm=%s \\\n' "${norm}"
    printf '  --activation=%s \\\n' "${activation}"
    printf '  --seed=%s \\\n' "${seed}"
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

primary_patches=(4 16)
primary_lrs=(1e-4 1e-3)
primary_norms=(LN RMS BN BNc BNs SBN SBNc SBNs CFBN CFBNc CFBNs SeqBN DSeqBN)

for patch_size in "${primary_patches[@]}"; do
  for lr in "${primary_lrs[@]}"; do
    for norm in "${primary_norms[@]}"; do
      generate_job "mechanism_grid" "${patch_size}" "${norm}" "${lr}" "0"
    done
  done
done

confirm_seeds=(1 2)
confirm_specs=(
  "16 1e-3 BN"
  "16 1e-3 BNs"
  "16 1e-3 CFBN"
  "16 1e-3 CFBNs"
  "16 1e-3 SBN"
  "16 1e-3 SBNs"
  "4 1e-4 BN"
  "4 1e-4 BNs"
  "4 1e-4 CFBN"
  "4 1e-4 CFBNs"
  "4 1e-4 SBN"
  "4 1e-4 SBNs"
)

for seed in "${confirm_seeds[@]}"; do
  for spec in "${confirm_specs[@]}"; do
    read -r patch_size lr norm <<< "${spec}"
    generate_job "multiseed_confirm" "${patch_size}" "${norm}" "${lr}" "${seed}"
  done
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

chmod +x "${gen_dir}/z_bash_execute.sh"
echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
echo "Run with: cd ${gen_dir} && bash z_bash_execute.sh"

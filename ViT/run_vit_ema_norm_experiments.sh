#!/usr/bin/env bash
set -euo pipefail

# Focused EMA-norm diagnostics for ViT/CIFAR-10.
#
# EMA over image patches is order-dependent, so this is a diagnostic experiment
# rather than a primary replacement for SBN/CFBN. The sweep focuses on patch4
# and patch8, where previous SBN/CFBN runs were stable and competitive, with a
# small patch16 risk check.

dir_name="${DIR_NAME:-exp-vit-ema-norm}"

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

dataset="${DATASET:-cifar10}"
image_size="${IMAGE_SIZE:-32}"
batch_size="${BATCH_SIZE:-256}"
epochs="${EPOCHS:-200}"
display_every="${DISPLAY_EVERY:-1}"
optimizer="${OPTIMIZER:-adamw}"
weight_decay="${WEIGHT_DECAY:-0.1}"
dropout="${DROPOUT:-0.0}"
drop_path_rate="${DROP_PATH_RATE:-0.1}"
lr_method="${LR_METHOD:-cos}"
activation="${ACTIVATION:-gelu}"
subjectname="${WANDB_PROJECT:-ViT-EMA-Norm}"

dataset_root="${DATASET_ROOT:-${repo_root}/dataset}"
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

generate_job() {
  local phase="$1"
  local arch="$2"
  local patch_size="$3"
  local norm="$4"
  local lr="$5"
  local seed="$6"
  local norm_cfg="$7"

  if (( patch_size > image_size )); then
    return
  fi

  local norm_tag lr_tag batch_tag dpath_tag cfg_tag
  norm_tag="$(sanitize "${norm}")"
  lr_tag="$(sanitize "${lr}")"
  batch_tag="$(sanitize "${batch_size}")"
  dpath_tag="$(sanitize "${drop_path_rate}")"
  cfg_tag="$(sanitize "${norm_cfg}")"

  local base_string="execute_${phase}_${arch}_${dataset}_img${image_size}_patch${patch_size}_${norm_tag}"
  if [ "${norm_cfg}" != "-" ]; then
    base_string="${base_string}_${cfg_tag}"
  fi
  base_string="${base_string}_${activation}_lr${lr_tag}_bs${batch_tag}_drop${dropout}_dpath${dpath_tag}_wd${weight_decay}_s${seed}_${optimizer}"
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
    printf '  --lr=%s \\\n' "${lr}"
    printf '  --lr-method=%s \\\n' "${lr_method}"
    printf '  --optimizer=%s \\\n' "${optimizer}"
    printf '  --weight-decay=%s \\\n' "${weight_decay}"
    printf '  --dropout=%s \\\n' "${dropout}"
    printf '  --drop-path-rate=%s \\\n' "${drop_path_rate}"
    printf '  --norm=%s \\\n' "${norm}"
    if [ "${norm_cfg}" != "-" ]; then
      printf '  --norm-cfg=%s \\\n' "${norm_cfg}"
    fi
    printf '  --activation=%s \\\n' "${activation}"
    printf '  --seed=%s \\\n' "${seed}"
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

arch="${ARCH:-vit_small}"
primary_patch_sizes=(4 8)
primary_lrs=(1e-4)
primary_seeds=(0)
ema_momenta=(0.02 0.05 0.1 0.2)

control_norms=(LN SBN CFBN CFBNs DSeqBN)
ema_norms=(EMASBN EMASBNc EMASBNs EMACFBN EMACFBNc EMACFBNs)

for seed in "${primary_seeds[@]}"; do
  for patch_size in "${primary_patch_sizes[@]}"; do
    for lr in "${primary_lrs[@]}"; do
      for norm in "${control_norms[@]}"; do
        generate_job "control" "${arch}" "${patch_size}" "${norm}" "${lr}" "${seed}" "-"
      done
      for momentum in "${ema_momenta[@]}"; do
        for norm in "${ema_norms[@]}"; do
          generate_job "ema_patch_sweep" "${arch}" "${patch_size}" "${norm}" "${lr}" "${seed}" "momentum=${momentum}"
        done
      done
    done
  done
done

# Patch16 was unstable for several normalization families in prior runs, so keep
# this as a minimal risk check instead of a full momentum sweep.
for norm in LN SBN CFBN EMASBN EMACFBN EMACFBNs; do
  if [[ "${norm}" == EMA* ]]; then
    generate_job "patch16_risk" "${arch}" 16 "${norm}" "1e-4" 0 "momentum=0.1"
  else
    generate_job "patch16_risk" "${arch}" 16 "${norm}" "1e-4" 0 "-"
  fi
done

multiseed_specs=(
  "4 SBN -"
  "4 CFBN -"
  "4 CFBNs -"
  "4 EMASBN momentum=0.1"
  "4 EMACFBN momentum=0.1"
  "4 EMACFBNs momentum=0.1"
  "8 SBN -"
  "8 CFBN -"
  "8 EMASBN momentum=0.1"
  "8 EMACFBN momentum=0.1"
)
multiseed_seeds=(1 2 3)

for seed in "${multiseed_seeds[@]}"; do
  for spec in "${multiseed_specs[@]}"; do
    read -r patch_size norm norm_cfg <<< "${spec}"
    generate_job "ema_multiseed" "${arch}" "${patch_size}" "${norm}" "1e-4" "${seed}" "${norm_cfg}"
  done
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

chmod +x "${gen_dir}/z_bash_execute.sh"
echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
echo "Run with: cd ${gen_dir} && bash z_bash_execute.sh"

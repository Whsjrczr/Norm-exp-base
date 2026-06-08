#!/bin/bash
set -euo pipefail

dir_name=exp-vit-norm-scaling

script_dir="$(cd "$(dirname "$0")" && pwd)"
gen_dir="${script_dir}/${dir_name}"
mkdir -p "${gen_dir}"

cp "$0" "${gen_dir}/gen_script.sh"

dataset=cifar10
image_size=32
batchsize=256
epochs=200
display_every=1
optimizer=adam
dropout=0.0
lr_method=cos
activation=relu
subjectname="ViT-Norm-Scaling"
dataset_root="/home/dlth/norm-exp-code/dataset"
output_root="/home/dlth/norm-exp-code/Norm-exp-base/ViT/results/${dir_name}"
python_bin="/home/dlth/miniconda3/envs/norm-base/bin/python"

CUDA_VISIBLE_DEVICES=0
num_once=1
launch_cnt=0

arches=(vit_small vit_base)
seeds=(0 1 2)
norms=(CFBN CFBNs SBN DSeqBNs SeqBN)

: > "${gen_dir}/z_bash_execute.sh"

sanitize() {
  local value="$1"
  value="${value//+/p}"
  value="${value//,/x}"
  value="${value//./p}"
  printf '%s' "${value}"
}

add_job() {
  local phase="$1"
  local arch="$2"
  local patch_size="$3"
  local lr="$4"
  local norm="$5"
  local weightdecay="$6"
  local droppath="$7"
  local seed="$8"

  if (( patch_size > image_size )); then
    return
  fi

  local lr_tag
  lr_tag="$(sanitize "${lr}")"
  local wd_tag
  wd_tag="$(sanitize "${weightdecay}")"
  local dpath_tag
  dpath_tag="$(sanitize "${droppath}")"
  local baseString="execute_${phase}_${arch}_${dataset}_img${image_size}_patch${patch_size}_${norm}_${activation}_lr${lr_tag}_bs${batchsize}_drop${dropout}_dpath${dpath_tag}_wd${wd_tag}_s${seed}_${optimizer}"
  local fileName="${baseString}.sh"
  echo "Generating ${baseString}"
  cat > "${gen_dir}/${fileName}" <<EOF
#!/usr/bin/env bash
cd "\$(dirname "\$0")/../.."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${python_bin} /home/dlth/norm-exp-code/Norm-exp-base/ViT/vit.py \\
  --arch=${arch} \\
  --dataset=${dataset} \\
  --dataset-root=${dataset_root} \\
  --im-size=${image_size},${image_size} \\
  --patch-size=${patch_size} \\
  --batch-size=${batchsize} \\
  --epochs=${epochs} \\
  --lr=${lr} \\
  --lr-method=${lr_method} \\
  --optimizer=${optimizer} \\
  --weight-decay=${weightdecay} \\
  --dropout=${dropout} \\
  --drop-path-rate=${droppath} \\
  --norm=${norm} \\
  --activation=${activation} \\
  --seed=${seed} \\
  --print-f=${display_every} \\
  --output=${output_root} \\
  --visualize \\
  --wandb_project="${subjectname}" \\
  --no-save-checkpoint
EOF

  echo "nohup bash ${fileName} > output_${baseString}.out 2>&1 &" >> "${gen_dir}/z_bash_execute.sh"
  launch_cnt=$((launch_cnt + 1))
  if (( launch_cnt % num_once == 0 )); then
    echo "wait" >> "${gen_dir}/z_bash_execute.sh"
  fi
}

for arch in "${arches[@]}"; do
  for seed in "${seeds[@]}"; do
    for norm in "${norms[@]}"; do
      # Patch8: current stable setting.
      add_job patch8_main "${arch}" 8 1e-4 "${norm}" 0.1 0.1 "${seed}"

      # Patch16: CFBN needs low LR; compare wd=0.0 vs wd=0.1 explicitly.
      add_job patch16_wd0 "${arch}" 16 1e-4 "${norm}" 0.0 0.1 "${seed}"
      add_job patch16_wd01 "${arch}" 16 1e-4 "${norm}" 0.1 0.1 "${seed}"
    done
  done
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

echo "Generated ${launch_cnt} jobs under ${gen_dir}/"

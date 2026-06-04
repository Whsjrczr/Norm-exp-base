#!/bin/bash
set -euo pipefail

dir_name=exp-cfbn-patch16-diagnostics

script_dir="$(cd "$(dirname "$0")" && pwd)"
gen_dir="${script_dir}/${dir_name}"
mkdir -p "${gen_dir}"

cp "$0" "${gen_dir}/gen_script.sh"

arch=vit_small
dataset=cifar10
image_size=32
batchsize=256
epochs=200
display_every=1
optimizer=adam
dropout=0.0
lr_method=cos
activation=relu
seed=0
subjectname="ViT-CFBN-patch16-diagnostics"
dataset_root="/home/dlth/norm-exp-code/dataset"
output_root="/home/dlth/norm-exp-code/Norm-exp-base/ViT/results/${dir_name}"
python_bin="/home/dlth/miniconda3/envs/norm-base/bin/python"

CUDA_VISIBLE_DEVICES=0
num_once=1
launch_cnt=0

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
  local patch_size="$2"
  local lr="$3"
  local norm="$4"
  local weightdecay="$5"
  local droppath="$6"

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

# 1) Primary patch16 LR sweep. This tests whether 1e-3 caused the 10% collapse.
for norm in CFBN CFBNs CFBNc; do
  for lr in 1e-3 3e-4 1e-4 3e-5; do
    add_job lr_sweep_patch16 16 "${lr}" "${norm}" 0.1 0.1
  done
done

# 2) Remove stochastic depth. If collapse disappears, drop-path interacts badly
# with the short patch16 token sequence.
for norm in CFBN CFBNs; do
  for lr in 1e-3 1e-4; do
    add_job no_droppath_patch16 16 "${lr}" "${norm}" 0.1 0.0
  done
done

# 3) Weight-decay sensitivity at the safer low LR.
for norm in CFBN CFBNs; do
  for weightdecay in 0.0 0.01 0.1; do
    add_job wd_sweep_patch16 16 1e-4 "${norm}" "${weightdecay}" 0.1
  done
done

# 4) Controls at patch16. These anchor the failure against norms already known
# to survive patch16 in earlier SeqBN validation runs.
for norm in LN SBN SeqBN SeqBNs DSeqBN DSeqBNs; do
  for lr in 1e-3 1e-4; do
    add_job controls_patch16 16 "${lr}" "${norm}" 0.1 0.1
  done
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
